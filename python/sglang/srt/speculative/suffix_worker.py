"""
Suffix decoding worker that reuses NGRAMWorker with a cache adapter.

This is a thin wrapper that replaces NgramCache with SuffixCacheAdapter,
allowing all the tree-based verification logic to be reused.
"""

import json
import logging
import os
from typing import Optional

from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.ngram_worker import NGRAMWorker
from sglang.srt.speculative.suffix_cache_adapter import SuffixCacheAdapter
from sglang.srt.managers.io_struct import UpdateWeightsFromTensorReqInput

logger = logging.getLogger(__name__)


class SpecStatsLogger:
    """
    Writes per-step, per-request speculative decoding stats to a JSONL file.

    Enable by setting the SUFFIX_STATS_FILE environment variable:
        SUFFIX_STATS_FILE=/tmp/spec_stats.jsonl

    Each line is a JSON object:
        {
          "step": int,              # global decode-step counter
          "req_id": str,            # SGLang request ID
          "draft_score": float,     # sum of per-token suffix-cache probs (can exceed 1)
          "draft_score_avg": float, # draft_score / usable_drafts, mean per-token prob in [0,1]
          "first_token_prob": float,# probs[0], frequency of the first draft token in cache
          "probs": List[float],     # per-position cache frequency for draft positions 1..8
          "match_len": int,         # context suffix matched in the cache (longer = better)
          "draft_len": int,         # actual draft tokens proposed (excl. padding)
          "accept_len": int,        # tokens accepted by target model (incl. bonus token)
          "accept_rate": float      # accept_len / (draft_len - 1), or 0 if draft_len <= 1
        }

    Note: probs[i] is the suffix-cache frequency (NOT the target model probability).
    In path mode (default), probs[i] maps directly to draft position i+1 (root excluded).
    """

    def __init__(self):
        self._file = None
        self._step = 0
        output_path = os.environ.get("SUFFIX_STATS_FILE", "")
        if output_path:
            self._file = open(output_path, "w", buffering=1)
            logger.info("[SpecStatsLogger] Writing spec stats to %s", output_path)

    @property
    def enabled(self):
        return self._file is not None

    def log(self, req_ids, draft_stats, accept_lengths):
        if not self.enabled:
            return
        for rid, stats, accept_len in zip(req_ids, draft_stats, accept_lengths):
            usable_drafts = stats["draft_len"] - 1  # exclude root node
            probs = stats["probs"]
            record = {
                "step": self._step,
                "req_id": rid,
                "draft_score": round(stats["score"], 4),       # sum(probs), can exceed 1
                "draft_score_avg": round(stats["score"] / usable_drafts, 4) if usable_drafts > 0 else 0.0,  # mean per-token prob, in [0,1]
                "first_token_prob": round(probs[0], 4) if probs else 0.0,
                "probs": [round(p, 4) for p in probs[:8]],     # draft positions 1..8
                "match_len": stats["match_len"],
                "draft_len": stats["draft_len"],
                "accept_len": int(accept_len),
                "accept_rate": round(int(accept_len) / usable_drafts, 4) if usable_drafts > 0 else 0.0,
            }
            self._file.write(json.dumps(record) + "\n")
        self._step += 1

    def close(self):
        if self._file:
            self._file.close()
            self._file = None


class SuffixWorker(NGRAMWorker):
    """
    Suffix decoding worker that inherits from NGRAMWorker.

    The only difference is using SuffixCacheAdapter instead of NgramCache.
    All tree-based verification logic is inherited from NGRAMWorker.
    """

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        # Call parent __init__ which sets up all the infrastructure
        super().__init__(
            server_args,
            gpu_id,
            tp_rank,
            dp_rank,
            moe_ep_rank,
            nccl_port,
            target_worker,
        )

        self.ngram_cache = SuffixCacheAdapter(
            draft_token_num=server_args.speculative_num_draft_tokens,
            max_batch_size=self.max_batch_size,
            max_tree_depth=server_args.speculative_suffix_max_tree_depth,
            max_cached_requests=server_args.speculative_suffix_max_cached_requests,
            max_spec_factor=server_args.speculative_suffix_max_spec_factor,
            min_token_prob=server_args.speculative_suffix_min_token_prob,
        )
        self._stats_logger = SpecStatsLogger()

    def _prepare_draft_tokens(self, batch):
        """
        Override to pass FULL token sequences to the cache adapter.

        NGRAMWorker passes only last N tokens, but the suffix cache needs:
        1. Full prompt for start_request()
        2. Full sequence for suffix tree building
        3. Request identity tracking
        """

        bs = batch.batch_size()

        self.ngram_cache.synchronize()
        batch_req_ids = []
        batch_prompts = []
        batch_tokens = []
        for req in batch.reqs:
            # Pass request ID for stable tracking
            batch_req_ids.append(req.rid)
            # Pass prompt separately (for cache initialization)
            batch_prompts.append(req.origin_input_ids)
            # Pass FULL token sequence (prompt + outputs), not just last N
            full_tokens = req.origin_input_ids + req.output_ids
            batch_tokens.append(full_tokens)

        req_drafts, mask = self.ngram_cache.batch_get(
            batch_req_ids, batch_prompts, batch_tokens
        )
        total_draft_token_num = len(req_drafts)

        assert (
            total_draft_token_num == bs * self.draft_token_num
        ), f"{total_draft_token_num=}, {bs=}, {self.draft_token_num=}"

        return req_drafts, mask

    def _update_ngram_cache(self, batch, next_token_ids=None):
        """
        Override to pass FULL token sequences for cache updates.
        """
        batch_req_ids = []
        batch_tokens = []
        batch_prompts = []
        for i, req in enumerate(batch.reqs):
            # Pass request ID for stable tracking
            batch_req_ids.append(req.rid)
            # Pass prompt separately for proper cache initialization
            batch_prompts.append(req.origin_input_ids)
            # If next_token_ids is provided (spec disabled case), append it to output_ids
            # for cache update without modifying req.output_ids
            if next_token_ids is not None:
                output_ids_with_new = req.output_ids + [next_token_ids[i]]
                full_tokens = req.origin_input_ids + output_ids_with_new
            else:
                # Normal case: spec was enabled, output_ids already includes verified tokens
                full_tokens = req.origin_input_ids + req.output_ids
            batch_tokens.append(full_tokens)

        self.ngram_cache.batch_put(batch_req_ids, batch_tokens, batch_prompts)

    def forward_batch_generation(self, batch):
        """
        Override to capture draft stats vs accept lengths for correlation logging.
        Only active when SUFFIX_STATS_FILE env var is set.
        """
        # Snapshot req_ids and check if spec verify will run this step.
        # Spec verify runs when: not extend mode AND spec_algorithm is not NONE.
        will_run_spec = (
            self._stats_logger.enabled
            and not batch.forward_mode.is_extend()
            and not batch.spec_algorithm.is_none()
        )
        req_ids_snapshot = [req.rid for req in batch.reqs] if will_run_spec else None

        result = super().forward_batch_generation(batch)

        if will_run_spec and req_ids_snapshot:
            spec_info = batch.spec_info
            draft_stats = self.ngram_cache._last_draft_stats
            if (
                spec_info is not None
                and hasattr(spec_info, "accept_length")
                and draft_stats
            ):
                accept_lengths = spec_info.accept_length.tolist()
                self._stats_logger.log(req_ids_snapshot, draft_stats, accept_lengths)

        return result

    def update_weights_from_tensor(self, recv_req: UpdateWeightsFromTensorReqInput):
        return self.target_worker.update_weights_from_tensor(recv_req)