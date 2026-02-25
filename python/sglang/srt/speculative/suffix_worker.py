"""
Suffix decoding worker that reuses NGRAMWorker with a cache adapter.

This is a thin wrapper that replaces NgramCache with SuffixCacheAdapter,
allowing all the tree-based verification logic to be reused.
"""

import logging
from typing import Optional

from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.ngram_worker import NGRAMWorker
from sglang.srt.speculative.suffix_cache_adapter import SuffixCacheAdapter

logger = logging.getLogger(__name__)


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
