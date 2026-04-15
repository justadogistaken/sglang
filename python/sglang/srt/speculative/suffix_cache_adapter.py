"""
Cache adapter that wraps the suffix decoding backend cache to provide
the same interface as NgramCache.

This allows NGRAMWorker to use suffix decoding without modification.
"""

import logging
import os
from collections import deque
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class SuffixCacheAdapter:
    """
    Adapter that wraps SuffixDecodingCache to match NgramCache interface.

    NGRAMWorker expects:
    - batch_get(batch_tokens: List[List[int]]) -> Tuple[np.ndarray, np.ndarray]
      Returns (draft_tokens, tree_mask) as flat numpy arrays
    - batch_put(batch_tokens: List[List[int]]) -> None
      Updates cache with verified tokens
    - synchronize() -> None
      No-op for suffix cache
    - reset() -> None
      Clears all cached data
    """

    def __init__(
        self,
        draft_token_num: int,
        max_batch_size: int,
        max_tree_depth: int = 24,
        max_cached_requests: int = 10000,
        max_spec_factor: float = 1.0,
        min_token_prob: float = 0.1,
    ):
        """
        Args:
            draft_token_num: Fixed number of draft tokens (for padding)
            max_tree_depth: Maximum depth for suffix tree
            max_cached_requests: Maximum number of cached requests
            max_spec_factor: Maximum speculation factor
            min_token_prob: Minimum token probability threshold
        """
        # Lazy import to avoid error when Suffix Decoding is not used
        from arctic_inference.suffix_decoding import SuffixDecodingCache

        self.suffix_cache = SuffixDecodingCache(
            max_tree_depth=max_tree_depth,
            max_cached_requests=max_cached_requests,
        )
        self.draft_token_num = draft_token_num
        self.max_batch_size = max_batch_size
        self.max_tree_depth = max_tree_depth
        self.max_spec_factor = max_spec_factor
        self.min_token_prob = min_token_prob

        # Debug toggles (set env e.g. SUFFIX_DEBUG_TREE=1 to dump first batch)
        self.debug_tree_dump_remaining = int(
            os.environ.get("SUFFIX_DEBUG_TREE", "0"))

        # Track state by SGlang request ID (stable identifier)
        # Map: sglang_req_id → (arctic_req_id, last_length)
        self.req_state = {}

        # Preallocate buffers to avoid per-step allocations
        self.max_total_drafts = self.max_batch_size * self.draft_token_num
        self.draft_buffer = np.empty((self.max_total_drafts, ), dtype=np.int64)
        self.mask_buffer = np.empty(
            (self.max_batch_size, self.draft_token_num, self.draft_token_num),
            dtype=bool,
        )

        # Per-step draft stats populated by batch_get(), read by SuffixWorker
        # after verify() to correlate draft quality with accept rate.
        # Each entry: {"score": float, "match_len": int, "draft_len": int}
        self._last_draft_stats: list = []

    def _cleanup_inactive_requests(self, active_req_ids: set[str]):
        """Stop backend requests that are no longer active in SGlang."""
        inactive_req_ids = [
            rid for rid in self.req_state.keys() if rid not in active_req_ids
        ]
        for rid in inactive_req_ids:
            cache_req_id, _ = self.req_state.pop(rid)
            if cache_req_id in getattr(self.suffix_cache, "active_requests",
                                       set()):
                self.suffix_cache.stop_request(cache_req_id)

    def _get_or_create_cache_req_id(self, sglang_req_id: str,
                                    prompt: List[int],
                                    tokens: List[int]) -> tuple:
        """Get or create a backend request ID for the given SGlang request.

        Args:
            sglang_req_id: Stable request ID from SGlang
            prompt: Prompt tokens only (no generated tokens)
            tokens: Full token sequence (prompt + outputs)

        Returns: (arctic_req_id, last_length)
        """
        if sglang_req_id not in self.req_state:
            # Use SGlang request ID directly as backend request ID
            cache_req_id = sglang_req_id

            # Initialize the request in suffix cache with ONLY the prompt
            self.suffix_cache.start_request(cache_req_id, prompt)

            # Track: [arctic_req_id, last_length]
            # IMPORTANT: Set last_length to prompt length since the backend already has the prompt
            self.req_state[sglang_req_id] = [cache_req_id, len(prompt)]

        cache_req_id, last_length = self.req_state[sglang_req_id]
        return cache_req_id, last_length

    def batch_get(
        self,
        batch_req_ids: List[str],
        batch_prompts: List[List[int]],
        batch_tokens: List[List[int]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get draft tokens for a batch of token sequences.

        This is called BEFORE verification with the current state.
        We speculate based on the current tokens.

        Args:
            batch_req_ids: List of SGlang request IDs (stable)
            batch_prompts: List of prompt tokens (no generated tokens)
            batch_tokens: List of token sequences (prompt + output tokens)

        Returns:
            Tuple of:
            - draft_tokens: np.ndarray of shape (batch_size * draft_token_num,)
            - tree_mask: np.ndarray of shape (batch_size * draft_token_num * draft_token_num,)
        """
        batch_size = len(batch_req_ids)
        self._last_draft_stats = []  # reset each step

        if batch_size == 0:
            return np.empty((0, ), dtype=np.int64), np.empty((0, ), dtype=bool)

        if batch_size > self.max_batch_size:
            raise ValueError(
                f"Batch size {batch_size} exceeds configured max_batch_size={self.max_batch_size}"
            )

        total_draft_tokens = batch_size * self.draft_token_num
        draft_view = self.draft_buffer[:total_draft_tokens]
        mask_view = self.mask_buffer[:batch_size]
        mask_view.fill(False)

        active_req_ids = set(batch_req_ids)
        self._cleanup_inactive_requests(active_req_ids)

        for idx, (sglang_req_id, prompt, tokens) in enumerate(
                zip(batch_req_ids, batch_prompts, batch_tokens)):
            cache_req_id, last_length = self._get_or_create_cache_req_id(
                sglang_req_id, prompt, tokens)

            # Ensure cache includes the latest verified tokens before speculation.
            current_length = len(tokens)
            if current_length > last_length:
                new_tokens = tokens[last_length:current_length]
                if cache_req_id in self.suffix_cache.active_requests:
                    self.suffix_cache.add_active_response(
                        cache_req_id, new_tokens)
                    self.req_state[sglang_req_id][1] = current_length
                    last_length = current_length
                else:
                    logger.warning(
                        f"[BATCH_GET {idx}] Suffix cache req {cache_req_id} not active when updating!"
                    )

            # Extract pattern from end of tokens (up to max_tree_depth)
            pattern_start = max(0, len(tokens) - self.max_tree_depth)
            pattern = tokens[pattern_start:]

            # Speculate using suffix cache
            draft = self.suffix_cache.speculate(
                cache_req_id,
                pattern,
                max_spec_tokens=self.draft_token_num,
                max_spec_factor=self.max_spec_factor,
                min_token_prob=self.min_token_prob,
            )

            # Convert to fixed-size arrays
            draft_ids = list(draft.token_ids)
            draft_parents = list(draft.parents)
            draft_ids, draft_parents = self._reorder_tree_bfs(
                draft_ids, draft_parents)

            context_token = tokens[-1] if tokens else 0
            draft_ids, draft_parents = self._inject_root_node(
                draft_ids, draft_parents, context_token)

            # Pad or truncate to match draft_token_num (includes root node at index 0)
            original_draft_len = len(draft_ids)
            if original_draft_len < self.draft_token_num:
                pad_len = self.draft_token_num - original_draft_len
                draft_ids.extend([0] * pad_len)
                draft_parents.extend([0] * pad_len)
            elif original_draft_len > self.draft_token_num:
                draft_ids = draft_ids[:self.draft_token_num]
                draft_parents = draft_parents[:self.draft_token_num]
                original_draft_len = self.draft_token_num

            self._last_draft_stats.append({
                "score": draft.score,
                "match_len": draft.match_len,
                "draft_len": original_draft_len,
                # probs[i] = suffix-cache frequency of draft position i+1 (root excluded).
                # In path mode (use_tree_spec=False, current default) the order is linear
                # and maps directly to draft positions 1..N after _inject_root_node.
                "probs": list(draft.probs),
            })

            start = idx * self.draft_token_num
            end = start + self.draft_token_num
            draft_view[start:end] = draft_ids

            # Build tree mask from parent structure
            # Token i can attend to token j if j is an ancestor of i
            mask = mask_view[idx]
            if original_draft_len > 0:
                for i in range(original_draft_len):
                    mask[i, i] = True  # Self-attention
                    parent_idx = draft_parents[i]
                    while parent_idx >= 0 and parent_idx < self.draft_token_num:
                        mask[i, parent_idx] = True
                        parent_idx = draft_parents[parent_idx]

            if self.debug_tree_dump_remaining > 0 and original_draft_len > 0:
                logger.warning(
                    "[SUFFIX DEBUG] req=%s, original_draft_len=%d, masked_len=%d, draft_ids=%s",
                    sglang_req_id,
                    original_draft_len,
                    len(draft_ids),
                    draft_ids,
                )
                logger.warning(
                    "[SUFFIX DEBUG] mask=\n%s",
                    mask.astype(int),
                )
                self.debug_tree_dump_remaining -= 1

        tree_mask = mask_view.reshape(-1)[:total_draft_tokens *
                                          self.draft_token_num]

        return draft_view, tree_mask

    def batch_put(
        self,
        batch_req_ids: List[str],
        batch_tokens: List[List[int]],
        batch_prompts: Optional[List[List[int]]] = None,
    ):
        """
        Update the global cache with verified tokens.

        This method is called in two scenarios:
        1. After normal speculative decoding verification (tokens already updated via batch_get)
        2. When speculative decoding is disabled due to batch size threshold (need to update cache)
        3. External cache update from other instances (need full prompt + response)

        Args:
            batch_req_ids: List of request IDs
            batch_tokens: List of full token sequences (prompt + response for external update,
                          or just the updated portion for internal update)
            batch_prompts: Optional list of prompt-only sequences. Required when updating
                          cache for requests that are not currently active (external update
                          or spec disabled from start). If None, tokens are treated as
                          continuation of existing requests.
        """

        return None
        # Cleanup requests that are no longer active in the current batch
        # This is important when speculative decoding is disabled, as batch_get won't be called
        # active_req_ids = set(batch_req_ids)
        # self._cleanup_inactive_requests(active_req_ids)

        # for idx, (sglang_req_id, tokens) in enumerate(
        #     zip(batch_req_ids, batch_tokens)
        # ):
        #     if not tokens:
        #         continue

        #     # Check if this request is already being tracked (normal spec flow)
        #     if sglang_req_id in self.req_state:
        #         cache_req_id, last_length = self.req_state[sglang_req_id]
        #         current_length = len(tokens)

        #         # If we have new tokens to add (normal spec flow)
        #         if current_length > last_length:
        #             new_tokens = tokens[last_length:current_length]
        #             if cache_req_id in self.suffix_cache.active_requests:
        #                 # Update via add_active_response for active requests (incremental update)
        #                 self.suffix_cache.add_active_response(cache_req_id, new_tokens)
        #                 self.req_state[sglang_req_id][1] = current_length
        #             else:
        #                 # Request is not active anymore, need to re-add to cache
        #                 # Use prompt if provided, otherwise use the tokens as both prompt and response
        #                 if batch_prompts is not None and idx < len(batch_prompts):
        #                     prompt = batch_prompts[idx]
        #                     response = tokens[len(prompt):] if len(tokens) > len(prompt) else []
        #                 else:
        #                     # Fallback: treat all tokens as prompt, no response
        #                     prompt = tokens
        #                     response = []

        #                 self._add_completed_request_to_cache(
        #                     sglang_req_id, prompt, response
        #                 )
        #                 self.req_state.pop(sglang_req_id, None)
        #     else:
        #         # Request not tracked (spec was disabled from start, or external update)
        #         # Need prompt to properly add to cache
        #         if batch_prompts is not None and idx < len(batch_prompts):
        #             prompt = batch_prompts[idx]
        #             response = tokens[len(prompt):] if len(tokens) > len(prompt) else []
        #         else:
        #             # Fallback: treat all tokens as prompt (less effective but still works)
        #             prompt = tokens
        #             response = []

        #         self._add_completed_request_to_cache(sglang_req_id, prompt, response)

    def _add_completed_request_to_cache(self, req_id: str, prompt: List[int],
                                        response: List[int]):
        """
        Add a completed request's tokens to the global cache.
        
        This uses the standard flow: start_request -> add_active_response -> stop_request
        to properly cache the prompt + response in the global suffix tree.
        """
        try:
            # Start the request (adds prompt to local tree and allocates slot in global tree)
            self.suffix_cache.start_request(req_id, prompt)

            # Add response tokens (updates both local and global tree)
            if response:
                self.suffix_cache.add_active_response(req_id, response)

            # Stop the request (removes from local tree, keeps in global tree)
            self.suffix_cache.stop_request(req_id)
        except ValueError as e:
            # Handle case where request might already be cached or active
            if "already active" in str(e):
                # Request is already active, try to add response and stop
                try:
                    if response:
                        self.suffix_cache.add_active_response(req_id, response)
                    self.suffix_cache.stop_request(req_id)
                except ValueError:
                    pass  # Ignore errors in cleanup
            elif "already cached" in str(e) or "not active" in str(e):
                # Try to evict and re-add
                try:
                    if req_id in self.suffix_cache.cached_requests:
                        self.suffix_cache.evict_cached_response(req_id)
                    self._add_completed_request_to_cache(
                        req_id, prompt, response)
                except ValueError:
                    pass  # Ignore errors in retry

    def synchronize(self):
        """No-op for suffix cache (no async operations)."""
        pass

    def reset(self):
        """Clear all cached data."""
        # Stop all active requests
        for cache_req_id in list(self.suffix_cache.active_requests):
            self.suffix_cache.stop_request(cache_req_id)
        # Clear tracking
        self.req_state.clear()
        logger.info("[SUFFIX ADAPTER] Cache reset")

    def _reorder_tree_bfs(
            self, token_ids: List[int],
            parents: List[Optional[int]]) -> Tuple[List[int], List[int]]:
        """
        Reorder nodes so parents always precede their descendants.

        reconstruct_indices_from_tree_mask assumes this layout; the backend emits
        score-ordered nodes, so we re-topologize the list before building masks.
        """
        n = len(token_ids)
        if n <= 1:
            return token_ids, parents

        children: List[List[int]] = [[] for _ in range(n)]
        roots: List[int] = []
        for idx, parent in enumerate(parents):
            if parent is None or parent < 0 or parent >= n:
                roots.append(idx)
            else:
                children[parent].append(idx)

        if not roots:
            roots = [0]

        order: List[int] = []
        visited = [False] * n
        for root in roots:
            if visited[root]:
                continue
            queue = deque([root])
            while queue:
                node = queue.popleft()
                if visited[node]:
                    continue
                visited[node] = True
                order.append(node)
                for child in children[node]:
                    if not visited[child]:
                        queue.append(child)

        # Append any detached nodes (should not happen, but keep deterministic order).
        for idx in range(n):
            if not visited[idx]:
                order.append(idx)

        if order == list(range(n)):
            return token_ids, parents

        remap = {old_idx: new_idx for new_idx, old_idx in enumerate(order)}
        reordered_ids = [token_ids[old_idx] for old_idx in order]
        reordered_parents: List[int] = []
        for old_idx in order:
            parent = parents[old_idx]
            if parent is None or parent < 0:
                reordered_parents.append(-1)
            else:
                reordered_parents.append(remap.get(parent, -1))

        return reordered_ids, reordered_parents

    def _inject_root_node(self, token_ids: List[int], parents: List[int],
                          context_token: int) -> Tuple[List[int], List[int]]:
        """
        Insert the latest verified token as index 0 so the layout matches NGRAM.
        """
        rooted_ids = [context_token]
        rooted_parents = [-1]
        for parent_idx in parents:
            if parent_idx < 0:
                rooted_parents.append(0)
            else:
                rooted_parents.append(parent_idx + 1)
        rooted_ids.extend(token_ids)
        return rooted_ids, rooted_parents
