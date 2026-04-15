# Dynamic Speculative Decoding Switching

This document describes the batch-size-based dynamic speculative decoding strategy
implemented in the `auto_enable_suffix_spec` branch, the correctness issues encountered
during development, and how they were resolved.

## Overview

Speculative decoding (NGRAM/SUFFIX) improves throughput for small batches by accepting
multiple tokens per step. For large batches, however, the GPU is already well-utilised
and the spec overhead is counterproductive; overlap scheduling is preferred instead.

The dynamic switching strategy:

| Batch size | Mode |
|---|---|
| `<= speculative_disable_batch_size_threshold` | Spec decoding enabled, overlap disabled |
| `> speculative_disable_batch_size_threshold` | Spec decoding disabled, overlap enabled |

### Enabling the Feature

```bash
python -m sglang.launch_server \
    --model-path <model> \
    --speculative-algorithm SUFFIX \
    --speculative-num-draft-tokens 24 \
    --speculative-disable-batch-size-threshold 4 \
    --enable-overlap-schedule \
    --port 30000
```

Set `--speculative-disable-batch-size-threshold 0` (default) to disable dynamic
switching and always use spec decoding.

---

## Architecture

### Key Code Paths

| Component | File | Role |
|---|---|---|
| `_update_batch_spec_state` | `scheduler.py` | Sets `batch.spec_algorithm` and `batch.enable_overlap` per iteration |
| `update_running_batch` | `scheduler.py` | Calls `_update_batch_spec_state` then `prepare_for_decode` |
| `event_loop_overlap` | `scheduler.py` | Main loop; handles result queuing and cross-iteration ordering |
| `prepare_for_decode` | `schedule_batch.py` | Allocates `out_cache_loc`, sets `input_ids`; returns early when spec is active |
| `forward_batch_generation` | `ngram_worker.py` | Drives spec verify; contains rollback logic for non-spec→spec transition |
| `process_batch_result_decode` | `scheduler_output_processor_mixin.py` | Processes forward results, calls `release_kv_cache` for finished requests |

### Transition Invariants

**Non-spec → spec** (`batch.spec_algorithm` changes from `NONE` to `NGRAM/SUFFIX`):
- `prepare_for_decode()` returns early: no `out_cache_loc` or `input_ids` allocated.
- `ngram_worker.forward_batch_generation` detects the stale `out_cache_loc`/`input_ids`
  left by the previous non-spec iteration and rolls them back before calling
  `prepare_for_verify`.

**Spec → non-spec** (`batch.spec_algorithm` changes from `NGRAM/SUFFIX` to `NONE`):
- `batch.output_ids` is reset to 1 token per request (the last accepted token).
- `prepare_for_decode()` runs normally: allocates 1 `out_cache_loc` slot per request.

**Overlap → non-overlap** (switching to spec):
- After `pop_and_process()`, the current batch may contain requests already finished
  by that call. For non-overlap (spec) batches, `filter_batch()` is called immediately
  after `pop_and_process()` to remove those requests before `run_batch()`.

---

## Bugs Found and Fixed

### Bug 1 — Double-free crash via `pop_overallocated_kv_cache` assert

**Symptom:** `AssertionError: Overallocated KV cache already freed` during
`pop_overallocated_kv_cache()`.

**Root cause (commit `7c4d8b5d0`):** An earlier commit added `release_kv_cache` inside
the early-continue branch of `process_batch_result_decode`, guarded by
`batch.enable_overlap`. Because `pop_and_process()` runs *after*
`get_next_batch_to_run()`, a request R that finishes in `pop_and_process` is still
present in the current batch. The early-continue fired and called `release_kv_cache`
a second time, crashing the assert.

**Fix (`scheduler_output_processor_mixin.py`):**
- Made the early-continue unconditional (`req.finished() or req.is_retracted`).
- Removed `release_kv_cache` from the early-continue (see Bug 3 below for the correct
  conditional re-introduction).

---

### Bug 2 — Spec batch processes already-freed `req_pool_idx`

**Symptom:** Silent data corruption or KV leak when a request finishes in
`pop_and_process` but its `req_pool_idx` (freed by `cache_finished_req`) gets
reallocated to a new request before `_prepare_for_speculative_decoding` writes to it.

**Root cause:** In the overlap event loop, `get_next_batch_to_run()` runs *before*
`pop_and_process()`. When the batch switches to spec mode (`enable_overlap=False`),
no extra filtering was done after `pop_and_process()`, so the spec batch could still
reference a `req_pool_idx` that `cache_finished_req` had already freed.

**Fix (`scheduler.py`, `event_loop_overlap`):**
After `pop_and_process()`, if the current batch is a non-overlap (spec) batch, call
`batch.filter_batch()` immediately. This is safe because `prepare_for_decode()`
returned early for spec batches (no `out_cache_loc` was allocated yet for this
iteration).

```python
if batch is not None and not batch.enable_overlap:
    batch.filter_batch()
    if batch.is_empty():
        batch = None
    self.cur_batch = batch
```

---

### Bug 3 — KV leak when request finishes during spec verify

**Symptom:** `token_to_kv_pool_allocator memory leak detected!` on idle check.
Error: `available=59666, evictable=0, protected=6, max_total=59679` — 7 tokens
unaccounted for, 6 tokens still locked in the radix tree.

**Root cause:** `_fill_requests` (in `ngram_info.py`) appends accepted tokens to
`req.output_ids` and calls `req.check_finished()` *before* `process_batch_result_decode`
runs. The early-continue in `process_batch_result_decode` (from the Bug 1 fix) fired for
these requests and skipped `release_kv_cache`, so:
- `cache_finished_req` was never called → `dec_lock_ref(req.last_node)` never called
  → `protected_size` remained positive.
- KV slots allocated by `prepare_for_verify` were never freed or inserted into the
  radix tree → unaccounted tokens.

**Two distinct cases the early-continue must handle:**

| Case | `req.finished()` | `req.kv_overallocated_freed` | Action |
|---|---|---|---|
| Overlap: finished in previous `pop_and_process` | `True` | `True` (already freed) | Skip — prevent double-free |
| Spec: finished in `_fill_requests` (current iter) | `True` | `False` (not freed yet) | Call `release_kv_cache` |
| Retracted | `False` | `False` | Skip — KV belongs to re-queued req |

**Fix (`scheduler_output_processor_mixin.py`):**

```python
if req.finished() or req.is_retracted:
    if req.finished() and not req.kv_overallocated_freed:
        # Finished during spec verify; release_kv_cache not called yet.
        release_kv_cache(req, self.tree_cache)
        req.time_stats.completion_time = time.perf_counter()
    continue
```

`req.kv_overallocated_freed` is set to `True` inside `pop_overallocated_kv_cache()`
(called by `release_kv_cache`), making it a reliable sentinel.

---

### Bug 4 — Shape mismatch on spec → non-spec transition

**Symptom:**
```
RuntimeError: shape mismatch: value tensor of shape [17, 8, 128]
cannot be broadcast to indexing result of shape [5, 8, 128]
```
Crash in `flashattention_backend.forward_decode` → `set_kv_buffer`.

**Root cause:** After spec verify, `run_batch` sets:
```python
batch.output_ids = batch_result.next_token_ids  # = verified_id
```
`verified_id` is a *flat* tensor of all accepted tokens:
`shape = [sum(accept_lengths + 1)]` (e.g., 17 elements for 5 requests).

In the next iteration, `_update_batch_spec_state` switches the batch to non-spec mode
and calls `prepare_for_decode()`, which does:
```python
self.input_ids = self.output_ids  # 17 tokens
self.out_cache_loc = alloc_for_decode(...)  # 5 slots (1 per req)
```
The forward pass then tries to write 17 KV entries into 5 slots → shape mismatch.

**Fix (`scheduler.py`, `_update_batch_spec_state`):**
Reset `batch.output_ids` to 1 token per request when transitioning spec → non-spec,
using the last accepted token from each request's per-req `output_ids` list:

```python
if not batch.spec_algorithm.is_none():
    # Transitioning spec → non-spec: rebuild output_ids as 1 token per req.
    batch.output_ids = torch.tensor(
        [req.output_ids[-1] if req.output_ids else req.origin_input_ids[-1]
         for req in batch.reqs],
        dtype=torch.int64,
        device=batch.device,
    )
    batch.spec_info = None
    batch.spec_algorithm = SpeculativeAlgorithm.NONE
```

---

## Testing the Switching Boundary

To reliably trigger all four transitions, set a low threshold so the batch size
oscillates around it frequently:

```bash
# Server: threshold=2 → batch<=2 uses spec, batch>=3 uses overlap
python -m sglang.launch_server \
    --model-path <model> \
    --speculative-algorithm SUFFIX \
    --speculative-num-draft-tokens 24 \
    --speculative-disable-batch-size-threshold 2 \
    --enable-overlap-schedule --port 30000
```

```bash
# Test 1: high-frequency oscillation (short requests, moderate rate)
python3 -m sglang.bench_serving --backend sglang --port 30000 \
    --dataset-name random --random-input-len 64 --random-output-len 32 \
    --num-prompts 300 --request-rate 6 --host 127.0.0.1

# Test 2: mostly spec (low rate, small batches)
python3 -m sglang.bench_serving --backend sglang --port 30000 \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 --request-rate 2 --host 127.0.0.1

# Test 3: mostly overlap (high rate, large batches)
python3 -m sglang.bench_serving --backend sglang --port 30000 \
    --dataset-name random --random-input-len 512 --random-output-len 256 \
    --num-prompts 200 --request-rate 10 --host 127.0.0.1
```

**Expected healthy output:** `accept len: X.XX` (>1) in decode logs when spec is active;
no `token_to_kv_pool_allocator memory leak` on idle; no shape-mismatch crashes.

---

## Key Design Notes for Future Changes

1. **`pop_and_process` runs after `get_next_batch_to_run`** — Any request finished by
   `pop_and_process` is still in the current batch. Always filter before using
   `req_pool_idx` if spec is enabled.

2. **`batch.output_ids` semantics differ by mode** — In non-spec mode it is a
   `[bs]` tensor (1 token/req). After spec verify it is `verified_id` (multi-token
   flat). Code that reads `batch.output_ids` must be aware of this.

3. **`kv_overallocated_freed` as release sentinel** — Use this flag rather than
   `batch.enable_overlap` to detect whether `release_kv_cache` has been called. It is
   immune to mode transitions and set atomically inside `pop_overallocated_kv_cache`.

4. **`prepare_for_decode` returns early for spec** — Any state that `prepare_for_decode`
   normally sets (`input_ids`, `out_cache_loc`, `seq_lens`) is the responsibility of
   `_prepare_for_speculative_decoding` when spec is active. Keep these two paths in sync.
