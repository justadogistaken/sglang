# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SGLang is a high-performance LLM/VLM serving framework. This fork (`auto_enable_suffix_spec` branch) extends upstream SGLang with **suffix speculative decoding** — a draft token proposal method using suffix trees built from prompt and generated tokens, backed by the `arctic-inference` library.

## Common Commands

```bash
# Install for development
cd python && pip install -e ".[dev]"

# For suffix decoding support
pip install arctic-inference==0.1.1

# Format code (isort + black on modified files)
make format

# Launch server
python -m sglang.launch_server --model-path <model> --port 30000

# Launch with suffix speculative decoding
python -m sglang.launch_server --model-path <model> \
  --speculative-algorithm SUFFIX \
  --speculative-num-draft-tokens 24

# Run a single test file
cd test/srt && python3 test_srt_endpoint.py

# Run a single test case
python3 test/srt/test_srt_endpoint.py TestSRTEndpoint.test_simple_decode

# Run the suffix decoding test
python3 test/srt/test_suffix_speculative_decoding.py

# Run a full test suite
cd test/srt && python3 run_suite.py --suite per-commit
```

## Architecture

### Multi-Process Pipeline

Requests flow through ZMQ IPC across three main processes:

```
HTTP Client
  → HTTP Server (FastAPI, main process)
  → TokenizerManager (main process)  [ZMQ]
  → Scheduler (subprocess per TP group)  [ZMQ]
  → DetokenizerManager (subprocess)  [ZMQ]
  → back to TokenizerManager → HTTP response
```

### Key Modules

| Module | Path | Role |
|---|---|---|
| `Engine` | `python/sglang/srt/entrypoints/engine.py` | Top-level Python API; spawns all subprocesses |
| `HTTP Server` | `python/sglang/srt/entrypoints/http_server.py` | FastAPI, OpenAI-compatible endpoints |
| `TokenizerManager` | `python/sglang/srt/managers/tokenizer_manager.py` | Tokenizes requests, dispatches via ZMQ |
| `Scheduler` | `python/sglang/srt/managers/scheduler.py` | Batch scheduling, KV cache management |
| `DetokenizerManager` | `python/sglang/srt/managers/detokenizer_manager.py` | Token IDs → text |
| `TpModelWorker` | `python/sglang/srt/managers/tp_worker.py` | Tensor-parallel GPU worker |
| `ModelRunner` | `python/sglang/srt/model_executor/model_runner.py` | Model forward passes, CUDA graph management |
| `RadixCache` | `python/sglang/srt/mem_cache/radix_cache.py` | Prefix caching via radix tree |

### Batch Data Structures

Requests are progressively wrapped: `ScheduleBatch → ModelWorkerBatch → ForwardBatch`.
All inter-process message types (dataclasses) are in `python/sglang/srt/managers/io_struct.py`.

### Scheduler Composition

`Scheduler` is composed from many mixins in `python/sglang/srt/managers/`:
- `SchedulerOutputProcessorMixin` — handles batch results for prefill, decode, and speculative decoding
- `SchedulerMetricsMixin`, `SchedulerProfilerMixin`, `SchedulerDPAttnMixin`
- `SchedulerPPMixin`, `SchedulerDisaggregationDecodeMixin/PrefillMixin`
- `SchedulerUpdateWeightsMixin`, `SchedulerRuntimeCheckerMixin`

### Speculative Decoding

All speculative algorithms are registered in `python/sglang/srt/speculative/spec_info.py` (`SpeculativeAlgorithm` enum):
`NONE`, `EAGLE`, `EAGLE3`, `NEXTN`, `STANDALONE`, `NGRAM`, `SUFFIX`.

**Suffix decoding** (this branch's key addition):
- `SuffixWorker` (`speculative/suffix_worker.py`) — extends `NGRAMWorker`, swaps `NgramCache` for `SuffixCacheAdapter`
- `SuffixCacheAdapter` (`speculative/suffix_cache_adapter.py`) — wraps `arctic_inference.suffix_decoding.SuffixDecodingCache` to match the `NgramCache` interface
- `SuffixVerifyInput` (`speculative/suffix_info.py`) — extends `NgramVerifyInput` with `SpecInputType.SUFFIX_VERIFY`
- `SuffixCacheServer` (`speculative/suffix_cache_server.py`) — optional HTTP server for distributed cache sync (RL rollout use cases)

The suffix cache builds suffix trees from prompt and output tokens, proposes draft tokens by frequency, and reuses the existing NGRAM tree-verify infrastructure for GPU verification.

## Repository Layout

```
python/           Main sglang Python package
  sglang/srt/     SGLang Runtime (server backend)
sgl-kernel/       Custom C++/CUDA kernel library
sgl-router/       Rust-based load balancer
test/srt/         Backend runtime tests
test/lang/        Frontend DSL tests
benchmark/        Benchmark scripts
```
