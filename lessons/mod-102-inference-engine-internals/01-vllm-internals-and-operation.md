# Chapter 01 — vLLM Internals and Operation

> You will run vLLM in production. When it misbehaves, "redeploy and hope" is not a strategy. This chapter is the five-layer map you read when something breaks.

## Motivation

vLLM is the default open-source LLM serving engine for a reason: it ships OpenAI-compatible endpoints, continuous batching, PagedAttention, prefix caching, and multi-GPU parallelism in one Python entrypoint. It is also fast-moving upstream, which means an engine that worked last quarter can OOM, deadlock, or regress TTFT this quarter after a routine `pip install vllm -U`.

To operate it under SLO you need two things this chapter delivers:

1. A **mental map of the source** at the exact resolution a production outage requires — low enough to point at a scheduler bug, high enough that you don't drown in individual kernel calls.
2. A **working knowledge of the engine-args surface** — the knobs that gate throughput, latency, and memory, and the knobs that are footguns if misread.

Everything below is a planning assumption; the vLLM source and release notes are the authoritative references (`resources.md`). <!-- needs-research: vLLM internal module names, class names, and flag defaults rotate between releases. Before quoting the exact names in a runbook, verify against the vLLM version your fleet runs. -->

## The five-layer anatomy

Every modern LLM engine — vLLM, SGLang, TensorRT-LLM — has roughly the same five layers. We use vLLM as the canonical example; the next two chapters cross-reference it.

```
    ┌───────────────────────────────────────────────────────┐
    │ 1. API frontend         (FastAPI / entrypoints)       │
    │    HTTP + SSE, OpenAI-compatible                      │
    └───────────────────────────────────────────────────────┘
                      │       request objects
                      ▼
    ┌───────────────────────────────────────────────────────┐
    │ 2. Engine (AsyncLLMEngine)                            │
    │    request admission, output streaming                │
    └───────────────────────────────────────────────────────┘
                      │       scheduling calls
                      ▼
    ┌───────────────────────────────────────────────────────┐
    │ 3. Scheduler                                          │
    │    continuous batching, chunked prefill, preemption   │
    └───────────────────────────────────────────────────────┘
                      │       KV allocation requests
                      ▼
    ┌───────────────────────────────────────────────────────┐
    │ 4. KV allocator / block manager  (PagedAttention)     │
    │    block tables, prefix cache, swap/recompute         │
    └───────────────────────────────────────────────────────┘
                      │       forward-step calls
                      ▼
    ┌───────────────────────────────────────────────────────┐
    │ 5. Model runner + Worker/Executor                     │
    │    CUDA graphs, attention backends, TP collectives    │
    └───────────────────────────────────────────────────────┘
```

### Layer 1 — API frontend

The OpenAI-compatible server (`vllm serve ...` or `python -m vllm.entrypoints.openai.api_server`). FastAPI-based. Exposes `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/metrics`, `/health`. Translates OpenAI request shapes into the engine's internal `SamplingParams` / `PoolingParams`.

What breaks here: SSE flushing (ITL jank; module 111), client-disconnect handling, authentication, request validation. Measured as *TTFT above P99 even though scheduler iteration time is normal* → check this layer first.

### Layer 2 — Engine (`AsyncLLMEngine`)

The async wrapper around the core engine step loop. Owns the request queue, dispatches each API request as a `SequenceGroup`, and streams outputs back as `RequestOutput` deltas. The engine step is the heartbeat: on each tick the scheduler picks a batch, the model runner executes a forward pass, and completed tokens stream out.

What breaks here: async hang (an `await` not completing because an engine step died silently), request-leak (requests enter but never finish), queue-depth explosion. Watch `vllm:num_requests_running`, `vllm:num_requests_waiting`.

### Layer 3 — Scheduler

The brain. On every engine step the scheduler:

- Admits new `waiting` requests into the `running` batch (continuous batching).
- Decides which running requests run a **prefill** iteration vs. a **decode** iteration (continuous batching mixes them; chunked prefill splits long prefills).
- Allocates / frees KV blocks via the block manager (layer 4).
- Preempts running requests when the KV pool is exhausted (swap-to-CPU or recompute).

Knobs: `--max-num-seqs`, `--max-num-batched-tokens`, `--enable-chunked-prefill`, `--max-num-partial-prefills`, scheduler-delay (release-dependent). The scheduler is where goodput is made or lost; module 103 goes deeper.

What breaks here: preemption-rate spike, chunked-prefill misconfigured, scheduler loop stalled by a slow worker. Watch `vllm:num_preemptions_total`, `vllm:scheduler_running_queue_size`, `vllm:time_in_queue_requests_seconds`.

### Layer 4 — KV allocator / block manager (PagedAttention)

The KV cache is split into fixed-size blocks (default `block-size=16` tokens; verify in your version). Each sequence has a **block table** mapping its logical tokens to physical blocks. Blocks can be shared across sequences for prefix caching (reference-counted, copy-on-write on divergence).

Operations:

- `allocate(seq)` — assign enough physical blocks to hold the prompt.
- `append_slot(seq)` — one more block (or a slot in the current block) for the next generated token.
- `free(seq)` — return blocks when a sequence finishes.
- `swap_out(seqs)` / `swap_in(seqs)` — move KV to CPU memory on preemption; swap back when memory frees.
- `fork(seq)` — copy-on-write for n-best / beam / speculative decoding.

What breaks here: fragmentation (free blocks exist but can't satisfy a request's contiguous-allocation need), prefix-cache cliff (the cache was warm, then a prompt-template change invalidates it), swap thrash (preemption rate climbs because CPU swap is slow). Watch `vllm:gpu_cache_usage_perc`, `vllm:cpu_cache_usage_perc`, `vllm:prefix_cache_queries_total`, `vllm:prefix_cache_hits_total`.

### Layer 5 — Model runner + Worker / Executor

The forward-pass executor. The `ModelRunner` turns the scheduler's batch into a padded, device-side tensor batch, calls the model's `forward()`, and returns sampled tokens. The `Worker` wraps a single process bound to a single GPU; the `Executor` wraps the multi-worker coordination (TP across one node, PP across stages, Ray-based multi-node, or the newer multiprocess executor — verify current default for your version).

The forward pass is where **CUDA graphs** apply. For a given input shape vLLM captures a CUDA graph of the decode step and replays it on every subsequent step of that shape — amortising kernel-launch overhead to zero. `--enforce-eager` disables graph capture (slower, but kinder when debugging a kernel or when shape variability causes recompile storms).

What breaks here: CUDA-graph recompile storm (shape variability forces repeated capture), `torch.compile` re-trace (if enabled; version-dependent), NCCL hang (a worker dies mid-collective and the surviving ranks block forever), slow-rank straggler (one GPU runs the forward at the pace of the slowest member of the TP group). Chapter 04 catalogues these.

## Request lifecycle, end to end

Trace a single `POST /v1/chat/completions` call through the layers:

1. **Ingress.** FastAPI handler validates the request, translates `messages` + `tools` into prompt tokens, builds `SamplingParams`. If streaming, the handler opens an SSE response.
2. **Submit.** The handler calls `engine.add_request(request_id, prompt_token_ids, sampling_params)`. The engine enqueues a `SequenceGroup` in the scheduler's `waiting` queue.
3. **Admit (scheduler).** On the next engine step the scheduler pulls from `waiting` while there is KV budget. Prefill is scheduled first (either as a single big prefill or as the first chunk if chunked prefill is on).
4. **Prefill (model runner).** The runner builds the input batch, runs a forward pass, writes KV into blocks allocated by the block manager, samples the first token.
5. **Stream.** The engine yields the first token back through `AsyncLLMEngine`'s output stream; the SSE handler flushes it. **TTFT** is measured here.
6. **Decode loop.** On each subsequent step the scheduler runs a decode iteration over the current `running` batch. One token per step per sequence. Each step's wall-time divided by batch size is **TPOT**.
7. **Preemption, maybe.** If a new large request lands and the KV pool is full, the scheduler evicts the lowest-priority running sequence (swap-out or free-and-recompute). The evicted sequence is paused; its remaining tokens are stalled until it is re-admitted. **Preemption rate** is measured here.
8. **Finish.** When the sequence hits a stop condition (EOS, max tokens, stop-string match, client disconnect), the engine emits the final chunk, frees the KV blocks, and closes the SSE stream. **E2E** is measured here.

If you can walk this lifecycle in a code tour (scheduler → block manager → runner → back) you are oriented. Module 103 and 104 drill into the hard edges.

## The engine-args surface

Below is the operator-level cheatsheet. **Every value is a planning assumption; verify against `vllm serve --help` for your version.** <!-- needs-research: default values and flag names shift between vLLM releases. The semantics below are stable; the defaults are not. -->

### Model, tokenizer, parallelism

- `--model <hf-id-or-path>` — the HF model id or local path. vLLM downloads from the HF Hub by default.
- `--tokenizer <id-or-path>` — override if the tokenizer differs from the model artifact.
- `--trust-remote-code` — required for models with custom modeling code. Review the code before enabling in production — it executes on load.
- `--tensor-parallel-size N` (`--tp-size`, `-tp`) — shard tensors across N GPUs within a node. N must divide attention head count and MLP hidden dim; typical values are powers of two bounded by GPUs-per-NVLink-domain (8 on HGX H100/H200/B200; 72 on GB200 NVL72).
- `--pipeline-parallel-size N` (`--pp-size`, `-pp`) — split layers across N stages. Use when TP alone can't fit the model inside one NVLink domain. Costs: pipeline bubbles for small micro-batches, extra inter-stage activation transfer.
- `--distributed-executor-backend <ray|mp>` — Ray for multi-node, multiprocess for single-node (verify current default). Ray adds a dependency and a cluster surface; multiprocess is lighter but single-node only.

### Capacity and batching

- `--max-model-len N` — the maximum (prompt + output) sequence length the engine will accept. Caps KV allocation per sequence. Set to the longest context your workload needs; **do not** set to the model's maximum unless the fleet has the KV budget for it.
- `--max-num-seqs N` — the hard cap on concurrent in-flight sequences. Serves as a batch-size ceiling for the scheduler.
- `--max-num-batched-tokens N` — the hard cap on tokens per engine step (prefill tokens + decode tokens combined). The single most load-bearing knob for the throughput-vs-latency Pareto. Larger → higher throughput, higher TTFT variance; smaller → tighter TTFT, lower throughput. Module 103 drills this.
- `--block-size N` — PagedAttention block size in tokens. Default typically 16 or 32 (version-dependent). Smaller blocks → finer-grained prefix sharing, more allocator metadata overhead; larger blocks → coarser sharing, less overhead.
- `--gpu-memory-utilization F` — fraction of GPU memory the engine may use (default around 0.9). The engine subtracts weight + activation workspace and gives the rest to the KV pool. Lower to leave headroom for other processes; raise to grow the KV pool (and tolerate more concurrency).
- `--swap-space N` — gigabytes of CPU swap space per worker for preempted KV. Larger swap → tolerates more preemption without recompute; too-small swap → recompute path kicks in.

### Caching and prefill shape

- `--enable-prefix-caching` — turn on prompt-prefix-level KV reuse (block-hash-based). First-class win for chat, RAG, agent traffic with shared system prompts. **Verify current default** for your version — recent releases default this to on.
- `--enable-chunked-prefill` — split long prefills into chunks interleaved with decode steps (SARATHI-shape). Keeps TPOT stable while a long prompt is still being prefilled; moderately raises TTFT on that single request. **Verify current default** — recent releases default this to on when supported.
- `--max-num-partial-prefills N` — cap on how many partial prefills can be interleaved into a single step.

### CUDA graphs vs. eager execution

- `--enforce-eager` — disable CUDA-graph capture. Use for:
  - Debugging a kernel-level crash (graphs make traces harder to read).
  - Workloads with high shape variability where graph capture thrashes.
  - Model-dev iteration where fast cold-start beats steady-state throughput.
- Absence of `--enforce-eager` → graphs on, which is the default. The decode step is captured as a CUDA graph per batch-size bucket; prefill is typically run in eager mode (version-dependent).
- `--compilation-config` / `-O<N>` — the compile-level knob (`-O0` disables, `-O3` the aggressive setting). Verify names in your version; this is a moving target as vLLM integrates `torch.compile` more deeply.

### LoRA

- `--enable-lora` — allow per-request LoRA adapters.
- `--max-loras N` — number of adapters that can be active in a single batch.
- `--max-lora-rank N` — ceiling on LoRA rank; sets memory per adapter.
- `--lora-modules name=path,...` — pre-register adapters at startup; requests select by `model=name`.
- Deep multi-adapter serving — Punica / S-LoRA depth — is module 106.

### Speculative decoding

- `--speculative-model <id-or-path>` — a smaller "draft" model proposing continuations.
- `--num-speculative-tokens N` — speculation window length. Larger window → more throughput *if* acceptance is high; otherwise a latency regression.
- `--speculative-decoding-method <draft|medusa|eagle|ngram>` — the algorithm. Flag name and set of supported methods are release-dependent; verify against the vLLM speculative-decoding guide (`resources.md`).

Operationally: speculative decoding is a conditional win. Always measure the **acceptance rate** per workload before leaving it on in production (module 108).

### Observability

- `--disable-log-requests` — turn off per-request access-log emission (loud at scale).
- `--metrics-port <N>` — Prometheus-format metrics endpoint. Instrument this; it is the primary SLO feed (module 110).

## A real `vllm serve` command, annotated

A production-shape launch for a 70B dense chat model on an 8× H200 node:

```bash
vllm serve \
    meta-llama/Llama-3.3-70B-Instruct \
    --tensor-parallel-size 8 \
    --max-model-len 32768 \
    --max-num-seqs 256 \
    --max-num-batched-tokens 16384 \
    --gpu-memory-utilization 0.92 \
    --swap-space 16 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --port 8000 \
    --metrics-port 8001
```

Reading each line against the five layers:

- `--tensor-parallel-size 8` → layer 5. All eight H200s share the forward pass over NVLink / NVSwitch inside the HGX node.
- `--max-model-len 32768` → layer 4 budget: the KV allocator reserves blocks against this ceiling per sequence.
- `--max-num-seqs 256`, `--max-num-batched-tokens 16384` → layer 3 scheduler caps.
- `--gpu-memory-utilization 0.92`, `--swap-space 16` → layer 4 pool sizing.
- `--enable-prefix-caching` → layer 4 (block hashing on).
- `--enable-chunked-prefill` → layer 3 (SARATHI-shape prefill interleaving).

This is the launch; it is not the end state. The command gets re-tuned every time traffic shape or model shifts. The engine args are a *dial board*, not a one-shot config.

## Debugging mental checklist

When vLLM misbehaves, walk the layers top-down:

1. **Ingress (layer 1):** are requests arriving at all? Is SSE flushing? Are 4xx/5xx hidden in the gateway?
2. **Engine queue (layer 2):** are requests entering `waiting` and leaving `running`? Is `num_requests_waiting` climbing without bound?
3. **Scheduler (layer 3):** preemption rate? queue time? Is chunked prefill thrashing?
4. **Block manager (layer 4):** GPU cache usage approaching 100%? Prefix cache hit-rate dropped off a cliff? Swap space exhausted?
5. **Runner / worker (layer 5):** step time climbing? Any CUDA errors in `dmesg`? NCCL log noise? One GPU utilisation visibly lower than the rest?

If a top-down pass doesn't localise the problem in five minutes, the next move is a code tour of the suspect layer against the release notes of the vLLM version you deployed. Chapter 04 catalogues the specific failure modes you'll meet.

## Boundary notes

- **The kernels** vLLM dispatches to — Flash Attention, FlashInfer, PagedAttention CUDA kernels, FP8 GEMM epilogues — are the `ai-infra-performance` peer's territory. You know *that they exist* and *which engine version pulled a new one in*. If a kernel is broken, open an issue to the engine (and, if the engine's using an external kernel library, upstream from there) — do not patch kernel source.
- **Autoscaling vLLM pods** on Kubernetes is module 109; this chapter covers one pod.
- **OpenAI-compatibility edge cases** (tool-call schema drift, structured-output edge cases, streaming chunk-boundary bugs) are module 111; this chapter covers the plumbing, not the API shape.

## Summary

- vLLM has a five-layer anatomy: API frontend → engine → scheduler → KV allocator → model runner + worker. Every production outage lives in one of those layers.
- The scheduler and the block manager are where goodput is made or lost. The runner and worker are where correctness and crashes happen.
- The engine-args surface is the dial board: parallelism (`--tensor-parallel-size`, `--pipeline-parallel-size`), capacity (`--max-model-len`, `--max-num-seqs`, `--max-num-batched-tokens`, `--gpu-memory-utilization`), caching (`--enable-prefix-caching`, `--enable-chunked-prefill`), graphs (`--enforce-eager`), adapters (`--enable-lora`), speculation (`--speculative-model`).
- Flag defaults rotate between releases. Treat the defaults quoted here as planning assumptions; verify against `vllm serve --help` for the version your fleet runs.
- Operation reads source. If you cannot find the scheduler loop, the block manager, and the model runner in vLLM's tree in under five minutes, you cannot run it under SLO.

Chapter 02 does the same five-layer walk for SGLang, where RadixAttention and the structured-generation front-end change the picture.
