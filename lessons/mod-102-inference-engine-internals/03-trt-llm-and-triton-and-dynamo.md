# Chapter 03 — TensorRT-LLM, Triton, and NVIDIA Dynamo

> vLLM and SGLang are "define the model in Python, serve it." TensorRT-LLM is "compile the model to a hardware-specific binary, then serve it." The operator model is fundamentally different. This chapter is the map.

## Motivation

TensorRT-LLM (TRT-LLM) is the NVIDIA-authored, open-source library that compiles a model into an optimised engine artifact (`.engine`) tuned for a specific GPU generation, numeric format, parallelism topology, and batch-size envelope. The compiled artifact is typically served through **Triton Inference Server** (via Triton's TRT-LLM backend) or through **NVIDIA Dynamo** for multi-node and disaggregated topologies.

Three operator consequences of the compile-ahead design:

1. **There is an explicit *build* step** between a model artifact and a serving fleet. Changes to the model, numeric format, TP/PP topology, or supported batch-size range all require a rebuild. Changes that are a config edit in vLLM / SGLang can be a 20-minute rebuild and re-upload in TRT-LLM.
2. **Max per-GPU throughput on NVIDIA hardware** is historically where TRT-LLM has been strongest. The compile step can bake in fused kernels, numeric-format choices (FP8 / FP4), and parallelism topology into one artifact.
3. **The hardware target is pinned.** An engine built for H200 is not interchangeable with one built for B200. The fleet-rotation plan has to include engine rebuilds — this is the operator overhead you pay for the throughput ceiling.

Everything below is a planning assumption; TRT-LLM release notes, the Triton docs, and the Dynamo repo are authoritative. Build-command names, backend config names, and Triton-side parameters evolve release-to-release — the semantics below are stable, the exact flag spellings may not be. Verify against the TRT-LLM / Triton / Dynamo versions your fleet runs.

## The build pipeline — convert → quantize → build → serve

Four steps, in order:

```
   HF checkpoint              TRT-LLM checkpoint          Quantized checkpoint
   (safetensors)      ─────►   (shards per rank)   ────►  (FP8/FP4/INT8/AWQ)
         │                              │                            │
         │                              │                            │
         │                              ▼                            │
         │                   calibration data,                       │
         │                   quantization recipe                     │
         │                              │                            │
         ▼                              ▼                            ▼
   ┌────────────────────────────────────────────────────────────────────┐
   │  trtllm-build  →   .engine  (hardware/topology/precision-pinned)   │
   └────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
                            Triton tensorrtllm_backend   ──► clients
                            (or Dynamo frontend)              over HTTP/gRPC
```

Each step is its own operational responsibility.

### Step 1 — Convert

Translate the model's weights into the TRT-LLM checkpoint format: a per-rank sharded directory with a config describing dtype, parallelism, and head/dim counts. One shard per TP (or TP×PP) rank; the shard count is baked in at convert time and must match the shard count the build and serving steps expect.

Tooling (names rotate — verify current): per-architecture Python conversion scripts in the TRT-LLM repo (`convert_checkpoint.py` for most families), or the `trtllm-build` CLI's conversion subcommand depending on the release. The convert step is cheap but version-sensitive — a convert script from an older release will often fail on a newer model family.

### Step 2 — Quantize (optional)

Apply a quantization recipe to the TRT-LLM checkpoint. Options typically include:

- **FP8** (E4M3 and/or E5M2) — on Hopper+ hardware with the full FP8 numeric path. Usually requires calibration data for scaling factors.
- **FP4 / NVFP4 / MXFP4** — on Blackwell+. Requires calibration.
- **INT8 / INT4** — with or without AWQ or GPTQ pre-quantization.
- **AWQ / GPTQ** — pre-quantized checkpoints produced by a separate tool, then converted into the TRT-LLM checkpoint format.
- **SmoothQuant** — activation-smoothing applied before INT8 quantization.

Tooling: NVIDIA's **TensorRT Model Optimizer** (`modelopt`) is the typical path for FP8 / FP4 calibration today; verify current recommendations against the TRT-LLM docs. AWQ / GPTQ checkpoints can come from the HF ecosystem and be imported.

The quantization-recipe-selection decision is deep enough to be its own module (mod-105). This chapter notes only *where in the pipeline it sits*. The payoff is engine artifacts half or quarter the size of FP16 with proportional throughput and memory-footprint wins — contingent on an accuracy-regression gate that confirms the recipe is safe for the serving workload.

### Step 3 — Build

Compile the (optionally quantized) checkpoint into a `.engine` artifact with `trtllm-build`:

```bash
trtllm-build \
    --checkpoint_dir ./llama-70b-fp8-tp8 \
    --output_dir   ./llama-70b-fp8-tp8-engine \
    --gemm_plugin   fp8 \
    --max_batch_size 256 \
    --max_input_len 32768 \
    --max_seq_len   65536 \
    --max_num_tokens 16384 \
    --use_paged_context_fmha enable \
    --paged_kv_cache enable \
    --tp_size 8
```

Operator-level reading of each flag:

- `--checkpoint_dir` / `--output_dir` — the convert-step output and the compile-step output.
- `--gemm_plugin` — selects the GEMM kernel path and precision. Values track NVIDIA's current kernel suite (`fp16`, `bf16`, `fp8`, `fp4` and variants depending on release). This is where the compile step hard-codes a numeric path.
- `--max_batch_size` — hard ceiling on concurrent sequences the compiled engine will serve.
- `--max_input_len` / `--max_seq_len` — context-length bounds compiled into the engine. Setting them larger raises activation workspace and KV-cache budget; setting them smaller caps what the engine can accept.
- `--max_num_tokens` — ceiling on prefill + decode tokens per step. The TRT-LLM equivalent of vLLM's `--max-num-batched-tokens`.
- `--use_paged_context_fmha enable` / `--paged_kv_cache enable` — turn on paged KV / paged context FMHA (fused multi-head attention). The paged-KV path in TRT-LLM is the equivalent of vLLM's PagedAttention block tables; it is what makes continuous batching workable.
- `--tp_size 8` — tensor-parallel degree compiled in. The resulting engine will *require* 8-way serving; it cannot be re-sharded at runtime.

Important consequences of the build step:

- The engine is **pinned to the TP (and PP) degree** compiled in.
- The engine is **pinned to the max shape envelope** (batch size, input length, sequence length) compiled in. Requests outside the envelope are rejected or truncated depending on the serving layer.
- The engine is **pinned to the hardware generation.** An engine built on an H200 host can run on H200 (and typically compatible-generation H100 with the same compute-capability path); it cannot be moved to Blackwell or Ada without a rebuild.
- The build takes minutes to tens of minutes depending on model and quantization. Budget time for it in the release pipeline.

### Step 4 — Serve

Two primary paths for the serving layer:

1. **Triton Inference Server + `tensorrtllm_backend`** — the mature, canonical path. Triton loads the engine from a model repository, exposes HTTP + gRPC endpoints, and runs TRT-LLM's **in-flight batching** scheduler on top of the compiled engine.
2. **NVIDIA Dynamo** — the newer orchestration layer for distributed and disaggregated serving. Dynamo can front TRT-LLM workers (as well as vLLM and SGLang workers) and manage prefill/decode split, KV transport, and cross-node routing.

The two are complementary. Triton serves a single-node fleet of TRT-LLM engines; Dynamo scales out across nodes and across prefill/decode roles.

## Triton Inference Server + the TRT-LLM backend

### The model repository

Triton's primary configuration surface is a filesystem layout — the **model repository** — not a flag set. A minimal TRT-LLM deployment has a model directory per logical model:

```
model_repository/
└── llama3-70b/
    ├── config.pbtxt         # Triton model config
    └── 1/                   # version subdirectory
        └── llama3-70b.engine  # the compiled TRT-LLM artifact (plus metadata)
```

In practice production deployments use the **ensemble** or **BLS (business-logic scripting)** pattern that chains a tokenizer step, the TRT-LLM engine, and a de-tokenizer step together. The TRT-LLM repo ships reference `tensorrtllm_backend` configs (`preprocessing`, `tensorrt_llm`, `postprocessing`, and the top-level ensemble `tensorrt_llm_bls`) you adapt and version. Verify current recommended configs against the backend docs.

### The TRT-LLM backend config — the knobs that matter

Within the backend's `config.pbtxt`, the parameters that gate operational behaviour:

- `max_batch_size` — must be ≤ the engine-time `--max_batch_size`.
- `max_tokens_in_paged_kv_cache` / `kv_cache_free_gpu_mem_fraction` — the KV-pool sizing knobs (equivalent to vLLM's `--gpu-memory-utilization` + related). One or the other, depending on how you configure the backend.
- `enable_kv_cache_reuse` — turn on prefix-cache-equivalent block reuse.
- `enable_chunked_context` / chunked-context parameters — in-flight batching's SARATHI-shape prefill chunking.
- `batching_strategy` / `batch_scheduler_policy` — in-flight batching mode (names rotate). This is the TRT-LLM equivalent of vLLM's scheduler loop.
- `decoding_mode` / speculative-decoding parameters — if the engine was built with speculative decoding support, this is where runtime speculative decoding is enabled.
- `lora_cache_optimal_adapter_size` and friends — multi-LoRA runtime settings when the engine supports it.

Every name above has rotated at least once across TRT-LLM backend releases; cross-check the exact `config.pbtxt` field names against the version of `tensorrtllm_backend` you are deploying, and treat them as the shape of the surface, not the final spelling.

### Triton-level operational knobs

Independent of TRT-LLM:

- `--http-port`, `--grpc-port`, `--metrics-port` — surface endpoints.
- Model-control mode (poll / explicit / none) — how model-repo changes are picked up.
- Logging verbosity, response caching, concurrency per model instance.
- OpenAI-compatible layer — Triton has grown an OpenAI-API-shape adapter (verify current state); for teams that need OpenAI-shape in front of Triton, this is the layer.

### In-flight batching

NVIDIA's term for continuous (iteration-level) batching inside the TRT-LLM backend. The scheduling model is the same as vLLM's and SGLang's — new requests join running batches between iterations rather than waiting for a batch window — but the implementation lives inside the C++ `tensorrt_llm` executor. Knobs are exposed through the backend `config.pbtxt`.

## Multi-GPU engine building

A multi-GPU TRT-LLM engine is a *set* of per-rank engine files, built in one `trtllm-build` run with `--tp_size N` (and optionally `--pp_size M`). The resulting output directory contains one `.engine` per rank, and the serving process launches `tp_size × pp_size` worker processes, each loading its own shard.

Operational implications:

- **The convert step must shard the checkpoint to match.** If you convert for TP=8 and build for TP=4 the build will error.
- **The serving container must be launched with the right number of GPUs visible.** Triton's model-instance group config and / or MPI launcher (`mpirun`) selects the ranks.
- **Changing TP degree at the fleet level is a rebuild.** You cannot resharded at runtime. A fleet of H200 nodes at TP=8 cannot absorb an engine built for TP=4 without a rebuild.

## FP8 / FP4 engine building

The engine artifact bakes in the numeric path. Three operator-level considerations:

1. **The hardware tensor-core path must exist.** FP8 requires Hopper+; FP4 (NVFP4 / MXFP4) requires Blackwell+. An FP8 build-flag on an older generation will either fall back to emulation (defeating the point) or fail; verify behaviour against the TRT-LLM release notes.
2. **Calibration data matters.** FP8 and FP4 recipes typically require calibration data to set per-tensor scaling factors. Calibration data should be *representative* of production traffic — not an arbitrary HF dataset.
3. **The accuracy-regression gate is non-optional.** The accuracy drop from FP16 → FP8 → FP4 is model- and workload-dependent. Promoting a quantized engine without an eval gate is a near-certain future incident. (The gate itself is mod-105 territory.)

The payoff when the path works: dramatically lower weight-memory footprint, higher per-GPU throughput, and a lower cost-per-token at the same SLO. The failure mode when the path doesn't work: silent accuracy regression that an eval surfaces only after rollout.

## NVIDIA Dynamo — the disaggregated serving layer

Dynamo is NVIDIA's open-source distributed inference serving platform. Its design axis is **multi-node orchestration** across heterogeneous workers, with first-class support for:

- **Disaggregated prefill/decode (P/D) serving** — a dedicated pool of prefill workers (compute-heavy, memory-bandwidth-insensitive) and a dedicated pool of decode workers (memory-bandwidth-heavy, latency-sensitive), connected by a KV transport that moves prefill-produced KV to the decode worker.
- **KV-aware routing** — the frontend router consults KV state (per-worker prefix availability, decode-worker queue depth) when picking the destination for an incoming request.
- **Multi-engine backends** — Dynamo can orchestrate TRT-LLM, vLLM, and SGLang workers. The orchestration layer is engine-agnostic; the engine remains your choice per workload.
- **KV transport (NIXL and other transports)** — the inter-worker KV byte stream, over NVLink / NVSwitch when possible and over RDMA when crossing nodes.
- **Autoscaling and LLM-aware scheduling** — scaling policies that understand prefill vs. decode load differently.

### When to introduce Dynamo

Not a day-one choice. Reach for Dynamo when:

- Single-node TRT-LLM / vLLM / SGLang has stopped scaling for a given model + traffic shape.
- The workload has distinctly asymmetric prefill vs. decode profiles (long prompts + short outputs favour P/D split; short prompts + long outputs don't).
- The model is large enough that a single node cannot host it and inter-node PP across nodes is the alternative. Dynamo's P/D split can beat naive PP for latency.

The deep disaggregation discussion — DistServe / Splitwise architecture, cross-node KV transport performance tuning, scheduling policy on top of the two pools — is module 107. This chapter notes *where Dynamo sits* in the stack: above the engines, below the API gateway, orchestrating.

### Reading Dynamo at the five-layer level

Dynamo bends the five-layer anatomy a little:

```
   ┌───────────────────────────────────────────────────────┐
   │ Dynamo frontend (HTTP, SSE)                           │
   │ KV-aware router, autoscaler                           │
   └───────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼────────────────────┐
          ▼                   ▼                    ▼
   Prefill worker       Prefill worker        Decode worker
   (TRT-LLM/vLLM/        ...                   (TRT-LLM/vLLM/
    SGLang)                                     SGLang)
          │                                        ▲
          └─────── KV transport (NIXL / RDMA) ─────┘
```

The familiar five layers still live inside each worker; Dynamo is the layer *around* them. Debugging in a Dynamo deployment adds two new question classes: *is the router making good decisions* and *is the KV transport actually fast enough*.

## Choosing: TRT-LLM standalone, through Triton, or via Dynamo

A short decision ladder:

- **Standalone TRT-LLM** (`trtllm-serve` or the executor API directly) — development and benchmarking. Not usually your production surface.
- **Triton + `tensorrtllm_backend`** — the canonical production path for single-node TRT-LLM serving. Choose this unless Dynamo adds something you need.
- **Dynamo + TRT-LLM workers** — multi-node scale, disaggregated P/D, heterogeneous worker pools. The operational cost is one more orchestration layer; the win is topology that the single-node path cannot reach.

## The operator cost of the TRT-LLM path, honestly

What you give up compared to vLLM / SGLang:

- **Agility.** A model-swap that's a `vllm serve <new-id>` on vLLM is a convert + build + repo-upload on TRT-LLM.
- **Hardware portability.** Compile for Hopper, deploy on Hopper. Rotating fleets to Blackwell is a rebuild.
- **Simpler debug surface.** You now debug across the convert step, the build step, the engine runtime, Triton's layer, and (if used) Dynamo's layer — more surface area for a bug to hide in.

What you get:

- **Peak NVIDIA throughput.** When the build pipeline fits the workload, the per-GPU throughput ceiling is often higher than Python-defined engines reach today.
- **First-class numeric formats.** FP8 and FP4 land here first and most completely.
- **Enterprise support path.** Via NIM / NVIDIA AI Enterprise if the organisation needs commercial support.

The decision is workload-shaped, not religious. Many fleets run two engines on purpose: vLLM / SGLang for fast-moving model lineups and TRT-LLM for the one or two stable, high-volume models where the throughput ceiling pays for the operator cost.

## Debugging mental checklist

Walk the pipeline top-down when something misbehaves:

1. **Convert step** — did the checkpoint convert correctly? Any shape warnings in the convert log? Does the shard count match the TP target?
2. **Quantize step** — was calibration done on representative data? Did the accuracy-regression gate pass?
3. **Build step** — did `trtllm-build` complete without warnings? Is the `--max_*` envelope wide enough for production traffic? Is the GEMM plugin consistent with the hardware target?
4. **Triton model repo** — are the three ensemble pieces (preprocessing, tensorrt_llm, postprocessing) wired correctly? Does the backend config match the engine's compiled envelope?
5. **Triton runtime** — is the backend reporting healthy? Is in-flight batching enabled? Are the KV-pool and chunked-context settings correct for the workload?
6. **Dynamo (if present)** — is the router making KV-aware decisions? Is prefill/decode load balanced? Is the KV transport saturating or underutilised?

## Boundary notes

- **New kernels** TRT-LLM ships (new FP4 attention, new fused epilogues) are the performance peer's territory. You adopt a release that includes them, test it, and roll it out. You do not patch the kernel source.
- **Multi-node disaggregated serving topology design** — DistServe / Splitwise reading, cross-node KV transfer cost analysis — is module 107.
- **Quantization recipe selection and the accuracy-regression gate** — module 105.
- **Autoscaling TRT-LLM pods on Kubernetes** — module 109.

## Summary

- TRT-LLM is a *compile-ahead* engine: convert → quantize → build → serve. The compiled artifact is pinned to hardware, numeric format, parallelism, and shape envelope.
- Triton Inference Server + `tensorrtllm_backend` is the canonical production serving path. The model-repository config is a filesystem-shaped equivalent of vLLM's launch flags; in-flight batching is the equivalent of continuous batching.
- Multi-GPU engines are built with `--tp_size` / `--pp_size`; the degree is baked in and cannot be resharded at runtime.
- FP8 (Hopper+) and FP4 (Blackwell+) require hardware tensor-core paths and calibration data; the accuracy-regression gate is non-optional before promotion.
- NVIDIA Dynamo sits above the engines as the disaggregated-serving orchestration layer: KV-aware routing, prefill/decode split, multi-engine backends, cross-node KV transport.
- The operator cost of the TRT-LLM path is agility and portability; the operator win is peak NVIDIA throughput and first-class low-precision numerics.
- Every flag spelling in this chapter is a planning assumption; verify against the exact TRT-LLM / Triton / Dynamo versions your fleet runs.

Chapter 04 steps back from any one engine and drills the seven production failure modes you'll see across all three.
