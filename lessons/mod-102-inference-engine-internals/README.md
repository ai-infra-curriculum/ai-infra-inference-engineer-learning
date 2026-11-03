# Module 102 — Inference-Engine Internals and Operational Fluency

> **Role:** AI Inference Engineer · **Level:** L35 (AI Infrastructure family) · **Hours:** 18

The first operational module in the AI Inference Engineer track. Module 101 taught you where this role sits and what vocabulary it uses. This module cracks open the three engines that role actually runs — vLLM, SGLang, and TensorRT-LLM — reads enough of each to debug a production outage, and drills the common failure modes you'll meet in on-call rotation.

By the end of it you can:

- Read the source of **vLLM**, **SGLang**, and **TensorRT-LLM** at the five-layer level a production outage demands: HTTP ingress → scheduler → KV allocator → model runner → worker/executor → streamed token egress.
- Build and operate a **vLLM** deployment end-to-end — launch flags, engine args, `--tensor-parallel-size`, `--pipeline-parallel-size`, `--max-model-len`, `--max-num-seqs`, `--max-num-batched-tokens`, `--enable-prefix-caching`, `--enable-chunked-prefill`, `--enforce-eager` vs. CUDA-graph capture, speculative-decoding flags, LoRA module config, OpenAI-compatible `vllm serve`.
- Build and operate an **SGLang** deployment — RadixAttention tuning, structured-generation front-end, data-parallel attention, cache-aware routing, zero-overhead batch scheduler configuration.
- Build and operate a **TensorRT-LLM** deployment — the four-step build workflow (convert → quantize → build engine → serve), Triton Inference Server backend wiring, in-flight batching config, FP8 / FP4 engine building, multi-GPU engine building, and NVIDIA Dynamo-based disaggregated serving.
- Characterise the deprecation state of **Hugging Face TGI** and the maintained alternatives (Hugging Face Inference Endpoints, Text-Generation-WebUI) at compare-point depth — not operational depth.
- Diagnose common engine failure modes — OOM on long-context, prefix-cache cliff, allocator fragmentation, CUDA-graph recompile storm, `torch.compile` re-trace, NCCL hang on worker crash, slow-rank straggler in TP — and apply the right knob or workaround.

> **Prerequisite:** [Module 101 — The Inference Engineer Position on the Infra Ladder](../mod-101-inference-engineering-position/README.md). The vocabulary (TTFT / TPOT / cache-hit / goodput) and the capability map from module 101 are assumed here.

## Boundary against `ai-infra-performance`

This module **reads engines to debug and tune**. It does **not re-implement the attention kernel they dispatch to** or author new fused kernels. Kernel craft lives at the sibling `ai-infra-performance` role (peer, level 35). Where the two tracks meet — Flash Attention upgrades, FP4 kernels, speculative-decoding verification kernels — the performance peer ships the kernel and lands it in engine source; you ship the engine upgrade in the fleet, the SLO comparison, and the rollback plan.

If you find yourself opening a `.cu` file to edit a GEMM epilogue, you've crossed the boundary. Open an issue to the performance peer instead.

## Chapters

1. [**vLLM Internals and Operation**](01-vllm-internals-and-operation.md) — the five-layer anatomy (engine → scheduler → KV allocator / block manager → model runner → worker-executor); request lifecycle from `vllm serve` HTTP ingress to streamed SSE egress; the engine-args surface you'll tune in production.
2. [**SGLang Internals and Operation**](02-sglang-internals-and-operation.md) — RadixAttention and its tree-shaped KV cache; the zero-overhead batch scheduler; data-parallel attention for MoE; cache-aware worker routing; the structured-generation front-end.
3. [**TensorRT-LLM, Triton, and Dynamo**](03-trt-llm-and-triton-and-dynamo.md) — the compile-ahead build pipeline (convert → quantize → build → serve); the Triton Inference Server TRT-LLM backend; in-flight batching; FP8 / FP4 engine building on Hopper and Blackwell; multi-GPU engine building; Dynamo as the disaggregated-prefill-decode orchestrator above a TRT-LLM / vLLM / SGLang worker fleet.
4. [**Engine Failure Modes and Triage**](04-engine-failure-modes-and-triage.md) — the seven production failure modes you'll see repeatedly (OOM, prefix-cache cliff, allocator fragmentation, CUDA-graph recompile storm, `torch.compile` re-trace, NCCL hang, slow-rank straggler) and the triage move for each.
5. [**TGI Deprecation and the Maintained Alternatives**](05-tgi-deprecation-and-maintained-alternatives.md) — Hugging Face TGI's deprecation signal and current state; HF Inference Endpoints and Text-Generation-WebUI as the maintained compare-points; the operator-level translation table from a TGI-shaped runbook to a vLLM / SGLang / TRT-LLM world.

## Exercises

The exercises take each chapter into a hands-on walkthrough you keep as a reference for the rest of the track:

1. [**vLLM scheduler and engine walkthrough**](exercises/exercise-01-vllm-scheduler-and-engine-walkthrough.md) — read vLLM source bottom-up, launch a local server, run a traffic probe, map metrics back to the code paths.
2. [**SGLang radix and scheduler walkthrough**](exercises/exercise-02-sglang-radix-and-scheduler-walkthrough.md) — exercise RadixAttention under a prefix-heavy workload, read the batch scheduler, tune cache knobs, verify cache-hit telemetry.
3. [**TRT-LLM engine build and Triton serve**](exercises/exercise-03-trt-llm-engine-build-and-triton-serve.md) — run the convert → quantize → build → serve pipeline against a small model and expose it through Triton.
4. [**Engine failure-mode triage drill**](exercises/exercise-04-engine-failure-mode-triage-drill.md) — reproduce each of the seven failure modes locally (or stage-replay them from logs) and author a one-page triage runbook per mode.
5. [**Deprecated vs. maintained engine landscape review**](exercises/exercise-05-deprecated-vs-maintained-engine-landscape-review.md) — audit TGI's current maintenance posture, place HF Inference Endpoints and Text-Generation-WebUI on the capability map, and author a TGI → vLLM / SGLang / TRT-LLM translation table.

## Resources

See [`resources.md`](resources.md) for primary / citable references for the engine sources, release notes, papers, and vendor documentation this module draws on.

## Where this fits in the curriculum

- **Before:** [Module 101 — The Inference Engineer Position on the Infra Ladder](../mod-101-inference-engineering-position/README.md)
- **After:** `mod-103-request-scheduling-and-batching` (goes deep on the scheduler and batching knobs introduced here) and `mod-104-kv-cache-management` (goes deep on the KV-allocator primitives introduced here).
- **Peer-track hand-off:** when the diagnosis in a drill lands on a kernel-level fix (new Flash Attention version, FP4 kernel bug), the hand-off is to [`ai-infra-performance-learning`](https://github.com/ai-infra-curriculum/ai-infra-performance-learning). This module teaches you to *recognise* that hand-off, not to execute it.
