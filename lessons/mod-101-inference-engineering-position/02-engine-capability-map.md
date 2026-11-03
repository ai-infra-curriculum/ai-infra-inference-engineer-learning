# Chapter 02 — The Inference Engine Capability Map

> You will spend the rest of this curriculum operating one or more of these engines. This chapter is the field guide.

## Motivation

"Which engine should we use?" is a bad question with a good answer. The good answer is: *what is the model, what is the fleet, what is the traffic shape, what is the operator skill-set, and what is the exit cost?* An inference engine is a long-lived operational commitment — your Prometheus dashboards, your runbooks, your canary workflow, and half your senior engineers' muscle memory all get specialised to it.

This chapter maps every major engine onto the same capability axes so you can make that commitment with eyes open — and so that when an upstream maintainer breaks something in the engine you picked, you know which neighbour you can borrow from.

## The capability axes

Every engine in this chapter is characterised along the same ten axes. Modules 102–114 go deep on each one; here we just place the engines.

1. **Scheduling model** — static batch, dynamic batch, continuous (iteration-level) batching, chunked prefill
2. **KV-cache primitive** — contiguous allocation, PagedAttention block table, RadixAttention prefix tree, tiered KV store
3. **Attention / fused-kernel path** — Flash Attention version, custom fused kernels, vendor library (cuDNN, cuBLAS, oneDNN)
4. **Quantization matrix** — FP16 / BF16, FP8 (E4M3 / E5M2), FP4 / NVFP4 / MXFP4, INT8, INT4, AWQ, GPTQ, SmoothQuant
5. **Parallelism** — tensor (TP), pipeline (PP), expert (EP), data (DP), sequence (SP) parallelism
6. **Disaggregated prefill–decode** — unified vs. P/D-split, cross-worker KV transport
7. **Multi-adapter serving** — static LoRA, dynamic LoRA swap, S-LoRA / Punica-style multi-LoRA
8. **Speculative decoding** — draft-target, Medusa, EAGLE, Lookahead, n-gram
9. **API surface** — OpenAI-compatible chat/completions/embeddings, Triton gRPC/HTTP, custom
10. **Hardware backends** — NVIDIA (Hopper / Ada / Blackwell), AMD (ROCm MI300 / MI325), AWS Trainium / Inferentia, Google TPU, Intel Gaudi, Apple Metal, CPU

Treat the axes as *claims to verify against today's docs* — engines change fast, and a feature that was experimental last quarter is often default this quarter. Any specific feature claim below is cross-checked by `resources.md`; before you commit, verify against the engine's current release notes.

## The open-source engines

### vLLM — the generalist

Open-source LLM serving engine originating from the Sky Computing Lab at UC Berkeley; now a community project under the PyTorch Foundation umbrella. The reference implementation of **PagedAttention** and **continuous batching** — primitives most other engines have since adopted.

- **Scheduling:** continuous batching, chunked prefill, prefix caching, preemption with swap/recompute
- **KV-cache:** PagedAttention (fixed-size blocks with a per-request block table)
- **Attention:** Flash Attention via the FlashInfer / Flash Attention backends; vLLM also ships its own CUDA / Triton kernels
- **Quantization:** FP16 / BF16; FP8 (E4M3) on Hopper+; FP4 / NVFP4 on Blackwell+; INT8 / INT4; AWQ, GPTQ, SmoothQuant via upstream model-compressor
- **Parallelism:** TP, PP, EP (for MoE), DP attention
- **Disaggregation:** P/D-disaggregation support present in V1; cross-worker KV transport via NIXL and plug-in backends
- **Multi-adapter:** dynamic LoRA loading via `--enable-lora`
- **Speculative decoding:** draft-target, n-gram, EAGLE (verify current state against release notes)
- **API:** OpenAI-compatible server (`vllm serve`) with `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`
- **Hardware:** NVIDIA (CUDA), AMD (ROCm), AWS Neuron (Trainium / Inferentia), Intel (CPU / XPU / Gaudi), Google TPU — coverage varies by feature

Operator profile: strong default choice when you want OpenAI-compatibility plus most modern serving features out of the box, and when you can tolerate upgrade churn in exchange for a fast-moving upstream.

### SGLang — the structured-generation specialist

Open-source engine from LMSYS. Known for **RadixAttention** (a radix-tree-indexed prefix cache), a zero-overhead batch scheduler, and a strong structured-output / constrained-decoding path.

- **Scheduling:** continuous batching, cache-aware routing between workers
- **KV-cache:** RadixAttention — prefix sharing indexed by token-sequence prefix, with eviction policies
- **Attention:** FlashInfer, Flash Attention
- **Quantization:** FP16 / BF16, FP8, GPTQ, AWQ (verify current list in release notes)
- **Parallelism:** TP, EP, DP attention (particularly strong on DP attention for MoE)
- **Disaggregation:** P/D-disaggregated serving mode available
- **Multi-adapter:** LoRA loading supported
- **Speculative decoding:** EAGLE and draft-target
- **API:** OpenAI-compatible plus an SGLang front-end language for structured programs
- **Hardware:** NVIDIA primary; AMD and other backends in various states of maturity

Operator profile: strong choice when the workload leans heavily on *prefix sharing* (chat with long system prompts, RAG with repeated retrieved context, agentic traffic with shared tool preambles) or when structured output / grammar-constrained decoding is a first-class requirement.

### TensorRT-LLM — the NVIDIA-native optimiser

NVIDIA-authored, open-source library that *compiles* a model into an optimised TensorRT engine (`.engine` artefact), typically served through **Triton Inference Server** or **NVIDIA Dynamo**. Different operator model than vLLM / SGLang: there is an explicit *build* step (convert → quantize → build), and the resulting engine is tied to a hardware target.

- **Scheduling:** in-flight batching (NVIDIA's term for continuous batching), chunked context
- **KV-cache:** paged KV cache, prefix caching
- **Attention / fused kernels:** NVIDIA's own kernel suite, Flash Attention variants
- **Quantization:** full NVIDIA numeric-format matrix including FP8 (E4M3 / E5M2), FP4 / NVFP4, INT8, INT4, AWQ, GPTQ, SmoothQuant — this is TRT-LLM's historical strength on NVIDIA hardware
- **Parallelism:** TP, PP, EP; multi-GPU engines
- **Disaggregation:** via NVIDIA Dynamo
- **Multi-adapter:** LoRA supported in the build / runtime path
- **Speculative decoding:** draft-target, Medusa, EAGLE (verify current support in release notes)
- **API:** served via Triton (HTTP / gRPC) or Dynamo
- **Hardware:** NVIDIA-only

Operator profile: strong choice when the fleet is NVIDIA-only, maximum per-GPU throughput matters, and the team can absorb the "compile engine per hardware target" workflow. Weaker fit when you need hardware portability or when the model lineup rotates fast enough that re-building engines is operationally painful.

### NVIDIA Dynamo — the disaggregated / distributed scheduler

NVIDIA's open-source **AI inference platform** for scaling generative-AI serving across many nodes. Dynamo focuses on *distributed inference orchestration*: disaggregated prefill–decode serving, KV-aware routing, LLM-aware autoscaling, and multi-engine support (it can front a vLLM, SGLang, or TRT-LLM worker). Think of Dynamo as the layer *above* the engine, not a replacement for it.

- **Scheduling:** request-level routing across prefill workers and decode workers
- **KV-cache:** cross-worker KV transport (NIXL and other transports), KV-aware routing
- **Disaggregation:** this is Dynamo's primary design axis
- **Multi-engine:** can orchestrate heterogeneous backends
- **Hardware:** NVIDIA primary

Operator profile: reach for Dynamo when a single-node engine (vLLM / SGLang / TRT-LLM) has stopped scaling and you need to split prefill and decode across different worker pools, or when a very large model needs multi-node serving topology coordination.

### Triton Inference Server — the multi-framework generalist

NVIDIA's long-standing, open-source, multi-framework model-serving server. Not LLM-specific — it serves TensorRT, PyTorch, ONNX, TensorFlow, Python-backend, and TRT-LLM models through a uniform HTTP / gRPC API.

- **Scheduling:** dynamic batching, sequence batching; LLM-specific scheduling via the TRT-LLM backend
- **API:** Triton HTTP / gRPC; OpenAI-compatible layer available via community/add-ons
- **Hardware:** NVIDIA primary (with CPU / other backends for non-LLM models)

Operator profile: the right choice when the fleet serves a *mix* of LLM and non-LLM models (classical ML, embeddings, vision) and a single serving plane is operationally cheaper than two. In an LLM-only shop, a dedicated LLM engine (vLLM / SGLang / TRT-LLM via Dynamo) is usually a tighter fit.

### llama.cpp — the edge / local inference baseline

C/C++ inference engine built around `ggml` tensor code, targeting CPU and heterogeneous accelerator paths (CUDA, Metal, Vulkan, HIP, SYCL). Owns the **GGUF** model format. Dominant choice for on-device, developer-laptop, and small-footprint deployment.

- **Scheduling:** server mode supports batching; the project's historical focus is single-stream decoding
- **Quantization:** rich GGUF quantization matrix (Q2–Q8 variants, imatrix calibration, K-quants)
- **API:** OpenAI-compatible server (`llama-server`) in addition to the C API
- **Hardware:** broadest hardware coverage of any engine in this chapter — CPU, NVIDIA, AMD, Apple Metal, Vulkan devices

Operator profile: not usually a production-fleet choice for GPU-based serving at scale, but the reference point for *compare-across-hardware* work and for edge / local deployment questions you will field from the AI Engineering consumer tracks.

### MLC-LLM — the compiler-first portable path

Open-source project built on **Apache TVM** / **Relax**. Compiles a model to a hardware-specific runtime, with particular strength on mobile GPUs, WebGPU, and the long tail of accelerators. Overlaps conceptually with TRT-LLM (compile-to-engine) but on an open, portable compiler.

- **Scheduling:** continuous batching in the serving runtime (`mlc_llm serve`)
- **Quantization:** 4-bit / 8-bit quantization paths
- **API:** OpenAI-compatible server
- **Hardware:** broad — NVIDIA, AMD, Apple, Qualcomm, WebGPU, mobile targets

Operator profile: niche in a datacentre GPU fleet; valuable when the serving footprint stretches to mobile, embedded, or in-browser targets, or when you need a hardware target that neither vLLM nor TRT-LLM covers well.

## The deprecated / archived historical compare-point

### Hugging Face TGI — Text Generation Inference

Historically one of the four major open-source LLM serving engines (alongside vLLM, SGLang, TRT-LLM). Hugging Face has signalled that TGI's maintenance posture has changed and that recommended serving for new deployments points elsewhere (`vllm`, HF Inference Endpoints, or the engine that fits the hardware). <!-- needs-research: confirm the exact current status of TGI (archived / maintenance-only / community-forked) and the Hugging Face announcement date before next revision; CURRICULUM.md references a 2026 archival signal from the sibling performance-engineer cycle. -->

Why we still teach it as a compare-point: a lot of production fleets still run TGI, a lot of blog posts and tutorials still cite it, and you will meet operators whose mental model is TGI-shaped. Knowing where TGI sits on the axes above lets you translate advice from a TGI world to a vLLM / SGLang / TRT-LLM world without getting burned.

## The proprietary / commercial compare-points

You will interact with these as either (a) *build-vs-buy* alternatives you evaluate in a vendor RFC (module 114), or (b) fall-back providers you route *through* when self-hosted capacity is short. You do not operate them, but you must be able to reason about them.

- **NVIDIA NIM** — NVIDIA Inference Microservices. Containerised, enterprise-supported serving built on TensorRT-LLM / Triton / Dynamo, delivered as pullable images from the NVIDIA registry. The managed-NVIDIA option for shops that want TRT-LLM-grade performance without owning the build pipeline.
- **Fireworks AI** — commercial inference provider. Public positioning emphasises a proprietary serving stack and OpenAI-compatible endpoints. <!-- needs-research: current product-specific feature claims (FireAttention variants, speculative-decoding offerings) change frequently; verify against vendor docs before committing. -->
- **Together AI** — commercial inference provider offering OpenAI-compatible endpoints for a catalogue of open-weight models.
- **Baseten** — model-deployment platform; serves models as auto-scaled endpoints with support for custom model code via Truss.
- **Modal** — serverless compute platform; not inference-specific but widely used to host custom inference workloads behind an HTTP endpoint.
- **Groq** — custom LPU hardware plus a managed inference service exposing OpenAI-compatible endpoints; distinctive latency profile driven by the hardware design.
- **Cerebras** — wafer-scale WSE hardware plus a managed inference service; distinctive throughput profile at long context, again driven by hardware design.

Routing rule for commercial compare-points: build-vs-buy, resilience (overflow routing when self-host is over quota), and workloads whose economics don't justify self-hosting. The module-114 RFC is where you make this decision formally.

## How to use the map

- **Picking an engine for a new workload** — read the capability axes against the workload. Long-context chat with high prefix reuse → the RadixAttention path (SGLang) deserves a close look. NVIDIA-only fleet maximising per-GPU throughput on a stable model lineup → TRT-LLM + Dynamo earns the compile-step tax. Heterogeneous hardware or fast-rotating model lineup → vLLM's generality usually wins.
- **Debugging an upstream regression** — know which other engines implement the same primitive differently. If vLLM's prefix-caching path is misbehaving, SGLang's RadixAttention path is a useful mental cross-check even if you cannot migrate.
- **Interviewing a candidate** — ask them to place a workload on the capability axes before they name an engine. "vLLM" as an answer with no axis-reasoning is a red flag.

## Summary

- The engine market has three serious open-source generalists for datacentre GPU serving (vLLM, SGLang, TensorRT-LLM) plus NVIDIA Dynamo and Triton as the orchestration / multi-framework layer around them.
- llama.cpp and MLC-LLM cover edge / portable / compiler-first use cases that still matter for the role.
- TGI is a historical compare-point; you must know it to read older runbooks and blog posts.
- The commercial stacks (NIM, Fireworks, Together, Baseten, Modal, Groq, Cerebras) are build-vs-buy references and overflow routes, not day-to-day operational targets.
- You compare engines along a fixed set of axes — scheduling, KV cache, kernels, quantization, parallelism, disaggregation, adapters, speculative decoding, API, hardware. Everything else is marketing.

Chapter 03 locks in the vocabulary that labels those axes.
