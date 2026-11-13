# Chapter 06 — Edge / Local / On-Device Inference: llama.cpp, MLX, MLC-LLM, ExecuTorch, Build-Time Decision

> Most of this curriculum is about inference that happens on your GPU. This chapter is about inference that happens on your user's phone. The decision between the two is a build-time architecture choice, not a runtime routing one, and the inference engineer is the person the product team asks when they are trying to make it. This chapter gives you enough of a compare-point on the four dominant on-device runtimes to answer well.

## Motivation

Several products in 2026 ship their inference on the client device, not through an API. The reasons stack up:

- **Privacy.** A health app, a journaling app, a therapy assistant, a password-less biometric check — the data must not leave the device. No network path, no serving platform, no logging.
- **Offline.** A travel app that drafts translations without connectivity; a productivity app that works on a plane; a field-service tool that runs in a basement.
- **Cost per user.** For a free-tier product with millions of users, serving every model call on central GPUs is a line item that can dominate the operating budget. Moving inference to the device shifts that cost to a zero-marginal-cost component (the user's hardware they already own).
- **Latency floor.** A network round-trip to a US-east data centre from Sydney is ~200 ms minimum. A local model has no round-trip; the first token appears when the model's first token appears.

The product team usually encounters this as "should we use llama.cpp for this?" or "is MLX viable on iPhones?" The inference engineer owns the compare-point: what each runtime is, what it runs well on, which model formats it consumes, and which of the product's constraints it meets.

By the end of this chapter you can:

- Name the four dominant on-device runtimes (llama.cpp, Apple MLX, MLC-LLM, PyTorch ExecuTorch) and the device / accelerator / model-format each targets.
- Walk the decision axes that drive server-vs-edge at build time (privacy, offline, cost per user, latency floor, model-footprint, device fragmentation, update cadence).
- Produce a one-page RFC that lets a product manager, a mobile engineering lead, and an inference engineer sign off.
- Call out the operational asymmetries you lose at the edge (fleet observability, atomic rollout, secret management, abuse prevention).

## The four dominant runtimes

### llama.cpp

- **Project.** <https://github.com/ggml-org/llama.cpp> (originated as `ggerganov/llama.cpp`; the canonical path is the `ggml-org` GitHub org as of this writing — re-check before pinning in a long-lived doc, since open-source project homes can shift).
- **What it is.** A C/C++ runtime for GGUF-quantized models. CPU-first, with GPU offload via Metal (Apple), CUDA (NVIDIA), ROCm (AMD), Vulkan, OpenCL, SYCL, and Vulkan-based mobile backends.
- **Model format.** GGUF — self-describing quantized tensor files. Quantization types include `Q2_K` through `Q8_0` and newer `IQ`-prefixed types; the per-weight bit budget ranges from ~2 bits (very small) to 8 bits (near-lossless).
- **Target devices.** Everything. Desktop (Linux, macOS, Windows), server (CPU and GPU), mobile (via Android NDK and iOS via Metal), embedded (Raspberry Pi, Jetson).
- **Strengths.** Broad model coverage (new architectures land quickly in `ggml`); broadest device support; the reference runtime for any "can this model run on arbitrary hardware" question.
- **Weaknesses.** Not the fastest runtime on any specific device (dedicated per-platform runtimes beat it on their home turf); mobile distribution requires bundling a native library per architecture; the GGUF format is not a standard you'd pick for a pure-Apple or pure-NVIDIA stack.

### Apple MLX

- **Project.** <https://github.com/ml-explore/mlx>.
- **What it is.** Apple's array / ML framework for Apple-silicon devices. Lazy evaluation, unified memory, Metal under the hood. Companion projects `mlx-lm` (LLMs) and `mlx-swift` (iOS integration).
- **Model format.** MLX-native (safetensors + model-specific metadata); loaders for Hugging Face repos; quantization via the MLX quantize utilities (4-bit and 8-bit block-wise).
- **Target devices.** Apple silicon only — M1/M2/M3/M4 Macs, iPads with M-series chips, iPhones (via Metal + Core ML / MLX-Swift integration).
- **Strengths.** Fastest open-source LLM runtime on Apple hardware; unified memory (no CPU↔GPU copy); first-party support means it keeps pace with Apple's accelerator roadmap.
- **Weaknesses.** Apple-only — if your product has a cross-platform mobile client, you can't use MLX for the Android half. Model coverage is smaller than llama.cpp (though the gap is closing fast).

### MLC-LLM (TVM-based)

- **Project.** <https://github.com/mlc-ai/mlc-llm>.
- **What it is.** An Apache TVM-based compiler for LLMs that targets every mobile and desktop GPU with a WebGPU / Metal / Vulkan / CUDA backend. The companion WebLLM project runs LLMs in the browser via WebGPU.
- **Model format.** MLC-compiled artefacts (per-target shared libraries plus weight files). The `mlc_llm` CLI compiles a Hugging Face model for a specific backend.
- **Target devices.** Mobile (iOS Metal, Android Vulkan), desktop (CUDA, ROCm, Metal), browser (WebGPU).
- **Strengths.** Only runtime that does *browser* LLM inference well via WebGPU; broad backend coverage; TVM compilation produces competitive kernels.
- **Weaknesses.** The compile step adds build-time friction (per-target compilation artefact); less mature quantization coverage than llama.cpp; the TVM dependency is a lot of machinery to pull into a mobile app.

### PyTorch ExecuTorch

- **Project.** <https://github.com/pytorch/executorch>. Documentation <https://pytorch.org/executorch/>.
- **What it is.** The on-device runtime from the PyTorch team, successor to PyTorch Mobile. Takes a PyTorch model through `torch.export`, lowers it via Edge Dialect to a backend-specific partitioned graph (XNNPACK, Core ML, Qualcomm QNN, MediaTek NeuroPilot, Vulkan), and ships the resulting `.pte` file plus a per-platform runtime.
- **Model format.** ExecuTorch `.pte` (a serialised Edge Dialect graph with weights).
- **Target devices.** Mobile (iOS via Core ML backend, Android via XNNPACK / Qualcomm QNN / MediaTek), embedded, desktop CPU.
- **Strengths.** Direct path from a PyTorch training / fine-tuning stack to the deployed on-device model; first-party integration with Core ML and the mobile accelerator drivers from Qualcomm and MediaTek; a serious contender for production iOS / Android LLM deployments in 2026.
- **Weaknesses.** Newer than the above three; the backend partitioner is still maturing for some model architectures; the tooling ergonomics are improving but not yet as smooth as llama.cpp's "download a GGUF, run."

### The decision-tree sketch

A rough mapping between product shape and runtime default:

```
Cross-platform mobile, broad model coverage, team familiar with native code
                           → llama.cpp
Apple-only (iOS + macOS), best-in-class Apple-silicon performance
                           → MLX (consider ExecuTorch + Core ML backend if PyTorch-native)
Browser / web, no app installation step
                           → MLC-LLM (WebLLM)
PyTorch-native training pipeline, production mobile with Core ML / QNN accelerators
                           → ExecuTorch
Research / desktop power user, maximal device coverage
                           → llama.cpp
```

The defaults above are rough and will shift year to year; use them as the starting point of the RFC, not the final answer.

## The axes that drive server-vs-edge

Six axes the RFC should evaluate:

### 1. Privacy and data-residency

- **Server.** Request text crosses the network to the serving fleet; the fleet logs sampled requests for debugging; the data lives in the cloud provider's infrastructure.
- **Edge.** Request text stays on the device; no network path; no server-side logs.
- **Question to ask.** Does the product carry data types (PHI, PII, financial, minors') where a server-side logging policy is a legal risk that an on-device execution model dissolves?

### 2. Offline / network-dependence

- **Server.** Zero functionality without network; zero functionality with slow network (TTFT inflates with RTT).
- **Edge.** Full functionality offline; latency is local.
- **Question to ask.** Is offline a product requirement or a product feature? A travel / field / basement / airplane use case usually pushes to edge.

### 3. Cost per user

- **Server.** Pay-per-request (your serving fleet's GPU-hours allocated proportional to usage); variable with user activity.
- **Edge.** One-time cost of distributing the model binary (download bandwidth) and ongoing model-update bandwidth; marginal cost per request is zero.
- **Question to ask.** For the target DAU and QPS-per-DAU, what is the yearly serving cost (module 110 chapter 04 for the formula), and does moving to edge save more than the engineering build cost?

### 4. Latency floor

- **Server.** Network RTT + queue + prefill + decode. In the best case (nearby region, warm prefix cache) can be hundreds of ms; in the worst case (far region, bursty load) seconds.
- **Edge.** Model load + prefill + decode on local hardware. Model load is one-time; prefill is proportional to context length; decode is proportional to response length. No network component.
- **Question to ask.** Is the product UX sensitive to a sub-100-ms first-token floor? Is the user on a 4G connection where even the best server adds 200 ms?

### 5. Model footprint

- **Server.** Any model size you can afford (70B, 405B, MoE). The user never downloads weights.
- **Edge.** The model must fit the device's storage and RAM. Current practical ceiling for a phone: 3–7B parameters quantized to 4 bits (1.5–4 GB on disk, 2–5 GB RAM). Desktop ceiling: 30–70B quantized.
- **Question to ask.** Does the product's quality requirement need a model that doesn't fit in the smallest supported device's budget? If yes, edge is off the table for that product unless you split (smaller model on-device + server for the quality-critical path).

### 6. Device fragmentation and update cadence

- **Server.** Single deployment target; update by canarying a new replica set; rollback with a config change.
- **Edge.** Many device classes (iOS vs. Android, different SoC generations, different RAM tiers); updates are gated by app-store release cycles (days to weeks) and user adoption (slow tail). A security-critical model regression discovered Monday takes until the user updates the app to reach them.
- **Question to ask.** Does the model need to be updatable in under a week? Is the product willing to maintain multiple model classes per device tier?

### Secondary axes to consider

- **Abuse prevention.** On-device models can be jailbroken, extracted, and run outside the app with no server-side policy to enforce. If the product requires moderation or policy enforcement, edge is harder.
- **Observability.** Edge inference emits no server-side logs by default. Diagnosing a model regression in the field means client-side telemetry with user consent and the privacy trade-off that implies.
- **Battery and thermal.** Running an LLM on a phone warms the phone and drains the battery. Sustained usage is a real UX constraint; episodic usage is fine.
- **Hardware-acceleration diversity.** The Apple Neural Engine, Qualcomm Hexagon NPU, MediaTek APU, and the Pixel TPU each want different model formats. ExecuTorch's backend partitioner and MLC-LLM's compilation-per-target are the two paths that handle this cleanly.

## A minimal server-vs-edge RFC template

The output of chapter 06 is a one-pager. A reasonable shape:

```
TITLE: On-device vs. server inference for <product feature>
AUTHOR: <inference engineer>
REVIEWERS: <product manager>, <mobile engineering lead>, <security / privacy>
STATUS: proposal | adopted | rejected

1. Context
   - The feature: one paragraph.
   - The model it needs (size, latency envelope, quality floor): one paragraph.
   - The current implementation (if any): one paragraph.

2. Decision axes (chapter 06 table)
   For each of privacy, offline, cost-per-user, latency-floor, model-footprint, device-fragmentation:
   - The product constraint.
   - The server-side answer.
   - The edge-side answer.
   - Winner for this feature.

3. Runtime choice (if edge)
   - llama.cpp / MLX / MLC-LLM / ExecuTorch — one row each, pro and con on this product's constraints.
   - Picked runtime: __
   - Model format and quantization: __

4. Trade-offs we accept
   - (if edge) Observability degrades — mitigated by __ client telemetry.
   - (if edge) Update cadence is app-store-gated — mitigated by __ server-side kill-switch / fallback.
   - (if server) Latency floor is RTT + server — mitigated by __ regional presence.
   - (if server) Per-user cost scales — mitigated by __ caching / batching discipline.

5. Rollout plan
   - Phase 1 (M+0): __
   - Phase 2 (M+1): __
   - Kill criteria: __
   - Rollback path: __

6. Open questions
   - __
   - __
```

Exercise 06 asks you to produce one of these end to end against a candidate product feature.

## Hybrid: on-device for the common case, server for the long tail

Not every decision is binary. Common hybrid patterns:

- **Small on-device + large server.** A 3B model on-device handles 80 % of user requests (short / simple). Hard requests (longer context, more complex reasoning) fall back to a server-hosted 70B. The client judges when to fall back; the server endpoint is the same as a pure-server deployment would be.
- **On-device first-token, server continuation.** The on-device model streams the first N tokens (fast TTFT), while a server-side request is dispatched in the background; if the server response arrives before the on-device model finishes, it takes over. Hides server latency behind the local model's TTFT.
- **On-device for private data, server for everything else.** Private context (user notes, health data) stays on the device; the on-device model handles any query that touches that context. Non-private queries go to the server for higher quality.

Hybrids multiply the operational surface (now you have two model stacks to maintain, not one) but are the common end state for products with a mixed workload.

## What you lose at the edge

Three capabilities the server-side stack from modules 102–111 provides that edge deployments do not:

- **Atomic model rollout and rollback.** A server-side canary (module 113 preview) is a config change; an edge canary is a staged app-store release that takes days to propagate and can be stuck when users don't update.
- **Centralised observability.** Server logs, traces, and metrics (module 110) tell you what the fleet is doing. Edge inference emits nothing by default; the only path to telemetry is opt-in client reporting, which carries a privacy cost.
- **Secret management, policy enforcement, abuse prevention.** API keys, rate limits, moderation (module 111 chapter 06; module 113 safety) all live at the server. An extracted on-device model has no server-side policy to enforce.

The RFC must call out which of these are being given up and what the mitigation is. "No server-side logging" is a deliberate feature if privacy is the driver; it is a disaster if observability was load-bearing.

## Three failure modes

### "We shipped a 7B on-device model; mid-range Androids thermal-throttle after five minutes"

Common cause: the quality bar was set against the flagship device; the long tail of mid-range SoCs can't sustain the throughput.

Fix: tier by device class — smaller model on mid-range, larger on flagship; or cap sustained usage; or fall back to server for sessions that exceed a thermal budget. The RFC's model-footprint axis should have caught this.

### "A model regression shipped; we can't roll back for a week"

Common cause: the product team expected to ship fixes at server-cadence; the app-store update cycle is days to weeks; some users never update.

Fix: a server-side kill-switch flag that the app checks on launch and can disable the on-device model path (falling back to server or degraded UX); build this into the first release, not after the incident.

### "The on-device model works great on wifi, badly on first launch"

Common cause: the weights are downloaded on first launch over the user's network; the download is slow or fails intermittently; the user sees "Loading…" for minutes.

Fix: ship the weights in the app bundle when the model is small enough; otherwise, a staged download with resume-on-failure and a clear progress UI; for large models, download on wifi-only with a user prompt.

## Summary

- Edge / on-device inference is a build-time architecture choice driven by privacy, offline requirements, cost-per-user, latency floor, model footprint, and device fragmentation / update cadence.
- Four dominant runtimes in 2026: llama.cpp (broadest coverage, CPU + GPU), MLX (Apple-only, best-in-class on Apple silicon), MLC-LLM (TVM-compiled, browser via WebGPU), PyTorch ExecuTorch (PyTorch-native, first-party mobile accelerator support).
- The inference engineer's deliverable is a one-page RFC that walks the six axes, picks a runtime, and names the trade-offs the product accepts.
- Hybrids (on-device + server) are common end states; they double the operational surface but match products with mixed workloads.
- At the edge you lose atomic rollout, centralised observability, and server-side policy enforcement; the RFC must call out mitigations for each.
