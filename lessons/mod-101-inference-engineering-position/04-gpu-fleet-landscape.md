# Chapter 04 — The GPU Fleet Landscape at Operator Depth

> You pick the fleet before you pick the engine. This chapter is deep enough to make that pick without pretending you can author the kernels that run on it.

## Motivation

Hardware choices lock in your numeric-format matrix, your parallelism topology, and your supplier risk for 18–36 months. "We'll use Hopper" and "we'll use Blackwell" produce different FP8 vs. FP4 defaults, different NVLink topologies, different Dynamo-vs-TRT-LLM feasibility, and different cost per 1 M tokens.

The depth you need in this role is **operator-depth**, not **kernel-authoring depth**:

- You need to know: compute capability generation, memory capacity and bandwidth, intra-node interconnect, inter-node interconnect, numeric formats supported in hardware, matrix / tensor core capability tier, whether a given engine has mature support.
- You do not need to know: SASS instruction scheduling, warp-level primitive latencies, TMA (Tensor Memory Accelerator) programming contract at the instruction level, or how to design a kernel against WGMMA / UTCCMMA primitives. That's the performance peer at level 35.

If you catch yourself writing a kernel, you've crossed the boundary.

## The decision axes

A GPU-fleet decision reduces to:

1. **Compute per device** — matrix / tensor core generation, peak FLOPs at relevant precision (FP16, FP8, FP4)
2. **Memory per device** — HBM capacity (sets the model-and-KV-cache ceiling), HBM bandwidth (sets decode TPOT)
3. **Intra-node interconnect** — NVLink / NVSwitch generation, bandwidth, topology; whether TP across the full node is cheap or expensive
4. **Inter-node interconnect** — InfiniBand / RoCE generation, per-port bandwidth; whether PP / EP across nodes is cheap or expensive
5. **Numeric-format support** — which low-precision formats have hardware tensor-core paths (not emulation)
6. **Engine maturity** — which inference engines treat this hardware as a first-class target vs. "supported in principle"
7. **Supply, price, and procurement timeline** — the thing you'll actually spend most of your calendar on

## NVIDIA — the dominant datacentre vendor

### Hopper (H100 / H200 / HGX H100 / HGX H200)

Launched 2022 (H100) and 2024 (H200). The current workhorse of the production LLM-serving fleet.

- **Numeric formats** — introduced hardware FP8 (E4M3, E5M2) support alongside BF16 / FP16. Transformer Engine libraries from NVIDIA leverage FP8 end-to-end for both training and inference.
- **Memory** — H100 SXM: 80 GB HBM3; H200: 141 GB HBM3e with substantially higher bandwidth than H100. H200 is the "H100 with more memory" refresh.
- **Interconnect** — NVLink 4 (900 GB/s aggregate per GPU); HGX H100 / H200 ships 8 GPUs with NVSwitch for all-to-all intra-node bandwidth.
- **Form factors** — SXM (datacentre HGX boards), PCIe (lower bandwidth, lower TDP), NVL (dual-GPU boards for high-density).
- **Engine maturity** — first-class on vLLM, SGLang, TRT-LLM, Dynamo. Default assumed hardware target for most inference tutorials.

Operator fit: default production target today for anything that needs FP8 and either doesn't need (or doesn't yet want to pay for) Blackwell.

### Ada Lovelace (L40 / L40S / L4)

Datacentre line based on the Ada architecture. Lower memory capacity and lower NVLink topology than Hopper; stronger per-watt economics for smaller models.

- **Numeric formats** — FP16 / BF16, FP8 (hardware-accelerated), INT8
- **Memory** — L40 / L40S: 48 GB GDDR6; L4: 24 GB GDDR6. Not HBM — bandwidth is lower than Hopper HBM3.
- **Interconnect** — PCIe-only in most configurations; no NVLink Switch fabric. TP across many Ada GPUs is bandwidth-constrained.
- **Engine maturity** — fully supported by vLLM, SGLang, TRT-LLM for the models that fit.

Operator fit: smaller models, high density per rack, cost-constrained workloads where a Hopper SKU is overspec. Not a first choice for a 70B+ dense model or a long-context KV-heavy workload.

### Blackwell (B100 / B200 / HGX B200 / Blackwell Ultra / GB200 NVL72 / GB300 NVL72)

Launched 2024–2025. NVIDIA's current flagship generation.

- **Numeric formats** — hardware FP4 support (NVFP4 and microscaling MXFP4 variants) added on top of the Hopper FP8 / FP16 / BF16 matrix. The FP4 path is the signature capability of the generation.
- **Memory** — B200: 192 GB HBM3e; Blackwell Ultra: higher-still HBM3e capacity. <!-- needs-research: lock Blackwell Ultra per-GPU HBM capacity against the current NVIDIA spec sheet; the SKU matrix is still shifting. -->
- **Interconnect** — NVLink 5; much higher aggregate per-GPU bandwidth than NVLink 4.
- **GB200 / GB300** — rack-scale designs that pair Blackwell GPUs with Grace CPUs over NVLink-C2C. **NVL72** configurations unify 72 GPUs behind a single NVLink Switch domain, exposing them as a single very-large coherent compute pool for very-large-model serving and training.
- **Engine maturity** — TRT-LLM / Dynamo lead on day-one Blackwell support; vLLM / SGLang follow close behind. FP4 support in a given engine lags the hardware ship date by months; verify current state against release notes.

Operator fit: new-build fleets targeting the FP4 numeric path and very large models (dense 100B+, DeepSeek-V3-scale MoE, long context). The NVL72 rack is the SKU for workloads where cross-GPU NVLink bandwidth is the binding constraint.

### Vera Rubin — the pre-announcement generation

The announced next-generation NVIDIA platform after Blackwell. NVIDIA has publicly named Rubin / Rubin Ultra / Vera CPU as the planned successor; <!-- needs-research: production ship dates, exact HBM / interconnect specs, and the inference-engine support timeline all shift quarterly — refresh against NVIDIA GTC announcements before next cycle. --> for this role, Rubin is a *planning horizon* input (vendor roadmaps for capital-budget cycles) rather than a hardware you operate today.

Operator fit: factor into 18-month fleet-refresh plans; don't commit tooling to it until there is a shipped SKU and engine support.

## AMD — the credible second source

### Instinct MI300X / MI325X

The current production AMD data-centre GPUs. The strategic story: a credible second source to Hopper-class performance, pitched hard at the LLM-serving use case because of the very large HBM capacity per device.

- **Numeric formats** — FP16 / BF16, FP8, INT8; MI325X extends the FP8 matrix. <!-- needs-research: confirm FP4 support status on current and next AMD generation against AMD's ROCm matrix. -->
- **Memory** — MI300X: 192 GB HBM3; MI325X: 256 GB HBM3e. Very large per-device HBM capacity — a differentiating story for serving large dense models with fewer GPUs per deployment.
- **Interconnect** — Infinity Fabric for GPU-to-GPU inside a node; ROCm RCCL for the collective-communication library equivalent to NCCL.
- **Engine maturity** — vLLM, SGLang, and TRT-LLM-adjacent paths (via AMD's own stack) all have ROCm support with varying feature parity; LLM-serving on AMD has moved from "experimental" to "deployable" over the last 24 months. Feature parity to NVIDIA is **not** uniform across engines — verify your specific feature set (FP8, speculative decoding, prefix caching, LoRA, disaggregated serving) against the ROCm support matrix before committing.

Operator fit: workloads that benefit from large per-GPU HBM (serving a very large model on fewer devices); shops with a strategic second-source mandate; procurement situations where NVIDIA supply is constrained.

## AWS silicon — Trainium / Inferentia

AWS's own accelerator line, accessible only via EC2.

- **Trainium (Trn1 / Trn2)** — training-oriented; Trn2 is the current generation.
- **Inferentia (Inf1 / Inf2)** — inference-oriented; Inf2 is the current generation.
- **Neuron SDK** — the compile-ahead toolchain. Models are compiled to Neuron artefacts; the resulting artefact targets a specific accelerator generation.
- **Engine maturity** — vLLM has upstream Neuron support; AWS maintains a Neuron plug-in path. Feature parity with CUDA / ROCm backends is more limited.

Operator fit: AWS-committed shops with volume commitments on Neuron capacity; workloads that fit the Neuron-supported model lineup; cost-structure optimisations when EC2 pricing on Neuron is favourable vs. equivalent GPU SKUs.

## Google TPU — v5p / v5e / v6 (Trillium / Ironwood)

Google's custom accelerator, accessible via Google Cloud (and, for research, Kaggle / TRC).

- **v5p** — scale-up training TPU; v5e — serving-oriented, cost-optimised; v6 is the current generation branded as Trillium, with Ironwood as the inference-focused successor announced for 2025. <!-- needs-research: confirm Ironwood general availability date, inference-engine support, and compare the v6 Trillium vs. Ironwood positioning against the current Google Cloud docs — the TPU product line names rotate. -->
- **Interconnect** — ICI (Inter-Chip Interconnect) at pod scale; distinctive "pod-of-chips" topology that is logically very different from an NVIDIA 8-GPU HGX node.
- **Engine maturity** — vLLM TPU backend exists; JAX / Pallas kernels are the native path; Pathways and XLA compile models to TPU-executable artefacts. LLM-serving on TPU is deployable but the operator ergonomics (debugging, profiling, kernel-swap flexibility) differ enough from GPU-serving that your runbooks need separate chapters.

Operator fit: Google Cloud-committed shops; workloads that match the TPU-native model lineup; serving at pod scale where the ICI topology beats NVLink / IB equivalents.

## Operator-depth reasoning: what to actually care about

The four questions you should be able to answer for any proposed fleet SKU without opening the kernel source:

### 1. Can the model fit? (memory capacity)

Rule of thumb for a dense Transformer at inference: ~2 × parameter-count bytes for FP16 / BF16 weights, ~1 × for FP8, ~0.5 × for FP4. Add KV cache per in-flight sequence (sequence-length × 2 × num-layers × num-kv-heads × head-dim × bytes-per-element). Add activation workspace.

If the budget doesn't fit in a single device, you need TP / PP / EP across devices — which means an interconnect question (next).

### 2. Will the interconnect carry it? (NVLink / NVSwitch / IB / RCCL)

TP collective bandwidth is the sum of all-reduces across every transformer block for every forward step. On NVLink-connected GPUs inside a node, this is cheap. Across PCIe-only connected GPUs or across nodes without IB / RoCE at HGX-class bandwidth, TP throughput collapses.

Rule of thumb: TP sizes above the number of GPUs sharing a single NVLink Switch domain is usually a mistake. For HGX H100 / H200 / B200 that's 8; for GB200 NVL72 / GB300 NVL72 that's 72.

### 3. Does the numeric format match? (FP8 / FP4 feasibility)

- Hopper and MI300-class → FP8 is the aggressive-but-safe default for serving once you have an accuracy-regression gate (module 105)
- Blackwell → FP4 is in play if the engine and model combination supports it; else fall back to FP8
- Ada / older → stay on FP16 / BF16 for correctness unless you have the engine and model to validate FP8

An FP8 or FP4 claim without a hardware tensor-core path is emulation and does not deliver the throughput you want.

### 4. Does the engine support it as a first-class target?

"Supported" varies between "day-one on NVIDIA's reference blog" and "experimental PR not yet in a release". Before standardising a fleet on a hardware target, verify against the engine's release notes that the features *you depend on* (FP8, chunked prefill, prefix caching, LoRA, disaggregated serving) are release-grade on that hardware.

## Cross-vendor comparison lens

| Axis | NVIDIA Hopper / Blackwell | AMD MI300 / MI325 | AWS Trainium / Inferentia | Google TPU v5p / v6 / Ironwood |
|---|---|---|---|---|
| Engine support (vLLM / SGLang / TRT-LLM) | first-class, immediate | growing, varying by engine | vLLM + Neuron plug-in, narrower | vLLM TPU backend + XLA; narrower |
| FP8 hardware path | Hopper+ | MI300+ | <!-- needs-research --> | <!-- needs-research --> |
| FP4 hardware path | Blackwell+ | <!-- needs-research --> | <!-- needs-research --> | <!-- needs-research --> |
| HBM per device | up to ~192 GB (B200) | up to 256 GB (MI325X) | varies by generation | varies by generation |
| Intra-rack interconnect | NVLink / NVSwitch (up to 72-GPU NVL72) | Infinity Fabric | AWS proprietary | ICI pod |
| Open ecosystem for ops | large | moderate | AWS-only | GCP-only |
| Supply risk | demand-constrained | second-source story | locked to AWS capacity | locked to GCP capacity |

The table is the structure, not the answer. Numbers rotate; the axes don't. For every row, check the current spec sheet when you're actually making the call.

## Boundary notes

- The kernel-authoring side of hardware depth — "author a WGMMA GEMM against Hopper's TMA" — is the performance peer at level 35. You know *that it exists* and *what engine releases depend on it*.
- The driver / kernel-module / BMC side of hardware operations — NVIDIA driver versioning, DCGM configuration, GPU operator in Kubernetes — is partly the `ai-infra-engineer` prerequisite (level 20) and partly module 109 of this track. You know the operational interface.
- Procurement and capital planning — multi-year supply commitments, reserved instance pricing, cross-cloud arbitrage — is partly leadership scope (module 114's RFC) and partly the `ai-infra-architect` role above you (level 40).

## Summary

- Hardware decisions lock in numeric formats, parallelism feasibility, engine support, and 18–36 months of operator tooling.
- NVIDIA is the dominant target; Hopper is today's workhorse, Blackwell is the active flagship, Rubin is the planning horizon.
- AMD MI300 / MI325 is a deployable second source with large-HBM strengths and uneven feature parity.
- AWS Neuron and Google TPU are credible cloud-locked paths that pay off for cloud-committed shops.
- The four operator-depth questions — can the model fit, will the interconnect carry it, does the numeric format match, does the engine support it — are enough to pick a fleet without owning kernel-authoring depth.

Chapter 05 ties the vocabulary and the hardware map together into a platform charter.
