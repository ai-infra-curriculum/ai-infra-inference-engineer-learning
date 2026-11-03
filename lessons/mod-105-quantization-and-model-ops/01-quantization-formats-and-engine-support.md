# Chapter 01 — Quantisation Formats and Engine Support

> There is no single "quantised model." There is a two-dimensional grid of (what-is-quantised × how-many-bits × calibration-style) crossed with a hardware grid of (what-your-GPU-can-execute). The operator's first fluency is reading those grids without a reference open.

## Motivation

A product team asks "can we quantise this?" The engineer with answer-fluency replies, in sixty seconds: "On your H100s you have four live options — BF16 baseline, FP8 (weights-and-activations), weight-only INT4 via AWQ or GPTQ, and INT8 with SmoothQuant for activations if the model behaves. On Blackwell add FP4 / NVFP4 / MXFP4 as a fifth. vLLM supports all five; TRT-LLM supports all five with a build step; SGLang supports the first three. For most chat workloads today the FP8 route gives the best speed / memory / accuracy point. If accuracy regresses, drop to AWQ-INT4 next, not FP4."

That is this chapter. By the end you can draw the format × engine × hardware matrix on a whiteboard, name what each format quantises (weights only vs. weights + activations vs. KV), and know which hardware generation unlocks which Tensor Core.

<!-- needs-research: engine support for each format rotates between releases. Verify the specific matrix below against the exact engine version your fleet runs — vLLM, SGLang, and TRT-LLM all add and deprecate quantisation backends across releases. -->

## The three axes of a quantisation format

Any quantisation format is a point in three independent axes. Reading a format name without knowing the axes it names is how operators mis-match formats to engines.

### Axis 1 — What is quantised

- **Weight-only.** The weight tensors are stored in low precision; activations stay in BF16 / FP16. The GEMM is a mixed-precision operation that dequantises the weight on the fly. **AWQ**, **GPTQ**, naive weight-only INT8 — all weight-only. Weight-only halves (INT8) or quarters (INT4) model-weight HBM but does not change activation precision, so KV cache and intermediate activations stay full size.
- **Weight-and-activation.** Both weights and activations are stored in low precision; the GEMM runs natively at the quantised type. **FP8**, **INT8 with activation quantisation**, **SmoothQuant** (which is a *recipe* that enables INT8 weight-and-activation). These unlock the hardware's low-precision Tensor Cores — real speedup, not just memory savings.
- **KV cache.** A separate quantisation choice from weight / activation. FP8 KV cache (sometimes INT8 or INT4) halves the per-token KV byte-math from Module 104. Can be combined with any weight / activation precision above. Engines expose it as a separate flag (`kv-cache-dtype` in vLLM; equivalent elsewhere).

The three sub-axes compose. "FP8 weights, FP8 activations, FP8 KV" is one configuration; "INT4 weights (AWQ), BF16 activations, BF16 KV" is another. They are independent switches.

### Axis 2 — Precision (how many bits)

- **16-bit** — BF16, FP16. Baseline. BF16 is the standard today for training and most serving.
- **8-bit** — FP8 (floating point), INT8 (integer). Two different number systems; the hardware treats them differently. FP8 has two sub-formats: **E4M3** (more mantissa, less range — the default for weights and activations in most engines) and **E5M2** (more exponent, more range — used for gradients in training; occasionally for activations with heavy outliers). INT8 is a signed integer with per-tensor or per-channel scale.
- **4-bit** — FP4 (NVFP4, MXFP4 — see below), INT4. Native hardware support exists only on Blackwell+ for FP4; INT4 is a software format with on-the-fly dequant.
- **Sub-4-bit** (experimental) — INT2, 1-bit (BitNet). Research-stage; not production.

**Key hardware detail:** the GPU's Tensor Core must *natively support* the format for the throughput gains to materialise. INT4 on Ampere runs faster than BF16 only because the dequant-to-BF16-and-GEMM pipeline is memory-bandwidth-limited; the Tensor Core still runs BF16 GEMM. FP8 on Hopper runs native FP8 GEMM — the Tensor Core computes `fp8 × fp8 → fp32` directly. FP4 on Blackwell runs native FP4 GEMM.

### Axis 3 — Granularity

How finely you quantise. For a weight matrix of shape `[out_features, in_features]`:

- **Per-tensor.** One scale (and zero-point) for the entire matrix. Cheapest to store (one scalar per tensor), lowest fidelity. Fine for well-behaved weight distributions; often not fine for activations.
- **Per-channel.** One scale per output channel (`out_features` scales total). More fidelity, modest storage. Common for weights.
- **Per-token.** One scale per activation token. Dynamic — computed online. Common for activation quantisation under SmoothQuant / FP8.
- **Per-group.** Weights grouped in chunks along the input dimension (group sizes 32, 64, 128 are common); one scale per group. The AWQ / GPTQ default. Trades a small amount of storage for substantially better accuracy on sensitive layers.
- **Block / microscaling.** The MXFP4 / NVFP4 model: tiny blocks (typically 16 or 32 elements) each with their own scale, encoded in the data layout itself. Trades hardware complexity for very fine-grained scaling.

The granularity choice interacts with both the format and the hardware: Blackwell MXFP4 is 32-element microscaling baked into the Tensor Core; AWQ INT4 is group-128 software scaling the GEMM kernel honours during dequant.

## The hardware map — which Tensor Core unlocks which format

| Generation | GPU examples | Native Tensor Core formats | What this unlocks |
|---|---|---|---|
| Volta / Turing | V100, T4 | FP16, INT8 | FP16 training, INT8 inference (post-training quantisation) |
| Ampere | A100, A10, A30 | BF16, FP16, TF32, INT8, INT4 (A100), FP64 TC | BF16 serving default; INT8 weight-and-activation; INT4 weight-only (via software dequant) |
| Ada Lovelace | L4, L40, L40S, RTX 40xx | BF16, FP16, FP8 (E4M3 / E5M2), INT8 | First Ada-class consumer / inference-SKU FP8 |
| Hopper | H100, H200, H800 | BF16, FP16, FP8 (E4M3 / E5M2), INT8, TF32, FP64 TC | FP8 at datacenter scale; Transformer Engine; FP8-accumulate GEMM |
| Blackwell | B100, B200, GB200, GB300 | BF16, FP16, FP8, FP4 (NVFP4, MXFP4), INT8 | FP4 native; second-generation Transformer Engine; much larger HBM3e |
| AMD CDNA 3 | MI300X, MI325X | BF16, FP16, FP8, INT8 | FP8 serving on AMD; ROCm path |
| TPU v5e / v5p / Trillium | — | BF16, INT8 | TPU inference stack; non-NVIDIA path |

<!-- needs-research: AMD and TPU format support rotates; verify current ROCm / JAX-on-TPU support against vendor docs for the generation your fleet runs. -->

**The operator's rule:** before you choose a format, know which Tensor Core your target hardware exposes. FP4 on Hopper is a software fiction (it runs through dequant to FP8 or BF16); FP8 on Ampere is the same (dequant to BF16). You can still *store* the weights at that precision for HBM savings, but you will not get the throughput win.

## The format map — what each name actually means

### FP16 and BF16

The two 16-bit floating-point types. **FP16** has 5 exponent bits and 10 mantissa bits (range ±65 504); **BF16** has 8 exponent bits and 7 mantissa bits (range ±3.4e38, matching FP32). BF16's wider range is why modern training and serving default to it — fewer overflow pitfalls, same memory as FP16. For this module, "16-bit baseline" means BF16 unless stated otherwise.

### FP8 — E4M3 and E5M2

Introduced by Micikevicius et al. (Nvidia / Arm / Intel, 2022; see [`resources.md`](resources.md)) and shipped at scale on Hopper. The eight bits are allocated two ways:

- **E4M3** — 4 exponent bits, 3 mantissa bits. Range roughly ±448 (and no infinity encoding in the common variant — see the FP8 spec). Default for **weights and activations** on most engines; the better fidelity-per-bit choice for forward-pass tensors.
- **E5M2** — 5 exponent bits, 2 mantissa bits. Range roughly ±57 344 (with infinities and NaNs). Used in training for **gradients** (wider range matters more than fidelity there). In inference, used occasionally for activations in models with heavy outliers; rarely for weights.

Both require **per-tensor scale factors** stored alongside the data so values can be mapped into the FP8 representable range. On Hopper the Transformer Engine automates scale maintenance; manually configured engines must set the scales at calibration or runtime.

**Note on KV cache FP8:** when an engine says "FP8 KV cache," it is almost always E4M3 for K and V tensors with a per-head or per-channel scale. The accuracy impact on long-context recall is model- and workload-dependent; validate per-model (see chapter 04).

### FP4 — NVFP4 and MXFP4

Introduced with Blackwell (NVIDIA) and standardised across the Open Compute Project (OCP) ecosystem as part of the Microscaling (MX) formats specification. Two 4-bit floating-point variants:

- **NVFP4** — NVIDIA's variant, 4 bits per element in a block of 16 elements with one FP8 shared scale per block. Supported natively by Blackwell Tensor Cores for FP4 GEMM.
- **MXFP4** — OCP Microscaling format: 4-bit elements in a block of 32 elements with one shared E8M0 (8-bit exponent-only) scale per block. Also supported on Blackwell.

Both are *block-scaled*: the fine-grained-scale-per-block structure is what makes 4-bit precision tolerable in a modern Tensor Core without catastrophic accuracy loss. The block is part of the format, not a software choice; the hardware GEMM reads the block and its scale together.

**MXFP6 and MXFP8** are the 6-bit and 8-bit MX variants; MXINT8 is the integer form. These are also part of the OCP MX spec. Serving-side adoption of MXFP6 / MXFP8 is still emerging; FP4 is where the practical weight-compression wins live on Blackwell.

<!-- needs-research: NVFP4 vs. MXFP4 serving tooling matures through 2025–2026. Verify which variant your TRT-LLM / vLLM build actually emits and which the Blackwell / Vera Rubin generation you target natively executes. -->

### INT8

An 8-bit signed integer type. Two patterns:

- **Weight-only INT8.** Weights stored as INT8 with per-channel scale; activations stay BF16. Mixed-precision GEMM. Historically useful on Ampere where FP8 did not exist; largely superseded by FP8 on Hopper+.
- **Weight-and-activation INT8.** Both tensors are INT8. The GEMM runs on INT8 Tensor Cores (available since Volta). Historically throughput-competitive with BF16; accuracy depends heavily on activation calibration (outliers break per-tensor INT8 — see **SmoothQuant** in chapter 02).

INT8 remains useful for Ampere-only fleets, for AMD / Intel inference stacks that lack full FP8 support, and for edge / mobile where INT8 Tensor Cores are the standard.

### INT4 — weight-only, software-dequant

INT4 is almost always **weight-only**: weights stored at 4 bits with group-wise scale (and optionally zero-points), activations in BF16, GEMM kernel dequantises on load. The dequantise-and-GEMM fusion is where kernels like Marlin (Grootendorst et al., 2024; see [`resources.md`](resources.md)) and the AWQ / GPTQ kernel families earn their speedup — the compute is memory-bandwidth-bound at low batch size, and halving the weight bytes read halves the stall.

INT4 weight-only is the standard low-bit choice on Ampere / Ada / Hopper where native FP4 is unavailable. On Blackwell+, FP4 generally wins on both quality and throughput for the same weight-bytes; INT4 remains relevant for compatibility with existing checkpoints.

### AWQ — Activation-aware Weight Quantisation

A **weight-only INT4 recipe**, introduced by Lin et al. (MIT / NVIDIA, 2023 — see [`resources.md`](resources.md)). The core idea: not all weight channels matter equally for the output; identify the salient channels (those that interact with large activation magnitudes) and *move* the quantisation scale off those channels so they stay well-represented. Formally, apply a per-channel scaling transform `s` that is absorbed into the previous layer's output and the current layer's weights such that the quantised result tracks the full-precision result on the calibration activations.

**Operator takeaways:**
- AWQ produces a **weight-only INT4** checkpoint; activations stay BF16.
- Granularity: **group-128** is the common default (one scale per 128-weight group along the input dimension).
- A **calibration dataset** is required (small, ~128–512 samples typically) — see chapter 02.
- Output is a `.safetensors` with INT4 weights, FP16 scales, and the per-layer transform baked in.
- Supported in vLLM (via the AWQ kernel), SGLang, TRT-LLM (via a dedicated converter), and llama.cpp (via GGUF format with Q4 variants).

### GPTQ — Generative Pre-trained Transformer Quantisation

Another **weight-only INT4 (or INT3 / INT8) recipe**, introduced by Frantar et al. (ISTA, 2022; see [`resources.md`](resources.md)). Different mechanism from AWQ: GPTQ performs a **Hessian-aware column-block update** on each weight matrix, quantising one column block at a time and updating the remaining unquantised columns to compensate for the error introduced.

**Operator takeaways:**
- GPTQ produces a **weight-only INT4** checkpoint (same storage shape as AWQ).
- Granularity: **group-128** is the common default; `act-order` (reorder columns by activation importance) is a common additional flag.
- A **calibration dataset** is required.
- The algorithm is more sensitive to calibration data than AWQ in some reports; see chapter 02 for the operator-depth read.
- Supported in vLLM, SGLang (via the GPTQ kernel; `exllama` / `exllamav2` is a specific optimised backend), TRT-LLM (via a converter), and llama.cpp (via GGUF Q4 variants).

### SmoothQuant

A **weight-and-activation INT8 recipe**, introduced by Xiao et al. (MIT, 2022; see [`resources.md`](resources.md)). The problem it solves: in large language models, activation tensors have *channel-wise outliers* — a few channels have 100× the magnitude of the median. Per-tensor INT8 activation quantisation on such channels collapses to zero for the median channels. The SmoothQuant transform **redistributes the magnitude** between activations and weights per channel (multiply the activation channel by `1/s`, multiply the weight channel by `s`), smoothing the activation distribution so per-tensor INT8 works.

**Operator takeaways:**
- SmoothQuant produces a **weight-and-activation INT8** checkpoint.
- Granularity: per-tensor activation (post-smoothing), per-channel weight.
- A **calibration dataset** is required (to estimate per-channel smoothing factors).
- Supported in TRT-LLM primarily; vLLM / SGLang coverage varies — verify. Mostly superseded by FP8 on Hopper+.
- Historical importance: the first recipe to make INT8 serving of large LLMs reliable at scale.

### Other recipes worth naming briefly

- **HQQ (Half-Quadratic Quantisation):** another weight-only INT4 recipe, non-GPTQ-family. See `mobiusml/hqq`.
- **QLoRA / NF4:** a weight-only 4-bit format used primarily for fine-tuning (not serving). Mentioned so you can recognise and reject it in a serving context.
- **GGUF Q-formats** (Q4_0, Q4_K_M, Q5_K_M, Q6_K, Q8_0, etc.): the llama.cpp quantisation families; weight-only, software-dequant on CPU and increasingly on GPU. The chapter-03 cross-format work is where these come in.

## The engine × format matrix (as of recent releases)

The matrix drifts release-to-release. Verify against your pinned engine version; the shape below is the one you should expect.

| Format | vLLM | SGLang | TRT-LLM | llama.cpp (GGUF) | ONNX Runtime |
|---|---|---|---|---|---|
| BF16 / FP16 | ✅ | ✅ | ✅ | ✅ (via conversion) | ✅ |
| FP8 (E4M3) weights + activations | ✅ (Hopper+) | ✅ (Hopper+) | ✅ (Hopper+) | — | partial |
| FP8 KV cache | ✅ (flag) | ✅ (flag) | ✅ (flag) | — | — |
| FP4 (NVFP4 / MXFP4) | ✅ (emerging, Blackwell) | emerging | ✅ (emerging, Blackwell) | — | — |
| INT8 weights-and-activations | ✅ | ✅ | ✅ | — | ✅ |
| INT8 weight-only | ✅ | ✅ | ✅ | ✅ (Q8_0) | ✅ |
| INT4 weight-only (AWQ) | ✅ | ✅ | ✅ (converter) | partial | partial |
| INT4 weight-only (GPTQ) | ✅ | ✅ | ✅ (converter) | partial | partial |
| SmoothQuant (W8A8) | partial | partial | ✅ | — | — |
| GGUF Q4 / Q5 / Q6 / Q8 | partial (via GGUF loader) | partial | — | ✅ (native) | — |

<!-- needs-research: cells marked "partial" or "emerging" change release-to-release. Validate for your pinned engine. The authoritative sources: vLLM `docs.vllm.ai/en/latest/features/quantization/`, SGLang quantisation docs, TRT-LLM "model quantization" guide. -->

**How to read the matrix:**
- ✅ = first-class support, usually via a documented flag + converter.
- partial = supported for some models, not all; or supported via a plug-in that is not production-ready.
- "converter" means TRT-LLM requires an explicit convert → quantise → build step; it is not a runtime flag.
- Blank = not supported in this engine; use another engine or convert via GGUF / safetensors.

## Where KV cache precision fits

KV cache precision is a *separate* flag from weight / activation precision. The three common options:

- **BF16 / FP16 KV** — default. 2 bytes per token, per layer, per head, per head-dim-element.
- **FP8 KV** — 1 byte. Halves KV footprint. Available in all major engines on Hopper+ (verify flag). Accuracy impact depends on model (chapter 04 gates it).
- **INT4 KV** — 0.5 bytes with packing. Experimental; narrow engine support; non-trivial accuracy consideration. Not recommended without a model-specific validation.

Operationally: enabling FP8 KV halves the per-token KV footprint from Module 104's byte-math. On a 128 k-context deployment this is often what makes "does it fit" flip from no to yes. The chapter-04 accuracy gate here is the sign-off.

## The "which format should I pick" decision tree

The simplest version. Numbers are rules of thumb; validate per model.

```
1. Is accuracy regression tolerable at zero? (Hard eval requirement.)
   → stay BF16 / FP16. Skip quantisation.

2. Am I memory-bound (model does not fit) or throughput-bound (fits but slow)?
   → memory-bound: focus on reducing weight bytes first (INT4 weight-only via AWQ / GPTQ, or FP4 on Blackwell).
   → throughput-bound: focus on activation + weight quantisation to unlock low-precision Tensor Cores (FP8 on Hopper+; INT8 on Ampere).

3. What hardware am I on?
   → Hopper+ with no Blackwell: FP8 is the default. Fall back to AWQ-INT4 if accuracy regresses.
   → Blackwell+: FP4 (NVFP4 / MXFP4) is the default. Fall back to FP8 if accuracy regresses.
   → Ampere / Ada without FP8: AWQ or GPTQ INT4 for memory; SmoothQuant INT8 or weight-only INT8 for a modest step.
   → AMD / TPU / edge: separate matrix, use vendor docs.

4. Do I have a calibration dataset that matches production traffic distribution?
   → yes: AWQ / GPTQ / SmoothQuant become viable (they need calibration).
   → no: FP8 with per-tensor scales from an engine-side calibration pass; or BF16.

5. Is my serving stack vLLM / SGLang / TRT-LLM / llama.cpp?
   → cross-reference the matrix above. If my engine doesn't support my format of choice, pick the next-best supported format.

6. Can I combine with FP8 KV cache?
   → If memory-bound at long context: yes, almost always worth it (with accuracy gating).
```

The decision tree is a starting point. The chapter-04 accuracy gate is what tells you whether the picked format actually works for your model and workload.

## Common misreadings

- **"FP4 is just FP8 at half the bits."** No — FP4 is a *block-scaled* format where fine-grained scaling is baked into the data layout. The MXFP4 32-element block with an E8M0 scale is the mechanism that makes 4-bit precision tolerable at all. A naive "cast to FP4" that ignores the block scaling is useless.
- **"Weight-only INT4 gives me 4× speedup."** It gives you 4× *weight memory* savings. Throughput speedup depends on whether your workload is memory-bandwidth-bound (yes: close-to-4× on batch=1 decode) or compute-bound (no: BF16 compute still runs). Measure per workload.
- **"AWQ and GPTQ are interchangeable."** They produce the same storage shape (INT4 weight-only) but via different algorithms with different accuracy / calibration-sensitivity profiles. Pick based on your model's observed behaviour, not by preference.
- **"SmoothQuant is deprecated."** On Hopper+, mostly yes — FP8 solves the same problem (activation quantisation) with better hardware support. On Ampere-only fleets and for some non-NVIDIA stacks, SmoothQuant is still relevant.
- **"KV cache quantisation and weight quantisation must match."** No — they are independent. BF16 weights with FP8 KV is a valid and common configuration. FP8 weights with BF16 KV is also valid (though uncommon).
- **"If my engine supports it, it works."** Engines list format support in their docs; actual accuracy on *your* model requires the chapter-04 gate. "Supported" means "the kernel runs"; it does not mean "the output is still correct."

## Summary

- Every quantisation format is a point in three axes: **what is quantised** (weight-only / weight+activation / KV), **precision** (16 / 8 / 4 / sub-4 bits), and **granularity** (per-tensor / per-channel / per-token / per-group / block).
- Hardware generations unlock specific formats: **Ampere** for INT8 and INT4-via-software-dequant; **Hopper / Ada** for FP8 (E4M3 / E5M2); **Blackwell+** for FP4 (NVFP4 / MXFP4).
- The important named recipes are **AWQ** (weight-only INT4, activation-aware scaling), **GPTQ** (weight-only INT4, Hessian-aware column-block update), **SmoothQuant** (weight-and-activation INT8, outlier redistribution), and **FP8** (floating-point low-precision on Hopper+).
- **KV cache precision is independent** of weight / activation precision; typically BF16 or FP8.
- The engine × format matrix drifts across releases; verify per pinned version.
- The decision tree is **memory-bound vs. throughput-bound**, **what Tensor Core your hardware exposes**, **whether you have a calibration dataset**, and **what your engine supports**. The accuracy gate (chapter 04) is the authority on whether the choice actually works.

Chapter 02 moves from "what is each format" to "how does each algorithm actually produce one" — AWQ, GPTQ, SmoothQuant, and FP8 at operator depth, with the boundary to the performance peer for algorithm-design depth.
