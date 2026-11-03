# Chapter 02 — Quantisation Algorithms at Operator Depth

> You will not re-implement AWQ. You will decide between AWQ and GPTQ on a Tuesday, defend the choice at review on Thursday, and roll back from the one that regressed on Friday. For all three days you need to know what the algorithms actually do.

## Motivation

Chapter 01 named the formats and their engine support. This chapter opens the four algorithms the operator will actually produce or consume — **AWQ**, **GPTQ**, **SmoothQuant**, and **FP8** — and goes one layer below "what is the output format" to "what does the recipe actually do, and what are the knobs." The goal is not kernel-depth; the goal is operator-depth. When a quantised variant fails the chapter-04 accuracy gate, you need to be able to look at the recipe knobs — calibration dataset, group size, granularity, number of samples — and ask the right next question.

The algorithm-design depth itself — "should the Hessian-approximation in GPTQ use diagonal or block-diagonal, should AWQ absorb the scale into the preceding norm or the preceding linear" — is kernel craft and lives at the performance peer at [`ai-infra-performance-learning`](https://github.com/ai-infra-curriculum/ai-infra-performance-learning). This chapter teaches you how to recognise an algorithm-depth question and route it there, not how to answer it.

<!-- needs-research: tool implementations for each algorithm vary (AutoAWQ, AutoGPTQ, llm-compressor, modelopt, TensorRT-Model-Optimizer, etc.). Flag spellings and default group sizes drift. Verify against your pinned tooling version. -->

## The structural picture — forward pass with quantised weights

A transformer forward pass is dominated by GEMM operations: `Y = X · W^T + b`. In the quantised world:

```
Y = X · dequant(W_q, scales, zero_points)^T + b
```

Where `W_q` is the stored low-precision tensor and the dequantisation is either (a) done by a fused kernel just before the GEMM (weight-only INT4; AWQ / GPTQ), or (b) absorbed into a native low-precision GEMM (FP8 / FP4; INT8 Tensor Cores). The quantisation algorithms differ in **how `W_q`, `scales`, and `zero_points` are chosen**, not in what the forward pass looks like at inference.

For weight-and-activation quantisation, `X` is also quantised, and `scales` have an activation-side component the engine (or an earlier calibration pass) computed.

## AWQ — the algorithm, operator view

### The idea in one paragraph

Weights interact with activations channel-wise. Some weight channels pair with large-magnitude activation channels; those channels dominate the output. If quantisation error on a dominant channel is large, the output degrades. AWQ's observation is that you can **move the scale** off the dominant weight channels, into the preceding layer, by picking a per-channel scaling vector `s` that satisfies `W' = W * s`, `X' = X / s`, and then quantise `W'` instead of `W`. The weight tensor's dynamic range is now more uniform, so quantisation error on the dominant channels is reduced. The `X / s` operation is absorbed into the preceding layer's output transformation (bias / layer-norm scaling), so inference-time cost is zero.

The scale `s` is chosen per channel to minimise reconstruction error on a **calibration set** — a few hundred prompts whose activations are measured.

### The knobs the operator sees

- **`bits`** — 4 or 3 (3-bit is experimental; 4-bit is standard). Determines the stored precision.
- **`group_size`** — the group-wise granularity along the input dimension. **128 is the near-universal default.** Smaller (64, 32) → more scales, slightly larger artefact, better fidelity. Larger (256) → less storage, worse fidelity. The paper and most tools default 128.
- **`zero_point`** / **`sym` vs. `asym`** — symmetric quantisation (no zero-point, one scale per group) vs. asymmetric (scale + zero-point per group). AWQ's reference implementation uses asymmetric; some kernels only support symmetric. **Verify your target engine's kernel requirements before producing an asymmetric AWQ.**
- **`calib_data`** — the calibration dataset. Typically 128–512 samples of text (or multimodal input for VLM models). The samples must match the distribution of production traffic in important ways — see "calibration dataset" below.
- **`version`** — AWQ has had revisions; `gemm` vs. `gemv` kernel variants, static vs. dynamic scale derivation. Verify tooling version.
- **`w_bit`** vs. **`q_group_size`** — AutoAWQ's specific flag spellings. Different tools use different names; the semantics are the above.

### Operator takeaways

- **AWQ is `W' = W * s`, then quantise `W'`.** The scale `s` is computed to minimise reconstruction error on calibration data; the per-channel scale is then **absorbed** into the preceding layer so inference-time overhead is zero.
- **Calibration data matters, moderately.** AWQ is less sensitive to calibration-set choice than GPTQ in common reports; it still matters.
- **Group-128 asymmetric INT4 is the default.** If accuracy regresses, try group-64 before dropping to GPTQ.
- **Produces a `.safetensors` plus a scale / zero-point table.** The engine (vLLM / SGLang / TRT-LLM / etc.) must have an AWQ-aware kernel to run the dequant-and-GEMM fusion; without it, you lose most of the throughput win.

The algorithm-depth question — "why does AWQ work better than naive weight-only INT4 on channels with high-magnitude activations" — is kernel craft. The operator-depth question — "my AWQ variant regresses on code-generation but not on dialog" — is this module (chapter 04 gates it; the fix is almost always a calibration-data change).

## GPTQ — the algorithm, operator view

### The idea in one paragraph

Weight-only INT4 again, but a different mechanism. Instead of a per-channel scale transform (AWQ), GPTQ **quantises one column block of the weight matrix at a time** and, after quantising each block, **updates the remaining un-quantised columns** to compensate for the error the quantisation of this block introduced. The update is **Hessian-aware**: GPTQ computes an approximate second-order correction using the inverse of the empirical Hessian of the reconstruction loss (computed on calibration activations). The compensation push error into the still-un-quantised columns, which get their own chance to absorb or further propagate it. By the end, the quantised weight matrix minimises reconstruction error on the calibration set to a much higher degree than naive round-to-nearest.

### The knobs the operator sees

- **`bits`** — 4, 3, or 2 (2-bit is research; 4-bit is standard).
- **`group_size`** — same semantics as AWQ. **128** is the default; **-1** means per-column (no grouping, maximum fidelity, more scales).
- **`act_order` (also `desc_act`)** — reorder columns by activation importance before quantisation. Improves accuracy at the cost of a slightly more complex dequant-time index. Verify the engine's AWQ kernel supports `act_order`; some kernels (Marlin's older variants) require `act_order=False`.
- **`sym` vs. `asym`** — same as AWQ.
- **`damp_percent`** — the Hessian damping factor (used in the inverse approximation to improve numerical stability). Default 0.01; rarely changed by operators.
- **`calib_data`** — same role as AWQ; usually 128–1024 samples. GPTQ is often reported as **more sensitive to calibration-set choice** than AWQ.
- **`blocksize`** — the column block size for the per-block quantisation loop. Default 128; algorithm-design knob, rarely touched.

### Operator takeaways

- **GPTQ is sequential: quantise block, update remainder, repeat.** The result is a weight matrix whose quantised form has minimum reconstruction error on the calibration set to within the algorithm's approximations.
- **Calibration data matters, substantially.** Switching from `wikitext` to a code corpus changes which columns look salient. If your production traffic is code-heavy, calibrate on code.
- **`act_order` matters at the margin.** Turn it on if your engine supports it and you need every point of accuracy; turn it off if the engine kernel doesn't support it.
- **Produces a `.safetensors` with the same storage shape as AWQ** (INT4 weight-only, group-wise scales). Different *content* — the quantised values are different — but the file layout is compatible with most AWQ-aware engines, as long as the engine's loader knows which flavour it is reading.

The algorithm-depth question — "why is Hessian-aware compensation stable at group-128 but unstable at per-column without damping" — is kernel craft. The operator-depth question — "GPTQ vs. AWQ, which should I ship" — is: produce both, run both through the chapter-04 gate, pick the one that passes with better margin on your canary.

## SmoothQuant — the algorithm, operator view

### The idea in one paragraph

Activation tensors in LLMs have **channel-wise outliers**: a few channels have 100× the magnitude of the median. Per-tensor INT8 quantisation of activations clips the median to near-zero because the one scale has to span the outlier. SmoothQuant does the AWQ-like move but on activations: pick a per-channel smoothing factor `s`, multiply activations by `1/s` (absorbed into the preceding layer), multiply weights by `s`. Choose `s` to balance the dynamic ranges: smaller when activation outliers are heavy, larger when weight outliers are heavy. Then quantise activations per-tensor to INT8 (reasonable now) and weights per-channel to INT8 (as before).

### The knobs the operator sees

- **`alpha`** — the migration strength. `alpha=0.5` is a common default; `alpha=1.0` moves all outlier burden to weights; `alpha=0.0` leaves activations unchanged. Different models have different optimal `alpha` values (often in the 0.5–0.85 range).
- **`calib_data`** — activation statistics are computed from calibration data; the same discipline as AWQ / GPTQ.
- **`per_channel`** / **`per_token`** — activation granularity. `per_token` is dynamic (computed online at inference), typically better quality, modestly more expensive at runtime.
- **`weight_bits`** / **`act_bits`** — SmoothQuant is usually W8A8 (both 8-bit), but higher-order variants exist.

### Operator takeaways

- **SmoothQuant is a *preparation step* for INT8 serving, not a new format.** The output is a modified weight tensor + an activation-smoothing recipe the engine applies at runtime. The engine's INT8 kernel then runs on the smoothed tensors.
- **`alpha` is the one knob you tune per model.** Lower `alpha` for models with heavy activation outliers; higher for models with heavy weight outliers.
- **Mostly superseded by FP8 on Hopper+.** FP8 handles activation outliers natively via per-tensor scale factors; SmoothQuant's redistribution is unnecessary when the hardware already has an 8-bit floating-point representation.
- **Still relevant on Ampere / Ada without FP8, and on AMD / Intel inference stacks.**

The algorithm-depth question — "what is the right `alpha` for Llama-3-70B's late-layer MLPs" — is kernel craft. The operator-depth question — "SmoothQuant-INT8 vs. BF16, which should I ship on this A100 fleet" — is this module.

## FP8 — the algorithm, operator view

### The idea in one paragraph

Native 8-bit floating-point in hardware. On Hopper, the Tensor Core computes `fp8 × fp8 → fp32` directly; the result is cast back to BF16 / FP8 for the next op. Because FP8 is floating-point, each tensor needs a **per-tensor scale factor** (and sometimes a per-row / per-column scale for better fidelity) so values are mapped into the FP8 representable range. The scale factors are either (a) **computed statically** at calibration time (same discipline as INT8 calibration) or (b) **maintained dynamically** (the Transformer Engine automates this, keeping a running max of the tensor magnitude and updating the scale each step).

### The knobs the operator sees

- **Sub-format**: **E4M3** for forward-pass tensors (weights, activations); **E5M2** for backward-pass (training only). Serving-only inference pipelines use E4M3 almost exclusively.
- **Scale strategy**: **static** (calibrated at convert time; the scales are baked into the artefact) vs. **dynamic** (computed at runtime from running statistics; higher accuracy, modest runtime cost). TRT-LLM supports both; vLLM varies.
- **Scale granularity**: **per-tensor** (one scale per weight matrix / activation tensor) vs. **per-row / per-column** (one scale per output / input row). Finer granularity → better fidelity → modestly larger scale table.
- **KV cache FP8** is a *separate* flag — chapter 01 noted this. Combine or do not combine with FP8 weights-and-activations independently.
- **Calibration data** — static FP8 needs calibration data; dynamic FP8 does not (it builds the scales online).

### Operator takeaways

- **FP8 is the production default on Hopper+ when the accuracy gate allows it.** Memory halved, native Tensor Core throughput, drop-in on most engines.
- **E4M3 for forward pass; E5M2 is a training-side detail.**
- **Static vs. dynamic scales is a trade-off.** Static is simpler and the artefact is self-contained; dynamic adapts to shift and is more robust to calibration-distribution mismatch.
- **FP8 KV cache is independent.** Many production deployments ship FP8 weights-and-activations + BF16 KV in generation 1 and add FP8 KV in generation 2 after validating the KV accuracy impact separately.
- **The Transformer Engine (TE)** is NVIDIA's library that automates dynamic scale maintenance on Hopper. Most engines on Hopper use TE under the hood or an equivalent.

The algorithm-depth question — "should the FP8 GEMM epilogue dequantise with per-tile scales or per-tensor scales for the best SM-occupancy / register-usage trade-off" — is kernel craft. The operator-depth question — "FP8 regresses at 128 k context; is it the KV or the weights" — is this module (chapter 04 gate; isolate by turning each independent flag off).

## FP4 — the algorithm, operator view (brief)

FP4 is new enough on serving stacks that the operator interface is still consolidating. The two serving-relevant facts:

- **It is block-scaled by design.** The format carries a shared scale per block (NVFP4: block-of-16 with FP8 scale; MXFP4: block-of-32 with E8M0 scale). You do not choose "scale granularity" at a higher level — the format is the granularity.
- **The quantisation recipe is similar in shape to FP8** — calibration (or dynamic running-max), per-block scale calibration, static or dynamic strategies. The main difference is the much tighter representable range per element, which makes outlier-robustness tooling (SmoothQuant-style migration, Hadamard rotations, or AWQ-shaped channel scaling) much more useful at FP4.

<!-- needs-research: production FP4 serving in TRT-LLM / vLLM matures through 2025–2026. The current state of outlier mitigation (QuaRot / SpinQuant / rotated weights) interacting with FP4 serving is a moving target. Verify against the version your fleet runs. -->

**Operator posture for FP4 right now (2026):** treat it as an advanced variant that currently requires more validation than FP8. Target Blackwell. Expect the accuracy gate to need additional outlier-mitigation steps for sensitive models.

## The calibration dataset — the under-respected input

Every weight-based quantisation recipe above (AWQ, GPTQ, SmoothQuant, static FP8) needs a calibration dataset. The dataset is small — hundreds to a few thousand samples — but its *composition* has first-order effect on the resulting quantised model's accuracy.

### What calibration data does

The algorithms compute activation statistics (magnitudes, Hessian approximations, per-channel means) on the calibration data. Those statistics determine which weight channels / dimensions / blocks are "sensitive" and get preserved vs. "insensitive" and get compressed. If the statistics are biased (calibration data differs from production in distribution), the sensitivity map is biased, and the resulting quantised model under-serves the production distribution.

### Calibration-data choices

- **Pretraining-style corpus (wikitext, c4):** the default in many open-source scripts. Works well for general-purpose chat models. Fails for domain-specific models where production traffic is heavily biased (code, legal, scientific).
- **Instruction-following corpus (dolly, alpaca, open-platypus):** better fit for instruct-tuned chat models.
- **Task-specific corpus:** samples drawn from the actual target workload. Best fit; hardest to assemble (needs representative traffic logs with PII scrubbing and consent).
- **Synthetic data:** generated from a reference model to approximate production. Fine as a complement; dangerous as the sole source (introduces distribution-reflection bias).

**The operator's rule:** if the quantised variant regresses on a specific task that the base model handles well, the first intervention is **recalibrate with data that includes that task**. The second intervention is a different algorithm (AWQ ↔ GPTQ). The third is a different format (INT4 → FP8). Not the other way around.

### Sample count

- **AWQ:** 128–512 samples typical; marginal returns above 512.
- **GPTQ:** 128–1024 samples; GPTQ benefits more from more samples than AWQ, up to a model-dependent plateau.
- **SmoothQuant:** 512–2048 samples; needs enough to estimate activation outlier channels stably.
- **Static FP8:** depends on the engine; TRT-LLM's calibrator typically uses 32–1024 samples for scale calibration. Dynamic FP8 uses no calibration (statistics are online).

### Sample length

Match the production prompt length as closely as practical. Calibrating on 512-token samples for a 32 k-context deployment often misses outlier patterns that only appear in long contexts.

## Granularity — per-tensor vs. per-channel vs. per-group

Granularity is the lever that trades storage for accuracy. For an operator, three decision points:

- **Weight granularity.** Per-tensor is cheapest; per-channel is nearly free and almost always worth it; per-group (group 128 or 64) is the standard for INT4. Smaller groups → more scales → better fidelity → modestly larger artefact and more dequant-time overhead.
- **Activation granularity.** Per-tensor (one scale per activation tensor) is standard for INT8 after SmoothQuant and for static FP8. Per-token (one scale per token) is dynamic, better quality, slightly more expensive. Per-channel activation quantisation exists but is unusual (the kernel pattern is harder).
- **KV cache granularity.** Per-head or per-layer per-head scale factors are the common pattern for FP8 KV. Finer is better, within the limits of the kernel.

**When granularity matters most:**
- **Models with heavy weight outliers** (certain MoE experts, certain late-layer MLPs): finer granularity helps.
- **Low-bit weights** (INT4 and below): finer granularity (group-64 or group-32) is often necessary to clear the accuracy bar.
- **Mixed precision** across layers: some tooling supports per-layer precision choices (first and last layer BF16, body INT4). The granularity knob composes with the per-layer choice.

**When granularity matters least:**
- **FP8 weights-and-activations** (hardware-native format with well-behaved dynamic range): per-tensor is usually enough.
- **INT8 weights with per-channel scales**: per-channel is already the standard; per-group is overkill.

The chapter-04 accuracy gate is where you discover whether your granularity choice is adequate.

## Where this stops and the performance peer picks up

This chapter has deliberately stayed at operator-depth. The algorithm-design depth — what the AWQ paper would call "the choice of search space for `s`," what GPTQ would call "the error-correction propagation order," what SmoothQuant would call "the per-layer `alpha` learning rule," what FP8 would call "the per-tile scaling kernel" — is kernel craft. If you ever find yourself asking those questions, the hand-off is to [`ai-infra-performance-learning`](https://github.com/ai-infra-curriculum/ai-infra-performance-learning). This module teaches you to:

- **Recognise** that an algorithm-depth question is being asked.
- **Scope** it precisely enough for the peer (the model, the layer, the symptom, the data).
- **Receive** the resulting change (a new recipe variant, a kernel optimisation) and integrate it into your serving stack.
- **Validate** the result with the chapter-04 accuracy gate.

The operator-depth questions you *do* own: which algorithm, which format, which granularity, which calibration data, which engine. Those are the five levers you reach for in a production review.

## Common misreadings

- **"AWQ and GPTQ differ in storage format."** They produce the same storage shape — INT4 weight-only with group-wise scales. Different content, same file layout.
- **"More calibration samples is always better."** Marginal returns plateau; off-distribution samples actively hurt. 128–512 well-chosen samples beat 10 000 off-distribution samples for AWQ.
- **"SmoothQuant is a format."** It is a recipe that produces INT8 weight-and-activation weights. The engine still runs the standard INT8 kernel; SmoothQuant's work is the per-channel smoothing baked into the weights and the activation-time scale.
- **"FP8 is parameter-free."** Static FP8 needs calibration data for scale factors. Dynamic FP8 (Transformer Engine style) does not, but has runtime cost for statistic maintenance.
- **"Lower-bit always means smaller."** True for weights (bits-per-element scales). Not necessarily true for the full artefact — scale tables, zero-point tables, and per-layer metadata add bytes. On very fine granularity (group-32) at 4 bits, the overhead can approach 10% of the stored size.
- **"The calibration dataset doesn't matter if I use the right algorithm."** It does matter. Algorithm choice compensates for some mismatch but cannot fully substitute for representative calibration.
- **"Mixed-precision layer choice (first-layer BF16, body INT4) is always a win."** Often yes, sometimes neutral, occasionally hurts throughput because the heterogeneous kernel path is slower than the uniform one. Validate.

## Summary

- **AWQ** is weight-only INT4 via **per-channel activation-aware scaling** that is absorbed into the preceding layer. Group-128 asymmetric is the default. Less calibration-sensitive than GPTQ.
- **GPTQ** is weight-only INT4 via **Hessian-aware column-block updates**. Group-128 default; `act_order` is a kernel-compatibility knob. More calibration-sensitive than AWQ.
- **SmoothQuant** is weight-and-activation INT8 via **per-channel activation smoothing** that migrates outlier magnitude from activations to weights. `alpha` is the one key knob. Mostly superseded by FP8 on Hopper+.
- **FP8** is **native hardware floating-point low precision** (E4M3 for forward pass). Static vs. dynamic scale strategy; per-tensor vs. per-row granularity. The production default on Hopper+.
- **FP4** is **block-scaled by design** (NVFP4 / MXFP4) with native support on Blackwell+. Treat as advanced; expect additional outlier-mitigation steps.
- **Calibration data is first-order important** — composition, sample count, sample length. Match to production distribution.
- **Granularity trades storage for accuracy.** Group-128 for INT4 weights, per-tensor or per-row for FP8, per-head for KV. Finer is better up to a point.
- **Algorithm-design depth is at the performance peer.** This module teaches you to recognise and route the hand-off, not to execute it.

Chapter 03 moves from "how is the quantised variant produced" to "how is it packaged, converted across formats, and validated to still match the original" — the model-format operations layer that sits between the recipe and the engine.
