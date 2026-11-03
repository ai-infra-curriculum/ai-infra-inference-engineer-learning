# Module 105 — Quantization Deployment and Model-Format Operations

> **Role:** AI Inference Engineer · **Level:** L35 (AI Infrastructure family) · **Hours:** 16

Modules 102–104 configured the engine and operated the KV cache for whatever weights someone handed you. This module is where the *weights themselves* become a serving concern. A production inference fleet does not run the Hugging Face checkpoint it was handed — it runs a *quantised, format-converted, lineage-tagged, accuracy-gated* variant that halves (or quarters) HBM, triples throughput, and still produces outputs the eval team will sign off on. The decisions behind that artefact — which precision, which algorithm, which format, which gate, which hand-off to the ML-platform peer — are what this module teaches.

By the end of it you can:

- Deploy quantised model variants on real engines and know which engine supports which format: **FP16 / BF16** baseline, **FP8** (E4M3 / E5M2) on Hopper+ and newer, **FP4 / NVFP4 / MXFP4** on Blackwell+, **INT8**, **INT4**, **AWQ**, **GPTQ**, **SmoothQuant**. Draw the engine × format × hardware matrix on a whiteboard.
- Convert and serve across the common model-format artefacts — **Hugging Face `safetensors`**, **GGUF** (llama.cpp family), **TensorRT-LLM compiled engines**, **ONNX** — and validate that the conversion did not change outputs beyond a stated tolerance.
- Author an **accuracy-regression gate** for a quantisation change — tolerance thresholds on perplexity and task-specific scores, a canary dataset, a bootstrap confidence interval. Know where this gate stops and the deep eval-methodology work at [`model-evaluation-engineer`](https://github.com/ml-engineering-curriculum/model-evaluation-engineer-learning) begins.
- Author a **release contract** with the ML-platform peer at [`ai-infra-ml-platform`](https://github.com/ai-infra-curriculum/ai-infra-ml-platform-learning) — artefact format, metadata schema (model-card slice), lineage pointer, quantisation-config hash, licence field, signing hook.
- Reason about **quantisation algorithms at operator depth** — what AWQ, GPTQ, SmoothQuant, and FP8 actually do; why the calibration dataset matters; when per-tensor vs. per-channel vs. per-group matters. Cite the boundary to [`ai-infra-performance`](https://github.com/ai-infra-curriculum/ai-infra-performance-learning) for algorithm-design depth.
- Deploy a **model-swap on the serving fleet without serving-interruption** — dual-load constraints, drain, blue-green, in-flight swap, atomic cutover.

> **Prerequisites:** [Module 101](../mod-101-inference-engineering-position/README.md) (vocabulary — quantisation terms, hardware generations), [Module 102](../mod-102-inference-engine-internals/README.md) (vLLM / SGLang / TRT-LLM internals; convert-build-serve), [Module 103](../mod-103-request-scheduling-and-batching/README.md) (what the scheduler calls through to), and [Module 104](../mod-104-kv-cache-management/README.md) (KV precision as a byte-math input — this module's choices feed back into the KV budget there).

## Boundary against `ai-infra-performance`

Quantisation has two sides. One side is the **algorithm** — how AWQ picks salient channels and how it moves scale from activation to weight; how GPTQ performs the Hessian-aware weight update per column block; how SmoothQuant redistributes outlier magnitudes between activation and weight; how an FP8 GEMM epilogue produces bit-accurate scales. That side is kernel craft; it lives at the performance peer. If your diagnosis lands on "the FP8 GEMM is slower than Marlin / cutlass should produce" or "the AWQ dequant kernel is memory-bandwidth-bound where it shouldn't be," the fix is a `.cu` / `.cuh` / Triton-DSL / CUTLASS edit — route it to the performance peer.

The other side is the **production operation**: choosing the right format for the right engine on the right hardware, running the conversion workflow, validating output equivalence, gating the accuracy regression, hand-shaking with the ML-platform peer on the artefact contract, and swapping the fleet from one quantised variant to another without dropping requests. That is this module.

Rule of thumb: *if the fix is to a kernel source file (`.cu` / `.cuh` / Triton kernel / CUTLASS epilogue), that is the performance peer. If the fix is to a quantisation flag, a calibration dataset, an accuracy threshold, an artefact-metadata field, or a model-swap runbook, that is this module.*

## Boundary against `model-evaluation-engineer`

The accuracy gate this module authors is a **release gate on a quantisation change**, scoped to a stated tolerance on a stated canary set, using documented statistical primitives (bootstrap CI, paired difference tests). It is deliberately narrow — enough to say "this quantised variant is or is not safe to swap in" with defensible evidence.

The broader eval-methodology discipline — building the benchmark suite, calibrating thresholds against task families, reasoning about leakage and dataset decay, correlating offline eval with online outcomes — lives at [`model-evaluation-engineer`](https://github.com/ml-engineering-curriculum/model-evaluation-engineer-learning) (peer, level 30). This module teaches you to *configure* the gate, interpret the result, and escalate the ambiguous cases. It does not teach you to design the benchmark suite.

Rule of thumb: *if the question is "does this quantised variant regress within our stated tolerance on our chosen canary set," that is this module. If the question is "is our canary set the right one, is our tolerance right, is our statistical method sound," that is the model-evaluation-engineer peer.*

## Boundary against `ai-infra-ml-platform`

This module ships a quantised model artefact. The ML-platform peer catalogues and distributes it. The handshake is a *contract* — what fields the artefact carries, where lineage points, what the metadata schema requires, how signing works. Authoring the contract is this module's output; running the registry and training-serving handoff is the ML-platform peer's.

Rule of thumb: *if the fix is to the artefact's contents, metadata, or quantisation config, that is this module. If the fix is to the registry service, the artefact-signing CA, or the training-to-serving CI pipeline, that is the ML-platform peer.*

## Boundary against module 113

Module 113 owns the production-rollout mechanics for a model swap *as a general discipline* — canary, shadow, response-diff, auto-rollback — against SLO signals. This module owns the serving-side **mechanics** of swapping a quantised model (dual-load, drain, in-flight weight reload) and the **accuracy gate** that is one of the signals module 113 reads. 113 is the rollout framework; 105 is the quantisation-specific swap primitive and the accuracy-gate axis.

## Chapters

1. [**Quantisation Formats and Engine Support**](01-quantization-formats-and-engine-support.md) — the format map (FP16 / BF16, FP8 E4M3 / E5M2, FP4 / NVFP4 / MXFP4, INT8, INT4, weight-only AWQ / GPTQ, activation-plus-weight SmoothQuant); which hardware generation unlocks which (Ampere → INT8 Tensor Cores; Hopper → FP8; Ada / Blackwell → FP4); which engine implements which; where weight-only vs. activation-plus-weight is the right axis to think along; the "what does this format cost in bits, bytes, and accuracy" cheatsheet.
2. [**Quantisation Algorithms at Operator Depth**](02-quantization-algorithms-at-operator-depth.md) — what AWQ, GPTQ, SmoothQuant, and FP8 actually do in one page each; why the calibration dataset matters; when per-tensor vs. per-channel vs. per-group granularity matters; the boundary to the performance peer for algorithm-design depth.
3. [**Model-Format Conversion and Round-Trip Validation**](03-model-format-conversion-and-roundtrip.md) — the four formats that matter in production (safetensors, GGUF, TRT-LLM engines, ONNX); the conversion recipes; the "did this conversion preserve outputs" validation discipline (logit-diff tolerance, per-prompt sampled-output diff, perplexity-delta); the common pathologies.
4. [**Authoring an Accuracy Regression Gate**](04-accuracy-regression-gate.md) — the canary dataset, perplexity and task-specific metric, bootstrap confidence interval, pass / fail / escalate thresholds; the gate as a CI job; the escalation path to the model-evaluation-engineer peer.
5. [**Release Contract with the ML-Platform Peer**](05-release-contract-with-ml-platform.md) — the artefact format, the metadata schema (model-card slice), the lineage pointer, the quantisation-config hash, the licence field, the signing hook; the "what breaks if a field is missing" audit.
6. [**In-Flight Model Swap on the Serving Fleet**](06-in-flight-model-swap.md) — dual-load capacity, drain protocol, blue-green between replicas, in-flight weight reload where supported, atomic routing cutover, the model-version header for client observability.

## Exercises

The exercises carry each chapter into a hands-on drill you keep as a reusable artefact for the rest of the track:

1. [**Deploy an FP8 Model on TRT-LLM and vLLM**](exercises/exercise-01-deploy-fp8-model-on-trt-llm-and-vllm.md) — produce an FP8 variant of one stock model; serve it on each engine; measure HBM savings, prefill / decode throughput, and output-sample equivalence against the BF16 baseline.
2. [**Deploy AWQ and GPTQ Weight-Only Serving**](exercises/exercise-02-deploy-awq-and-gptq-weight-only-serving.md) — produce AWQ and GPTQ variants of one stock model; serve each; measure HBM savings, latency delta vs. BF16, and output-sample equivalence; compare AWQ vs. GPTQ on the same base model.
3. [**safetensors → GGUF → TRT-LLM Engine Round-Trip and Validate**](exercises/exercise-03-safetensors-gguf-trt-engine-roundtrip-and-validate.md) — convert one model across three formats; serve each; drive the identical prompt set through each; measure logit-level drift, perplexity delta, and sampled-output equivalence; publish the round-trip table.
4. [**Accuracy Regression Gate Authoring**](exercises/exercise-04-accuracy-regression-gate-authoring.md) — assemble a canary dataset, pick a perplexity + task-specific metric pair, write the bootstrap CI, codify as a CI job, and gate one real quantised variant with it; document the pass / fail / escalate thresholds.
5. [**ML-Platform Handoff Contract Drill**](exercises/exercise-05-ml-platform-handoff-contract-drill.md) — draft the artefact + metadata contract with the ML-platform peer (or in their stead); walk through a quantised-variant handoff end-to-end; demonstrate the lineage pointer, quant-config hash, licence field, and signing hook.
6. [**In-Flight Model Swap Dry Run**](exercises/exercise-06-in-flight-model-swap-dry-run.md) — swap one running fleet from BF16 to FP8 (or AWQ) under live simulated load without dropping requests; measure latency, error-rate, and KV-cache preservation at every stage; author the runbook from the drill.

## Resources

See [`resources.md`](resources.md) for primary / citable references — the AWQ, GPTQ, SmoothQuant, and FP8 papers; the NVIDIA Hopper and Blackwell architecture whitepapers; the safetensors, GGUF, TensorRT-LLM, and ONNX specifications; the engine-specific quantisation docs; and the model-card / artefact-signing standards the handoff contract builds on.

## Where this fits in the curriculum

- **Before:** [Module 104 — KV-Cache Management](../mod-104-kv-cache-management/README.md). This module's precision choices feed directly into the KV byte-math there (FP8 KV halves per-token KV footprint; INT4 weights change the weight budget but not KV).
- **After:** [`mod-106-multi-model-and-adapter-serving`](../mod-106-multi-model-and-adapter-serving/README.md) (adapter / LoRA swap and multi-tenant serving use this module's weight-loader and swap mechanics); [`mod-107-distributed-inference-topologies`](../mod-107-distributed-inference-topologies/README.md) (quant choice interacts with TP sharding and FP8/FP4 GEMM collectives); [`mod-113-production-safety-and-rollouts`](../mod-113-production-safety-and-rollouts/README.md) (canary / blue-green rollouts that consume this module's accuracy gate as a release signal).
- **Peer-track hand-off:** when a quantisation drill lands on a GEMM kernel finding (FP8 GEMM slow, AWQ dequant stall, mixed-precision epilogue regression, CUTLASS block-copy regression), the hand-off is to [`ai-infra-performance-learning`](https://github.com/ai-infra-curriculum/ai-infra-performance-learning). This module teaches you to *recognise and route* that hand-off, not to execute it.
- **Peer hand-off (eval):** when the accuracy gate surfaces an ambiguous result — "regression is within noise on perplexity but outside noise on one task" — the methodology hand-off is to [`model-evaluation-engineer-learning`](https://github.com/ml-engineering-curriculum/model-evaluation-engineer-learning). This module teaches you to recognise the escalation and bring a complete artefact (dataset, metric, CI, pre-registered threshold) to the conversation.
- **Peer hand-off (registry):** the artefact + metadata contract is authored here and *operated* by [`ai-infra-ml-platform-learning`](https://github.com/ai-infra-curriculum/ai-infra-ml-platform-learning). This module teaches you the serving-side half of the handshake and what to require of the registry side.
