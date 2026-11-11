# Module 108 — Speculative-Decoding Deployment and Operation

> **Role:** AI Inference Engineer · **Level:** L35 (AI Infrastructure family) · **Hours:** 12

Modules 102–107 served tokens one at a time per sequence. Decode is memory-bandwidth bound: each new token forces a full pass over the weights to read them out of HBM. The idea of **speculative decoding** (Leviathan et al., 2023; Chen et al., 2023 — see [`resources.md`](resources.md)) is that if a cheap *drafter* can propose `k` plausible next tokens and the target model can *verify* all `k` with roughly one pass worth of HBM traffic, you can produce multiple tokens per target forward pass whenever the drafter is right — and fall back to one-token-per-pass whenever it isn't. The whole promise is bounded by one number: the **acceptance rate**.

This module is where speculative decoding stops being a paper and becomes a deployment, a Prometheus series, and a lifecycle. By the end of it you can:

- **Deploy draft-target speculative decoding** on vLLM / SGLang / TensorRT-LLM — pick a draft that matches the target's tokenizer and distribution, size the draft against the target (parameters, HBM, latency), set the speculative length `k`, and measure acceptance rate against your production workload.
- **Deploy self-speculative schemes** — Medusa (multi-head), EAGLE (feature-level), Lookahead decoding — in the engines that support them, and reason about when they beat draft-target (no second checkpoint, tokenizer parity for free, lower draft-step cost; but harder to re-train, engine-coupled).
- **Monitor acceptance rate in production** — per-tenant, per-model-pair, per-workload-class (code, chat, long-context, tool-use) — and alert on acceptance-rate drift that signals a draft-model lifecycle action rather than an engine bug.
- **Author the lifecycle of a draft model** — retrain on target drift, versioned-together promotion with the target checkpoint, rollback discipline that treats draft-rollback as a first-class action (and never leaves a `(target_v2, draft_v1)` pair in production unverified).
- **Reason about when speculative decoding hurts latency** — low acceptance that doesn't amortise the draft-step overhead, batches that are already compute-bound (prefill, large batch, long decode across many sequences), and workloads where draft-step overhead on the critical path inflates TTFT past its SLO.
- **Cite the boundary to the peers** — speculative-decoding *kernels* (fused draft+verify, tree-attention masks, bonus-token handling) live at `ai-infra-performance` (peer, L35); draft-model *fine-tuning* lives at `fine-tuning-engineer` (peer, L30). This module operates the deployment and the lifecycle, not the kernel and not the training loop.

> **Prerequisites:** [Module 102](../mod-102-inference-engine-internals/README.md) (engine internals — the places speculative decoding hooks into the scheduler loop and the model runner), [Module 103](../mod-103-request-scheduling-and-batching/README.md) (continuous batching — speculative decoding changes how many tokens a batch produces per iteration), [Module 104](../mod-104-kv-cache-management/README.md) (KV cache — speculative schemes write multiple candidate tokens into the KV and commit or discard based on verification), [Module 107](../mod-107-distributed-inference-topologies/README.md) (TP sharding — draft and target are both sharded; the collectives from 107 compose with the extra draft step).

## The core framing

Speculative decoding is **free throughput when the drafter is right, and overhead when the drafter is wrong**. The two numbers that govern everything in this module are:

- **Acceptance rate `α` (per workload class).** The fraction of draft tokens that the target accepts. Papers report `α ∈ [0.4, 0.9]` depending on draft-target closeness and workload; your fleet will have its own curve.
- **Draft-step cost `c_d` relative to the target step `c_t`.** The drafter costs roughly `c_d / c_t ∈ [0.02, 0.2]` of a target step depending on the scheme. The lower this ratio, the lower the acceptance rate speculative decoding can tolerate before it stops paying.

The speed-up ceiling (Leviathan et al.) is approximately `(1 - α^(k+1)) / ((1 - α)(1 + c_d · k / c_t))` for speculative length `k`. The ceiling is unreachable in practice (contention, batching dynamics, memory pressure) but it names the shape — a sharp drop-off below `α ≈ 0.5` or so, and a diminishing return above `α ≈ 0.8`. The operator job is to **measure `α` on real traffic**, pick `k` to match, and alert when `α` drifts.

The dividing line that recurs: *operate the acceptance rate and the lifecycle; defer the kernel and the draft training to the peers.* Below.

## Boundary against `ai-infra-performance`

Speculative-decoding **kernels** live at [`ai-infra-performance`](https://github.com/ai-infra-curriculum/ai-infra-performance-learning) — the fused draft+verify kernel, the tree-attention masks that let the target verify a tree of candidates in one pass, the bonus-token handling that extracts the "free" token when all `k` drafts are accepted, the TP-aware draft forward pass, the CUDA-graph re-trace discipline when `k` changes dynamically. If your diagnosis lands on *"tree-attention mask is wrong at `k=5`"* or *"the fused verify kernel's epilogue stalls at this batch shape"* — route to the performance peer.

The **production operation** is this module: picking the drafter, picking `k`, deploying through the engine's speculative-decoding guide, measuring acceptance, authoring the lifecycle. Chapter 06 writes the hand-off bundle.

Rule of thumb: *if the fix is to a `.cu` / `.cuh` / Triton-DSL / engine-internal mask layout, that is the performance peer. If the fix is to a launch flag, a drafter choice, a speculative length, a per-tenant policy, or a lifecycle contract, that is this module.*

## Boundary against `fine-tuning-engineer` (L30)

Draft-model **training and fine-tuning** live at [`fine-tuning-engineer`](https://github.com/ai-infra-curriculum/fine-tuning-engineer-learning) — distillation from the target, data mix for draft training, DPO / preference training of drafters where applicable, loss landscape of EAGLE's feature-level objective, Medusa head training, and the evaluation harness that scores a draft candidate on target-equivalence. If your diagnosis lands on *"we need a new draft trained on more of our code traffic"* — the request is to the fine-tuning peer; chapter 04 writes the hand-off and the versioning contract.

Rule of thumb: *if the fix is "retrain the drafter," that is the fine-tuning peer. If the fix is "swap the drafter to a different checkpoint we already have," "change `k`," "change which tenants get speculative decoding," or "roll back to the previous `(target, draft)` pair," that is this module.*

The exact scope of the `fine-tuning-engineer` track at this org — specifically whether draft-model distillation is in-scope at level 30 or escalates to a research org — depends on the sibling track's charter; verify the hand-off interface in chapter 04 and chapter 06 against that charter when the curriculum lands.

## Boundary against module 103

Module 103 owns the **scheduler loop** — continuous batching, chunked prefill, admission control, preemption. Speculative decoding changes **what counts as one iteration's work**: a decode step at `k=4` produces up to 4 tokens per sequence per iteration if all are accepted, or 1 token if none are. Chapter 03 of module 103 teaches the scheduler; chapter 05 of this module names when the scheduler's decisions make speculative decoding a net loss (batches already compute-bound, long prefill without decode on the critical path).

## Boundary against module 107

Module 107 owns the **TP / PP / EP topology**. Draft-target speculative decoding runs the drafter on the same topology as the target in some engines, on a different (smaller) topology in others. The draft's TP degree affects draft-step cost `c_d` — a 1B draft at TP=8 may be *slower than* at TP=1 because of collective overhead on a tiny model. Chapter 01 of this module names the sizing trade-off; chapter 07 of 107 is the broader topology view.

## Chapters

1. [**Draft-Target Speculative Decoding — Mechanics, Sizing, and Deployment**](01-draft-target-speculative-decoding.md) — the draft-and-verify loop; picking a draft model (tokenizer match, distribution closeness, parameter ratio); sizing the draft against the target in HBM and in step-cost; speculative length `k`; the throughput-vs-TTFT trade-off; a vLLM / SGLang / TRT-LLM deployment walkthrough.
2. [**Self-Speculative Schemes — Medusa, EAGLE, Lookahead Decoding**](02-self-speculative-schemes.md) — the three alternatives to draft-target; what each one is (multi-head, feature-level, trigram cache); what each one needs (training a Medusa head, training EAGLE's draft, no training for Lookahead); when each one beats draft-target and when it doesn't.
3. [**Acceptance-Rate Monitoring — Per-Tenant, Per-Model-Pair, Per-Workload-Class**](03-acceptance-rate-monitoring.md) — the metric set (`spec_tokens_proposed`, `spec_tokens_accepted`, `spec_acceptance_rate`, `spec_overhead_latency`); the axis set to slice by (tenant, model pair, workload class, prompt-length bucket); the alert set that signals a lifecycle action vs. an engine bug.
4. [**Draft-Model Lifecycle — Retrain, Versioned Promotion, Rollback**](04-draft-model-lifecycle.md) — the trigger set for draft retrain (target drift, acceptance-rate drift, new workload class); the versioning contract (`target_v2` cannot promote without a verified `(target_v2, draft_v2)` pair); the rollback discipline (if you roll back the target, roll back the draft; never leave an unverified pair in production).
5. [**When Speculative Decoding Hurts Latency — Regimes and Benchmarks**](05-when-speculative-hurts-latency.md) — the two regimes where speculative decoding is a loss (acceptance too low to offset draft overhead; batch already compute-bound); a benchmark methodology that catches both; what to do when you catch them (lower `k`, raise acceptance threshold, disable speculative for the affected class, autoscale differently).
6. [**Boundaries — Performance Peer (L35) and Fine-Tuning Engineer (L30)**](06-boundary-to-performance-and-fine-tuning.md) — the three-way table; the hand-off bundle to the performance peer when the diagnosis is a kernel; the hand-off bundle to the fine-tuning peer when the diagnosis is draft retraining; the three common cross-boundary mistakes.

## Exercises

Each exercise carries a chapter into a reusable artefact — a deployment, a monitoring dashboard, a lifecycle contract, a benchmark report.

1. [**Draft-Target Speculative Deployment on vLLM**](exercises/exercise-01-draft-target-speculative-deployment-on-vllm.md) — stand up a draft-target speculative pair on vLLM, measure acceptance rate against three workload classes, publish the `k` sweep and the per-class recommendation.
2. [**Medusa and EAGLE Deployment Walkthrough**](exercises/exercise-02-medusa-and-eagle-deployment-walkthrough.md) — deploy a Medusa-equipped checkpoint and/or an EAGLE pair in the engine that supports it, compare acceptance rate and end-to-end TPOT to the draft-target baseline from exercise 01.
3. [**Acceptance-Rate Monitoring and Alerts**](exercises/exercise-03-acceptance-rate-monitoring-and-alerts.md) — instrument the engine's speculative metrics into Prometheus, build the Grafana dashboard with the per-tenant / per-model-pair / per-workload-class slicing, author the alert set.
4. [**Draft-Model Lifecycle Contract**](exercises/exercise-04-draft-model-lifecycle-contract.md) — write the one-page lifecycle contract for `(target, draft)` pairs on your fleet: promotion gate, rollback trigger, retrain trigger, ownership, and the hand-off bundle to the fine-tuning peer.
5. [**When Speculative Hurts Latency — Benchmark**](exercises/exercise-05-when-speculative-hurts-latency-benchmark.md) — build the benchmark that catches the two regimes (low acceptance, compute-bound batch); run it against your fleet; publish the fleet's "disable speculative when ..." policy.

## Resources

See [`resources.md`](resources.md) for primary / citable references — Leviathan et al. and Chen et al. (the original speculative-decoding papers), Medusa and EAGLE (and EAGLE-2 / EAGLE-3), Lookahead decoding, SpecInfer (tree-based verification), vLLM / SGLang / TensorRT-LLM speculative-decoding documentation, and the peer-track pointers.

## Where this fits in the curriculum

- **Before:** [Module 107 — Distributed Inference Topologies and Disaggregated Serving](../mod-107-distributed-inference-topologies/README.md). Speculative decoding composes with TP: both drafter and target are typically sharded, and the extra draft forward pass's collective cost is a line item in the chapter-05 "when speculative hurts" analysis.
- **After:** [`mod-109-autoscaling-fleet-and-resilience`](../mod-109-autoscaling-fleet-and-resilience/README.md) (autoscaling must respect the drafter's HBM footprint — a replica is a `(target_shard, draft_shard)` pair, not just a target shard), [`mod-110-observability-slo-and-cost`](../mod-110-observability-slo-and-cost/README.md) (chapter 03's metrics feed the SLO methodology; speculative acceptance is a cost-per-1M-tokens line item), [`mod-113-production-safety-and-rollouts`](../mod-113-production-safety-and-rollouts/README.md) (the lifecycle contract from chapter 04 is a specific instance of the canary / blue-green discipline).
- **Peer-track hand-off (performance):** every operator-depth chapter in this module names the explicit kernel-level cutoff where you stop debugging and route to [`ai-infra-performance-learning`](https://github.com/ai-infra-curriculum/ai-infra-performance-learning). Chapter 06 formalises the hand-off bundle.
- **Peer-track hand-off (fine-tuning):** chapter 04 writes the retrain-request bundle to [`fine-tuning-engineer-learning`](https://github.com/ai-infra-curriculum/fine-tuning-engineer-learning) (sibling track; verify the exact repo name against the organisation's layout when that curriculum lands) — the one-page spec that triggers a new draft checkpoint.
