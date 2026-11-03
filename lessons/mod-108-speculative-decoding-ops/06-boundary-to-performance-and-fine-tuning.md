# Chapter 06 — Boundaries: Performance Peer (L35) and Fine-Tuning Engineer (L30)

> Speculative decoding touches three tracks. This module operates the deployment and the lifecycle. The performance peer owns the kernels the operator dispatches to. The fine-tuning engineer owns the drafter's training. Each of the three cross-boundary mistakes is a wasted week; this chapter writes the fence.

## Motivation

Three recurring triage escalations go to the wrong owner:

1. **"Speculative is slow."** Operator sees `α = 0.68`, draft overhead ratio `0.35`, and files a bug against the engine project. The engine team closes it with "your draft is sharded at TP=4 on a 1B model — of course the draft overhead is high; try TP=1 for the draft." The right fix was a config change this module owns.

2. **"The draft's `α` is low on our workload — fix the kernel."** Operator opens a performance-peer ticket about low acceptance. The peer closes it with "the acceptance rate is a function of the drafter's training, not the kernel. Your `α = 0.42` on reasoning traffic is a training-data mix issue, not a kernel issue — route to the fine-tuning peer." A week was spent on the wrong escalation.

3. **"Retrain the draft against our traffic."** Operator opens a fine-tuning request without the chapter-04 bundle — just "it's slow." The fine-tuning peer asks for the measurements, the mix, the acceptance criteria. Three days later the operator submits a complete bundle and the work starts.

Each of these is a boundary error. This chapter writes the three-way table, the two hand-off bundles, and the three recurring mistakes.

By the end of it you can:

- State what this module owns, what `ai-infra-performance` owns, and what `fine-tuning-engineer` owns — in table form.
- Write the hand-off bundle to the performance peer (what config, what trace, what measurement, what to try first).
- Write the hand-off bundle to the fine-tuning peer (what traffic sample, what acceptance criteria, what workload mix).
- Avoid the three common cross-boundary mistakes.

## The three-way boundary table

| Concern | This module (`ai-infra-inference-engineer`, L35) | `ai-infra-performance` (peer, L35) | `fine-tuning-engineer` (peer, L30) |
|---|---|---|---|
| Draft-target scheme deployment | ✅ configure, deploy | — | — |
| Medusa / EAGLE / Lookahead deployment | ✅ configure, deploy | — | — |
| Speculative-length `k` choice | ✅ measure, set | — | — |
| Draft model selection | ✅ pick existing checkpoint, size | — | ✅ (train new drafter if requested) |
| Draft model training / fine-tuning | — | — | ✅ |
| Draft distillation recipe | — | — | ✅ (owns loss, data, optimiser) |
| Medusa head training | — | — | ✅ |
| EAGLE drafter training | — | — | ✅ |
| Fused draft+verify kernel | — | ✅ | — |
| Tree-attention mask | — | ✅ | — |
| Bonus-token kernel | — | ✅ | — |
| TP-aware draft forward pass | — | ✅ | — |
| CUDA-graph re-trace discipline for dynamic-`k` | — | ✅ | — |
| Per-tenant / per-class acceptance monitoring | ✅ | — | — |
| Acceptance baseline on canary | ✅ (measure) | — | — (consumes as acceptance criterion) |
| Pair-level lifecycle contract (ch 04) | ✅ | — | — (consumes as retrain request shape) |
| Rollback discipline | ✅ | — | — |
| Benchmark of "when speculative hurts" (ch 05) | ✅ | — (if the fix is kernel) | — (if the fix is draft training) |
| Per-request / per-class disable policy | ✅ | — | — |
| Autoscaling to avoid compute-bound regime | ✅ (defers to mod-109) | — | — |
| Sampler compatibility matrix (what bypasses speculative) | ✅ documents, routes | ✅ owns if kernel-side | — |
| Workload characterisation (prompt-shape, generation-shape) | ✅ | — | ✅ (consumes for retrain data mix) |

## The hand-off to the performance peer

### When to hand off

- Chapter 03's Alert 5 ("draft overhead ratio climb") fires and the pair's config is unchanged, the draft weight-SHA matches the registry, and the engine version matches the pair record.
- The engine's startup log confirms speculative is enabled, but TPOT doesn't drop even on a workload where `α > 0.7` (acceptance looks healthy, speedup doesn't materialise).
- Nsight timeline shows the draft forward pass dominating the iteration time rather than overlapping with target verify. The fused kernel isn't firing / isn't efficient.
- Tree-attention mask looks wrong (correctness issue — rare, but a kernel-level bug if it happens; output distribution regresses on greedy regression suite).
- CUDA graph re-trace storm on dynamic-`k`: the engine is re-tracing on every `k` change rather than hitting a cached graph.
- Published benchmark numbers for your (engine version, hardware, pair) exist and are substantially better than you observe, after the chapter-01 health checks pass and placement is verified.

### The hand-off bundle

Package before escalation:

```
# Deployment identity
engine:                vllm 0.7.1 / sglang 0.4.0 / trt-llm 0.14 (be specific)
model_pair:            llama31-70b-fp8__draft-llama32-1b-bf16
target_name@sha:       meta-llama/Meta-Llama-3.1-70B-Instruct-FP8 @ <sha>
draft_name@sha:        meta-llama/Llama-3.2-1B-Instruct @ <sha>
scheme:                draft-target  / medusa / eagle / lookahead
k (static or dynamic): 4 (static) / [2,8] dynamic
tp_target:             4
tp_draft:              4 (or 1, if sharding diverges)
quantisation:          fp8 target, bf16 draft

# Hardware
nodes:                 2 × 8× H100 SXM 80GB
interconnect:          NVLink intranode, 400 Gbps IB internode
driver / cuda / nccl:  535.129.03 / 12.4 / 2.21.5

# Chapter 01 health checks (verified)
Startup log confirms speculative:                 YES
Metrics endpoint shows non-zero spec counters:    YES
A/B latency sanity on high-α workload:            FAIL (TPOT didn't drop)
Sampler bypass counter near-zero:                 YES
Greedy regression suite output-equivalence:       PASS

# Acceptance health (chapter 03)
Fleet α (1h window):                              0.71
Per-class α (chat / code / long_context / reasoning):  0.73 / 0.86 / 0.65 / 0.44
Deploy-time baseline α:                                0.72 / 0.86 / 0.66 / 0.46
Drift:                                                 negligible

# Observed symptom
TPOT P50 (speculative on):                             42 ms
TPOT P50 (speculative off, same traffic sample):       48 ms
Expected speedup from ch 01 ceiling formula at α=0.71, k=4, c_d/c_t=0.1:  ~2.1×
Observed speedup:                                      ~1.14×
Delta unaccounted for:                                 ~45%

# spec_draft_overhead_ratio
Current:                                               0.42
Expected on this draft/target pair:                    ~0.10

# Nsight trace
Attached: ns-rep for a 20-iteration window on a canary sequence.
Observations:
  - Draft forward passes serialise (expected — autoregressive).
  - Target verify kernel takes 2.8 ms where published benchmark shows 1.1 ms at this shape.
  - Specific kernel suspected: fused_verify_attn. See frame <t_ms> in the attachment.

# Already tried (chapter 01 + chapter 06 placement)
- NCCL env: NCCL_P2P_LEVEL=NVL, NCCL_IB_HCA set, debug clean on startup.
- Draft TP: tested TP=1 for draft (replicated across group) → worse (as expected
  for small-model-under-TP, chapter 01).
- Dynamic k: forced static k=4 → same.
- Downgraded engine to previous release → same.
- Chapter 06 placement check: nvidia-smi topo -m clean; NUMA pinning verified;
  nccl-tests match spec on bare node pair.

# Specific question
Why is fused_verify_attn 2.5× published spec on this (target, draft, hardware)
combination? Suspicion: the fused verify's mask-compute isn't vectorising at k=4.

# What this module is NOT asking
- "Make it faster." — not actionable.
- "Rewrite the engine." — out of scope.
- The acceptance rate itself — that is a drafter issue, not a kernel issue.
```

The performance peer takes this bundle and makes it their first-class task. The operator does not re-write the kernel; the operator defines the test and the measurement.

### What NOT to send to the performance peer

- "Speculative is slow." Too vague. See the three-way table — scope the question.
- A low acceptance rate without a drafter-level investigation (that's fine-tuning peer territory).
- A per-tenant SLO breach without the per-class triage (that's chapter 03 + chapter 04 before escalation).
- A screenshot of a Grafana dashboard without the Nsight trace or the measurement delta. The peer needs a specific hypothesis to test.

## The hand-off to the fine-tuning peer

### When to hand off

- Chapter 04's Trigger 1 (target drift forced a new target rev; draft is stale).
- Chapter 04's Trigger 2 (durable workload shift; new mix needs to be represented).
- Chapter 04's Trigger 3's residual cause (no engine / target / mix explanation; the drafter is simply aging against drifting traffic).
- Chapter 04's Trigger 4 (new workload class with volume; drafter wasn't trained to cover it).
- Chapter 04's Trigger 5 (architecture upgrade: new EAGLE variant, new Medusa recipe).

### The hand-off bundle (recap from chapter 04)

Chapter 04 wrote the full retrain-request bundle; summarised here with the specific fields that distinguish a complete request from a dropped one:

```
# Retrain request — draft for <target_name@sha>

## Context
- Current pair record (link to registry).
- Trigger: <ch 04 trigger name + evidence>.
- Date requested; needed by.

## Measurements that motivate
- Per-class α table — current and historical (link to ch 03 panel snapshot).
- Workload-mix snapshot: class × share by request count and by token count,
  prompt-length distribution per class, generation-length distribution per class.
- Observed failure modes of the current draft (which class and how badly).

## Acceptance criteria for the new draft (operator-set)
- Primary: paired α on canary traffic ≥ thresholds per class.
- Secondary: c_d / c_t ≤ threshold; HBM footprint ≤ threshold; output-distribution
  preservation (greedy regression suite).

## What this module provides
- Canary traffic sample (scrubbed, representative, N requests).
- Access to the current pair's baseline table.
- A test harness that measures α for a candidate draft.
- An on-call engineer to co-validate on canary.

## What the fine-tuning peer provides
- Retrained draft checkpoint + weight SHA.
- Training recipe summary (one page).
- Any changes the engine needs to know about.

## Decision rights
- Peer owns the retraining recipe.
- This module owns the acceptance criteria and the canary acceptance.
- Both sign off on the pair before promotion.
```

### What NOT to send to the fine-tuning peer

- A retrain request without the mix snapshot. The peer cannot train against "it's slow."
- A retrain request where the actual cause is a kernel issue or a config issue. Chapter 04's Trigger 3 explicitly asks the operator to rule out engine / target / mix before escalating.
- A retrain request without acceptance criteria. The peer needs to know what "better" means in numbers.

## The three common cross-boundary mistakes

### Mistake 1 — opening a performance-peer ticket for an acceptance issue

**Shape.** Operator sees `α = 0.42` on reasoning traffic; files a bug against the kernel team.

**Why it's wrong.** The acceptance rate is the drafter's function (given a fixed target). The kernel does not change which tokens the drafter proposes or which the target accepts. The low `α` is a drafter-training problem, not a kernel problem.

**Fix.** Chapter 04 Trigger 2 (if durable workload shift) or Trigger 3 (if ambiguous drift). Hand off to the fine-tuning peer, not the performance peer.

**Prevention.** Chapter 03's dashboard structure separates `α` panels from latency panels. The operator who triages `α` instinctively routes to fine-tuning; the one who triages latency routes to performance (after chapter 06's health-check gate).

### Mistake 2 — opening a fine-tuning request for a kernel issue

**Shape.** Operator sees TPOT didn't drop as expected on a workload where `α = 0.78`; files a retrain request against the fine-tuning peer.

**Why it's wrong.** `α = 0.78` is a healthy drafter. The lack of speedup is in the kernel / scheduler / config, not the drafter. A retrain of the drafter would not change TPOT; it might change `α` to `0.80`, which doesn't address the operator's symptom.

**Fix.** Chapter 06's hand-off bundle to the performance peer. The gate between the two: *is `α` the problem, or is the speedup-per-token the problem?*

**Prevention.** Chapter 03's `spec_draft_overhead_ratio` and `spec_tokens_per_iter` isolate the "speedup per accepted token" from the "fraction of tokens accepted." Operators triage on the right one.

### Mistake 3 — asking the fine-tuning peer to do operator work

**Shape.** Operator says "we want a draft that is fast enough for `k = 8` on reasoning workloads at batch=24." The peer is being asked to make a drafter that gets around the compute-bound regime (chapter 05 Regime 2) by raising `α` enough to pay even at high overhead.

**Why it's wrong.** The compute-bound regime is not a drafter problem — the target is the bottleneck, not the drafter. A better drafter still has to wait for the target's compute-bound verify step. The fix is a disable policy (chapter 05) or an autoscaler change (module 109).

**Fix.** Chapter 05's policy. Don't ask the peer to solve what is operator-owned.

**Prevention.** Chapter 05's benchmark characterises the regime; chapter 05's policy enforces the response. The retrain request to the peer is for low-`α` classes, not for compute-bound-batch regimes.

## Composition with other modules in this track

- **Module 102 (engine internals).** The specific engine's speculative-decoding hook points, flag surface, metric endpoint. This chapter's hand-off bundles pull from 102's engine-mental-model.
- **Module 103 (scheduler).** The compute-bound-batch regime in chapter 05 Regime 2 is in dialogue with 103's "when does batch become compute-bound" treatment. The scheduler's admission control is where the chapter 05 policy enforces.
- **Module 104 (KV cache).** Speculative schemes write candidate tokens into the KV and commit or discard based on verification. The byte-layout is 104's; the commit / discard logic is engine-internal.
- **Module 105 (quantization).** A quantization re-pack of the target changes the output distribution and triggers chapter 04 Trigger 1 (target drift). The quantization promotion gate and the pair-promotion gate interlock.
- **Module 107 (topology).** The draft's TP degree is a chapter 01 sizing decision; the compute-bound regime (chapter 05 Regime 2) is affected by the target's TP degree. The topology view at 107 and the pair view here compose.
- **Module 109 (autoscaling).** A replica is a `(target_shard, draft_shard)` pair. The autoscaler scales pairs, not shards. Chapter 05 Shape 3 (autoscale earlier to stay below `B*`) is a 109 policy.
- **Module 110 (observability / SLO / cost).** Chapter 03's metrics feed 110's SLO methodology. Speculative's cost accounting (chapter 03 `cost/1M-tokens`) is a 110 line item.
- **Module 113 (rollouts).** Chapter 04's pair-promotion canary ramp is a specific instance of 113's canary / blue-green discipline.

## The sign-off a well-run fleet publishes for speculative decoding

Every deployed `(target, draft)` pair ships a one-page sign-off:

```
Fleet: <name>
Pair: <model_pair slug>
Target: <name>@<sha>
Draft:  <name>@<sha>  (or Medusa heads @sha, EAGLE drafter @sha, or "lookahead")
Scheme: draft-target / medusa / eagle / lookahead
Engine version: <engine>@<sha>
k default: <value or dynamic policy>

Baseline paired α per class (ch 04 Rule 1):
    chat:            <α> ± <ci>  (canary <date>, <N> samples)
    code:            <α> ± <ci>
    long_context:    <α> ± <ci>
    reasoning:       <α> ± <ci>
    ...

Baseline TPOT P50/P99 per class:
    chat:            <ms> / <ms>
    code:            <ms> / <ms>
    ...

Disable policy (ch 05):
    workload_class ∈ { <list> }  → bypass
    batch_size      > <B_disable> → bypass (if engine supports)
    max_tokens      < <N>         → bypass (if applicable)

Monitoring (ch 03):
    Dashboard URL: <link>
    Alert pack:    <link>

Promotion history:
    <previous pair> → <this pair>  on <date> by <engineer>
    Canary ramp:    1% / 5% / 25% / 50% / 100% soak times
    Rollback plan:  roll to <previous pair record>

Open peer-boundary items:
    Performance: <list of open tickets with the hand-off bundle attached>
    Fine-tuning: <list of open retrain requests with the ch 04 bundle attached>

Next quarterly audit: <date>
Signed by: <on-call operator>, <fine-tuning counterpart>, <performance counterpart>
```

This document is reviewed quarterly (chapter 04 Practice 2) and updated whenever the pair changes.

## Summary

- Speculative decoding touches three tracks: operator (this module), kernel (`ai-infra-performance`, peer L35), drafter training (`fine-tuning-engineer`, peer L30).
- Three-way table (above) names ownership per concern. The two hand-off bundles are documented templates — a performance-peer bundle (config + health-check status + Nsight trace + specific question) and a fine-tuning-peer bundle (traffic sample + mix snapshot + acceptance criteria).
- Three common cross-boundary mistakes: opening a kernel ticket for an acceptance issue; opening a fine-tuning request for a kernel issue; asking fine-tuning to solve a compute-bound regime.
- The gate between performance and fine-tuning: *is `α` the problem, or is the speedup-per-token the problem?* Chapter 03's dashboards structure the question.
- Every deployed pair ships a one-page sign-off: pair record, baseline tables, disable policy, monitoring links, open peer-boundary items, next audit.
- Compose with other modules: 102 (engine), 103 (scheduler), 104 (KV), 105 (quantisation triggers lifecycle), 107 (topology), 109 (autoscale), 110 (SLO / cost), 113 (rollouts).
