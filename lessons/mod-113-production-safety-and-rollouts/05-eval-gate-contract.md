# Chapter 05 — The Eval-Gate Contract with `ai-eval-engineer` and `model-evaluation-engineer`

> A model promotion is not an inference engineer's unilateral decision. Two upstream peer tracks produce the evals that gate promotion — application-layer evals from `ai-eval-engineer` (L30) and capability / benchmark evals from `model-evaluation-engineer` (L30). The contract between those three roles decides what passes, what fails, and what degrades gracefully. This chapter writes it.

## Motivation

Chapter 01's canary gate includes an eval-gate step. Chapter 02's blue-green assumes a pre-cutover eval has passed. Chapter 04's post-mortem produces fixtures that back-feed into eval sets. Every one of those gates is produced by someone *other* than the inference engineer: the eval peers.

Without a contract, the common failure modes are:

- The ML-eval peer's benchmark gate passes but the AI-eval peer's application eval regresses, and nobody notices because the two live in different repos with different owners.
- The AI-eval peer's rubric shifts between model versions and the "pass" signal is comparing apples to a different apple; the inference engineer trusts a gate that no longer means what it used to.
- A regression is discovered in production and back-fed to "the eval set" — but there are two eval sets and nobody owns the one that would have caught it.
- A silent-drift incident surfaces, both eval peers deny ownership, and the post-mortem's back-feed action item has nowhere to land.

The contract names: *which eval owns which failure mode*, *what gate blocks promotion*, *what gate forces rollback*, *what gate degrades gracefully with a feature-flag*, and *who signs what by when*. This chapter writes the contract template and names the hand-offs.

By the end of this chapter you can:

- Place a specific eval on the right peer track.
- Draft a contract that encodes pass / fail / degrade semantics with thresholds the three roles agreed on.
- Specify who signs what and when during a release window.
- Hand-off a post-incident regression fixture to the right owner.
- Avoid the five contract failure modes that make eval gates ceremonial rather than load-bearing.

## The three roles

| Role | Track | Level | Owns |
|---|---|---|---|
| **Inference engineer** | `ai-infra-inference-engineer-learning` | 35 | Production operation; the canary / blue-green / rollback mechanic; the cost SLO; `X-Model-Version` attribution; incident response for inference. Adopts the eval verdicts. |
| **Application eval engineer** | [`ai-eval-engineer-learning`](https://github.com/ai-engineering-curriculum/ai-eval-engineer-learning) | 30 | Application-layer evals — behaviour on real user tasks, task-specific rubrics, user-visible regression detection, feature-eval sets. Produces application-eval verdicts. |
| **Model evaluation engineer** | [`model-evaluation-engineer-learning`](https://github.com/ml-engineering-curriculum/model-evaluation-engineer-learning) | 30 | Capability / benchmark evals — MMLU, GPQA, HumanEval, GSM8K, task-family benchmarks, held-out academic evals, perplexity on reference corpora. Produces benchmark-eval verdicts. |

The two eval peers are distinct roles in distinct curriculum families; they are not the same person wearing two hats.

### Who produces what — the division table

| Eval kind | Who runs it | Example signals |
|---|---|---|
| Benchmark score (MMLU, GPQA, HumanEval, GSM8K, task-family benchmarks) | `model-evaluation-engineer` | Absolute score; bootstrap CI vs. last-known-good. |
| Perplexity on held-out corpora | `model-evaluation-engineer` | Perplexity delta ≤ tolerance. |
| Application behaviour on product tasks (summarisation quality on your product's inputs, tool-call correctness, retrieval-faithfulness for RAG, etc.) | `ai-eval-engineer` | Rubric score; LLM-judge score; task-specific metric. |
| Refusal-rate / safety eval | `ai-eval-engineer` (often in partnership with a safety / alignment team if separate) | Refusal rate per safety category; jailbreak-resilience. |
| Streaming behaviour (TTFT / TPOT under eval traffic) | Inference engineer (this role); eval traffic from the above two. | Latency SLOs under evaluation workload. |
| Response-equivalence against last-known-good (shadow-rollout output diffs) | Inference engineer produces the mechanic; eval peers score it. | Embedding similarity, schema conformance, refusal delta. |
| Cost-per-token under eval traffic | Inference engineer. | $ / 1M tokens for the eval workload shape. |
| Runtime-specific regressions (NCCL hang, speculative-decoding acceptance, etc.) | Inference engineer. | Chapter 04 fixtures. |

The rule of thumb: *if the eval measures "what the model produces as a function," it is an eval-peer concern. If it measures "what the inference system does when serving the model," it is this role's concern.*

## The contract template

A one-page contract between the three roles. Author it per release stream (one for the chat model family, one for the embedding family, one for each distinct serving product). Keep it versioned, reviewed on a cadence (module 114's calendar), and signed.

```
# Eval-Gate Contract — <product / model family> — effective <YYYY-MM>

## Signatories
- Inference engineering lead: <name>
- Application eval engineering lead: <name>
- Model evaluation engineering lead: <name>
- Product partner (optional, for product-visible regressions): <name>

## Gate set

### Benchmark gate (owned by model-eval, run per promotion)
- Benchmarks: <list; e.g. MMLU-5shot, GSM8K-strict, HumanEval-plus, long-context-NIAH, etc.>
- Tolerance: candidate score not more than <X> absolute percentage points below last-known-good,
  verified with bootstrap CI at <confidence level>.
- Verdict: PASS / FAIL.
- SLA on verdict: <time budget> after candidate availability.

### Application eval gate (owned by ai-eval, run per promotion)
- Eval sets: <list; e.g. internal-product-rubric-v4, tool-call-correctness-v2, retrieval-faithfulness-v1>
- Tolerance: candidate rubric / judge score not more than <Y> points below last-known-good; refusal-rate delta within ± <Z>.
- Verdict: PASS / FAIL.
- SLA on verdict: <time budget> after candidate availability.

### Shadow-rollout gate (owned by inference, scored by ai-eval + model-eval jointly)
- Mechanic: shadow traffic at <M%> mirror rate for <N hours>.
- Equivalence tolerance against primary: <embedding similarity ≥ A>, <refusal rate delta ≤ B>, <schema conformance ≥ C>.
- Verdict: PASS / FAIL.

### SLO gate (owned by inference, checked post-canary)
- SLO axes: TTFT_P95, TPOT_P95, ITL_P99, cache-hit, cost/1M-tokens — all per workload class.
- Tolerance: SLO targets as published in module 110 chapter 02; no burn-rate alert on candidate-scoped metrics during ramp.
- Verdict: PASS / FAIL.

## Gate semantics

| Gate outcome | Action |
|---|---|
| Benchmark fail | BLOCK promotion. No canary. Return to upstream with the regression report. |
| Application eval fail | BLOCK promotion. No canary. Return to upstream. |
| Shadow-rollout equivalence fail | DOWNGRADE to extended shadow period (longer dwell, larger sample) + application eval review. If still failing after extended period, BLOCK. |
| SLO gate fail on canary | AUTO-ROLLBACK (chapter 01). Return to upstream with runbook referenced. |
| Any gate fail *after* promotion to 100% | ROLLBACK via chapter 02; file incident per chapter 04. |

## Graceful degradation

For non-critical application-eval failures (minor rubric regressions within tolerance, specific task-class dips):
- The feature using the regressed class is placed behind a feature flag with the degraded version.
- The AI-eval peer owns the feature-flag's roll decision with the product partner.
- Inference serves both versions (chapter 02 in-place dual-load pattern, or routed per-feature).

## Post-incident fixture back-feed
- Chapter 04 regression fixtures land on the right eval peer per the division table.
- Fixture SLA: fixture merged into eval set within <N business days> of the post-mortem.
- The inference engineer provides the reproducer; the eval peer reviews and merges.

## Review cadence
- Contract reviewed quarterly.
- Threshold values re-negotiated when a new model family is introduced, when a benchmark is retired, or when the SLO contract changes.
```

### Explaining the semantics

- **BLOCK** is the clean case — a gate fail means the candidate does not proceed. The chapter 01 / 02 rollout machinery never starts. Upstream (the model-training peer, the quantisation author, the engine maintainer) owns the fix.
- **DOWNGRADE** is the escape hatch — a shadow-rollout equivalence mismatch *might* be noise from sampling, might be a real but minor shift; the right response is more data, not a flat block.
- **AUTO-ROLLBACK** is the chapter 01 mechanism — SLO failures during a live ramp don't need human decisioning; the rollback is automatic and an incident is filed for the RCA.
- **ROLLBACK** (post-promotion) is chapter 02's mechanism — slower, costlier, but the right tool when a regression surfaces after the ramp completed.
- **Graceful degradation** is the pragmatic case — the eval peer and the product partner decide whether a specific minor regression is worth blocking the whole release or routing around with a feature flag.

## Post-incident fixture back-feed — the operational flow

Chapter 04's regression-fixture back-feed is a cross-role workflow. The contract names it explicitly:

1. **Incident resolved; post-mortem drafted.** Inference engineer identifies a fixture — a reproducible `(prompt, params, seed, expected behaviour, workload class)` that would have caught the regression if run as part of the pre-promotion gate.
2. **Fixture routed to the right peer.** From the division table above: an application-level regression → `ai-eval-engineer`; a capability regression → `model-evaluation-engineer`; a runtime regression (NCCL-specific long-context trigger, etc.) → stays with inference and goes into the canary acceptance gate.
3. **Peer reviews the fixture.** The eval peer checks relevance, deduplicates against existing fixtures, adjusts the eval-set schema, merges a PR.
4. **Fixture becomes a gate.** Next promotion runs the fixture; a regression now fails the appropriate gate pre-promotion rather than reaching production.
5. **Tracking.** The post-mortem's action item is "merged" only when the fixture PR is merged. A monthly audit (module 114) closes stale items.

### SLA discipline

A fixture that takes six weeks to merge is a fixture that didn't exist for the next release. The contract names the SLA — typically 5–10 business days from post-mortem to merged fixture. Missing the SLA is a cross-team conversation at the module-114 cadence, not an inference-engineer-chases-eval-peer email.

## On-call interaction pattern

Three cross-role interactions happen outside release windows:

- **Eval regression outside a release.** An automated eval run (e.g. nightly benchmarks) regresses on the production model with no recent deploy. Could be: eval-infrastructure change, dataset refresh that changed difficulty, a newly-discovered sampling issue. The eval peer owns investigation; the inference engineer confirms no silent deploy / config change on the serving side.
- **Production anomaly with no SLO alert.** A product metric shifts; the inference engineer checks `X-Model-Version` and deploy events. If no change, escalate to the AI-eval peer with a request for an unplanned eval run against the current primary.
- **Pre-release question on eval tolerance.** A new model family's first promotion raises "is 0.94 the right cosine threshold for this product?" Convene the three roles plus the product partner; adjust the contract; re-release.

## Five contract failure modes

- **Threshold drift.** Thresholds originally picked for a 7B model are too permissive for a 70B model (or vice versa). Fix: thresholds reviewed when model class changes.
- **Overlap without ownership.** Both eval peers own "refusal rate" and neither signs off. Fix: the division table names a primary and a secondary.
- **Gate ceremony.** The gate "passes" because nobody actually looks at the output. Fix: a dashboard that shows the gate result alongside the rollout step; chapter 01's ramp refuses to proceed until the dashboard reads green.
- **Fixture back-feed deferred.** Post-mortem action closed, PR never merged. Fix: audit at module 114's monthly cadence; the audit is the forcing function.
- **Rollback without eval re-run.** After a rollback, the team moves on without running the gates on the primary again — a bad primary stays. Fix: contract specifies post-rollback re-eval.

## Worked example — a tool-call correctness regression caught by application eval

Context: candidate `mymodel-70b/v2-fp8`. Benchmark gate (MMLU, GSM8K, HumanEval, long-context NIAH) passed cleanly. Shadow-rollout equivalence score on open chat: 0.96 cosine — within tolerance.

- **Application eval peer** runs the product's tool-call correctness eval set (v2, maintained by them). Candidate scores 92 % vs. last-known-good 95 %; tolerance is 2 %; verdict FAIL.
- **Contract gate outcome:** BLOCK promotion. Chapter 01 ramp never starts.
- **Upstream investigation** (not this role): the FP8 quantisation applied aggressive scaling on an MLP layer that the tool-call path happens to be sensitive to. A revised quantisation re-runs the gates two weeks later and passes.
- **Operational consequence** here: the inference engineer did not touch production. Cost-of-failure was the eval run, not an incident.

The *point* of the contract is that this is boring: the gate blocks, upstream fixes, release re-runs, no outage.

## Summary

- Two upstream eval peers at L30 produce the evals that gate inference promotion; the inference engineer adopts those verdicts and runs the SLO / cost gate.
- The contract names the division of ownership, the gate set, the semantics (BLOCK / DOWNGRADE / AUTO-ROLLBACK / graceful degradation), and the post-incident back-feed SLA.
- Thresholds are product decisions — reviewed quarterly and when model class changes.
- Fixture back-feed into the correct eval set is a blocking post-mortem action item; a monthly audit enforces follow-through.
- On-call cross-role interactions follow the division table; escalate quickly when ownership is ambiguous.
- Chapter 01's canary and chapter 02's blue-green assume this chapter's contract is in place; chapter 04's post-mortem's back-feed lands here; chapter 06's deprecation uses the contract's signatories as the consumer-migration sign-off.
