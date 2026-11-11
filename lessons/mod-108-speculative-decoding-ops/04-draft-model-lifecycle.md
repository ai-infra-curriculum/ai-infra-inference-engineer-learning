# Chapter 04 — Draft-Model Lifecycle: Retrain, Versioned Promotion, Rollback

> A draft is a second artefact on your platform's shelf. Its correctness is defined by its acceptance rate against a specific target, on a specific workload. When the target changes, the workload changes, or the acceptance rate drifts, something in the pair has to move. This chapter names the triggers, the promotion contract, and the rollback discipline.

## Motivation

A team runs Llama-3.1-70B as target with a distilled 1B draft, `α ≈ 0.72` on their canary set. Three events happen over six months:

1. **Target rev.** The platform promotes `target_v1` (Llama-3.1-70B-Instruct @ sha-A) to `target_v2` (Llama-3.1-70B-Instruct-fp8 @ sha-B — same base, re-quantised with the module-105 pipeline). The draft is unchanged. Acceptance rate drops from `0.72` to `0.58` because the quantised target's output distribution shifted slightly. Nobody noticed until TPOT P99 breached. Fix: distil a new draft against the FP8 target; promote the *pair* `(target_v2, draft_v2)` as a unit.

2. **Workload rev.** A new customer onboards whose traffic is 70 % reasoning (chain-of-thought). The existing draft was distilled against a chat-weighted mix; its acceptance on reasoning is `0.42`. Fix: retrain the draft with a 30 % reasoning weight; promote as `(target_v2, draft_v3)`.

3. **Rollback.** The next target revision (`target_v3`, a merge from upstream) regresses the eval harness. The platform rolls back the target to `target_v2`. Someone forgets to roll back the draft, leaving `(target_v2, draft_v4)` in production — a pair that was never co-validated. Acceptance drops; a few workload classes see TPOT regress; a confusing week ensues. Fix: the lifecycle contract says *the pair is the unit, never roll back one without the other.*

Each of these is a lifecycle event with a well-defined trigger, deliverable, and rollback. This chapter writes them.

By the end of this chapter you can:

- Enumerate the triggers for a draft retrain (target drift, workload drift, acceptance drift, new workload class, draft architecture upgrade).
- Author the versioned-pair promotion contract — "the pair `(target_vX, draft_vY)` is the deployable unit; neither side promotes alone."
- Author the rollback discipline — if you roll back the target, roll back the draft to the paired version; never leave an unverified pair in production.
- Write the hand-off bundle to the fine-tuning peer when a retrain is the right action, and the rejection criteria that distinguish "retrain" from "the engine is broken."
- Carry the same lifecycle shape to Medusa, EAGLE, and Lookahead with the appropriate simplifications (Medusa / Lookahead have one artefact to version; EAGLE has two like draft-target).

Specific org-level model-registry patterns and promotion-ritual tooling (MLFlow, Weights & Biases, in-house) are organisation-dependent; the contract below is engine-and-tool-agnostic — implement it with whichever registry your org already runs.

## The pair as the unit

### What "the pair" means

A deployable speculative serving config consists of (at minimum):

- `target_name@sha` — the target checkpoint and its weight SHA (safetensors or TRT engine hash).
- `draft_name@sha` — the draft checkpoint and its weight SHA. (For Medusa / Lookahead, there is no separate draft; see below.)
- `engine_version@sha` — the inference engine's pinned version.
- `k_default` — the speculative length (or the dynamic-`k` policy).
- `scheme` — draft-target / Medusa / EAGLE / Lookahead.
- `baseline_α(workload_class)` — the paired acceptance baseline measured on canary traffic for each workload class.

Call this tuple a **pair record**. It is the atomic unit of promotion and rollback. The registry stores one row per pair; the engine configuration references one pair.

For Medusa / Lookahead, `draft_name@sha` is replaced by `scheme`'s intrinsic artefact (Medusa heads bundled with the target checkpoint — one artefact; Lookahead has no draft weights). The invariant — "paired baseline co-measured, co-promoted, co-rolled-back" — still holds; the number of files on disk is just smaller.

### Why the pair is the unit

Acceptance rate is a property **of the pair**, not of either side alone. A target's expected `α` is undefined without naming the draft. A draft's expected `α` is undefined without naming the target. Treating either side as a stand-alone artefact invites the three failure modes in the motivation.

The pair-level discipline mirrors module 105's quantization promotion (the quantised model + its eval harness are one artefact) and module 113's canary / blue-green discipline for inference changes.

## The trigger set — when to act on the lifecycle

Five triggers. Each has a defined signal, a defined action, and a defined boundary (what you do vs. what the fine-tuning peer does vs. what the engine team does).

### Trigger 1 — Target drift (the target changed; the draft didn't)

**Signal.** A target revision is in the deploy pipeline — new quantization, new fine-tune, new upstream merge, new engine-build-time optimisation that changes the output distribution.

**Action.**

- Distil / fine-tune a new draft against the new target. (Fine-tuning peer's deliverable.)
- Baseline the new pair on canary. (This module.)
- Promote `(target_new, draft_new)` atomically. (This module + module 113's rollout discipline.)

**Boundary.** The draft retrain is the peer's work; the pair baseline + atomic promotion is this module's work.

### Trigger 2 — Workload drift (the target and draft didn't change; the workload did)

**Signal.** Chapter 03's alerts fire — fleet acceptance drift, per-class collapse, or a tenant-level regression — and the correlation points to a workload shift (new tenant class, new prompt shape, new generation shape) rather than to the engine or the artefacts.

**Action.**

- Characterise the shift. What class now accepts worse? What is the volume of that class? Is the shift durable (tenant onboarding, product change) or transient (an experiment)?
- If durable and the volume is large: request a draft retrain with the new mix represented. Hand off to the fine-tuning peer.
- If durable and the volume is small: consider disabling speculative for the affected class / tenant (chapter 05 covers the policy) rather than retraining.
- If transient: ride it out; keep speculative disabled for the slice if the SLO is burning; return to normal when the experiment ends.

**Boundary.** Characterising the shift and deciding the posture is this module. The retrain itself (if invoked) is the peer.

### Trigger 3 — Acceptance drift (nothing obvious changed; `α` is sliding)

**Signal.** Chapter 03's rolling-baseline alerts fire; the fleet-level and per-class panels are drifting slowly, no release correlates, no clear workload shift.

**Action.** The three candidate causes to investigate in order:

1. **Silent target drift.** Was there a quantization re-pack, a tokenizer re-canonicalisation, or a sampler-defaults change on the target side that didn't register as a formal rev? If yes → treat as Trigger 1.
2. **Silent workload drift.** Did the tenant mix shift under the top-N tracking (the long tail added up to a shift)? If yes → treat as Trigger 2.
3. **Engine / stack regression.** Was there an engine version bump, a CUDA driver update, a library update that changed draft behaviour? If yes → route to the engine docs / performance peer (chapter 06).

If none of the above account for the drift, the draft itself may simply be aging — the training data mix is drifting from current traffic over months. Retrain against a fresh traffic snapshot. Peer deliverable.

### Trigger 4 — New workload class

**Signal.** A product change introduces a new workload class — a new tool-use pattern, a new structured-output schema, a new reasoning mode. Chapter 03's taxonomy picks it up as its own class; its acceptance baseline is unknown.

**Action.**

- Measure. Enable speculative; let the metric accumulate for enough samples to be legible.
- Decide. If `α` is above the class break-even threshold (chapter 05 computes), keep enabled. If below, disable for the class until a draft that covers it is available.
- Baseline. The paired baseline for this class is a new row in the pair record.
- Optionally request a draft retrain. If the class is strategically important, update the training mix to include it. Peer deliverable.

### Trigger 5 — Draft architecture upgrade

**Signal.** A new paper / open-source release suggests the draft architecture can be improved (e.g. a stronger EAGLE variant, a Medusa upgrade, a different draft family that matches the tokenizer and shows better `α` in published numbers).

**Action.**

- Evaluate on canary. Build the candidate `(target_vX, draft_vZ)`, baseline on canary, compare `α` and TPOT to the incumbent.
- Promote only if the paired baseline wins by a margin that justifies the lifecycle churn. "Wins" has a specific numeric threshold your fleet agrees on (e.g. `α` by `≥0.05` on the dominant class and no class regresses by more than `0.03`).

**Boundary.** The upgrade *decision* is this module. The upgrade *implementation* (new EAGLE drafter, new Medusa heads) is the peer.

## The promotion contract

Four rules. Every pair promotion follows them.

### Rule 1 — Co-baseline on canary

Every new pair record must have canary measurements for `(α, TPOT, cost/1M-tokens)` per workload class **before** it goes to production. The canary runs under the same engine version, same TP topology, same `k`, same sampler config as the production target.

The canary measurement window is long enough to accumulate statistical significance per class. The output is the pair's baseline table:

```
| workload_class | α (canary) | α 95% CI | TPOT_P50 | TPOT_P99 | cost/1M |
|---|---|---|---|---|---|
| chat           | 0.73       | ±0.02    | 22 ms    | 44 ms    | $X      |
| code           | 0.86       | ±0.01    | 17 ms    | 32 ms    | $Y      |
| long_context   | 0.65       | ±0.03    | 28 ms    | 58 ms    | $Z      |
| reasoning      | 0.44       | ±0.04    | 51 ms    | 95 ms    | $W      |
```

A pair without a baseline table cannot promote. A pair with a baseline that regresses a class beyond the promotion threshold cannot promote without an explicit sign-off.

### Rule 2 — Promote the pair atomically

In the serving config, the pair is one object. Change `target_name@sha` → `draft_name@sha` must change in the same commit, the same deploy, the same rollout. The engine config files, the registry entries, the autoscaler's "what artefact am I serving" labels all move together.

Mechanically:

- One deploy YAML / Helm values entry per pair.
- The deploy pipeline rejects a diff that changes only `target_name@sha` without a correspondingly staged `draft_name@sha`.
- The engine's startup log emits both SHAs; the health check verifies both.
- Chapter 03's `model_pair` label is the pair record's slug; a mismatch in the metric tells the operator the deploy is split.

### Rule 3 — Canary-ramp with speculative-specific gates

A pair's canary ramp follows module 113's canary pattern with speculative-specific gates:

- Ramp 1 % → 5 % → 25 % → 50 % → 100 %, with soak at each step.
- At each step, verify:
  - Fleet `α` ≥ canary baseline `α` − 0.03 per class.
  - TPOT P99 ≤ canary baseline × 1.1 per class.
  - No new bypass spike (chapter 03's Alert 4).
  - No per-tenant SLO breach (chapter 03's Alert 3).
- Auto-rollback (Rule 4) on any gate failure.

### Rule 4 — Rollback is first-class and atomic

If the pair fails a gate (acceptance below baseline, SLO breach, bypass spike) or if any downstream rollback is triggered on the target or draft side, the rollback is **of the pair**, not of one side.

**Explicit rule:** *if you roll back the target, you roll back the draft to the version paired with that target. Never leave `(target_vX, draft_vY)` in production unless the pair record names it as a verified configuration.*

Mechanically:

- The registry stores every pair record as immutable. "Rollback" means re-selecting a previous pair record and re-applying its config.
- The deploy pipeline rejects a diff that rolls back `target_name@sha` without correspondingly rolling back `draft_name@sha`.
- An auto-rollback button (if your pipeline has one) rolls the pair; the pipeline does not expose "roll back only the target."
- The on-call runbook mentions the pair discipline on every rollback entry.

## The hand-off to the fine-tuning peer

When a trigger lands on "retrain the draft," the operator writes a one-page retrain request to the fine-tuning peer. The bundle:

```
# Retrain request — draft for <target_name@sha>

## Context
- Target: <name>@<sha>  (link to registry entry)
- Current draft: <name>@<sha>  (link to registry entry)
- Reason for retrain: <trigger — target drift / workload drift / α drift / new class / architecture upgrade>
- Date requested: <date>; needed by: <date>

## Measurements that motivate the request

Baseline paired α per class (link to chapter 03 panel snapshot):
<attach the full per-class α table — current and 14-day history>

Workload-mix snapshot that the retrain should target:
<class × share of traffic by request count, by token count>
<prompt-length and generation-length distributions per class>

Expected failure modes the current draft shows:
<e.g. "poor long-prompt behaviour (α=0.5 vs 0.72 on short)">

## Acceptance criteria for the new draft

Primary metric: paired α on canary traffic.
  - Must meet: α(chat) ≥ 0.70, α(code) ≥ 0.80, α(long_context) ≥ 0.60,
               α(reasoning) ≥ 0.50 (adjust per fleet).
  - No class regresses by more than 0.03 vs. current draft.

Secondary metrics:
  - Draft-step cost ratio c_d / c_t ≤ <threshold> on canary measurements.
  - Draft HBM footprint ≤ <bytes> (so the serving config doesn't need re-sizing).
  - Output distribution preserved under rejection sampling (greedy regression suite passes).

## What this module provides

- A canary traffic sample (<N> requests, representative class mix, scrubbed).
- Access to the current pair's baseline table.
- A test harness that measures α for a candidate draft against the current target
  on the canary sample.
- An on-call engineer to co-validate the new pair on canary within the week after
  the fine-tuning peer ships a candidate.

## What the fine-tuning peer provides

- The retrained draft checkpoint + weight SHA.
- The training mix / recipe summary (one page; so this module can reason about
  future workload-drift triggers).
- Any changes to the draft's forward pass that the engine needs to know about
  (new tokenizer, new vocab, new model class) — flagged for ch 01's deployment
  walkthrough review.

## Decision rights

- The fine-tuning peer owns the retraining recipe (data, loss, optimiser, epochs).
- This module owns the acceptance criteria and the canary acceptance.
- Both sign off on the pair before promotion.
```

A retrain request without the above fields comes back from the peer. A peer deliverable without the acceptance criteria met goes back for iteration. Chapter 06 covers the full boundary.

## The dash of lifecycle hygiene

Beyond the triggers and the contract, four practices keep the lifecycle honest:

### Practice 1 — Pair registry is append-only

Every pair that ever ran in production has a row in the registry — target SHA, draft SHA, engine version, baseline table, canary results, promotion date, retire date (if applicable). Append-only; nothing deleted. The registry is the oracle for "what pair was serving when this metric fired."

### Practice 2 — Quarterly pair audit

Once a quarter, review:

- Every active pair's current `α` vs. its deploy-time baseline. Pairs that have drifted >`0.05` without a trigger ever having been raised get a scheduled retrain.
- Every retired pair that is still "warm" in the registry — is it still worth keeping as a rollback candidate, or has the engine moved past it?
- Every trigger that fired in the quarter — did the response match the trigger? If drift triggers kept getting "disable for class" responses, maybe the fleet needs a different draft family (not a retrain of the same architecture).

### Practice 3 — Pair records in CI

The pair record for each deployed fleet lives in the serving config repo. Every commit that touches `target_name@sha` is CI-checked against a correspondingly-named `draft_name@sha`. Mismatches fail CI. The pair record is a first-class config file, not an operator note.

### Practice 4 — Baseline snapshots in storage

The canary baseline tables are stored (object store, config repo, timeseries baseline table). Chapter 03's drift alerts query them. If the storage is corrupted or lost, the alerts degrade silently — build a canary-replay harness that can re-measure a pair's baseline on demand from a frozen traffic sample.

## Three example lifecycle walks

### Walk 1 — FP8 target promotion forces a draft retrain

**Trigger.** Target team completes module-105 FP8 conversion of Llama-3.1-70B. Expected engine latency improvement is 1.4×. The target team wants to ship Monday.

**Action timeline.**

- *Mon*: FP8 target lands in staging; current draft (BF16, distilled against BF16 target) is kept.
- *Tue*: Canary A/B. Pair `(target_fp8, draft_bf16)` acceptance measured: `α(chat)` drops from `0.72` to `0.62`. The target's output distribution shifted enough to matter.
- *Wed*: Decision: do not promote the pair until a new draft is distilled against the FP8 target. Open a retrain request to the fine-tuning peer.
- *T+2 weeks*: Peer ships `draft_v2`. Canary pair `(target_fp8, draft_v2)` → `α(chat) = 0.74`. Meets acceptance criteria.
- *T+2 weeks + canary ramp*: Pair promoted atomically through the chapter-04 Rule 3 ramp.

### Walk 2 — Reasoning workload onboards; draft doesn't cover it

**Trigger.** A new tenant launches with 60 % reasoning traffic. Chapter 03's per-tenant panel shows their `α = 0.44`; their TPOT P99 is 95 ms vs. their SLO of 60 ms. Chapter 03's Alert 3 is firing.

**Action.**

- Immediate: enable the "disable speculative for this tenant's reasoning class" fallback (chapter 05). TPOT falls to the non-speculative baseline (~60 ms); SLO just-barely holds.
- Characterise: this tenant's traffic is projected to grow 10× over the quarter. The reasoning class is a durable product direction, not a transient experiment.
- Retrain request to the fine-tuning peer: new draft with reasoning class represented in training mix at 20 % (up from current ~5 %).
- When the new draft lands: canary → promote the pair.

### Walk 3 — Target rollback forces draft rollback

**Trigger.** `target_v3` (freshly promoted) regresses the eval harness: a module-113 canary catches a quality drop. The target team rolls back to `target_v2`.

**Action timeline.**

- The deploy pipeline tries to roll back `target_name@sha` from `v3-sha` to `v2-sha`.
- The CI check (Practice 3) blocks the diff: `target_name` rolled back without a corresponding `draft_name` rollback.
- Deploy engineer re-authors the diff as "roll back the pair record from `(v3-sha, draft_v3)` to `(v2-sha, draft_v2)`."
- CI passes; pair rolls back atomically.
- Chapter 03's post-deploy acceptance alert (Alert 6) does not fire because the paired baseline for `(v2, draft_v2)` is still in the registry and matches current `α`.

Without the pair discipline: the pipeline would roll back only the target, leaving `(target_v2, draft_v3)` in production. `draft_v3` was distilled against `target_v3`'s distribution; against `target_v2` it accepts worse (hypothetically, drops by `0.08`). The next week is spent debugging "the engine regressed after we rolled back."

## Lifecycle shape for Medusa, EAGLE, and Lookahead

The invariants above apply, with simplifications:

- **Medusa.** One artefact (the target with Medusa heads). The "pair" is `(target+heads_name@sha, scheme=medusa)`. Retrain means retraining the Medusa heads (fine-tuning peer); promotion is still atomic (the heads and the target weights ship as one file). No two-sided version drift risk.
- **EAGLE.** Two artefacts (target + EAGLE drafter). Same discipline as draft-target. The EAGLE drafter has its own retrain cycle driven by the same triggers.
- **Lookahead.** One artefact (the target). No draft to retrain. The "lifecycle" is just the target's lifecycle. Chapter 03's alerts for Lookahead focus on the n-gram cache's effectiveness on the current workload — if the workload shifts away from repetitive patterns, Lookahead's `α` collapses and the operator disables Lookahead for the affected classes (chapter 05) rather than retraining.

Chapter 03's monitoring is scheme-agnostic; chapter 04's lifecycle is scheme-aware (number of artefacts changes).

## The boundary to the peers

This chapter does not teach:

- **How to distil a draft.** The loss (KL / feature matching / preference), the data mix, the optimiser, the epochs. Fine-tuning peer.
- **How to train Medusa heads or an EAGLE drafter.** Same peer.
- **How the engine loads a `(target, draft)` pair atomically.** Engine design; the operator consumes the behaviour.
- **The model-registry software.** MLFlow, W&B, in-house — organisational / platform choice.

Those live at [`fine-tuning-engineer`](https://github.com/ai-infra-curriculum/fine-tuning-engineer-learning) (training) and at the engine project (loading). This chapter is the operator's lifecycle view: *what triggers an action, what the action's bundle looks like, what the promotion contract is, how rollback works.*

## Summary

- The pair `(target, draft, engine, k, scheme, baseline_α(class))` is the deployable unit. Promote atomically; roll back atomically; version the pair, not either side.
- Five triggers: target drift, workload drift, acceptance drift, new workload class, draft architecture upgrade. Each has a defined action and a defined boundary.
- Four promotion rules: canary-baseline before promotion, atomic pair promotion, canary ramp with speculative-specific gates, atomic rollback.
- The retrain hand-off to the fine-tuning peer is a one-page bundle: context, measurements, acceptance criteria, decision rights. Chapter 06 extends.
- Four hygiene practices: append-only pair registry, quarterly pair audit, pair records in CI, baseline snapshots in storage.
- Medusa / Lookahead have one artefact; EAGLE / draft-target have two. The pair discipline simplifies but doesn't disappear.
- Draft training is the peer's deliverable; pair acceptance, canary baselining, atomic promotion / rollback, and the lifecycle contract are this module's.
