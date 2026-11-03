# Exercise 04 — Draft-Model Lifecycle Contract

**Time:** 2.5 hours · **Output:** one lifecycle contract document, one pair registry schema, one retrain-request bundle example, one CI check specification, one rollback runbook

## Goal

Make [Chapter 04](../04-draft-model-lifecycle.md) concrete by **authoring the lifecycle contract** that governs every `(target, draft)` pair on your fleet. The deliverable is the one-page document that your platform team, the fine-tuning peer, and the engine team all sign off on — the thing that prevents the three motivating failures in chapter 04 from happening on your watch.

## Problem statement

Your platform runs speculative decoding (exercise 01 / 02). The monitoring is in place (exercise 03). The next six months will bring at least one target revision, at least one draft retrain, and at least one rollback. If the lifecycle contract isn't written before those events happen, the first one will teach it the hard way — three days of post-mortem, a leaked `(target_v2, draft_v1)` pair in production, a tenant SLO breach that gets traced back to "we forgot to update the pair record."

Write the contract today.

## Prerequisites

- Chapter 04 of this module read; chapter 06 skimmed (you need the hand-off bundles).
- Exercise 03 completed — the dashboard, baseline storage, and alerts the contract depends on.
- A sketch of your organisation's model-registry setup (MLFlow, W&B, in-house). The contract is tool-agnostic but must be instantiatable in your registry.
- Representative examples of the artefacts the contract tracks — one target checkpoint and one draft from your fleet (even if just paper references).

## Requirements

Five deliverables.

### Deliverable A — the lifecycle contract (one-page document)

A one-page document that answers every question in the chapter-04 contract shape:

```
# Speculative-Decoding Pair Lifecycle Contract — <fleet_name>

## Scope
Applies to every (target, draft) pair deployed on <fleet_name>.
Pair types covered: draft-target, Medusa, EAGLE, Lookahead.

## The pair as the unit
Pair record fields:
  - target_name@sha
  - draft_name@sha (or scheme-intrinsic artefact for Medusa / Lookahead)
  - engine_version@sha
  - k_default (or dynamic-k policy)
  - scheme
  - baseline_α(workload_class) — canary table

## Promotion contract
Rule 1 (co-baseline on canary): <contract text>
Rule 2 (atomic pair promotion): <contract text>
Rule 3 (canary-ramp with speculative-specific gates): <contract text; name gates>
Rule 4 (rollback is first-class and atomic): <contract text>

## Trigger set
Trigger 1 (target drift): <who detects, what action, where to route>
Trigger 2 (workload drift): <...>
Trigger 3 (acceptance drift): <...>
Trigger 4 (new workload class): <...>
Trigger 5 (draft architecture upgrade): <...>

## Decision rights
Who signs off on a pair before promotion.
Who can trigger a rollback.
Who authors the retrain request to the fine-tuning peer.

## Hygiene practices
Pair registry: append-only, located at <registry URL / path>.
Quarterly audit: cadence, participants, output.
CI check: location of the enforcement.
Baseline snapshots: storage, retrieval.

## References
- Monitoring dashboard: <URL> (exercise 03)
- Alert pack: <URL> (exercise 03)
- Retrain-request template: <URL> (Deliverable C)
- Rollback runbook: <URL> (Deliverable E)
```

### Deliverable B — the pair registry schema

A schema (SQL table, JSON Schema, Pydantic model, or protobuf — pick what your platform runs) for the pair-record. Fields the chapter-04 contract demands:

```sql
CREATE TABLE pair_record (
    id                      SERIAL PRIMARY KEY,
    slug                    TEXT UNIQUE NOT NULL,       -- e.g. 'llama31-70b-fp8__draft-llama32-1b-bf16'
    target_name             TEXT NOT NULL,
    target_sha              TEXT NOT NULL,
    draft_name              TEXT,                        -- NULL for Medusa / Lookahead
    draft_sha               TEXT,
    scheme                  TEXT NOT NULL,               -- 'draft-target' | 'medusa' | 'eagle' | 'lookahead'
    engine_name             TEXT NOT NULL,
    engine_version          TEXT NOT NULL,
    k_default               INTEGER,                     -- NULL for schemes with no explicit k
    k_policy                JSONB,                       -- optional dynamic-k config
    scheme_params           JSONB,                       -- scheme-specific knobs (eagle topk, medusa num_heads, ...)
    baseline_canary_date    TIMESTAMP NOT NULL,
    baseline_table          JSONB NOT NULL,              -- per-workload-class α, TPOT, cost/1M
    promoted_at             TIMESTAMP,
    retired_at              TIMESTAMP,
    promoted_by             TEXT,
    sign_off_record         JSONB,                       -- who signed, when, what they verified
    notes                   TEXT
);
-- append-only; no UPDATE / DELETE outside corrections with audit trail.
```

(Or equivalent in your registry's shape.) The schema must support the chapter-04 Practice 1 "append-only" discipline and the Practice 4 "baseline snapshots in storage" discipline.

### Deliverable C — an example retrain-request bundle

Instantiate the chapter-04 retrain-request bundle against a plausible scenario on your fleet. The scenario:

- Target `v2` was promoted (module 105 FP8 re-pack).
- The current `(target_v2, draft_v1)` pair's acceptance has drifted on reasoning class (α from 0.46 to 0.34).
- The new draft is requested.

Write the full one-page retrain request as the chapter-04 template specifies. Attach the canary α table (from exercise 01 or synthesised) and the workload-mix snapshot.

### Deliverable D — the CI check specification

A specification for the chapter-04 Practice 3 CI check that prevents partial promotion / rollback:

```
# CI check: pair-integrity

## Purpose
Prevent the deploy pipeline from merging a change that modifies target_name/sha
without a correspondingly staged draft_name/sha change, or vice versa — and
to prevent any serving config from referencing a pair_record that is not in
the registry.

## Trigger
On every pull request that modifies <path to serving config>.

## Checks
1. If `target_name` or `target_sha` changed in the diff, verify:
   a. `draft_name` and `draft_sha` also changed in the same diff (unless the
      scheme is Medusa or Lookahead — then the single-artefact rule allows the
      target change alone, but requires a baseline refresh).
   b. The new `(target_sha, draft_sha, scheme, engine_version)` tuple exists in
      the pair_record registry with status = `canary_passed`.
2. If `draft_name` or `draft_sha` changed, mirror check (a) and (b) for target.
3. If `engine_version` changed, verify the current pair_record has a canary
   measurement under the new engine version.
4. Rollback diffs (reverting a previous commit's pair change) are subject to
   the same checks — a rollback that leaves (old_target, new_draft) in the config
   fails CI.

## Failure modes
- Diff changes only one side → block with error "pair promotion is atomic; see
  <contract URL>."
- Pair not in registry → block with error "promote the pair through the canary
  pipeline first."
- Pair in registry but not canary_passed → block with error "pair canary not
  yet green."

## Pseudo-code (or actual script skeleton)
<concise sketch>
```

A GitHub Action, GitLab CI job, or whatever your pipeline runs — written so another engineer could implement it.

### Deliverable E — the rollback runbook

A one-page runbook for the chapter-04 Rule 4 atomic rollback. Shape:

```
# Pair Rollback Runbook

## When to invoke
- Chapter 03 Alert 6 (post-deploy acceptance regression) fires.
- A target-side rollback is triggered (quality regression, incident, etc.);
  the paired draft must roll with it.
- Any pair-level SLO breach that is traced to the current pair.

## Pre-rollback checklist
- Identify the previous pair_record (slug + id).
- Verify it is still in the pair_record registry.
- Verify the previous engine_version is still runnable on the fleet (if the engine
  has moved past it, this is a different incident; involve the engine team).

## Rollback steps
1. Set serving config `pair_record_id` to the previous pair's id.
2. Run the CI pair-integrity check (Deliverable D) on the diff.
3. Deploy through the normal canary ramp — but accelerated (1h soak per step
   instead of per-day).
4. Watch Chapter 03 alerts.
5. If all green at 100%, close the incident.

## What to NOT do
- Do NOT roll back only `target_name@sha` without rolling back `draft_name@sha`.
  CI should block this; if it somehow doesn't, abort the deploy.
- Do NOT skip the canary ramp. The previous pair's registry record is a historical
  truth, not a current-state guarantee. The fleet, the traffic, and the engine may
  have moved.

## Post-rollback
- Open a post-mortem in the standard template.
- Update the pair registry with the retirement timestamp of the regressed pair
  (chapter 04 Practice 1).
- Open a chapter-06 hand-off ticket if the root cause is in a peer's scope
  (fine-tuning for a draft issue; performance for an engine / kernel issue).
```

## Starter guidance

- **Write the contract against the fleet you actually have.** The chapter-04 template is a skeleton. If your platform has a different registry, a different canary pipeline, a different CI, adapt the field names and paths. The invariants (pair-as-unit, atomic promote/rollback, append-only registry) are the invariants.
- **Decision rights are the hard part.** Write them explicitly. "Who signs off" with no name attached is a contract gap.
- **Reference exercise 03 by URL.** The dashboard, alerts, and runbook from exercise 03 are the enforcement arm of the contract. If exercise 03's artefacts aren't addressable, the contract is paper.
- **Instantiate against a scenario in Deliverable C.** A retrain-request template with no example is a template; with an example, it's a tool.
- **Mock the CI check even if you can't run it.** Deliverable D is a spec, not necessarily running code. But the spec must be concrete enough that another engineer could write the implementation.
- **Keep it one page where possible.** Contracts that run to 10 pages don't get read. If you need more than one page for the Deliverable A main contract, cut.

## Fallbacks for constrained environments

- **No registry infra.** The schema in Deliverable B can be a `pair_records.yaml` checked into the repo. Append-only enforced via code review rather than database constraint. Document the simplification.
- **No CI system.** Deliverable D becomes a plain-text spec with a shell-script skeleton; wire-up follows when CI exists.
- **No rollback automation.** Deliverable E becomes a manual runbook; the invariant (atomic pair rollback) is enforced by on-call discipline rather than by tooling.
- **No fine-tuning peer in the org.** Deliverable C targets whoever would own the retrain (an external vendor, a research team, a wait-list). The hand-off bundle shape doesn't change.
- **Pair volume is tiny (one or two pairs ever).** The contract is overkill for a single pair; shrink to the minimum (promotion gate, rollback runbook) and skip the registry schema and audit cadence.

## Acceptance criteria

1. **Deliverable A** is a one-page document that answers every item in the chapter-04 contract shape — scope, pair-as-unit, four promotion rules, five triggers, decision rights, four hygiene practices.
2. **Deliverable B** is a schema that supports every chapter-04 pair-record field and enforces append-only (database constraints, immutable storage, or documented process).
3. **Deliverable C** is a complete, concrete retrain-request bundle for a specific scenario — not a template with placeholders.
4. **Deliverable D** is a CI check specification concrete enough for implementation — trigger, checks, failure modes, pseudo-code.
5. **Deliverable E** is a rollback runbook that references the pair registry, the CI check, and the chapter-03 alerts.
6. All five deliverables are internally consistent — a term defined in A has the same meaning in B / C / D / E.

## Common failure modes

- **Contract without enforcement.** The CI check (Deliverable D) is the teeth of the contract. Without it, the contract is a wish.
- **Separate target / draft registry entries.** Chapter 04 names the pair as the unit; if your schema has one row per target and one per draft, you're set up to leak unverified pairs. Pair-as-row is the invariant.
- **No baseline storage.** Chapter 03 and 04 both depend on retrievable deploy-time baselines. If exercise 03 skipped this, exercise 04 must fix it.
- **Rollback not atomic.** The number-one chapter-04 motivating failure. Deliverable E's "do NOT" section exists specifically to prevent this; CI (Deliverable D) catches it in review.
- **Retrain requests without acceptance criteria.** Chapter 04's hand-off bundle requires them. Deliverable C must include them concretely.
- **Promoting a pair without canary measurement for every class the fleet serves.** Deliverable A Rule 1 is explicit; CI (Deliverable D) check 1(b) is the enforcement.
- **Over-engineering.** Lifecycle discipline matters; a 20-page contract that no one reads matters less than a one-page contract everyone reads.

## Keep it

The contract (Deliverable A) is the fleet's permanent artefact; refresh it on major process changes (new engine, new scheme, new org structure). The retrain-request bundle (Deliverable C) is the template for every subsequent chapter-04 Trigger. The rollback runbook (Deliverable E) is the on-call's reference during an incident. Exercise 05 benches against the "when speculative hurts" cases that often trigger a chapter-04 action.
