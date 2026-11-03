# Exercise 05 — Eval-Gate Contract with Peer Tracks

**Time:** 4 hours · **Output:** one signed eval-gate contract between the inference engineer and the two upstream eval peers (real or simulated), negotiated against two or more candidate model promotions, with a clear dispute-resolution path and a quarterly review date on the calendar.

## Goal

Make [Chapter 05](../05-eval-gate-contract.md) concrete. Author the contract as a real document the three roles would sign; negotiate it against at least two concrete promotion scenarios to pressure-test the thresholds; publish it with a schedule for review.

## Problem statement

Your fleet currently promotes models via ad-hoc hand-offs: the eval peers run some benchmarks, slack-message the result, and the inference engineer reads the message and decides. This is fine until it fails — the failure is the chapter 05 silent-drift case where nobody owns the specific regression and nobody signs the specific threshold. Fix it by writing the contract.

The contract is a real document, co-authored (or simulated co-authored if the peer roles aren't available in your organisation).

## Prerequisites

- Chapter 05 of this module read; chapters 01, 04, and 06 for the cross-references.
- Chapter 02 of module 110 (SLO authoring) as the source for the SLO gate thresholds the contract will reference.
- Access to the peer tracks' documentation: [`ai-eval-engineer-learning`](https://github.com/ai-engineering-curriculum/ai-eval-engineer-learning) and [`model-evaluation-engineer-learning`](https://github.com/ml-engineering-curriculum/model-evaluation-engineer-learning) — review their scope so the contract reflects what they actually produce.
- At least two candidate promotion scenarios to use as test cases.

## Requirements

Four deliverables.

### Deliverable A — the authored contract

Use the chapter 05 template literally. Fill in:

- **Signatories** — real names if you have the peers, placeholder role labels if simulated.
- **Benchmark gate** — specific benchmarks owned by `model-evaluation-engineer`; tolerance thresholds justified; SLA on verdict.
- **Application eval gate** — specific eval sets owned by `ai-eval-engineer`; tolerance thresholds justified; SLA on verdict.
- **Shadow-rollout gate** — specific equivalence scorers from Exercise 01; thresholds justified; dwell time justified.
- **SLO gate** — specific SLO targets from module 110 chapter 02; burn-rate alerting from Exercise 01.
- **Gate semantics table** — what BLOCKs, what triggers AUTO-ROLLBACK, what DOWNGRADEs, what degrades gracefully.
- **Graceful degradation protocol** — who decides on a feature-flag routing path when a minor eval regression surfaces.
- **Post-incident fixture back-feed** — SLA for merging fixtures, routing table of who owns which kind of fixture.
- **Review cadence** — quarterly review date on the calendar.

### Deliverable B — threshold defence

For each threshold in the contract, write 2–3 sentences in a separate "threshold rationale" appendix:

- Why *this* benchmark and not another?
- Why *this* tolerance and not stricter / looser?
- What's the signal-to-noise on the benchmark? Would a 1 % regression here be noise or real?
- What's the consumer-visible consequence of a tolerance miss?

If you don't know, write "to be calibrated after N months of data" explicitly. An unknown is better than a false precision.

### Deliverable C — the negotiation drill (two scenarios)

Pressure-test the contract against at least two concrete promotion scenarios. Write up each:

**Scenario 1 — a clean promotion.** Candidate passes benchmark gate (+0.5 %), passes application gate (within tolerance), passes shadow-rollout equivalence (cosine 0.97 vs. primary), passes SLO gate under canary. Walk through the contract step by step; confirm it promotes; note any step where the contract was ambiguous.

**Scenario 2 — a mixed-signal promotion.** Candidate passes benchmark gate but regresses 1.5 % on one specific application eval (within gate tolerance of 2 %). What happens? Does it promote? Who decides? Walk the contract; identify the specific clauses that apply; confirm the decision is clean or document what needs to be added to the contract to make it clean.

Optional third scenario: a candidate that fails the benchmark gate but that upstream really wants to ship because cost drops 40 %. What is the escape path? Walk the contract.

### Deliverable D — dispute-resolution path

Explicitly state what happens when the three signatories disagree:

- Inference engineer wants to ship; both eval peers say no — what overrides?
- Eval peers want to ship; inference engineer sees SLO burn risk — what overrides?
- Product partner overrides the eval peers' no — under what policy? Is there a product-manager veto, a shared decision, or a formal escalation?

The default per chapter 05 is: "eval peers block, no override." But many organisations deviate; document your org's actual escalation path.

## Starter guidance

- **Co-author if you can.** Walk the draft contract through a 30-minute conversation with an `ai-eval-engineer` and a `model-evaluation-engineer` (even if those people wear different formal titles in your org — the roles exist somewhere). The conversation reveals what thresholds are negotiable and what are load-bearing.
- **Simulate the roles if you can't.** Read the peer tracks' curricula; adopt the voice of each role when walking the contract alone; the role-play catches what cross-role dialogue would.
- **Don't invent thresholds.** If you don't know the right cosine-similarity threshold for your workload, say so and plan to calibrate; writing "0.94" because it sounds plausible is worse than writing "to be calibrated."
- **The graceful-degradation clause is where most contracts fail.** Most eval regressions are not catastrophic; they're small. The contract has to say *clearly* what the team does with a small regression — route a feature flag, extend shadow, or block. Make this decision at contract-time, not at-regression-time.
- **Review cadence is calendar-backed.** Put the next review date on the actual calendar. The contract will drift otherwise.

## Fallbacks for constrained environments

- **No peer roles to sign:** complete the contract as a sole author, document the roles the contract assumes, and note the limitation. When real peers are available, this document is the opening draft of the conversation.
- **No benchmark infrastructure:** describe which benchmarks the contract would cover if you had them; name the ownership and threshold shape; this is the forward-plan artefact.
- **No historical eval data for calibration:** use "to be calibrated after N months" for thresholds; do not invent precision you don't have.

## Acceptance criteria

1. **Deliverable A** uses the chapter 05 template literally; every section is filled.
2. **Deliverable B** defends every threshold with 2–3 sentences; unknowns are stated as unknowns, not invented.
3. **Deliverable C** walks the contract against two scenarios; each scenario identifies clauses that applied, clauses that were ambiguous, and any contract additions needed.
4. **Deliverable D** names the dispute-resolution path; the policy is a named policy, not implicit.
5. The contract has signatories (real or simulated) and a dated review cadence.
6. The contract integrates with Exercise 01's canary harness — the gate verdicts the harness checks are the ones named in this contract.

## Common failure modes

- **Threshold drift.** Thresholds for a 7B model used for a 70B model. Fix: review cadence catches this when model class changes.
- **Overlap without ownership.** Both eval peers "own" refusal rate; neither signs off. Fix: primary and secondary explicit.
- **Gate ceremony.** The gate passes because no one actually looks at the output. Fix: the ramp controller (Exercise 01) refuses to start until the gate dashboard reads green.
- **Fixture back-feed deferred.** Post-mortem action never merged. Fix: SLA named in contract; module-114 cadence audit.
- **Ambiguous override path.** Who gets the final say when the three roles disagree? Fix: Deliverable D.

## Keep it

The contract is a durable document. It lives in version control, has a review cadence, has signatories. It is referenced by Exercise 01's canary harness (as the pre-flight gate), by Exercise 04's post-mortems (as the fixture-back-feed target), and by Exercise 06's deprecation runbook (as the sign-off list).
