# Exercise 05 — Cross-Peer Interface Contracts

**Time:** 5 hours · **Output:** four signed interface contracts (ML-platform peer, two eval peers, SRE peer, security peer) using the chapter 05 five-part shape, plus a cross-peer cadence that covers them weekly and quarterly.

## Goal

Make [Chapter 05](../05-cross-peer-interfaces.md) concrete. Author the written contracts with each of the four remaining peer tracks, using the same discipline exercise 04 used for the performance peer. The exercise is complete when all five peer interfaces for this platform are written, signed (dry or real), and plugged into the cadence from exercise 01.

## Problem statement

Chapter 05 generalises the chapter 04 interface pattern across four peers:

- **ML-platform peer (L30, AI Infrastructure family)** — model handoff, registry, metadata, lineage, signing, deprecation.
- **Eval peers (L30 × 2 — `model-evaluation-engineer` in ML Engineering, `ai-eval-engineer` in AI Engineering)** — benchmark and application evals, respectively.
- **SRE peer (L30, AI Infrastructure family)** — methodology primitives, incident command for availability incidents.
- **Security peer (L35, AI Infrastructure family)** — inference-edge security, detectors, weight signing, rate-limit tiers, red-team evidence.

Author one contract per peer (treat the two eval peers as a shared contract with two signers, following chapter 05's framing). The five-part contract shape, applied once per peer, is the deliverable.

## Prerequisites

- Chapter 05 read; chapter 04 completed (its contract pattern is the template).
- Module 110 chapter 07 read (SRE-peer boundary at the module level).
- Module 113 chapter 05 completed (eval-gate contract at the module level — this exercise situates it in the program).
- Exercise 01 completed (the cadence into which these contracts plug).
- Exercise 04 completed (the performance-peer contract — same shape, one peer).

## Requirements

Six deliverables.

### Deliverable A — ML-platform peer contract

A document with the five sections:

1. **Boundary statement.** What ML-platform owns (registry, lineage, training-to-serving handoff). What this role owns (production operation from artefact ingress to response egress). The 20 % shared (artefact format and metadata; promotion decision; deprecation decision).
2. **Mutual deliverables.** The deliverables table from chapter 05 — model artefact, model metadata, quantisation artefact, serving configuration, deploy-event feedback, deprecation notice, model-regression evidence. For each: direction, format, when.
3. **Ticket template.** The "new model version, please promote" ticket template with all required fields.
4. **Response SLA.** P0–P2 table with the ML-platform peer's SLAs.
5. **Shared incident path.** How a bad-artefact incident is co-owned.

### Deliverable B — eval-peers contract

A single document with two signers (one per peer). The five sections:

1. **Boundary statement.** What the eval peers each own (model-evaluation: capability / benchmark evals; ai-eval: application / rubric evals). What this role owns (the gates that consume the eval evidence). The 20 % shared (threshold setting; fixture back-feed; live-eval regression ownership).
2. **Mutual deliverables.** Benchmark eval results; application eval results; pre-flight eval sign-off; live-eval regression alert; fixture back-feed; the module 113 chapter 05 eval-gate contract as the anchor artefact.
3. **Ticket template.** The "pre-flight eval request" ticket template.
4. **Response SLA.** P0–P2 table.
5. **Shared incident path.** Live-eval regression on production as the archetype.

Note that module 113 chapter 05 already defines the detailed eval-gate contract; this deliverable *situates* that contract in the program, with program-scope boundary, SLA, and cadence additions.

### Deliverable C — SRE peer contract

Chapter 05 structure. Boundary statement specifies methodology primitives in, inference-specific application out. The 20 % shared is availability-incident command, SLO contract review, on-call rotation design. Deliverables include the SLO contract and burn-rate alert definitions (this role → SRE), the incident-command and post-mortem methodology (SRE → this role), the on-call rotation plan (this role → SRE for review). Ticket template is the "SLO contract change" ticket or the "burn-rate ratio consultation" request. SLA covers P0 (availability), P1 (SLO change), P2 (methodology consultation). Shared incident path has the "which incident mode" decision in the first ten minutes.

### Deliverable D — security peer contract

Chapter 05 structure. Boundary statement specifies policy and primitives in, production operation out. The 20 % shared is policy evaluation at serving; weight-signing and supply chain; incident response for edge attacks. Deliverables include edge-security policy, prompt-injection detector, signed-weight manifest, rate-limit tier recommendations (security → this role), serving telemetry indicative of attacks (this role → security), red-team evidence per major promotion, shared edge-incident runbook. Ticket template is the "red-team evidence for model X" request or the "policy update" ticket. SLA covers P0 (active attack), P1 (red-team evidence needed), P2 (policy revision). Shared incident path has security-peer incident command with this role's mitigation operation.

### Deliverable E — the cross-peer cadence

A one-page document that plugs the four peer interfaces (plus the exercise-04 performance-peer interface) into the chapter 01 cadence:

- **Weekly SLO review rotation** — which peer attends which week (chapter 05's rotation is a reasonable default; adapt).
- **Quarterly migration-roadmap review** — all five peer leads present; what each brings (the counter-sign on their interface contract's review).
- **Ad-hoc peer sync** — when (what triggers an off-cadence sync, who schedules).
- **Contract review schedule** — each contract is re-signed on a cadence; name the dates.

### Deliverable F — the cross-peer failure-mode review

A one-page self-review addressing chapter 05's common cross-peer failure modes:

- **Implicit interface.** Any peer where the interface was unwritten before this exercise? Fix: the contract.
- **Interface written but not consulted.** Any contract from a prior iteration that is now stale? Fix: re-sign cycle.
- **No reverse direction.** Do any of your contracts only flow one way? Fix.
- **Lead-turnover blind spot.** Have any peer leads changed without re-signature? Fix.
- **Methodology-SRE conflation.** Is the SRE peer treated as "the pager peer" rather than the methodology partner? Fix.

## Starter guidance

- **Treat the eval peers as one contract with two signers.** Chapter 05 explains why; the capability eval and the application eval are genuinely different work by different people, but the gate they produce is one gate.
- **The security peer is often under-contracted.** Many teams have never written a security interface for inference specifically. Use the opportunity to make a first draft; expect iteration.
- **SRE peer contract is primarily a methodology adoption, not a deliverable exchange.** The SLO contract and the incident playbook *are* the deliverables; the artefacts cross less than they do for the other peers.
- **Signing matters.** Even if you dry-sign, name the roles and dates explicitly; a contract without signers reads as a sketch.
- **The cross-peer cadence is the chapter-01 cadence with the peer rotation added.** Don't re-invent the cadence; add the peer-attendance column to it.

## Fallbacks for constrained environments

- **No peers at all:** write each contract as if addressing a real peer; dry-sign; note in the preamble that peer involvement is a TODO.
- **Some peers present, not others:** get real signatures where possible; stand-in for the rest.
- **One-off / individual-contributor situation (no formal peers):** draft the contracts as the written form of what would be the peer interfaces if the organisation grew; the artefacts are still useful as a self-discipline and as a transition plan.

## Acceptance criteria

1. **Deliverable A** has the five sections for the ML-platform peer; the mutual-deliverables table covers at least seven deliverables; ticket template has the required fields.
2. **Deliverable B** has the five sections for the eval peers (one document, two signers); live-eval regression is clearly named as a shared incident.
3. **Deliverable C** has the five sections for the SRE peer; the "which incident mode" decision in the first ten minutes is named.
4. **Deliverable D** has the five sections for the security peer; the weight-signing protocol and the detector-in-production discipline are explicit.
5. **Deliverable E** integrates all five peer interfaces into the exercise-01 cadence with a clear weekly rotation and a quarterly review slot.
6. **Deliverable F** reviews the five cross-peer failure modes with honest assessment and proposed fixes.
7. All four contracts are signed (real or dry) with named signers and dates.
8. Each contract has a review cadence named in its own body.

## Common failure modes

- **Copy-paste contract across peers.** Each peer has a genuinely different artefact exchange and a different failure profile. Do not reuse language without specialising.
- **Ticket template without priority mapping.** Each peer's tickets map to the same P0/P1/P2 severity your org uses; don't invent per-peer severity scales.
- **SRE contract that reads as "SRE owns on-call."** The real contract is methodology. Rewrite if drifting.
- **Security contract that stops at "red-team sign-off."** The deploy-time detector in production is where most teams under-invest; cover it.
- **Cadence that lists meetings but names no outputs.** Each cadence slot has a deliverable (minutes, tickets created, contract re-signed). Name them.

## Keep it

The four contracts, the cross-peer cadence, the failure-mode review. Combined with exercise 04's performance-peer contract, this exercise 05 completes the full peer-interface portfolio for your platform. Chapter 01's quarterly cadence re-reviews all five; the contract-review schedule in Deliverable E carries that forward.
