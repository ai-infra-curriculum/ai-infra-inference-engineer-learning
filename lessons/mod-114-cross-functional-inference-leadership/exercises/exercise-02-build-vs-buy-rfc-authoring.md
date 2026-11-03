# Exercise 02 — Build-vs-Buy RFC Authoring

**Time:** 6 hours · **Output:** one complete inference-platform RFC (8–15 pages) with workload decomposition, full TCO models for each option, a risk register, a migration path, triggers that reverse the decision, and sponsor sign-off (simulated acceptable).

## Goal

Make [Chapter 02](../02-build-vs-buy-rfc.md) concrete. Author a real, defensible build-vs-buy RFC for a plausible decision facing your fleet. The exercise is complete when a sponsor could sign the document and a successor engineer could re-open it when a trigger fires.

## Problem statement

Pick a plausible inference-platform decision your fleet faces. The archetype candidates:

- Move a specific workload class (long-tail embedding, transcription, free-tier chat) from self-host to a managed provider.
- Move a workload class from a managed provider back to self-host.
- Stand up a greenfield hybrid platform from scratch.
- Decide whether to migrate an existing self-host platform from one hardware class to another (H100 → H200, add Blackwell).
- Decide whether to add a secondary vendor as failover for an existing self-host or vendor path.

If your fleet is simulated, pick a realistic hypothetical and state the assumptions clearly in the RFC's "Current state" section. Reality of the numbers matters; realism of the assumptions matters more.

## Prerequisites

- Chapter 02 read; chapter 03 skimmed (the vendor scorecard it references; this exercise doesn't require completing exercise 03 first but benefits from thinking about scorecard criteria in parallel).
- Module 110 chapter 04 (cost per 1M tokens) completed or available — this exercise reuses its arithmetic.
- Access to at least one managed-inference provider's pricing page and SLA documentation (public).
- A rough fleet model: GPU count, SKU, utilisation estimate, token forecast per workload class.

## Requirements

Ten sections (the RFC's chapter 02 structure, plus an appendix). Each section is a required deliverable.

### Deliverable A — problem statement

One paragraph. The forcing function: cost pressure, SLO pressure, capacity pressure, product requirement, strategic direction. Name the concrete trigger; vague "we should consider this" is not a forcing function.

### Deliverable B — current state

What the platform looks like today. Engines (vLLM / SGLang / TRT-LLM / Triton / Dynamo); hardware; routing; SLOs per workload class; cost per workload class (from module 110 chapter 04 if available, else a defensible estimate). Include the current `$ / 1M tokens` number for each workload class.

If simulated, state the simulation parameters:

```
fleet: 40 H100-SXM5 nodes × 8 GPUs = 320 H100
models: Llama-3.1-70B-Instruct FP16 (primary), [list others]
routing: LiteLLM-shape in-house router
average utilisation: 55 %
peak utilisation: 85 %
token volume: X billion / month per workload class
```

### Deliverable C — options under consideration

For each of self-host, managed inference (named vendors — at least two for serious consideration), and hybrid variants, one paragraph describing the option plus a two-sentence "what wins here" summary.

### Deliverable D — decomposition

The matrix from chapter 02: workload class × tenant tier × model class × region × reliability tier (use the axes that matter to your decision; it's valid to only use two or three). State which cells are *in scope* for the RFC's decision.

### Deliverable E — TCO model (the arithmetic)

For *each option*, a line-item TCO spreadsheet or `cost-model.yaml` + a one-paragraph summary per option. Required line items:

For self-host:

- hardware amortised (from module 110 chapter 04; explicit assumptions: on-demand / reserved / spot mix)
- underutilised capacity
- warm pool (module 109)
- model weights storage
- observability
- egress
- software licences (if any)
- people cost (name whether included or separately tracked)
- on-call overhead
- opportunity cost (note; harder to quantify)
- exit cost

For managed inference:

- usage billed (per-vendor rate × forecast traffic)
- overage buffer (the forecast-error allowance)
- minimum commits (if any)
- data transit
- observability integration
- security and compliance overhead
- integration people cost
- incident ownership premium (support-tier fees)
- lock-in discount erosion (model the renewal explicitly)
- exit cost

For hybrid:

- Sum of per-cell TCOs (named by cell)
- Router cost
- Cross-option observability
- Failover rehearsal cost

Produce a per-option annual TCO number for the horizon of the RFC (typically 12 months). Show arithmetic; the appendix carries the spreadsheet.

### Deliverable F — risk register

For *each option*, a scored risk register with columns: `risk | likelihood | impact | owner | mitigation`. At least six risks per option. Cover the representative classes from chapter 02 (capacity, people, hardware, software supply chain, driver/firmware for self-host; SLA, price, feature parity, data boundary, exit cost, lock-in for managed; split discipline, observability gap, failover drift for hybrid).

Likelihood: low / medium / high. Impact: quantified where possible (dollars, hours of downtime, number of consumer teams affected); qualitative otherwise.

### Deliverable G — recommendation

The specific placement for each cell of the decomposition. Not a one-liner; a decomposed verdict. Include the paragraph-level structure from chapter 02's worked structure example:

- Recommendation (specific placements).
- TCO delta (appendix pointer).
- Decision reverses if (triggers).

### Deliverable H — migration path

Using chapter 01's eight-field migration-plan shape:

- Problem, scope, target state, peer touch points, stages, rollback branch, consumer communication plan, evidence of done.

Explicitly name where in the stage sequence rollback becomes expensive (becomes a second migration). Explicitly name the consumer-track communication.

### Deliverable I — open questions

What the RFC does *not* resolve. Who answers each; by when.

### Deliverable J — sign-off

A signature block for platform lead, sponsor, peer-track leads whose work the decision touches, finance partner. Simulate the signatures if peers are unavailable but name the roles explicitly and the dates of promised sign-off.

### Appendix

The TCO spreadsheet or `cost-model.yaml`. Supporting exhibits for the current-state numbers (benchmark data, utilisation measurements). If the RFC references a vendor scorecard, the scorecard pointer (exercise 03 produces it).

## Starter guidance

- **Numbers over adjectives.** "Fireworks is cheaper" is a vibe; "Fireworks is $0.90 / 1M output at current forecast vs. our self-host TCO of $2.40 / 1M output; $300k / year delta at current volume" is an RFC.
- **Price the renewal.** Promotional year-1 pricing that rises at renewal is the most-missed TCO line item. Model it explicitly.
- **Price the exit.** Even a great vendor decision needs an exit cost estimate. If you don't know, say "unknown; recommend rehearsed exit by Q[X]."
- **Decompose honestly.** "All-or-nothing self-host" is almost never the real recommendation. Force yourself to look at per-cell placements.
- **Risk register's job is to make the decision defensible.** A risk register of three bland risks is not defensible. Push to at least six meaningful risks per option.
- **Peer sign-off is an actual gate.** Even if simulated, name the peers and what they'd want to see. If you cannot answer "would the ML-platform peer sign this today," the migration-path section is weak.

## Fallbacks for constrained environments

- **No real cost data:** use publicly available on-demand GPU prices from the major clouds plus publicly-posted vendor pricing to assemble a defensible synthetic model; be explicit that the model is synthetic.
- **No access to a real vendor contract:** read the vendor's public terms-of-service and pricing page; model the TCO conservatively (treat ambiguity as cost, not savings); mark `<!-- needs-research: ... -->` on any clause that would need legal review in reality.
- **No fleet of your own:** pick a plausible hypothetical and state its parameters in the current-state section.

## Acceptance criteria

1. **Deliverable A** has a concrete, named forcing function; no "we should think about" phrasing.
2. **Deliverable B** has current-state numbers (real or clearly-labelled synthetic) including `$ / 1M tokens` per workload class.
3. **Deliverable C** evaluates at least two named managed-inference vendors plus self-host plus at least one hybrid variant.
4. **Deliverable D** has a genuine decomposition (at least two axes); some cells are explicitly out of scope with reasons.
5. **Deliverable E** has line-item TCO for each option; the renewal risk is priced for managed; the hybrid overhead is priced.
6. **Deliverable F** has at least six risks per option, scored with likelihood / impact / owner / mitigation.
7. **Deliverable G** gives a per-cell recommendation, not a verdict.
8. **Deliverable H** applies the chapter-01 migration-plan shape and names the rollback-expensive stage.
9. **Deliverable I** names 2–5 open questions with owners.
10. **Deliverable J** has signature blocks for the required roles; signatures either real or clearly marked simulated.
11. The appendix has the TCO spreadsheet / YAML and the supporting exhibits.

## Common failure modes

- **The unit-cost shootout.** The RFC compares `$ / 1M tokens` on one axis and picks the lower. Price the rest of the TCO.
- **The strategic abstract.** The RFC argues conviction without numbers. Add arithmetic.
- **The binary pick.** No decomposition. Force at least one cell placement differently from the overall verdict.
- **Exit-cost zero.** Vendor exit priced as zero because nobody asked. Price it.
- **Wishful SLA.** Vendor's marketing uptime copied as fact. Check historical status; name credit schedule.
- **No triggers.** "Our recommendation is X" with no "we revisit if Y." The RFC is already stale.

## Keep it

The RFC. The TCO spreadsheet / YAML. The risk register. The migration path. Exercise 03 of this module uses the RFC's managed-inference options as the vendor-eval targets. Exercise 05 uses the peer-touch-points list as the starting point for the cross-peer contracts. If you ever face a real build-vs-buy decision on this fleet, this RFC is the template.
