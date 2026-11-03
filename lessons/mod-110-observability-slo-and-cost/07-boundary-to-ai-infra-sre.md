# Chapter 07 — Boundary to `ai-infra-sre`: SRE Methodology at L30, Inference Specialisation at L35

> The SRE methodology — SLI / SLO / error budget / burn-rate alerts / incident command / postmortems — is one body of knowledge, taught at the L30 `ai-infra-sre` track. This module did not re-teach it; it *adopted* it for inference-shaped signals. The shared boundary is "SRE at L30, inference-specialised at L35." This chapter formalises the mapping, the hand-off cases, and the three common cross-boundary mistakes.

## Motivation

Six real cross-boundary situations an inference engineer runs into:

1. "What is a good burn-rate multiplier for a 30-day SLO window?"
2. "Our chat TTFT P95 is drifting; what alert pack catches it fast enough to act?"
3. "A production incident paged four teams; who is commander?"
4. "A GPU in our fleet is throwing XID 79 errors; what is the on-call protocol?"
5. "Our incident postmortem template asks for 'affected users'; how do I count users when the service is a token stream?"
6. "My on-call engineer missed a page because the burn-rate alert fired below the pager threshold; how do I design multi-window alerting?"

Three of those are *SRE methodology* questions (1, 3, 6) — the answers are at [`ai-infra-sre-learning`](https://github.com/ai-infra-curriculum/ai-infra-sre-learning) (L30). Two are *inference-specific* (2, 5) — the answers are in this module. One is *shared* (4) — the GPU-specific hardware event belongs here, but the on-call ritual belongs to the SRE track.

By the end of this chapter you can:

- Decide, for any ambiguous situation, whether it is an SRE-methodology question or an inference-specialisation question.
- Point your SRE partner at the methodology they should teach the broader org, not the inference extensions you added on top.
- Avoid the three common cross-boundary mistakes.
- Hand off cleanly in both directions.

## The mapping table

| Topic | L30 (`ai-infra-sre`) | This module (L35) |
|---|---|---|
| **SLI definition — general** | yes | — |
| **SLI definition — token-weighted inference SLIs (chapter 02)** | — | yes |
| **SLO target negotiation — general** | yes | — |
| **SLO target negotiation — per-workload-class latency targets (chapter 02)** | — | yes |
| **Error budget math — general** | yes | — |
| **Per-token error budget (chapter 02)** | — | yes |
| **Burn-rate alert design — multi-window multi-burn-rate pattern** | yes | — |
| **Burn-rate alerts applied to TTFT / TPOT / cache / cost (chapter 02)** | — | yes |
| **Prometheus / Grafana base install** | shared (also at `ai-infra-engineer` L20) | — |
| **DCGM-exporter wiring, engine-metrics scraping (chapter 03)** | — | yes |
| **OTel span layout for prefill-then-decode request (chapter 03)** | — | yes |
| **Log / trace sampling at QPS (chapter 03)** | general (sampling patterns) | inference-specific budget math |
| **Cost SLO methodology** | yes (cost as an SLO is general) | yes (cost-per-1M-token model with credits) |
| **Business-metric tie-in — translating technical to business** | general (SRE as a product practice) | inference-specific memo shape |
| **Incident command — roles (commander, scribe, communicator)** | yes | — |
| **Incident command — GPU-specific pages (XID, ECC DBE)** | shared | triggers from here |
| **Postmortem discipline — template, framing, retrospective** | yes | — |
| **Postmortem of an inference incident — token-seconds impact, SLO burn, business dollar impact** | — | yes (uses the chapter 04 cost model and chapter 06 KPI tie-in) |
| **On-call rotation — rosters, escalation, hand-off** | yes | — |
| **Runbook writing — general shape** | yes | — |
| **Inference-specific runbooks (TTFT regression, cache cliff, preemption spike, NCCL hang)** | — | yes (also in module 113) |

Rule of thumb: *the SRE track teaches "how SRE works." This module teaches "what SRE methodology looks like when the signal is tokens, the GPU is the cost, and the request is a stream."*

## Ambiguous cases — how to decide

Four ambiguous situations and the correct hand-off:

### Case 1 — a cross-team incident

A chat latency regression correlates with a release in the API gateway (owned by another team), a KV allocator change (owned by the inference team), and a cloud-provider incident (owned by the SRE team). Who is commander?

- Incident command is an **SRE-methodology** question — the commander-vs-scribe role split, the escalation protocol, the communication cadence. Go to L30.
- The *technical investigation* on the inference side is this module's domain — chapter 01's axis-set bisection, chapter 03's dashboards, chapter 05's Pareto context.

The hand-off: the SRE track owns the ritual; the inference engineer brings the *evidence* (axis-set decomposition, SLO burn status, cost impact) into the ritual.

### Case 2 — a new SLO proposal

Product asks for a new SLO on "agent completion time." Who owns the design?

- The **shape of the SLO document** — SLI definition, target, window, error budget, alert pack — is an SRE-methodology question. The chapter-02 template in this module follows the SRE methodology, not the other way around.
- The **inference-specific content** — what SLIs exist for an agent workload, what targets are realistic given the physics, how to burn-rate alert on a composite signal — is this module.

The hand-off: the SRE partner reviews the document for methodology ("you used P99 for a 500-samples-per-day class; that's noisy"); the inference engineer produces the content.

### Case 3 — a bad GPU

A GPU is throwing XID 79 errors (chapter 01's hardware-event axis). Who diagnoses and who pages?

- The **inference-side signal** — the DCGM metric, the engine's correlation (did the replica crash, did a sequence get corrupted) — is in this module (chapter 01). The hardware fault page is triggered here.
- The **on-call ritual** — how the page is answered, when to escalate to NVIDIA support, how to drain the replica, when to replace the hardware — is SRE and infrastructure methodology. Module 109's drain-and-replace is adjacent; module 113's runbook is the specific instance.

The hand-off: this module's chapter 01 dashboards detect the fault; module 113's chapter on inference-specific runbooks writes the runbook; the SRE on-call ritual answers the page.

### Case 4 — a broken cost dashboard

The cost-per-1M-token number drops 30 % overnight on the dashboard. Who investigates?

- A **real** cost drop is an inference-engineering question — chapter 04 attribution says the credit model drifted, or a release landed, or workload mix changed. Inference engineer.
- A **fake** cost drop (dashboard bug, cost-model config error, metric relabel failure) is a chapter-03 plumbing question. Inference engineer.
- A **cost SLO breach** (whether real or fake) is an SRE methodology question about how to alert, who to escalate, when to disable the feature. Shared.

The hand-off: this module investigates the signal; the SRE partner reviews whether the alert pack caught it correctly.

## The hand-off bundle — inference → SRE

When you need the SRE track's methodology for an inference-shaped problem, the hand-off bundle:

```
## What I need help with
{specific SRE-methodology question: burn-rate design / incident command / postmortem shape}

## Relevant inference context
- Axis set: {chapter 01 reference}
- SLO document: {link}
- Dashboards: {links to fleet / per-class / per-replica / cost}
- Current error-budget status: {percent consumed, window, burn rate}
- Any active alerts: {names, when fired}

## What I've tried
{inference-engineer side work already done}

## The question for the SRE partner
{concrete, SRE-methodology-shaped question; not "fix my inference problem"}
```

Three disciplines:

- **Ask an SRE-shaped question.** Not "help me debug this latency issue" but "help me design the burn-rate alert pack for this SLO."
- **Bring the inference evidence.** The SRE partner doesn't know your fleet's axis set; your handoff teaches them enough to help.
- **Capture the answer in this module's template.** When the SRE partner answers, write the answer into your chapter-02 template so the next time you don't need to ask.

## The hand-off bundle — SRE → inference

When an SRE asks the inference team to produce an inference-shaped signal for a general SRE workflow (a company-wide SLO review, a cross-platform incident retro), the bundle in the other direction:

```
## What's needed
{SRE need: "inference's token-weighted availability for the all-hands SLO review"}

## What the inference team provides
- SLI definition in SRE-language: {chapter 02 reference}
- Dashboard URL that renders it: {chapter 03 reference}
- Error-budget status: {percent consumed, window, burn rate}
- Known caveats: {chapter 02 and 04 caveats — token-weight vs. request-weight, idle-time credit, class slicing}

## Narrative for the SRE review
"Chat availability this week was 99.94 % token-weighted vs. the 99.9 % SLO. Primary burn: 
 a 10-minute XID-79 incident on replica <X> at <time> cost <N> token-seconds of error budget.
 Review'd incident retro: {link}."
```

The inference engineer translates the token-shaped data into the SRE's RED-metric-shaped language without hiding the inference specifics. "Availability 99.94 %" is intelligible to any SRE; "token-weighted" is the specific that an SRE reviewing inference should know.

## The three common cross-boundary mistakes

### Mistake 1 — re-inventing SRE methodology

Symptom: an inference team invents a new burn-rate formula from scratch because they didn't know the SRE workbook existed.

Cost: hours of work on something already done well elsewhere, usually arriving at a worse answer than the workbook's recipe.

Fix: before authoring a methodology primitive, check the SRE track and the SRE workbook. If the answer is there, cite it and move on.

### Mistake 2 — treating inference as a generic service

Symptom: an SRE reviews an inference SLO document and treats it as a generic service — "P99 request latency < 2s" — without the chapter-02 per-class decomposition or token weighting.

Cost: the SLO masks the regressions the inference team can actually act on (chapter 02 motivation example A).

Fix: the SRE partner is reviewing for methodology; the inference engineer is responsible for the content. If the SRE suggests "use P99 request latency," the inference engineer argues for the chapter-02 shape with evidence.

### Mistake 3 — unclear incident command

Symptom: an inference incident has three engineers investigating three different axes in parallel and no one running the ritual. The incident goes long because the comms are unclear.

Cost: longer incident, worse postmortem, lower user trust.

Fix: the SRE track teaches incident command. If your team doesn't have a designated commander ritual, that is the SRE partner's call to action — not an inference-engineer-written process. Chapter 113's runbook work in the inference module names the inference-specific runbook content; the ritual on top is SRE-owned.

## Shared responsibilities — not everything can be split

Some concerns are genuinely shared. The right pattern is a joint document, not a split one:

- **Postmortem of an inference incident.** The postmortem *template* is SRE-owned. The *content* on the inference side — token-seconds of SLO burn, cost impact in dollars, root cause in the axis-set vocabulary — is inference-owned. Co-author.
- **SLO review ritual.** Weekly cross-functional review that includes SRE, inference, product, finance (chapter 06 ritual). The SRE partner owns the methodology facilitation; the inference engineer owns the content presentation.
- **On-call rotation for the inference service.** The rotation design is SRE; the runbook content is inference; the paging thresholds are co-authored.

The principle: methodology is L30, content is L35, *co-authored artifacts have both names on them*.

## When the SRE track lands

At time of writing, the L30 `ai-infra-sre` track is a sibling under the `ai-infra-curriculum` organisation; its curriculum may be in development. If a specific L30 chapter is referenced from this module and the sibling chapter does not yet exist, the convention is to:

- Mark the reference with a `<!-- needs-research: L30 sibling chapter pending; verify when the track lands -->`.
- Give the inference engineer enough methodology inline to proceed without the sibling.
- Promote the sibling reference and remove the inline methodology when the sibling chapter lands.

<!-- needs-research: verify the exact chapter mapping to `ai-infra-sre-learning` once that track's curriculum is published. The references in this module to "the SRE track's chapter on SLO methodology" etc. should be updated to point at specific chapter URLs. -->

## Summary

- SRE methodology (SLI, SLO, error budget, burn-rate alert, incident command, postmortem) lives at `ai-infra-sre` (L30). Don't re-teach it; adopt it.
- Inference specialisation (axis set, token-weighted SLI, per-workload-class targets, DCGM / NCCL / engine metrics, cost model with credits, Pareto reasoning, business-metric memo) lives in this module (L35). Teach it in SRE's vocabulary.
- Ambiguous cases map cleanly if you ask "is this a methodology question or an inference-content question?"
- Hand-off bundles go both directions. The inference engineer translates inference-shape to SRE-shape when asked; the SRE partner reviews inference SLOs for methodology.
- Three common cross-boundary mistakes: re-inventing methodology, treating inference as a generic service, unclear incident command. All three have the same root — unclear ownership — and the same fix — explicit boundary.
- Co-authored artifacts (postmortems, SLO reviews, runbooks) have both names on them; neither owner tries to write the whole thing alone.
