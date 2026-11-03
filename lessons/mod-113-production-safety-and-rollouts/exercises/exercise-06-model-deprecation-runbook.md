# Exercise 06 — Model-Deprecation Runbook

**Time:** 4 hours · **Output:** one complete deprecation runbook — candidate report, announcement, migration guide, deprecation-warning emission, decommission timeline — for a model currently served on your fleet (or a plausible stand-in), with a simulated consumer-peer review and a published artefact.

## Goal

Make [Chapter 06](../06-model-deprecation-runbook.md) concrete. Pick a model, author the complete deprecation runbook, socialise the migration guide with a (real or simulated) consumer-peer review, and publish the artefact so the actual deprecation can proceed on the published schedule.

## Problem statement

You have one or more models on your fleet that are candidates for deprecation (traffic decay, replacement available, cost / value misalignment, licence / compliance). Pick one and run the full deprecation runbook against it. If no real candidate exists, construct a plausible stand-in (a hypothetical v1 being replaced by a hypothetical v2).

## Prerequisites

- Chapter 06 of this module read; chapter 05 for the eval contract that gates the replacement.
- Chapter 02 of module 110 for the usage / cost telemetry the candidate report draws on.
- Access to (or awareness of) the three L25 consumer peer tracks — `llm-application-developer`, `rag-engineer`, `applied-ai-engineer` — and their communication channels.
- A model deployment with `X-Model-Version` emission already live (Exercise 02 should produce this).
- Access to your fleet's response-header / API-metadata surface so the deprecation signal can be emitted in practice (even if the actual signal firing is simulated for the exercise).

## Requirements

Five deliverables.

### Deliverable A — the deprecation-candidate report

Author the one-page report per chapter 06's template:

```
# Deprecation candidate: <model_name>
## Current state (serving footprint, capacity, traffic, cost, revenue attribution, consumers)
## Replacement (name, eval gate results, behaviour diff, latency / cost delta)
## Rationale (usage decay / cost / compliance / operational)
## Proposed timeline (announce, warning, throttle optional, decommission, rollback)
## Sign-off (inference, consumers, product, compliance if relevant)
```

Fill with real numbers from your fleet's telemetry where possible; use realistic hypothetical numbers if the model is a stand-in. Justify every number.

### Deliverable B — the grace-period calibration

Explicitly justify the grace period in writing:

- Longest consumer release cadence applicable → floor on the grace period.
- Behavioural-drift magnitude between deprecated and replacement → modifier.
- Traffic volume and blast radius of migration mistakes → modifier.
- Model maturity (experimental, production, flagship) → category-based range.
- Industry norm cross-check (OpenAI / Anthropic / Google deprecation policies as references).

Output: a specific number (e.g. "90 days") with a 1-paragraph justification.

### Deliverable C — the migration guide

Written *for* the consumer peers. Three sections per chapter 06:

1. **What's different between the deprecating model and the replacement** — API, behaviour, cost, latency. Quantitative where possible (eval-score deltas, TTFT delta, $/token delta).
2. **How to migrate** — code-change (one-line SDK model-name swap is typical), prompt / behaviour tuning (link to application-eval peer's regression report if applicable), testing (how to run the consumer's own evals against the replacement), routing (feature-flag pattern).
3. **Operational timeline** — when the deprecation warning starts; when (optional) throttling applies; when decommission happens; what happens after decommission to callers still using the deprecated model.

Include a code sample for the SDK-level migration, pinned to the OpenAI Python SDK (or your fleet's actual client).

### Deliverable D — the deprecation-warning emission

Implement (or demonstrate how you would implement) the deprecation signal:

- **Response header** — `X-Model-Deprecation: <model_name>; decommission=<date>; replacement=<model_name>; guide=<url>`. Add to the server-side response middleware; verify it's set on every response for the deprecated model before first-byte.
- **Response-body metadata** — if your API shape has a metadata field, include a `deprecation` object.
- **`GET /v1/models/{model_name}`** — ensure the endpoint returns a `deprecation` object for the model.
- **Server-side counter** — emit `deprecation_warning_responses_total{model_name}` so adoption progress is visible on a dashboard.

Include a dashboard panel showing the counter — the adoption-progress panel consumers' leads will watch.

### Deliverable E — the simulated consumer-peer review

Walk the migration guide through at least one (real or simulated) consumer-peer reviewer. Record:

- What was clear.
- What was missing (common: specific code examples, specific prompt-pattern breakage, specific cost impact).
- What was wrong (common: assumed consumer release cadence is faster than reality).
- Revisions made.

If simulated, play the role of a consumer-peer by reading the three L25 tracks' curricula and consider what that persona would ask. The more concretely the simulated reviewer is realised, the more real the feedback.

## Starter guidance

- **Pick a real candidate if possible.** The exercise is more valuable when the result is a document that will actually trigger a deprecation. If your fleet has a legacy model at low utilisation, use it.
- **Measure the capacity held by the candidate.** Chapter 06's point is capacity-reclaim; the report should name a specific GPU-hour number. If your fleet doesn't measure this, chapter 04 of module 110 explains how.
- **The grace period is a product decision.** Don't pick it alone; the consumer leads should be consulted or at minimum simulated-consulted. 90 days feels long to the inference engineer; it is often short to the consumer teams.
- **Deprecation warning is non-breaking.** Consumers who ignore the header continue to function. The warning is a *notice*, not a 400 error. Verify this in your implementation.
- **The rollback window matters.** Chapter 06 names a week of drain-but-alive after decommission. If your fleet can't afford a week of retained capacity, document the trade-off.
- **Communicate on a schedule.** Reminders at T+14, T+30, T+75 (week-ahead), T+88 (48-hour reminder) are in the chapter 06 template. Don't skip these.

## Fallbacks for constrained environments

- **No real candidate:** use a plausible hypothetical (deprecating a 7B in favour of an 8B) and document "this is a dry-run; artefact is reusable."
- **No consumer peers to review:** read the L25 track curricula and simulate; note in Deliverable E that reviews were simulated.
- **Can't actually emit headers:** write the server-side middleware snippet and verify it runs; demonstrate the response shape against a test endpoint.
- **No dashboard / counter:** describe the panel you would build; include the PromQL query and sketch the layout.

## Acceptance criteria

1. **Deliverable A** is a one-page candidate report with real or plausible numbers, justified rationale, and sign-off placeholders.
2. **Deliverable B** names a specific grace period with a defended justification.
3. **Deliverable C** is a consumer-ready migration guide with API-level, behavioural, cost, and latency diffs, plus an SDK-level migration code sample.
4. **Deliverable D** implements or demonstrates the three deprecation signals (header, body metadata, models endpoint) plus the server-side counter.
5. **Deliverable E** has a reviewer's feedback captured and revisions made.
6. The artefact set is publishable — tomorrow you could announce T-0 against this material.

## Common failure modes

- **Silent deprecation.** Skipping the announcement and just decommissioning. Fix: announcement is step one; without it, do not proceed.
- **Grace period too short.** Consumers can't migrate; features break; emergency extension needed. Fix: calibrate against longest consumer cadence.
- **Deprecation header invisible.** Consumer SDK doesn't surface headers; nobody sees the warning. Fix: dual-mechanism (header + body + models endpoint).
- **Rollback window skipped.** Decommission on T+90, capacity returned at T+90; a consumer surfaces a blocker at T+91. Fix: retain drained-but-alive for a week.
- **Capacity reclaim forgotten.** The replicas scale to zero but GPU budget stays on the team's ledger. Fix: explicit reclaim step at T+hard-decommission.

## Keep it

The candidate report, the migration guide, the header implementation, the dashboard panel definition, and the consumer-peer review notes. Together they form a publishable artefact: the actual deprecation can proceed on this schedule. The pattern is reusable — the next model deprecation uses this as the template.
