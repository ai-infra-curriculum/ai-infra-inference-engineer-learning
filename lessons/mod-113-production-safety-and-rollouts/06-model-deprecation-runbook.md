# Chapter 06 — Model-Deprecation Runbook: Grace Period, Migration Guide, Decommission Timeline

> Every inference platform accumulates models. Not every model earns its continued capacity. Deprecation is the discipline that removes a model from the serving surface *without* breaking the consumer tracks that depend on it and *without* leaving the GPU budget held by a zombie fleet. This chapter writes the runbook.

## Motivation

Three consumer tracks at L25 in the AI Engineering family — `llm-application-developer`, `rag-engineer`, `applied-ai-engineer` — consume the inference API this role produces. When a model name they depend on is removed, their features break. When it is kept forever, the inference fleet runs replicas at low utilisation, holding GPU budget that should serve newer models. Deprecation is the controlled transition between those extremes.

Industry prior art (OpenAI, Anthropic, Google deprecation policies) converges on a shape: an announcement, a grace period, a deprecation-warning header on live responses, a migration guide, a decommission date. The *grace period length* is the design knob — too short and consumers can't migrate; too long and the fleet carries dead weight. The right length is a product decision shaped by consumer-release cadence and model-use maturity.

This chapter writes the runbook as a template you adapt per model, names the capacity-reclaim step that closes the loop, and lists the cross-track hand-offs.

By the end of this chapter you can:

- Decide whether a model is a deprecation candidate.
- Pick a grace period calibrated against consumer release cadence.
- Author a deprecation announcement, a migration guide, and a decommission timeline.
- Instrument live responses with deprecation warnings that are visible but non-breaking.
- Execute the decommission with rollback-exit points and GPU-budget reclaim.
- Avoid the five common deprecation failure modes.

## When to deprecate

A model is a candidate for deprecation when one or more of these hold:

- **Usage decayed.** Traffic below a threshold (set by the fleet's capacity economics) for a sustained window. Common signature: a model with 3 % of overall traffic held for the last three months.
- **Replacement available and preferred.** A newer version (`v2` beats `v1` on the eval gate across the board; chapter 05's contract) covers the same use cases at lower cost or higher quality.
- **Cost exceeds revenue attribution.** Cost per 1M tokens at the current utilisation (module 110 chapter 04) exceeds the model's revenue contribution; the fleet is subsidising the model.
- **Compliance / licensing.** A model's licence expires or shifts; a jurisdiction requires its removal; a security review recommends against it.
- **Operational cost.** The model's engine version is diverging from the fleet standard (needs a special image, a special CUDA toolkit, a special kernel branch) and the carrying cost on the team exceeds the model's value.

The deprecation decision is cross-role — the inference engineer proposes, the consumer peers sign off, the product partner approves. It is not unilateral.

### The deprecation candidate report

The artefact that opens a deprecation is a one-pager:

```
# Deprecation candidate: <model_name>

## Current state
- Serving footprint: <N replicas × <hardware>; capacity reserved: <X GPU-hours/day>
- Traffic (last 30 days): <requests/day>, <tokens/day>
- Workload classes: <chat, code, long-context, agent, batch — which?>
- Cost/1M tokens: <$>
- Revenue attribution: <$ or "not quantifiable">
- Known consumers: <list of teams, products, or anonymised cohorts>

## Replacement
- Proposed replacement: <model_name_v2>
- Eval gate results (chapter 05): <PASS summary with thresholds>
- Expected behaviour diff: <summary of what changes for consumers>
- Latency / cost delta: <TTFT/TPOT/$ change>

## Rationale
- Usage decay? Cost? Compliance? Operational?

## Proposed timeline
- Announce: <date>
- Deprecation-warning-in-response starts: <date>
- 10% rate-limit applied: <date (optional)>
- Decommission: <date>
- Rollback window: <date+N>

## Sign-off
- Inference engineering: <name>
- Consumer peers: <names from L25 tracks>
- Product partner: <name>
- Compliance (if relevant): <name>
```

Publish it; collect sign-offs; schedule.

## The grace period — calibrating length

The grace period is the time between *deprecation announcement* and *decommission*. The right length depends on:

- **Consumer release cadence.** A consumer on a monthly release train needs 2–3 cycles to adopt a replacement: ~60–90 days. A consumer on continuous-deploy can migrate in a sprint: ~2 weeks.
- **Behavioural-drift magnitude.** A replacement that behaves identically needs less consumer-side work; a replacement with significant behavioural drift needs longer because client-side evals and prompt adjustments are required.
- **Traffic volume.** A high-volume model's consumers need more time because migration mistakes have larger blast radius.
- **Industry norm cross-check.** Prior art ranges from 3 months (minor model updates) to 12+ months (foundational deprecations like OpenAI's `text-davinci-003`); see the resources chapter for citations.
- **Maturity.** An experimental / preview model can carry a shorter grace period than a GA model; be explicit in the announcement what SLA the model shipped under.

A common shape for a production model:

| Model use | Grace period |
|---|---|
| Experimental / preview | 30 days |
| Production, minor replacement (v1 → v1.1) | 60 days |
| Production, significant replacement (v1 → v2 with behaviour diff) | 90–180 days |
| Flagship / foundational | 180–360 days |

Pick the band; justify the number; name it in the announcement.

<!-- needs-research: cross-check your fleet's chosen grace periods against the authoritative deprecation policies at https://platform.openai.com/docs/deprecations, https://docs.anthropic.com/en/docs/legacy-model-deprecations, and the Google Cloud / Vertex AI model-versioning docs. Industry norms shift; freshness matters. -->

## The deprecation announcement

A single document, published to the consumer tracks' usual channels (internal docs, dev portal, release-notes feed) and to the API itself via response metadata (next section).

Contents:

- **Model name and version** being deprecated.
- **Decommission date** in UTC.
- **Replacement model** name and a one-paragraph summary of behavioural diff.
- **Migration guide** link (see next section).
- **Response-metadata deprecation signal** — the header / field consumers can read programmatically to detect usage of a deprecating model.
- **Contact and office hours** — the inference-platform's channel for migration questions.
- **Rollback / exit options** — is there a paid extension, a special-case exception path, a parallel-version serving plan?

## The migration guide

Written *for* the consumer tracks. Three sections:

1. **What's different between the deprecating model and the replacement** —
   - API-level: any change to request shape (`messages`, `tools`, `response_format`), parameter defaults, token limits, context window.
   - Behaviour-level: eval-measured deltas (rubric scores, refusal rates, length distribution); specific prompt patterns that behave differently; patterns that are still equivalent.
   - Cost-level: `$ / 1M tokens` change; possible speed change.
   - Latency-level: TTFT / TPOT change under representative workload.
2. **How to migrate** —
   - Code change: the one-line-of-code change in the OpenAI SDK / HTTP body / LiteLLM config to swap model names.
   - Prompt / behaviour tuning: known cases where the consumer-side prompt may need adjustment; links to the application-eval peer's regression report (chapter 05).
   - Testing: how to run the consumer's own evals against the replacement before switching.
   - Routing: for consumers using a feature-flag, what the flag should do during migration (both-models-tested-then-switch pattern).
3. **Operational timeline** —
   - When the deprecation-warning header will start firing.
   - When any rate-limiting throttles apply (if the deprecation policy includes throttling).
   - Decommission date, plus the policy on what happens if a consumer still uses the deprecated model after the date (hard error, routing-to-replacement with a warning, etc.).

The guide is best co-authored with at least one consumer-peer reviewer; they catch the client-side issues the inference engineer can't see.

## The deprecation-warning signal on live responses

Consumers who missed the announcement should still see a warning. Three mechanisms:

- **Response header.** `X-Model-Deprecation: <model_name>; decommission=<date>; replacement=<model_name>; guide=<url>`. Chapter 02 already requires `X-Model-Version` as the attribution axis; add this header alongside it. Non-breaking for consumers that ignore unknown headers.
- **Response-body metadata.** OpenAI's `/v1/responses` and other structured response shapes have metadata or `meta` fields where a `deprecation` object can be included — this is more visible to clients using SDKs that surface metadata. Check the API shape before adding.
- **Deprecation announcement endpoint.** `GET /v1/models/{model_name}` returns a `deprecation` object — well-defined, programmable, discoverable.

Emission discipline:

- The header is live from the deprecation-announce date through decommission.
- It is logged server-side as a counter (`deprecation_warning_responses_total` with `model_name` as a bounded label) so you can track adoption progress.
- The response is not broken; existing clients continue to function. Deprecation is a *notice*, not an incident.

## The decommission timeline

A concrete schedule with named gates. Example for a 90-day period:

| Day | Action |
|---|---|
| T-0 (announce) | Announcement published; `X-Model-Deprecation` header live on responses; migration guide published; dashboard shows per-consumer usage of the deprecating model. |
| T+14 | Office-hours session with consumer-peer leads to answer migration questions. |
| T+30 | First migration-adoption milestone check — if adoption < threshold, escalate (slack / meetings / 1:1s with consumer leads). |
| T+60 | Optional: begin rate-limit throttling for callers still using the deprecating model (contentious — use only with explicit policy and consumer sign-off). |
| T+75 | Decommission-week-ahead reminder sent to all remaining consumers. |
| T+85 | 48-hour reminder to the last consumers still on the model. |
| T+90 (decommission) | Router stops accepting the deprecated model name; responses return `404 model_not_found` or route to replacement with a header indicating auto-routing. |
| T+90 to T+97 | Rollback window — the deprecated model is drained but not yet decommissioned; if a critical consumer surfaces a blocker, the router can be flipped back on. |
| T+97 | Hard decommission — deprecated model unloaded; GPU capacity returned to the fleet (next section). |

Three disciplines:

- **Communication is scheduled, not opportunistic.** Reminders go out at the named times automatically; a Slack message per week to the broader team; a dashboard panel for consumer leads.
- **Rollback is a *real* option until the decommission date.** Chapter 02 blue-green rollback mechanic still applies if a decommission goes sideways on day T+90.
- **Hard decommission is distinct from stop-serving.** Hard decommission is when the model is unloaded, replicas are deprovisioned, and the capacity is released. If something goes wrong between T+90 and T+97, you can still bring the model back without a fresh deploy.

## Capacity-reclaim — closing the loop

The decommission's point is often capacity recovery. The reclaim step:

1. **Measure the capacity that was held** by the deprecating model — `N` replicas × hardware × hours = GPU-hours/month. Record it.
2. **Deprovision the reserved replicas.** `kubectl scale` or the equivalent to zero; node pool downscales (module 109). Verify the GPU budget is returned to the fleet pool, not held by orphaned resources.
3. **Re-budget the capacity.** The reclaimed GPU budget feeds the fleet's next scaling need — a new model's rollout, a larger fleet for the replacement, a cost-reduction opportunity. Chapter 02 of module 114 (leadership) tracks this.
4. **Document the reclaim.** The deprecation runbook's final entry is "capacity reclaimed: X GPU-hours/month" — closes the loop, provides evidence that deprecations are value-adding, feeds the next deprecation decision (which model is next).

## Cross-track hand-offs

Three hand-offs during a deprecation:

- **With the eval peers (chapter 05's contract).** The eval set often needs updating: fixtures that reference the deprecated model's behaviour are pruned or retargeted at the replacement. Owned by the eval peer; prompted by the inference engineer.
- **With the consumer peers at L25.** The migration guide is co-reviewed; the consumer-peer leads escalate migration blockers to the inference engineer; the office-hours session is co-hosted.
- **With the ML-platform peer at L30 (`ai-infra-ml-platform`).** The model registry entry is marked deprecated; the artefact is retained for the rollback window and then archived per the platform's retention policy.

## Five deprecation failure modes

- **Silent deprecation.** No announcement; the model is just removed on a Tuesday. Consumers break. Fix: the announcement is the *first* step; without it, don't decommission.
- **Grace period too short for consumers.** The team ships a 30-day deprecation on a production model; two of three consumer teams miss it. Fix: calibrate against the longest-cadence consumer, not the fastest.
- **Deprecation header not readable.** The header is emitted, but the SDK consumers use doesn't surface it; they never see the warning. Fix: dual-mechanism (header + body metadata + `GET /v1/models` response).
- **Rollback window skipped.** Decommission happens on T+90, capacity returned at T+90; a consumer surfaces a blocker at T+91 and there's nothing to roll back to. Fix: keep the model drain-but-alive for a week after decommission.
- **Capacity reclaimed but not re-budgeted.** The replicas scale to zero, the GPU budget stays on the team's bill, nobody uses it. Fix: the reclaim step explicitly re-assigns the capacity to the fleet pool.

## Worked example — deprecating a legacy chat model

Context: `mymodel-7b/v1`, a year-old 7B chat model, 2 % of traffic, replaced by `mymodel-8b/v1` (newer, cheaper, higher eval scores). Three consumer teams (two at L25 `llm-application-developer`, one at L25 `rag-engineer`); consumer release cadences biweekly, monthly, and continuous-deploy.

- **T-14:** deprecation-candidate report drafted; sign-offs obtained.
- **T-0:** announcement published; `X-Model-Deprecation` header live; migration guide live, co-reviewed by the three consumer leads; grace period set to 60 days (biweekly consumer's two cycles plus margin).
- **T+14:** office hours held; 2 of 3 consumer teams have migrated to `mymodel-8b/v1`. One team has an issue — tool-call eval on their specific tool set is 3 % below with the replacement; escalates via chapter 05 contract. AI-eval peer confirms the regression is real but minor; product partner approves graceful-degradation routing (chapter 05) — the specific tool-call feature continues to use `mymodel-7b/v1` behind a feature flag; all other usage migrates.
- **T+30:** adoption dashboard shows traffic on `mymodel-7b/v1` down 90 %; remaining traffic is the feature-flagged tool-call case.
- **T+60 (decommission):** `mymodel-7b/v1` routing turned off; the tool-call feature is temporarily routed to `mymodel-8b/v1` with an auto-routing header. Feature-flag-owning consumer team confirms degraded behaviour is acceptable for the week-long rollback window.
- **T+67:** hard decommission; GPU capacity reclaimed; 4 replicas × 8× H100 freed up; re-budgeted to `mymodel-8b/v1` fleet for an autoscaling-floor increase.
- **Post:** retrospective on the migration; the tool-call regression becomes a chapter 05 eval fixture for the next replacement model.

The whole thing is boring. That is the point.

## Summary

- Deprecation is a controlled transition between "keep it forever" and "break the consumers." The grace period is the design knob.
- The runbook: candidate report → sign-offs → announcement + migration guide + deprecation header → scheduled decommission → rollback window → hard decommission → capacity reclaim.
- Grace period calibrates against consumer release cadence; typical range 30–180+ days.
- Deprecation warnings are non-breaking; emit via header + body metadata + `GET /v1/models` endpoint.
- Rollback remains available until the hard decommission date, usually a week after stop-serving.
- Capacity reclaim is the loop-closing step; measure and re-budget the recovered GPU-hours.
- Cross-track hand-offs are with the eval peers (chapter 05), the L25 consumer peers, and the ML-platform peer.
- This chapter completes the module: from change to rollout (01, 02), to runbook (03), to incident (04), to upstream contract (05), to downstream contract (06).
