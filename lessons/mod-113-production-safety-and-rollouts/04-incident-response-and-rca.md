# Chapter 04 — Incident Response and RCA for Inference: Triage, Overlays, Version Diffs, Post-Mortem, Fixture Back-Feed

> SRE methodology gives you the shape of incident response — command, scribe, communicator, blameless framing, action items. Inference specialises the content: the first axis you look at is TTFT decomposition, not error rate; the RCA overlays model-version diffs on metric time-series; the post-mortem's most important action item is often "add this to the eval set." This chapter plugs the inference-specific content into the SRE-methodology container.

## Motivation

A generic SRE incident goes: alert fires → commander assigns roles → investigators pull dashboards → fix is applied → service recovers → post-mortem. The content of "pull dashboards" is where inference diverges.

An inference TTFT regression won't yield to "look at the HTTP 5xx rate" — the incident is slow success, not failure. A silent accuracy regression won't yield to *any* latency dashboard — the signal is external. An NCCL hang might show as zero throughput on one replica while every "service up" probe passes. The incident commander (methodology from the SRE peer) needs an *inference triage pattern* that handles all of these, and an RCA artefact that encodes *inference-specific evidence*: traces, metric overlays, model-version diffs, config diffs, and the deploy-event timeline on one page.

Chapter 03 wrote the runbook per failure mode. This chapter writes the connective tissue — triage for when you don't know which runbook, RCA for after the fix, post-mortem for the record, and the fixture back-feed that prevents recurrence.

By the end of this chapter you can:

- Triage an unknown inference incident against the axis set from module 110 in under five minutes.
- Assemble the RCA artefact with traces, metric overlays, and version / config diffs on one timeline.
- Write a blameless post-mortem in the shape the SRE peer's methodology expects, with inference-specific content.
- Back-feed a regression fixture into the eval set and the canary acceptance gate so this incident cannot recur silently.
- Avoid the five common failure modes of inference incident response.

## Triage — the first five minutes

The inference triage pattern is specifically the chapter 01 axis set from module 110 inspected in order. It is the thing that is called *before* a runbook gets picked, when the on-call engineer doesn't yet know which runbook applies.

The sequence:

1. **Where is the burn? Request-side or system-side?**
   - A latency / error-rate SLO alert fired → *request-side*; continue to step 2.
   - A hardware / infrastructure signal fired (XID, driver, worker crash loop) → *system-side*; jump to runbook 6 / 7 / 5.
   - No alert fired but a *product metric* dropped → *silent drift*; jump to runbook 4.
2. **Which replica / fleet / region?** Split SLO panel by `replica_id` (bounded) and `region`. If one → drain + investigate there. If all → fleet-wide; continue.
3. **Which axis of the burn?** Specifically on the decomposition from module 110 chapter 01:
   - TTFT high → runbook 1 (continue bisecting queue vs. prefill vs. first-decode).
   - TPOT high → check preemption rate; probably runbook 3.
   - ITL P99 high → preemption spike (runbook 3) or NCCL stragglers (runbook 5).
   - Cache-hit dropped → runbook 2.
   - Admission rejects spiked → runbook 8.
   - Error rate / KV OOM aborts → runbook 9.
   - Client disconnects / streaming closes → runbook 10.
4. **Did we deploy anything?** Check `X-Model-Version` histogram shift and the deploy-event timeline for the past 4 hours. If yes → rollback is a candidate before deeper diagnosis (chapters 01 / 02 rollback mechanics).
5. **Pick the runbook.** From step 3, the matching runbook is the next page.

This is explicitly *inference triage*; incident-command roles (commander, scribe, communicator) come from the SRE peer's methodology and sit on top. The commander calls the triage above; the scribe records the step 1 → step 5 sequence and its findings; the communicator updates stakeholders.

### Time budget

- Step 1: ≤ 1 min (an eyeball at the dashboard).
- Step 2: ≤ 2 min.
- Step 3: ≤ 2 min.
- Step 4: ≤ 1 min (change calendar is bookmarked).
- Step 5: hand to the matching runbook.

If the triage doesn't land in < 10 minutes, something is wrong with the triage pattern — add that case to the post-mortem.

## The RCA artefact — overlays and diffs on one page

Root-cause analysis is the step after the fix. It asks: *what actually happened, in what order, caused by what, and detected how*. The artefact is a single page (literally, printable) containing:

- **Incident timeline.** Horizontal time axis from first anomalous metric to recovery; markers for alerts fired, actions taken, rollbacks initiated, user-visible events, external events (vendor outage).
- **Metric overlays.** Three or four metric time-series at the incident window, aligned to the same time axis: the primary SLO axis (e.g. TTFT P99), one or two correlated axes (queue depth, preemption rate, FB_USED), and the deploy events as vertical markers.
- **Model-version diff.** The `X-Model-Version` label distribution over the window; which versions were serving which fraction of traffic; which versions showed the regression.
- **Config diff.** Any engine flags, scheduler parameters, router weights, feature flags, or infrastructure changes active in the window — ideally as a `git diff`-shaped snippet from your config-as-code repo.
- **Trace samples.** Two to five representative OTel traces from the regression window, chosen because they expose the chapter 01 TTFT decomposition that led to the diagnosis.
- **Cause chain.** A narrative paragraph (1–3 sentences) from trigger to effect: *"deploy T-0 raised `max-num-seqs`; KV pressure triggered preemption for long-context sequences starting T+5; ITL P99 breached SLO at T+8; auto-rollback fired T+9; service restored T+10."*

Three disciplines:

- **Overlays reveal correlation; diffs reveal cause.** A metric overlay by itself says "both of these spiked at the same time"; the version / config diff says *why*. Never publish an RCA without both.
- **The timeline is in UTC.** Multi-region, multi-timezone teams; UTC-only avoids reader errors.
- **Keep a known-good baseline on the panels.** Same panel, same query, previous 7-day same-hour overlay — this is what distinguishes "weird" from "normal."

### RCA assembly tooling

In practice the RCA is a document — a post-mortem Confluence / Notion / markdown page — plus a Grafana link with a shared time range. A few disciplines that scale:

- **Dashboard permalinks are time-bounded.** Grafana's `from=` and `to=` query params; save the exact window on the RCA page so the reader six months later sees the same thing you did.
- **Deploy events are a Prometheus annotation.** Every chapter 01 / 02 ramp step emits an annotation; the RCA overlay reads them automatically.
- **Config diffs live in a repo.** All engine configs, scheduler flags, router weights, and feature flags are code; every change is a PR; `git log --since=...` is the config change feed for the incident window.

## Post-mortem — blameless, inference-flavoured

The SRE peer's methodology writes the blameless-post-mortem template. Chapter 04 of this module *specialises the content* so the inference-engineer reader (and the eval-peer, consumer-peer, and product-partner reviewer) get what they need from it.

The inference-flavoured sections, layered on top of the SRE-methodology post-mortem template:

1. **Summary.** What happened, impact (tokens affected, users affected, revenue if quantifiable), duration. One paragraph.
2. **Triage trace.** The five triage steps with what each one showed; useful for runbook refinement.
3. **Root cause.** The cause chain from the RCA artefact, in one paragraph.
4. **Contributing factors.** Inference-specific: was the SLO target too permissive? Was the canary gate's response-diff tolerance too loose? Did the shadow-rollout fixture lack long-context coverage?
5. **What went well.** Auto-rollback fired / didn't fire; the runbook handled it / was missing; the on-call engineer found it in N minutes using which panel.
6. **What went badly.** The usual list; be specific to inference.
7. **Action items.**
   - **Code / config action items** (runbook of the usual shape).
   - **Eval-set back-feed action items** — described in the next section; this is the inference-specific piece.
   - **Canary-gate back-feed action items** — add the regression's signature as a canary acceptance gate for future rollouts.
   - **Runbook back-feed action items** — if the triage missed the case, add it to the chapter 03 top-ten or add the one-off to a longer list.

The blameless framing is methodology; the content is specialised.

## Regression-fixture back-feed — the step that prevents recurrence

The single most important inference-specific post-mortem action: *add this incident's pattern to the gates that would have caught it before promotion*.

Three places it lands:

- **Eval set (chapter 05 contract with the eval peers).** The regression's representative inputs become a held-out eval case. The fixture carries: `(input, expected equivalence tolerance against last-known-good, task-score threshold, workload-class tag)`. The eval peer owns the eval set; this role contributes the inference-specific fixture.
- **Canary acceptance gate (chapter 01).** The regression's signature on the axis set becomes a per-metric gate. Example: an NCCL-hang incident caused by a specific long-context pattern becomes a canary gate that requires "≥ N long-context requests through the candidate with ITL P99 within tolerance before promoting."
- **Runbook (chapter 03).** If the triage missed the case, add it to the top-ten or the long-tail runbook list; schedule a quarterly review.

### Back-feed discipline

- **The fixture is reproducible.** Prompt, params, seed, expected behaviour; stored in version control.
- **The fixture has an owner.** When a model is retired (chapter 06), its fixtures are reviewed for relevance.
- **The fixture back-feed is tracked.** The post-mortem's action item has a real PR and real merge; a monthly audit confirms the fixtures made it into the gates.

The failure mode: a well-written post-mortem with action items that are never implemented. Build a cadence that enforces follow-through (module 114's leadership cadence picks this up).

## Multi-service incidents — the cross-cutting case

When the inference regression is caused by *another team's change* (an ingress upgrade, a shared KV cache service outage, a DNS flap, a vendor API degrade affecting the fallback router), the incident is cross-cutting and:

- **Incident command is shared.** The SRE peer's methodology for multi-service commander-across-teams applies; this role contributes the inference-specific triage and RCA content.
- **The RCA names the cross-cutting dependency.** The deploy-events timeline must include the other team's deploys and vendor status updates.
- **Post-mortem is co-signed.** Action items belong to one owner each; the cross-team ones have a named owner on the other team.

Don't try to do this part alone; the SRE peer's methodology exists specifically for cross-service incident command.

## Five common failure modes

- **Alert fired on fleet-scoped metric; regression was candidate-scoped.** Dilution by the primary hides the alert. Fix: chapter 01 candidate-scoped version-labelled alerts.
- **Rollback skipped because "this change couldn't have caused it."** The rollback is cheap; the diagnosis is expensive; rollback first, diagnose from a safe state. Counter-example: silent drift (runbook 4) where diagnosis *is* the rollback decision.
- **RCA without version / config diff.** Overlay shows correlation; without the diff you're guessing. Fix: require both on every published RCA.
- **Post-mortem's action items never land.** Fix: a monthly audit (module 114) that closes or escalates stale action items.
- **Fixture back-feed deferred "for a quieter week."** The incident recurs in six months because the gate never shipped. Fix: the back-feed PR is a *blocking* action item on the post-mortem, not a "nice to have."

## Worked example — a cache-hit cliff incident

Minimally fictionalised sequence, structured against this chapter's pattern.

- **09:14 UTC** — `TTFT_P99(chat)` SLO burn-rate alert fires. On-call commander opens incident.
- **09:15** — Triage step 1: request-side burn, not system-side; step 2: fleet-wide; step 3: TTFT, decomposed → prefill dominates; step 4: a flag change deploy happened T-20 min on the KV allocator. Candidate runbook: runbook 2 (cache-hit cliff) via the prefill-dominated signal.
- **09:17** — Dashboard confirms `cache_hit_rate` dropped from 0.62 to 0.08 at T-20.
- **09:19** — Rollback the flag change via chapter 01 mechanic; new cache config propagates in 2 minutes.
- **09:22** — `cache_hit_rate` recovers to 0.58; `TTFT_P99(chat)` returns to SLO within one scrape interval.
- **09:30** — Incident closed; RCA commissioned.
- **Next day** — RCA artefact: timeline with 09:14 alert, 09:19 rollback, 09:22 recovery; overlay of TTFT / cache-hit / prefill-tokens-per-second; config diff showing the KV allocator flag change (`--prefix-cache-ttl` from 3600 to 60 seconds, accidentally shipped as part of an unrelated refactor); trace samples showing cold-prefix prefills.
- **Post-mortem action items** —
  - Code: block refactors from changing cache-related flags without an explicit flag changelog entry (automation PR).
  - Fixture: add a regression fixture that drives a high cache-hit workload and asserts `cache_hit_rate ≥ 0.5` on candidate (canary-gate back-feed, exercise 01 in chapter 05).
  - Runbook: refine runbook 2's first-bisection to include "check for cache-TTL config changes in last 4 hours" as step 2.5.
  - Cadence: schedule a quarterly review of KV-allocator flags under the module-114 change calendar.

Every incident does not need to be this clean; this is the shape you aim for.

## Summary

- Triage is chapter 01's axis set, inspected in a fixed order, timeboxed to ten minutes. It happens before a runbook is picked.
- The RCA artefact overlays metric time-series with version / config diffs on one page; both are required.
- The post-mortem adopts the SRE peer's blameless framing and adds inference-specific sections (triage trace, back-feed action items).
- Fixture back-feed into the eval set (chapter 05) and the canary acceptance gate (chapter 01) is the step that prevents recurrence; make it a blocking action item.
- Cross-cutting incidents escalate to the SRE peer's multi-service incident-command methodology; this role contributes the inference content.
- Chapter 03 handles the known-shape incidents; this chapter handles the glue, the forensic artefact, and the loop that closes the gap.
