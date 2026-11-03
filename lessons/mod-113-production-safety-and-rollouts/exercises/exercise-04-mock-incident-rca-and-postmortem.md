# Exercise 04 — Mock Incident: RCA and Post-Mortem

**Time:** 6 hours · **Output:** one incident drill (induced or replayed), one RCA artefact with metric overlays and config diffs on a single page, one blameless post-mortem authored to the shape chapter 04 specifies, and one merged PR for the regression-fixture back-feed.

## Goal

Make [Chapter 04](../04-incident-response-and-rca.md) concrete. Run the inference-triage pattern from first page to resolved; assemble the RCA artefact in the required shape; write the post-mortem with inference-specific sections; and *close the loop* by merging the back-feed fixture into the eval set or the canary gate. The exercise is complete when the fixture is a durable artefact that would catch the same regression next time.

## Problem statement

Pick one of these three incident shapes and exercise it end-to-end:

- **Induced TTFT regression** (recommended for first attempt): on a staging replica, reduce `max-num-batched-tokens` or raise `max-num-seqs` beyond safe limits; drive representative traffic until TTFT breaches a documented SLO; run the drill from the burn-rate alert through to recovery.
- **Induced cache-hit cliff:** change the prefix-cache TTL to near-zero mid-traffic; watch the cache-hit rate drop and TTFT climb; drill.
- **Replayed silent-drift incident:** take a historical post-mortem (yours or a public one) and replay the triage flow using chapter 04's pattern against the recorded data.

Whichever you pick, the deliverables are the same.

## Prerequisites

- Chapters 03 and 04 of this module read.
- Module 110's SLO authoring (chapter 02) completed so you have real burn-rate alerts to fire.
- A staging fleet with the module-110 observability stack (dashboards, traces, alerts).
- Access to a chat application (you can author ad-hoc load) or a traffic replay tool.
- The ten runbooks from Exercise 03 (if run); otherwise the chapter 03 template and your best effort at the first-bisection step.
- A repo where eval-set fixtures or canary-gate fixtures live (chapter 05 contract's fixture surface, or your own staging-equivalent).

## Requirements

Five deliverables.

### Deliverable A — the incident drill

Run the drill end-to-end:

1. Induce / replay the regression (document the induction).
2. The burn-rate alert fires (or you observe it on the dashboard; if replaying, open the recorded alert in the historical time range).
3. **Triage trace.** Follow chapter 04's triage pattern — five steps, time-boxed. Record each step's action, the panel / log / metric consulted, what it showed, and what it ruled in or out.
4. Pick the matching runbook from Exercise 03. If no runbook matches, note that gap — that's a post-mortem action item.
5. Apply the first fix. (For induced regressions, revert your induction. For replayed incidents, do not re-apply any fix — just walk through what the runbook would have you do.)
6. Observe recovery. Note time from first-burn to recovery.

For drills that don't pick a known runbook on step 4, document what the engineer did instead — this is the "incident where the triage missed" case chapter 04 explicitly cares about.

### Deliverable B — the RCA artefact

One printable page containing:

- **Incident timeline** (UTC horizontal axis) with markers for first anomalous metric, burn-rate alert fired, each triage step, fix applied, service recovered, incident closed.
- **Metric overlays** — at minimum the SLO axis at issue (e.g. TTFT P99) and two correlated axes (queue depth, cache-hit rate, FB_USED, preemption rate — whichever is relevant); the window around the incident; a 7-day same-hour reference overlay for context.
- **Model-version attribution** — which `X-Model-Version` values served what fraction of traffic during the window.
- **Config diff** — a `git diff` or diff-shaped snippet showing the change that caused the incident (your induction, or the recorded change if replaying).
- **Trace samples** — two to five representative OTel traces from the window, chosen to expose the specific bisection that led to the diagnosis.
- **Cause chain** — a 1–3 sentence narrative from trigger to effect to detection to fix.

Render it as a single PDF or markdown page. If multi-page, trim; the point is it's readable in one glance.

### Deliverable C — the blameless post-mortem

Written to chapter 04's inference-flavoured template, built on top of the SRE peer's methodology framework:

1. **Summary** — what happened, impact (tokens affected, users affected if known, duration).
2. **Triage trace** — the five triage steps with findings.
3. **Root cause** — the cause chain.
4. **Contributing factors** — inference-specific (SLO target permissiveness, gate tolerance, missing axis in observability).
5. **What went well.**
6. **What went badly.**
7. **Action items:**
   - Code / config.
   - **Eval-set back-feed** (names the eval peer per chapter 05 contract).
   - **Canary-gate back-feed** (specific chapter 01 gate addition).
   - **Runbook back-feed** (specific chapter 03 revision or new entry).

Blameless framing throughout — no names of responsible engineers; language is about the system, not people.

### Deliverable D — the fixture back-feed (merged)

Pick one of the action items from Deliverable C and actually ship it:

- **Eval-set fixture:** add the regression's representative inputs to an eval set as a reproducible fixture `(prompt, params, seed, expected behaviour, workload class, tolerance)`. Open a PR; merge (or simulate the merge with a review approval if the real eval-set repo isn't accessible).
- **Canary-gate fixture:** add a rule to the chapter 01 gate config — e.g. "candidate must serve ≥ N long-context requests with ITL P99 within tolerance before promoting." PR and merge against the Exercise 01 harness.
- **Runbook addition:** revise or add a chapter 03 runbook entry to cover the gap exposed by the triage. PR against the Exercise 03 runbook set.

A merged PR, not a draft. "I'll do it next quarter" is the failure mode chapter 04 names.

### Deliverable E — reflection

Short write-up (one page):

- Where in the triage did the chapter 04 pattern work well?
- Where did it lag?
- Which runbook from Exercise 03 fit? If none, what was missing?
- What did you learn about the module-110 axis set — was any axis missing that would have accelerated the triage?

## Starter guidance

- **Induce the regression on staging.** Don't exercise this on production unless you have explicit approval; the chapter 04 drill is cheap in staging and expensive in production.
- **Time-box the triage.** Chapter 04's budget is 10 minutes to pick a runbook. Hold yourself to it during the drill; if you exceed, note *why* in Deliverable E.
- **Capture dashboard screenshots live.** Grafana's time-bounded permalinks are gold for the RCA; set them during the drill.
- **The RCA fits on one page.** Multi-page RCAs don't get read by the next on-call engineer six months later. Trim.
- **The post-mortem is blameless.** Language like "the engineer misread the dashboard" is wrong; "the dashboard layout led the on-call engineer to the queue panel before the TTFT decomposition panel" is right.
- **The back-feed is a merged PR.** Not a ticket, not a draft, not a plan. A PR, reviewed, merged, with a verification that the gate / fixture would catch the regression.

## Fallbacks for constrained environments

- **No staging fleet to induce:** replay a public post-mortem (e.g. the OpenAI / Anthropic / Google incident retrospectives) against chapter 04's pattern; your RCA artefact is a re-statement of theirs in this module's shape.
- **No eval-set repo:** simulate the back-feed in a local repo and show the fixture definition. State the limitation; the discipline of writing it matters more than the merge target.
- **No OTel traces:** the RCA's trace samples section can be log excerpts instead. Document the shortfall.

## Acceptance criteria

1. **Deliverable A** — the drill ran end-to-end; each triage step is logged with its finding.
2. **Deliverable B** — the RCA is a single printable page with timeline, overlays, version attribution, config diff, trace samples, and cause chain.
3. **Deliverable C** — the post-mortem has the inference-flavoured sections; it is written blamelessly; action items are specific and assigned.
4. **Deliverable D** — a merged PR back-feeds at least one action item into a durable gate or fixture.
5. **Deliverable E** — a reflection that evaluates the chapter 04 pattern and identifies observability gaps.
6. The drill artefact set is reusable — the next on-call engineer can use the RCA as a reference for a similar future incident.

## Common failure modes

- **Triage without time-box.** Expands to fill available time; the "10 minutes to pick a runbook" discipline slips. Fix: explicit timer during the drill.
- **RCA without config diff.** Overlay shows correlation; you're guessing at cause. Fix: diff is non-optional.
- **Post-mortem with vague action items.** "Improve observability" is not actionable. Fix: each action item names a specific commit, PR, or dashboard change.
- **Back-feed deferred.** The point of the loop is closed; a never-shipped fixture is a never-closed loop. Fix: merge as part of this exercise.
- **Drill done on production.** Blast radius too high; use staging.

## Keep it

The RCA artefact, the post-mortem, the merged fixture PR, and the reflection. These become the shape of the real post-mortems your team will run next quarter. The reflection in particular feeds the module-114 cadence as evidence for what the team should invest in next.
