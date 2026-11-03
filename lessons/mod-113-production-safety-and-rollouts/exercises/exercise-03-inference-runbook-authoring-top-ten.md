# Exercise 03 — Inference Runbook Authoring: the Top-Ten

**Time:** 8 hours · **Output:** ten one-page runbooks for your fleet, authored against the chapter-03 template, each linked to the module-110 axis set, each dry-run against induced or recorded traffic patterns, with an on-call sign-off that the sheet is usable at 2 AM.

## Goal

Make [Chapter 03](../03-inference-runbooks.md) concrete. Write the ten runbooks the on-call rotation will actually reach for; dry-run each one so the first-bisection step is tested before it is read in anger.

## Problem statement

Your fleet's current runbook library is a handful of inherited pages, generic "service is slow" documents, and scattered debug tips. Chapter 03 named the ten inference-specific incidents the top 80 % of on-call hours land in. Author those ten against your fleet, your engines, your SLOs, and your dashboards — so the next on-call engineer finds the right page in under a minute and the right bisection in under five.

## Prerequisites

- Chapter 03 of this module read; chapter 04 for the RCA / escalation pattern each runbook references.
- A working fleet with observability (modules 110 chapters 01 and 03; module-specific debug surfaces from 102 / 103 / 104 / 107 / 109).
- Access to the module 110 dashboards and traces.
- At minimum one historical incident per runbook, or the ability to induce the pattern (module 110 exercise 01's induced-regression suite is the base source for TTFT / preemption / queue-saturation).
- An on-call peer or a simulated reviewer to sign off on the sheets.

## Requirements

Three deliverables, replicated across the ten runbooks.

### Deliverable A — the authored runbook set

One markdown file per runbook. Each file uses the chapter-03 template literally:

```
# Runbook: <title>
## Owned by
## Symptoms (you're looking at this if...)
## First bisection (do these in order)
## Most likely causes and first fixes
## Escalation
## Rollback
## References
```

The ten runbooks (chapter 03 names them):

1. **TTFT regression**
2. **Cache-hit cliff**
3. **Preemption spike**
4. **Silent accuracy regression / output drift**
5. **NCCL hang**
6. **GPU driver / XID / ECC error**
7. **Worker crash loop**
8. **Admission-reject flood**
9. **KV OOM**
10. **Streaming-close storm**

Discipline:

- **Specific, not generic.** Flags are your engine's actual flags, pinned to version. Metric names are the ones your dashboards use. Replica identifiers follow your Kubernetes naming.
- **One page rendered.** No scroll on a laptop. The on-call engineer at 2 AM gets the gist in 90 seconds.
- **Commands are copy-pasteable.** `kubectl drain my-ns/my-replica-7`, not `drain the problem replica`.
- **Escalation names reachable humans.** `#oncall-inference-platform` or an email alias with routing, not "SRE."
- **Dashboard links are time-bounded.** Preserve the pattern you'd expect during an incident.

### Deliverable B — the axis-set linkage

Each runbook's first-bisection step opens the right module-110 dashboard panel. Make this explicit — include a one-line table in each runbook:

```
| Axis to read first                      | Dashboard panel link                |
|-----------------------------------------|-------------------------------------|
| ttft_seconds_bucket / queue_seconds / ... | <grafana link with time range var> |
```

For each runbook, name the exact axis from module 110 chapter 01 that the on-call engineer reads first. If the axis is missing from your fleet, that's a gap — document it as a finding and open a ticket to close it.

### Deliverable C — dry-run against induced or recorded traffic

Each runbook is tested *before* publication. Three ways to dry-run:

- **Tabletop:** walk the runbook with an on-call peer; have them play "incident commander" and confirm the runbook's language leads them to the right diagnosis without prior knowledge.
- **Induced regression:** reproduce the incident pattern on a staging fleet (module 110 exercise 01 produces TTFT / queue-saturation / preemption; other patterns may need a specifically-induced scenario — e.g. kill a replica to exercise crash-loop).
- **Historical replay:** take a recorded incident's dashboard data (saved time range) and have an engineer who did not run the original incident use the runbook against the historical data.

For each of the ten runbooks, at least one dry-run method; record for each runbook:

- Which method you used.
- Whether the runbook correctly led to the right diagnosis.
- What revision was needed after the dry-run.
- Who signed off (name + date).

## Starter guidance

- **Start with the runbooks you can dry-run cheaply.** TTFT regression and preemption spike can be induced easily on a staging fleet; silent accuracy regression needs a recorded incident or a shadow-rolled regressed candidate — do the easier ones first.
- **Write each runbook from a real or hypothetical incident.** Open with "the on-call engineer is paged at 09:14 UTC because...". If you can't open with a concrete scenario, you don't have the symptoms right.
- **Base rate matters.** For "Most likely causes," order by frequency in your fleet, not by generic plausibility. Empirical ordering comes from historical post-mortems; if you don't have enough history, say so and plan to re-order after six months of incidents.
- **Cross-reference, don't duplicate.** Multiple runbooks will land on module 102's engine-failure-modes chapter; link, don't paste.
- **Keep a "long tail" file.** Ten runbooks aren't everything. Keep a separate file of one-liners for the long-tail incidents (ingress DNS flap, cert rotation, image-pull slow, specific model-registry outage) so the on-call engineer has somewhere to look.
- **Review date in the header.** `last reviewed: 2026-11-03` matters; stale runbooks are worse than no runbook.

## Fallbacks for constrained environments

- **No staging fleet to induce regressions:** rely more heavily on tabletop and historical replay. A simulated incident (on a notebook, with a scripted dashboard replay) is better than nothing; document the limitation.
- **No real on-call rotation:** simulate the review with a peer unfamiliar with the specific incident patterns. Their fresh read is similar to an on-call engineer's.
- **No historical incident record:** write the runbook anyway with placeholder "common cause" ordering, and commit to re-ordering after six months of logged incidents.

## Acceptance criteria

1. **All ten runbooks** authored, one per file, using the template literally.
2. **Each runbook fits on one page** when rendered; commands are copy-pasteable; escalation is reachable.
3. **Axis-set linkage** present for every runbook — the first-bisection step points to the specific module-110 panel.
4. **Dry-run evidence** for each — method, result, revisions, sign-off.
5. **Long-tail file** exists with one-liners for incidents not in the top ten.
6. **Review cadence** scheduled — a calendar event / cadence hook that re-reviews each runbook quarterly.

## Common failure modes

- **Encyclopaedic.** 20 pages per runbook; on-call never reads them. Fix: hard-cap at one page.
- **Opinion-free.** "Investigate the cache." Fix: name likely causes in order; recommend a first action.
- **Stale.** Engine-version-specific flags from six versions ago. Fix: review cadence + last-reviewed date.
- **Over-generic.** Runbook could apply to any service. Fix: specifics (flags, metric names, replica identifiers).
- **No fallback path.** First fix exhausts; runbook has no plan B. Fix: escalation is explicit.

## Keep it

The ten runbooks go into the on-call team's runbook directory (git repo or docs system). The dry-run results and sign-offs stay as a log of what has actually been tested. The long-tail file is the compost heap for the next quarter's top-ten revision.
