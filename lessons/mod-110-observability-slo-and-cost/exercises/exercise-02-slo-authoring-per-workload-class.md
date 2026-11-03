# Exercise 02 — SLO Authoring per Workload Class

**Time:** 3 hours · **Output:** one SLO document per workload class (3 classes), one burn-rate alert pack, one review-ritual proposal

## Goal

Make [Chapter 02](../02-slo-design-for-inference.md) concrete by **writing** the SLO documents for three workload classes on your fleet, authoring the burn-rate alert pack on top, and proposing the review cadence that keeps the documents alive.

## Problem statement

Chapter 02 argued that a one-number fleet-wide SLO hides per-class regressions and that per-token error budgets differ from per-request budgets. Prove it today by writing three one-page SLO documents, authoring the alerts that enforce them, and presenting the package in the shape your SRE partner (L30) and your product partner would sign off on.

The deliverable is prose and config, not infrastructure — this exercise is authored on paper (or Markdown) and backed by live data where you have it.

## Prerequisites

- Chapter 02 of this module read; chapter 01 as context.
- Access to recent production (or benchmark) latency data for your fleet — the SLO targets must be set against real distributions, not vendor marketing.
- Exercise 01 wired (not strictly required, but the dashboards it produces make target-setting much easier).
- A conception of the `ai-infra-sre` methodology primitives (SLI, SLO, error budget, burn-rate alert, multi-window multi-burn-rate pattern — see the Google SRE workbook at <https://sre.google/workbook/alerting-on-slos/>).

## Requirements

Four deliverables.

### Deliverable A — the workload-class list

Name three workload classes on your fleet (if you don't have a multi-class fleet, simulate with sub-categories of one class). For each, a short characterisation:

```
Class: chat
  Prompt shape: {description + distribution stats}
  Output shape: {description + distribution stats}
  Traffic fraction: {%}
  TTFT sensitivity: {low / medium / high} because {reason}
  TPOT sensitivity: {low / medium / high} because {reason}
  Current latency P95 / P99: {from exercise 01's dashboard or a one-off measurement}

Class: {next class}
  ...
```

Pick classes from chapter 02's list (`chat`, `code`, `long_context`, `batch`, `agent`, `reasoning`) or author your own if your product has a class the chapter doesn't cover. Three is the minimum; the discipline is interesting at three.

### Deliverable B — three one-page SLO documents

One document per class, in the chapter-02 template. Example filled template:

```
# Service: chat

## SLIs
- TTFT: engine-admit → first-token, from histogram `inference_ttft_seconds{workload_class="chat"}`.
- TPOT: first-token → stream-close averaged, from histogram `inference_tpot_seconds{workload_class="chat"}`.
- ITL_P99: per-token jitter, from histogram `inference_itl_seconds{workload_class="chat"}`.
- Availability: tokens-served / tokens-requested.
- Cost: $ per 1M output tokens (from chapter 04 model).

## Targets (28-day rolling window)
- TTFT_P95 < 500 ms
- TTFT_P99 < 1000 ms
- TPOT_P95 < 50 ms
- ITL_P99 < 300 ms
- Availability > 99.9% token-weighted
- Cost_per_1M_output_tokens < $4.00

## Error budget
- Latency: 0.1% of tokens served in window, token-weighted.
- Availability: 0.1% of tokens served in window.

## Alerts
- Burn-rate: 1h × 14.4 (page), 6h × 6 (page), 3d × 1 (ticket) — see Deliverable C.
- Cost: daily rollup; alert if 7-day running cost exceeds target by 10%.

## Ownership
- Primary on-call: {rotation name}
- SRE partner: {name}
- Product partner: {name}

## Review cadence
- Weekly ops review: burn rate, dashboard walk.
- Monthly: SLO target review.
- Quarterly: full target re-negotiation with product.

## Justification for the targets
- TTFT_P95 < 500ms: {reason — citation to product analytics, competitor baseline, or user research}
- TPOT_P95 < 50ms: {reason — 1/50ms = 20 tokens/sec, matches comfortable reading pace}
- Availability > 99.9%: {reason — downstream SLO requirement, commercial-tier commitment}
- Cost < $4.00: {reason — business margin or alternate-provider comparison}

## Change log
- {date}: initial SLO.
```

Three disciplines:

- Each target has a *justification*. "Because P95 is a good number" is not one; "because our competitor benchmarks at 480 ms" or "because our product analytics show session-abandonment climbs above 800 ms" are.
- Named owners on every line. An unowned SLO is an unreviewed SLO.
- A change log. If you move a target, the next on-call needs to know why.

### Deliverable C — the burn-rate alert pack

The three-alert pattern per SLO axis per class (chapter 02's multi-window multi-burn-rate recipe). Working Prometheus-rule YAML:

```yaml
groups:
- name: inference_chat_slos
  interval: 30s
  rules:
    # 1-hour fast-burn page
    - alert: ChatTTFTFastBurn
      expr: |
        (sum(rate(inference_ttft_seconds_bucket{workload_class="chat", le="0.5"}[1h]))
         / sum(rate(inference_ttft_seconds_count{workload_class="chat"}[1h])))
        < (1 - (1 - 0.95) * 14.4)        # 95th-percentile SLO burned 14.4× fast
      for: 2m
      labels: {severity: page, class: chat, slo: ttft}
      annotations:
        summary: "Chat TTFT fast-burning the SLO"
        runbook_url: "{link to runbook}"

    # 6-hour slow-burn page
    - alert: ChatTTFTSlowBurn
      expr: |
        (sum(rate(inference_ttft_seconds_bucket{workload_class="chat", le="0.5"}[6h]))
         / sum(rate(inference_ttft_seconds_count{workload_class="chat"}[6h])))
        < (1 - (1 - 0.95) * 6)
      for: 15m
      labels: {severity: page, class: chat, slo: ttft}
      annotations:
        summary: "Chat TTFT slow-burning the SLO"
        runbook_url: "{link to runbook}"

    # 3-day ticket (slow drift)
    - alert: ChatTTFTDrift
      expr: |
        (sum(rate(inference_ttft_seconds_bucket{workload_class="chat", le="0.5"}[3d]))
         / sum(rate(inference_ttft_seconds_count{workload_class="chat"}[3d])))
        < (1 - (1 - 0.95) * 1)
      for: 2h
      labels: {severity: ticket, class: chat, slo: ttft}
      annotations:
        summary: "Chat TTFT drifting toward SLO exhaustion"
        runbook_url: "{link to runbook}"

    # TPOT / availability / cost burn-rate alerts similarly
```

Three disciplines:

- Verify the burn-rate multipliers (14.4 / 6 / 1 for a 30-day window) against the SRE workbook's canonical table for your chosen window length.
- Each alert has a runbook URL. The runbook may stub for now; chapter 07 and module 113 fill in the content.
- Don't alert per-tenant. Class-level burn covers tenant-induced regressions through the token-weighted denominator.

### Deliverable D — the review-ritual proposal

A one-page proposal for the cadence that keeps these documents alive:

```
# SLO review ritual

## Weekly (every Tuesday 15:00 UTC)
Attendees: inference on-call, SRE partner, product partner.
Agenda (30 min):
  - Burn-rate traffic lights (per class, per SLO): 2 min.
  - Any red: root cause + action: 10 min.
  - Any ticket-severity alerts open: 5 min.
  - Pending memos / proposals (chapter 06): 10 min.
  - AOB: 3 min.

## Monthly (first Thursday of the month 16:00 UTC)
Attendees: Weekly + finance partner + eng director.
Agenda (60 min):
  - Monthly SLO status summary.
  - Cost review (chapter 04 dashboard).
  - Target negotiation if any class has had > 2 consecutive monthly misses or > 2 consecutive monthly passes-with-margin.
  - Upcoming initiatives + impact estimate.

## Quarterly (end of quarter)
Attendees: Monthly + VP level.
Agenda (90 min):
  - Full re-authoring of SLO targets based on past-quarter data.
  - Business-metric tie-in (chapter 06): KPI-per-TTFT-ms, cost-per-revenue-token, pending Pareto moves.
  - Next quarter priorities.
```

The ritual is the *venue* chapter 02 and chapter 06 assume. Without a venue, documents go un-read and prioritisation drifts to ad-hoc email.

## Starter guidance

- **Set targets against real distributions, not aspirations.** If your current P95 TTFT is 800 ms, setting a target of 300 ms is setting a target you will blow 90 % of the time; the alert is useless. Pick a target slightly below current P95 (or slightly below where you expect to be after exercise 05's Pareto tune).
- **Three classes are enough for this exercise.** 7 classes is a real fleet's shape; 3 is the pedagogical minimum. If your fleet has one class, split synthetically by prompt length.
- **Availability token-weighted may require a middleware shim.** Most engines don't emit `tokens_streamed_total` as a counter; you might need to synthesise from the stream.
- **Burn-rate multipliers depend on window length.** The 14.4 / 6 / 1 recipe is for a 30-day window; verify against the SRE workbook for your chosen window.
- **Cost SLO may be a placeholder.** If the chapter-04 cost model isn't built yet, the cost SLO target can cite a known quote (e.g. "below Fireworks' published price of $0.90") and exercise 04 fills in the detail.

## Fallbacks for constrained environments

- **No production data.** Use exercise 01's induced-regression data as baseline. Note the caveat; real production data refines.
- **One-workload fleet.** Split by `prompt_length_bucket` as a stand-in. The SLO shape is identical; the numbers are just more speculative.
- **No Prometheus / Alertmanager running.** The alert pack as YAML is a valid deliverable; the wiring is exercise 03.
- **No designated product or SRE partner.** Put a placeholder role name in the owner lines and note that owner assignment is a pending prerequisite for sign-off.

## Acceptance criteria

1. **Deliverable A** names three workload classes with distinguishing characteristics; each class is a shape, not a tenant.
2. **Deliverable B** has three one-page SLO documents, each with targets, justifications, owners, error budget, and a change log.
3. **Deliverable C** has a working alert pack covering at least TTFT and availability per class; multipliers cited to the SRE workbook or equivalent source.
4. **Deliverable D** has a review-ritual proposal with specific cadences, attendees, and agendas.
5. Targets are grounded in real distributions (exercise 01 data, production data, or cited external baselines) — not fabricated.
6. Availability SLI is token-weighted (chapter 02), not request-weighted.

## Common failure modes

- **One SLO for the fleet.** The whole point of chapter 02 is that per-class slicing matters. If your deliverable is "P99 < 2 s for the fleet," you missed the exercise.
- **Request-weighted availability.** A partial stream is partial credit; chapter 02's SLI is token-weighted. Easy to get wrong.
- **P99 alerts on low-QPS classes.** Long-context may be 500 requests/day; P99 on 500 samples is 5 samples. Alert on P95; use P99 as a forensic dashboard metric only.
- **Fabricated justifications.** "Because P95 < 500 ms is industry standard" is not a justification. Cite product data, external benchmarks, or user research.
- **Burn-rate without a window.** `(1 - 0.999) × 14.4` depends on the window length; naming the window is part of the SLO.
- **No named owners.** An unowned SLO is a failed SLO. The ritual needs humans; the humans need to be on the document.
- **Missing cost SLO.** Chapter 02's framing is five SLOs, not four. Cost is a first-class SLO with the same discipline as latency.

## Keep it

The three SLO documents, the alert pack, and the review-ritual proposal are the fleet's SLO artefacts. Every subsequent ritual (chapter 06 memo, module 113 postmortem, module 114 vendor eval) cites them. Exercise 03 wires the alerts; exercise 04 produces the cost-SLO content; exercise 06 writes memos against them.
