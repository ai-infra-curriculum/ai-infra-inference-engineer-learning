# Exercise 05 — Scheduler Metrics into Prometheus

**Time:** 2 hours · **Output:** one Prometheus + Grafana wiring, five dashboards, five alert rules, one provoked-alert verification log, one one-page scheduler-observability cheatsheet

## Goal

Make [Chapter 05](../05-instrumenting-the-scheduler.md) concrete by wiring your engine's `/metrics` endpoint into Prometheus, building the five load-bearing dashboards, authoring one alert rule per SLO axis, and **provoking each alert to verify it fires with the correct severity and runbook link**.

## Problem statement

You have tuned the scheduler across exercises 02–04. Observability is what makes the tuning durable: when traffic shifts, when the engine upgrades, when the workload drifts, the alert fires before a user does. This exercise lands that observability.

The deliverable is a running Prometheus / Grafana / alert stack, scraping your engine, with every alert in [chapter 05](../05-instrumenting-the-scheduler.md) tested by provocation.

## Prerequisites

- The engine from exercises 02–04, launched with `--metrics-port` exposed.
- A local Prometheus install (binary, Docker container, or a dev K8s cluster with Prometheus Operator). Grafana for dashboards. Alertmanager (optional but recommended for the alert verification step).
- An existing or new workload trace (reuse the exercise-02 load generator).

A minimal docker-compose with Prometheus + Grafana is enough; no need for Kubernetes.

## Requirements

Deliver five artefacts.

### Artefact A — Prometheus scrape config

A `prometheus.yml` (or scrape-config fragment) that scrapes your engine. Minimum:

```yaml
scrape_configs:
  - job_name: 'inference-engine'
    scrape_interval: 10s
    metrics_path: /metrics
    static_configs:
      - targets:
          - 'localhost:8001'
        labels:
          engine: <vllm | sglang | trt-llm>
          pool: <pool-name>
          model: <model-id-slug>
          region: local-dev
```

Document the exact engine-launch command alongside, with `--metrics-port` set to match.

Capture `curl http://localhost:8001/metrics | head -40` to confirm the metrics are flowing. Count the number of metric families — a healthy vLLM / SGLang export has dozens.

### Artefact B — Five dashboards

Author (in Grafana or an equivalent) the five dashboards from [chapter 05](../05-instrumenting-the-scheduler.md):

1. **Scheduler overview** — one row per axis (queue depth, admission rate, preemption rate, prefill / decode mix, iteration time).
2. **Admission / queue dynamics** — queue depth over time, arrival vs. admission rates, time-in-queue histogram, request-aborted rate.
3. **Prefill vs. decode mix** — prefill fraction, iteration-time distribution (if available), chunked-prefill chunk count.
4. **Preemption and KV pressure** — preemption rate, GPU and CPU cache usage, swap-in / swap-out counts.
5. **SLO roll-up** — TTFT P50/P95/P99, TPOT P50/P95/P99, E2E P95, request-aborted rate.

Each dashboard: at least four panels, with legible axis labels and sensible y-ranges. Export the JSON (or screenshots at minimum).

Metric spellings depend on your engine version. Chapter 05 uses vLLM-shaped names as examples. **Verify the exact spellings in your version's metrics reference** — vLLM has renamed and reshaped metrics across releases. Document any name drift.

### Artefact C — Five alert rules

Author a Prometheus rules file (PromQL expressions) with one rule per SLO axis. Chapter 05 gives the template; adapt for your engine's actual metric names. Minimum:

1. `TTFT_P95_Breach` — pages when TTFT P95 > SLO for 5m.
2. `TPOT_P95_Breach` — pages when TPOT P95 > SLO for 5m.
3. `Preemption_Thrash` — pages when preemption rate > 2% for 5m.
4. `Queue_Time_P95_Breach` — pages when queue-time P95 > threshold for 5m.
5. `KV_Cache_Saturation` — tickets when GPU cache usage > 90% for 10m.

Each rule must have:

- A specific severity (`page` or `ticket`).
- A `for:` clause.
- An `annotations.runbook` link (even if the runbook is a stub — it must exist as a URL / path).
- Labels (`pool`, `engine`, `model`, `severity`, `slo_axis`).

Keep the rules in version control alongside the pinned config from exercise 02.

### Artefact D — Alert verification log

For each of the five alerts, **provoke it on purpose and record what happens**:

1. **TTFT_P95_Breach**: run the exercise-02 workload with `max-num-batched-tokens` dropped to a value that cannot admit long prompts quickly. Confirm TTFT P95 rises above the threshold on dashboard 5; confirm the alert fires in Prometheus `/alerts`; confirm (if Alertmanager is wired) a notification lands with the runbook link.
2. **TPOT_P95_Breach**: launch the engine with chunked prefill off and a long-prompt load; confirm TPOT P95 rises and the alert fires.
3. **Preemption_Thrash**: run the overcommit workload from exercise 04 with `max-num-seqs` low; confirm the preemption rate crosses 2% and the alert fires.
4. **Queue_Time_P95_Breach**: run the load generator at 2× the Pareto-chosen admission rate; confirm the arrival queue backs up and the alert fires.
5. **KV_Cache_Saturation**: run a long-prompt + long-output workload; confirm GPU cache usage sustains > 90% and the ticket-level alert fires after the `for:` window.

For each: a one-paragraph log entry with the timestamp, the provocation command, the metric value at firing time, and a screenshot (or copy of Prometheus alert state).

After each verification, **return the knobs to the exercise-02 setpoint and confirm the alert resolves cleanly within the `for:` window plus a few scrape intervals.** An alert that fires but doesn't resolve is as bad as one that doesn't fire.

### Artefact E — One-page scheduler-observability cheatsheet

A single page another operator can use to:

- Launch the engine with `/metrics` on.
- Point Prometheus at it.
- Pull the five dashboards (or at least a one-line "where to find each dashboard").
- Know the five alert rules and the first-check metric for each.
- Know the three biggest "metric drift" risks across engine versions (vLLM renames, SGLang / TRT-LLM naming differences, missing metrics in older releases).

Reuse the exercise-01 inner-loop cheatsheet shape; keep both pinned together.

## Starter guidance

- **Start the stack with docker-compose.** A Prometheus + Grafana + (optional) Alertmanager docker-compose is 20 lines of YAML and avoids installing anything system-wide. There are many public examples; see [`resources.md`](../resources.md).
- **Pull metrics first, draw dashboards second.** Confirm every metric you reference in a dashboard actually exists in your engine's `/metrics` output. A dashboard that references a renamed metric looks empty — easy to miss until it's load-bearing.
- **Histograms are the lingua franca.** Latency axes (TTFT, TPOT, E2E, queue time, inter-token latency) should be histograms, not gauges or counters. `histogram_quantile(0.95, rate(..._bucket[5m]))` is the standard incantation.
- **Alert `for:` clauses prevent flapping.** 1-minute `for:` is too jumpy; 5-minute is the practical minimum for SLO alerts. For `KV_Cache_Saturation`, 10 minutes avoids pages on transient bursts.
- **Alert verification is the critical step.** An un-provoked alert is an un-trusted alert. If an alert does not fire when you deliberately engineer the condition, the expression is wrong. Fix before shipping.
- **Keep rules file-based and version-controlled.** `scheduler-alerts.yml` next to `serving-config/<pool>.yaml`. PRs on alerts get the same review as PRs on code.
- **Mute the stack during exercise 04 or similar provoked runs.** You will intentionally trip alerts; silence Alertmanager notifications for the duration unless you want to spam a channel.
- **Label hygiene.** Add `pool`, `engine`, `model`, `region`, `tenant` (if applicable) to every metric at scrape time. Chapter 05 is explicit; this is the step most teams get lazy about and regret in year 2.

## Fallbacks for constrained hardware

- **No Grafana?** Prometheus's built-in Graph view is enough to validate the dashboards; export JSON for the five dashboards and note that Grafana rendering is deferred.
- **No Alertmanager?** Prometheus `/alerts` endpoint shows firing alerts in-stack; use that to verify firing even without a notification channel.
- **Single-engine single-pool.** The multi-label hygiene matters more at fleet scale; for a dev run, the labels are documented in Artefact E and applied at the first production rollout.

## Acceptance criteria

1. **Artefact A** shows a running Prometheus scraping the engine, with ≥ 15 metric families scraped and labelled.
2. **Artefact B** has five dashboards, each with ≥ 4 panels, each panel referencing metrics that exist in your engine version (verified against `curl /metrics`).
3. **Artefact C**'s five alert rules each have `for:`, `severity`, `annotations.runbook`, and proper labels. The runbook URLs resolve (even to a stub page).
4. **Artefact D** has a verification log for each of the five alerts, with a provocation command, the metric value, and the firing state captured. Each alert also *resolves* within a reasonable window after the knob is restored.
5. **Artefact E** fits on one page and reuses the exercise-01 cheatsheet's structure.
6. The whole stack is captured in a reproducible form (docker-compose + rules files + dashboard JSON) that another engineer can bring up in under 15 minutes.

## Common failure modes

- **Alert names look right; expressions silently never fire.** Expression typos in PromQL are easy. Verification by provocation is the only check that works.
- **Metric drift across engine versions.** vLLM has renamed metrics (e.g., removed trailing underscores, moved from `vllm:` prefix to no-prefix in some versions). Verify every metric name against your installed version's `/metrics` output.
- **Dashboards with empty panels.** Usually a metric name typo or a missing label. Fix at authoring time.
- **`for: 0s` alerts.** Pages flap on every 10-second scrape jitter. Set `for:` to at least 5 minutes for SLO alerts.
- **No runbook link.** The on-call clicks "runbook" when the page fires. If it's not there, the page is just noise.
- **Alerts tested only in staging, never provoked.** Fire them. The verification step is non-negotiable.

## Keep it

The Prometheus config, dashboards, and alert rules are the production observability bundle. They sit alongside the pinned config, the chunked-prefill setpoint, and the preemption-policy choice from exercises 02–04. Together they are the full "scheduler ships to production" artefact set. Module 110 extends this with cost accounting, cross-fleet dashboards, and the SLO methodology.
