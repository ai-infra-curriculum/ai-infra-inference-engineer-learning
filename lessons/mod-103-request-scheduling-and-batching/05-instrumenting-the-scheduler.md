# Chapter 05 — Instrumenting the Scheduler

> If you cannot see the scheduler's state on a dashboard, you cannot defend an SLO. This chapter is the minimum metric set, the derived SLOs built on top of them, and the alert feed into module 110.

## Motivation

Chapters 01–04 wrote the operator's model of the scheduler and the knobs you turn against an SLO. This chapter closes the loop: the metrics that tell you whether your tuning *is* holding, and the alerts that fire *before* a user notices.

The job split with module 110:

- **This chapter** — the scheduler-specific axes, their Prometheus wiring, and the alert *expressions*. The things a scheduler-tuning engineer owns end-to-end.
- **Module 110** — the broader SLO engine (error budgets, burn rates, incident declarations, cost-per-token), the dashboards shared across the whole inference platform, and the methodology for setting / revising SLOs.

If you are the person who tunes `--max-num-batched-tokens`, you own the alerts in this chapter. If you are the person who sets the TTFT SLO policy for the organisation, that is module 110.

<!-- needs-research: exact metric names, labels, and histogram-bucket defaults are engine-version dependent (vLLM has renamed and reshaped metrics across releases; SGLang and TRT-LLM have different naming entirely). Confirm the current spelling for your engine version; the axes below are stable, the strings are not. -->

## The minimum axis set

Five axes. Every production scheduler needs all five. Each one is one of: a Prometheus counter, a Prometheus gauge, or a Prometheus histogram.

### Axis 1 — Queue depth

What: how many requests sit in the arrival (waiting) queue right now, and how long they've been there.

Metrics (vLLM-nameshape; map to your engine):

- `vllm:num_requests_waiting` — gauge, current depth.
- `vllm:num_requests_running` — gauge, current running set size (bounded by `max-num-seqs`).
- `vllm:time_in_queue_requests_seconds` — histogram, time spent waiting before first admission.

Why it matters: a queue that is persistently non-zero is the single most direct TTFT bottleneck. If depth is low (0–5) you are not admission-limited. If depth is persistently > 10× the running batch, admission is saturated.

Derived SLO: queue time P95 < some threshold (e.g. 100 ms). Queue time ≈ 0 is often achievable at light load; the alert should fire when it drifts above the threshold.

### Axis 2 — Admission rate

What: how many requests the scheduler admits into the running set per second. Compare to the arrival rate — if admission < arrival for sustained periods, the queue is growing.

Metrics:

- `vllm:request_success_total`, `vllm:request_aborted_total` — counters.
- Derived: `rate(request_success_total[1m])` as admission throughput; `rate(num_requests_waiting[1m])` as net queue delta.

Why it matters: it's the headline for how fast the engine is making forward progress against the inbound load.

### Axis 3 — Preemption rate

What: how often the scheduler evicts a running sequence to free KV.

Metrics:

- `vllm:num_preemptions_total` — counter.
- Derived: `rate(num_preemptions_total[1m]) / rate(scheduler_iterations_total[1m])` — fraction of iterations with a preemption. (If `scheduler_iterations_total` isn't exposed in your version, use requests/sec as a denominator — it's a looser proxy.)

Why it matters: directly implements chapter 04's dashboard patterns. Low: healthy. Elevated: capacity pressure. Sustained high: thrash or policy failure.

Alert levels:

- Ticket: preemption rate sustained > 0.5% for 10 minutes.
- Page: preemption rate sustained > 2% for 5 minutes.

### Axis 4 — Prefill / decode token mix

What: on each iteration, how many tokens are prefill vs. decode. Combined with iteration count, this tells you where the engine's wall-clock went.

Metrics (release-dependent; the shape is standard):

- `vllm:prompt_tokens_total` — counter of prefill tokens processed.
- `vllm:generation_tokens_total` — counter of decode tokens produced.
- Derived: `rate(prompt_tokens_total[1m]) / (rate(prompt_tokens_total[1m]) + rate(generation_tokens_total[1m]))` — prefill fraction of recent token work.
- Optional (version-dependent): a histogram of batched-tokens-per-iteration, split by phase.

Why it matters: a prefill-dominant iteration is slow for the decoders in it (chapter 01). Watching the mix tells you whether chunked prefill (chapter 03) is doing its job and lets you reason about why a cost or latency axis shifted.

### Axis 5 — Iteration time

What: wall-clock duration of one engine step. The direct TPOT driver.

Metrics:

- Not always exposed natively. Derive it from TPOT:
  - `vllm:time_per_output_token_seconds` — histogram, approximates per-iteration wall-clock when the batch is in pure-decode (one token per sequence per step).
- Or compute from `request_success_total` × average output length × TPOT = total iteration time; divide out.
- Or (if your engine supports it) explicit `vllm:scheduler_iteration_duration_seconds` or `vllm:iteration_latency_seconds` histograms.

Why it matters: for a decode-dominant batch, iteration-time ≈ TPOT. For mixed batches, iteration-time rises above TPOT during prefill-heavy steps — and that is exactly where chunked-prefill (or lack of it) shows up.

## Additional axes worth having

Not strictly in the five, but every mature instrumentation set carries them:

### KV cache utilisation

- `vllm:gpu_cache_usage_perc` — gauge, 0.0–1.0.
- `vllm:cpu_cache_usage_perc` — gauge, for swap pool.

Direct view of the resource preemption is competing for. Alert at ≥ 90% sustained; investigate if ≥ 95%.

### Prefix cache hit rate

- `vllm:prefix_cache_hits_total`, `vllm:prefix_cache_queries_total` — counters.
- Derived: `rate(prefix_cache_hits_total[5m]) / rate(prefix_cache_queries_total[5m])`.

Load-bearing for workloads with shared prefixes (chat, RAG, agentic). A cliff here cascades into higher prefill cost, longer admission times, and preemption. Module 104 goes deep.

### End-to-end request latency

- `vllm:e2e_request_latency_seconds` — histogram.
- Derived: `histogram_quantile(0.95, rate(e2e_request_latency_seconds_bucket[5m]))`.

The user-visible truth. All scheduler tuning serves this. If internal metrics look fine but E2E P95 is bad, the gap is in the API frontend layer (module 102 chapter 01's layer 1).

### GPU utilisation (from DCGM-exporter, not the engine)

- `DCGM_FI_DEV_GPU_UTIL`, `DCGM_FI_DEV_SM_ACTIVE`, `DCGM_FI_DEV_MEM_COPY_UTIL` — DCGM-exporter metrics.

Scheduler metrics tell you about work the engine *has*. DCGM tells you whether the GPU is actually busy with it. A scheduler sitting on a hot batch with DCGM reporting 20% SM-active is a signal: something downstream (worker, driver, kernel) is the limit, not the scheduler.

## Wiring metrics into Prometheus

Three pieces: the engine's `/metrics` endpoint, a scrape config, and a dashboard set.

### Engine endpoint

- vLLM: `--metrics-port <N>` or the engine args to expose Prometheus-format `/metrics`. By default on port 8000 or similar; verify your version. Example: `vllm serve ... --metrics-port 8001`, scrape `http://host:8001/metrics`.
- SGLang: the launcher exposes metrics; path and port set via server args.
- TensorRT-LLM via Triton: Triton exposes Prometheus metrics on its own port (default 8002), with the TRT-LLM backend contributing scheduler and KV stats.

### Prometheus scrape config

Minimum config:

```yaml
scrape_configs:
  - job_name: 'inference-engine'
    scrape_interval: 5s
    metrics_path: /metrics
    static_configs:
      - targets:
          - 'engine-pod-0.inference:8001'
          - 'engine-pod-1.inference:8001'
        labels:
          engine: vllm
          pool: chat-prod-us
          region: us-east-1
```

Scrape interval of 5s is reasonable for scheduler metrics (fast-moving; need high-resolution histograms). Keep it as low as you can afford without Prometheus storage pain.

**Label hygiene:** every metric should carry `pool`, `engine`, `model`, `region`, and (if multi-tenant; module 106) `tenant`. These labels let one dashboard serve the whole fleet while slicing by workload.

### Dashboard set

Five Grafana (or equivalent) dashboards:

1. **Scheduler overview** — one row per axis above. The "is the engine healthy" view.
2. **Admission / queue dynamics** — queue depth over time, arrival vs. admission rates, time-in-queue histogram, request-aborted rate.
3. **Prefill vs. decode mix** — prefill fraction, chunked-prefill chunk count distribution, iteration time by phase.
4. **Preemption and KV pressure** — preemption rate, GPU/CPU cache usage, swap-in / swap-out counts, preempted-sequences-pending gauge.
5. **SLO roll-up** — TTFT P50/P95/P99, TPOT P50/P95/P99, E2E P95, request-aborted rate, cost-per-M-tokens (feeds module 110).

Dashboard 5 is the one leadership sees. Dashboards 1–4 are the ones you look at when 5 goes red.

## Alerting

For each SLO axis, one alert rule, each with a runbook link. Keep rules small and specific; a single alert that fires on three unrelated causes is worse than three alerts that fire on their own causes.

### Example alert rules (PromQL-shaped; verify metric spellings against your engine)

```yaml
groups:
  - name: inference-scheduler-slo
    interval: 30s
    rules:

      - alert: TTFT_P95_Breach
        expr: |
          histogram_quantile(0.95,
            rate(vllm:time_to_first_token_seconds_bucket{pool="chat-prod-us"}[5m])
          ) > 0.4
        for: 5m
        labels:
          severity: page
          slo_axis: ttft
        annotations:
          summary: "TTFT P95 breach on {{ $labels.pool }}"
          runbook: "https://wiki.example.com/runbooks/ttft-regression"

      - alert: TPOT_P95_Breach
        expr: |
          histogram_quantile(0.95,
            rate(vllm:time_per_output_token_seconds_bucket{pool="chat-prod-us"}[5m])
          ) > 0.05
        for: 5m
        labels:
          severity: page
          slo_axis: tpot
        annotations:
          summary: "TPOT P95 breach on {{ $labels.pool }}"
          runbook: "https://wiki.example.com/runbooks/tpot-regression"

      - alert: Preemption_Thrash
        expr: |
          rate(vllm:num_preemptions_total[5m])
            / clamp_min(rate(vllm:request_success_total[5m]), 1) > 0.02
        for: 5m
        labels:
          severity: page
          slo_axis: preemption
        annotations:
          summary: "Preemption rate >2% of successful requests on {{ $labels.pool }}"
          runbook: "https://wiki.example.com/runbooks/preemption-thrash"

      - alert: Queue_Time_P95_Breach
        expr: |
          histogram_quantile(0.95,
            rate(vllm:time_in_queue_requests_seconds_bucket{pool="chat-prod-us"}[5m])
          ) > 0.1
        for: 5m
        labels:
          severity: page
          slo_axis: admission
        annotations:
          summary: "Admission queue backing up on {{ $labels.pool }}"
          runbook: "https://wiki.example.com/runbooks/admission-queue"

      - alert: KV_Cache_Saturation
        expr: vllm:gpu_cache_usage_perc{pool="chat-prod-us"} > 0.9
        for: 10m
        labels:
          severity: ticket
          slo_axis: kv_pressure
        annotations:
          summary: "KV cache sustained >90% on {{ $labels.pool }} — preemption risk"
          runbook: "https://wiki.example.com/runbooks/kv-saturation"
```

### Alert hygiene rules

- **One cause per alert.** `TPOT_P95_Breach` should fire on TPOT breach, not on preemption-rate breach — those have different runbooks even when they correlate.
- **`for:` guards against flapping.** 5 minutes is the minimum useful window for scheduler-level SLO alerts; faster blips are usually noise or intra-burst behaviour that smooths itself out.
- **Severity split**: page = user-facing SLO is breached now. Ticket = underlying resource is pressured but SLO is still met. A page requires acknowledgement; a ticket is reviewed in the next working-hour cycle.
- **Every alert needs a runbook link.** The link is the first thing the on-call clicks. If the runbook doesn't exist yet, authoring it is part of landing the alert.
- **Mute on-call during known drains.** Model deploys, canary rollouts (module 113), and capacity planned drains should silence relevant alerts. Automate the mute; a human remembering is error-prone.

## Burn-rate alerts (preview of module 110)

For an SLO like "TTFT P95 < 400 ms over 30 days with 99.9% availability," the raw instantaneous metric is noisy. The right abstraction is a **burn-rate alert**: alert when the current error rate is burning the monthly error budget faster than N× sustainable.

The full burn-rate methodology lives in module 110 (and the SRE methodology sits at `ai-infra-sre`, level 30). For this chapter, know that:

- Instantaneous P95 alerts (above) are useful for *fast* detection of a regression.
- Burn-rate alerts are useful for *budget-oriented* detection — i.e. "this month's budget is at risk" — and they are less sensitive to a 2-minute spike.
- Mature setups run both. The instantaneous alerts page the on-call; the burn-rate alerts shape the planning and rollout cadence.

## Verifying the alerts

An alert you have never seen fire is an alert you cannot trust. Exercise 05 asks you to:

1. Launch the scheduler with intentionally bad knobs (over-admit; under-provision KV; over-aggressive chunk size).
2. Watch each metric cross its threshold on the dashboard.
3. Confirm each alert fires with the correct severity and the runbook link lands.
4. Return the knobs to the chapter-02 setpoint and confirm the alerts resolve.

Alerts are *infrastructure*. Treat them as code: version-controlled, tested in staging, reviewed as part of PRs.

## Boundary with module 110

What this chapter owns:

- The scheduler-specific axes (queue depth, admission rate, preemption rate, prefill/decode mix, iteration time).
- The Prometheus wiring for an engine (`/metrics` endpoint, scrape config, label hygiene).
- The direct SLO alerts (TTFT, TPOT, preemption, queue time, KV saturation).

What module 110 owns:

- The SLO *methodology* — how targets are chosen, how error budgets are computed, how burn-rate alerts are structured, how incidents are declared, how error-budget policy maps to deploy cadence.
- The cross-fleet dashboards — multi-pool overviews, cost-per-M-tokens roll-ups, provider comparisons.
- The OpenTelemetry trace layer and the request-id propagation discipline across API frontend → scheduler → runner (useful for debugging specific requests, not for aggregate SLOs).
- The integration with the organisation's broader on-call / incident platform.

Rule of thumb: if the alert expression references a scheduler-internal metric, it belongs here. If it is a composite error-budget alert that aggregates multiple pools or a derived cost metric, it belongs in module 110.

## Boundary notes

- **Metric names rotate.** This chapter uses vLLM-shaped names as examples. Any time you move engines or upgrade a release, re-verify the exact spellings and dashboard templates. The axes are stable; the strings are not. [`resources.md`](resources.md) points at each engine's current metrics reference.
- **DCGM is a separate scrape**, not a scheduler metric. It lives next to the scheduler metrics on dashboards 1 and 5 but comes from `dcgm-exporter`, not the engine. Module 109 governs the GPU-fleet observability.
- **OpenTelemetry traces** are a parallel channel useful for per-request debugging; they are not a substitute for Prometheus metrics for SLO alerting. Module 110 and module 111 (API protocols) cover tracing.

## Summary

- The five minimum scheduler axes: **queue depth**, **admission rate**, **preemption rate**, **prefill/decode token mix**, **iteration time**. Plus KV utilisation, prefix cache hit rate, and E2E latency as common additions.
- Wire the engine's `/metrics` endpoint into Prometheus with hygienic labels (`pool`, `engine`, `model`, `region`, `tenant`). Scrape interval 5–15 s.
- Build five dashboards: scheduler overview, admission / queue, prefill vs. decode, preemption / KV, SLO roll-up.
- One alert per SLO axis, one cause per alert, `for:` guard against flapping, severity split (page / ticket), runbook link mandatory. Instantaneous P95 alerts for fast detection; burn-rate alerts (module 110) for budget-oriented detection.
- Verify alerts by provoking them. An unfired alert is an untrusted alert.
- Module 110 owns the SLO methodology and cross-fleet aggregation; this chapter owns the scheduler-level instrumentation and alerts that feed it.

The module exercises take you through this end-to-end — chapter 05 is the one you keep as a reusable artefact every time a new engine lands in your fleet.
