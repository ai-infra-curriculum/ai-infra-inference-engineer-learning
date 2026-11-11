# Chapter 03 — Acceptance-Rate Monitoring: Per-Tenant, Per-Model-Pair, Per-Workload-Class

> One number — the acceptance rate `α` — decides whether speculative decoding is a 2× speedup, a wash, or a slowdown. One fleet aggregate hides the three or four workload-class curves that actually drive the signal. This chapter instruments the dimensions that matter and writes the alerts that trigger a lifecycle action.

## Motivation

A platform runs Llama-3.1-70B with a 1B draft, speculative length `k = 4`. The fleet-level dashboard reports `α = 0.69`, TPOT P50 = 24 ms, and TPOT P99 = 48 ms — healthy. Then two things happen in a week:

1. A new tenant (an internal code-review bot) starts sending high volumes of code-completion traffic. The fleet-level `α` creeps up to `0.74`; the TPOT P50 drops to 21 ms. Looks like a win.
2. An existing tenant (customer support chat) starts a prompt-engineering experiment that pushes outputs toward longer, more reasoning-shaped answers. The fleet-level `α` is still `0.74`; the fleet-level TPOT is still 21 ms. For *this tenant specifically*, their `α` has dropped from `0.70` to `0.44` and their TPOT has risen from 32 ms to 61 ms — breach of their TPOT SLO. The ticket is "customer support is slow" and the operator spends two days bisecting.

The fleet-level aggregate hid the per-tenant regression. The alert that would have caught it is a per-tenant acceptance-rate drift alert, and the dimensional slicing that makes it legible is `tenant × model_pair × workload_class`. This chapter walks the metric set, the slicing, and the alert conditions that distinguish "retrain the draft" from "the engine is broken" from "the tenant changed their workload."

By the end of it you can:

- Enumerate the specific speculative metrics each engine exposes (and the ones you must synthesise from traces because the engine doesn't expose them).
- Define `tenant`, `model_pair`, and `workload_class` as metric labels in a way that is cardinality-safe for Prometheus.
- Build the Grafana dashboard panels: fleet aggregate, per-tenant, per-model-pair, per-workload-class, with the `α` curve, the TPOT curve, and the derived "net speedup vs. non-speculative" curve.
- Write the alert set: `α` drift, draft bypass rate, per-tenant SLO breach, and the "did a release regress acceptance" post-deploy check.
- Distinguish an alert that means "draft-model lifecycle action" (chapter 04) from one that means "engine regression" (route to engine docs / performance peer) from one that means "workload changed" (route to the tenant).

Specific metric names and label-set discipline rotate between engine versions — vLLM, for example, exposes `vllm:spec_decode_draft_acceptance_rate` alongside counters like `vllm:spec_decode_num_accepted_tokens` and `vllm:spec_decode_num_draft_tokens` (see <https://docs.vllm.ai/en/latest/features/speculative_decoding/acceptance_metrics/>). SGLang and TRT-LLM each publish their own metrics references; verify names against the pinned release.

## The metric set

### What the engine should expose (verify names per engine)

| Metric (shape) | What it measures | Why it matters |
|---|---|---|
| `spec_tokens_proposed_total{...}` | Counter — tokens the drafter proposed | Denominator of acceptance rate |
| `spec_tokens_accepted_total{...}` | Counter — tokens the target accepted | Numerator of acceptance rate |
| `spec_bonus_tokens_total{...}` | Counter — "free" bonus tokens from the final target position | Full speedup accounting |
| `spec_requests_total{...}` | Counter — requests that used speculative decoding | Base rate for bypass computation |
| `spec_bypassed_total{...}` | Counter — requests that bypassed speculative (sampler incompatibility, structured output, etc.) | Catches silent fallback |
| `spec_draft_latency_seconds{...}` | Histogram — per-iteration draft overhead | Direct `c_d` measurement |
| `spec_verify_latency_seconds{...}` | Histogram — per-iteration verify latency (target forward pass under speculative) | Compare to non-speculative decode latency |
| `spec_num_accepted_per_iter{...}` | Histogram — tokens committed per iteration | Distribution matters, not just mean |
| `spec_k{...}` | Gauge — current speculative length (static or dynamic) | Correlate with acceptance; detect auto-adjust |

Not every engine exposes all nine; chapter 03 names the gap and the trace-side synthesis.

Derived metrics (always compute these in Prometheus / recording rules, never pre-compute in the engine):

```
# Acceptance rate (per aggregation window — e.g. 5m rolling)
spec_acceptance_rate = rate(spec_tokens_accepted_total[5m]) / rate(spec_tokens_proposed_total[5m])

# Mean tokens committed per iteration (speed contribution)
spec_tokens_per_iter = (rate(spec_tokens_accepted_total[5m]) + rate(spec_bonus_tokens_total[5m]))
                       / rate(spec_iterations_total[5m])

# Bypass rate
spec_bypass_rate = rate(spec_bypassed_total[5m]) / rate(spec_requests_total[5m])

# Draft-overhead ratio (vs. target verify)
spec_draft_overhead_ratio = sum(rate(spec_draft_latency_seconds_sum[5m]))
                            / sum(rate(spec_verify_latency_seconds_sum[5m]))
```

Treat these as the fleet's **first-class speculative signals**; chart them in Grafana; alert on them in Prometheus.

Not all engines expose every metric listed above — the enumerations live in each engine's metrics reference (vLLM's speculative-decoding counters, SGLang's metrics server, TRT-LLM's Triton-level metrics). For any metric the engine does not expose, synthesise from request-level traces or add a middleware shim that records it from engine callbacks.

### What the engine typically does NOT expose (synthesise from traces)

- **Per-tenant acceptance rate.** The engine's speculative counters are usually global or per-request-at-request-close. To get per-tenant, you need to tag each request with its tenant at admission and roll up the per-request counters.
- **Per-workload-class acceptance rate.** The engine doesn't know your workload taxonomy. Tag at admission (e.g. route an API header to a label; infer from prompt-shape heuristics; have the client assert it).
- **Per-model-pair acceptance rate.** If your fleet runs multiple `(target, draft)` pairs behind one router, roll up per pair.
- **Mean prompt length / mean generation length by class.** Correlation inputs for the alerts.
- **Diurnal / weekly baseline for `α`.** Baseline is not a metric; it's a reference set that lives in a recording rule, a stored baseline, or a seasonal-aware alert (chapter 03's alert section).

The synthesis pattern:

```
request_start_timestamp, tenant, model_pair, workload_class
request_end_timestamp,   tokens_in, tokens_out, tokens_accepted,
                         tokens_proposed, bonus_tokens, bypassed_flag
```

A per-request record keyed on tenant + model_pair + workload_class, written to a trace store (OTel span attributes, a Postgres append log, or Prometheus histograms with labels — careful with cardinality). From that record all the dimensional metrics derive.

## Labels — the cardinality-safe shape

Three labels dominate. Define each with cardinality discipline:

### `tenant`

- **Source.** API auth → tenant ID. One tenant per API key / token / project / organisation; the mapping is your platform's concern.
- **Cardinality.** Hundreds to tens of thousands depending on fleet. **Do not** label Prometheus metrics with every tenant unless you have a plan for the cardinality. Common patterns:
  - **Top-N tenants + "other" bucket.** Promote the top 50 tenants by QPS to explicit labels; aggregate the long tail into one `tenant="other"` series. Review quarterly.
  - **Separate per-tenant time-series out of the hot metrics path.** Record per-tenant metrics into a different TSDB with longer retention and lower sampling; keep the hot path to tier labels.
  - **Tenant *tier* instead of tenant ID.** `tenant_tier="gold"` / `silver"` / `bronze"`. Low cardinality, serves SLO alerting; per-tenant drill-down happens off-dashboard.
- **Lifetime.** Tenants come and go. Build the label registry so a disappeared tenant doesn't inflate the series carousel; set appropriate `metric_relabel_configs` or recording-rule drops.

### `model_pair`

- **Source.** The engine config — `(target_name@sha, draft_name@sha)` for draft-target, `(target_name@sha, "medusa")` for Medusa, `(target_name@sha, "eagle@sha")` for EAGLE, `(target_name@sha, "lookahead")` for Lookahead.
- **Cardinality.** Low — one fleet typically runs one or a handful of pairs.
- **Normalisation.** `model_pair` is a short slug (`llama31-70b-fp8__draft-llama32-1b-bf16`), not a full HF path. Keep the SHA in a separate label if drift tracking requires it; drop before high-cardinality dashboards.

### `workload_class`

- **Source.** Either client-asserted (header: `x-workload-class: chat | code | long_context | reasoning | tool_use | structured_output`) or server-inferred (prompt-shape heuristic: prompt length > 4000 → `long_context`; tool-call JSON schema present → `tool_use`; etc.).
- **Cardinality.** Keep to ~5–10 classes. The point of the class is that `α` differs by class in a way that is actionable — if two classes behave identically on the acceptance curve, merge them.
- **Starter taxonomy** (adjust to your fleet):
  - `chat` — open-ended conversational, medium prompt, medium generation
  - `code` — code generation / completion
  - `long_context` — prompt > `X` tokens (set `X` from your fleet's 90th percentile)
  - `reasoning` — chain-of-thought / thinking-mode traffic
  - `tool_use` — JSON tool-call traffic (if speculative isn't bypassed; chapter 01's bypass counters)
  - `structured_output` — JSON-schema-constrained
  - `embedding` — if your fleet also serves embeddings (speculative irrelevant; still a label for exclusion)

The acceptance-rate differences across these classes are often large — `α(code) > α(chat) > α(long_context) > α(reasoning)` is a common ordering, though it varies by target and draft. Measure, don't assume.

## Baselines and drift

For an alert to be legible, "drift" must be defined against a baseline. Three practical baselines:

### Baseline 1 — paired A/B at deploy time

Every `(target, draft)` promotion (chapter 04) ships with an acceptance baseline on a frozen canary traffic sample. The pair's `α(class)` curve from canary becomes the deployment-time baseline. Drift is "current `α(class)` vs. this baseline."

**Pro:** Deterministic. The baseline is tied to the artefact.
**Con:** The baseline ages. If the workload shifts, the baseline doesn't follow.

### Baseline 2 — rolling 14-day

A rolling 14-day average of `α(class)` per `(tenant, model_pair, workload_class)` tuple. Drift is "current 1-day vs. rolling 14-day."

**Pro:** Follows the workload. Noise-robust.
**Con:** Can mask a slow drift that is still real (14-day moves with it).

### Baseline 3 — seasonal

Diurnal / weekly pattern — "today's weekday 2 pm `α` vs. last 4 weeks' weekday 2 pm `α`." Needed if your fleet has a strong tenant-time-of-day pattern.

**Pro:** Catches small drifts that get smeared out of a rolling average.
**Con:** Complex to maintain; alert fatigue if poorly tuned.

Pick one or two baselines; don't try to run all three on every series. The deploy-time baseline is the minimum; the rolling baseline is useful for class-level panels; the seasonal baseline is a specialisation.

## The dashboard — minimum shape

Four panels at minimum:

### Panel 1 — Fleet aggregate speculative health

```
Panels:
  - spec_acceptance_rate (fleet-level, 5m window, with 1h and 24h moving average overlays)
  - spec_tokens_per_iter (fleet-level, with k on a second axis)
  - spec_bypass_rate (fleet-level)
  - spec_draft_overhead_ratio (fleet-level)
  - TPOT P50 / P99 (speculative vs. non-speculative A/B if applicable)
```

### Panel 2 — Per-workload-class breakdown

```
For each workload_class in [chat, code, long_context, reasoning, tool_use, ...]:
  - spec_acceptance_rate{workload_class="..."}
  - spec_tokens_per_iter{workload_class="..."}
  - TPOT P99{workload_class="..."}
  - Request rate{workload_class="..."}
Row-sorted by request rate so the dominant classes are above the fold.
```

### Panel 3 — Per-tenant table (tiered)

```
Table rows: top-N tenants by QPS, plus "other"
Columns:
  - tenant / tenant_tier
  - QPS
  - α (24h)
  - α delta vs. baseline (deploy-time)
  - TPOT P99
  - SLO status (ok / near / breach)
Sortable by α delta so regressions float to the top.
```

### Panel 4 — Per-model-pair comparison

```
If the fleet runs multiple pairs (e.g. a draft-target legacy pair and an EAGLE pair rolling in):
  - α and TPOT by model_pair
  - Request-rate by model_pair
  - Headline: "which pair wins per class"
Used during promotion windows (chapter 04).
```

Supplementary panels — draft loading log, draft weight-SHA, engine version — are for the lifecycle-triage overlay (chapter 04).

## The alert set

Six alerts form the minimum set. Each names:

- **Signal.** What series + threshold + window.
- **Trigger means.** What the operator does next (route to the lifecycle chapter 04, the engine docs, or the tenant).
- **Suppression / cardinality note.** Avoid noise.

### Alert 1 — Fleet acceptance rate drift

```
Signal:   spec_acceptance_rate (fleet, 1h) < spec_acceptance_rate (deploy-time baseline) - 0.08
          sustained for 1h
Severity: warn (if 0.08 drift); page (if 0.15 drift)
Trigger means:
  - Is a release in flight? If yes, this is a release regression → rollback (ch 04).
  - Did a tenant's traffic mix change? Check Panel 2 per-class / Panel 3 per-tenant.
  - Did the target or draft change but the pair wasn't re-baselined? → ch 04 contract violation.
  - If the drift is secular across all classes and no release correlates → the draft may be
    due for a lifecycle action (retrain against drift; ch 04).
Suppression: suppress during a planned canary; resume after promotion.
```

### Alert 2 — Per-class acceptance rate collapse

```
Signal:   spec_acceptance_rate{workload_class="X"} < 0.3 for 15m
          AND rate(spec_requests_total{workload_class="X"}[15m]) > <low_qps_floor>
Severity: warn
Trigger means:
  - This class is spending draft steps for no return. If the class is small, consider
    disabling speculative for that class (chapter 05 covers the policy).
  - If the class is large, this is a lifecycle trigger (chapter 04) OR a signal that the
    class should be routed to a different (target, draft) pair.
Suppression: by class; a new class should enter observation mode before alerting.
```

### Alert 3 — Per-tenant SLO breach via speculative regression

```
Signal:   TPOT_P99{tenant="T"} > SLO_T for 15m
          AND spec_acceptance_rate{tenant="T", 1h} < baseline - 0.1
Severity: page (if SLO breached)
Trigger means:
  - Correlation (not causation) that the tenant's TPOT regression is driven by their
    workload's acceptance collapse. Route the tenant to the "disable speculative"
    fallback (ch 05), or page the on-call to decide on per-tenant intervention.
Suppression: per-tenant; use tenant-tier for top-N table.
```

### Alert 4 — Bypass rate spike

```
Signal:   spec_bypass_rate > 0.1 for 10m
Severity: warn
Trigger means:
  - A sampler combination, a structured-output surge, or a logprobs=True surge is
    causing the engine to silently bypass speculative. The acceptance metric looks
    unchanged; the TPOT doesn't improve because speculative isn't firing.
  - Route to the API team to understand which clients are triggering the bypass; this
    is usually a client-shape change.
Suppression: by request-shape; add a second metric that labels why speculative bypassed
            if the engine exposes a reason.
```

### Alert 5 — Draft overhead ratio climb

```
Signal:   spec_draft_overhead_ratio > <threshold, typically 0.3 for small-draft deployments> for 1h
Severity: warn
Trigger means:
  - The draft step is taking more of the critical path than designed. Verify:
      - Draft weight-SHA matches the configured artefact (ch 04 pair integrity).
      - Draft is sharded as expected (TP mismatch — the draft at TP=target may be
        slower than TP=1 for a tiny model; chapter 01 names this).
      - CUDA-graph re-trace isn't firing on draft shape changes (chapter 05).
  - If baseline draft overhead was 0.08 and now reads 0.35, something structural changed.
Suppression: cross-reference with engine-version label; a release correlation points to
            engine side.
```

### Alert 6 — Post-deploy acceptance regression (release gate)

```
Signal:   spec_acceptance_rate{model_pair=current, 1h} < spec_acceptance_rate{model_pair=previous, 1h} - 0.05
          within 24h of a promotion
Severity: page (holds the deploy)
Trigger means:
  - The newly-promoted (target, draft) pair is accepting less than the previous pair.
    If the pair was baselined on canary before promotion (ch 04), this is a canary-vs-prod
    traffic divergence; investigate.
  - If the pair wasn't baselined, this is a lifecycle contract violation; roll back.
Suppression: tied to the deploy pipeline; disabled during planned ramp windows after
            the ramp completes with explicit sign-off.
```

Supplementary alerts (not in the minimum set but worth building when the fleet is mature):

- Per-engine-version acceptance regression (release-of-engine, not of model).
- Per-rank acceptance drift (if TP > 1 and the draft has sharding, one rank's logs may show divergence from the others — route to performance peer).
- `k` auto-adjust excursion (if the engine supports dynamic `k` and it drifts outside the configured range).

## Three example triage walks

### Walk 1 — Fleet aggregate `α` drops; one tenant drives it

**Observation:** Alert 1 fires. Fleet `α` dropped from `0.69` to `0.59` over 24h. Panel 1 confirms.

**Panel 3 check:** Tenant `customer_support` has `α = 0.45`, down from `0.70`. Their class mix shifted — `reasoning` traffic went from 5 % to 35 % of their volume.

**Decision tree:**

- The *fleet* draft is doing its job on everything else — Panel 2 shows other classes unchanged.
- The *customer_support* tenant's new reasoning traffic is poorly matched by the current draft.

**Action:**

- Short-term: disable speculative for `customer_support` + `reasoning` class, or lower `k` for that slice. Preserve the SLO.
- Medium-term: triage with the tenant — is the reasoning traffic a durable shift (we need a draft that covers it) or an experiment (we revert in a week)?
- Long-term (if durable): request a draft retrain against the new mix (chapter 04 hand-off).

### Walk 2 — Fleet `α` steady; TPOT regressed

**Observation:** Fleet TPOT P99 crept up from 48 ms to 61 ms over a week. Alert 1 did NOT fire; `α` is unchanged.

**Panel 1 check:** `spec_draft_overhead_ratio` crept from `0.09` to `0.27`.

**Diagnosis:** The draft's step cost has risen, not the acceptance rate. Candidates:

- Engine upgrade changed draft handling (recent release correlates? → investigate release notes).
- Batching dynamics changed (more / larger prompts in flight → draft overhead amortises worse).
- Something in the stack is forcing CUDA-graph re-trace on each iteration (chapter 05 covers).

**Action:** Alert 5 should have fired at `0.3`; verify the threshold, tighten if needed. The fix is not a draft lifecycle action — it's engine / stack. Route accordingly.

### Walk 3 — Everything looks fine on the dashboard; a tenant complains

**Observation:** Tenant `premium_chat` opens a ticket: "our p99 latency is bad." Fleet dashboard, Panel 1, Panel 2 all green.

**Panel 3 check:** `premium_chat` has `α = 0.68`, TPOT P99 = 44 ms. Fleet average. Looks fine.

**Deeper dig:** `premium_chat`'s workload includes a batch of `tool_use` requests that are *bypassing* speculative (`spec_bypassed_total{tenant="premium_chat"}` is non-zero). On those, TPOT is non-speculative — ~60 ms, above `premium_chat`'s stricter SLO.

**Diagnosis:** Alert 4 should have caught this but was suppressed below the `0.1` threshold at fleet level because the bypass is confined to one tenant's `tool_use` slice.

**Action:** The dashboard needs a per-tenant bypass rate; the alert threshold is lower per-tenant than fleet-wide. Update instrumentation; retroactively confirm.

## Metric hygiene

- **Labels are forever.** Adding a label to a metric multiplies the series. Think before adding; sunset ruthlessly.
- **Histograms for latencies; counters for events; gauges sparingly.** `spec_k` as a gauge is fine (it changes rarely). `spec_tokens_proposed_total` as a counter is fine. `spec_acceptance_rate_current` as a gauge is a smell — compute from counters, don't persist as a gauge.
- **Recording rules for derived metrics.** The acceptance rate and the tokens-per-iter are computed in Prometheus recording rules, not re-exported by the engine. Keeps the engine's exposition simple and makes rules reusable.
- **Baseline storage.** Deploy-time baselines live in a separate store (a `baselines.yaml`, a config map, a database table) with the `(model_pair, workload_class) → α_baseline` mapping. Alert queries read it; recording rules read it; dashboards read it.
- **Trace sampling.** The per-request trace that powers tenant / class dimensional roll-ups does not need to be 100 %. 1–10 % sampling for high-QPS tenants plus 100 % for sensitive tenants / top-SLO classes is a common configuration.
- **Retention.** Fleet-level: 30 days hot. Per-tenant: 7–14 days hot, longer in cold storage. Per-request traces: 24 h hot.

## The boundary to the peers

This chapter does not teach:

- **How the engine computes acceptance in the first place** — the kernel-level counter inside the fused verify. Performance peer.
- **How to drive a draft retrain** — chapter 04 writes the hand-off bundle to the fine-tuning peer; this chapter only triggers it.
- **General SRE methodology for alerts** — error budgets, on-call rotations, incident runbooks. Peer track `ai-infra-sre` (level 30); module 110 adopts the methodology for inference-specific signals.
- **The TSDB / observability stack itself** — Prometheus, Grafana, OpenTelemetry. This module uses them; module 110 covers inference-specific SLO engineering on them end-to-end.

## Summary

- Instrument speculative decoding with nine metrics (proposed / accepted / bonus / requests / bypassed / draft-latency / verify-latency / num-accepted-per-iter / `k`). Not every engine exposes all; synthesise the gap from per-request traces.
- Slice by three dimensions: `tenant`, `model_pair`, `workload_class`. Keep each cardinality-safe (tenant tier or top-N + other; model-pair as a short slug; 5–10 workload classes).
- Define baselines explicitly — deploy-time baseline is minimum; rolling 14-day is useful for class panels; seasonal is for mature fleets. Alert against the baseline, not against absolute thresholds.
- Four dashboard panels (fleet aggregate, per-class, per-tenant table, per-model-pair) and six alerts (fleet drift, per-class collapse, per-tenant SLO breach, bypass spike, draft overhead climb, post-deploy regression).
- Three triage walks: one-tenant drives fleet drop; TPOT regressed without acceptance regression; dashboard green but a tenant complains. Each has a different owner.
- Chapter 04 consumes the alerts: fleet drift → draft retrain; bypass spike → API / client; draft overhead climb → engine / performance peer. The metrics decide which boundary to escalate to.
