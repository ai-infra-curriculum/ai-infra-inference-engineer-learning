# Chapter 02 — SLO Design for Inference: Per-Workload-Class Latency, Per-Token Error Budgets, Cache and Cost SLOs

> The SLO primitives — SLI, SLO target, error budget, burn-rate alert — are the same for inference as for any service. The *units* are not. A request is a token stream, so the error-budget unit is a token, not a request. A workload is not one thing — chat, code, long-context, and batch have different latency physics, and an SLO that averages them is an SLO that masks regressions. A cache-hit rate is the thing a cost SLO *depends on*, not the thing a cost SLO *is*. This chapter translates the SRE methodology onto inference.

## Motivation

Two SLO documents, both from real inference teams, both wrong in opposite ways:

- **Document A: "P99 request latency < 2 s."** One number. The team meets it 99.95 % of the time. They also run a long-context summarisation workload where P99 is 45 s and nobody has ever complained — because their other workload is chat, and the chat P99 is 1.8 s, and chat is 95 % of traffic, so the fleet-level P99 is the chat P99. A long-context regression that doubles the summarisation P99 doesn't move the SLO at all. The SLO is masking the signal.
- **Document B: "P99 TTFT < 300 ms, P99 TPOT < 50 ms, P99 ITL < 200 ms, P99 queue time < 100 ms, P99 prefill time < 200 ms, availability > 99.95 %, cache-hit > 60 %, preemption rate < 0.5 %, cost < $3 per 1M output tokens."** Nine SLOs. The team reviews zero of them. Every week, four are red and nobody can tell if that means an outage. Alert fatigue is total; the business partner has stopped reading the report.

A good SLO document is narrower than A and much narrower than B. By the end of this chapter you can:

- Author a latency SLO per workload class (chat, code, long-context, batch, agent / tool-use) with distinct TTFT and TPOT targets.
- Define an availability SLO in token-seconds served, not in requests completed.
- Express a cache-hit SLO as a cost-SLO dependency rather than a standalone target.
- Express a cost SLO in `$ / 1M output tokens` with a per-class budget.
- Compute an error budget at per-token granularity and the multi-window multi-burn-rate alerts on top of it.
- Pick a 28-day window, a target, and a burn-rate alert pack a stakeholder will sign off on.

This chapter assumes you have read the SLO chapter of the [`ai-infra-sre`](https://github.com/ai-infra-curriculum/ai-infra-sre-learning) track or equivalent — the methodology primitives (SLI, SLO target, error budget, burn rate, multi-window multi-burn-rate alerts, see the Google SRE workbook at <https://sre.google/workbook/alerting-on-slos/>). The L30 peer teaches *what* those terms mean; this module teaches *how to apply them to inference*.

## The workload-class split

The first move is naming the workload classes. The classes are not tenants (chapter 01's cardinality rule: never label on tenant); they are *shapes*:

| Class | Prompt shape | Output shape | TTFT sensitivity | TPOT sensitivity |
|---|---|---|---|---|
| **chat** | short (100–2k tokens), conversational | short to medium (50–500 tokens), streaming | high — user watches the first token | medium — user reads at ~250 ms / token |
| **code (completion)** | medium (500–4k tokens), IDE-scraped | short (10–200 tokens), one shot | very high — must beat 500 ms or fall below tab-trigger threshold | high — latency on close completions kills UX |
| **long-context** | long (10k–200k tokens), document-centric | variable (summary / Q&A); often one response | medium — user accepts seconds for a long document | medium — the prefill dominates E2E |
| **batch** | mixed, offline | variable, bulk | low — none; the request is async | low — total throughput is the budget |
| **agent / tool-use** | medium, with tool-call interleaving | multi-turn with tool calls | high TTFT per turn — latency compounds | high TPOT per turn — same |
| **reasoning** | medium | long with "thinking budget"; may be partially hidden | medium | medium, but output tokens per request is much higher |

Three disciplines:

- **Pick 4 – 6 classes, not 20.** Each class needs its own target, its own alert, its own dashboard row. Twenty classes is a dashboard no one reads.
- **Classes are externally visible categories, not tenants.** "Chat" is a shape, not "customer X's chat workload." The point is to separate shapes whose SLOs *should* be different.
- **Classes are set at admission.** The engine doesn't know what workload class a request is; the API layer must tag it. Chapter 01 of module 111 (API design) is where the tag is introduced; chapter 03 of this module names the Prometheus label.

A representative class list for a team starting from scratch: `chat`, `code`, `long_context`, `batch`, `agent`. Reasoning gets its own class when you have a reasoning model in production.

## Per-class latency SLOs

Each class gets a TTFT target and a TPOT target. The targets are distinct numbers because the physics are distinct: TTFT is dominated by prefill compute and queueing; TPOT is dominated by decode memory bandwidth and batching. Optimising one does not automatically improve the other; a target that covers both forces an honest trade-off.

An example class SLO sheet (illustrative — your fleet's numbers are its own):

```
chat:
  TTFT_P95 < 500 ms
  TTFT_P99 < 1000 ms
  TPOT_P95 < 50 ms
  TPOT_P99 < 100 ms
  ITL_P99 < 300 ms         # jitter SLO; catches preemption / NCCL stalls
  availability > 99.9%
  window: 28 days

code:
  TTFT_P95 < 300 ms
  TTFT_P99 < 500 ms
  TPOT_P95 < 40 ms
  TPOT_P99 < 80 ms
  ITL_P99 < 250 ms
  availability > 99.9%
  window: 28 days

long_context:
  TTFT_P95 < 10 s           # prefill-dominated; long context is slow
  TTFT_P99 < 25 s
  TPOT_P95 < 60 ms
  TPOT_P99 < 120 ms
  ITL_P99 < 400 ms
  availability > 99.5%
  window: 28 days

batch:
  (no latency SLO; async workload)
  availability > 99.9%
  deadline_miss_rate < 1% (if deadlines are a contract)
  window: 28 days

agent:
  TTFT_P95 per turn < 500 ms
  TPOT_P95 per turn < 50 ms
  turns_to_first_answer_P95 < 4
  window: 28 days
```

Three design choices to argue through:

- **P95 as the alerting SLI, P99 as the diagnostic.** P99 is noisy at low QPS classes (long_context may see 500 requests / day, not 500 / second); a P99 on 500 samples is 5 samples. Alert on P95; dashboard P99 for forensic.
- **ITL_P99 is the jitter SLO.** A clean TPOT_P95 masks a 500 ms stall on every 20th token (a preemption signature). The jitter SLO catches it.
- **Long-context TTFT is in seconds, not milliseconds.** Prefill on 100k tokens takes seconds even on an H100. Setting a 500 ms TTFT target for long-context is setting a target you will miss 95 % of the time; the SLO is then useless as a signal.

## Availability — token-seconds served, not requests completed

A naive availability SLI for inference is "fraction of requests that returned a 200." That mostly works — until a request streams 100 tokens and then errors out. The request is a "failure" but the user got 100 tokens of value. The naive SLI over-counts failures for requests that partially succeeded and under-counts failures for requests that completed with mangled tokens.

The inference-shaped availability SLI is **token-seconds served over token-seconds requested**:

```
availability_SLI = sum(tokens_streamed_for_successful_response) 
                 / sum(tokens_would_have_been_streamed_at_target_pace)
```

Approximations acceptable in practice:

- `tokens_streamed_total` as a counter, incremented per emitted token.
- `tokens_requested_total` as `requests_accepted * expected_mean_output_tokens`, or as the sum of `max_tokens` requested when that's available.
- `tokens_requested_total` can be approximated by the total tokens the user asked for (`max_tokens`), understanding that the number is an upper bound (users set higher caps than they need).

An alternative, simpler SLI that works for teams starting out: *"fraction of streams that complete without error and emit at least one token."* Each cut has trade-offs; the point is to not count a 100-token partial stream as a 100 % failure.

## The cache-hit SLO — a cost-SLO dependency

A cache-hit rate target is **not** an end SLO. Product doesn't care if the cache hits 60 % or 40 %; product cares whether the request is cheap and fast. The cache-hit rate is a *leading indicator* of the cost SLO and (indirectly) of the TTFT SLO (chapter 04 covers the math).

The right shape:

- **Dashboard the cache-hit rate.** `cache_hit_rate = cache_hit_tokens / cache_attempt_tokens`, split by `workload_class`.
- **Alert on cache-hit-rate drop as a cost-SLO leading indicator.** A cache-hit rate falling from 60 % to 20 % is a cost SLO burn in slow-motion: the fleet will spend roughly 2× on prefill compute within the SLO window. See chapter 04 for the exact credit math.
- **Do not promise cache-hit rate to product.** It is an internal metric. The SLO you promise is cost per 1M tokens or TTFT; the cache is a means.

A typical operational threshold per class (illustrative):

```
chat:            cache_hit_rate > 40% (typical range 30-60%)
code:            cache_hit_rate > 50% (IDE context is highly repeated)
long_context:    cache_hit_rate > 20% (documents vary; common prefixes rare)
agent:           cache_hit_rate > 60% (tool-use prompts have common prefixes)
```

Each threshold is a *dashboard line* with an alert at the leading-indicator level; the cost SLO downstream is what you report to stakeholders.

## The cost SLO — `$ per 1M tokens`

The SLI:

```
cost_SLI_output = (GPU_cost_per_hour × GPU_hours_billed_for_decode)
                 / (output_tokens_streamed / 1e6)

cost_SLI_input  = (GPU_cost_per_hour × GPU_hours_billed_for_prefill)
                 / (input_tokens_processed / 1e6)
```

The SLO:

```
cost_output_per_1M < $target_output
cost_input_per_1M  < $target_input
```

The targets are per-class. Chapter 04 derives the numbers; the shape here is that cost is an SLO like latency is an SLO — with a target, a window, an error budget, and an alert.

Example per-class cost sheet (illustrative — your numbers depend on model size, GPU, utilisation, and provider):

```
chat:
  cost_output_per_1M < $4.00  (goal), < $6.00 (ceiling)
  cost_input_per_1M  < $0.80

code:
  cost_output_per_1M < $5.00
  cost_input_per_1M  < $1.50  (large IDE contexts)

long_context:
  cost_output_per_1M < $3.00
  cost_input_per_1M  < $2.00  (prefill is the cost line item)

batch:
  cost_output_per_1M < $2.00  (uses spot / off-peak)
```

Three disciplines:

- **Include idle time in the denominator.** A fleet with 60 % utilisation costs 1.67× more per token than a 100 %-utilised fleet of the same size. Chapter 04 details.
- **Track input and output cost separately.** Prefill and decode have different bottlenecks (compute vs. memory bandwidth); treating "cost per token" as one number hides where the money goes.
- **Report cost per 1M output tokens as the primary number.** Output tokens dominate the user-visible cost and are what external pricing pages (OpenAI, Anthropic, Fireworks) quote.

## Error budgets — per-token, not per-request

Standard SRE SLO math: a 99.9 % SLO over 28 days produces an error budget of 0.1 % of events. In an HTTP service, "events" is requests. In inference, requests are not atomic; a request produces many tokens; a fleet that serves 10k requests / sec producing 500 tokens each serves 5M tokens / sec.

The per-token error budget:

```
window = 28 days
tokens_served_in_window = sum(tokens_streamed_total) over window
SLO_target = 99.9%

error_budget_tokens = tokens_served_in_window × (1 - 0.999)
                   = tokens_served_in_window × 0.001
```

For a workload-class latency SLO like "TTFT_P95 < 500 ms":

```
failing_requests = requests where TTFT > 500 ms
# but we want to weight by the request's token value:

failing_tokens = sum(output_tokens for request r 
                      where TTFT(r) > 500 ms)

error_budget_tokens_consumed = failing_tokens
error_budget_remaining = error_budget_total - failing_tokens
```

Why this matters: a 10 % failure on 1 % of requests (long-context) is a 10 % token-weighted failure if long-context requests emit 10 × the tokens. Request-weighted error budgets hide this; token-weighted budgets expose it.

The practical version — SLO review once a week:

```
Weekly SLO review (per class):
  - error_budget_total = tokens_in_window × (1 - SLO)
  - error_budget_consumed = failing_tokens
  - burn_rate = (error_budget_consumed_this_window / window_elapsed_fraction) 
              / (error_budget_total)
```

A burn rate of `1.0` means "on track to exactly exhaust the budget by window end"; `> 1.0` is "will exceed"; `< 1.0` is "under budget."

## Multi-window multi-burn-rate alerts

The SRE workbook's canonical alerting pattern (<https://sre.google/workbook/alerting-on-slos/>) adapted to inference:

```
# 1-hour fast-burn page
expr: |
  (sum(rate(tokens_failing_latency_total{class="chat"}[1h])) 
   / sum(rate(tokens_total{class="chat"}[1h])))
  > (1 - 0.999) * 14.4   # burn rate 14.4× → consumes 2% of monthly budget in 1h
for: 2m
labels: {severity: page}

# 6-hour slow-burn page
expr: |
  (sum(rate(tokens_failing_latency_total{class="chat"}[6h])) 
   / sum(rate(tokens_total{class="chat"}[6h])))
  > (1 - 0.999) * 6     # burn rate 6× → consumes ~5% of monthly budget in 6h
for: 15m
labels: {severity: page}

# 3-day ticket (slow drift)
expr: |
  (sum(rate(tokens_failing_latency_total{class="chat"}[3d])) 
   / sum(rate(tokens_total{class="chat"}[3d])))
  > (1 - 0.999) * 1     # burn rate 1× → on track to exhaust budget
for: 2h
labels: {severity: ticket}
```

The 14.4 / 6 / 1 multipliers are from the SRE workbook's standard recipe for a 30-day SLO. Verify the exact multipliers against the workbook for your window length and your false-positive / false-negative tolerance.

<!-- needs-research: the burn-rate multipliers (14.4, 6, 1, etc.) depend on the window length and the long-window / short-window ratio you choose. The SRE workbook's canonical table assumes a 30-day window; verify for your organisation's window choice. -->

One alert per class per SLO axis is correct. One alert per *tenant* is a cardinality disaster (chapter 01's trap 1). If one tenant blows the SLO, the class-level alert catches it; if one tenant's workload is mis-routed, the burn-rate alert surfaces it as a class-level burn; per-tenant breakdown is a *dashboard*, not an alert.

## The SLO document — what to write down

A one-page SLO document per class. It is the artefact that your SRE partner, your product partner, and your finance partner all sign off on. The template:

```
# Service: {class, e.g. "chat"}

## SLIs
- TTFT: engine-admit → first-token, from histogram `ttft_seconds{class="chat"}`.
- TPOT: first-token → stream-close averaged, from histogram `tpot_seconds{class="chat"}`.
- ITL_P99: per-token jitter, from histogram `itl_seconds{class="chat"}`.
- availability: tokens-served / tokens-requested, as defined in chapter 02.
- cost: $ per 1M output tokens, from chapter 04 model.

## Targets (28-day rolling window)
- TTFT_P95 < 500 ms
- TPOT_P95 < 50 ms
- ITL_P99 < 300 ms
- availability > 99.9% token-weighted
- cost_per_1M_output_tokens < $4.00

## Error budget
- latency: 0.1% of tokens, token-weighted.
- availability: 0.1% of tokens, token-weighted.

## Alerts
- burn-rate: 1h × 14.4, 6h × 6, 3d × 1 (see chapter 02 of mod-110).
- cost: daily rollup; alert if 7-day running cost exceeds target by 10%.

## Ownership
- Primary on-call: {rotation}
- SRE partner: {owner}
- Product partner: {owner}

## Review cadence
- Weekly ops review: burn rate, dashboard walk.
- Monthly: SLO target review, cost target review.
- Quarterly: full target re-negotiation with product.

## Change log
- 2025-Q4: initial SLO.
- ...
```

Three disciplines to carry:

- **One page. No exceptions.** If it doesn't fit on one page, something is not essential.
- **Named owner on each line.** An SLO without a named human is an SLO no one reviews.
- **Change log.** When you move a target, the history tells the next on-call why. Un-documented SLO drift is how a 2 s target becomes a 5 s target over three quarters.

## Common failure modes

- **"Fleet P99 latency."** Averages across classes; masks the long-context regression. Split by class.
- **Request-weighted error budget.** 1 % of requests blown may be 10 % of tokens blown. Token-weight the SLI.
- **One cache-hit SLO for the fleet.** Each class has a different cache-hit baseline. Dashboard per class; don't promise a fleet-level number.
- **Cost SLO without idle adjustment.** A cost SLO that divides only by *billed* GPU-hours (ignoring idle) understates cost. Chapter 04 fixes it.
- **Burn-rate alerts on raw rates, not SLI rates.** `rate(ttft_slow_total[5m])` on its own fires on a traffic spike. The right form is `rate(ttft_slow[1h]) / rate(ttft_total[1h])`, which is a dimensionless burn.
- **No window in the SLO.** "99.9 % availability" without a window is unmeasurable. Always specify (28-day rolling is a sensible default).
- **Alerting on P99 for a low-QPS class.** 500 samples a day times 1 % tail = 5 samples. P95 is the alerting SLI; P99 is diagnostic.
- **SLO drift without review.** Quarterly re-negotiation is a non-optional ritual; if an SLO is always green or always red, it needs a change.

## Summary

- Pick 4 – 6 workload classes; each gets its own SLO. Classes are externally-visible shapes, not tenants.
- Each class: TTFT target, TPOT target, ITL jitter target, availability target, cost target.
- Availability is token-seconds served, not requests completed — a partial stream is partial credit, not zero credit.
- Cache-hit rate is a leading indicator for the cost SLO; dashboard it, don't promise it externally.
- Cost SLO is `$ / 1M output tokens` with a per-class target; chapter 04 derives the number.
- Error budgets are per-token, not per-request. Burn-rate alerts use the SRE workbook's multi-window multi-burn-rate pattern (14.4 / 6 / 1 for a 30-day window; verify for your window).
- One-page SLO doc per class, with named owners and a change log.
- The SLO is a contract, not a lossy summary. If the number masks a class-level regression, it's the wrong number.
