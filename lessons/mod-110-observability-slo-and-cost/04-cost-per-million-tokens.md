# Chapter 04 — Cost per 1M Tokens: Amortised GPU-Hour, Prefix-Cache Credit, Speculative Credit, Cost SLO

> External pricing pages (OpenAI, Anthropic, Fireworks, Together, Baseten) quote `$ per 1M input tokens` and `$ per 1M output tokens`. If you self-host, you need to compute the same number for your own fleet, and the computation is not "cost of a GPU-hour divided by how many tokens a GPU produced." Utilisation is not 100 %, prefill and decode cost different amounts, the prefix cache makes some tokens almost free, speculative decoding changes the effective rate, and the controlplane / egress / storage / logging bill is not zero. This chapter builds the model.

## Motivation

A platform director asks: "we spend $800k / month on H100s. Fireworks will sell me the same Llama-3.1-70B at $0.90 / 1M output tokens. What is *our* cost per 1M output tokens? Should we keep self-hosting?"

A naive answer:

```
$800k / month
model: Llama-3.1-70B
fleet: 40 × H100 nodes × 8 GPUs = 320 GPUs
tokens served: ~2.5B output tokens / month (from logs)
cost per 1M output tokens = 800_000 / 2500 = $320 per 1M tokens
```

That number is almost certainly wrong — it ignores utilisation, prefill vs. decode, cache credit, and whether "tokens served" includes partial / errored streams. Three different teams will produce three different numbers for the same fleet. The right model is reproducible, auditable, and comparable to the provider's quoted number.

By the end of this chapter you can:

- Compute the amortised GPU-hour cost including idle, controlplane, egress, and storage overhead.
- Split prefill cost from decode cost on first principles.
- Apply the prefix-cache credit so cache-hit tokens are not double-counted.
- Apply the speculative-decoding credit so the acceptance rate flows into the per-token number.
- Produce a cost-per-1M-input-tokens and cost-per-1M-output-tokens number you can compare to a provider quote.
- Author a cost SLO per workload class with the same discipline as a latency SLO.

Chapter 06 then uses these numbers in the business-metric tie-in; this chapter is the arithmetic.

## The amortised GPU-hour

Start with the raw hardware bill. A fleet GPU's "cost per hour" is not the sticker price — it is the amortised cost once you include everything that scales with the fleet:

```
GPU_sticker_cost_per_hour          = on-demand, reserved, or spot rate from the provider
+ controlplane_cost_per_gpu_hour    = cluster (K8s controlplane, etcd, LB, control-node) / GPU count
+ egress_cost_per_gpu_hour          = network egress / GPU count
+ storage_cost_per_gpu_hour         = model weights, logs, snapshots / GPU count
+ observability_cost_per_gpu_hour   = Prometheus, logs, traces / GPU count
+ amortised_people_cost_per_gpu_hr  = optional: SRE / inference engineers / GPU count
=  GPU_amortised_cost_per_hour
```

Illustrative decomposition for a mid-size fleet (your numbers will differ):

```
on-demand H100 (3-year RI):      $2.00 / hr
controlplane:                    $0.05 / hr   (shared across ~500 GPUs)
egress:                          $0.10 / hr   (traffic-dependent)
storage:                         $0.03 / hr   (weights pre-staged on NVMe)
observability (metrics+traces):  $0.08 / hr   (Prometheus + Tempo + Loki)
amortised people:                $0.30 / hr   (optional; often tracked separately)
---
GPU_amortised_cost_per_hour:     $2.26 / hr   (hardware-only: $2.18 / hr)
```

Three disciplines:

- **On-demand vs. reserved vs. spot — pick the mix your fleet actually runs.** The weighted blend is what goes into the amortisation.
- **Amortise, don't allocate.** Controlplane cost is shared across all GPUs; divide, don't guess. Keep a spreadsheet or a `cost-model.yaml` with the exact divisors.
- **People cost is optional in the per-token number but required in total cost of ownership (TCO).** Chapter 06's business-case memo always includes TCO; the per-token number sometimes doesn't, to allow a clean comparison with a provider's per-token price.

## Utilisation — the fleet isn't 100 % busy

The naive formula divides cost by tokens served *as if* the GPU was always serving. A real fleet runs at 40 % – 80 % utilisation; the remaining hours are billed but produce no tokens. The utilisation-adjusted cost:

```
utilisation = 
    tokens_served_in_window 
    / (peak_tokens_per_gpu_hour × GPU_count × hours_in_window)
```

Where `peak_tokens_per_gpu_hour` is the fleet's demonstrated peak — measured during a saturated-batch benchmark, not a vendor spec sheet. Chapter 05's Pareto sweep produces this number.

The cost-per-million output tokens, utilisation-adjusted:

```
cost_per_1M_output_tokens = 
    (GPU_amortised_cost_per_hour × hours_in_window × GPU_count) 
    / (output_tokens_served_in_window / 1e6)
```

This is the correct form. The utilisation factor is hidden in the denominator — `output_tokens_served_in_window` is what the fleet *actually produced*, not what it *could have produced at 100 %*. A 40 %-utilised fleet pays 2.5× the fully-utilised cost for the same tokens.

Worked example:

```
window: 30 days = 720 hours
GPU count: 320
GPU_amortised_cost_per_hour: $2.26
total_cost = 720 × 320 × $2.26 = $520,704

output_tokens_served = 2.5 × 10^9 = 2500 M
cost_per_1M_output_tokens = 520_704 / 2500 = $208 per 1M output tokens

(If the fleet had been at 100 % utilisation producing 5.5 × 10^9 output tokens,
 the cost per 1M would have been $95 instead.)
```

The $208 number is comparable to the Fireworks $0.90 — self-hosting this fleet is ~230× more expensive than renting, given these assumptions. That is a *huge* gap; chapter 06 names why the arithmetic may be right and the conclusion (migrate everything) may be wrong (SLA, data residency, model customisation, control).

## Input tokens vs. output tokens — different physics

Input tokens are prefill; output tokens are decode. They have different costs per token because the GPU is doing different work.

### Prefill cost per token

Prefill is **compute-bound**. The forward pass over N input tokens uses the full model weights once and `O(N)` attention. For a saturated batch, prefill throughput approaches the GPU's tensor-core peak:

```
prefill_tokens_per_sec_per_gpu ≈ (model_FLOPs_per_token^-1) × GPU_peak_FLOPs × efficiency
```

For Llama-3.1-70B on H100 at TP=4, prefill throughput is ~10k – 30k tokens / sec per replica depending on prompt length and batch (verify against vLLM / SGLang benchmarks for your pin).

```
cost_per_1M_input_tokens = 
    (GPU_cost_per_hour × seconds_per_hour) 
    / (prefill_tokens_per_sec × 1e6)
```

For the example above (320 GPUs, 80 % utilised, ~20k tokens/s/GPU during prefill):

```
if 20% of GPU-time spent in prefill, averaged:
  effective_prefill_tokens_per_second = 0.2 × 320 × 20_000 = 1.28M tokens/sec

cost_per_1M_input_tokens ≈ (320 × $2.26 × 0.2) / (1.28 × 3600) 
                        ≈ $0.031 per 1M input tokens
```

Prefill is cheap *per token* because tensor cores push a lot of tokens / second when saturated. The number looks absurdly low because it is per-token, averaged over a fleet that does both prefill and decode.

### Output (decode) cost per token

Decode is **memory-bandwidth-bound**. Every token re-reads the model weights from HBM; batching helps, but the ceiling is `HBM_bandwidth / weight_size_bytes` tokens / second across the batch. For Llama-3.1-70B on H100 at TP=4, decode throughput is ~1k – 3k tokens / sec per replica (across all sequences in the batch).

```
decode_tokens_per_sec_per_gpu ≈ (model_weight_bytes^-1) × HBM_bandwidth × efficiency × batch_size
```

```
cost_per_1M_output_tokens = 
    (GPU_cost_per_hour × seconds_per_hour) 
    / (decode_tokens_per_sec × 1e6)
```

For the example (80 % utilised, ~2k tokens/s/GPU during decode, averaged):

```
if 80% of GPU-time spent in decode:
  effective_decode_tokens_per_second = 0.8 × 320 × 2000 = 512k tokens/sec

cost_per_1M_output_tokens ≈ (320 × $2.26 × 0.8) / (0.512 × 3600)
                         ≈ $0.31 per 1M output tokens
```

Decode is more expensive per token because tokens / second is 10× lower than prefill. The right order of magnitude; match against the direct amortisation above (which came out at $208 under less favourable assumptions; the delta is the utilisation and the saturated-decode assumption).

<!-- needs-research: the throughput numbers above are illustrative; measure your fleet against vLLM `benchmark_serving.py`, SGLang's benchmarks, or your production load generator. Model, hardware, parallelism, quantisation, and sequence-length distribution all materially change them. -->

Three disciplines:

- **The provider pricing you compare against is almost all decode cost.** OpenAI's `output token` price is what they bill you per output token; the prefill (input) price is typically 1/3 to 1/5 of the output price. Your input / output split should look similar.
- **Prefill-dominated workloads shift the economics.** Long-context inference is prefill-heavy; the per-token cost is dominated by `cost_per_1M_input_tokens`. Chat is decode-heavy; the per-token cost is dominated by `cost_per_1M_output_tokens`.
- **Chunked prefill blurs the split.** If your scheduler piggybacks decode onto prefill chunks (SARATHI / vLLM chunked-prefill), attribution gets messier. The engine's metrics need to track prefill-tokens separately from decode-tokens; chapter 01 names the axis.

## The prefix-cache credit

A prompt token that is served from the prefix cache (PagedAttention's automatic prefix caching, SGLang's RadixAttention, see module 104) does not consume prefill compute. It still consumes HBM for the cached KV blocks, and the request still queues, but the prefill cost is roughly zero for the cached tokens.

The credit:

```
effective_input_tokens_billed = total_input_tokens - cache_hit_tokens
cost_per_1M_input_tokens (credited) = 
    GPU_cost_prefill_only / effective_input_tokens_billed / 1e6
```

Example impact:

```
raw_input_tokens:      100M / day
cache_hit_tokens:       45M / day (45% cache hit)
effective_billed:       55M / day
effective_cost_saving:  45% on input-token cost
```

Three disciplines:

- **Cache hit ≠ free.** The KV blocks live in HBM, which has an opportunity cost (less room for in-flight sequences, lower effective batch size, module 104 details). For most workloads the net is a win; for a cache that rarely hits, the HBM cost may exceed the compute saved.
- **Attribute the credit to input, not output.** A cache hit saves prefill compute; decode still runs the same. The output-token cost does not get a cache credit.
- **Cost-SLO alert on cache-hit drop.** From chapter 02: a 60 % → 20 % cache-hit rate is a 40 % jump in effective input-token volume. If that happens overnight, the cost SLO burns through the budget quickly.

## The speculative-decoding credit

Speculative decoding (module 108) produces multiple tokens per target forward pass when the drafter is accepted. The credit shows up on the output side: for a given GPU-hour of decode, you produce more accepted tokens.

The effective decode throughput under speculative decoding:

```
effective_decode_tokens_per_step = 1 + α × k        (approximate; see module 108 chapter 01)
```

Where `α` is the acceptance rate and `k` is the speculative length. If `α = 0.7` and `k = 4`, you produce ~3.8× more tokens per step than non-speculative, minus the draft-step overhead:

```
speedup = (1 + α × k) × (c_t / (c_t + k × c_d))
       ≈ 3.8 × (1 / (1 + 4 × 0.1))      # c_d/c_t ≈ 0.1 for a representative drafter
       ≈ 2.7×
```

Which means the decode cost per output token drops by ~2.7× for that class / pair:

```
cost_per_1M_output_tokens (speculative) 
    = cost_per_1M_output_tokens (baseline) / 2.7
```

Three disciplines:

- **The credit is per-class.** Code and agent workloads often have higher `α` than long-context; the credit is not uniform. Chapter 03 of module 108 instruments `α` per class — use those labels.
- **The credit is credited to the pair.** `(target, draft)` is the unit of accounting. A different draft at the same acceptance-rate still has a different `c_d` and a different credit.
- **The credit disappears when speculation is disabled.** Dashboard the credit explicitly so when a release bypasses speculation (module 108 chapter 04 Rule 4) the finance partner isn't surprised.

## Putting it together — the cost model

A concrete `cost-model.yaml` for a fleet:

```yaml
model: llama-3.1-70b
hardware:
  gpu: h100-80gb
  gpus_per_replica: 4         # TP=4
  replicas: 80
  gpu_hour_sticker: 2.00      # from RI contract
  overhead:
    controlplane: 0.05
    egress: 0.10
    storage: 0.03
    observability: 0.08
  gpu_hour_amortised: 2.26

throughput:                   # from chapter 05 Pareto sweep, measured
  peak_prefill_tokens_per_s_per_gpu: 20000
  peak_decode_tokens_per_s_per_gpu: 2000

utilisation:
  prefill_gpu_hours_fraction: 0.20    # from a week of production data
  decode_gpu_hours_fraction:  0.80

cache_credit:
  hit_rate_by_class:
    chat: 0.45
    code: 0.55
    long_context: 0.25
    agent: 0.60

speculative_credit:
  enabled_classes: [chat, code, agent]
  acceptance_rate_by_class:
    chat: 0.68
    code: 0.75
    agent: 0.70
  k: 4
  c_d_over_c_t: 0.10

# Derived numbers (recompute daily):
derived:
  cost_per_1M_input_tokens_uncached:   0.08
  cost_per_1M_input_tokens_blended:    0.05       # weighted by cache-hit mix
  cost_per_1M_output_tokens_baseline:  0.85
  cost_per_1M_output_tokens_blended:   0.34       # weighted by spec / non-spec mix
```

Three disciplines:

- **Daily recompute.** The cost model's inputs drift (utilisation, cache hit, acceptance rate). A nightly job recomputes `derived` and writes it to the config map the dashboards read.
- **Version the model.** `cost-model.v1.yaml`, `v2.yaml`. Chapter 06's memos cite the version; "cost per 1M tokens is $X per the v3 model."
- **Keep the raw inputs auditable.** Finance partners always ask "where did this number come from." The YAML answers.

## The cost SLO

A cost SLO has the same shape as a latency SLO. Target, window, error budget, burn-rate alerts.

Example from chapter 02 repeated with the arithmetic from this chapter grounding it:

```yaml
class: chat
cost_slo:
  target:      cost_per_1M_output_tokens < $4.00
  ceiling:     cost_per_1M_output_tokens < $6.00      # soft ceiling, warn
  window:      28 days (rolling)
  error_budget_dollars: tokens_served_in_window × ($4.00 - $measured) / 1e6
  burn_rate_alert:
    - threshold: 1-day rolling cost > $5.00
      for:       2h
      severity:  warn
    - threshold: 1-day rolling cost > $6.00
      for:       15m
      severity:  page
```

The dashboard (chapter 03 dashboard 4) surfaces: measured cost / 1M tokens per class, measured vs. target, trend over 28 days, the three credit components (raw, cache credit $, spec credit $).

The weekly review ritual: did we move cost, what moved it, is the move explainable from a credit change / utilisation change, or did an engine release regress something? Chapter 06 operationalises this as a cross-functional memo.

## Reconciling with provider pricing

When comparing to a Fireworks / Together / Baseten quote:

- Their `$ per 1M output` is a *fully-loaded blended* number — their fleet's utilisation, their engineering amortisation, their margin.
- Your self-hosted number can be apples-to-apples only if you *include* overhead + utilisation + credits.
- The provider's quote is a floor on efficiency; if you cannot match it on cost, the question is whether the SLA / data-boundary / customisation / control is worth the premium. Chapter 06 and module 114 take this up.

A table that makes the comparison honest:

```
                           self-hosted    Fireworks    Together    Baseten
cost_per_1M_input_tokens      $0.05        $0.20       $0.20       $0.18
cost_per_1M_output_tokens     $0.34        $0.90       $0.88       $0.75
TTFT_P95 (chat)               380 ms       420 ms      440 ms      380 ms
data_residency                yes          partial     partial     yes
custom_adapters               yes          partial     partial     yes
SLA_credit_terms              bespoke      standard    standard    standard
```

The right conclusion from a table like this is almost never "migrate everything to the cheapest"; it is "do we accept the data-residency / customisation trade-off at X % of volume."

## Common failure modes

- **"Cost per token = bill / tokens served."** Ignores utilisation, input vs. output split, credits. Always wrong.
- **Pricing on the sticker, not the amortised rate.** On-demand sticker is 2 – 4× the reserved rate; your fleet's rate is a blend.
- **Not splitting input from output.** Prefill and decode have different costs; a single number hides the chapter-05 Pareto shape.
- **Double-counting the cache credit.** If the cache hit is applied to input *and* output, you count the saving twice.
- **Speculative credit applied to classes it's disabled on.** The config must gate the credit by class.
- **No audit trail.** When finance asks "why did cost drop 15 % this month," you cannot reconstruct the credit changes. The `cost-model.vN.yaml` plus change log is the artefact.
- **People / GPU-amortised cost double-counted with TCO.** If payroll is already tracked in TCO, don't add it into the per-token model. Chapter 06 separates.
- **Idle GPU-hours not surfaced.** Dashboard 4 should call this out explicitly — "40 % idle" is an actionable signal, not a hidden drag.

## Summary

- The amortised GPU-hour is sticker + controlplane + egress + storage + observability (+ people, optionally).
- Utilisation is captured in the denominator: tokens actually served, not tokens that *could have been* served.
- Input (prefill) tokens are compute-bound and cheap per-token; output (decode) tokens are memory-bandwidth-bound and expensive per-token. Report both.
- Prefix cache credits input cost only; attribute by class.
- Speculative decoding credits output cost by `(1 + α × k)` roughly; attribute by `(target, draft)` pair and by class.
- Author a cost SLO per class with the same discipline as a latency SLO.
- The provider-quote comparison is apples-to-apples only with all overhead / credit terms accounted for; migration decisions are then a product trade-off, not an arithmetic one.
- Chapter 05 produces the Pareto sweep that grounds the throughput numbers; chapter 06 translates these costs into a business memo.
