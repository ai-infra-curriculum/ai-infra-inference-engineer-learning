# Chapter 02 — Scheduler Configuration for a Stated SLO

> You will be asked, more than once, "what should `--max-num-batched-tokens` be?" The honest answer is "it depends on the workload and the SLO" — but the correct answer is a defended value that you tuned against a Pareto sweep. This chapter is that method.

## Motivation

Chapter 01 established the inner loop and the three resource caps. This chapter turns them into a configuration against a *stated* workload and a *stated* SLO. The audience is you, authoring a scheduler config on a Monday morning for a workload about to go live on Friday.

The method is deliberately boring:

1. Write down the **workload shape** — RPS, prompt-length distribution, output-length distribution.
2. Write down the **SLO** — TTFT target, TPOT target, admission reject policy, cost target.
3. Compute a **first-pass config** from arithmetic.
4. **Measure** against the real workload (or a reproducible trace of it).
5. **Sweep** the two load-bearing knobs; plot the Pareto; pick a defensible point.
6. **Codify** the config as a regression test and an alertable SLO.

Tuning is a feedback loop, not a one-shot calculation. The numbers below (defaults, cost cliffs, typical values) are planning assumptions — verify against your engine version and your hardware. <!-- needs-research: specific flag defaults, metric names, and the exact iteration-time budget per hardware SKU rotate by engine version and GPU generation; the method in this chapter is stable, the numbers are not. -->

## Write down the workload shape

Four axes. If you cannot cite each one you are not ready to tune.

### Arrival rate (RPS)

Peak RPS, mean RPS, and the shape of the arrival process (Poisson, bursty, bimodal with a cron-driven peak). The 95th-percentile burst over a short window is often more load-bearing than the mean, because it is where the scheduler's admission queue overflows.

Minimum useful breakdown:

- Mean RPS over the business day.
- P95 RPS over a 1-minute window.
- Burst factor: `P95 / mean` — a value of 2–3 is routine, 5–10 means the fleet must be sized for the burst, not the mean.

### Prompt-length distribution

Not an average — a distribution. The scheduler cost of a prompt is roughly linear in length (compute per token × tokens), so a P99 prompt that is 10× the mean dominates P99 TTFT.

Minimum:

- P50, P95, P99 prompt length in tokens.
- Any multi-modal structure (chat has a long system prompt + short user turns; RAG has a long retrieved context + short query; agentic has long tool-call histories).

### Output-length distribution

Decode cost scales linearly with output length. P50 output of 100 tokens vs. P50 of 1000 tokens is a 10× difference in KV footprint per request and in decode-step count.

Minimum:

- P50, P95, P99 output length.
- Fraction of max-length (hit-max) responses — those are the ones that pin KV budget for the longest.

### Workload mix

Is this a single workload or a mix (chat + embeddings + batch jobs)? A mixed workload with wildly different prompt/output shapes is not one tuning problem; it is often two or three *separate pools* with their own scheduler configs. Chapter 04 (preemption) and module 106 (multi-tenant) go deeper; here, just name the mix and decide whether to tune one pool or split.

### Example workload spec

For the chapters below we'll use a concrete example:

```
Workload: "chat-prod-us"
Mean RPS: 60 ; P95-1min RPS: 180 (burst factor 3)
Prompt length: P50 1200, P95 4000, P99 8000 tokens (long system prompt + short user)
Output length: P50 180, P95 900, P99 1600 tokens
Hit-max fraction: 2%
Model: Llama-3.3-70B-Instruct on 8× H200 (TP=8), FP16
max-model-len cap: 16384
```

Everything that follows is tuned against this spec, as a worked example. Treat the specific numbers as scaffolding; apply the method to your workload.

## Write down the SLO

Three numbers from module 101's vocabulary, plus the admission policy.

- **TTFT P95** — the latency SLO at session start. Example: 400 ms.
- **TPOT P95** — the per-output-token latency SLO (steady-state generation). Example: 50 ms (~20 tokens/sec visible rate).
- **Admission policy** — if the arrival queue overflows, what happens? Reject with 429? Queue indefinitely? Shed to a cheaper backup?
- **Cost ceiling** — amortised GPU-hour per 1M input tokens / 1M output tokens target. Lets you say "no" to a config that hits the latency SLO but at 2× the cost target. Module 110 goes deep on cost accounting; here, keep a target in mind.

A TTFT SLO without a TPOT SLO is a trap: you can hit TTFT by over-admitting, then watch TPOT explode as every iteration drags under a huge batch. Both must be stated.

## First-pass config from arithmetic

Three load-bearing knobs: `max-num-seqs`, `max-num-batched-tokens`, and (if available) scheduler-delay. The chunked-prefill knob is chapter 03; the preemption knob is chapter 04. Pretend both are at their safe defaults for now.

### Pick `max-num-seqs`

A pure-decode step is memory-bandwidth-bound. Throughput (tokens/sec) rises roughly linearly with batch size until weight-load is amortised; past that point adding more sequences does not add throughput and *does* add KV footprint and preemption risk.

The practical heuristic: there is a "knee" where additional sequences stop helping. For 70B dense models on 8× H200 that knee is somewhere in the 64–256 range (verify empirically; the exact number depends on GPU, dtype, and attention backend). Smaller models have higher knees.

Start-of-tuning rules of thumb:

- A rule of thumb only — your empirical sweep is authoritative.
- Begin at `max-num-seqs = 128` for a dense 70B on a tensor-parallel 8× H100/H200. Begin at 256 for a 13B. Begin at 32 for a 405B TP=8 (bigger model, tighter KV budget).
- Floor: you need at least enough slots to serve the mean RPS × P95 request duration without the arrival queue filling. If mean RPS × mean duration is 60 × 10s = 600 concurrent requests but `max-num-seqs = 128`, 472 requests are always in the arrival queue and TTFT is dominated by queue wait.
- Ceiling: whatever makes the KV pool at `max-model-len × max-num-seqs × bytes-per-token` fit with headroom. On an 8× H200 fleet with 1.1 TB total HBM, after model weights (~140 GB for 70B FP16 across the TP group) and activation workspace, the KV pool is in the ~500 GB range. At FP16 with (very rough) ~160 KB per 1k-token KV for a 70B dense model (verify for your model's exact head config and dtype — numbers vary by model config), 500 GB divided by ~160 KB / 1k-token = ~3M tokens of KV budget. At `max-model-len = 16384` that's ~190 sequences at worst case. In reality nothing like 100% of sequences hit max-length, so you can push `max-num-seqs` well past 190 and let the preemption policy absorb the tail.

### Pick `max-num-batched-tokens`

This is the knob that gates the wall-clock duration of each engine step.

- **Low value (e.g. 2048)**: each step is fast and consistent. TPOT variance is tight. Long prompts take many iterations to prefill — TTFT on long prompts suffers. Throughput ceiling is lower.
- **High value (e.g. 32768)**: long prefills finish in one or two iterations — TTFT on long prompts is good. Any iteration that includes a chunky prefill is a long iteration — TPOT jitter is worse. Throughput ceiling is higher.
- **Default** varies by engine; verify. vLLM has shipped different defaults across versions; typical production values are in the 8192–16384 range for 70B-class models.

A sanity check: `max-num-batched-tokens` must be ≥ `max-num-seqs` (otherwise a pure-decode step can't even process one token per running sequence).

For our worked example (P95 prompt 4000 tokens, TTFT target 400 ms, TPOT target 50 ms), start at `max-num-batched-tokens = 8192`. Reasoning: a P95 prompt (4000 tokens) can be prefilled in one or two chunks. If chunked prefill is on with a chunk of 1024, each iteration mixes ~1024 prefill tokens + ~N decode tokens (where N ≤ `max-num-seqs`), making iteration time predictable.

### Pick scheduler-delay

If present in your engine, start it off. Scheduler-delay helps when arrival rate is bursty and the batch would otherwise be dispatched before a cohort of concurrent requests arrives, but it costs you TTFT in the common case. Only enable after measuring that the batch is consistently small (coalesce opportunity exists).

### Pick the priority / fairness policy

For a single-tenant workload: FIFO with a starvation guard is almost always fine. For multi-tenant: a fair-share policy keyed on tenant id, with per-tenant token-rate quotas. Multi-tenant depth is module 106; this chapter assumes single-tenant.

### Example first-pass config

For the chat-prod-us workload above, first pass:

```bash
vllm serve meta-llama/Llama-3.3-70B-Instruct \
    --tensor-parallel-size 8 \
    --max-model-len 16384 \
    --max-num-seqs 128 \
    --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.92 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --swap-space 16 \
    --metrics-port 8001
```

This is the *starting* point. The next three steps make it defensible.

## Measure against the real workload

A config you have not run is not a config; it's a hypothesis.

### Build a reproducible load generator

The load generator must replay your actual workload shape, not a synthetic bench with fixed prompt lengths:

- **Prompt and output lengths from your distribution**, not from the model's `max-model-len` and a uniform random. The scheduler behaves qualitatively differently on P99 prompts.
- **Arrival process** matching your production arrival shape — Poisson at mean RPS for a baseline; add bursts of 3× for 1 minute to simulate P95 bursts; add a longer sustained ramp to find the overload cliff.
- **Streaming** — if production streams, the generator must consume SSE and record first-token time (TTFT), not just response-complete time.
- **Realistic concurrency** — spawn enough concurrent clients to actually reach your target RPS.

Many engines ship a benchmarking script (vLLM's `benchmarks/` directory; `benchmark_serving.py`). Those are a good starting point; extend them to your distribution. See [`resources.md`](resources.md) for links.

### Measure the SLO axes under the first-pass config

Minimum metrics to capture:

| Axis | Metric (vLLM-nameshape; verify in your version) | What it tells you |
|---|---|---|
| TTFT P50/P95/P99 | `vllm:time_to_first_token_seconds` | Admission + prefill latency |
| TPOT P50/P95/P99 | `vllm:time_per_output_token_seconds` | Decode-step latency |
| ITL P95 | `vllm:inter_token_latency_seconds` (or derived) | Streaming jitter |
| E2E P95 | request duration | End-to-end from client view |
| Throughput | output tokens/sec, requests/sec | Goodput |
| Preemption rate | `vllm:num_preemptions_total` rate | Policy headroom |
| GPU cache usage | `vllm:gpu_cache_usage_perc` | KV budget pressure |
| Queue time P95 | `vllm:time_in_queue_requests_seconds` | Arrival-queue backup |
| Running / Waiting | `vllm:num_requests_running`, `vllm:num_requests_waiting` | Scheduler state |

### Common patterns and what they mean

- **TTFT high, TPOT fine, queue time high.** Admission is bottlenecked. Either `max-num-seqs` is too low (fleet can't fit the concurrency), or `max-num-batched-tokens` is too low (prefills take many iterations to admit). Try raising one and re-measuring.
- **TTFT fine, TPOT high and jittery.** Prefill is contaminating decode steps. Either reduce `max-num-batched-tokens`, enable chunked prefill (chapter 03), or shrink the chunk size.
- **Throughput plateaued despite more RPS.** Something is capping you. If GPU cache usage is at 100% and preemption rate is climbing, KV is the cap — reduce `max-model-len` or raise `--gpu-memory-utilization` or add fleet. If GPU is memory-bandwidth-bound at the configured batch, you're on the physical ceiling — only parallelism, quantization, or a kernel upgrade helps.
- **Preemption rate > 1% of iterations.** You're over-admitting; the policy is doing emergency work every step. Lower `max-num-seqs`, raise swap space, or revisit the preemption policy (chapter 04).

If the first-pass config meets SLO comfortably: step 5 is a cheap sweep to document the headroom. If it misses SLO: step 5 is the actual tuning work.

## Sweep the Pareto

The two most load-bearing knobs are `max-num-seqs` and `max-num-batched-tokens`. Everything else is a second-order correction.

### Design the sweep

A minimum useful sweep:

- 4 values of `max-num-seqs`: {64, 128, 192, 256}
- 4 values of `max-num-batched-tokens`: {4096, 8192, 16384, 32768}

That's 16 configs. For each one, run the same load generator at the same RPS target for enough time to get stable P95s (5–10 minutes once warmed; the first minute of each run is warmup — discard).

If 16 configs is too expensive, start with a diagonal (increase both together) and refine around the best point.

### Plot the Pareto

Two plots:

1. **TTFT P95 vs. throughput** — a point per config. The lower-left corner is latency-good + throughput-bad; the upper-right is throughput-good + latency-bad. The frontier is the Pareto front; nothing dominates it.
2. **TPOT P95 vs. throughput** — same shape; separate concern.

Overlay the SLO targets (horizontal line at TTFT 400 ms; horizontal line at TPOT 50 ms). Any config above either SLO line is disqualified. The chosen config is the highest-throughput point still below both SLO lines, with a margin.

### Pick the config

- Choose the Pareto-frontier point with a comfortable margin (SLO ≥ 20–30% of headroom). The margin absorbs drift from arrival-pattern surprises, model updates, and traffic growth. Zero-margin is zero-headroom.
- Prefer the config with lower preemption rate when two are near-equal on SLO. Preemption is latent risk.
- Prefer the config with lower `max-num-seqs` when two are near-equal — less KV footprint = more headroom for a traffic shift.

For our worked example a plausible outcome: `max-num-seqs=128`, `max-num-batched-tokens=16384`, chunked-prefill on with a chunk of 2048 (chapter 03). TTFT P95 at 320 ms, TPOT P95 at 42 ms, preemption rate 0.3%. All numbers conditional on hardware, model, kernel version, and traffic shape — never ship numbers you didn't measure.

### A note on sweep budget

A full 16-config sweep on a 70B model on an 8× H200 node is a lot of GPU-hours. Pragmatic compromises:

- Use a smaller model (7B or 13B) for the shape of the Pareto curve; verify the chosen setpoint on the real model with a shorter run.
- Run on a reserved dev node, not production — scheduler behaviour is sensitive to the hardware and software versions, so use the production image on a dev node, not a dev image on anything.
- Reuse the sweep. Author it as a reproducible script (see exercise 02) so the next model deploy doesn't start from zero.

## Codify the config and the SLO

The output of this chapter is not just a launch command — it's the SLO contract and the regression tests that defend it.

### Pin the config in a version-controlled file

```yaml
# serving-config/chat-prod-us.yaml
model: meta-llama/Llama-3.3-70B-Instruct
tp: 8
max_model_len: 16384
max_num_seqs: 128
max_num_batched_tokens: 16384
gpu_memory_utilization: 0.92
enable_prefix_caching: true
enable_chunked_prefill: true
# Reasoning:
# - max-num-seqs=128: Pareto knee for 70B dense on 8xH200 (sweep 2026-11-02)
# - max-num-batched-tokens=16384: TTFT P95 target 400ms met with 20% headroom
# - chunked prefill on; chunk size tuning in chapter 03
```

The comments are not decoration. They are the "why" the next operator reads when traffic shifts and the config needs re-tuning.

### Author the SLO alert set

For each SLO axis, one Prometheus alert, each with a runbook link:

- `TTFT_P95 > 400ms for 5m` → page.
- `TPOT_P95 > 50ms for 5m` → page.
- `Preemption rate > 1% of iterations for 10m` → ticket (not a page unless it correlates with SLO).
- `GPU cache usage > 90% for 10m` → ticket (capacity pressure, not a page unless SLO is also degrading).
- `Queue time P95 > 100ms for 5m` → page (arrival queue is backed up).

Chapter 05 wires these up end-to-end. Module 110 governs the overall SLO stack.

### Author the regression test

A nightly / weekly job that re-runs the sweep's chosen config against a fixed trace and compares TTFT / TPOT / throughput / preemption to the pinned baseline. Any drift above a threshold opens a ticket.

Without this, every library bump, engine upgrade, or hardware firmware update becomes a surprise. With it, the surprise is caught in the regression harness before it reaches production.

## When the first-pass arithmetic is wrong

Three patterns where the method above doesn't converge cleanly:

### The workload is really two workloads

A "chat" pool that has 5% of requests with 32k-token prompts and 95% with 500-token prompts cannot be tuned as one. The 32k-token requests dominate P99 TTFT and push the scheduler toward large `max-num-batched-tokens` and large KV budget; the 500-token requests want a tight batch. Split the pool.

Operationally: run two engines, route by prompt length at the API gateway. Each pool has its own config.

### Prefill cost dominates total GPU-hour

If prefill tokens outweigh decode tokens (prompt-heavy, output-light — RAG with 10k context and 100-token answers), the economics change. Normal batching reaches throughput limits because each step is spending most of its budget on prefill. Options: prefill-decode disaggregation (module 107), prefix caching (if the context repeats; module 104), or paying for more prefill-oriented capacity.

### The TTFT target is physically impossible

If the P95 prompt is 10k tokens and the TTFT target is 100 ms, no scheduler config will hit it on a 70B model on reasonable hardware — the prefill itself takes longer than 100 ms. The right move is to renegotiate the SLO with the SLO owner (modules 110, 113, 114) or to re-shape the workload (shorter prompts, pre-prefill with prefix caching).

Say so in writing. Do not spend a week tuning for a target that physics rules out.

## Summary

- A scheduler config is only defensible against a **stated workload** (RPS, prompt + output distributions, mix) and a **stated SLO** (TTFT, TPOT, admission policy, cost).
- The three first-pass arithmetic calls: `max-num-seqs` from the decode knee + KV budget, `max-num-batched-tokens` from the TTFT / TPOT trade-off on your prompt distribution, scheduler-delay typically off.
- **Measure against a reproducible load generator** that mimics the real distribution. A fixed-length benchmark lies about tail latency.
- **Sweep the Pareto** of `max-num-seqs` × `max-num-batched-tokens`; plot TTFT and TPOT vs. throughput; pick the frontier point with ≥20–30% SLO headroom and the lower preemption rate.
- **Codify the chosen config** as a version-controlled file with reasoning in comments, **a Prometheus alert per SLO axis**, and **a regression test** that re-runs the sweep against a fixed trace.
- If the first-pass does not converge, the workload may be two workloads (split the pool), prefill may dominate (disaggregation), or the SLO may be physically impossible (renegotiate).

Chapter 03 goes deep on chunked prefill — the single most consequential knob for the TTFT / TPOT trade-off, and the one whose default sometimes needs changing in production.
