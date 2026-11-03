# Exercise 04 — Dynamic Adapter Loader with Eviction

**Time:** 4 hours · **Output:** one simulated adapter workload with Zipfian popularity, one running fleet with sized warm set, one eviction policy implementation, one measurement table, one defence of warm-set size

## Goal

Make [Chapter 04](../04-dynamic-adapter-loader-with-eviction.md) concrete by **simulating a long-tail adapter workload** (50–200 adapters, Zipfian popularity), **sizing the warm set**, **implementing an eviction policy**, and **measuring cold-start latency, hit rate, and HBM footprint**. By the end you can defend a warm-set-size decision against the measured workload.

## Problem statement

Your platform has signed up 150 customers, each with their own adapter. The fleet cannot hold 150 adapters in HBM; some must live on the shelf (host DRAM) or in cold storage. The question: which, how many, and under what policy. The deliverable is the measurement and the policy defence.

## Prerequisites

- A multi-LoRA deployment from exercise 01 (ideally expanded to 50+ adapters). If you only have 20, re-use them 2–3× under different IDs for the simulation (not strictly accurate for a real customer workload but acceptable for the exercise).
- The engine's dynamic adapter load / unload endpoints enabled (vLLM has `/v1/load_lora_adapter` and `/v1/unload_lora_adapter` in recent versions; verify). If not available, simulate by restarting the engine with different `--lora-modules` lists; much slower but exercises the same decisions.
- A workload generator (Python with `asyncio` + `httpx`, or `locust`, or similar).
- Chapter 04 of this module read.

## Requirements

Deliver five artefacts.

### Artefact A — Simulated workload with Zipfian popularity

Build a workload generator that:

- Draws from a pool of 50–200 adapters (use `exercise-01`'s registry or an expanded version).
- Assigns each adapter a popularity rank 1 … N.
- Samples the next request's adapter from a Zipfian distribution with parameter `s ∈ [0.8, 1.2]` (close to typical real-world skew).
- Sends 1 000–10 000 requests at steady 20–50 RPS.
- Records which adapter each request used.

Reference: a Zipfian PMF `P(rank k) ∝ 1/k^s`. For `s = 1.0` and N = 100, the top-10 adapters account for roughly 60 % of traffic; the top-20 for 73 %; the long tail distributes thinly across ranks 50 … 100.

Commit the generator and the generated workload trace (CSV of request → adapter mapping) so the exercise is reproducible.

### Artefact B — Warm-set sizing analytics

Before deploying, do the sizing math:

1. **Available HBM for adapters** on your GPU after base model + KV pool + reserves. Use chapter-01's byte-math.
2. **Per-adapter bytes.** Measure one adapter's actual safetensors size; verify against the formula.
3. **Zipfian popularity curve.** From the generated workload, plot cumulative popularity (y-axis: cumulative fraction of requests; x-axis: top-K adapters).
4. **Hit-rate targets.** For 90 %, 95 %, 99 % hit rate, read the required warm-set size off the curve.
5. **HBM check.** Confirm each warm-set size fits in the HBM budget.
6. **Recommendation.** Pick a warm-set size that balances HBM utilization against hit-rate target.

Deliverable: a short analysis document with the plot, the sizing table, and the recommendation.

### Artefact C — Fleet with sized warm set and eviction policy

Deploy the engine with the recommended warm-set size:

```
vllm serve <base-model> \
    --enable-lora \
    --max-loras <N_concurrent_in_batch> \
    --max-lora-rank <max_rank> \
    --max-cpu-loras <N_shelf> \
    --lora-modules <initial_pin_set> \
    --max-model-len <chosen> \
    --gpu-memory-utilization 0.90
```

The initial `--lora-modules` is the pin set — your decision on which 10–20 adapters to always keep warm (top-10 by popularity, say).

Implement one eviction policy. Two options:

**Option 1 — Rely on engine's built-in LRU + explicit pin.**

If your engine's dynamic load endpoint already evicts LRU-style, implement the pin set and let the engine manage the rest. The code is the pin set + the load / unload calls.

**Option 2 — Author a sidecar loader with a custom policy.**

A separate Python process that:

- Watches request metadata (which adapter was requested).
- Maintains a popularity counter (sliding window).
- Decides when to pre-load an adapter (if a request arrived and the adapter is not warm, trigger a load).
- Decides when to evict (adapter not used in last T seconds, warm set is full and a new adapter needs to come in).
- Calls the engine's load / unload endpoints to execute.

Pick the harder option if time permits; the simpler is acceptable for the exercise.

Document:

- The eviction policy's state (what counters it tracks).
- The policy's inputs (last-use time, use count, adapter size, priority tag).
- The decision function (pseudocode is fine).
- The audit record emitted on each load / unload.

### Artefact D — Measurement table

Replay the Artefact-A workload against the fleet from Artefact C.

Measure per request:

- Tier hit (`tier_1` warm, `tier_2` shelf-copy, `tier_3` cold-fetch, or `pin`).
- Cold-start latency decomposition (fetch_ms, load_ms, verify_ms, warm_up_ms).
- TTFT (overall).
- Which adapter.

Aggregate:

- **Hit rate per tier.** Fraction of requests served from each tier.
- **Cold-start latency distribution.** P50, P99 per tier transition.
- **TTFT distribution with and without cold-start.** Two histograms.
- **Eviction count.** Over the workload window.
- **Warm-set churn.** Number of distinct adapters that entered Tier 1 at any point.
- **Per-adapter residency time.** Average time each adapter spent in Tier 1.

Publish as a table or dashboard. Expected shape:

- Pinned adapters: 100 % Tier-1 hit, zero cold-start.
- Top-K popular adapters (above pin set): mostly Tier-1 with occasional Tier-2 misses when they age out and come back.
- Long-tail adapters: a mix of Tier-2 and Tier-3; higher cold-start cost per request.

### Artefact E — Defence of warm-set size

A short write-up answering:

- Did the chosen warm-set size hit the target hit rate (90 % / 95 % / 99 %)?
- If not, what would change? (Larger warm set? Different eviction policy? More workers for adapter affinity?)
- What is the cold-start-TTFT impact on the SLO? Specifically: how many requests pay a > 500 ms TTFT because of cold-start?
- What is the per-adapter cost (cold-start work spent per request, aggregated)?
- Would you ship this policy to production? What is the alert you would set to catch a regression?

Call out: if the warm set was too small (hit rate missed), what happened — thrashing? Long tail dominating? One-shot workloads evicting hot adapters?

## Starter guidance

- **Start with N=50 adapters, not 200.** The dynamics are already interesting at 50; scale up when the plumbing is reliable.
- **Instrument before measuring.** Add structured logging to the loader; the measurement is only as good as the records.
- **Separate pin-set load (startup) from dynamic load.** The engine's startup load is one-time; dynamic loads during the workload are the Tier-2/3 events you care about.
- **For option 2 (custom sidecar), use the engine's HTTP endpoints — don't try to inject into the engine's process.** The HTTP interface is slower but it is the interface you'd use in production.
- **Replay the same workload trace multiple times under different policies.** Enables apples-to-apples comparison.
- **Measure a *baseline*** (no custom eviction, engine LRU only) and your custom policy. Report the delta.
- **For the "long tail" cold-start measurements, consider using an actual object store (S3 / MinIO) or at least a non-HF-cache path.** If every "cold" load is actually warm in the HF cache, you're not measuring the Tier-3 cost.

## Fallbacks for constrained environments

- **Engine does not support dynamic adapter load/unload.** Simulate by restarting the engine with different `--lora-modules` lists. The measurement is cruder but the structural work (sizing, policy design) is the same.
- **Fewer than 50 adapters available.** Re-use adapters under different synthetic IDs (acknowledge the limitation in the write-up).
- **No object store available.** Use the local filesystem as "cold storage"; the cold-start cost is understated but the model holds.
- **No sliding-window counters.** A fixed counter is fine for the exercise (reset at each run); the write-up calls out the limitation.

## Acceptance criteria

1. **Artefact A** is a reproducible workload generator with a Zipfian popularity model and a committed trace.
2. **Artefact B** has the sizing analysis with popularity curve, HBM check, and defensible warm-set recommendation.
3. **Artefact C** has the fleet deployed with the sized warm set and the eviction policy running (built-in or custom).
4. **Artefact D** has per-tier hit rates, cold-start latency distributions, TTFT distributions, eviction counts, and churn metrics.
5. **Artefact E** defends the warm-set choice against the measured results, calls out any mis-sizing, and proposes an alert for regression.
6. All five artefacts live in version control; the workload trace is reproducible from a seed.

## Common failure modes

- **Zipfian parameter too small (near 0).** Workload approaches uniform; the long tail is huge; no warm set can achieve a high hit rate. Pick s ∈ [0.8, 1.2].
- **Pin set includes a one-shot adapter.** Pin set is pollution; HBM wasted. Pin only measurably hot adapters.
- **HF cache fakes Tier 3.** First "cold" load is actually a cache hit; cold-start times look suspiciously fast. Clear cache or use a non-cached path.
- **Measurement window too short.** 100 requests across 50 adapters is thin; many adapters have 0–2 requests. Run 1 000+ for stable statistics.
- **Dynamic load endpoint deadlocks under concurrent requests.** The engine serialises load calls; parallel loads become sequential. Rate-limit the loader; serialise explicitly.
- **Hit-rate math off-by-one at boundaries.** A pinned-set hit is Tier-1; a *re-hit after eviction* is Tier-2 or 3. Define the taxonomy at the start.
- **Policy churns at warm-set boundary.** Oscillation between evicting and re-loading the same adapter. Add a small hysteresis.
- **Warm set occasionally larger than HBM allows.** Transient OOM; engine misbehaves. Enforce the warm-set cap strictly.

## Keep it

The workload generator, sizing analysis template, eviction-policy sidecar (if written), and measurement harness all compose with exercise 05 (per-tenant isolation changes the eviction decisions) and feed module 110's per-adapter cost accounting. The failure modes feed module 113's adapter-rollout runbook.
