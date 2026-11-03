# Chapter 04 — Dynamic Adapter Loader and Eviction

> A thousand adapters do not fit in HBM at the same time, and they do not need to. The question is which ones are hot, which cold, which stay pinned, and what it costs to be wrong.

## Motivation

Chapter 01 showed that a few hundred LoRAs fit on one H100 at rank 16, FP16 storage. But production fleets routinely carry **thousands** of adapters — one per customer, one per fine-tune experiment, one per language variant. The fleet's HBM cannot hold them all. The warm-set-plus-cold-tier model is unavoidable: the hot subset lives in HBM, the rest live in host DRAM or on object storage, and a loader moves adapters across tiers in response to traffic.

The cost of being wrong is measurable:

- **Over-pinning** → wasted HBM, lower batch sizes, lower throughput, higher cost per inference.
- **Under-pinning** → more cold-start loads, higher TTFT on first-use requests, broken latency SLOs for customers whose adapter is not resident.
- **Wrong eviction policy** → the adapter the next request needs was evicted a second ago, round-trip through object storage inflates the first-request latency.

By the end of this chapter you can:

- Draw the three-tier adapter residency model (HBM, host DRAM, cold storage) and name what lives in each.
- Specify an eviction policy (LRU, LFU, cost-aware with cold-start-latency penalty) and defend it against a measured workload.
- Size the warm set against HBM budget, cold-start latency, and target cache-hit-rate.
- Instrument the loader with the right axes (hit-rate per tier, cold-start-latency distribution, warm-set occupancy, promotion / eviction rate).
- Compose this loader with chapter 03's router — adapter residency becomes a routing signal.

<!-- needs-research: vLLM, SGLang, and TRT-LLM each handle adapter residency differently; some expose explicit `max_cpu_loras` and a dynamic `/load_lora_adapter` endpoint, others defer to engine-internal LRU. Verify current state against your pinned engine docs. -->

## The three-tier residency model

Adapters live in one of three tiers at any moment:

```
┌──────────────────────────────────────────┐
│ Tier 1 — HBM resident (warm set)         │   10s to 100s of adapters per GPU
│   Loaded in GPU memory; usable in the    │   Fast: zero load cost at request time
│   current engine iteration.              │   Expensive: HBM is scarce
├──────────────────────────────────────────┤
│ Tier 2 — Host DRAM cache (shelf)         │   100s to 1000s per host
│   In CPU memory; usable after a          │   Medium: ~100 ms to copy to HBM
│   DRAM-to-HBM copy.                      │   Cheaper: host DRAM is plentiful
├──────────────────────────────────────────┤
│ Tier 3 — Cold storage (object store)     │   All adapters the platform knows about
│   On S3 / GCS / local disk / HuggingFace │   Slow: 500 ms–10 s to fetch and copy
│   Hub.                                   │   Essentially free (bytes-at-rest)
└──────────────────────────────────────────┘
```

The **warm set** is Tier 1. The **shelf** is Tier 2. The **cold tier** is Tier 3. Each tier has a different cold-start cost and a different capacity bound. The loader moves adapters across tiers in response to traffic.

### Capacity bounds

Chapter 01's byte-math per adapter times the number of resident adapters is the Tier-1 bound:

```
HBM available for adapters = HBM_total
                            - base_model_bytes
                            - KV_pool_bytes
                            - activations
                            - reserves

Tier 1 capacity (count) = HBM_available_for_adapters / bytes_per_adapter
```

For a 7B BF16 base, 40 GiB KV pool, 76 MiB adapters at r=16 FP16 (chapter 01's example), Tier 1 bound is ~200 adapters per H100.

Tier 2 (host DRAM) is typically much larger — tens of GiB available per host, meaning thousands of adapters at the same per-adapter cost. The engine typically exposes `--max-cpu-loras` or equivalent (verify spelling for your engine).

Tier 3 is effectively unbounded; the only limit is object-store capacity and the organisational scope of adapters you choose to serve.

## The lazy-load lifecycle

A request arrives referencing `adapter_id`. The engine (or the loader-sidecar) resolves the adapter's state:

1. **Hit in Tier 1 (HBM resident).** The engine selects the adapter's slot; the batched LoRA path (chapter 01) uses it in the next iteration. **Zero load cost.**
2. **Hit in Tier 2 (DRAM resident).** The engine must copy the adapter from host DRAM to HBM. This triggers:
   - Select a Tier-1 slot (evict an existing warm adapter if full — see "eviction policy").
   - DMA transfer DRAM → HBM. Order of 10s to 100s of ms for a 76-MiB adapter over PCIe Gen4; much faster if the engine streams the copy concurrently with the current iteration.
   - Register the adapter at its slot and update the registry.
   - The *first* request using this adapter pays this cost; subsequent requests are Tier-1 hits.
3. **Hit in Tier 3 (cold storage).** The engine (or a sidecar) must:
   - Fetch the adapter bytes from object storage. 500 ms – 10 s depending on network, object size, and whether the object is in a CDN or cold-tier S3.
   - Deserialize and validate (hash check against the registry's SHA; reject on mismatch — chapter 05's lineage check).
   - Copy to DRAM (fast, ~100 ms).
   - Copy to HBM per Tier-2 flow above.
   - Register.
   - The *first* request on this adapter pays the full cost; subsequent requests are at least Tier-2 hits.
4. **Miss (adapter unknown).** The adapter ID is not in the registry. The engine must reject or query the control plane to see if it should load this adapter at all. Rejecting is the safe default; the alternative is a security hole (anyone can request arbitrary adapters be loaded).

### Where the lazy-load logic lives

Three implementations:

- **Engine-internal.** vLLM / SGLang support in-process LoRA loading from `--lora-modules` at startup and (in some versions) a `/load_lora_adapter` endpoint for dynamic loading. The engine owns the LRU across its warm set.
- **Loader sidecar.** A separate process watches the adapter-request stream, pre-loads adapters it predicts will be used, and uses the engine's load endpoint to populate the warm set. Allows richer policy than the engine's built-in LRU.
- **Load-on-request router hook.** The router (chapter 03) checks whether the adapter is in the engine's warm set *before* routing; if not, it either (a) blocks and triggers a load, (b) routes to a different fleet that has the adapter, or (c) sends to a provider that supports arbitrary adapters.

Option (c) is interesting: Fireworks, Together, and some other providers offer "dynamic LoRA" APIs where you can upload a LoRA and serve it. On a self-hosted fleet, the cold-start cost of a never-seen adapter is 500 ms – 10 s; on a provider's shared fleet, it may be 100 ms if the provider has already warmed it. The router can account for this and route cold-first requests to the provider, warm ones to self-host — a two-tier *fleet* arrangement that mirrors the two-tier *adapter* residency.

<!-- needs-research: provider APIs for LoRA (Fireworks' "dedicated LoRA" / Together's "custom LoRA" / Anyscale) evolve rapidly. Verify current offer, pricing, cold-start characteristics against each provider's current docs. -->

## Eviction policies

When Tier 1 is full and a Tier-2-or-3 hit requires a slot, which Tier-1 adapter is evicted?

### LRU (least-recently-used)

The default. Evict the adapter that has been idle longest. Simple, performant, well-understood. Works well when traffic is **temporally local** — the adapter used N seconds ago is likely to be used again soon.

**Failure mode:** a one-shot workload (an adapter used once for a batch job) can displace a frequently-used adapter from the warm set. Even if the one-shot adapter is never seen again, the eviction has already happened.

### LFU (least-frequently-used)

Evict the adapter with the fewest total requests over a window. Works well when traffic is **spatially skewed** (Zipfian — a few adapters account for most traffic). Resistant to the one-shot workload issue.

**Failure mode:** adapters that were popular a month ago but are no longer served still hog the warm set because their total count is high. Mitigation: aging / windowed LFU.

### Cost-aware (penalty-weighted)

Each adapter's eviction cost is weighted by its **cold-start latency** (the SLO-visible cost of re-loading if evicted). Adapters that are expensive to reload (large size, cold-tier only, slow network) get an eviction penalty. The policy evicts the adapter with the lowest `(recency × frequency / reload_cost)` score — or similar.

**Failure mode:** complexity. The weighting needs to be tuned; the latency measurement must be captured and kept fresh. Worth it when adapter sizes and reload costs vary substantially.

### Priority / pin set

Independent of the above: some adapters are **pinned** — never evicted. The pin set is the set of adapters with the strictest SLOs (large customers, latency-critical workloads). Pin set + LRU over the remainder is the practical production pattern.

```yaml
adapter_registry:
  - id: "customer-7f3a-support"
    status: active
    residency_class: pinned     # never evicted
    priority: p0
  - id: "customer-7f3a-chat"
    status: active
    residency_class: hot        # eligible for warm set, LRU within
    priority: p1
  - id: "exp-eval-batch-32"
    status: active
    residency_class: cold-first # prefer not to warm; use shelf
    priority: p3
```

The engine's warm-set manager honours `residency_class` ("pinned" never evicts; "hot" LRU within the pool; "cold-first" lives in Tier 2 by default and only promotes to Tier 1 on repeated use).

## Warm-set sizing

How many adapters should Tier 1 hold? This is a trade-off between HBM used and cold-start-latency incidence.

### The three inputs

1. **HBM budget available for adapters.** Measurable from the engine's startup log (HBM remaining after base model, KV pool, reserves).
2. **Per-adapter byte count.** Measurable from the adapter files, or computed from rank × target modules × base-model dims (chapter 01's byte-math).
3. **Target warm-set hit rate.** Driven by the SLO on cold-start latency. If the SLO allows 1 % of first-request TTFTs to be degraded, then the warm set should hit on 99 % of requests.

### The sizing procedure

Analytically:

```
Tier 1 capacity (count) = HBM_available / bytes_per_adapter
                        [upper bound]

Required capacity for hit rate H = cumulative_popularity_fraction(H)
                                 × total_active_adapters
                        [workload-driven]

Final warm set = min(Tier 1 capacity upper bound,
                     required capacity for hit rate)
```

If the required capacity exceeds Tier 1 capacity, you have three options:

- Quantise the adapters (store at FP8 or INT8 — halves bytes per adapter).
- Add more workers (more aggregate HBM; smaller per-worker warm sets but adapter affinity can route popular adapters to specific workers — "adapter stickiness").
- Accept the lower hit rate and account for cold-start-TTFT budget in the SLO.

### The Zipfian intuition

Most real-world adapter traffic follows a Zipfian / power-law distribution: a few adapters are extremely hot; many are long-tail. If the top 20 adapters account for 80 % of requests, warming only the top 20 captures 80 % of traffic — a very good ROI. The sizing exercise is: measure the popularity curve, pick the hit-rate target, read off the required warm-set size.

Instrumentation to drive the sizing:

- `adapter_request_count` (per adapter, over a rolling window).
- Rank-order popularity (top-N, cumulative).
- Current warm-set occupancy (which adapters are resident).
- Hit rate per tier (`tier_1_hit`, `tier_2_hit`, `tier_3_load`).

## Cold-start cost accounting

Each tier transition is a cost. Account them:

- **Tier 2 → Tier 1 copy.** DRAM → HBM over PCIe. Order of 50–150 ms for a 76-MiB FP16 adapter over Gen4 x16; faster on Gen5, slower on Gen3. Overlap with current iteration where supported.
- **Tier 3 → Tier 2 fetch.** Object store → DRAM. Dominated by network and object store latency; order of 500 ms – 10 s. Mitigated by concurrent fetches, prefetch, local DRAM caches on each host.
- **Hash verification.** Order of 10s to 100s of ms depending on adapter size. **Non-optional** (chapter 05's signing / verification).
- **CUDA graph recompile.** If adapter use triggers a graph recompile (chapter 01's warm-up stall), add that cost. Avoid by warm-up at pin.

Account them per request:

```json
{
  "request_id": "req_abc123",
  "adapter_id": "customer-7f3a-support",
  "adapter_residency": "tier_2",
  "cold_start_ms": {
    "fetch": 0,
    "load": 95,
    "verify": 12,
    "register": 2,
    "warm_up": 0,
    "total": 109
  },
  "ttft_total_ms": 320,
  "ttft_without_cold_start_ms": 211
}
```

The split is what lets you decide where to invest optimisation.

## Composing with chapter 03's router

Adapter residency is a **routing signal**. Three composition patterns:

### 1. Adapter affinity (sticky routing)

The router prefers to send a request for adapter X to the worker that already has X in Tier 1 (or at minimum in Tier 2). Avoids unnecessary cold-start.

```python
async def pick_backend_with_adapter_affinity(request):
    candidates = backends_serving(request.model)
    # Each backend reports its warm set in its /health metadata.
    warm_backends = [b for b in candidates if request.adapter_id in b.warm_set]
    if warm_backends:
        return pick_by_load(warm_backends)
    shelf_backends = [b for b in candidates if request.adapter_id in b.shelf]
    if shelf_backends:
        return pick_by_load(shelf_backends)
    # Otherwise any backend; it will cold-start.
    return pick_by_load(candidates)
```

### 2. Cold-start deferral to a provider

For a request on an adapter whose cold-start cost on self-host would blow the SLO, route to a provider that already has the adapter warm (if commercially / contractually possible). After the request completes, pre-warm the adapter on self-host for the next request.

### 3. Pre-warm on prediction

The router (or a sidecar) observes the request pattern and pre-warms adapters it predicts will be requested — on login, on a session-start event, on time-of-day patterns. Trades HBM for lower cold-start-latency incidence.

## Instrumentation axes

The loader is observable along these axes (feeds module 110):

- `adapter_requests_total{adapter_id, tier_hit}` — counter of requests per adapter per tier that served them.
- `adapter_cold_start_latency_seconds{adapter_id, from_tier, to_tier}` — histogram of the tier-transition latencies.
- `adapter_warm_set_occupancy{worker, adapter_id}` — gauge of which adapters are resident where.
- `adapter_eviction_total{adapter_id, reason}` — counter of eviction events, labelled by reason (LRU, forced, revoked).
- `adapter_load_errors_total{adapter_id, reason}` — counter of load failures (hash mismatch, out of memory, source unreachable).

Derive the SLIs:

- Overall warm-set hit rate = `tier_1_hit / total_requests`.
- Cold-start-TTFT impact = (sum of cold-start latencies) / (number of requests). Important: TTFT distribution without cold-start vs. with.
- Per-adapter popularity, rank-ordered. The eviction policy should match the shape of this curve.
- Warm-set churn rate (evictions / minute). High churn → warm set too small *or* workload too uniform.

## Common failure modes

### Thrashing between Tier 1 and Tier 2

Two adapters A and B both exceed the Tier-1 size when co-resident. Request pattern alternates A-B-A-B. Each request forces an eviction of the other. Warm set is paying the Tier 2 → Tier 1 cost every request. Observable as: eviction count per minute >> request count per adapter per minute. Fix: increase Tier 1 size, split adapters across workers (adapter affinity on the router), or pin one of the two.

### Hot adapter evicted by one-shot

A batch job runs with a fresh adapter; LRU moves one of the hot adapters out. Next request to the evicted adapter pays cold-start. Fix: pin the hot set; use "cold-first" residency class for batch adapters (chapter 03 can tag batch traffic to inform the loader).

### Object store rate-limit on cold-start storm

A new model variant drops; the fleet simultaneously fetches dozens of adapters from object storage; the object store rate-limits; cold-starts time out. Fix: rate-limit concurrent fetches in the loader; pre-warm during the model-swap window (chapter 05 of module 105).

### Hash mismatch on verify

The adapter in object storage differs from the registry's expected SHA. The loader rejects; the request fails. Could be a legitimate corrupt artefact; could be a signed-contract violation (chapter 05). Fix: alert on hash-mismatch events; investigate as a tamper or supply-chain issue, not just a bug.

### Adapter version drift

The registry says adapter `v4` is active; the engine has `v3` loaded in HBM; the request arrives for `v4`. The engine must reload. If `v3` is still eligible for the shelf, this is a cheap swap; if the registry promotes `v4` to pinned, `v3` must be evicted. The in-flight-swap mechanics from module 105 chapter 06 apply — don't drop in-flight requests on `v3` while promoting `v4`.

### CPU shelf OOM

The shelf (`--max-cpu-loras`) grows unbounded as adapters are loaded. Host DRAM exhausts; OS kills the worker. Fix: cap the shelf explicitly; evict shelf entries with their own LRU; the shelf is not infinite.

### Pre-warm storms a cold region

The router pre-warms adapters based on session-start events; a traffic spike on new-session triggers simultaneous pre-warms on hundreds of adapters; object store spikes; TTFT on currently-serving requests regresses. Fix: rate-limit pre-warms; back off on object-store latency; prioritise live requests over pre-warm.

## The boundary to the performance peer

This chapter does not teach:

- How the DRAM → HBM copy's kernel-level efficiency depends on PCIe BAR configuration or GPUDirect.
- How to re-layout adapter weights for faster load (adapter-weight packing).
- How to overlap adapter transfer with the current iteration's forward pass.

Those are hardware / kernel questions. If your cold-start latency has a specific *transport* cost you cannot work around (DRAM-to-HBM copy is faster than theory says it should be or slower than another workload achieves), route to the performance peer with the measurement.

## Summary

- Adapter residency is a three-tier model: HBM (warm set), host DRAM (shelf), object storage (cold). Each tier has different capacity and cold-start cost.
- The lazy-load lifecycle is: Tier 1 hit → free; Tier 2 hit → ~100 ms copy; Tier 3 hit → 500 ms–10 s fetch + copy + verify.
- Eviction policy is LRU by default, LFU for Zipfian traffic, cost-aware when adapter reload cost varies. A pin set separates the "never evict" tier from the LRU pool.
- Warm-set sizing is a trade-off between HBM budget, per-adapter bytes, and target hit rate. The Zipfian intuition is that a small warm set often captures most traffic.
- Cold-start cost is accounted per request; the TTFT split (with cold-start vs. without) is where optimisation decisions get made.
- Composes with chapter 03's router: adapter residency is a routing signal (adapter affinity, cold-start deferral, pre-warm).
- Common failures are thrashing, one-shot eviction of hot adapters, cold-start storms, hash mismatches, shelf OOM. All are configuration / policy issues this chapter owns.
- Kernel-level transport efficiency routes to the performance peer.
