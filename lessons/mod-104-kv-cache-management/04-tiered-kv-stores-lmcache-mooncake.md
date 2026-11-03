# Chapter 04 — Tiered KV Stores with LMCache / Mooncake

> The KV cache in HBM is small, fast, and expensive. The KV cache on NVMe is huge, slow, and cheap. A tiered store lets you buy the SLO you want by placing each byte on the tier that matches its access pattern.

## Motivation

Prefix caching (chapters 02–03) retains KV in the GPU's HBM. HBM is the smallest and most expensive tier in the serving stack — a single H100 node has 80 GB per GPU, and most of that is already weights, activations, and the live-sequence working set. A chat workload with a 2 000-token system prompt + growing conversation state can accumulate hundreds of MB of cacheable KV per active conversation. Retaining millions of such conversations in HBM is a non-starter.

**Tiered KV stores** solve this. The idea is standard-issue systems engineering adapted to KV:

- Keep the hot KV in **HBM** (nanosecond-latency, GB-scale).
- Spill the warm KV to **host DRAM** (microsecond-latency over PCIe, 100 GB–few TB-scale).
- Spill the cold KV to **NVMe or a remote object store** (millisecond-latency, TB–PB-scale, cheap).

The eviction policy becomes a **tiering policy**: which blocks go to which tier, when, and on what signal. The TTFT a request sees now depends on which tier its prefix matched. A hit in HBM is free; a hit in DRAM pays PCIe transfer; a hit in NVMe pays flash-read latency; a hit in a remote store pays network RTT. A miss pays full prefill.

Two production systems anchor the chapter: **LMCache** (open source; see [`resources.md`](resources.md)) and **Mooncake** (Moonshot AI's disaggregated serving system with a KV store component; the paper is on arXiv). Both expose a similar abstraction: a tiered KV store with HBM / DRAM / disk tiers and policies for moving blocks between them.

<!-- needs-research: LMCache and Mooncake are moving targets. Specific API shapes, which backends are supported, and which engines they integrate with change frequently. Verify integration status against the current version of your chosen engine. -->

## The three tiers in operator terms

### Tier 0 — HBM

- **Where it lives:** on-GPU HBM, inside the engine's PagedAttention pool.
- **Latency to attention:** zero — the attention kernel reads directly.
- **Capacity:** tens of GB per GPU (minus weights, activations, working set).
- **Cost signature:** this is your most expensive GB; every byte of cache here is a byte that cannot host a live sequence.

### Tier 1 — Host DRAM

- **Where it lives:** CPU-attached RAM on the inference node.
- **Latency to attention:** PCIe transfer time to copy a block to HBM before the attention kernel reads it. Roughly 100 μs–few ms per MB on PCIe Gen4; faster on PCIe Gen5. (Verify your platform's bandwidth with `nvidia-smi topo -m` and a `cudaMemcpy` benchmark.)
- **Capacity:** hundreds of GB to a few TB per node, depending on config.
- **Cost signature:** DRAM is roughly an order of magnitude cheaper per GB than HBM. Cheap enough to hold the "warm" tier of cache for most workloads.
- **Tooling:** same mechanism as the chapter-01 swap-to-CPU pool, but used as a *cache*, not an *overflow buffer*.

### Tier 2 — NVMe

- **Where it lives:** local NVMe SSD on the inference node.
- **Latency to attention:** flash-read latency (tens of μs for sequential, more for random) + transfer over PCIe. Milliseconds per MB in practical terms; depends on NVMe generation and queue depth.
- **Capacity:** TB per device; tens of TB per node with multiple drives.
- **Cost signature:** NVMe is two orders of magnitude cheaper per GB than DRAM. Useful for very long-tail prefixes (per-user conversation history retained across days) or for extremely large prefix sets (massive shared document corpora).

### Tier 3 — Remote object store or distributed KV store

- **Where it lives:** cross-node — a distributed KV store (Redis-like, Mooncake-style) or an object store (S3-like) accessed over the network.
- **Latency to attention:** network RTT (millisecond-order on an in-datacenter network, tens-of-ms over a WAN) + transfer bandwidth-limited by the network link.
- **Capacity:** effectively unbounded.
- **Cost signature:** per-GB cost is comparable to NVMe or cheaper, but with a *fetch cost* that only pays for very long-tail or very cross-worker-shared KV.
- **Operational signature:** this tier is a fleet asset, not a per-node asset. Changes the mental model from "my worker's cache" to "the fleet's cache."

Not every deployment uses all four tiers. Many production systems run tier 0 + 1 (HBM + DRAM); others add NVMe for very long-tail chat; a few go all the way to a distributed KV store for truly fleet-scale sharing.

## The tiered-store abstraction

LMCache and Mooncake both expose roughly this API to the engine (the specifics differ; this is the operator-visible shape):

```
# When a prompt arrives:
hit = store.lookup(prompt_tokens)   # returns {tier, matched_prefix_length, handle}

if hit.tier == "hbm":
    # already in the GPU pool; just point the block table at it
    attach_blocks_from_hbm(hit.handle)

elif hit.tier == "dram":
    # copy from DRAM to HBM before dispatch
    blocks = allocate_hbm_blocks(hit.matched_prefix_length)
    pcie_copy(hit.handle, blocks)
    attach_blocks(blocks)

elif hit.tier == "nvme":
    # flash-read into DRAM, then PCIe-copy into HBM
    # (some implementations may read NVMe → HBM directly via GPU Direct Storage; verify)
    ...

elif hit.tier == "remote":
    # network fetch into DRAM, then PCIe-copy into HBM
    ...

else:  # miss
    # proceed with full prefill as usual
    ...

# At the end of the request / on scheduler-signalled eviction, blocks
# move down the hierarchy on policy:
store.on_evict_from_hbm(blocks)   # may push to DRAM
store.on_evict_from_dram(blocks)  # may push to NVMe
# ... etc.
```

The engine calls into this surface at admission (lookup) and at eviction (demotion). The store calls back into the engine at promotion time (when a lower-tier hit requires pulling blocks up into HBM).

### Who implements what

- **Engine side**: the PagedAttention block manager, the radix index, the attention backend that reads KV from (potentially-recently-promoted) HBM blocks.
- **Store side**: the DRAM pool, the NVMe backend, the remote-store client, the promote / demote logic, the eviction policies at each tier.

LMCache and Mooncake are the "store side." They integrate with vLLM / SGLang via a plugin or forked engine (verify current integration path; this is where release cadence matters). TRT-LLM's tiered-KV features are in its KV-cache manager (also moving; verify).

## Tier policies — demotion, promotion, eviction

The policy decisions are richer than at a single tier.

### Demotion (moving blocks down the hierarchy)

When HBM is nearing full and the eviction policy fires:

- **Default**: write the evicted blocks into DRAM, where they remain cached. Pay the PCIe cost once; future hits on this prefix skip prefill but pay another PCIe to come back up.
- **Alternative**: skip DRAM for very large / very cold entries, write directly to NVMe or the remote store. Reduces DRAM pressure for workloads whose "warm" set exceeds DRAM.

Decision factor: how big is the "warm" working set vs. DRAM capacity? If warm fits in DRAM, demote one tier at a time. If warm > DRAM, consider skipping tiers.

### Promotion (moving blocks up the hierarchy)

When a lookup hits in a lower tier:

- **Default**: promote the matched blocks to HBM for this request's attention. Keep them in HBM at least until this request finishes. After that, LRU rules.
- **Alternative**: for frequently-hit prefixes, keep them pinned in HBM. For once-hit prefixes, demote them back immediately after use (no HBM slot kept). Depends on access-frequency estimation.

### Eviction within a tier

Each tier has its own eviction policy. Typically LRU-shape at every tier, with the lower tiers having slower eviction (because capacity is larger). TTL as a conversation-state primitive still applies.

### The write-back vs. write-through decision

When blocks are first created in HBM and are prefix-cacheable (long prefix likely to be reused):

- **Write-through**: eagerly copy to DRAM at creation. Keeps the DRAM tier always up-to-date; on HBM eviction, the data is already in DRAM. Costs continuous PCIe.
- **Write-back**: only copy on eviction. Saves PCIe when the cache is churning fast enough that most entries get evicted before any lookup. Pays PCIe at eviction time, which is bursty.

Most tiered-store implementations default to write-back. Write-through is useful when PCIe is cheap relative to GPU time (a workload bottlenecked on decode rather than data movement).

## The TTFT / cost trade-off

The operator-facing cost-benefit framing. For each tier, measure:

| Tier | Lookup latency on hit | Capacity | $ / GB per month (rough) | Prefill tokens avoided per hit |
|---|---|---|---|---|
| HBM | ~0 | 10–60 GB per GPU | high | all |
| DRAM | 100 μs – few ms | 100 GB – TB per node | mid | all |
| NVMe | ms per MB | TB per device | low | all |
| Remote | ms – 10s of ms | ~unbounded | lower per-GB; non-trivial fetch cost | all |

Then:

- A miss costs full prefill. For a 2 000-token prompt at 2 000 tokens/sec of prefill throughput, that is ~1 second of prefill.
- A tier-0 hit saves ~1 second of prefill at zero cost.
- A tier-1 hit saves (1 second − PCIe transfer time). For 2 000 tokens × ~400 KB/token-of-KV-at-70B-scale ≈ 800 MB; at PCIe-Gen4 ~10 GB/s that is 80 ms transfer. Net saving: 1 s − 80 ms ≈ 920 ms. Still an order-of-magnitude TTFT win.
- A tier-2 hit saves (1 second − flash-read + PCIe). NVMe sequential read of 800 MB at ~3 GB/s is ~270 ms; still ~700 ms saved.
- A tier-3 hit over a 10 GbE link transfers 800 MB in ~700 ms; margin is tight. On a 100 GbE link, 70 ms; strong win.

The numbers above are illustrative; verify against your platform. The decision framework:

- Use **DRAM** whenever its hit rate × TTFT saving × traffic volume justifies the DRAM budget. On most prefix-caching-friendly workloads, this is cheap and strongly positive.
- Use **NVMe** when the warm set exceeds DRAM and you have NVMe headroom. Benefit depends on how long-tailed access is; for conversation state over days, strongly positive.
- Use a **remote tier** only when the warm set exceeds node-local capacity *and* the fetch cost beats full prefill. Mostly justified by cross-worker sharing (fleet-wide dedup) rather than per-worker overflow.

### A worked numerical example

A chat fleet runs 100 RPS, 70B model, 2 000-token average prompt, 50% of requests benefit from prefix caching with 1 500-token average cached prefix.

Prefill cost without caching: 100 × 2 000 = 200 000 prefill tokens/sec of raw compute demand.

With tier-0 (HBM) prefix caching at 50% hit rate on 1 500-token prefixes: 100 × 0.5 × 1 500 = 75 000 prefill tokens/sec saved. New demand: 125 000 tokens/sec. ~37% compute saved.

With tiers 0 and 1 and some of the misses actually being tier-1 hits (say 20% of previously-missing traffic now hits DRAM): extra 100 × 0.2 × 0.5 × 1 500 = 15 000 tokens/sec saved. Minus PCIe cost on those tier-1 hits — one PCIe trip per request × 1 500-token KV copy ≈ 60 ms per request, times 10 requests/sec = 600 ms/sec of PCIe = ~60% PCIe saturation. If PCIe is spare, free win. If PCIe is saturated by live-sequence KV traffic already, tiered caching trades.

The arithmetic is simple; the numbers must come from your workload. Exercise 04 walks the method.

## LMCache — the open-source abstraction

LMCache is a KV-cache management layer that integrates with vLLM (primary target) and is being integrated into other engines. The operator-facing primitives:

- A **cache server** process managing DRAM (and optionally NVMe / remote tiers).
- An **engine-side client** that intercepts the engine's KV lookups / evictions and routes to the cache server.
- A **transport** layer (shared memory for same-node, network for cross-node) that moves blocks between the engine and the cache.
- **Policies** for which blocks to store, when to demote, when to evict.

What LMCache buys you:

- A tier-1 (and optionally tier-2 / tier-3) prefix cache without touching the engine's internals.
- Cross-worker KV sharing (one engine's cached KV is accessible to another's) — a step toward fleet-wide dedup.
- A policy surface the operator can tune separately from the engine's own caching flags.

What it costs:

- Additional moving parts to operate (another process, another metric endpoint).
- Integration bugs and version-skew pain between the LMCache version and the engine version.

Verify integration status, performance claims, and feature coverage against the LMCache repo and the engine's docs. The integration surface moves.

## Mooncake — the paper and the pattern

Mooncake (Moonshot AI's serving architecture; paper on arXiv — see [`resources.md`](resources.md)) describes a disaggregated inference system with a dedicated KV cache store spanning prefill and decode workers. The key operator-visible concepts:

- **KV cache as a first-class fleet asset.** Workers do not own their KV; the store does. Workers lease blocks for the duration of a request.
- **Chunked prefill-decode disaggregation.** Prefill workers write KV to the store; decode workers read from it. This mirrors the DistServe / Splitwise topologies (module 107); Mooncake's contribution is the explicit, well-designed cache layer.
- **Tiered storage with cross-node coherence.** DRAM + NVMe + network transports; a block can be fetched from any peer.

Mooncake is less of a drop-in product (as of writing) and more of a *reference architecture* for building tiered KV at fleet scale. If you are building a disaggregated serving stack from scratch, the paper is required reading. If you are deploying vLLM-with-LMCache, Mooncake is background.

<!-- needs-research: Mooncake's open-source status and integration with mainline engines is evolving. Verify the current state of upstream integration before committing architectural work. -->

## When *not* to deploy a tiered store

- **Workloads with low prefix-cache hit rate** (< 20%). You are moving around KV nobody is reading. Overhead with no benefit.
- **PCIe-starved platforms** where the live-sequence KV traffic already saturates PCIe. Tiered caching's demote / promote cost compounds with existing traffic.
- **Deployments without a strong operator case for DRAM budget.** DRAM is "cheap" per GB but still a budget item; if your platform is memory-constrained, the DRAM cache is a tax.
- **Early-stage deployments without baseline measurements**. Deploy prefix caching in tier 0 first, measure, then layer tiers.

The right order of operations: baseline prefill cost → tier-0 caching → tier-1 caching → tier-2 if warranted → distributed tier if warranted.

## Operational signatures

### Healthy tiered store

- Per-tier hit-rate distribution is monotonic — more hits in tier 0 than tier 1, more in tier 1 than tier 2. Reversals suggest a routing or policy bug.
- Promotion latency is bounded and matches the tier's expected hardware latency. If tier-1 promotions take 500 ms instead of 80 ms, PCIe or memcpy is contending; investigate.
- Pool occupancy at each tier stabilises below cap. If any tier is at 100% sustainedly, the eviction policy is behind and admission into that tier is being dropped.

### Pathologies

- **Promotion storms**: a burst of requests hitting lower tiers triggers mass promotion into HBM, which evicts live sequence KV, which triggers preemption. Signature: HBM occupancy spikes, preemption rate spikes, TTFT P99 jumps. Mitigation: cap promotion rate, prefer eviction of cached (not live) blocks on HBM pressure.
- **Hot-spot shards**: in a distributed store, one shard holds a hot prefix and is overloaded. Signature: cross-shard latency variance. Mitigation: replicate the hot prefix.
- **Stale-cache serves**: a prefix was invalidated upstream (document changed) but the lower tier still has it. Serves wrong answers. Mitigation: version-in-prefix, chapter 03.

## Common misreadings

- **"More tiers always helps."** More tiers always *costs* — more moving parts, more pathology surface. Each tier must earn its keep with a measured hit rate × TTFT delta.
- **"NVMe is just slow DRAM."** Flash wear, write amplification, queue-depth sensitivity, SSD-firmware behaviour. Treat it as a storage system, not an extension of RAM.
- **"The remote tier is just a bigger NVMe."** Network introduces concurrency and failure modes that single-node tiers do not have. Design for fetch timeouts, retries, and partial reads.
- **"LMCache or Mooncake handles all this for me."** They handle the mechanics. The operator still owns the policy — which workloads cache, how much DRAM to budget, how to version prefixes, how to alert on tier health.
- **"Tier-1 hit is as good as tier-0."** Tier-1 hits pay PCIe cost, which can be substantial for large matched prefixes. On a PCIe-constrained platform, a tier-1 hit is less of a win than a tier-0 hit.

## Summary

- A **tiered KV store** lets you buy TTFT with cheaper memory tiers — HBM for hot, DRAM for warm, NVMe for cold, remote store for fleet-shared or very-long-tail.
- The abstraction is **lookup / promote / demote / evict**, with per-tier policies for each.
- **LMCache** is the leading open-source tiered-KV layer for vLLM and increasingly other engines. **Mooncake** is a reference architecture for fleet-scale tiered caching integrated with disaggregated serving.
- The **TTFT / cost trade-off** is per-tier arithmetic: `TTFT saved = prefill time avoided − transfer / fetch cost`. Each tier is justified only if that is positive at the measured hit rate.
- **Operational signatures** — healthy tiering is monotonic per-tier hit-rate and bounded promotion latency; pathologies are promotion storms, hotspot shards, and stale serves.
- **Order of deployment**: baseline → tier 0 (HBM, chapters 02–03) → tier 1 (DRAM) → tier 2 (NVMe) → distributed tier if warranted. Each tier must earn its budget.
- **The algorithms** — radix tree, block manager, transport layer — are kernel / systems craft. **The runtime object** — the tiered budget, the policy, the dashboard — is yours.

Chapter 05 steps back from the cache to the pool itself: given a model, a context length, and a GPU SKU, how much KV budget do you actually have, and how many concurrent sequences does that admit?
