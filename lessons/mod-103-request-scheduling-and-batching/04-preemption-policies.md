# Chapter 04 — Preemption Policies

> The KV pool is the first resource that goes bankrupt under load. Preemption is the engine's response. The policy you pick decides whether the first symptom is a throughput cliff, a tail-latency cliff, or an ignored starving request.

## Motivation

Admission is optimistic by default: the scheduler lets a request in when there is budget *right now*, not when there is budget for the request's worst-case lifetime. Decode-only workloads make this easy — every sequence grows one KV slot per step, slowly. Mixed workloads, long outputs, and bursty arrivals make it hard: the pool was fine three iterations ago, now it's full, and a new high-priority request wants in.

Preemption is the escape hatch. The scheduler **evicts** one or more running sequences, frees their KV, and admits the new request (or keeps the batch viable). Later, the preempted sequence is restored and resumes.

The policy decision has two dimensions:

1. **How** to evict — swap the KV to CPU for later restore, or discard it and recompute the sequence from the start.
2. **Who** to evict — which running sequence loses its slot. Fairness, starvation, and priority live here.

Both dimensions have real cost. Pick wrong and the SLO cliff is quick.

As always: specific flag names, metric names, and policy sets rotate across engines and releases. The *shape* of the trade-off is stable. <!-- needs-research: vLLM's default preemption mode and metric spelling, SGLang's eviction policy, and TRT-LLM's `max_num_tokens_in_paged_kv_cache` / block-reuse semantics have shifted across releases. Confirm current defaults against the engine version your fleet runs. -->

## When preemption fires

Three triggers; two legitimate, one that signals a different bug.

### Trigger 1 — New admission wants KV that doesn't fit

The scheduler decides to admit a new request. It calls into the block manager; the block manager reports fewer free blocks than needed. The scheduler must choose: refuse the admission, or evict a running sequence to free blocks for the new one.

Legitimate: this is the pure capacity case. Preemption rate here directly reflects traffic × capacity.

### Trigger 2 — Running sequence outgrew its initial budget

A long-generating sequence has produced far more output tokens than typical. On every step it needs one more KV slot. Eventually the pool runs out even without new admissions.

Legitimate: this is the long-output tail case. More common with reasoning-model workloads (long think-traces) and agentic ones (long tool-call histories growing in-place).

### Trigger 3 — Fragmentation

Enough free *blocks* exist, but the allocator cannot find a contiguous enough region for the request's layout. In PagedAttention systems this is rare (blocks are small and tracked per-sequence) but possible under certain KV-sharing patterns.

Not quite "preemption" by itself, but often observed alongside — the symptom looks the same from the dashboard. Chapter 04 of module 102 covers fragmentation as its own failure mode.

If preemption rate correlates with "GPU cache usage percent" being near 100%, you are in trigger 1 or 2. If preemption rate is high but cache usage is modest, trigger 3 is a candidate — route it as a module-104 investigation.

## Policy dimension A — how to evict

Two canonical modes. Most engines support both and let you choose.

### Mode 1 — Swap to CPU

The evicted sequence's KV blocks are copied from GPU HBM to CPU DRAM (`swap_out`). Later, when GPU capacity frees, they are copied back (`swap_in`). The sequence resumes decode from where it left off; its KV is intact.

Operationally:

- Requires a configured swap space (vLLM `--swap-space` in GB per worker; equivalents in other engines).
- The cost is PCIe bandwidth: KV for a long sequence can be tens to hundreds of MB per layer, times many layers. For a 70B TP=8 config this is often enough that the swap round-trip takes tens to hundreds of ms per sequence.
- Pros: no wasted computation. The sequence picks up exactly where it was.
- Cons: adds PCIe traffic (contends with other transfers; can throttle at HBM-saturation moments). Requires enough CPU DRAM. The round-trip latency is a direct TTFT-restore cost.

### Mode 2 — Recompute

The evicted sequence's KV is simply freed. When re-admitted, the engine re-prefills it from the original prompt — the KV is reconstructed from scratch.

Operationally:

- No swap space needed; no PCIe cost.
- Pros: cheap when preemption is rare and sequences are short-prompt. No DRAM required for staging.
- Cons: wasted GPU work. For a 10k-token-prompt sequence, re-prefill is a lot of compute. If preemption happens repeatedly on the same sequence (thrash), the compute waste compounds.

### Picking the mode

The break-even is roughly:

> `swap_rtt_cost(sequence) ≈ recompute_cost(sequence)` at some sequence-length threshold.

For short-prompt workloads (prompt ≤ 1k tokens): recompute is usually cheaper. The prefill of a short prompt is a few milliseconds; the swap round-trip can easily exceed it.

For long-prompt workloads (prompt ≥ 4k tokens): swap is usually cheaper, assuming you have the CPU DRAM and PCIe bandwidth. Re-prefilling a 10k-token prompt costs much more than copying its KV out and back.

For long-output workloads: swap is almost always right. The KV at preemption time is prompt + many output tokens — recompute means re-prefilling the prompt *and* losing all generated output progress (depending on engine; some engines keep output tokens and recompute only the KV from them, which is similar cost).

### Engine specifics (verify current defaults in your release)

- **vLLM**: historically supports both; default has been `swap` when `--swap-space` is configured, with recompute as the fallback when the swap pool is full. Verify in your version.
- **SGLang**: policy and preference set differently; check the server-arg docs for the current policy knob.
- **TensorRT-LLM / Triton**: the TRT-LLM batch manager exposes a scheduler-policy knob (e.g. `GUARANTEED_NO_EVICT` vs. `MAX_UTILIZATION`); the KV-cache reuse mechanism interacts with eviction.

### Hybrid: swap-then-recompute

Some engines swap until the swap pool is exhausted, then fall back to recompute. If you cap swap space artificially low, you force recompute; sized generously, you get swap-only until pool size becomes the bottleneck. The vLLM `--swap-space` sizing decision is downstream of this: how much CPU DRAM is reasonable to dedicate per worker?

Rule of thumb: enough swap to hold the KV of the maximum number of sequences you expect to preempt simultaneously, at your P95 (not max) prompt+output length. In practice 8–32 GB per worker is a common range; verify against your workload's KV footprint.

## Policy dimension B — who to evict

The scheduler must pick one or more running sequences to evict. The choice interacts with fairness, SLO, and starvation.

### Common selection rules

- **LIFO (last-in-first-out)** — evict the most recently admitted. Intuition: the newcomer hasn't invested much; let the old-timers keep running. Minimises throughput loss (short investment to lose). Can starve newcomers forever under sustained load.
- **FIFO (longest-running)** — evict the oldest. Intuition: it's near-done anyway, give the slot to a newer request. Only works if "near-done" is actually true; otherwise it is a goodput disaster.
- **Lowest-priority** — evict the sequence with the lowest explicit priority (per-tenant or per-request). Needs a trust model.
- **Largest-KV-footprint** — evict the sequence whose eviction frees the most blocks. Operationally efficient; starvation-unfair.
- **Oldest-arrival with age guard** — a variant that evicts the oldest, but refuses to evict any sequence whose "age" (iterations in flight) exceeds a threshold. Prevents a long-running sequence from being perpetually victimised.

Default policies vary by engine. The *shape* that matters: whichever is default, you should know what happens to a sequence that gets preempted twice. Does the engine protect it? Does its priority rise? If not, you may be at risk of starvation.

### Starvation guards

A sequence that is preempted every time it's about to make progress is **starving**. Starvation is a correctness-adjacent bug: the client sees arbitrarily high tail latency (minutes to never), the sequence consumes resources without producing output, and the request-id trails forever.

Guards you want to see in the policy:

- **Promotion on preemption**: a preempted sequence's priority rises when it is re-admitted. Second preemption promotes further.
- **Preemption cap per sequence**: after N preemptions the sequence is pinned (will not be evicted again this run).
- **Admission freeze under thrash**: if the preemption rate exceeds a threshold, the scheduler stops admitting new requests until it drops. New arrivals wait rather than causing cascades.
- **Timeouts**: a sequence that has been running for X seconds without producing output is terminated with an explicit error, not left to starve silently.

Not every engine ships every guard. The ones to verify for your deployment: preemption cap, admission freeze, and per-tenant quota.

## Dashboard patterns and triage

Four recognisable patterns. Memorise them.

### Pattern 1 — Steady low preemption rate

Preemption rate hovers in the 0.1–0.5% of iterations range; GPU cache usage sits in 70–90%; TTFT and TPOT are on-SLO.

Interpretation: the scheduler is working. Preemption is exercised, not thrashed. This is the healthy state at the edge of capacity.

Action: none. Keep the alert threshold set (e.g. page at 2%).

### Pattern 2 — Preemption-rate spike correlated with traffic burst

Rate rises to 2–10% for a few minutes during a traffic surge, then falls back as the burst ends. Cache usage pinned at 100% during the spike.

Interpretation: capacity is right-sized to the mean, undersized to the burst. The scheduler is doing what it should (keeping the system alive) but TTFT and TPOT during the spike may breach SLO.

Action options:

- Add capacity (fleet autoscale on burst; module 109).
- Shed load at the edge (429 some requests; API gateway).
- Enable or lengthen scheduler-delay / backoff.
- Verify preemption policy picks appropriate victims (not newcomers you need).

### Pattern 3 — Swap thrash

Preemption rate sustained high; CPU memory saturated; PCIe traffic high; TTFT elevated; throughput down.

Interpretation: you are in swap-exhaustion. Swapping *out* works fine; swapping *in* takes long enough that by the time the sequence is restored, another preemption cycle fires. The system is churning.

Action: lower `max-num-seqs` or raise fleet capacity. Short-term: switch to recompute mode (the recompute cost may be lower than the swap round-trip under this regime). Longer-term: capacity planning — you cannot swap your way out of a sustained undersizing.

### Pattern 4 — Silent starvation

Preemption rate normal in aggregate; a specific long-running request vanishes from the output stream for minutes; its request-id's wait-time clock climbs; the user gives up and retries, generating a second request that may also starve.

Interpretation: the eviction-selection policy is picking the same victim repeatedly, no guard is active.

Action: enable a per-sequence preemption cap and / or priority-promotion-on-preemption. Add a timeout. Verify the request-id log shows the thrash. This is a *correctness* fix, not a tuning fix.

## Interaction with chunked prefill and prefix caching

Preemption composes with the two chapter-03 and module-104 mechanisms, with wrinkles.

### Preemption + chunked prefill

A sequence can be preempted mid-prefill. In swap mode, the partial KV is saved and restoration continues from the next chunk. In recompute mode, the whole prefill restarts — any chunk-level progress is lost. If your workload has very long prompts (10k+ tokens) and non-trivial preemption rate, recompute mode is more painful than it is on short prompts; prefer swap or raise capacity.

### Preemption + prefix caching

A preempted sequence's KV may be *partially* reusable from the prefix cache after re-admission. If the sequence shares a long system prompt with other sequences (chat workloads), prefix caching can make the recompute path cheaper than the swap path (because much of the "recompute" is actually a cache hit). This narrows the swap-vs-recompute gap for long-prompt chat workloads with warm caches. Module 104 drills prefix caching; it is a load-bearing input to this decision.

## Choosing a policy — the practical heuristic

For a brand-new deployment with no measurements yet:

1. **Mode**: start with **swap** if your prompts are long (P50 ≥ 2k tokens) and you have CPU DRAM to spare. Start with **recompute** if your prompts are consistently short (P95 ≤ 1k tokens) or you don't have swap room. If unsure, start swap; it fails less badly.
2. **Swap space**: 16 GB per worker is a reasonable neutral default for 70B-class models; verify by sizing to P95 prompt+output KV × expected concurrent preemptions.
3. **Selection**: whatever the engine default is, verify it includes a starvation guard. If not, enable one.
4. **Alerting**: page on sustained >2% preemption rate; ticket on >0.5%. Instrument swap-pool usage and recompute-count separately.

Then: measure. Preemption behavior under *your* workload is the only authoritative input. Exercise 04 drives that measurement.

## When preemption shouldn't be happening

Three cases where the right answer is not to tune preemption policy but to remove the pressure:

### Case 1 — Over-admission by misconfiguration

If `max-num-seqs` is far above the KV budget supports at P95 prompt+output length, you've asked the scheduler to admit more than can run. Preemption is permanent. Fix: lower `max-num-seqs`, raise `--gpu-memory-utilization`, or cap `--max-model-len`.

### Case 2 — Prefix cache is cold

If workloads with shared prefixes are not getting prefix-cache hits (misconfigured prompts; module 102 chapter 04's "prefix-cache cliff"), each sequence is holding the full KV when it could hold the suffix KV. Fixing the cache can relieve KV pressure dramatically.

### Case 3 — Capacity is wrong

Sometimes the right fix is "add fleet." Scheduler tuning cannot create resource that is not there. If every policy and knob is already tight and SLO is still breached, the message to leadership is "we need more capacity" — module 109 (autoscaling) and the project-level RFC (module 114) are where that conversation lives.

## Summary

- Preemption fires when the KV pool cannot admit a new or growing sequence. It is a *healthy* mechanism at low rates and a symptom at high ones.
- **Two policy dimensions**: how to evict (swap-out vs. recompute) and who to evict (LIFO / FIFO / lowest-priority / largest-footprint, with starvation guards).
- **Swap vs. recompute**: swap wins on long prompts and long outputs if swap bandwidth and CPU DRAM are sufficient. Recompute wins on short prompts, in low-DRAM environments, or when swap is thrashing.
- **Starvation is a correctness issue**: ensure the policy has at least one of preemption-cap, priority-promotion-on-preemption, or a per-request timeout.
- Four dashboard patterns to memorise: steady low rate (healthy), burst-correlated spike (capacity), swap thrash (undersized), silent starvation (unfair policy).
- Preemption composes with chunked prefill (chapter 03) and prefix caching (module 104) — a warm prefix cache narrows the swap-vs-recompute gap on long-prompt chat.
- When nothing works, the right answer is often **not more tuning** — it's lowering `max-num-seqs`, fixing the prefix cache, or adding capacity.

Chapter 05 wires the scheduler's internal state — queue depth, admission rate, preemption rate, prefill/decode token mix, iteration time — into Prometheus and defines the alerts that catch each pattern above before it reaches a user.
