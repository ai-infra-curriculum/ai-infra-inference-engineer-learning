# Exercise 04 — Tiered KV Store with LMCache or Mooncake

**Time:** 4 hours · **Output:** one deployed tiered-KV stack, one per-tier hit-rate report, one TTFT-vs-tier breakdown, one cost / TTFT trade-off table, one pathology drill

## Goal

Make [Chapter 04](../04-tiered-kv-stores-lmcache-mooncake.md) concrete by **standing up a two- or three-tier KV store** on top of your engine (HBM + DRAM at minimum; add NVMe if available), **replaying a chat trace** against it, **measuring which tier each lookup lands in** and the TTFT at each tier, and **authoring a cost / TTFT trade-off table** you can use to decide whether a tier earns its budget. Also drill one pathology (promotion storm) so you recognise it.

## Problem statement

Exercise 03 maximised prefix-cache benefit in HBM alone. The next lever is to extend the cache into cheaper memory — DRAM, NVMe, or a distributed store. The cost of adding each tier is additional complexity and PCIe / NVMe / network traffic; the benefit is a larger effective cache and more hits. This exercise forces the trade-off into the open.

Pick a tiered-KV layer:

- **LMCache**: the leading open-source option; integrates with vLLM primarily. Verify current integration status for your version.
- **Mooncake**: reference-architecture-heavy; useful if you are willing to run their reference stack.
- **Engine-native tiered KV**: some engines (TRT-LLM has some; verify) expose tiered cache natively — if yours does, use it.

If none of these is practically deployable in your environment, see the fallback below.

## Prerequisites

- A working single-worker engine from previous exercises.
- Enough host DRAM to host a 2–8 GiB DRAM-tier cache (16+ GiB free host memory is comfortable).
- (Optional) An NVMe drive for the tier-2 experiment.
- The chat workload generator from exercise 03.
- Admin on the host (installing a cache process, mounting NVMe).

## Requirements

Deliver five artefacts.

### Artefact A — Deployed tiered-KV stack

Stand up the chosen tiered-KV layer alongside your engine. Document:

- The exact tiered-KV version and integration method (plugin, forked engine, sidecar process).
- The DRAM budget (bytes allocated to the DRAM tier).
- The NVMe mount point and budget, if included.
- The policy configuration — write-back vs. write-through, demotion rule, eviction policy per tier.
- The launch command and the health-check that confirms the layer is live.

Confirm the layer is actually in the request path by running a known-identical request twice and watching the second request's trace: it should log a tier-1 (DRAM) hit, not a full prefill.

### Artefact B — Per-tier hit-rate report

Replay the chat workload from exercise 03 for 10 minutes against the tiered stack. Scrape per-tier metrics every 15 seconds.

Report:

1. **Overall cache-hit rate** (across all tiers).
2. **Per-tier hit distribution**: of all hits, what fraction landed in HBM vs. DRAM vs. NVMe vs. miss?
3. **Promotion traffic**: how many blocks were pulled from DRAM / NVMe up into HBM during the run? What was the total bytes moved?
4. **Demotion traffic**: how many blocks were pushed from HBM into DRAM? From DRAM into NVMe?
5. **Pool occupancy at each tier**, averaged over the run.

Narrative (≤ 300 words) interpreting the shape of the per-tier distribution. Monotonic (chapter 04 says: expect more hits in higher tiers)? If not, why?

### Artefact C — TTFT-vs-tier breakdown

Correlate each request's TTFT with the tier it hit. Produce:

- A histogram or box plot of TTFT with four series: `tier-0 hit`, `tier-1 hit`, `tier-2 hit` (if applicable), `miss`.
- Report P50, P95, P99 TTFT for each series.
- Compute the delta vs. miss for each tier. This is the "benefit per tier."

For a tier-1 hit at your platform's PCIe bandwidth, back-of-envelope the expected transfer time (`matched_bytes / PCIe_bandwidth`). Compare to the measured delta between tier-0 and tier-1 TTFT. The two should agree within a factor of 2; if they disagree by an order of magnitude, something is contending (concurrent live-sequence KV traffic, kernel-launch bottleneck).

### Artefact D — Cost / TTFT trade-off table

A one-page table whose rows are tier configurations and columns are cost and TTFT:

| Config | DRAM budget | NVMe budget | Hit rate (tier-0 / 1 / 2 / miss) | TTFT P99 | Approx. $ per month (DRAM $ + NVMe $ + overhead) |
|---|---|---|---|---|---|
| HBM only (exercise 03 baseline) | 0 | 0 | ... | ... | ... |
| HBM + 8 GiB DRAM | 8 GiB | 0 | ... | ... | ... |
| HBM + 32 GiB DRAM | 32 GiB | 0 | ... | ... | ... |
| HBM + 8 GiB DRAM + 100 GiB NVMe | 8 GiB | 100 GiB | ... | ... | ... |

Use rough cloud pricing for the cost column (DRAM-per-GiB-month and NVMe-per-GiB-month from your cloud's instance pricing; order-of-magnitude is fine). State the pricing source.

Narrative (≤ 300 words) picks a config as the production candidate and defends it. "HBM + 32 GiB DRAM wins because [measured delta between X and Y configurations]." If NVMe is included, defend whether its hit rate × TTFT delta justifies the budget.

### Artefact E — Promotion-storm pathology drill

Deliberately trigger a promotion storm:

1. Fill the DRAM tier with cached prefixes (replay the chat workload for 20 minutes).
2. Restart the engine process but *not* the DRAM-tier cache. The DRAM is now full of warm entries; the HBM pool is empty.
3. Resume traffic at the normal RPS. Every request hits DRAM and promotes a prefix into HBM.
4. Observe:
   - HBM occupancy spike.
   - Preemption rate (does the promotion traffic evict live-sequence KV?).
   - TTFT P99 spike (promotion contention).
   - PCIe utilisation.

Document the pathology — what metrics spiked, how long before the system stabilised, which backpressure mechanism (if any) kicked in to limit the storm.

Propose a mitigation (cap promotion rate, prefer evicting cached blocks over live blocks on HBM pressure, pre-promote top-K prefixes on cold start). Implement the mitigation if your tiered-KV stack exposes the knob; otherwise document what the mitigation would look like as a configuration proposal.

## Starter guidance

- **LMCache deployment path** is evolving. The current LMCache docs are authoritative; integration with vLLM is typically a forked `vllm serve` or a plugin. Verify before committing to the integration method.
- **Verify the stack before measuring**. Send one request twice; the second must show a cache hit (via the LMCache / tiered-KV metric endpoint, not just via the engine's prefix-cache metric). If it doesn't, the integration is wrong — fix before continuing.
- **DRAM budget is a knob**. Try 4 GiB, 16 GiB, 64 GiB. The hit-rate curve as DRAM grows is informative.
- **NVMe is optional**. For a 4-hour exercise, skip NVMe unless you have the drive mounted and the tiered-KV stack supports it. The DRAM tier alone is enough to make the per-tier framing concrete.
- **For the promotion-storm drill**, make the DRAM tier survive the engine restart by keeping it in a separate process (LMCache ships as a separate server). If your integration method keeps DRAM state inside the engine process, you'll have to simulate the pathology differently (e.g. drain the HBM pool while traffic continues).
- **Fallback: no tiered KV available**. Simulate tier-1 with a hand-rolled DRAM-backed cache in a Python sidecar that intercepts prefix lookups and holds blocks in RAM. The measurements won't be production-realistic, but the per-tier framing is still exercisable. State the limitation in your writeup.

## Fallbacks for constrained hardware

- **No NVMe**: do the two-tier variant (HBM + DRAM). Omit the tier-2 column from Artefact D.
- **Low host DRAM** (< 16 GiB): run with a 2–4 GiB DRAM budget. Hit rate will be lower; the framing holds.
- **No LMCache / Mooncake available**: implement the hand-rolled sidecar described above. State in writeup.
- **CPU-only engine**: all code paths work; TTFT numbers won't reflect production. Focus on hit-rate and bytes-moved axes.

## Acceptance criteria

1. **Artefact A** demonstrates the tiered-KV layer is actually in the request path with a verifiable test (second-identical-request shows a tier-1 hit).
2. **Artefact B** reports per-tier hits with monotonic shape (more hits in higher tiers), or explains convincingly why not.
3. **Artefact C** reports measured TTFT per tier; the measured tier-1 transfer time is within a factor of 2 of the back-of-envelope PCIe-bandwidth prediction.
4. **Artefact D** compares at least three tier configurations and defends a chosen production candidate with measured data.
5. **Artefact E** either produces a visible promotion storm (quantified) or explains why the setup could not provoke one and proposes an alternative drill.
6. All five artefacts live in version control; tiered-KV versions are pinned.

## Common failure modes

- **Tiered-KV layer not actually in path.** The engine's own prefix cache catches the hit before the tiered layer sees anything. Verify by instrumenting the tiered layer's lookup directly.
- **DRAM pool collides with host OS**. If DRAM budget approaches physical RAM, the OS pages out and your "DRAM-tier" latency becomes disk-latency. Watch for swap activity.
- **PCIe saturated by other traffic.** If live-sequence KV traffic already saturates PCIe, tier-1 promotions queue up and TTFT delta explodes. Diagnose with `nvidia-smi dmon -s pucvmet`.
- **Promotion storm not triggerable.** If your setup auto-rate-limits promotion, the pathology won't fire. Note it and propose how it would present on a less-protected stack.
- **Cost column based on list price, not realised cost.** Pick a cloud or on-prem reference and state it; don't mix.

## Keep it

The per-tier TTFT breakdown (Artefact C) and the trade-off table (Artefact D) feed into module 110's cost-per-token work. The pathology drill (Artefact E) is a template for the module 113 incident-response runbook family.
