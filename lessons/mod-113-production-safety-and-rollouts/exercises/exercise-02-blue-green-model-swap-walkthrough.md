# Exercise 02 — Blue-Green Model Swap Walkthrough

**Time:** 6 hours · **Output:** one successful blue-green or drain-and-swap of a model version across a representative replica fleet, with the atomic routing cutover demonstrated against in-flight streaming sessions, the `X-Model-Version` header verified on every response, a measured rollback-time budget, and a feasibility-matrix artefact for your hardware.

## Goal

Make [Chapter 02](../02-blue-green-and-model-swap.md) concrete. Pick the right pattern for your fleet's hardware (full-fleet blue-green, drain-and-swap, or in-place dual-load), execute the swap end-to-end on a realistic replica count, and prove the mechanic respects streaming semantics and rolls back fast.

## Problem statement

You have a model version in production (`v1`). A candidate (`v2`) has passed the chapter 05 eval gate and the chapter 01 shadow-rollout gate. You now need to transition the fleet from `v1` to `v2` without breaking active streaming sessions and with a rollback budget measured in minutes. Pick the pattern, execute, and document.

The goal is not just "it worked once"; the goal is "this is the ritual we'd run again, with these exact steps, and here is what we'd verify at each step."

## Prerequisites

- Chapter 02 of this module read; chapters 01 and 04 for cross-references.
- Chapter 02 of module 110 for the SLO definitions you'll honour during the swap.
- Module 102 chapter on engine startup and lifecycle for the shard-load and warm-up primitives.
- A fleet of at least 4 replicas serving one model (ideally 8+, so batched drains are meaningful).
- Two model builds — same family, different version, or same model under a different engine flag; must produce a distinct `X-Model-Version` value.
- A router capable of per-replica or per-pool weighted routing.
- Prometheus + Grafana with the module-110 axis set.

## Requirements

Six deliverables.

### Deliverable A — the dual-load feasibility matrix

Per chapter 02's feasibility matrix shape:

```
| Model     | Quant | Hardware     | TP  | One version FB_USED | Dual-load headroom | Verdict      |
|-----------|-------|--------------|-----|---------------------|--------------------|--------------|
```

Fill it in by *measuring* on your fleet, not by estimating. For your current model: run one replica at steady state, read `DCGM_FI_DEV_FB_USED` per GPU, record. Repeat for the candidate if its footprint differs (quantisation change). Compute dual-load headroom. State the verdict.

The verdict picks the pattern for the rest of this exercise. If the verdict is "No dual-load," you're doing drain-and-swap. If "Yes with margin," you may choose full-fleet blue-green or in-place dual-load.

### Deliverable B — the pattern choice and justification

Document in one paragraph:

- Which pattern (full-fleet blue-green, drain-and-swap, in-place dual-load).
- Why, given the feasibility matrix, the SLO targets from module 110, and the GPU budget.
- What the rollback mechanic is and its time budget.

### Deliverable C — the `X-Model-Version` emission

Verify (or implement, if missing) the `X-Model-Version` response header on both `v1` and `v2`. The header must:

- Be emitted before the first streamed chunk for SSE responses.
- Include a stable component (the API model name) and a specific version identifier.
- Be labelled on every histogram and counter as the `version` label (bounded cardinality: `active_versions ≤ 2` during swap, exactly 1 outside swap).

Include a small test that opens a 60-second streaming response against both versions and confirms the header is set on the first byte.

### Deliverable D — the drain sequence (for drain-and-swap) or the cutover sequence (for blue-green)

Author the sequence as a runnable script or a documented operator procedure:

**For drain-and-swap:**
```
for batch in batches:                             # e.g. 2 replicas per batch
    for replica in batch:
        router.set_weight(replica, 0)             # drain starts
        wait_until(replica.in_flight == 0 or drain_deadline)
        forcibly_close_remaining_streams(replica) # with clean SSE error
        stop_engine(replica)
        start_engine(replica, model=v2)
        wait_until(replica.readiness and replica.warmup_done)
        router.set_weight(replica, normal_weight)
    verify_slo_gates_green()                       # dwell between batches
    pause(cooldown)
```

Named parameters: `drain_deadline`, `batch_size`, `cooldown`, `warmup_time`. Defend each value in 1–2 sentences — grounded in your fleet's P99 stream duration and SLO budget.

**For full-fleet blue-green:**
```
provision_green_fleet(count=N, model=v2)
warmup_green_fleet()
smoke_test_green_fleet(fixture_count=200)
router.atomic_switch(from=blue, to=green)
observe_dwell(hours=D)
drain_blue_fleet()
after_safety_margin:
    decommission_blue_fleet()
```

Same discipline — parameters defended.

### Deliverable E — the drain-respects-streams demonstration

The hardest thing to get right. Build and run a small test:

1. Start the swap with a known long-stream workload running — several clients each issuing a request that emits 1000+ tokens over 30+ seconds.
2. Begin draining replicas while those streams are in flight.
3. Verify:
   - Streams that started before the drain complete on their original version (header stays consistent for the stream).
   - Streams started after the drain begins go to the new version.
   - No stream is interrupted mid-token (or if the drain deadline fires, the client gets a clean error, not a TCP RST).

Capture: a timeline (requests issued, drain events, stream completion events, per-stream model-version header) and the LB / router log confirming no mid-stream re-evaluation.

### Deliverable F — rollback-time measurement

Measure your pattern's rollback-time budget. Specifically:

- For full-fleet blue-green in the dwell phase: time to flip the router back to blue; measure in seconds.
- For drain-and-swap mid-roll: time from "stop the roll" to "all new traffic back on blue, candidate drained"; measure in minutes (likely).
- For drain-and-swap post-100%: time to reverse the roll back to blue; measure in roll-duration terms.

Run the measurement in practice — partway through the real swap, exercise the rollback path, and record. Then continue the swap to completion. Capture the recorded times.

## Starter guidance

- **Warm-up is not optional.** A cold replica serves its first requests at a TTFT that breaches SLO; treating a cold candidate as ready is where blue-green regressions happen. Pre-record representative traffic; replay before opening live weight.
- **Drain deadline from measurement, not from policy.** `kubectl` default `terminationGracePeriodSeconds` is 30; your longest streaming session may be 60+. Measure P99 stream duration per workload class, not a cross-fleet default.
- **Document what the LB actually does.** Service meshes have surprising semantics around streaming re-evaluation. Write the test (Deliverable E) first; if the LB re-evaluates mid-stream, change the LB *before* doing the real swap.
- **Retain blue after the swap.** Chapter 02 is explicit: retain at least an operational cycle. For this exercise, retain for 24 hours minimum (or document explicitly why your exercise environment can't).
- **Don't swap at peak.** If your fleet has daily traffic shape, run the swap in a trough; chapter 02's fleet-capacity analysis assumes some headroom during the swap.

## Fallbacks for constrained environments

- **One model only:** simulate `v2` as `v1` with a changed flag so the `X-Model-Version` header differs; the mechanic exercises even if outputs are equivalent.
- **Single-replica fleet:** you can only demonstrate in-place dual-load or a full-fleet blue-green with a provisioned second replica; drain-and-swap needs at least 2 replicas. Document the constraint.
- **No DCGM:** approximate FB_USED from `nvidia-smi --query-gpu=memory.used` snapshots; less granular but sufficient for the feasibility matrix.
- **No service mesh:** an Envoy or FastAPI-based router is sufficient; verify streaming semantics manually.

## Acceptance criteria

1. **Deliverable A** is a measured feasibility matrix with per-GPU HBM numbers from the running fleet.
2. **Deliverable B** names the pattern with a defensible rationale.
3. **Deliverable C** shows the header on first-byte for streaming responses on both versions, and the `version` label is bounded on metrics.
4. **Deliverable D** is a reproducible sequence (script or documented procedure) with each parameter defended.
5. **Deliverable E** demonstrates streaming-safety — streams in flight at the drain moment complete on their admitted version, with no mid-token interleaving.
6. **Deliverable F** reports measured rollback times for your pattern.
7. The swap completes successfully — fleet ends on `v2`; SLO panels stay green through the swap (or any breach is documented with cause).

## Common failure modes

- **Dual-load on a fleet that can't.** HBM OOM at startup; module 104 OOM runbook (chapter 03 of this module) fires.
- **Drain deadline shorter than longest stream.** Users see mid-token cuts; schema-conformance counter spikes falsely.
- **Interleaved versions mid-stream.** Router re-evaluates weight per chunk; response has mixed-version output.
- **Blue decommissioned before dwell complete.** Rollback requires a fresh deploy instead of a flag flip.

## Keep it

The feasibility matrix, the pattern decision document, the drain / cutover script, the streaming-safety test, the rollback-time measurement. This is the ritual your team runs on the next real model swap.
