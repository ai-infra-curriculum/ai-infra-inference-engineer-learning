# Chapter 02 — Blue-Green and In-Flight Model Swap: Dual-Load, Drain-and-Swap, Atomic Cutover, Model-Version Header

> Blue-green on a stateless microservice costs a few extra pod-minutes. Blue-green on a 70B replica costs a second fleet of GPUs or a careful drain-and-swap sequence. This chapter names which one applies when, how to cut over without breaking a 15-second streaming response mid-token, and what the `X-Model-Version` header does for every downstream observer.

## Motivation

Chapter 01's canary primitive handles the clean case — same outputs, gradual ramp, SLO gate, roll back on burn. Several change categories do not fit that mould:

- A new tokenizer with the model. Mid-stream version flip produces garbage; the API contract itself has shifted; the client has to see one version per request.
- A hard model-version change where shadow traffic (chapter 01) already produced the evidence, eval gates (chapter 05) have passed, and there's nothing more to learn at 1 % live.
- A rollback *from* a canary that promoted — now 100 % of the fleet is on the bad version and you need to swap back whole.
- An engine-upgrade that invalidates in-process state (KV allocator layout, CUDA graph signatures). Mixed-version replicas in one fleet would misbehave at the router.

For these you need *blue-green* or *in-flight model swap*: a ritual that moves the whole fleet from version A to version B as atomically as the streaming-session shape allows.

The challenge is GPU-specific. A 70B model at `TP=8` uses 70+ GB of HBM for weights plus tens of GB for KV. On an `8× H100 80GB` node, you *cannot* simultaneously hold the old and new weights in memory. On an `8× H200 141GB` node, you often can — barely, and with no KV headroom. The blue-green pattern that assumes "spin up green at 100 %, flip the LB, shut down blue" is not available on every fleet. Chapter 01's canary was free to ramp over hours; this chapter's ritual often has a dwell measured in seconds-to-minutes while the fleet is in a mixed state.

By the end of this chapter you can:

- Decide between full-fleet blue-green, drain-and-swap, and rolling model-swap for a given change.
- Reason about dual-load constraints — "can this replica fit both versions at once?" — per hardware class.
- Design a drain sequence that respects in-flight streaming sessions and bounded tail-latency.
- Make the routing cutover atomic from the client's perspective.
- Emit a correct `X-Model-Version` header and consume it in dashboards and client telemetry.
- Budget the rollback time so a failed swap is reversible in minutes, not hours.

## Three patterns, three fleet shapes

| Pattern | Spare capacity needed | When to pick | What breaks if you pick wrong |
|---|---|---|---|
| **Full-fleet blue-green** | 2× the fleet during cutover | Small models (fits twice per node), or fleets where GPU budget allows the overlap, or the change is high-risk and rollback-on-a-second's-notice is required. | You pay for a second fleet of expensive GPUs for the cutover window. |
| **Drain-and-swap (rolling)** | No extra capacity; one replica at a time goes offline, swaps version, comes back | Large models (one version per node), tight GPU budget, dual-load not possible. | Capacity drops during the roll; if traffic is near the SLO ceiling, you'll breach latency. |
| **In-place dual-load** (same replica holds both) | Enough HBM headroom per replica for both weights plus some KV | Mid-size models on big-HBM nodes (H200 / B200 / MI300X territory); engine supports multi-model co-location (module 106). | HBM pressure can preempt KV; cold-swap-path can still stall the first requests. |

The first question is always *do both versions fit on one replica at the same time?* The answer is per-model, per-hardware, per-engine. Test it, don't guess it.

### Dual-load feasibility — the back-of-envelope

For a model of parameters `P` quantised at `b` bytes per parameter:

```
weights_bytes = P * b
```

Plus optimizer-free activation overhead for inference (typically a few GB), plus KV block pool (per-token KV size × block count, from module 104).

For `P = 70 × 10^9` and FP8 (`b = 1`): ~70 GB for weights. On an `8× H100 80GB` system with `TP=8`, each GPU holds ~9 GB of weights plus activation plus KV. Loading a second version simultaneously doubles the weight share per GPU to ~18 GB, leaving ~40 GB for KV per GPU. On `H200 141GB`, you have ~90 GB of headroom per GPU — comfortably dual-loadable.

For `P = 70 × 10^9` and FP16 (`b = 2`): ~140 GB for weights. On `H100 80GB` you need `TP ≥ 2` to hold even one copy; two copies at `TP=8` would exceed per-GPU HBM; you cannot dual-load. Drain-and-swap or full-fleet blue-green.

<!-- needs-research: exact per-GPU HBM budget after engine, CUDA runtime, CUDA graphs, and NCCL communicator overhead varies substantially by engine release; always measure against your engine's reported `DCGM_FI_DEV_FB_USED` at steady state with headroom. -->

The output of this envelope is a *feasibility matrix* for your fleet:

```
| Model          | Quant | Hardware     | TP  | One version FB_USED | Dual-load headroom | Verdict      |
|----------------|-------|--------------|-----|---------------------|--------------------|--------------|
| mymodel-70b    | FP16  | 8× H100 80GB | 8   | 72 GB / GPU         | -4 GB              | NO dual-load |
| mymodel-70b    | FP8   | 8× H100 80GB | 8   | 48 GB / GPU         | 32 GB              | Yes w/ margin|
| mymodel-70b    | FP16  | 8× H200 141GB| 8   | 72 GB / GPU         | 69 GB              | Yes          |
```

The verdict picks the pattern.

## Full-fleet blue-green

The simple pattern. Blue is the current fleet, green is the candidate, both at full replica count:

1. **Provision green.** `N` candidate replicas, same TP / PP shape as blue; same autoscaling floor; reachable but *not yet routed to*.
2. **Warm-up green.** Pre-recorded traffic replay (see chapter 01); CUDA graphs captured; TTFT / TPOT stable; smoke-test fixture passes.
3. **Cut over.** Router weight for the API model name flips from blue to green in one atomic update. New requests go to green. Existing streaming sessions on blue finish on blue (see the in-flight-streams section below).
4. **Drain blue.** Blue stops accepting new requests; its in-flight sessions complete; its replicas scale to zero after a drain timeout.
5. **Observe.** Hold green at 100 % under watch for the dwell (hours to days); rollback remains available by flipping the router back to blue, which has been retained but idle.
6. **Decommission blue.** After the dwell plus a safety margin, release blue's GPU reservations back to the fleet pool.

The virtue: rollback is "flip the router again," which is seconds. The cost: 2× GPU for the dwell + decommission window.

### When this pattern is actually available

- You have the GPU budget (reserved capacity, spot that can be acquired reliably, a provider with elastic GPU supply).
- The model's weights and warm-up state fit on a *second* fleet of same shape.
- The ingress / router can hold two distinct backend pools under one logical model name with a weight-shift primitive.

For many L35 teams, this pattern is reserved for high-blast-radius changes (new model family, major engine upgrade) where the rollback speed justifies the cost.

## Drain-and-swap (rolling)

The pattern for tight budgets. One replica at a time:

1. **Mark a blue replica for drain.** Its weight in the router drops to 0 for new requests; existing requests continue.
2. **Wait for in-flight sessions to finish** or reach the drain-deadline.
3. **Shut the replica down.** Engine stops; model unloads; worker exits.
4. **Load the green version** on the same node (same `TP` shape; new weights). Engine starts up; warm-up runs; smoke-test passes.
5. **Mark the replica as green and ready.** Router opens traffic to it at normal weight; the next replica starts its drain.
6. **Repeat** until all blue replicas are green.

Three trade-offs:

- **Capacity dips while a replica is draining.** If the fleet normally runs at 60 % utilisation of its SLO ceiling, draining one of 10 replicas drops effective capacity by ~10 % — fine. Draining one of 3 replicas at 80 % utilisation — not fine; latency breaches during the roll.
- **Model startup is the critical path.** Loading a 70B model from disk can take 1–5 minutes (shard load, CUDA init, graph capture, warm-up). Faster start-up (preloaded shards, mounted model volume, warm NVMe) reduces the total roll time.
- **Draining a streaming workload takes its longest streams.** If chat sessions can stream for 60 s, the drain deadline should be at least 60 s; faster deadlines cut live sessions mid-stream. See the next section.

A representative Kubernetes-flavoured pseudo-rollout (illustrative):

```
for replica in replicas:
    router.set_weight(replica, 0)          # drain starts
    wait_until(replica.in_flight == 0 or drain_deadline)
    forcibly_close_remaining_streams(replica)
    kubectl.delete(replica)                 # old version off
    kubectl.apply(replica, image=GREEN)     # new version on
    wait_until(replica.readiness == "ready" and replica.warmup == "done")
    router.set_weight(replica, normal_weight)
    pause(cooldown)                         # let SLOs re-stabilize
```

The pause between replicas matters: if you roll too fast the fleet is mostly "draining or starting" and effective capacity is a fraction of nominal.

### Partial-fleet variants

A useful variant for very large fleets: *batched* drain-and-swap. Roll 10–25 % of the fleet at a time with gates between batches:

- Batch 1 (10 %) → gate on SLO burn-rate and response-diff (chapter 01's gate set, but applied at fleet share).
- Batch 2 (25 %) → gate again.
- …until 100 %.

This looks like a canary ramp at the replica level, but the *routing* is uniform across versioned replicas — the `X-Model-Version` header still attributes which version served which request, and the gates still have per-version metrics.

## In-place dual-load

If dual-load feasibility is "yes with margin," the engine can hold both weights at once and the router can shift traffic within a single replica. Three ways this looks in practice:

- **Multi-model engine primitive.** vLLM / SGLang / TRT-LLM can serve multiple model names from one process (module 106). You load the green model under `mymodel-70b-v2`, canary traffic for the public name `mymodel-70b` from v1 to v2, and unload v1 after the switch.
- **Multi-engine co-location.** Two engine processes on the same node, each with its own model, sharing MIG / MPS slots. Riskier operationally; module 106 is cautious about MPS under heavy KV.
- **LoRA-scale swap.** For an adapter change (module 106), the base model stays loaded; only the adapter weights swap. The dual-load constraint is trivial. Promotion is the canary / shadow pattern from chapter 01, scoped per adapter.

In-place dual-load collapses the drain window: there is no "drain this replica, stop it, restart it." The replica stays live; the router shifts. The trade-off is HBM pressure — KV budget shrinks while both versions are resident; module 104's over-admission runbook (chapter 03) is a risk.

## In-flight streams and atomic cutover

Streaming is what makes inference blue-green different from a REST API's blue-green. A chat completion can stream for seconds-to-minutes. A cutover that is "flip the router at T" means:

- At T, requests already issued and streaming stay on their assigned backend. The router is consulted on admission, not per-chunk.
- At T, new requests go to the new assignment.
- Requests issued *immediately before T* but whose connection only reaches the engine at T+ε are an edge case — they should land on the pre-T side (first admission wins). If they land post-T and are retried on error, the idempotency key honours one assignment per key.

Three semantics to get right:

- **Session affinity ≠ version pinning.** A chat session (several completions over minutes) does not need to stay on the same model version between completions; it needs each *completion* to stay on one version until end-of-stream. Document this with your product partner.
- **Drain deadline exceeds longest reasonable stream.** For agent workloads that stream for 60 s to minutes, a 10-second drain deadline forcibly aborts streams mid-answer. Set the deadline to your longest workload class's P99 stream duration or more.
- **Forcibly aborted streams must be visible.** When drain-deadline forces a mid-stream close, the ingress emits a clean SSE / WebSocket error (not a TCP RST) so the client can retry. The client's retry hits the new version with a fresh idempotency key or — if the key is the same and the server is strict — is rejected with `409 Conflict`.

### Router primitives

The router must support:

- **Backend selection at admission** that reads a per-model weight table.
- **Atomic weight update** (one request flips old→new; no interleave).
- **Per-connection stickiness** (once admitted, all chunks for that stream go to the same backend).
- **Idempotency-key-aware retry routing** (if the key is in-flight on backend X, retries go to X until the stream closes).

Service meshes (Istio, Linkerd) and ingress controllers (Envoy, NGINX with care, HAProxy) can do this; verify your pinned router's streaming semantics against a test case before trusting it.

## The `X-Model-Version` response header

The non-negotiable instrument for every pattern in this chapter (and chapter 01). Format:

```
X-Model-Version: mymodel-70b/2026-10-28-fp8-v3@sha256:abcd1234...
```

Three roles:

- **Client-side observability.** Clients can log the version alongside the request outcome; product teams can bisect quality / latency incidents against version without needing fleet access.
- **Dashboard attribution.** The `version` label on every histogram and counter is populated by the header's value (with bounded-cardinality hygiene — see below).
- **Post-incident forensics.** Every log line, every trace span, every captured response records the version. Chapter 04's RCA overlays this directly.

### Emission disciplines

- **Emit before first streamed byte.** For SSE, the header is on the HTTP response (sent before the body). Verify your framework actually emits headers before the first stream event.
- **Include a stable component and a semver-ish component.** The stable component (`mymodel-70b`) is the API-level name; the version component (`2026-10-28-fp8-v3`) is the specific build; the `sha256` is the weight digest. The dashboard uses the `(name, version)` tuple as the label; the sha is for forensics only.
- **Bounded cardinality.** The number of active model versions is small (`versions_in_active_rollout + recent_history ≤ ~5`). The sha is *not* a metric label; store it in logs and traces where high-cardinality is cheap.
- **Clients can tolerate the header being absent.** For legacy fleets or old engines, the header may be missing; dashboards should treat absence as a known class, not a crash.

## Rollback-time budget

The question every pattern in this chapter has to answer: *if we are 30 minutes into a swap and the SLO burn-rate alert fires, how fast can we be back on blue?*

| Pattern | Rollback mechanic | Typical time budget |
|---|---|---|
| Full-fleet blue-green (dwell phase) | Router weight flip | Seconds to a minute |
| Full-fleet blue-green (post-decommission) | Re-provision blue | Hours; same as initial deploy |
| Drain-and-swap, mid-roll | Pause the roll; existing-green stays green; next replicas skip drain-to-green | Minutes (depends on how many are already green) |
| Drain-and-swap, post-100% | Reverse rolling swap back to blue | Full roll duration again |
| In-place dual-load | Router weight flip within replicas | Seconds |

Three rules:

- **Don't decommission blue until you're sure.** For high-blast-radius changes, retain blue for at least one operational cycle (day, week — product decision) after the swap.
- **Rollback is a *first-class procedure*, not a theoretical capability.** Chapter 03's runbook set has a "rollback this swap" runbook and chapter 04's post-mortem checks whether it was invoked. Practice it.
- **The rollback itself can fail.** A newly-re-loaded blue might have cold CUDA graphs; a drained fleet might not re-scale fast enough. Chapter 04's RCA calls this out when relevant.

## Worked example — swapping a 70B chat model from FP16 to FP8 on an H100 fleet

Context: 32 replicas × 8× H100 80GB; FP16 `mymodel-70b/v1`; candidate `mymodel-70b/v2-fp8`; feasibility matrix says "dual-load not possible on H100 80GB with margin." Drain-and-swap selected.

1. **Pre-roll:** chapter 05 eval gate green; chapter 01 shadow-traffic rolled; pre-loaded FP8 shards on NVMe on every node.
2. **Roll config:** 4 replicas per batch (12.5 %); drain deadline 90 s (P99 chat stream + 20 s); readiness gate 5 min per replica (shard-load + warm-up + smoke).
3. **Roll:**
   - Batch 1 (4 replicas, replicas 1–4): drain, swap, warm; `X-Model-Version` on `v2-fp8` for those replicas; fleet is now 87.5 % blue, 12.5 % green; SLO gates green for 20 min dwell between batches.
   - Batch 2 (replicas 5–8): same. Fleet is 75 % blue, 25 % green. One burn-rate alert: `TPOT P95 on long_context` on v2-fp8 is +50 % from v1. Pause the roll. Investigate.
   - Investigation finds a FlashAttention variant pick under FP8 long-context; add engine flag `--attention-backend flashinfer`; re-build candidate image.
   - Rollback: `router.set_weight(v2_replicas, 0)`; the 8 green replicas drain to zero traffic; they get re-imaged with the updated flag; now re-join the roll at batch-1 position.
   - Roll resumes; all batches complete over ~2 hours total.
4. **Post-roll dwell:** 48 hours; blue retained at zero traffic but with weights on NVMe so revert is a 10-minute re-load, not a 90-minute deploy.
5. **Decommission:** blue's nodes return to the fleet pool after dwell plus a 72-hour safety margin.

The feasibility matrix picked the pattern; the drain deadline respected the streaming shape; the `X-Model-Version` header attributed the long-context TPOT regression to v2; the rollback used the pattern's inherent mechanic.

## Four common mistakes

- **Dual-loading on a fleet that can't.** HBM OOM at startup; module 104's OOM runbook fires; incident. Resolve by running the feasibility envelope *before* authoring the swap.
- **Drain deadline shorter than longest stream.** Users see mid-answer cuts; retries hammer the new version; schema-conformance counter spikes falsely; the rollout looks like a regression. Resolve by sampling actual P99 stream durations per workload class before setting the deadline.
- **Interleaved versions mid-stream.** Router re-evaluates weights per chunk; one response has v1's first paragraph and v2's second; user-visible garbage. Resolve by making router selection happen at admission only, documented and tested.
- **Blue decommissioned too fast.** An issue surfaces three days post-swap; blue is gone; recovery is a full re-deploy. Resolve with a retention policy on the prior version.

## Summary

- Blue-green and in-flight model swap are the rituals for changes that can't canary cleanly. Picking the pattern is a function of dual-load feasibility and GPU-budget constraint.
- Build the feasibility matrix per (model, quant, hardware, TP). Measure HBM headroom with the engine running; don't guess.
- Full-fleet blue-green has the fastest rollback but costs 2× GPU during cutover. Drain-and-swap costs no extra GPU but drops capacity during the roll. In-place dual-load collapses the drain window but pressures KV.
- Streaming respects admission-time version assignment; router never re-evaluates per chunk. Drain deadline exceeds longest reasonable stream.
- `X-Model-Version` is set on every response, before first byte, with bounded-cardinality label hygiene; it is the attribution axis for chapter 04 forensics.
- Rollback is a first-class procedure; retain the prior version for at least an operational cycle; practise reverting the swap.
- Chapter 01 handles the canary ramp before this chapter's cutover; chapter 03 writes the runbook for a swap gone sideways; chapter 04 runs the RCA; chapter 05 is the pre-flight eval gate.
