# Chapter 03 — Inference Runbooks: Template and the Top-Ten Set

> The on-call engineer who has never seen this incident reads the runbook. The runbook either describes the exact signal, the exact first bisection, and the exact first fix — or it doesn't, and the engineer starts guessing at 2 AM. The signals here are inference-shaped and no generic "web app is slow" runbook covers them. This chapter authors ten that do.

## Motivation

A runbook is not documentation; it is a decision tree compressed to one page. A good runbook answers four questions in order:

1. *Am I looking at this incident?* (Symptoms match.)
2. *Which of the half-dozen common causes is most likely?* (First bisection.)
3. *What do I do about the most likely cause?* (First fix or escalation.)
4. *Who do I page if the first fix doesn't work?* (Escalation.)

An inference engineer's runbook also has to assume the reader doesn't know which engine metric to look at, which DCGM counter to pull, which scheduler flag to touch. The chapter 01 axis set from module 110 is the vocabulary; the originating-module debug surface (102–109) is where fixes come from; this chapter composes them into ten decisions.

By the end of this chapter you can:

- Write a runbook that is usable in the first five minutes of an incident.
- Dispatch any of the ten common inference incidents to a first bisection against a specific axis.
- Identify when a runbook's first fix exhausts and the correct escalation is needed.
- Avoid the runbook antipatterns — stale, encyclopaedic, generic, or opinion-free.

## The runbook template

Every inference runbook in this module uses one template. The template has six sections; aim for one page of text when rendered, no scroll on a laptop:

```
# Runbook: <short imperative title>

## Owned by
<team / rotation>; last reviewed <YYYY-MM>

## Symptoms (you're looking at this if...)
- <one sentence per symptom>

## First bisection (do these in order)
1. <action>: expected signal if this is the right runbook; what it rules in or out.
2. <action>: …
3. <action>: …

## Most likely causes and first fixes
- Cause A (how to confirm) → fix (commands, flags, scripts).
- Cause B (how to confirm) → fix.
- Cause C (how to confirm) → fix.

## Escalation
- If first bisection step N doesn't narrow → escalate to <team / peer>.
- If first fix doesn't restore signal in <time budget> → open an incident at severity <N> and page <role>.

## Rollback
- If this followed a deploy in the past <window>, rollback procedure:
  <command / Argo / router weight / replica scale>.

## References
- Module <N> chapter <N> for the underlying primitive.
- Dashboard: <link>.
- Prior incidents: <IDs>.
```

Three disciplines:

- **Owned by and last reviewed are not decorative.** A runbook with no owner rots and nobody notices. Schedule reviews on a calendar (chapter 04 of module 114's cadence).
- **Commands are copy-pasteable.** Not pseudo-code. The on-call engineer at 2 AM will copy-paste whatever is in a code block; make it the right thing, pinned to your engine / router / cluster.
- **Escalation names a human-reachable path.** "Page SRE" is useless if your org has four SRE teams; "Page `#oncall-inference-platform` or `platform-oncall@`" is useful.

## The top-ten inference runbooks

Each runbook below is a *template-shape summary* — the structure, the axis to read first, the likely causes, the first fixes, the escalation. The exercise in this module (exercise 03) carries each one to the fully-authored one-pager against your fleet's specifics.

### Runbook 1 — TTFT regression

- **Symptoms:** `ttft_seconds_bucket` P95 or P99 up by >2× from steady state on one or more workload classes; often paired with an increase in `queue_seconds` or `prefill_seconds`.
- **First bisection:**
  1. Is it one replica or all replicas? Split TTFT by `replica_id`. If one replica → drain and escalate; if all → continue.
  2. Split TTFT into `queue_seconds + prefill_seconds + first_decode_seconds`. Which component dominates?
  3. If `queue_seconds` dominates → scheduler saturation; check `in_flight_sequences` and `queue_depth` (module 103).
  4. If `prefill_seconds` dominates → GPU-side; split by `prompt_length_bucket`.
  5. If `first_decode_seconds` dominates → suspect CUDA-graph re-trace, cold KV cache, or a scheduler-side first-token stall.
- **Most likely causes and fixes:**
  - Queue saturation → autoscale (module 109) should have fired; if it didn't, check the HPA signal path; temporary manual scale-out.
  - Prompt-length shift (workload change) → raise `max-num-batched-tokens` or lower `max-num-seqs`; module 103's trade-off table.
  - Prefill throughput regression → suspect recent deploy; check `X-Model-Version` histogram labels; rollback if a swap happened in the last hour (chapter 02 rollback mechanic).
  - CUDA-graph re-trace storm → engine-side; check logs for re-trace events; module 102's engine failure-modes chapter.
- **Escalation:** cluster-level if bisection doesn't localise within 15 min.
- **Rollback:** chapter 02 rollback if this followed a model swap or engine upgrade; chapter 01 rollback if a scheduler-flag change is live.
- **References:** module 110 chapter 01 (axis set), module 103 (scheduler), module 102 (engine).

### Runbook 2 — Cache-hit cliff

- **Symptoms:** `cache_hit_rate` drops sharply (e.g. from 0.6 steady to 0.1 within minutes); TTFT follows upward because recomputed prefills cost more; cost-per-1M-tokens rises.
- **First bisection:**
  1. Confirm the metric: `rate(cache_hit_total[5m]) / rate(cache_attempt_total[5m])` — is this a real drop or a metric path change (chapter 01 of module 110 cardinality trap)?
  2. Did the eviction rate on the KV cache spike? Module 104's allocator metrics.
  3. Did the prompt / prefix distribution shift? Check `prompt_length_bucket` and a rolling prompt-hash distribution.
  4. Did a recent deploy change prefix-cache config (size, eviction policy)? Check deploy events.
- **Most likely causes and fixes:**
  - Workload shift — a client rolled out a new prompt template; prefix sharing broke. Talk to the consumer; often the right fix is upstream.
  - KV budget shrank — a new model version has larger per-token KV footprint; block pool pre-allocated the same number of blocks for a smaller effective capacity. Module 104.
  - Prefix-cache config regression — eviction policy switched to LRU at too-aggressive TTL; raise TTL or switch to the engine's prefix-cache default.
  - Cache disabled by a flag flip — verify `--enable-prefix-caching` (vLLM) / the SGLang equivalent is still on.
- **Escalation:** module 104's maintainer / on-call if a KV allocator bug is suspected.
- **Rollback:** if a config change, revert the flag; if a new model version, chapter 02 rollback.
- **References:** module 104 (KV cache, prefix caching), module 110 chapter 04 (cost-per-token credit for cache hits).

### Runbook 3 — Preemption spike

- **Symptoms:** `preemption_events_total` rate goes from ~0 to >0.01 per in-flight seq per second; `ITL_P99` spikes on the dashboard while `TPOT_mean` stays close to normal; occasional `503` or request-aborted events if preemption exhausts the swap budget.
- **First bisection:**
  1. Is `FB_USED / FB_TOTAL` (DCGM, module 110) near 1.0? If yes → KV pressure.
  2. Is `max-num-seqs` above the model's safe limit at current prompt distribution? Module 103's over-admission signature.
  3. Is a new workload class pushing in with long prompts that exceed the block-swap-to-CPU budget?
- **Most likely causes and fixes:**
  - Over-admission — lower `max-num-seqs` by 20 %; see module 103 chapter on scheduler trade-offs.
  - KV budget regression — see runbook 2 for the shared cause pattern.
  - Preemption-policy change — if the deploy switched from `swap` to `recompute` (or vice versa) the latency profile shifted; revert if unintentional.
- **Escalation:** module 103 chapter owners if preemption remains > 0.005 after lowering `max-num-seqs`.
- **Rollback:** scheduler-flag change deploys follow chapter 01's rollback.
- **References:** module 103 (scheduling), module 104 (KV allocator, block swap).

### Runbook 4 — Silent accuracy regression / output drift

- **Symptoms:** *No latency alert fires*; cost / token looks normal; engine health green. The signal is *external*: a product metric dips (retention on a feature, a human-eval rate, a down-funnel KPI), or a scheduled offline eval reports a benchmark drop > tolerance, or a user-support ticket volume spikes.
- **First bisection:**
  1. What deploys happened in the last 72 hours? Check `X-Model-Version` histogram labels for shifts; check adapter promotions.
  2. Run the chapter 01 shadow-traffic / response-diff score offline against current primary vs. last-known-good version on a held-out fixture.
  3. Run the chapter 05 benchmark gate on current primary and on last-known-good.
- **Most likely causes and fixes:**
  - A model-version swap passed a canary / eval gate but the gate's threshold was too permissive for the specific regression — rollback (chapter 02) *immediately*; back-feed the regression as a permanent fixture (chapter 04).
  - An adapter promoted and is serving a cohort with a quality drop — rollback the adapter (module 106); back-feed.
  - A deeper engine-side regression altered sampler behaviour at `temperature=0` — module 102 engine-failure-modes chapter; probably rollback the engine upgrade.
- **Escalation:** chapter 05's eval peers; chapter 04's incident process. Silent drift incidents are often multi-day.
- **Rollback:** always rollback first and diagnose later for silent drift; the signal is unreliable enough that leaving the regressed version in place costs more than the rollback.
- **References:** module 105 (quantisation gate), chapter 05 (eval contract), module 108 (speculative draft-model drift).

### Runbook 5 — NCCL hang

- **Symptoms:** A `TP > 1` replica stops progressing; `in_flight_sequences > 0` but `decode_tokens_per_second = 0` for that replica; NCCL warnings or `NCCL_WATCHDOG` timeouts in the engine log; GPU `SM_ACTIVE` stuck at ~1.0 with no useful work; eventually a worker crash with an NCCL error.
- **First bisection:**
  1. Which rank is stuck? Each worker has an NCCL rank; logs name it.
  2. Is it a repeat on the same rank → likely hardware (one GPU / one NVLink pair); or different rank each time → likely software (allocator pattern, variable collective sizes, engine-release bug).
  3. Is the fabric saturated? Check `NVLINK_RX_BYTES` / `IB_BYTES`; a bad cable or a routing fault shows as a single-link saturation.
- **Most likely causes and fixes:**
  - Stuck rank, repeat offender → drain the replica; reboot the node; module 109 autoscaling should re-fill; file a hardware ticket with the XID / MLX5 logs.
  - Deadlocking collective pattern — if a recent engine upgrade introduced it, rollback (chapter 02).
  - Timeouts set too tight — `NCCL_TIMEOUT` / engine-side watchdog thresholds can mask progress on a very slow collective; verify against module 107's NCCL timing axis.
- **Escalation:** module 107 topology chapter authors / on-call; hardware ops for the node reboot and ticket.
- **Rollback:** engine-upgrade rollback if the pattern started post-deploy.
- **References:** module 107 (NCCL / topologies), module 102 chapter on NCCL-hang diagnostics.

### Runbook 6 — GPU driver / XID / ECC error

- **Symptoms:** `DCGM_FI_DEV_XID_ERRORS` increments; `DCGM_FI_DEV_ECC_DBE_VOL_TOTAL` increments (uncorrectable); worker logs show `CUDA error: an illegal memory access was encountered` or an unrecoverable driver fault; kernel logs show `NVRM: Xid (PCI:...)`.
- **First bisection:**
  1. Which XID? Look it up against NVIDIA's XID reference (<https://docs.nvidia.com/deploy/xid-errors/>).
  2. Is it a single GPU repeat offender, a single node repeat offender, or fleet-wide?
  3. Is DBE non-zero? Uncorrectable ECC = hard failure, replace the GPU.
- **Most likely causes and fixes:**
  - Single-GPU hardware fault → drain the node; power-cycle; if XID reoccurs, remove from pool; file hardware RMA.
  - Driver mismatch / recent driver upgrade → check `nvidia-smi` driver version against the engine's expected version; rollback or upgrade.
  - Thermal throttle → `DCGM_FI_DEV_GPU_TEMP` sustained > vendor spec; check cooling / rack power; schedule a maintenance window.
- **Escalation:** DC / hardware ops; module 109 for capacity impact.
- **Rollback:** driver rollback if the issue began at a driver update; chapter 02 blue-green rollback is not relevant here (the issue is hardware/driver, not model/engine).
- **References:** NVIDIA XID error reference; DCGM docs; module 102 chapter on hardware-signalled failure modes.

### Runbook 7 — Worker crash loop

- **Symptoms:** A pod / worker restarts repeatedly; Kubernetes `CrashLoopBackOff`; capacity drops because of the replica flapping in and out.
- **First bisection:**
  1. Log capture before crash — look for `CUDA out of memory`, `NCCL error`, `AssertionError: ...` from the engine, driver panic.
  2. Has anything changed in the last 24 h — image, config, model weights, node pool?
  3. Is it one replica, one node, or distributed?
- **Most likely causes and fixes:**
  - OOM at startup — model + KV pool exceeded HBM; a quantisation change raised memory beyond expected; lower `max-num-seqs`, lower KV block count, or rollback.
  - Health-check tuned too tight — the engine hasn't finished loading before the readiness probe fails; raise probe timeout.
  - Transient dependency failure — model registry / object-store slow; worker hits a download timeout; adjust pull policy and fallback to a cached shard path.
- **Escalation:** module 109 for the autoscaling / pod-template level; module 105 if the crash correlates with a model-weight / quantisation change.
- **Rollback:** chapter 02 rollback if the crash began at a model swap; chapter 01 rollback if a flag change.
- **References:** module 102 (engine lifecycle), module 109 (autoscaling).

### Runbook 8 — Admission-reject flood

- **Symptoms:** `admission_rejects_total` rate spikes; clients see `429`; fleet GPU utilisation is below expected (fleet has capacity it isn't using).
- **First bisection:**
  1. Fleet-scoped admission reject or per-tenant? Split by `tenant_tier`.
  2. Are per-tenant rate limits firing on a specific tenant's batch job (chapter 01 of module 110's cardinality trap pattern)?
  3. Is the scheduler rejecting on context length, prompt length, or total queue? Inspect reject-reason labels (module 103).
- **Most likely causes and fixes:**
  - Per-tenant rate limit tripped by a legitimate burst — negotiate a temporary raise with the tenant / platform team.
  - Context-length guard — a client is sending prompts beyond `max-model-len`; return a helpful error; chapter 05's migration guide may be the long-term fix.
  - Admission misconfig — `max-num-seqs` cap too low for the fleet's effective capacity; raise it (check KV budget first, see runbook 3).
- **Escalation:** fleet-level capacity management if admission-reject persists after per-tenant fixes.
- **Rollback:** scheduler config change — chapter 01 rollback.
- **References:** module 103 (admission / scheduling), module 106 (multi-tenant isolation).

### Runbook 9 — KV OOM

- **Symptoms:** Engine logs `KV allocator: no blocks available`; `preemption_rate` jumps immediately before; some requests are aborted mid-decode (`AbortedBy="kv_oom"`); dashboard panel 110-04 on `FB_USED` reads near 1.0.
- **First bisection:**
  1. `FB_USED` plateau or climb? Plateau → sized too tight; climb → leak.
  2. Was `max-num-seqs` recently raised? Prompt-length distribution recently longer?
  3. Any evidence of a prefix-cache runaway (chapter 04 of module 104)?
- **Most likely causes and fixes:**
  - Overcommitted KV — lower `max-num-seqs`; module 103 and 104 explain the trade-off.
  - Allocator leak — restart the replica as a short-term; file a bug with module 104's allocator owners; verify against the engine's reported reserved-vs-free blocks.
  - Prefix-cache too aggressive — tune TTL or cap total prefix-cache size (module 104).
- **Escalation:** module 104 owners for allocator-bug cases.
- **Rollback:** chapter 02 rollback if a model swap raised KV footprint; chapter 01 rollback if scheduler flag.
- **References:** module 104 (allocator, prefix cache), module 110 chapter 01 (axis: FB_USED vs. KV usage).

### Runbook 10 — Streaming-close storm

- **Symptoms:** Client-side disconnects spike; server logs show a burst of `ConnectionReset` on SSE stream writes; a correlated spike in back-end retries; often a network-path event rather than an engine event.
- **First bisection:**
  1. Is it ingress-side or downstream? Check load-balancer logs; are TCP RSTs showing on the LB → client side or LB → engine side?
  2. Is a client library hitting idle-timeout mid-stream? Chapter 02 drain or an ingress idle-timeout change is a common cause.
  3. Did a WAF / network rule flip? Confirm no policy change.
- **Most likely causes and fixes:**
  - Ingress idle-timeout shorter than stream — raise timeout; this is a common deploy-day regression.
  - Keep-alive config regression — check LB / ingress keep-alive settings; verify SSE / WebSocket heartbeat is still emitting.
  - Client bug — the client library upgraded and now closes on first byte; coordinate with the consumer peer (`llm-application-developer` at L25); workaround at the server by extending `keep-alive` message frequency.
- **Escalation:** networking / platform-ops; consumer-peer on-call if client-side.
- **Rollback:** LB / ingress config rollback is often the fix; verify change-calendar.
- **References:** module 111 (streaming / SSE / client-disconnect handling), module 109 (ingress).

## Runbook antipatterns

Three failure modes to avoid when authoring:

- **Encyclopaedic.** The runbook tries to cover all possible causes; it ends up a 20-page document that the on-call engineer never finishes reading. Fix: constrain to the top 3–5 causes by frequency; link out for the long tail.
- **Opinion-free.** The runbook says "investigate the cache"; it doesn't say "cache hit rate is at X < 0.3 and this is the signature, so the cause is almost always one of [A, B, C]." Fix: author with named likely causes, in order of base-rate frequency.
- **Stale.** The runbook references `vllm-engine-v0.4` flags; the fleet is on `v0.7` and the flags moved. Fix: `last reviewed` date in frontmatter; quarterly review cadence; CI that fails on unreferenced runbooks.

## Testing a runbook

A runbook that has never been read in anger is a hypothesis, not a procedure. Three ways to test:

- **Tabletop drill** — a quarterly session where the on-call team reads the runbook against a scripted synthetic incident and critiques it; tracked in chapter 04's post-mortem cadence.
- **Induced regression** — exercise 03 of module 110 induces TTFT / preemption / queue-saturation patterns; use those sessions to dry-run runbooks 1, 3, 8.
- **Historical replay** — take the last three real incidents' data; have an engineer who did not run them dry-run the current runbooks against the incident's recorded metrics. If the runbook doesn't land on the right cause, revise.

## Summary

- One template for every inference runbook: symptoms → first bisection → likely causes → first fixes → escalation → rollback → references.
- Keep each to one page; make commands copy-pasteable and pinned to the live fleet.
- The top-ten inference incidents are latency / cache / preemption / silent drift / NCCL / GPU-fault / crash-loop / admission-reject / KV-OOM / streaming-close; each has its own axis-set entry-point from module 110.
- `last reviewed` and `owned by` are non-decorative; schedule reviews or the runbook rots.
- Test every runbook before it is read at 2 AM — tabletop, induced-regression, or historical replay.
- Chapter 04 is where the runbook that didn't fit the incident becomes a post-mortem action item.
