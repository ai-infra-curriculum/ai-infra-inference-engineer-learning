# Chapter 01 — Canary and Shadow Rollouts: Response Diffing, Semantic Equivalence, SLO-Based Auto-Rollback

> A stateless web service canaries on request count and error rate. An inference service canaries on a token-stream, a stochastic decoder, a KV-cache that has to warm, an SLO measured in tokens-per-second, and a semantic-equivalence tolerance a product manager has to sign. This chapter names the primitives.

## Motivation

A quantisation change on a 70B model passes every unit test, every engine smoke-test, and every HTTP status check. It also drops a long-context benchmark score by 2 %. Every latency SLO stays green. Every GPU-utilisation dashboard stays green. The response-length distribution barely shifts. Two weeks later a product manager notices retention bending downward on power-users of long-document chat. The incident did not surface because *the deploy gate had no eye for semantic regression*.

That is the first failure mode this chapter prevents. The second is the opposite: a minor refactor to the engine's sampler changes bit-level outputs under `temperature=0` without any semantic shift, and an over-strict response-diff rolls the change back on byte inequality. The canary primitive has to be *sensitive to semantic regression* without being *tripped by stochastic non-determinism that was always there*. The acceptance threshold is the whole game.

By the end of this chapter you can:

- Decide between a canary rollout, a shadow rollout, and a hard cutover for a given change type.
- Author a response-diff score against a stated semantic-equivalence tolerance and defend the threshold.
- Author a canary-weight ramp with per-step promotion gates against the inference axis set.
- Wire SLO-based auto-rollback on multi-window multi-burn-rate alerts across TTFT / TPOT / cache-hit / preemption / admission-reject.
- Avoid the five cardinality / sampling / routing traps that make a canary useless.

## Canary vs. shadow vs. hard cutover — pick the right primitive

Three primitives, three change categories.

| Primitive | What it does | Change categories that fit |
|---|---|---|
| **Canary** | Route a small fraction of *live* traffic to the candidate; score per-request outcomes and SLOs; ramp up on success. | Scheduler-flag tweak, engine-version upgrade with stable API, routing-policy change, adapter promotion, non-breaking API change. |
| **Shadow** | Duplicate inbound traffic; send both to primary and candidate; return the primary's response to the client; score the candidate's response offline. | Quantisation / model-version change where output is expected to differ, speculative-decoding draft-model promotion, new attention backend, response-caching policy change. |
| **Hard cutover (blue-green or drain-and-swap)** | Replace the whole model version at once, guarded by pre-promotion eval and a fast rollback. | Changes that *must not* interleave versions mid-stream (e.g. a model with a new tokenizer where mixed-version outputs would be user-visible garbage), and changes where the dual-fleet / dual-load decision in chapter 02 drives the choice. |

Rule of thumb: **if the candidate is expected to produce different outputs for the same input, you are shadow-rolling. If output should be equivalent, you are canarying.** Mixing them produces a canary that false-alarms on every quantisation change, or a shadow-roll that never promotes because nobody runs the gate.

### Change-type → primitive map

| Change | Expected to change outputs? | Primitive | Rationale |
|---|---|---|---|
| Scheduler flag (`max-num-seqs`, chunked-prefill size) | No (only latency) | Canary | Response-equivalence tolerance is strict; the SLO deltas are what graduate. |
| Engine version (same model build) | Near-identical outputs | Canary with widened tolerance | Kernel non-determinism shifts; equivalence should allow small distribution-level drift. |
| Quantisation (FP16 → FP8 / INT4) | Expected shift | Shadow + eval-gate (chapter 05) + hard cutover | Outputs will differ pointwise; only a semantic / benchmark gate is meaningful. |
| Model fine-tune / new checkpoint of same model name | Expected shift | Shadow + eval-gate + canary-on-percentage | If passing eval, canary ramps the behavioural change; roll back on SLO burn or user-visible regression. |
| LoRA adapter promotion | Expected per-adapter shift | Shadow of adapter-tagged traffic + eval-gate | Scope the roll to the tenant / adapter route. |
| Speculative-decoding draft-model promotion | Outputs equivalent at `temperature=0`, latency shift | Canary + acceptance-rate gate | Draft-target decoding is equivalence-preserving; the gate is acceptance rate and TPOT. |
| Routing / load-balancer change | No output change | Canary on routing weights | Pure infra change; SLO-gate it. |
| Hardware swap (H100 → H200, add NVLink) | Negligible output change, big latency shift | Canary + goodput gate | Treat like a hardware-side "scheduler flag." |

The map is a starting point; your fleet will specialise it.

## Response diffing with semantic equivalence — the tolerance vocabulary

The oldest wrong idea in inference rollout is *byte-equal responses between old and new*. Byte-equality does not hold even on the *same* model, because:

- Non-associative floating-point: `(a + b) + c ≠ a + (b + c)`; batched decoding changes the reduction order.
- Kernel non-determinism: GEMM dispatch may pick different algorithms per batch shape.
- Scheduler effects: continuous batching mixes a given sequence with different neighbours across runs.
- Sampling: anything above `temperature=0` is explicitly stochastic. Even at `temperature=0`, the argmax ties can resolve non-deterministically under reduction reordering.

Response diff therefore means *distance* under a tolerance, not equality. The toolkit:

| Equivalence score | What it measures | When to use | Typical tolerance |
|---|---|---|---|
| **Exact-string equality** | Byte identity | Never for free-form text; sometimes for structured-output JSON on a fixed schema | 100 % only for parsed-JSON equivalence (field-by-field), not byte |
| **Edit distance (normalised)** | Character / token-level diff | Short deterministic outputs (classifications, labels, JSON values) | Threshold product-chosen; often 0.02–0.05 for structured outputs |
| **Token-level perplexity delta** | Model-scored shift in distribution | Comparing two model versions on held-out eval text | Small; defer to module 105's tolerance table |
| **Embedding similarity (cosine)** | Semantic similarity via an embedding model | Open-ended chat, summarisation, reasoning | Threshold product-chosen; typical 0.90–0.98 depending on task |
| **Rubric / LLM-as-judge score** | A judge model scores both responses against a task-specific rubric | Nuanced tasks where embeddings miss quality shifts | Threshold from product and the eval peer |
| **Structured-output schema conformance** | Does the candidate still emit valid JSON / tool-calls? | Any structured-output path | Pass / fail; schema-conformance regression is a *rollback* signal |
| **Refusal-rate delta** | How often did the candidate refuse vs. the primary? | Any alignment-affecting change | Threshold per workload class; refusal spike is a rollback signal |

Two realities to anchor:

- **The threshold is a product decision.** "Our cosine-similarity gate is 0.94" is not a math fact; it is a product promise about how different a response can be before the user notices. The eval peers (chapter 05) carry half of that decision; the inference engineer carries the SLO half.
- **Use several scores, not one.** An embedding score with a refusal-rate gate and a schema-conformance gate catches different regressions. A single-score gate is a single-point-of-failure gate.

<!-- needs-research: embedding-similarity thresholds vary by model-family and workload — defer absolute numbers to the eval peers and your own holdout data; chapter 05 formalises the hand-off. -->

### Response-diff emission shape

For a shadow-traffic rollout, on every mirrored request:

1. Capture `(prompt, params, seed)` plus the primary response.
2. Send the same `(prompt, params, seed)` to the candidate. (Seed matters — see below.)
3. Score equivalence with the per-task scorer(s).
4. Emit a per-request record (sampled, not every request if QPS is high) to a long-term store — Parquet on S3, BigQuery, Snowflake — so you can run offline aggregate analysis.
5. Emit summary metrics with bounded cardinality (per workload-class, per candidate-version):
   - `candidate_response_diff_score` as a histogram.
   - `candidate_refusal_rate` as a counter pair.
   - `candidate_schema_conformance_rate` as a counter pair.

The per-request log goes to an offline store because high-cardinality request-level data is the wrong fit for Prometheus. The histogram goes to Prometheus because the quantiles drive the auto-rollback (see below).

### Seeding and determinism caveats

A canary where candidate and primary receive different seeds is comparing two stochastic samples; the "diff" includes noise the user never sees. Shadow traffic should propagate `seed` deterministically if the API supports it; where it doesn't (reasoning models without an exposed seed), the equivalence gate must tolerate sampling noise, which means either higher thresholds or averaging over many requests per prompt.

## The canary-weight ramp with per-step promotion gates

The ramp is a sequence of traffic fractions with a *promotion gate* between each step. A standard conservative ramp:

| Step | Traffic to candidate | Minimum dwell | Promotion gate |
|---|---|---|---|
| 0 | 0 % (shadow-only or pre-flight) | pre-flight eval passed | Chapter 05 eval gate green; smoke test on candidate replicas green; warm-up complete |
| 1 | 1 % | 15 minutes | SLO burn-rate alerts green for 15 min; response-diff score within tolerance; schema-conformance ≥ primary |
| 2 | 5 % | 1 hour | 15-min deltas within tolerance; no new error classes; cost / token not above budget |
| 3 | 25 % | 2 hours | Same gates plus a *per-workload-class* check (long-context included if applicable) |
| 4 | 50 % | 2 hours | Same gates plus a 1-hour burn-rate check on the 30-day SLO window |
| 5 | 100 % | — | Hold at 100 % under watch for 24 hours before "complete" — rollback remains available |

Three disciplines:

- **Dwell times must exceed the burn-rate detection window** of the alerts you expect to catch regressions. If the 1-hour burn-rate alert needs 15 minutes of data to fire, a 10-minute dwell is a 1 % traffic ramp that cannot see the signal it is gating on.
- **Dwell counts tokens, not wall-clock.** A 15-minute dwell at 10 QPS is not comparable to a 15-minute dwell at 10 000 QPS. For a low-QPS canary, dwell by *tokens served on the candidate* (e.g. "500k tokens from the candidate before promotion").
- **Per-workload-class gates matter.** A 1 % canary at 10 000 QPS sees ~100 requests per second. If `long_context` is 1 % of traffic, the canary sees 1 long-context request per second — not enough to detect a regression specific to that class before the next ramp step. The gate at step 3 should *require* a minimum number of long-context samples to have passed, not just "15 minutes elapsed."

### Promotion gate logic

A promotion gate is a conjunction of:

- **SLO gates** — no burn-rate alert fired on any of the chapter-01 axis set from module 110 during the dwell: TTFT, TPOT, ITL-P99, cache-hit, preemption, admission-reject, error-rate, token-availability.
- **Response-diff gates** — primary scoring stat (e.g. P50 cosine similarity ≥ 0.94, refusal-rate delta within ± 0.02, schema-conformance ≥ 0.999) in a window.
- **Cost gates** — `$ / 1M tokens` not above the budget set in module 110 chapter 04.
- **Error-class gates** — no new 5xx subtype, no spike in a specific exception, no new NCCL warning in logs.

Any gate fail → halt the ramp and alert. Any *burn-rate* alert during a step → auto-rollback (see below).

## SLO-based auto-rollback

Auto-rollback moves the candidate's traffic weight to 0 without waiting for a human. It fires on:

- A multi-window multi-burn-rate alert on *any candidate-scoped* SLO.
- A response-diff score that drops below tolerance for a window.
- A schema-conformance rate drop.
- A refusal-rate spike beyond threshold.
- An error-rate spike localised to the candidate (requests tagged with candidate version).

The design rules:

- **Alerts are candidate-scoped, not fleet-scoped.** If the fleet TTFT P99 is 2.1 s but the *candidate's* TTFT P99 is 3.5 s while the *primary's* is 1.5 s, the fleet-wide number is a mix. Candidate-scoped alerts use a `version` label on histograms (bounded cardinality: at most `active_versions + 1`).
- **Rollback is fast.** Traffic-weight change should propagate in seconds (service mesh, ingress router, LB) — not minutes. The rollback itself must not require a redeploy.
- **Rollback preserves in-flight streams.** Requests already routed to the candidate finish on the candidate (unless the burn is a hang, in which case the circuit-breaker kicks — see module 102's engine failure-modes chapter). New requests go to the primary. Chapter 02 names the exact streaming-session semantics; canary respects the same rule.
- **Rollback emits an incident.** Any auto-rollback files an incident ticket automatically. Chapter 04 teaches what to do with it.

### Burn-rate alert shape for a canary

For an SLO of "P95 TTFT < 500 ms for chat," a candidate-scoped multi-window multi-burn-rate alert pair (SRE workbook pattern, specialised):

```
# Fast burn (short window, high sensitivity) — pages on ~2% of a 30-day budget consumed in ~1h
expr: >
  (
    sum(rate(ttft_seconds_bucket{le="0.5",workload_class="chat",version="candidate"}[1h]))
    /
    sum(rate(ttft_seconds_count{workload_class="chat",version="candidate"}[1h]))
  ) < (1 - 14.4 * (1 - 0.95))
for: 2m
severity: page-rollback

# Slow burn (longer window) — pages on ~5% of a 30-day budget consumed over ~6h
expr: >
  (
    sum(rate(ttft_seconds_bucket{le="0.5",workload_class="chat",version="candidate"}[6h]))
    /
    sum(rate(ttft_seconds_count{workload_class="chat",version="candidate"}[6h]))
  ) < (1 - 6 * (1 - 0.95))
for: 15m
severity: page-rollback
```

Details:

- The `version="candidate"` label is the whole point; use `{active_versions} + 1` labels only.
- `0.95` is the SLO target; the `14.4` and `6` are standard burn-rate ratios for the 1h and 6h windows against a 30-day SLO window (see SRE workbook).
- `page-rollback` is routed to the rollback automation, not to a human first. Chapter 04 teaches what the human does in the subsequent incident.

<!-- needs-research: exact Prometheus metric names (`ttft_seconds_bucket` here) are placeholders; the real labels and buckets come from module 110 chapter 03 and your pinned engine's exposition. Verify. -->

## Smoke test and warm-up before the first percent

A cold candidate replica fails any TTFT burn-rate within seconds of its first request — its KV cache is empty, its CUDA graphs aren't captured, its weights aren't paged in. Jumping straight to 1 % live traffic produces a false-positive rollback. The sequence before step 1:

1. **Deploy the candidate replicas.** `TP` / `PP` sized correctly (module 107); liveness probe passes.
2. **Warm them up with pre-recorded traffic.** Replay a representative traffic sample (chapter 03 of module 110's trace capture is one source) for a few minutes until TTFT and TPOT stabilise; CUDA graphs captured; prefix cache has representative hits.
3. **Smoke-test with a canonical fixture.** 50–200 fixed `(prompt, params, seed)` pairs with expected-equivalent outputs; equivalence scores above tolerance across the fixture; schema-conformance on every structured-output case.
4. **Only then** open the live canary weight to 1 %.

The warm-up step is where most rollout-automation tooling is thin; invest in it.

## Infrastructure choices — where the canary primitive lives

The canary runs at the layer that owns traffic routing. Common placements:

- **Service mesh** (Istio, Linkerd) — weighted `VirtualService` / `TrafficSplit`; good for Kubernetes fleets with existing mesh; emits the per-version metric labels naturally.
- **Progressive-delivery controllers** (Argo Rollouts, Flagger) — wrap a Kubernetes `Deployment` with the ramp-and-gate loop; integrate with Prometheus for the SLO gate; well-documented for stateless services, needs adaptation for streaming SSE and WebSocket (see chapter 02).
- **Application-level router** (LiteLLM-shape, in-house) — if your fleet already has a model-aware router (module 106), the canary weight is a routing-table field; latency is lowest but you write the ramp-and-gate yourself.
- **Ingress controller** (NGINX, Envoy) — feasible for simple HTTP paths; awkward for streaming because weights can change mid-stream if the ingress re-evaluates.

Three implementation rules:

- **Weight changes must not affect in-flight streams.** The router selects the backend at *admission*, not per-chunk. A router that re-evaluates mid-stream will interleave versions and the diff score will be nonsense.
- **The `X-Model-Version` response header is non-optional.** Chapter 02 names the header formally; the canary needs it on day one so the client telemetry can attribute per-version. Set it as early as possible in the response (before first token flushes for streaming) so SSE events are version-tagged.
- **Idempotency keys are honoured across versions.** A client retry during a ramp must land on the same version if the key matches an in-flight request — otherwise a retry shadows the canary into the primary count. This is usually a router feature; verify.

## Shadow traffic — the pattern for "outputs will differ"

Shadow traffic is a canary where the user never sees the candidate's output. Both primary and candidate receive the request; the primary's response is returned; the candidate's response is scored offline. The shape:

```
client -> ingress -> primary       (returns to client)
              \
               -> candidate (buffered; discarded after scoring)
```

Three mechanics:

- **Mirror at the request level, after authentication.** Do not mirror pre-auth; it duplicates billing and auth load.
- **Rate-limit the mirror.** For quantisation shadow-rolls, you likely don't need 100 % mirror; 10 % is a cost-effective rate for scoring.
- **Timebox the candidate.** Candidate responses are discarded after the scoring deadline (e.g. 2× the primary's E2E) so a slow candidate doesn't starve the mirror.
- **Separate the GPU budget.** Candidate replicas run on their own node pool so their occupancy doesn't preempt primary KV or scheduling. Module 109 names the warm-pool pattern.

### Shadow score aggregation

Shadow scores are aggregated offline, not alerted on individual requests. A typical pipeline:

- Per-request scoring record → object-storage (Parquet on S3).
- A nightly / hourly Spark / BigQuery / DuckDB job → summary table per `(candidate_version, workload_class, task_type)`.
- A dashboard panel for the summary table — the thing a product partner, the eval peer (chapter 05), and the inference engineer look at together.
- A hand-off to the eval peer: the shadow's scored-and-held fixtures become new eval data (chapter 05's fixture back-feed).

Shadow traffic does *not* promote by itself. It produces the evidence that lets a hard-cutover (chapter 02) or a canary (this chapter) be scheduled.

## Five canary traps and how to avoid them

- **Trap 1: byte-equal response diff.** Outputs will never be byte-equal; the canary false-alarms every time. The fix: semantic-equivalence scoring with a product-chosen threshold.
- **Trap 2: fleet-scoped SLO alert.** The candidate's regression is diluted by 99 % primary traffic; the alert never fires. The fix: candidate-scoped version-labelled alerts.
- **Trap 3: no warm-up.** The candidate is cold at step 1; latency SLO burns; auto-rollback fires on a false-positive. The fix: smoke test + warm-up before the first percent.
- **Trap 4: mid-stream version flip.** The router re-evaluates traffic weight mid-stream; a response is half old / half new. The fix: router selects at admission, not per-chunk.
- **Trap 5: per-workload-class blindness.** The ramp dwell is wall-clock, not sample-count; low-volume classes don't accumulate enough samples to detect their regressions. The fix: per-class token-count gates at each ramp step.

## Worked example — a quantisation change from FP16 to FP8

The change: a 70B chat model, served at `TP=8`, re-quantised from FP16 to FP8. Expected to shift output distribution; expected to roughly halve decode latency on the same hardware class. Walkthrough:

1. **Pre-flight eval** (chapter 05) — the ML-eval peer runs the capability benchmark on the FP8 build; threshold green. The AI-eval peer runs the application eval set; threshold green with a tolerance of ± 1.5 % on task scores.
2. **Shadow traffic rolled first** at 10 % mirror rate, for 24 hours — response-diff aggregates look like: P50 cosine similarity 0.96 to primary, refusal-rate delta +0.003, schema-conformance 0.9995 (primary is 0.9998), task-score distribution matches eval. Decision: proceed.
3. **Warm-up** — pre-recorded traffic for 10 minutes against the candidate replicas; TTFT median stabilises at 180 ms vs. primary 410 ms; CUDA graphs captured.
4. **Smoke test** — 200 canonical fixtures; all equivalence scores above tolerance; schema-conformance 100 %.
5. **Canary ramp:**
   - 1 % for 15 min → gates green (longer dwell than usual because candidate scoring continues against mirrored primary responses).
   - 5 % for 1 h → gates green; `$ / 1M tokens` down 28 %; per-workload-class check includes long-context (10 % of traffic in step 2, so ~0.5 % overall — adequate at this step).
   - 25 % for 2 h → one `page-rollback` alert on ITL-P99 for the `agent` class. Auto-rollback fires to 0 %. Incident opens. Chapter 04 kicks in.
6. **Investigation finds** a FlashAttention variant selection difference under long-context FP8 batches; a flag change on the candidate fixes it. New candidate build; restart from step 1.
7. **Second attempt** passes to 100 %; held for 24 hours under watch; rollout "complete." The failed ITL-P99 case becomes a permanent regression fixture (chapter 04 back-feed) for the next FP8 change.

Every one of those steps is enabled by *one* primitive from this chapter and one *axis* from module 110. Together, the two modules make a quantisation rollout a controlled, reversible operation instead of a product crisis.

## Summary

- Canary, shadow, and hard-cutover are three primitives for three change categories. Pick by whether outputs are expected to shift.
- Response diff is distance under a tolerance, not equality. Use a bundle of scores (embedding similarity, refusal-rate, schema-conformance, rubric) and defend the thresholds with the eval peers.
- Canary ramp is a sequence of (traffic %, dwell, gates). Dwell counts tokens, not wall-clock; per-class gates matter.
- Auto-rollback runs on candidate-scoped multi-window multi-burn-rate alerts across the inference axis set — plus response-diff, schema-conformance, and refusal-rate gates.
- Smoke-test and warm-up before the first percent, or the cold-candidate false-positive eats your rollout.
- Keep weights stable mid-stream, set `X-Model-Version` on day one, honour idempotency keys across versions.
- Shadow traffic is the pattern for changes where output *will* differ; it feeds eval-set back-feed and schedules the hard cutover (chapter 02).
- Chapter 02 picks up for changes that cannot canary cleanly; chapter 03 is the runbook surface when an auto-rollback fires; chapter 04 is the RCA; chapter 05 is the upstream eval contract.
