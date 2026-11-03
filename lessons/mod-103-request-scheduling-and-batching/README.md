# Module 103 — Request Scheduling, Continuous Batching, and Chunked Prefill

> **Role:** AI Inference Engineer · **Level:** L35 (AI Infrastructure family) · **Hours:** 16

Module 102 cracked each engine open and named the five layers. This module zooms in on layer 3 — the **scheduler** — and makes it operational. The scheduler is where goodput is made or lost: it decides which request runs this iteration, how prefill tokens and decode tokens are mixed into one batch, and what to do when the KV pool fills up. By the end of this module you can author a scheduler configuration that hits a stated TTFT / TPOT SLO under a stated traffic shape, defend it against a Pareto curve, and alert on it before it degrades.

By the end of it you can:

- Walk the **continuous-batching inner loop** at operator depth — iteration-level scheduling, token-level batching, the prefill vs. decode phase split, the arrival-vs-run queue split, and the admission-control decision. Point at each in the vLLM / SGLang / TRT-LLM source.
- Author **scheduler configurations** against a stated SLO and traffic shape — `max-num-seqs`, `max-num-batched-tokens`, `max-prefill-tokens`, prefill-chunk-size, scheduler-delay, priority / fairness. Defend the choices against a throughput-vs-latency Pareto curve.
- Deploy **chunked prefill (SARATHI-shape)** and reason about the **TTFT / TPOT trade-off** — when raising chunk size helps, when it hurts, and how the knob interacts with `--max-num-batched-tokens`.
- Deploy **preemption policies** — swap-out vs. recompute, PagedAttention block-level eviction, request-level starvation guards — and know which to prefer at a given KV-pool size and traffic shape.
- Instrument the scheduler — queue depth, admission rate, preemption rate, prefill / decode token mix, iteration time — in **Prometheus** and feed those into **SLO alerts** (depth lives in module 110).
- Cite the **boundary to the performance peer** — iteration-level scheduler knobs are owned here; the kernels those iterations dispatch to are owned at [`ai-infra-performance`](https://github.com/ai-infra-curriculum/ai-infra-performance-learning).

> **Prerequisites:** [Module 101](../mod-101-inference-engineering-position/README.md) (vocabulary — TTFT / TPOT / goodput) and [Module 102](../mod-102-inference-engine-internals/README.md) (five-layer anatomy; vLLM / SGLang / TRT-LLM entry points). The scheduler flags introduced at the surface in module 102 are the knobs we go deep on here.

## Boundary against `ai-infra-performance`

The scheduler is a *control-plane* object: it answers "which request runs this iteration, with what KV budget, mixing which prefill chunk and which decode step." The kernels that the chosen batch dispatches to — attention, GEMM, FP8 epilogues, Flash Attention variants — are owned at the peer. If your diagnosis lands on "the scheduler keeps picking the wrong batch shape," you own the fix. If it lands on "this attention kernel is 20% slower than the specialist's benchmark suggests," open an issue to the performance peer and ship the engine upgrade when their fix lands.

Rule of thumb: a scheduler fix is a config change, a policy change, or an engine PR to the scheduler module. A kernel fix is a `.cu` / `.cuh` / Triton-DSL file. Watch the file extension on the fix.

## Boundary against module 104 (KV-Cache Management)

The scheduler calls into the KV allocator every iteration — `allocate`, `append_slot`, `swap_out`, `swap_in`, `free`. This module treats the allocator as a *dependency*: we care that it has blocks, that it can swap, that it can evict. [Module 104](../mod-104-kv-cache-management/README.md) goes deep on how the allocator works internally — block tables, radix trees, tiered KV stores, prefix-cache invalidation. If a scheduler problem turns out to be a block-manager problem, follow the thread into module 104.

## Chapters

1. [**The Continuous-Batching Inner Loop**](01-continuous-batching-inner-loop.md) — iteration-level scheduling vs. request-level batching; how the arrival queue and run queue interact; the prefill phase vs. the decode phase, and why they are compute-bound vs. memory-bound; the admission-control decision each iteration; the mental model you need before any knob makes sense.
2. [**Scheduler Configuration for a Stated SLO**](02-scheduler-configuration-for-slo.md) — translating a workload spec (RPS, prompt-length distribution, output-length distribution, TTFT / TPOT target) into concrete values for `max-num-seqs`, `max-num-batched-tokens`, `max-prefill-tokens`, scheduler-delay, and the priority / fairness policy; the throughput-vs-latency Pareto curve as the tuning compass.
3. [**Chunked Prefill and the TTFT / TPOT Trade-off**](03-chunked-prefill-ttft-tpot-tradeoff.md) — the SARATHI insight (piggyback decode steps on chunked prefill); the chunk-size knob and its interaction with `--max-num-batched-tokens`; when chunk size helps (long prompts without blocking decode), when it hurts (short-prompt workloads where chunking is pure overhead), and how to pick a starting value.
4. [**Preemption Policies**](04-preemption-policies.md) — swap-out to CPU vs. recompute; PagedAttention block-level eviction; the recompute vs. swap break-even; starvation and the per-request age / priority guard; how to tell swap thrash from legitimate preemption on a dashboard.
5. [**Instrumenting the Scheduler**](05-instrumenting-the-scheduler.md) — the Prometheus axes every scheduler needs (queue depth, admission rate, preemption rate, prefill / decode token mix, iteration time), the derived SLOs built on top of them, and how the alert feed is wired to module 110's SLO engine.

## Exercises

The exercises carry each chapter into a hands-on drill you keep as a reusable artefact for the rest of the track:

1. [**Continuous Batching Inner-Loop Walkthrough**](exercises/exercise-01-continuous-batching-inner-loop-walkthrough.md) — trace one iteration end-to-end in a running engine; annotate the scheduler loop; print the admission decision for a captured workload trace.
2. [**Scheduler Tuning for a Stated SLO and Traffic Shape**](exercises/exercise-02-scheduler-tuning-for-stated-slo-and-traffic-shape.md) — given a workload spec and TTFT / TPOT target, author a scheduler config, defend it against a Pareto sweep, and document the regression test.
3. [**Chunked Prefill TTFT / TPOT Trade-off Sweep**](exercises/exercise-03-chunked-prefill-ttft-tpot-tradeoff-sweep.md) — sweep chunk size and `max-num-batched-tokens` on a long-prompt + decode-heavy mix; plot the Pareto; pick a production value and name the risks.
4. [**Preemption Policy Swap-vs-Recompute Drill**](exercises/exercise-04-preemption-policy-swap-vs-recompute-drill.md) — force the engine into preemption under both policies; measure recovery TTFT / throughput; author the policy-pick decision rule.
5. [**Scheduler Metrics into Prometheus**](exercises/exercise-05-scheduler-metrics-into-prometheus.md) — scrape the scheduler metrics, build the five load-bearing dashboards, author one SLO alert per axis, and verify by provoking it.

## Resources

See [`resources.md`](resources.md) for primary / citable references — the Orca paper, SARATHI / SARATHI-Serve, PagedAttention, DistServe, the vLLM / SGLang / TRT-LLM scheduler docs and source trees, and the Prometheus / Grafana references the exercises build on.

## Where this fits in the curriculum

- **Before:** [Module 102 — Inference-Engine Internals and Operational Fluency](../mod-102-inference-engine-internals/README.md) — the five-layer anatomy and the scheduler's surface-level flag set.
- **After:** [`mod-104-kv-cache-management`](../mod-104-kv-cache-management/README.md) (the KV allocator the scheduler calls into); [`mod-110-observability-slo-and-cost`](../mod-110-observability-slo-and-cost/README.md) (the SLO stack the scheduler metrics feed into).
- **Peer-track hand-off:** when a scheduler-tuning drill lands on a kernel-level finding (attention kernel slower than expected, FP4 GEMM epilogue regression), the hand-off is to [`ai-infra-performance-learning`](https://github.com/ai-infra-curriculum/ai-infra-performance-learning). This module teaches you to *recognise and route* that hand-off, not to execute it.
