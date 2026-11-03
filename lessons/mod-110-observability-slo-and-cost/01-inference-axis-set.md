# Chapter 01 — The Inference Axis Set: TTFT, TPOT, ITL, Cache-Hit, Preemption, DCGM, NCCL

> A Flask app's observability axis set is three numbers wide: request rate, error rate, latency P99. An inference engine's axis set is twelve numbers wide — because a request is not atomic, the GPU is not a CPU, the cache is a cost line item, and the collective is a critical-path contributor. This chapter names the twelve axes, what each one measures, how to emit each one, and the common failure modes that make an axis useless at the moment you need it.

## Motivation

An on-call engineer gets a page: "chat latency P99 is 4.1 s, breaching the 2 s SLO." Their dashboard has one panel — `histogram_quantile(0.99, http_request_duration_seconds_bucket)`. The panel is red. Everything else is "green" because everything else is CPU and memory, which don't move much for a GPU workload. They can see *that* something is wrong; they cannot see *where*.

In an inference stack, the "where" is one of six places:

- **Admission:** the request waited at the ingress layer before the engine saw it.
- **Queue:** the scheduler had too many sequences and the request waited in queue.
- **Prefill:** the GPU was busy encoding a long prompt and it took time.
- **First decode:** the first-token forward pass stalled (a cold batch, a KV cache miss, a slow attention backend).
- **Decode (TPOT):** steady-state per-token latency crept up (contention, HBM pressure, scheduler preemption).
- **Decode (ITL):** steady-state is fine on average but individual tokens stall (GC, NCCL stragglers, preemption).

A P99 of "the request" that aggregates all six is uninformative. An axis set that separates them turns the on-call hour from a guess into a bisection. By the end of this chapter you can:

- Name every axis in the inference set and the unit it is measured in.
- Decide whether each axis is a counter, a histogram, a gauge, or a sampled trace.
- Emit each axis from the engine, the scheduler, the KV allocator, or DCGM.
- Avoid the three cardinality traps that make an axis useless.
- Separate the "signal" axes (feed SLOs — chapter 02) from the "diagnostic" axes (feed dashboards and traces).

## The six latency axes

A request is a token stream. The latency decomposition that an inference engineer owns is:

| Axis | Unit | Boundary | What it measures |
|---|---|---|---|
| **E2E latency** | seconds | client-start → client-end | the full client-visible latency; includes network, retries, auth, routing, admission, queue, prefill, decode, stream-close |
| **TTFT** (time to first token) | seconds | engine-admit → first-token-emitted | critical path for the first byte of the response; dominates perceived latency for streaming UIs |
| **Queue time** | seconds | engine-admit → first-step-scheduled | subset of TTFT; the scheduler's share |
| **Prefill time** | seconds | first-step-scheduled → first-token-emitted | subset of TTFT; the GPU's share for encoding the prompt |
| **TPOT** (time per output token) | seconds / token | first-token → stream-close, averaged | steady-state decode rate; dominates total latency for long responses |
| **ITL** (inter-token latency) | seconds / token | per-token, individually | the TPOT distribution; captures jitter and stalls that an average hides |

Three invariants to carry through:

- `E2E ≈ network + admission + TTFT + (output_tokens - 1) × TPOT + stream-close`. The decomposition is additive; the dashboard must respect that.
- `TTFT = queue + prefill + first-token-step`. The three sub-components identify who is slow (scheduler, GPU-compute, or an engine-internal stall).
- `TPOT_mean` and `ITL_P99` can disagree by an order of magnitude. A clean TPOT with a `P99(ITL)` that spikes is the preemption signature; TPOT is fine on average but individual tokens stall for 500 ms when the engine evicts the sequence.

### Emission

The engine must emit the components, not just the aggregate. The modern engines (vLLM, SGLang, TensorRT-LLM through the Triton `tensorrtllm_backend`) expose Prometheus histograms covering most of these. The exact metric names drift across releases — the vLLM metrics reference is at <https://docs.vllm.ai/en/latest/serving/metrics.html>; SGLang's is in its documentation at <https://docs.sglang.ai/>; TRT-LLM / Triton exposes a metrics endpoint documented at <https://github.com/triton-inference-server/server/blob/main/docs/user_guide/metrics.md>.

<!-- needs-research: engine-specific metric names (`vllm:time_to_first_token_seconds`, `sglang:...`, `nv_inference_queue_duration_us`) rotate across releases. Verify against the pinned engine's `/metrics` output and release notes rather than against this chapter's example names. -->

Example instrumentation shape in Python (for a middleware you may still need to add — chapter 03 names when you have to synthesise versus when the engine already emits):

```python
# At the request's admission:
admission_time = time.monotonic()

# When the engine schedules it (first iteration):
queue_duration_seconds.labels(workload_class="chat").observe(time.monotonic() - admission_time)

# When the first token is emitted:
ttft_seconds.labels(workload_class="chat").observe(time.monotonic() - admission_time)

# For each subsequent token:
itl_seconds.labels(workload_class="chat").observe(time.monotonic() - last_token_time)
last_token_time = time.monotonic()

# At stream close:
tpot_seconds.labels(workload_class="chat").observe(
    (time.monotonic() - first_token_time) / max(1, output_tokens - 1)
)
```

Three emission disciplines to carry:

- **Use `time.monotonic()`, not `time.time()`.** NTP skew produces negative durations and lost samples.
- **Bucket histograms carefully.** TTFT buckets should span 10 ms → 60 s with roughly logarithmic spacing; TPOT buckets should span 1 ms → 1 s. Prometheus's defaults are optimised for HTTP services and give you poor resolution for inference.
- **Label by `workload_class`, not by `tenant`.** Chapter 02 and chapter 03 enforce this cardinality discipline; it is the first mistake every team makes.

## The scheduler-side axes

These describe the engine's internal state, not the per-request latency. They are *diagnostic* axes — some get SLOs (chapter 02), most feed dashboards and alerts.

| Axis | Unit | Shape | What it measures |
|---|---|---|---|
| **Prefill-decode token mix** | tokens / s | two counters | prefill tokens per second and decode tokens per second (two separate rates); their ratio reveals which regime the fleet is in |
| **Cache-hit rate** | ratio ∈ [0, 1] | counter / counter | fraction of prompt tokens served from the prefix cache (RadixAttention, vLLM auto-prefix-cache, SGLang) rather than recomputed |
| **Preemption rate** | ratio / events-per-second | counter | sequences preempted (swapped out or evicted) per unit time; signal of memory pressure and over-admission |
| **Admission-reject rate** | ratio / events-per-second | counter | requests rejected at admission (queue full, context too long, over-budget tenant); signal of back-pressure firing correctly |
| **In-flight sequences** | count | gauge | sequences currently in the batch; the thing the scheduler's `max-num-seqs` caps |
| **Queue depth** | count | gauge | sequences waiting to enter the batch; the primary autoscaler signal (chapter 02 of module 109) |
| **Prefill tokens-per-second** | tokens / s | gauge | GPU-side throughput for prefill; useful to compare across replicas |
| **Decode tokens-per-second** | tokens / s | gauge | GPU-side throughput for decode; the number cost-per-1M-tokens is derived from (chapter 04) |

Three failure modes to anticipate:

- **Cache hit rate 100 %.** Suspicious. Usually a metric bug (counting cache attempts that never fired), not a correctly-warmed cache. Verify against the engine's prefix-cache metrics directly (module 104, chapter on prefix caching).
- **Preemption rate `> 0.01` sustained.** The scheduler is thrashing. Usually `max-num-seqs` too high for the KV budget at the current prompt-length distribution. Chapter 05 (Pareto sweep) is where you find the right value; chapter 02 is where you alert on this.
- **Admission-reject rate `> 0` on an under-subscribed fleet.** Admission is rejecting on the wrong signal. Common cause: a per-tenant rate limit tripped by a batch job; the fleet itself has capacity. Dashboard should distinguish per-tenant admission reject from fleet admission reject.

Full scheduler-metric vocabulary is covered in module 103, chapter on scheduler metrics (`03-scheduler-metrics` or equivalent — verify current chapter numbering). This chapter catalogues them; module 103 teaches what each one *is*.

## The GPU-side axes — DCGM

DCGM-exporter (<https://github.com/NVIDIA/dcgm-exporter>, documentation at <https://docs.nvidia.com/datacenter/dcgm/>) is the GPU observability substrate — it runs as a DaemonSet, scrapes the GPU driver's metrics via NVML and DCGM's higher-level API, and exposes a Prometheus-compatible `/metrics` endpoint.

The DCGM axes an inference engineer reads daily:

| DCGM metric | Unit | What it measures | What to do with it |
|---|---|---|---|
| `DCGM_FI_DEV_SM_ACTIVE` | ratio ∈ [0, 1] | fraction of time the GPU's streaming multiprocessors (SMs) were active | reads 1.0 at single-request prefill *and* at saturated batch; useful for "GPU is doing something" but a bad SLI |
| `DCGM_FI_DEV_SM_OCCUPANCY` | ratio ∈ [0, 1] | fraction of SM-cycles where warps were resident | better than SM_ACTIVE for "is the kernel well-sized" |
| `DCGM_FI_DEV_FB_USED` | MiB | HBM in use | key signal for KV pressure and for "did we leak memory" |
| `DCGM_FI_DEV_FB_FREE` | MiB | HBM free | complement; the thing that drops before OOM |
| `DCGM_FI_DEV_POWER_USAGE` | Watts | GPU power draw | secondary utilisation signal; tracks compute better than SM_ACTIVE in a batched regime |
| `DCGM_FI_DEV_GPU_TEMP` | °C | GPU core temperature | signal for a thermal-throttle incident (module 113) |
| `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE` | ratio ∈ [0, 1] | tensor-core pipeline activity | reads low when the kernel isn't dispatching to tensor cores (e.g. fallback kernel on an unquantised GEMM) |
| `DCGM_FI_PROF_DRAM_ACTIVE` | ratio ∈ [0, 1] | DRAM pipeline activity | reads high in decode (memory-bandwidth-bound); reads lower in prefill (compute-bound). The diagnostic axis for prefill-vs-decode mix |
| `DCGM_FI_PROF_NVLINK_RX_BYTES` / `DCGM_FI_PROF_NVLINK_TX_BYTES` | bytes / s | per-GPU NVLink traffic | TP collective health across GPUs in a node |
| `DCGM_FI_PROF_PCIE_RX_BYTES` / `DCGM_FI_PROF_PCIE_TX_BYTES` | bytes / s | per-GPU PCIe traffic | PP / cross-node transfer health; disaggregated KV transfer (DistServe / NIXL) traffic |
| `DCGM_FI_DEV_XID_ERRORS` | count | NVIDIA kernel driver fault counter | hardware or driver incident signal; alert and page |
| `DCGM_FI_DEV_ECC_SBE_VOL_TOTAL` / `DCGM_FI_DEV_ECC_DBE_VOL_TOTAL` | count | correctable (SBE) and uncorrectable (DBE) ECC errors | DBE is a hard-failure signal; SBE drift is a canary for a bad GPU (module 113 wires this to incident alerts) |

The authoritative field-ID reference is <https://docs.nvidia.com/datacenter/dcgm/latest/dcgm-api/dcgm-api-field-ids.html>. The DCGM-exporter config file (`default-counters.csv` or similar) controls which fields are scraped; verify the current filename and layout against the exporter's README.

<!-- needs-research: DCGM profile metrics (`DCGM_FI_PROF_*`) require a profiling-capable GPU and may conflict with concurrent profilers (Nsight, CUPTI). The gating varies by GPU generation and by DCGM version; verify against the DCGM documentation and the GPU Operator release notes. -->

Three disciplines to carry:

- **SM_ACTIVE is not utilisation.** The CPU analogue — a CPU at 100 % busy — says the server is overloaded. A GPU at `SM_ACTIVE=1.0` says "a kernel was dispatched for the full window," which is true for one request and for a saturated batch. The utilisation that matters for cost (chapter 04) is `decode_tokens_per_second / peak_decode_tokens_per_second` — see chapter 04 for the derivation.
- **FB_USED is a plateau, not a leak.** The KV allocator (module 104) pre-allocates a block pool at startup. A high `FB_USED` sustained is normal; a `FB_USED` that drifts upward during a long run is a leak. Alert on the drift, not the absolute.
- **NVLink saturation is a TP bottleneck.** For a TP=8 replica on an H100 node, sustained NVLink traffic near the link's peak (~900 GB/s per NVLink v4 pair, ~1.8 TB/s aggregate on NVSwitch) means the collective is on the critical path. Chapter 05 (Pareto) uses this to recognise when more TP will not help.

## The NCCL time axis — for TP / PP workloads

For a replica with `TP > 1` or `PP > 1`, the collective time (AllReduce, AllGather, ReduceScatter, SendRecv) is a line item in the latency decomposition. The engine's forward pass is `compute + collective`, not `compute`. If the collective is 15 % of the decode step and you add a GPU to the replica without upgrading the fabric, decode gets *slower*.

Two emission paths:

- **Engine-exported.** vLLM, SGLang, and TRT-LLM expose timing hooks (sometimes gated behind a debug flag) that report per-collective time. Verify the current flag per engine release. The common pattern: the engine wraps `torch.distributed.all_reduce` with a `time.monotonic()` pair and records to a histogram.
- **Nsight Systems trace.** For a one-off bisection, record a Nsight Systems profile (<https://developer.nvidia.com/nsight-systems>) and read the NCCL time directly from the collective-ops section. This is a diagnostic tool, not a dashboard — see module 107's chapter on NCCL diagnostics for the methodology.

Suggested axis set to export when `TP > 1`:

| Axis | Unit | Shape |
|---|---|---|
| `nccl_time_allreduce_seconds` | seconds / collective | histogram |
| `nccl_time_allgather_seconds` | seconds / collective | histogram |
| `nccl_time_reducescatter_seconds` | seconds / collective | histogram |
| `nccl_bytes_allreduce_total` | bytes | counter |

The ratio `nccl_time_total / forward_time_total` is the "collective overhead" signal. For an inference workload on NVLink / NVSwitch, this ratio should sit well below 0.2 for a healthy replica; above that, chapter 05 (Pareto) and module 107's topology chapters say the parallelism shape is wrong for the model or the fabric.

<!-- needs-research: numeric thresholds for healthy "collective overhead" ratios depend heavily on model size, TP degree, and fabric (NVLink vs. InfiniBand vs. RoCE). Verify against your fleet's benchmark data rather than against a cross-fleet rule of thumb. -->

## The memory-pressure axis set

A composite signal, not one metric. The HBM pressure axis has two reliable sources:

- `DCGM_FI_DEV_FB_USED / DCGM_FI_DEV_FB_TOTAL` — HBM utilisation. Reads near 1.0 is "no more KV blocks will fit"; reads near 0 is "wasted budget."
- `vllm:kv_cache_usage_perc` (or the engine's equivalent) — a scheduler-visible KV-utilisation gauge. Verify the exact metric name against your engine.

The two together describe the pressure: the DCGM number says "the GPU is at X %"; the engine number says "the KV cache allocator is at Y %." The two disagree when the engine's block pool is sized smaller than the HBM available, which is a sizing decision from module 104.

Three failure modes:

- **High FB_USED, low KV cache usage.** Memory is held by someone — usually the model weights are larger than expected (did you load fp16 when fp8 was expected?) or an orphaned CUDA context. Diagnostic.
- **High KV cache usage, low preemption.** Scheduler is well-tuned; this is the correct steady state.
- **High KV cache usage, high preemption.** Over-admitting. Lower `max-num-seqs` or raise the preemption threshold; see module 103 and 104.

## Which axis is a counter vs. a histogram vs. a gauge vs. a trace

A common mistake: emitting everything as a Prometheus histogram. Prometheus histograms have fixed buckets, scrape at seconds granularity, and are summarised across a time window; they are not the right shape for every signal. The emission shape must match the question the axis answers.

- **Counter.** Monotonic, time-rate-able. Use for: cumulative tokens served, requests served, speculative tokens accepted, cache hits, cache attempts, admission rejects, preemption events. The derived rate (`rate(...[5m])`) is the operational signal.
- **Histogram.** Bucketed distribution with the ability to compute quantiles. Use for: TTFT, TPOT, ITL, queue-time, prefill-time, NCCL-time per collective. The P50 / P90 / P99 are the operational signals.
- **Gauge.** Point-in-time instantaneous value. Use for: in-flight sequences, queue depth, FB_USED, SM_ACTIVE, cache-utilisation. The last-value is the operational signal.
- **OTel trace.** A full span tree for one request. Use for: debugging a specific slow request, correlating admission + prefill + decode + downstream calls on one request, exploring "why did this request's TTFT spike at 2:03 AM" with a trace sample rather than a dashboard aggregate. Chapter 03 names the sampling policy.

A metric with the wrong shape is a metric that costs cardinality or hides signal. Three concrete examples:

- **Latency as a gauge.** `latency_last_seconds` loses the distribution. One slow request shows up as a spike; one hundred fast requests after it hide the spike. Use a histogram.
- **Token count as a histogram.** `tokens_per_request_bucket` with 10 buckets × 20 label combinations × 100 pods = 20k series for a signal that could be a counter. Use a counter; derive the distribution over the request-log if needed.
- **Cache-hit as a boolean gauge per request.** `cache_hit_last_request` doesn't aggregate. Use a counter pair (`cache_hit_total`, `cache_attempt_total`); the ratio is the operational signal.

## Cardinality discipline — the three traps

Prometheus is a cost multiplier on label cardinality. One careless label can turn a 100-series fleet into a 10M-series fleet. The traps are consistent across tracks:

- **Trap 1: `tenant_id` on the hot path.** 10k tenants × 20 metrics × 10 label combinations = 2M series. The cluster will spend more compute scraping than serving. The pattern: `tenant_tier` (`free` / `paid` / `enterprise` + a `top-N tenant_id for the top-N tenants + "other"`) carries most of the signal at bounded cardinality. The top-N table lives in a config and is refreshed daily.
- **Trap 2: `request_id` on anything other than a trace.** A unique per-request label is unbounded. The pattern: no `request_id` label on metrics; carry `request_id` *only* in OTel spans and in logs, with the trace sampling policy of chapter 03.
- **Trap 3: `model_name` with ad-hoc values.** If the client specifies the model and you label metrics by it, a typo or a drive-by test with `model="gpt-4o-latest-v2-test"` produces a permanent series. The pattern: validate against an allow-list at admission; reject or re-label unknown values.

A common cross-cutting check: Prometheus's `/api/v1/status/tsdb` reports the top label names by cardinality. Add it to the stack from day one and inspect weekly.

## Diagnostic vs. SLO axes — the split

Not every axis gets an SLO. Chapter 02 teaches the SLO methodology; this chapter sets the stage by naming which axes are candidates.

- **Candidate SLO axes** (will show up in chapter 02): `ttft_seconds` (per class), `tpot_seconds` (per class), `itl_seconds_P99` (per class, as a jitter SLO), `availability` (tokens served / tokens requested), `cache_hit_rate` (as a cost-SLO dependency), `cost_per_1m_tokens` (per class).
- **Operational axes** (dashboards + alerts, not SLOs): queue depth, in-flight sequences, preemption rate, admission-reject rate, prefill-TPS, decode-TPS, FB_USED, SM_ACTIVE, NVLink traffic, NCCL time per collective. These feed autoscaling (module 109) and diagnosis (module 113), not external contracts.
- **Hardware-event axes** (page immediately): XID errors, ECC DBE, driver crashes, OOM. These are incidents, not drift.

The dividing question: *does this axis go into a contract we tell product / finance?* If yes, it is an SLO axis and chapter 02 writes the target. If no, it is operational and lives on the dashboard.

## Worked example — decomposing a TTFT regression

An on-call page: "chat TTFT P99 spiked from 400 ms to 2.1 s at 09:14 UTC." The axis set makes this a bisection, not a guess:

1. **Is it one replica or all of them?** Split the TTFT histogram by `replica_id` (bounded cardinality — one label per pod). If one replica, drain it (module 109) and escalate to a cluster-level investigation; chapter 03 names the per-replica dashboard.
2. **Is it the queue, prefill, or first-decode step?** Split TTFT into `queue_time + prefill_time + first_decode_time` from the same engine. If `queue_time` dominated: scheduler is overloaded, autoscaler should have fired — check module 109's scaler signals. If `prefill_time` dominated: GPU-side; see 3. If `first_decode_time` dominated: a CUDA-graph re-trace is suspected (module 102).
3. **Prefill-side: is prefill-TPS down or are prompts longer?** Split by `prompt_length_bucket` (bounded cardinality). If prompts the same length and prefill-TPS dropped: a GPU-side regression — check `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE` for a dispatch regression and `DCGM_FI_DEV_FB_USED` for a KV pressure change. If prompts got longer: a workload shift — not an engine bug.
4. **GPU-side: is the HBM budget blown or the compute budget?** `FB_USED` and `KV_USED` tell you about memory; `TENSOR_ACTIVE` and `DRAM_ACTIVE` tell you about compute. One regression is usually visible in one place.

Every step uses an axis that exists because this chapter named it. The on-call engineer who has only `http_request_duration_seconds` cannot do this bisection; the on-call engineer who has the twelve-axis set reaches the right diagnosis in minutes, not hours.

## Summary

- The inference axis set is twelve numbers wide and the three-metric SRE RED-triangle is not enough.
- Latency is six axes: E2E, TTFT (= queue + prefill + first-decode), TPOT, ITL. Each has a different unit and a different operational meaning; aggregate them separately.
- Scheduler-side axes (cache-hit, preemption, admission-reject, prefill-decode mix, queue depth, in-flight) describe the engine's internal state. Some are SLO candidates (chapter 02); most are operational.
- GPU-side axes (via DCGM) describe the hardware. `SM_ACTIVE` is not utilisation; `FB_USED` is a plateau not a leak; `TENSOR_ACTIVE` and `DRAM_ACTIVE` distinguish prefill-regime from decode-regime.
- NCCL time is a first-class latency component for `TP > 1`. Export it; alert on the ratio to forward-time.
- Match emission shape to the question: counter for cumulative, histogram for distribution, gauge for instantaneous, trace for per-request forensic.
- Cardinality discipline: no `tenant_id` on the hot path, no `request_id` as a label, model-name allow-list.
- Chapter 02 picks the subset that gets an SLO; chapter 03 wires the stack; this chapter is the inventory.
