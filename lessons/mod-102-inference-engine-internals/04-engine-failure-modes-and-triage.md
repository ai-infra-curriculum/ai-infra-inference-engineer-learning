# Chapter 04 — Engine Failure Modes and Triage

> Seven failure modes recur across vLLM, SGLang, and TensorRT-LLM. If you can recognise, reproduce, and triage each one, you are already most of the way to an on-call shift that doesn't rely on luck.

## Motivation

Chapters 01–03 walked each engine's internals. This chapter walks the *symptom catalogue*. On an on-call shift the alert is almost never "a specific scheduler bug fired in a specific line"; it is "P99 TTFT is doubled," "requests are stuck," "a worker crashed," "cost is up 30% at the same traffic." The job is to go from symptom to localised layer-of-the-engine to a specific knob or fix — fast.

The seven failure modes below are the ones you will meet repeatedly. For each one we give:

- **The shape of the symptom** at the metric level — what a dashboard shows.
- **The cause** at the engine-internals level.
- **The triage move** that confirms the diagnosis.
- **The fix** — the knob, patch, or rollback.
- **The prevention** — what to add to the pre-release checklist so this one stops being a surprise.

All seven assume you have the vocabulary and knob-set from chapters 01–03 at hand. The symptoms are engine-agnostic; the knob names map to vLLM where quoted, with the SGLang / TRT-LLM equivalents in the chapters above. <!-- needs-research: specific metric names, flag spellings, and release defaults rotate between engine versions. Treat the examples in this chapter as the *shape* of the triage; verify current metric names and flag spellings against the engine release your fleet runs. -->

## 1. OOM on long-context

### Symptom

Requests with long prompts (close to `--max-model-len`) fail with a CUDA out-of-memory error, or the serving process is killed by the OOM-killer. May be silent at first — a few requests fail and are retried by the client; then a traffic burst takes the worker down.

### Cause

KV cache memory per sequence scales linearly with sequence length. The compiled engine (TRT-LLM) or the running engine (vLLM / SGLang) reserved its KV pool against a *rated* maximum concurrency; a cohort of long-context requests exceeds the pool and the scheduler either refuses to admit them or runs out mid-decode.

Three distinct underlying causes to distinguish:

1. **Admission misconfigured** — `--max-model-len` is set so high that a worst-case fleet of max-length requests cannot physically fit in the KV pool.
2. **Weight / activation workspace eating the pool** — the engine's weight + activation budget was underestimated at startup, leaving the KV pool smaller than planned.
3. **No per-request length cap** at the API gateway — client-sent prompts exceed the engine's envelope at traffic peaks.

### Triage move

- Check `vllm:gpu_cache_usage_perc` (or SGLang / TRT-LLM equivalent) at the time of the incident. Was it climbing toward 100% before the failure?
- Check the distribution of input length across the request stream. Did it shift (new prompt template, new agent trace, new tool preamble)?
- Check the launch command: what is `--max-model-len`? What is `--max-num-seqs`? Does `max-model-len × max-num-seqs` fit in the pool at the configured block size and dtype?
- If TRT-LLM: what is `--max_input_len` / `--max_seq_len` compiled into the engine? Is the backend's `max_tokens_in_paged_kv_cache` consistent?

### Fix

- Short term: lower `--max-model-len` or `--max-num-seqs` to a value where the worst case fits. Add a per-request length cap at the API gateway.
- Medium term: raise `--gpu-memory-utilization` (vLLM) / `mem-fraction-static` (SGLang) / backend KV-pool parameters (TRT-LLM) if there is headroom. Reduce `--block-size` (vLLM) if fragmentation is also present (see failure mode 3).
- Structural: split the workload. Serve short-prompt and long-prompt traffic on different engines or different pools, each tuned for its own envelope.

### Prevention

- Pre-release: run a traffic-shape probe at the longest realistic prompt length × the target concurrency. Document the KV footprint.
- Observability: alert on `gpu_cache_usage_perc` crossing a configurable threshold (not 100% — leave room to breathe).
- Capacity planning: when the model gets a bigger `max-model-len` upstream, the fleet's KV budget is **not** magically bigger. Reconcile explicitly.

## 2. Prefix-cache cliff

### Symptom

TTFT doubles overnight. Cache-hit rate drops from, say, 60% to near zero. No new deploy; no obvious upstream change. Possibly correlated with a change to a client's prompt template, a new agent framework rollout, or a RAG retriever upgrade on the application side.

### Cause

The prefix cache (vLLM's block-hashed PagedAttention cache or SGLang's RadixAttention tree) matches *token sequences*, not conceptual-prefix identity. A change in the first N tokens of the prompt — even one that is semantically irrelevant, like whitespace, a timestamp, or a reordered field — invalidates the shared prefix. The engine then prefills the full prompt for every request instead of just the suffix.

### Triage move

- Compare cache-hit rate before and after the regression (`vllm:prefix_cache_hits_total / vllm:prefix_cache_queries_total`, or SGLang equivalent).
- Dump a sample of recent prompts; diff the first few hundred tokens against last week's samples. Look for:
  - A new leading timestamp, request id, or UUID.
  - A changed order of system-prompt fields.
  - A new tool-schema preamble.
  - An agent framework that now prepends a changing per-request header.
- If TRT-LLM: confirm `enable_kv_cache_reuse` is on and that the backend config hasn't drifted on upgrade.

### Fix

- Short term: coordinate with the consumer that changed the prompt. Move the per-request varying content to the *end* of the prompt (after the stable prefix), not the beginning.
- Medium term: for workloads with structured prompts, define the prompt-prefix as a protocol contract: fields in a fixed order, non-varying content first. Pin this in the API contract.
- Structural: for truly dynamic prefixes (per-user memory, per-request personalisation), accept that the cache won't help and plan capacity against the cold-prefill cost.

### Prevention

- Observability: alert on a cache-hit-rate drop > N% over a short window.
- Change-management: prompt-template changes on the application side should be shared with the inference team before rollout. The platform charter (chapter 05 of mod-101) is where this contract lives.
- Testing: a canary that re-plays a sampled prefix pattern against staging should catch cliffs before production.

## 3. Allocator fragmentation

### Symptom

Preemption rate climbs even though `gpu_cache_usage_perc` is well below 100%. New requests are admitted and then preempted quickly; some are evicted multiple times. TTFT variance grows; the long tail widens.

### Cause

The KV pool has free blocks but not enough *in the right configuration*. For PagedAttention this usually means: the free list is fragmented across sequences that have released some blocks but still hold others, and a new sequence that needs a contiguous allocation (or a long run of blocks for its block table) can't be satisfied without evicting someone.

RadixAttention has a different flavour of fragmentation: the tree has many small nodes from past prefix branching; evictable nodes are small; servicing a large new suffix requires freeing many small nodes.

### Triage move

- Watch pool usage vs. preemption rate over the same window. Preemptions with pool < 85–90% usage is the signature.
- vLLM: check `--block-size`. Smaller blocks give finer sharing but worsen fragmentation for large allocations.
- SGLang: check node count in the radix tree over time. A tree node count that grows roughly linearly with request count signals heavy branching.
- Confirm the workload shape — is prefix locality low (many unique prefixes, little reuse)?

### Fix

- Short term: lower `--max-num-seqs` so fewer concurrent sequences compete for blocks.
- Medium term: raise `--block-size` (vLLM) if the workload prefers coarse allocation to fine prefix sharing. There is no universally better value; this is a per-workload tune.
- SGLang: if prefix churn is high and fragmentation outweighs reuse benefit, disable RadixAttention and run with the simpler allocator (verify current flag).
- Structural: segregate workloads by shape. Mixing one workload with high prefix locality and one with high churn on the same engine often fragments more than either alone would.

### Prevention

- Pre-release: workload characterisation. Know the prefix-churn rate before you turn RadixAttention / prefix-caching on with its default knobs.
- Observability: alert on preemption rate > N and pool usage < M at the same time (the fragmentation signature).

## 4. CUDA graph recompile storm

### Symptom

Step time has a bimodal distribution. Most steps run in the usual 20–30ms; a long tail of steps run 100ms+. The slow steps correlate with new request arrivals or batch-shape changes. Overall throughput is 20–40% below benchmark.

### Cause

CUDA graphs are captured per batch-shape bucket. When the engine encounters a batch shape it has not captured a graph for, it either:

- Falls back to eager execution for that step (slow but one-time).
- Captures a new graph on the critical path (slow and allocates memory).

If the workload presents many different batch shapes (unusual sequence-length bucketing, high variability in in-flight sequence count, speculative decoding with varying acceptance length), captures recur. Each capture is tens to hundreds of ms and allocates memory; the memory pressure can also cause the engine to evict existing captured graphs, which forces re-capture later.

### Triage move

- Pull step-time distribution from logs or metrics. Look for a bimodal split.
- Check whether `--enforce-eager` is off (graphs on) and whether the engine logs capture events.
- Correlate slow steps with arrival events or sequence-count transitions.
- If using speculative decoding: are acceptance lengths varying widely? Different accept counts produce different downstream batch shapes.

### Fix

- Short term: `--enforce-eager` turns graphs off and the problem goes away at the cost of baseline throughput. Acceptable as a stop-gap.
- Medium term: reduce the number of distinct batch shapes the engine sees. Pre-size shape buckets if the engine supports it (vLLM has a captured-batch-sizes list; verify current flag). Fix client-side batching that unnecessarily varies shape.
- Speculative decoding: cap the acceptance window to reduce downstream shape variance; module 108 drills this.

### Prevention

- Pre-release: run the benchmarking workload long enough to stabilise graph captures. The first N steps are expected to be slow (warmup); confirm the steady state is clean.
- Observability: alert on step-time P99 / P50 ratio climbing — a widening tail signals shape instability.

## 5. `torch.compile` re-trace

### Symptom

Similar to failure mode 4 but at a longer timescale. The engine runs well for a while, then enters a window of order-of-magnitude slower steps while something — not CUDA-graph capture — is in flight. New model code paths or new input types correlate. Memory usage climbs during the slow window.

### Cause

`torch.compile` traces model code paths on first-use and caches compiled kernels keyed on (shape, dtype, option) tuples. New shape or option combinations trigger a new trace. Traces are more expensive than CUDA-graph captures (compilation, not just launch-list recording), and the cache can be evicted under memory pressure, forcing re-trace.

Compile integrations in vLLM and SGLang are evolving. The problem profile is specific to the compile-level configuration (`-O0 … -O3` or equivalent flag) and the dynamic-shape behaviour of the compile backend.

### Triage move

- Check the engine logs for compile-trace events during the slow window.
- Check the compile-cache size and eviction behaviour (release-dependent).
- Correlate slow steps with first-of-kind requests — new prompt-length bucket, new output-length bucket, new sampling-param combination, new LoRA adapter activation.
- Test: launch with compile disabled and the problem should vanish.

### Fix

- Short term: lower the compile optimisation level, or disable compile entirely for the workload. The throughput loss is real but bounded.
- Medium term: identify the specific inputs that cause re-tracing and either pre-warm them (send a representative batch at startup) or stabilise the shape so the cache hits.
- Structural: pin to a release known to be stable for your model + compile-level combo. Compile-integration regressions are common at the current pace of engine development.

### Prevention

- Pre-release: a long-running soak test (hours, not minutes) at realistic traffic shape should surface re-trace behaviour.
- Observability: a compile-cache miss counter (if the engine exports one) is the direct signal; otherwise step-time tail is the indirect signal.

## 6. NCCL hang on worker crash

### Symptom

Tensor-parallel serving hangs. All ranks appear live from the process-monitor perspective; GPU utilisation is near zero; no new requests complete. Logs show one worker crashed earlier — a CUDA assertion, a Python exception, an OOM kill — but the other workers are blocked on an NCCL collective waiting for the dead rank to join.

### Cause

NCCL collectives (AllReduce, AllGather, ReduceScatter) require every rank to participate. If one rank dies mid-collective, the surviving ranks block indefinitely because, by default, NCCL has no deadline. The engine's own health check sees "process is alive" and doesn't know a collective is deadlocked.

### Triage move

- Confirm one worker died earlier — check logs across all ranks, not just rank 0.
- Check for the dead worker in `nvidia-smi` or Kubernetes pod state. Did an OOM-killer fire? Did a CUDA assertion kill it?
- Look for NCCL warnings about timeouts or watchdog triggers in the surviving ranks' logs.
- If using Ray / mpirun, is the launcher aware of the crash?

### Fix

- Short term: restart the entire serving unit (all ranks for the TP group). Partial restart is near-impossible without rebuilding the NCCL communicator.
- Medium term: enable NCCL watchdog / timeouts and configure the engine to exit when a watchdog fires. The torch distributed APIs expose timeout for collectives; some engines expose it as a flag — verify your version.
- Observability: on worker crash, trigger an immediate restart of the TP group, not a wait-and-see.

### Prevention

- Chaos testing: kill one worker during load and verify the fleet recovers within a bounded time. If it hangs, your health-check layer is misreading "alive but deadlocked" as healthy.
- Resource sizing: the first failure mode on this list (OOM) is a frequent *cause* of worker death that cascades into NCCL hang. Prevent OOM upstream and you prevent many hangs.

## 7. Slow-rank straggler in TP

### Symptom

Throughput at TP=N is substantially lower than N× the single-GPU baseline. All GPUs run; utilisation is uneven; one GPU shows consistently lower activity than the others. P99 TTFT is dominated by the slowest rank per step.

### Cause

Tensor-parallel execution is bulk-synchronous. Every step must wait for the slowest rank. Causes of a persistently-slow rank:

- **Hardware variance** — a GPU with slower memory (thermally throttled, degraded, or just a bad die), a different driver on one host, PCIe link width degraded on one slot.
- **Interconnect imbalance** — NVLink topology that's not fully-connected (unusual on HGX; common on custom builds).
- **Rank-pinning drift** — the engine binding worker N to physical GPU X, but X is also running something else (another workload, nvidia-smi queries, a telemetry agent).
- **Numa / CPU affinity** — the host CPU side of one rank is NUMA-mismatched to its GPU; CPU-side bookkeeping slows that rank's forward step.

### Triage move

- Watch per-rank step time (if the engine exports it) or per-rank GPU utilisation during a load test. Find the slow rank.
- Run `nvidia-smi` on the slow rank and look for thermal throttling, ECC errors, lower memory clock.
- Check PCIe link width on each GPU (`nvidia-smi -q | grep "PCIe Generation"` or similar). Is one at a lower generation / fewer lanes?
- Check NUMA pinning of worker processes.
- Compare the slow rank's host-level DCGM metrics against the fast ranks.

### Fix

- Short term: isolate the slow GPU from the TP group — drain the serving pod and reschedule on nodes with uniform GPUs. Mark the slow GPU for maintenance.
- Medium term: enforce NUMA / CPU pinning for serving workers.
- Hardware: coordinate with the data-centre / cloud operator to replace the degraded GPU or fix the slot.

### Prevention

- Pre-production: run per-rank micro-benchmarks on each new node and reject nodes whose per-rank throughput variance exceeds a threshold.
- Observability: a per-rank step-time metric is non-negotiable at scale.
- Fleet-level: track per-GPU DCGM metrics; flag outliers automatically.

## Reading the triage order

The seven modes are not independent. A frequent cascade:

- **A prompt-template change (failure mode 2) raises TTFT by causing cold prefills.** This causes KV churn, which raises **fragmentation (failure mode 3)**, which raises **preemption rate**, which raises step-time variance. If the engine was close to its KV ceiling, the long prompts now cause an **OOM (failure mode 1)**. If the OOM kills a worker in a TP group, the surviving ranks **hang on NCCL (failure mode 6)**.

One upstream prompt-template change can present as five different symptoms over an hour. Triage top-down, correlate across metrics, and start from the earliest signal in the metric timeline — not the loudest alert.

## A one-page triage runbook, condensed

| Symptom | First-check metric | First hypothesis | First knob |
|---|---|---|---|
| CUDA OOM / worker killed | `gpu_cache_usage_perc`, prompt-length distribution | Admission misconfig | lower `--max-model-len` or `--max-num-seqs` |
| TTFT doubled, no deploy | cache-hit rate before/after | Prompt-template change | diff recent prompt prefixes against last week |
| Preemption rate spike, pool < 90% | preemption rate vs. pool usage | Fragmentation | lower `--max-num-seqs`, tune `--block-size` |
| Bimodal step time | step-time P99/P50 ratio | CUDA-graph recompile | `--enforce-eager` as stop-gap |
| Slow window at long timescale | compile-cache miss counter | `torch.compile` re-trace | lower compile level, pre-warm shapes |
| TP serving hung, no completions | per-rank liveness vs. collectives | NCCL hang post-crash | restart TP group; verify NCCL timeout config |
| Throughput below N× baseline on TP=N | per-rank step time | Slow-rank straggler | drain slow GPU, fix NUMA pinning |

Pin this table to the on-call runbook. Expand each row into a per-engine playbook in the exercises for this module.

## Boundary notes

- **Kernel-level crashes** — a Flash Attention assertion, a FP8 GEMM epilogue bug — surface here as symptoms, but the fix lives at the performance peer. Your job is to pin a working engine version and open the issue; the peer ships the patch.
- **Driver / firmware** issues (ECC storms, driver mismatches between nodes, kernel-module ABI drift) sit at the `ai-infra-engineer` boundary (L20) with GPU-specific amendments covered in module 109. You recognise the signature; the base-platform team patches it.
- **Autoscaling and K8s-level recovery** (restart budgets, pod-disruption budgets, warm-pool management) are module 109 territory. This chapter triages the engine process; module 109 keeps the fleet of processes healthy.

## Summary

- Seven failure modes recur: OOM, prefix-cache cliff, allocator fragmentation, CUDA-graph recompile storm, `torch.compile` re-trace, NCCL hang, slow-rank straggler.
- Each has a dashboard-level signature and a specific first-knob fix. If you can recognise the signature inside five minutes, you have beaten the alarm.
- Failure modes cascade. One upstream change (prompt template, config drift, new model) can manifest as multiple symptoms over an hour. Triage from the earliest signal in the timeline, not the loudest alert.
- Prevention beats triage. Workload characterisation, pre-release soak tests, cache-hit-rate monitoring, per-rank step-time metrics, NCCL watchdogs — all cheap compared to the on-call cost of discovering a failure mode live.
- Kernel-level root causes defer to the performance peer; driver/firmware root causes defer to the base-platform engineer; your job is to localise and route accurately.

Chapter 05 steps outside the three serious engines and looks at the TGI deprecation story — because the field still runs TGI in a lot of places, and your on-call rotation will still meet it.
