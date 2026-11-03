# Chapter 01 — The Continuous-Batching Inner Loop

> The scheduler is the heart that beats once per engine step. Before you touch a knob, you need the mental model of that heartbeat — otherwise every tuning move is a guess.

## Motivation

In module 102 you walked the five layers and met the scheduler as *the box that admits requests and picks what runs this iteration*. That was enough to debug outages. It is not enough to tune for an SLO. To tune, you need the inner loop at an operator's resolution — exactly how one iteration decides what runs, why continuous batching is not just "larger batches," and what the admission decision is actually weighing.

This chapter is that resolution. Every subsequent chapter in module 103 — chunked prefill, preemption policies, metrics — assumes you can walk the inner loop in your head and name the step where each knob lands.

The algorithms themselves — iteration-level batching ("continuous batching") from Orca (OSDI 2022), chunked prefill from SARATHI, PagedAttention block allocation from vLLM — are documented in the papers listed in [`resources.md`](resources.md). This chapter is the operator's reading of those algorithms, not a re-derivation.

<!-- needs-research: specific class and function names in vLLM, SGLang, and TRT-LLM scheduler source rotate between releases. Verify names against the exact engine version your fleet runs before quoting them in a runbook. -->

## Request-level batching vs. iteration-level batching

The thing most operators call "continuous batching" is more precisely **iteration-level batching**. The distinction is load-bearing.

### Request-level (static) batching — the thing LLM serving does *not* do

A classical batch inference server collects N requests into one batch, runs the whole batch to completion, and emits all N responses. The batch is a fixed set from start to finish. If one request wants 10 output tokens and another wants 1000, the fast request waits for the slow one. GPU utilisation stays high only while all N are still generating; once requests start to finish, the batch *shrinks* but the compute cost per step does not drop proportionally (padding, kernel-launch overhead), so wall-clock throughput falls off a cliff.

For LLMs with heavy-tailed output-length distributions — most chat, agentic, and reasoning workloads — static batching is catastrophic. You pay the cost of the slowest request on every step for every request.

### Iteration-level batching — what every modern LLM engine actually does

Iteration-level batching makes the **batch membership decision every iteration**, not once per request:

```
every engine step:
    1. collect in-flight sequences (the "running" set)
    2. consult the "waiting" set — any arrivals that can fit?
    3. choose the batch: subset of running + possibly admit some from waiting
    4. for each sequence in the batch:
         - if it's new  → run a prefill token / prefill chunk
         - if it's live → run a decode token
    5. run ONE forward pass over the merged batch
    6. sample output tokens; stream any that are ready
    7. retire sequences that hit stop conditions
    8. if KV pool is full and a new large request wants in → preempt
    9. go to 1
```

Three consequences fall out:

1. **A sequence that finishes leaves the batch immediately.** No "wait for the slowest." The next iteration has one fewer sequence; a waiting request can take the slot the next tick.
2. **A new request can join mid-flight.** It doesn't wait for the current batch to drain. On the next step it is prefilled (possibly in chunks; chapter 03) and starts decoding alongside everyone else.
3. **Prefill and decode share one forward pass.** Within a single `model.forward()` call, some positions are prefill tokens (for a new request) and others are the next decode token of an in-flight request. The attention kernel handles the mixed workload (Flash Attention's "variable-length" / "prefill-decode-mixed" mode; version- and backend-dependent).

The Orca paper (OSDI 2022; [`resources.md`](resources.md)) is the canonical statement of this idea. Every production engine — vLLM, SGLang, TensorRT-LLM — implements some variant.

### Why GPU utilisation stays high

Each iteration dispatches *one* forward pass with a batch sized by the scheduler's choice. The scheduler's job is to keep that batch big enough to amortise kernel-launch overhead but small enough to fit in the KV budget and respect the per-sequence latency SLO. There is no "idle slot" waiting for a slow request — any slack is immediately handed to a waiting request, up to the scheduler's caps.

## The two queues

Every scheduler maintains at least two sets of sequences. The names rotate between engines; the structure is universal.

### The arrival (waiting) queue

Where new requests land after the HTTP ingress validates them and the engine wraps them in a `SequenceGroup` (vLLM nomenclature) / request object (SGLang / TRT-LLM nomenclature). This queue is **not** running on the GPU. It is a Python list / priority structure in the engine process.

Operational properties:

- Depth grows if arrival rate exceeds the admission rate the scheduler can sustain.
- Each request has a wait-time clock from the moment it enters.
- Admission from this queue is the single most important fairness decision the scheduler makes — see "Admission control" below.
- At the metric level this is `vllm:num_requests_waiting`, `vllm:time_in_queue_requests_seconds` or the SGLang / TRT-LLM equivalent.

### The run (running) queue

The sequences currently in flight — the ones the scheduler will choose from on the next iteration. For each sequence the scheduler tracks:

- Its state (prefill-not-started, prefill-in-progress, decoding, finished).
- Its allocated KV blocks (handed out by the block manager; module 104).
- Its progress (how many prompt tokens prefilled, how many output tokens produced).
- Its priority and age (for fairness policies and starvation guards; chapter 04).

A sequence enters this queue when admitted; leaves when it finishes, is preempted (and either swapped out or dropped-to-recompute), or is cancelled.

At the metric level: `vllm:num_requests_running`, `vllm:scheduler_running_queue_size`, or equivalents.

### Why the split matters

The arrival queue is a *latency* object: wait-time there is pure, unamortised TTFT. The run queue is a *throughput* object: time there is being spent on useful forward progress.

The scheduler's job is to move requests out of the arrival queue into the run queue **fast enough to meet the TTFT SLO**, without over-admitting into a run queue that can't support the TPOT SLO. Those two constraints pull in opposite directions. Chapter 02 is the arithmetic of balancing them.

## Prefill vs. decode — two radically different workloads in one engine

A single LLM request moves through two phases, and the engine runs the two differently.

### Prefill (prompt processing)

The engine takes the prompt tokens, runs them through the model, and fills the KV cache with the keys and values at every prompt position. Compute cost scales roughly linearly with prompt length. For a 2k-token prompt, a prefill step does ~2000× the attention+GEMM work of one decode step.

Characteristics:

- **Compute-bound.** With a long enough prompt (above a few hundred tokens; model- and backend-dependent) the arithmetic intensity is high enough that the GPU is doing actual math, not waiting on memory.
- **One shot per request** (unless chunked prefill is on — chapter 03). A traditional engine runs the whole prefill as a single forward pass.
- **Blocks everything else** while running, if run un-chunked. A long prompt's prefill step takes much longer than a decode step, which means every other in-flight request waits — TPOT spikes.

### Decode (generation)

One forward pass produces one output token per sequence. The forward is cheap *per token* but the arithmetic intensity is low: the model reads enormous weights to produce a single new token, and almost all of the time is spent loading weights and KV cache from HBM.

Characteristics:

- **Memory-bandwidth-bound.** Loading the KV cache for every attention head of every layer dominates. Larger batches amortise the weight-load; the KV cache does not amortise (every sequence has its own).
- **Many iterations per request.** A 500-token response is 500 decode steps.
- **Loves big batches.** Decode throughput scales roughly linearly with batch size until KV capacity caps it.

### Why mixing them is hard

If the scheduler is running decode-only (all sequences are already prefilled, no new ones admitted), the step is cheap and short — low TPOT. The moment it adds a long prefill to the batch, that single step gets much longer, which raises TPOT for every decode sequence sharing the step.

This is the central dynamic of the scheduler:

> A big prefill, scheduled naively, is a latency bomb for every decode sequence running at the same time.

Three mitigations exist:

1. **Chunked prefill** — split the long prefill into small chunks interleaved with decode steps so no single step is dominated by prefill. Chapter 03 is this knob.
2. **Prefill-decode disaggregation** — run prefill on one worker, decode on another, transfer KV between them. DistServe / Splitwise / NVIDIA Dynamo. Module 107.
3. **Batching caps** — cap `max-num-batched-tokens` so a single step cannot grow past a target wall-clock. Chapter 02.

All three cost something; all three are the right answer under different workload shapes.

## Admission control — the per-iteration decision

Every iteration, the scheduler decides which subset of waiting requests to admit into the running batch. The decision is constrained by three resources:

### Resource 1 — Sequence slots (`max-num-seqs`)

Hard cap on how many concurrent sequences can be in flight. Larger → more decode throughput (bigger batches amortise weight loads), but every slot carries a per-sequence overhead (block-table metadata, Python object, scheduler bookkeeping).

Typical production values are in the dozens to low hundreds. Choose based on the smallest batch size at which the GPU is "full" (decode is memory-bandwidth-bound; after a point adding sequences doesn't help). Chapter 02 drills this.

### Resource 2 — Token budget per step (`max-num-batched-tokens`)

Hard cap on `(prefill tokens scheduled this step) + (one decode token per running sequence)`. This is the knob that gates the wall-clock duration of each engine step.

Example: `max-num-batched-tokens = 8192`, `max-num-seqs = 128`. In pure-decode mode the step processes up to 128 tokens (one per sequence). The scheduler then has 8192 − 128 = 8064 token budget left over for prefill — it can admit a new request with a 4000-token prompt (whole prefill in one step), or two 3000-token prompts, or (with chunked prefill) a chunk of a long request.

If the budget is small (e.g. 2048), each step is fast — low TPOT variance — but a long prompt can take many iterations to prefill, raising TTFT for that request. If the budget is large (e.g. 32768), prefills finish faster — low TTFT on long prompts — but decode TPOT jitter is worse because any iteration that includes a chunky prefill is a long iteration.

### Resource 3 — KV cache (block manager budget)

Each sequence occupies KV blocks proportional to its current length. The block manager has a finite pool. Admission is bounded by "does this request's worst-case KV need (prompt + expected output) fit in what the pool has free?"

The scheduler's conservative-vs-aggressive stance here is a tuning dial:

- Conservative: refuse to admit if worst-case output can't fit. No preemption risk; low throughput at high load.
- Aggressive: admit optimistically; preempt someone else if the pool runs dry. Higher throughput; preemption risk.

The default varies by engine and release. Verify against your engine's docs; chapter 04 covers the preemption side.

### Priority and fairness

Beyond the hard caps the scheduler can order the waiting queue by:

- **FIFO** — simplest; prone to head-of-line blocking by a long prompt.
- **Priority** — explicit per-request / per-tenant priority. Needs a trust boundary (is the client authoritative?).
- **Shortest-prompt-first** — reduces mean TTFT at the cost of starvation for long-prompt requests.
- **Fair-share** — ensures no single tenant monopolises; typically a token-rate quota (module 106 goes deep).
- **Age-boosted priority** — a starvation guard that pushes an aging request forward.

Flag names and the set of supported policies rotate by engine. vLLM has typically defaulted to FIFO with starvation guards. SGLang's scheduler supports priority and has a lower-overhead loop. TRT-LLM / Triton exposes similar in-flight-batching knobs through the TRT-LLM backend config. Verify against the version your fleet runs.

### Scheduler delay

Some engines expose a scheduler delay — the loop waits a tiny budget (sub-millisecond) before committing a batch, hoping another request lands and the batch grows. Useful under bursty traffic; a liability on tail-sensitive workloads where the delay is pure added TTFT. Treat as a workload-specific knob; measure before enabling.

## One iteration, step by step

Trace one iteration of the inner loop on a realistic snapshot:

**Setup.** `max-num-seqs=64`, `max-num-batched-tokens=8192`, chunked prefill on with chunk size 1024. Current state:

- Run queue: 40 sequences, all mid-decode.
- Arrival queue: 3 new requests. Request A has a 500-token prompt; request B has a 6000-token prompt; request C has a 50-token prompt.
- KV pool: 70% used, with enough free blocks for ~1500 tokens.

**Step 1 — Scan running set.** The scheduler walks the 40 running sequences. Each gets one decode token's worth of budget — 40 tokens total committed. All 40 need one more KV slot; the block manager grants it.

**Step 2 — Admission scan.** Remaining token budget: 8192 − 40 = 8152. Remaining sequence budget: 64 − 40 = 24. Remaining KV budget: ~1500 tokens.

- Request A (500-token prompt): fits in 500 tokens of batch budget and 500 tokens of KV. Admit; prefill scheduled this step as 500 tokens.
- Request B (6000-token prompt): does not fit in one chunk; chunked-prefill mode picks the first 1024 tokens. 1024 tokens of batch budget needed; 1024 tokens of KV. The KV pool only has 1500 free. If we admit, we have 1500 − 500 (A) − 1024 (B first chunk) = -24. Over budget. The scheduler may defer B this iteration, OR trigger preemption of a lower-priority running sequence to free KV. Depends on policy.
- Request C (50-token prompt): tiny; if B is deferred, C fits trivially.

**Step 3 — Preemption decision (if B is admitted).** With `swap-out` preemption: pick the lowest-priority or longest-living running sequence, swap its KV to CPU, free its blocks. With `recompute` preemption: pick, free its blocks, note it must re-prefill when re-admitted.

**Step 4 — Build batch.** The runner receives a batch spec: 40 sequences × 1 decode token each + (either {A prefill of 500}, or {A prefill 500 + B chunk 1024}, or {A prefill 500 + C prefill 50} + preemption actions). Different scheduler policies pick different combinations.

**Step 5 — Forward pass.** One `model.forward()` call. The attention kernel handles the mixed prefill/decode positions. ~1 iteration of wall-clock, bounded by the longest of (compute for prefill tokens, memory loads for decode tokens + weights).

**Step 6 — Post-step.** Sampled tokens stream back through `AsyncLLMEngine`. Any sequences that hit EOS / stop condition retire; their KV frees. Loop.

This exact decision — three resources, N candidates, pick the subset — is what every scheduler tuning knob is modulating.

## Where to find the loop in source

For each engine, the loop lives in one place and the admission function in another. The exact names rotate; what to look for:

### vLLM

- Engine step loop — in the engine class (`AsyncLLMEngine` / core engine) under `vllm/engine/`. Look for a method named along the lines of `step()`, `_run_engine()`, or `engine_core_step()`.
- Scheduler module — `vllm/core/scheduler.py` (or the newer refactor; verify in your release). Look for a `schedule()` method; it returns a `SchedulerOutput`-shaped object containing which sequences run, which are prefilling, which decode, and which are preempted.
- Admission and preemption helpers — same file; grep for `_schedule_prefills`, `_schedule_decodes`, `_preempt`, `_swap_out`.

### SGLang

- Scheduler module — `python/sglang/srt/managers/scheduler.py` (verify path for your release). SGLang advertises a "zero-overhead batch scheduler"; the loop is intentionally tight.
- RadixAttention is tightly coupled to the scheduler; cache-aware batching lives alongside.

### TensorRT-LLM (via Triton `tensorrtllm_backend`)

- The scheduler is inside the TRT-LLM batch manager (C++). In the Python backend config (`config.pbtxt` / backend parameters) the operator-facing knobs include `max_num_sequences`, `max_tokens_in_paged_kv_cache`, `enable_chunked_context`, `scheduler_policy`.

Walk your engine's source once with the five layers from module 102 as the map and this chapter's inner-loop as the resolution. If you cannot point at `schedule()` in under a minute, exercise 01 will fix that.

## Common misreadings

- **"Continuous batching means always use the biggest batch."** No — continuous batching means the batch *membership* is reconsidered every iteration. The batch *size* is a separate decision bounded by three resources.
- **"Prefill is faster than decode so just run it inline."** No — prefill is faster *per token processed* but a long prefill is still many times more work than one decode step. Dropping a long prefill inline spikes TPOT for everyone sharing the iteration.
- **"If I raise `max-num-seqs` I get more throughput."** Only up to the point where decode is still memory-bandwidth-bound. Past that, you buy more KV footprint and more preemption risk, not more throughput.
- **"If I raise `max-num-batched-tokens` I get more throughput."** Up to a point. Past that, each step gets so long that TPOT variance ruins the SLO; and if KV can't keep up, preemption rate climbs.
- **"The scheduler is doing something clever with scheduling theory."** Mostly it's running a greedy policy under three resource caps. Clever is rare; configured is universal.

## Summary

- Modern LLM engines batch **at the iteration level** — the batch membership is chosen each engine step from (running sequences) ∪ (admitted-this-step requests).
- The scheduler maintains an **arrival queue** (not yet running) and a **run queue** (in flight). Latency is paid in the arrival queue; throughput happens in the run queue.
- **Prefill is compute-bound, decode is memory-bandwidth-bound.** Mixing them in one iteration is the central scheduler challenge.
- **Admission control** is bounded by three resources every iteration: sequence slots, token budget per step, KV cache capacity.
- The three resource caps — `max-num-seqs`, `max-num-batched-tokens`, KV-pool budget — are the knobs chapter 02 tunes against a stated SLO.
- Mitigations for the "long prefill blocks decode" problem are **chunked prefill** (chapter 03), **preemption policy** (chapter 04), and (across engines) **prefill-decode disaggregation** (module 107).
- If you cannot point at `schedule()` in your engine's source and name the admission function, you are not ready to tune its flags. Exercise 01 is the fix.

Chapter 02 turns this loop into a configuration — translating a workload spec and an SLO target into values for `max-num-seqs`, `max-num-batched-tokens`, scheduler-delay, and the priority policy, defended against a throughput-vs-latency Pareto curve.
