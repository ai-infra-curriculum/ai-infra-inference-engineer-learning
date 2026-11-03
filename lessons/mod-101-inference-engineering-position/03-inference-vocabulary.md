# Chapter 03 — The Inference Vocabulary

> Every term on this page shows up in modules 102–114 as a metric, a knob, or an SLO. This chapter is the anchor.

## Motivation

Inference-platform conversations fail cheaply when the vocabulary is loose. "The P99 latency is bad" is a sentence that could mean *time to first token*, *inter-token latency*, *end-to-end response time*, or *tail of the queue delay before scheduling* — four very different problems with four different owners. Fix the vocabulary first; the rest of the curriculum assumes it.

Each term in this chapter is defined once, linked to the metric unit it measures, and tied to the engine primitive that implements or exposes it. Where the glossary disagrees with your current fleet's dashboard labels, change the dashboard, not the glossary.

## Latency metrics

### TTFT — Time To First Token

The wall-clock time from request admission at the API gateway to the first streamed output token reaching the client. Includes queue delay, prefill compute, and the first-token network hop.

- Unit: milliseconds, usually reported as P50 / P90 / P99 over a sliding window
- Dominated by: prefill cost (prompt length × per-token prefill compute) and queue depth
- Engine primitive: measured by the serving front-end at request-handling boundaries; influenced by scheduler admission policy, chunked-prefill chunk size, prefix-caching hits
- SLO shape: typically a tight upper bound ("P99 TTFT ≤ X ms") because users perceive it as the "is it alive?" latency

### TPOT — Time Per Output Token

The average wall-clock time to generate one output token *after the first one*. In some engines this is called ITPL (inter-token-per-latency) or similar.

- Unit: milliseconds / token
- Dominated by: decode-phase compute per step, batch size, memory bandwidth, speculative-decoding acceptance rate
- Engine primitive: a per-token iteration of the continuous-batching inner loop; measured by averaging decode-step durations across a request's output tokens
- Relationship to TPS: 1 / TPOT ≈ tokens-per-second-per-request

### ITL — Inter-Token Latency

The *per-token* latency between consecutive streamed tokens as the user perceives them. In clean conditions ITL ≈ TPOT, but under preemption, batching stalls, or network buffering, ITL spikes above TPOT.

- Unit: milliseconds / token (distribution, not just mean)
- Dominated by: preemption events, batching glitches, GC-like scheduler stalls, SSE flushing behaviour
- Engine primitive: the output-stream pipe between the engine and the HTTP streaming layer; the ITL distribution is often a better signal of "janky streaming" than TPOT

### E2E — End-to-End latency

Total wall-clock time from request submission to the final output token (or final response, if non-streaming). For streaming workloads, E2E = TTFT + (output-token-count × TPOT), approximately.

- Unit: milliseconds
- Engine primitive: measured by the serving front-end
- When it matters: non-streaming consumers (batch pipelines, tool-call steps inside an agent loop) that care about *total* latency, not first-token perceived latency

## Throughput metrics

### Throughput

Total output tokens the fleet produces per second across all requests. The classical throughput metric for benchmarking.

- Unit: tokens / second (per-server, per-GPU, or fleet-wide)
- Dominated by: batch size, model architecture (dense vs. MoE), parallelism topology, kernel efficiency
- Trap: throughput and tail latency trade off against each other via batch size. "We hit 50 k tok/s" with no latency context is a marketing number, not an operational one.

### Goodput

Throughput *that satisfies the SLO*. If an engine serves 100 tok/s but half the requests violate the TTFT SLO, the goodput is 50 tok/s. Introduced in the DistServe paper (see `resources.md`) as a Pareto-honest alternative to raw throughput.

- Unit: tokens / second meeting SLO
- Why it matters: it is the only throughput number that corresponds to money the business can keep. Modules 109–110 build on goodput as the central operational metric.

### Cache-hit rate

Fraction of prompt tokens served by prefix-cache hits rather than re-prefilled. Measured per request or aggregated across a window.

- Unit: percent of prompt tokens
- Engine primitive: PagedAttention block-level prefix match (vLLM) or RadixAttention tree lookup (SGLang)
- Why it matters: a high cache-hit rate collapses TTFT and lowers prefill cost; a cliff in cache-hit rate (e.g., when prompt format changes) is a leading indicator of upcoming TTFT regression

### Preemption rate

Fraction of in-flight requests the scheduler preempted (swap-to-CPU or recompute) during a window, usually because the KV allocator ran out of blocks.

- Unit: preemptions / second, or % of scheduled iterations with ≥1 preemption event
- Engine primitive: the scheduler's eviction policy on top of PagedAttention
- Why it matters: a non-zero preemption rate indicates the fleet is memory-pressured; a growing preemption rate predicts imminent TTFT regression and goodput drop

## Scheduling and batching primitives

### Continuous batching (iteration-level batching)

Scheduling model where new requests can join the running batch *between* decode iterations rather than waiting for the batch to drain. Different from the older **static batching** (fixed batch, wait-to-start) and **dynamic batching** (variable batch, but still waits for a window). Introduced in Orca; adopted by every modern LLM engine.

- Engine primitive: the scheduler's per-iteration admission loop (vLLM's `Scheduler`, SGLang's batch scheduler, TRT-LLM's "in-flight batching")

### Chunked prefill

Scheduling technique (SARATHI-shape) that splits long prefill computations into chunks and interleaves them with ongoing decode steps, keeping decode TPOT stable while prefill still progresses.

- Knob: chunk size (tokens per prefill chunk)
- Engine primitive: `--enable-chunked-prefill` + `--max-num-batched-tokens` and `--max-prefill-tokens` (vLLM); SARATHI-serve for the original research reference
- Trade-off: smaller chunks → lower TPOT variance but longer TTFT; larger chunks → faster TTFT but more decode stalls

### Prefix caching

Reuse of KV-cache blocks for a prompt prefix that has been computed before (same system prompt, same tool-call preamble, same RAG retrieval). Prefix caching turns a worst-case prefill into a cache lookup when the prefix matches.

- Engine primitive: PagedAttention block hashing + block-table lookup (vLLM); RadixAttention tree node sharing (SGLang)
- Scope: usually per-engine; cross-worker prefix sharing is the harder tiered-KV problem (module 104)

### PagedAttention

KV-cache allocator design (vLLM) that chunks the KV cache into fixed-size blocks addressed via a per-request block table — the virtual-memory-for-KV-cache analogy. Enables fine-grained allocation, copy-on-write for shared prefixes, and swap-to-CPU preemption.

- Engine primitive: vLLM's block allocator; also adopted conceptually by TRT-LLM's paged KV cache and others
- Knob: block size (tokens / block)

### RadixAttention

KV-cache organisation (SGLang) that indexes shared prefixes with a radix tree so that any pair of requests sharing the same token prefix share the same KV allocation node, with reference counting and LRU eviction.

- Engine primitive: SGLang's radix tree + cache-aware worker routing
- Strength: workloads with heavy, structured prefix reuse (long system prompt, chat sessions, tool-call preambles)

### Disaggregated prefill–decode (P/D-disaggregation)

Topology that splits request handling into **prefill workers** (compute-heavy, batch-parallel) and **decode workers** (memory-bandwidth-heavy, latency-sensitive), connected by KV transport. Different workers optimise for different hardware utilisation profiles.

- Engine primitive: DistServe (research), Splitwise (research), NVIDIA Dynamo (production), vLLM V1 P/D-disaggregation (production)
- Why: a single node can't hit good TTFT *and* good TPOT at the same time without one of the two being bottlenecked. Splitting them lets each half run near its own hardware ceiling.

### Speculative decoding

Decoding scheme where a cheap "draft" model (or a lightweight auxiliary head) proposes a short token continuation, and the main "target" model verifies the continuation in a single step. Accepted tokens are kept; rejected tokens are discarded. If acceptance rate is high and draft cost is low, decoding throughput grows.

- Flavours: draft-target (two distinct models), Medusa (extra prediction heads on the target), EAGLE (feature-level drafting), Lookahead (self-speculative, no draft model), n-gram
- Engine primitive: a verification path in the decode loop that accepts/rejects speculative tokens; metric is the **acceptance rate**
- When it hurts: low acceptance rate × non-trivial draft cost = latency *regression*. Monitor acceptance rate per workload (module 108).

## Parallelism primitives

### TP — Tensor Parallelism

Partitions the weights of individual tensors (particularly the big MLP and attention projections) across GPUs within a tightly-coupled group (typically a single node over NVLink / NVSwitch). All ranks participate in every forward step; collective communication (AllReduce / AllGather / ReduceScatter) happens inside each layer.

- Engine primitive: `--tensor-parallel-size` (vLLM / SGLang / TRT-LLM)
- Where it fits: scaling a model that is too big to fit on one GPU across the GPUs of a single node

### PP — Pipeline Parallelism

Partitions whole *layers* across stages; each stage holds a contiguous slice of layers. Requests flow through stages like an assembly line. More amenable to looser interconnect (inter-node) than TP, at the cost of pipeline bubbles for small micro-batches.

- Engine primitive: `--pipeline-parallel-size`
- Where it fits: scaling across nodes when TP is bounded by intra-node interconnect

### EP — Expert Parallelism

For MoE (Mixture-of-Experts) models, partitions the *experts* across GPUs. Routing dispatches each token to its chosen experts, which may live on different GPUs. Introduces an AllToAll dispatch/combine pattern that is interconnect-sensitive.

- Engine primitive: `--expert-parallel-size` (vLLM / SGLang); engine-specific routing policy
- Where it fits: serving MoE models (DeepSeek-V3, Mixtral, Qwen-MoE) at scale

### DP attention

A pattern (common on MoE) where the attention block runs data-parallel across ranks while the MoE block runs expert-parallel. Different ranks hold different request subsets' attention state; the two parallelism axes compose. Not every engine exposes this cleanly; SGLang's DP-attention is a reference implementation.

### LoRA swap / multi-LoRA serving

Serving a shared base model with many LoRA adapters simultaneously. Each adapter is small; a single base-model deployment can time-slice or batch-across adapters. S-LoRA and Punica are the reference systems. Covered deeply in module 106.

- Engine primitive: `--enable-lora` plus a per-request `lora_name` field; the engine fuses adapter weights into the forward pass per request

### Model sharding

Umbrella term for *any* splitting of model state across devices — TP, PP, EP, FSDP-shape fully-sharded, ZeRO-style sharded optimizer state at training time. On the inference side, model sharding is near-synonymous with "TP + PP (+ EP if MoE)".

## API-surface primitives

### Streaming

Returning output tokens as they are produced rather than waiting for the full response. Over HTTP, almost always implemented as **Server-Sent Events (SSE)**; occasionally WebSocket. The streaming contract is what makes TTFT and ITL meaningful at the client.

- Engine primitive: HTTP SSE emission tied to the per-token output of the decode loop; backpressure and client-disconnect handling at the HTTP layer (module 111)

### Structured output

Constraint on the generated tokens so that the output is a parseable instance of some schema (JSON Schema, regex, grammar, Pydantic model). Enforced at the token-sampling layer so the model cannot emit tokens that would violate the schema.

- Engine primitive: Outlines / XGrammar backends (vLLM / SGLang), guided decoding modes
- Why it matters: structured output is the serving-side primitive that makes agent tool calls, JSON-mode responses, and function-calling reliable without client-side retries

## Hardware adjacency terms

These show up in chapters 02 and 04; they're defined here so the vocabulary is complete.

- **NVLink / NVSwitch** — NVIDIA's high-bandwidth intra-node GPU-to-GPU interconnect. The reason TP works inside a node but is painful across nodes.
- **InfiniBand / RoCE / RDMA** — the inter-node high-bandwidth fabric options. Where PP and EP cross machines.
- **DCGM** — NVIDIA Data Center GPU Manager. The source-of-truth agent for GPU-level metrics (SM activity, memory, power, ECC) that your Prometheus stack ingests.
- **MIG / MPS / DRA** — three distinct GPU-sharing mechanisms (hardware partition / soft time-share / Kubernetes-native Dynamic Resource Allocation). Covered in module 109.

## Mapping terms to engine primitives — reference table

| Vocabulary term | Engine primitive(s) that implement it |
|---|---|
| TTFT | Serving front-end timing between admission and first-chunk flush; influenced by scheduler + prefill + prefix-cache |
| TPOT | Decode-step timing from the continuous-batching loop |
| ITL | Streaming pipe between engine and HTTP layer |
| E2E | Request-level timing in the API gateway |
| Throughput | Fleet-wide token counter |
| Goodput | Throughput filtered by SLO conformance |
| Cache-hit rate | PagedAttention block-match counter / RadixAttention tree-hit counter |
| Preemption rate | Scheduler eviction counter |
| Continuous batching | Iteration-level admission loop in the scheduler |
| Chunked prefill | `enable-chunked-prefill` + chunk-size knob |
| Prefix caching | Block hashing (PagedAttention) or radix-tree sharing (RadixAttention) |
| PagedAttention | vLLM block allocator |
| RadixAttention | SGLang radix-tree KV organisation |
| Disaggregated prefill-decode | Separate worker pools + KV transport (Dynamo, DistServe, V1 vLLM P/D) |
| Speculative decoding | Draft-target / Medusa / EAGLE verification path in decode loop |
| Tensor parallelism | `tensor-parallel-size`, intra-node collectives |
| Pipeline parallelism | `pipeline-parallel-size`, inter-stage activations |
| Expert parallelism | `expert-parallel-size`, AllToAll dispatch |
| LoRA swap | `enable-lora` + per-request adapter id, adapter fusion in forward pass |
| Model sharding | Umbrella — TP + PP (+ EP) composition |
| Structured output | Outlines / XGrammar / guided-decoding backend |
| Streaming | SSE emission + backpressure + client-disconnect handling |

## Summary

- The latency vocabulary (TTFT / TPOT / ITL / E2E) and the throughput vocabulary (throughput / goodput / cache-hit / preemption) are the four axes of every inference SLO conversation.
- Each term binds to a concrete engine primitive. "TTFT" is not a vibe; it is a measurement tied to the scheduler and the prefix cache.
- The parallelism vocabulary (TP / PP / EP + DP attention) names *where* the model is split across devices; the KV-cache vocabulary (PagedAttention / RadixAttention / prefix caching) names *how* reusable state is kept.
- Goodput is the single most important term in this chapter — it is the metric that lets you compare engines, topologies, and vendors without being fooled by throughput-at-infinite-latency numbers.

Chapter 04 pins this vocabulary to the hardware you'll actually run it on.
