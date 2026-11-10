# Chapter 02 — SGLang Internals and Operation

> SGLang is not "vLLM with a different name." It is vLLM-adjacent in capability but very different in how it keeps KV state and in what it hands you as an API surface. This chapter is the operator-level walk.

## Motivation

Two workloads make SGLang the right default over vLLM:

1. **Prefix-heavy traffic** — long shared system prompts, RAG retrieval that repeats across requests, agent sessions that re-send a growing conversation plus a shared tool preamble. SGLang's **RadixAttention** turns these into cache lookups with reference-counted tree sharing.
2. **Structured generation at scale** — JSON Schema / grammar / Pydantic-constrained output where the throughput cost of constrained decoding has to be absorbed without crushing TPOT. SGLang's structured-output path is a first-class design axis, not an afterthought.

Everything else — scheduler loop, KV allocator shape, request lifecycle — rhymes with the five-layer map from chapter 01. We walk the same five layers for SGLang, then focus on the three places it is meaningfully different: RadixAttention, the zero-overhead batch scheduler, and cache-aware routing.

As with chapter 01, every specific flag or module-name claim below is a planning assumption. Internal module names, class names, launch-flag names, and defaults rotate between SGLang releases — before putting any of this in a production runbook, verify against the version your fleet runs (see resources.md).

## The five-layer anatomy, SGLang edition

```
    ┌───────────────────────────────────────────────────────┐
    │ 1. API frontend                                       │
    │    HTTP + SSE; OpenAI-compatible endpoints            │
    │    + SGLang front-end language runtime                │
    └───────────────────────────────────────────────────────┘
                      │
                      ▼
    ┌───────────────────────────────────────────────────────┐
    │ 2. Router / tokenizer-manager                         │
    │    cache-aware routing, DP attention dispatch         │
    └───────────────────────────────────────────────────────┘
                      │
                      ▼
    ┌───────────────────────────────────────────────────────┐
    │ 3. Scheduler  (zero-overhead batch scheduler)         │
    │    continuous batching, chunked prefill, speculative  │
    └───────────────────────────────────────────────────────┘
                      │
                      ▼
    ┌───────────────────────────────────────────────────────┐
    │ 4. KV allocator  (RadixAttention)                     │
    │    radix-tree prefix sharing, LRU eviction            │
    └───────────────────────────────────────────────────────┘
                      │
                      ▼
    ┌───────────────────────────────────────────────────────┐
    │ 5. Model runner + Worker                              │
    │    attention backends (FlashInfer, Flash Attention),  │
    │    TP / EP / DP-attention collectives                 │
    └───────────────────────────────────────────────────────┘
```

Three layers are near-identical to vLLM in spirit and need no re-explaining: the API frontend (layer 1), the scheduler (layer 3), and the model runner / worker (layer 5). The two layers that *are* different are the router / tokenizer-manager (layer 2) and the KV allocator (layer 4). We go deep on those next.

### Layer 1 — API frontend plus SGLang front-end language

Two surfaces, one process:

- **OpenAI-compatible HTTP** — `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, Pythonic `openai.OpenAI(base_url=...)` compatibility. The same contract vLLM exposes.
- **SGLang front-end language** — a Python DSL that lets a client express a *program* against the model (branching, fork/merge, structured generation, batched speculation, tool-call loops) as a single orchestration object the server executes. The DSL is optional; you can run SGLang as a drop-in OpenAI backend without touching it. Teams that need prompt-level programs (agents, structured workflows) prefer the DSL because it lets the server coalesce redundant prefills across branches.

### Layer 2 — Router and tokenizer-manager

Where SGLang parts company with vLLM. The router (sometimes called the "tokenizer manager" in older names) sits between the HTTP layer and the engine scheduler and does two things:

1. **Tokenizes** the incoming prompt and builds a sequence object.
2. **Routes** that sequence to a worker when the engine runs multiple workers behind a single endpoint — and, crucially, routes **cache-aware**: it tries to send sequences that share a token-prefix to the same worker, so the worker's RadixAttention cache absorbs the prefix instead of each worker having to prefill it.

Cache-aware routing is the SGLang-shape answer to "how do you keep RadixAttention useful when you scale out beyond one worker." The cross-worker alternative — move the KV between workers — is harder and lives in the tiered-KV / disaggregated-serving territory of module 104 and module 107.

### Layer 3 — Zero-overhead batch scheduler

Same shape as vLLM's scheduler (iteration-level continuous batching, prefill vs. decode interleaving, chunked prefill, preemption). The "zero-overhead" claim refers to the engineering target of keeping scheduler overhead off the critical path of each decode step — overlapping scheduler bookkeeping with the forward-pass GPU work rather than serialising it.

Knobs you tune at layer 3:

- `--max-running-requests` — SGLang's equivalent of `--max-num-seqs`.
- `--max-total-tokens` / `--max-prefill-tokens` — equivalents of vLLM's `--max-num-batched-tokens`; the single most load-bearing knob for the throughput-vs-latency Pareto.
- `--chunked-prefill-size` — chunk size when chunked prefill is enabled.
- `--schedule-policy` — scheduling policy (FCFS, cache-aware, priority — verify names and semantics in your version).
- `--enable-torch-compile` / compile-level flags — turn on torch-compile optimisation of the model code path when supported.

### Layer 4 — RadixAttention: the KV allocator

Everything above is familiar. Layer 4 is where SGLang's design is distinct.

#### The radix tree

Instead of hashing fixed-size blocks (PagedAttention's design), SGLang indexes KV state as a **radix tree over token sequences**:

- Each node holds a contiguous span of tokens plus the KV data for those tokens.
- Shared prefixes are shared *nodes*. If request A sends tokens `[s1, s2, s3, s4, s5]` and request B sends `[s1, s2, s3, s7, s8]`, the tree stores `[s1, s2, s3]` once with reference count 2, and branches to the two divergent suffixes.
- Reference counts drive eviction. A node that is referenced by at least one in-flight request cannot be evicted; a node with refcount 0 can be freed when the pool is pressured, in LRU-ish order.

The win is at the data-structure level: cache-hit lookup is a single tree walk from root along the request's token sequence; the first point of divergence is the prefill boundary. Deeply shared prefixes (long system prompts, repeated tool preambles) become one small node near the root that services many requests.

#### Eviction

Memory is still finite. When the KV pool fills, SGLang evicts the least recently used node whose refcount is zero. The algorithmic details and tie-breaks are version-dependent; the operator-level upshot is:

- **Workloads with high prefix locality** evict almost nothing — the hot nodes stay pinned.
- **Workloads with high prefix churn** (every request has a unique prompt, no shared prefix) evict aggressively, approaching the no-cache case. Then the cost of maintaining the tree is overhead with no benefit — turn RadixAttention off, or expect reduced returns.

#### Fragmentation

RadixAttention trades PagedAttention's "fixed-size blocks, allocator fragmentation possible" for "variable-size nodes, less fragmentation possible but tree walks and node splits as the cost." A node that is partially referenced may be split when a new request's prefix diverges partway through it; too many splits grow the tree.

Operationally: if you see RadixAttention's cache *size in bytes* climb faster than its *hit rate*, you have fragmentation / split overhead with limited reuse benefit. Reconsider the workload fit.

#### Metrics to watch

- Cache hit rate (per request, per worker, aggregated).
- Tokens evicted per second.
- Tree node count (proxy for fragmentation).
- Per-worker cache size vs. its KV pool budget.

Metric names rotate between versions; verify against the SGLang observability docs for yours.

### Layer 5 — Model runner, worker, and parallelism

Same shape as vLLM: a per-worker model-forward step, CUDA graphs for the decode path, attention backends plugged in (FlashInfer is the usual primary; Flash Attention is the alternative). What's additionally distinctive for SGLang:

#### DP attention

For MoE models, SGLang's **data-parallel attention** pattern lets each rank hold its own attention state (data-parallel across ranks for the attention block) while the MoE expert layer runs expert-parallel. Different ranks hold different *request subsets'* attention state; the two parallelism axes compose cleanly.

Flag family: `--dp-size` and friends (verify current flag names). You reach for DP attention when:

- The MoE model's attention is a decode-time bottleneck at scale.
- Expert parallelism alone leaves attention memory-bandwidth-starved on each rank.
- The traffic shape (many small requests, lots of concurrency) benefits from each rank running its own attention batch independently.

DP attention is one of the operator-visible advantages SGLang has over vLLM for large-MoE serving today; verify current state against the SGLang release notes before committing in production.

#### Tensor and expert parallelism

- `--tensor-parallel-size N` — same intra-node TP as vLLM.
- `--expert-parallel-size N` — EP for MoE.
- `--pipeline-parallel-size N` — PP across nodes (verify current support status).

### The structured-generation front-end

This is layer 1 territory but worth its own section.

SGLang supports structured output via grammar / JSON Schema / regex constraints enforced at the token-sampling layer: at each decode step, the next-token distribution is masked to the set of tokens that keep the output within the grammar. On high-constraint schemas this can be the dominant cost per step — the mask construction, not the LM forward, becomes the critical path.

The engine integrates with XGrammar / Outlines-shape libraries (verify current backend list for your version). Knobs:

- Guided-decoding mode selection (JSON Schema / regex / grammar).
- Grammar cache size (grammars compile once; the compiled automaton is cached).
- Fast-forward decoding (when the grammar forces a run of tokens, the engine can emit them without a sampling step).

For agent and tool-call-heavy workloads this is the layer that makes "guaranteed JSON" tractable at TPOT budgets. Module 111 goes deeper into the API contract; this chapter covers the mechanical reality that SGLang bakes it in.

## Request lifecycle, SGLang edition

Walk one `POST /v1/chat/completions` through the layers:

1. **Ingress.** HTTP handler validates the request, builds a sequence object with tokenized prompt and `SamplingParams`-equivalent. If the request uses the SGLang DSL, the server executes the program steps; otherwise it treats this as a single-generation call.
2. **Route.** The router consults the per-worker RadixAttention state. It picks the worker whose cache has the longest shared prefix with this request's tokens. The request is enqueued at that worker.
3. **Admit.** The worker's scheduler pulls from its waiting queue, allocating KV through RadixAttention: the tree walk finds the longest shared prefix, bumps refcounts, and reserves nodes for the new suffix.
4. **Prefill.** The runner computes KV for the suffix only (the prefix is already present). This is where **TTFT** is kept low: on a 10k-token prompt that shares a 9k-token prefix, the engine prefills 1k tokens, not 10k.
5. **Stream.** First token flushes through the HTTP layer.
6. **Decode loop.** Zero-overhead scheduler keeps bookkeeping off the critical path. Chunked prefill, if enabled, continues to interleave prefill work for other requests between decode steps.
7. **Finish.** Sequence completes; tree nodes have their refcounts decremented. Nodes that reach refcount 0 become candidates for LRU eviction later.

The *cache-aware routing + RadixAttention* combination means that on a prefix-heavy workload at steady state, most requests prefill only their suffix, and the engine spends most of its compute on decode. This is the operational payoff SGLang is engineered for.

## Launch surface — a representative command

A production-shape launch for a long-context chat workload with a shared system prompt, on an 8× H200 node:

```bash
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.3-70B-Instruct \
    --tensor-parallel-size 8 \
    --context-length 32768 \
    --max-running-requests 256 \
    --max-total-tokens 65536 \
    --mem-fraction-static 0.90 \
    --chunked-prefill-size 2048 \
    --schedule-policy lpm \
    --enable-torch-compile \
    --host 0.0.0.0 --port 30000
```

Reading each line against the five layers:

- `--tensor-parallel-size 8` → layer 5. Eight GPUs share the forward pass.
- `--context-length 32768` → layer 4 ceiling per sequence.
- `--max-running-requests 256`, `--max-total-tokens 65536` → layer 3 scheduler caps.
- `--mem-fraction-static 0.90` → layer 4 pool sizing.
- `--chunked-prefill-size 2048` → layer 3 chunk size.
- `--schedule-policy lpm` → layer 3 scheduler policy (longest-prefix-match-aware; verify current policy names). This is the SGLang-distinct knob — the scheduler itself favours cache-hitting requests.
- `--enable-torch-compile` → layer 5 compile-time optimisation.

All names and defaults rotate release-to-release. Verify every flag name, default, and schedule-policy value above against the SGLang release matching your fleet; the semantics are stable, the surface is not.

## When SGLang wins over vLLM, and when it doesn't

- **Wins** — shared system prompts, chat with growing conversation history, RAG with repeated retrieval context, agent sessions with shared tool preambles, structured output as the critical path, DP-attention-shaped MoE at scale.
- **Doesn't win** — every request has a unique prompt (prefix cache cold → RadixAttention is overhead), the engine ecosystem you depend on (observability, LoRA tooling, speculative-decoding breadth) is further along in vLLM, or the model you need lands day-one on vLLM but experimentally on SGLang.
- **Operational cost** — the "two engines in production" tax. Running both vLLM and SGLang means two sets of dashboards, two runbooks, two upgrade windows. Many fleets choose a single engine on operational grounds alone.

## Debugging mental checklist

Same five layers; the SGLang-specific moves:

1. **Ingress (layer 1):** OpenAI-compatible shape OK? SSE flushing? If the DSL is in use, is the program serialising to the server correctly?
2. **Router (layer 2):** is cache-aware routing active? Are requests being distributed in a way that keeps prefix locality, or are they ping-ponging across workers?
3. **Scheduler (layer 3):** schedule policy correct for the workload? Chunked prefill size appropriate? Zero-overhead scheduler not overlapping with the forward pass (if it is, step time climbs).
4. **RadixAttention (layer 4):** cache hit rate climbing or dropping? Node count growing linearly with request count (fragmentation sign) or sublinearly (good)? Eviction rate matching workload prefix churn?
5. **Runner / worker (layer 5):** DP-attention topology correct for the MoE? Rank utilisation balanced? Any NCCL noise?

The specific knobs change; the diagnostic order doesn't.

## Boundary notes

- **FlashInfer, Flash Attention, and the structured-output grammar backends** (XGrammar, Outlines) are external libraries SGLang integrates. Bugs in them are their maintainers'; your job is to pin a version that works, open issues upstream, and manage the version drift.
- **Cross-worker KV transfer** — moving a request's KV from one worker to another so a cache-miss worker can benefit from a cache-hit worker's state — is tiered-KV / disaggregated-serving territory (modules 104, 107).
- **DP-attention mathematics** — kernel-level math for the parallelism primitive — is the performance peer at L35. You operate the topology.

## Summary

- SGLang's five-layer anatomy mirrors vLLM's. Three layers are near-identical in shape; two are meaningfully different: the router / tokenizer-manager and the KV allocator.
- RadixAttention organises KV state as a radix tree over token sequences. It wins when prefix locality is high and becomes overhead when prefix churn is high.
- Cache-aware routing is how RadixAttention stays useful across multiple workers — the router sends prefix-sharing requests to the same worker.
- The zero-overhead batch scheduler keeps scheduler bookkeeping off the critical path. The structured-generation front-end makes JSON / grammar enforcement a first-class citizen.
- Data-parallel attention is SGLang's operator-visible advantage for large MoE serving at scale.
- Flag names and defaults rotate between releases. Treat this chapter as the *shape* of the surface; verify current names and defaults against the SGLang release notes for your fleet.

Chapter 03 crosses to TensorRT-LLM, where the shape of the engine inverts: an explicit compile step first, served through Triton, orchestrated for disaggregation by Dynamo.
