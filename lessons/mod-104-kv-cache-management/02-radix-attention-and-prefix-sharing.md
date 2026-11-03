# Chapter 02 — RadixAttention and Prefix Sharing at Operator Depth

> PagedAttention organises a sequence's KV. RadixAttention organises *the fleet's* KV — so a system prompt that 10 000 requests share is computed once and read 10 000 times.

## Motivation

Chapter 01 treated the block pool as a per-sequence resource: each request got its blocks on admit, held them until finish, and released them. That worked, but it left the biggest cost source in a modern inference fleet un-amortised: the **shared prefix**. In chat, every request begins with the same system prompt. In RAG, every answer to a hot document reuses the same retrieved context. In agentic workloads, every tool-call iteration re-sends an accumulating tool-use trace. Prefill is linear in prompt length — if the prefix is 2 000 tokens and traffic is 100 RPS, the engine recomputes 200 000 identical prefill tokens per second across the fleet.

**Prefix caching** fixes that. **RadixAttention** (Zheng et al., 2023; see [`resources.md`](resources.md)) is the data structure most engines now use to make it work: a radix tree of token sequences, indexed to the KV blocks that already hold their key-value tensors. When a new request arrives, the engine walks the tree with the request's prompt, finds the longest matching prefix, and reuses those blocks instead of recomputing them.

By the end of this chapter you can walk the radix cache's lookup / insert / evict path, name the three cache-eviction choices, understand why *routing* matters as much as the cache data structure, and know when prefix caching helps vs. hurts.

<!-- needs-research: prefix-cache implementation details differ substantively between SGLang (RadixAttention paper design), vLLM (automatic-prefix-caching, hash-based), and TRT-LLM (`enable_block_reuse`). Verify specific class names and flag defaults against the exact engine version you run. -->

## The problem prefix caching solves

Measure, on your fleet, the ratio

```
prefill_tokens_processed_per_second / prefill_tokens_unique_per_second
```

For a typical chat workload with a shared system prompt, this ratio is often 5–20×. Every one of those duplicate tokens is wasted compute on prefill — compute that is contending with decode for the GPU, driving up TTFT for whoever is being prefilled, driving up TPOT for whoever is decoding alongside.

If the KV for the shared prefix lives in the pool already, prefill does not need to run on it. The engine can walk the prefix in the cache, point the new request's block table at the existing physical blocks, and start prefill only at the first *diverging* token. The compute saving is linear in the shared prefix length; the TTFT saving is close to linear too, because prefill wall-clock is dominated by prompt length for prompts above a few hundred tokens.

The two things an operator must then arrange:

1. The cached KV must still be in the pool when the next request arrives — the cache must be **retained** (not immediately freed) and **evicted on policy** (not randomly).
2. The next request must reach the worker that holds the cached KV — this is the **cache-aware routing** problem. On a single worker this is automatic; on a fleet it is a router responsibility.

## The radix tree as a KV index

### What a radix tree is, briefly

A radix tree (compressed trie) is a tree where each edge is labelled with a *string* (not a single character), and paths from the root to a node spell out a key. In RadixAttention the keys are **token sequences** and the values are **block table ranges** (pointers to the physical KV blocks that hold those tokens' key-value tensors).

```
root
├── [sys_prompt_v1_tokens] → KV blocks [B1..B50]
│   ├── [user_turn_A_tokens] → KV blocks [B51..B55]
│   └── [user_turn_B_tokens] → KV blocks [B61..B64]
├── [sys_prompt_v2_tokens] → KV blocks [B70..B130]
│   └── [user_turn_C_tokens] → KV blocks [B131..B136]
└── [retrieved_doc_42_tokens] → KV blocks [B200..B280]
    └── [query_X_tokens] → KV blocks [B281..B286]
```

Three operations the operator cares about:

- **`match_prefix(tokens)`** — walk the tree from the root, consuming tokens from the input until a divergence; return the longest matching path and the KV blocks along it.
- **`insert(tokens, blocks)`** — add a new branch (or extend an existing one) with the KV blocks that were just computed.
- **`evict(bytes_to_free)`** — remove nodes to free blocks per the eviction policy (LRU, TTL, size-weighted).

### The vLLM variant

vLLM's "automatic prefix caching" uses a slightly different mechanism conceptually — hashing block-aligned token sequences to detect reuse — but the operator-visible contract is the same: a prompt arrives, blocks whose contents match get reused, blocks past the divergence point get allocated normally. The data structure is a hash table of block-aligned prefixes; the radix-tree abstraction above is the mental model to use regardless.

### The TRT-LLM variant

TRT-LLM exposes block-level reuse through configuration flags (names rotate; see the TRT-LLM docs). The user-facing semantic is the same: enable it, and prefill reuses KV for matched prefixes.

## The lookup path, step by step

What happens when a request `R` with 1800-token prompt lands on a worker that has prefix caching on:

1. **Tokenise the prompt** (as always).
2. **Call `match_prefix(prompt_tokens)`.** Returns a prefix length `k` and the KV blocks `[B_1..B_m]` covering the first `k` tokens.
3. **Build the block table.** The sequence's block table is initialised with `[B_1..B_m]` for the first `k` tokens. Reference counts on those blocks are incremented (another sequence is now holding them).
4. **Allocate fresh blocks for `prompt_tokens[k:]`.** Length `1800 - k`. If `k = 1500` the request only needs to prefill 300 new tokens' worth of KV.
5. **Schedule prefill for `prompt_tokens[k:]` only.** The engine runs the forward pass over the divergent suffix; the attention kernel reads KV from the shared blocks for the matched prefix and the newly-written blocks for the suffix.
6. **On the first decode step and all subsequent decode steps**, the sequence proceeds as any other sequence — the only artefact of prefix caching is that part of its block table points at shared blocks.
7. **On finish**, the sequence's unshared blocks are freed as usual (chapter 01). The shared blocks' reference counts drop by one; if they hit zero AND the eviction policy allows, they return to the free registry. Typically they do not hit zero quickly — other requests are referencing them.

Operational: step 3's reference-count increment is what *retains* the cached blocks past this request's lifetime. The eviction policy is what frees them later.

## Cache-aware routing — the fleet-level half of prefix caching

A single worker with a radix cache gives you prefix sharing *within* its own cache. In a fleet of N workers behind a load balancer, if the balancer is prefix-blind (round-robin, least-connections, random), the same shared system prompt will be cached on *every* worker — multiplying memory cost and giving you only `1/N` effective hit rate.

The fix is **cache-aware routing**: the router inspects the request's prefix and sends it to the worker most likely to have that prefix cached.

### The simplest implementation: a hash of the prefix

Hash the first `k` tokens of the prompt (`k` = system-prompt length or a configured prefix window). Use the hash to pick a worker. All requests sharing a prefix hit the same worker. Known as "sticky routing" or "consistent hashing on prefix."

Trade-off: the worker that owns a hot prefix sees all the load for that prefix. If prefixes are heavy-tailed (one tenant's system prompt is 10× more popular than everyone else's), this creates a hotspot. Mitigations:

- Replicate hot prefixes across M workers; route round-robin among them.
- Fall back to least-connections for the top-K hottest prefixes.
- Monitor per-worker load vs. per-worker cache-hit-rate; if they diverge, your hash is sending too much to one shard.

### SGLang's approach

SGLang ships a cache-aware router (sometimes called the "router" or "load-balancer"; verify in docs) that peeks at the prefix and dispatches to the worker with the highest `prefix_match_length` for the incoming request. If no worker has a usable match, it falls back to least-loaded.

### vLLM

vLLM does not ship a cache-aware router in the engine itself (verify current release). The router lives outside the engine — in a gateway, a `vllm serve` fronting fleet via an external LB, LiteLLM-shape routing, or Dynamo. The operator's job is to wire the router to the metrics that let it decide. Dynamo (NVIDIA) and other recent fleet routers advertise cache-aware routing explicitly.

### The operational implication

Prefix caching's observed benefit = (per-worker cache hit rate) × (per-request prefix share). Routing is what maximises the first factor. A fleet with excellent per-worker caches and a prefix-blind router leaves most of the benefit on the floor.

## Eviction policy — LRU vs. TTL vs. size-weighted

The cache grows forever if you let it. Eviction policy decides which cached blocks go when the pool nears full.

### LRU (least recently used)

Classic. Each access updates the node's last-access timestamp. When eviction is triggered, the tree's least-recently-used leaves (or internal nodes with ref-count = 0) are removed.

- **Pros.** Simple. Works well for workloads with locality.
- **Cons.** Does not distinguish "cheap to recompute" from "expensive to recompute." A 50-token prefix and a 5000-token prefix are treated the same.
- **Default for most engines.**

### TTL (time-to-live)

Each cached node gets a lifetime; after that it is evicted regardless of usage.

- **Pros.** Bounds staleness. Useful for **conversation state**: a conversation's accumulating prefix is cached for the duration of the session; after N minutes of inactivity it is dropped. This is the right policy for chat servers that cache per-conversation KV.
- **Cons.** Hot prefixes may be evicted before genuinely cold ones if the TTL fires early.
- **Common for conversation-aware caches** (chat servers maintaining per-session state).

### Size-weighted (prefer evicting cheap-to-recompute)

Evict in an order that prefers freeing small nodes before large ones (because the small ones are cheapest to re-fill if needed). Or weight by `age / size` so a short-cold entry goes before a long-warm one.

- **Pros.** Preserves the expensive-to-rebuild entries.
- **Cons.** Harder to implement; some engines support it, some do not.

### What to pick

- **Shared system prompts in chat** → LRU is fine; the prefix is referenced by every request, access-recency is always high.
- **Conversation state** → TTL on the conversation branch. The conversation's branch extends with each turn; it dies when the user walks away.
- **Hot RAG documents** → LRU is fine; frequency keeps them warm.
- **Mixed workload with a few giant cached prefixes** → consider size-weighted if your engine supports it, or manually pin the giant prefixes via API.

Verify what your engine actually supports; the TTL and size-weighted policies vary by implementation.

## When prefix caching helps — and when it hurts

### Helps

- **Chat with a shared system prompt** (long prompt, short varying user turn).
- **RAG where the retrieved document is on the hot set.**
- **Agentic / tool-use loops** where each iteration re-sends the growing trace.
- **Few-shot prompts** where the exemplars are stable.
- **Interactive conversation** where the session's prefix grows turn by turn.

For these, measure the hit rate (chapter 03 does the method); it should be 60–95% depending on workload homogeneity.

### Hurts

- **Workloads with no shared prefix.** Every prompt is unique (e.g. randomly-sampled few-shot, highly personalised prompts, embeddings-style one-shots). The cache occupies KV pool without producing hits. Pure overhead.
- **Workloads where the prefix diverges early.** If user queries diverge within the first 50 tokens on average, the hit depth is < 50 tokens and the benefit is in the noise. The pool is better used for live sequences.
- **Fleets with prefix-blind routing.** The cache is populated redundantly on every worker. Memory cost multiplies; benefit does not.

For these, **turn prefix caching off** and reclaim the KV budget for live sequences. The flag varies; look for the "automatic prefix caching" or "radix cache" switch in your engine's docs.

### The decision

Instrument. Prefix caching is a feature flag with a cost signature. If `cache_hits / (cache_hits + cache_misses) < ~30%` sustained, you are paying memory cost for little benefit. Over ~50% is clearly worth it. Between those is a judgment call; chapter 03 of this module is the method.

## Interaction with copy-on-write

Prefix caching turns every cached prefix into a *shared* block (reference count > 1). The first write past the shared span — the first new prefill or decode block that extends past the cached prefix — is a fresh block. The shared blocks themselves are read-only during attention (their KV never changes) so copy-on-write is simpler: there is no "the sibling wrote to a shared trailing block" because the shared blocks are always past the trailing edge of the shared prefix. Nonetheless, the ref-counting apparatus from chapter 01 is what retains and later releases the shared blocks.

## Interaction with chunked prefill

If chunked prefill is on (module 103, chapter 03), prefill of the *divergent suffix* is chunked as usual. The shared-prefix portion contributes zero chunks (there is no prefill to run; the KV is already in place). This means prefix-cache hits produce *strictly fewer* prefill chunks per request, which strictly reduces iteration-time pressure. The two features compose cleanly.

## Interaction with speculative decoding

Speculative decoding (module 108) uses a cheap draft to propose tokens; the target verifies. On the target's side, the proposed-token prefill is prefix-caching-friendly for the first iteration (the actual conversation prefix is cached) and runs as usual for the proposed suffix. No conflict.

## Where to find the prefix cache in source

### SGLang

- Radix cache — `python/sglang/srt/mem_cache/radix_cache.py` (or similar; verify path).
- Match / insert / evict methods — grep for `match_prefix` or `insert` on the radix-cache class.
- Cache-aware router — SGLang's router module, outside the engine process (verify current docs; path has moved).

### vLLM

- Automatic prefix caching — in the block manager module. Grep for `prefix` in `vllm/core/`. The implementation hashes block-aligned token sequences.
- Enable with `--enable-prefix-caching` (verify flag spelling for your release).

### TensorRT-LLM (via Triton)

- Enable with the KV-cache-reuse flag in the TRT-LLM runtime config / `config.pbtxt` (name varies by release; look in the KV-cache config section of the docs).

## Common misreadings

- **"Prefix caching is always on and always free."** It costs KV-pool memory to retain cached blocks. On a workload with no sharing it is pure overhead. Measure before enabling in production.
- **"A radix tree is just an implementation detail."** It also commits the engine to **block-aligned** prefix matching — the shortest cacheable prefix is one block (16 tokens by default). Prefixes that differ in their first block cannot share. Design workloads with this in mind (keep stable prefixes block-aligned, pad if necessary).
- **"Cache-aware routing is a nice-to-have."** It is 90% of the fleet-level benefit. Without it, each worker caches independently and most of your gains evaporate.
- **"LRU is always right."** For *conversation state* with inherent lifespan (chat session), TTL is often better. For *reference prefixes* (system prompts, RAG docs), LRU is right.
- **"Prefix caching will fix a cold-start problem."** Only after the cache is warm. A worker that just scaled up has an empty cache; its first N requests pay full prefill. Autoscaling (module 109) interacts with cache warm-up — scale-ups appear as temporary TTFT spikes until caches fill.

## Summary

- **Prefix caching** amortises prefill compute across requests that share a prompt prefix (chat system prompts, RAG retrieved contexts, agentic trace prefixes).
- **RadixAttention** indexes cached prefixes in a radix tree keyed by token sequences; the SGLang implementation is the canonical one; vLLM / TRT-LLM have equivalent capabilities under different names and data structures.
- **Lookup is `match_prefix(prompt_tokens)`** → longest matching prefix in blocks, incremented ref-count, block-table seeded with those blocks, prefill scheduled only for the divergent suffix.
- **Cache-aware routing** is the fleet-level half of the picture: route a request to the worker that already holds its prefix. Without this, per-worker caches populate redundantly and benefit evaporates.
- **Eviction policy** — LRU is the default and right for shared prefixes; **TTL** is right for conversation state; **size-weighted** is a niche but useful variant where supported.
- **When prefix caching helps:** chat, RAG, agentic loops, few-shot — anywhere prompts share a long head. **When it hurts:** fully-unique prompts, divergence-early workloads, fleets without cache-aware routing.
- **The algorithm is the kernel specialist's.** The runtime object — the retained cache, the eviction policy, the router — is yours.

Chapter 03 turns these mechanics into a deployment: enable prefix caching correctly on chat and RAG workloads, measure the hit rate as a first-class metric, defend the TTFT benefit, and name the cost of a miss.
