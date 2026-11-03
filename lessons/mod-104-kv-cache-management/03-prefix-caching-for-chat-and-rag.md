# Chapter 03 — Deploying Prefix Caching for Chat and RAG

> The flag is one line. The policy around it — which prefixes to cache, where to route them, how to measure hit rate as an SLO axis, how to recognise a cold-cache cliff on a dashboard — is a chapter.

## Motivation

Chapter 02 established the mechanism. This chapter is the deployment. The audience is you, authoring the engine config and the fleet routing for a workload where prefix caching will be the single largest TTFT improvement available — chat with a shared system prompt, RAG with hot documents, agentic loops with growing traces — and defending the choice with measurements.

Two workloads anchor the chapter: **chat** (shared system prompt + varying user turns + growing conversation state) and **RAG** (varying queries + retrieved-document prefix on hot documents). Both benefit from prefix caching. Both have failure modes if deployed carelessly.

<!-- needs-research: the exact spelling of the prefix-caching flag ("--enable-prefix-caching" in vLLM; different name in SGLang; TRT-LLM uses a different config key again) rotates between releases. Verify against the version your fleet runs. -->

## The two workloads at a glance

### Chat

```
system_prompt (shared across all users, ~1 500 tokens)
+ (optional) conversation history (grows with each turn, per user)
+ latest_user_message (varies, ~50 tokens)
```

Prefix structure: a shared head (system prompt), a per-conversation middle (history), a per-turn tail (new user message + any formatting tokens). Hit opportunity is huge: the system prompt is shared across every request; the conversation history is shared across every turn within one session.

### RAG

```
system_prompt (shared, ~200 tokens)
+ retrieved_doc_chunks (varies by query, but a hot set of docs recurs)
+ user_query (varies, ~30 tokens)
```

Prefix structure: shared system prompt + variable retrieved-doc section + per-query tail. Hit opportunity is moderate and skewed: the top-K most-popular docs recur frequently (Zipf-like); the long tail of rare docs never hits. Operators need to measure which portion of traffic falls on hot docs.

### What is *not* chat or RAG

- Pure one-shot API traffic with no shared prefix (e.g. per-request-crafted prompts with no template). Prefix caching has no opportunity here; turn it off.
- Highly personalised prompts where the "system" portion includes a per-user profile at the top. Only shares across the same user, which may or may not meet the hit-rate threshold.

## Enabling it, concretely

### vLLM

```bash
vllm serve <model> \
    --enable-prefix-caching \
    ...
```

Verify flag spelling for your release. On enable, the automatic-prefix-caching data structure (block-aligned hash) activates; new requests look up block-aligned prefixes against the cache before scheduling prefill.

### SGLang

SGLang defaults to RadixAttention on; the knob is the eviction policy and the cache size. Verify in the server-args reference.

### TensorRT-LLM

Enable KV cache reuse via the runtime config (name varies by release; look for the KV-cache-reuse section of the TRT-LLM docs).

### Verify it is actually on

After launch, send two identical prompts back-to-back and watch TTFT: the second should be substantially faster than the first. If TTFT is unchanged, prefix caching is not firing — either the flag is wrong, the engine does not support it on this backend, or the second request landed on a different worker. In the single-worker case the first two are the explanations; in the multi-worker case add "no cache-aware routing."

## Measurements the operator *must* publish

Prefix caching without instrumentation is a flag you cannot defend. These are the axes.

### Cache-hit rate

```
cache_hit_rate = cache_hits / (cache_hits + cache_misses)
```

Granularity: both at the per-request level ("did this request find any matching prefix?") and at the per-block level ("of this request's prompt blocks, how many were cache hits?").

**Hit-rate expectations by workload shape:**

- **Chat with a long shared system prompt and conversation state**: 60–95% block-level hit rate is realistic. Hits should concentrate on the system-prompt and conversation-history prefixes.
- **RAG with a long-tail document distribution**: 20–60% block-level. Lower because the long tail of unique queries drags the average down.
- **Agentic loops**: 70–90% because each tool-call iteration re-sends the growing trace.
- **Random / one-shot workloads**: < 10%. Turn prefix caching off.

Expose both metrics (per-request and per-block) in Prometheus. The per-block number is the one that correlates with TTFT savings; the per-request number is what tells you how many requests benefited at all.

### Prefix-cache-hit benefit on TTFT

For each request, compute

```
prefill_tokens_saved = cache_hit_block_count × block_size
estimated_ttft_saving ≈ prefill_tokens_saved / (prefill tokens per second at this batch mix)
```

Published as a histogram of TTFT savings per request, keyed by cache_hit vs. miss. In production dashboards:

- Panel: **TTFT P50 / P99 for cache-hit requests** vs. **TTFT P50 / P99 for cache-miss requests**. The gap is the benefit.
- Panel: **TTFT P50 of the top-10 longest cached prefixes** — the "big winners." Usually dominated by one or two system prompts.

### Cost of a cold miss

A cache miss costs:

- Full prefill compute on the entire prompt — the TTFT a non-caching engine would show.
- No incremental cost beyond that; the miss simply degrades to the default behaviour.

The thing to watch is the **first wave after a scale-up event or a cache flush**: every new worker starts with a cold cache. For the first few minutes, TTFT at the fleet level will track the cache-miss TTFT. If your TTFT SLO is tight, this transient is important — see "cold-cache cliff" below.

### Pool-occupancy share of the cache

```
cache_bytes_in_use / total_kv_pool_bytes
```

Prefix caching retains blocks past their owning request's finish. On a healthy cache this is 20–60% of the pool. If the cache has eaten > 80% of the pool, live sequences are starved and admission pressure is high — tune the eviction policy or lower the retention target.

### Hit-rate per prefix (optional, powerful)

If your engine exposes top-K cached prefixes with hit counts, publish them. "The top 5 prefixes are responsible for 70% of hits" is a one-line explanation of the workload's prefix structure and the single most useful artefact for capacity planning.

## The cold-cache cliff — the one operational surprise

Scenario: traffic is steady, hit rate is 80%, TTFT P99 is well under SLO. A worker scales up (new pod, Karpenter brought on a new node, canary cuts over). The new worker has an empty cache. For the first few minutes, requests routed to it experience full-prefill TTFT.

The fleet-level TTFT dashboard shows a step-up: not an outage, but a visible cliff. If the cliff crosses the SLO threshold, you have a problem.

Mitigations:

- **Cache-aware routing**: continue sending prefix-matching requests to warm workers for a grace period; only send a small fraction to the new worker. Lets its cache warm gradually.
- **Pre-warm the cache**: at pod-start, replay the top-K system prompts (or a curated priming set) to seed the cache before accepting traffic. Not a built-in feature of most engines; must be scripted at deploy time.
- **Readiness gating**: delay the new worker's addition to the LB until it has seen a configurable number of prefix-cache insertions. Operationally fragile; mention and move on.

The cliff is the single most common prefix-cache-related post-mortem. If TTFT SLO is tight and traffic is prefix-cache-sensitive, build the mitigation into the autoscaler before you need it.

## Chat: deploy pattern

### Shared system prompt

Treat the system prompt as a **cache primary**:

- Keep it *stable*. Versioning the system prompt invalidates the cache for every request. Deploy prompt changes behind a feature flag and expect a transient TTFT regression at cutover.
- Keep it *block-aligned*. If the system prompt is 1500 tokens, pad to 1504 (nearest multiple of 16) with innocuous tokens so the block boundary is predictable. Or ensure the next cached region (conversation history, user turn) always starts at a block boundary.
- Keep it *at the head of the prompt*. Any dynamic text before the system prompt (per-request timestamps, user IDs, request-specific metadata) poisons the cache. Move those to the end of the prompt or into a separate metadata header.

Common anti-pattern: injecting a timestamp or request ID at the top of the system prompt "for traceability." Every request has a unique prefix; hit rate is zero. Fix by moving the dynamic field to the tail.

### Conversation state

Treat each conversation as a growing cached prefix:

- The system prompt is the shared head.
- The conversation history (`user: ... \n assistant: ... \n user: ...`) extends the branch for that conversation.
- Each new turn is a new node extending the conversation's branch.

For this to work:

- **Sticky routing per conversation.** Route every request in a conversation to the same worker (hash by `conversation_id`). Otherwise each worker caches its own partial history and hit rate collapses.
- **TTL eviction**, not pure LRU, for conversation branches. A conversation that ends naturally should age out of cache even if the system prompt (its parent) remains hot. Chapter 02 goes deep.
- **Decide the TTL value** against the application's conversation-length distribution. If the median conversation lasts 5 minutes and the 95th percentile is 30 minutes, a TTL of 60 minutes is defensible. Shorter TTL → faster memory release but higher miss rate on long conversations.

### Explicit cache-key API (where available)

Some engines expose an API where the client can pass an explicit cache key (vLLM, SGLang; verify in your version). This lets the application opt into caching for a specific prefix without relying on the engine's token-level matching. Useful when:

- The prefix is binary / token-non-identical across versions but logically the same.
- You want to pin a hot prefix against eviction.
- You are implementing cross-worker cache sharing manually.

If available, treat as an optional precision tool, not a replacement for the automatic path.

## RAG: deploy pattern

### Retrieved-document prefixes

Each query has `(doc_1, doc_2, ..., doc_k)` concatenated as the retrieved context. For prefix caching:

- **Order matters.** `(doc_1, doc_2)` and `(doc_2, doc_1)` are two different prefixes with zero overlap. Sort retrieved docs deterministically — by relevance score, then by doc_id as a tie-breaker — so a given set of docs always produces the same prefix.
- **Chunk boundaries matter.** If a doc is split across retrieval chunks, concatenate the chunks in a deterministic order and emit the full concatenated prefix before the query. Random intra-doc chunking produces zero cache hits.
- **A doc should be block-aligned.** Pad doc separators (`\n\n`, `[DOC_END]`) so adjacent docs land on block boundaries. Mis-aligned docs produce partial hits that waste the first fractional block.

### Hot vs. tail

RAG hit rate is dominated by document frequency. If the top 1% of docs accounts for 50% of retrievals (common for FAQ-style QA), hit rate on that 1% is ~100% and on the rest is near zero.

- Publish **cache-hit rate stratified by doc-id**. Confirms whether the hot set is actually caching and whether the long tail is dragging the average.
- Decide **what to pin**. If your engine supports pinning (TRT-LLM: some versions do; SGLang: via policy; vLLM: via explicit cache-key API where available), pin the top-K hottest docs so they are never evicted.

### Cache invalidation on doc update

If a document in the hot set is updated:

- The cached KV is stale (wrong text underneath). Must be invalidated.
- Most engines do not expose a cache-invalidation API. The practical fix is to version the doc identifier in the prefix (e.g. `[DOC id=42 v=17]...`) so a new version produces a new prefix and the old cached one ages out.
- Alternative: restart workers on a cache-poisoning update. Operationally heavy; avoid if possible.

The versioning-in-prefix pattern is the standard answer.

## Fleet-level: wiring the cache-aware router

Chapter 02 named cache-aware routing as the fleet-level half of the picture. Deployment detail:

### Hashing scheme

Compute a hash of the first N tokens of the prompt (post-tokenisation). Use that hash to pick a worker. All requests sharing that prefix hit the same worker.

Common value for N: the length of the shared system prompt (block-aligned). For chat with 1 500-token system prompts, hash the first 1 504 tokens.

### Replication for hot prefixes

If the hot prefix is 10× more popular than the next, one worker is overloaded. Mitigate:

- Replicate the hot prefix across M workers. Route round-robin among them. Pay M× the KV cost for the hot prefix in exchange for load balance.
- Monitor per-worker request rate; if the top worker is 2× the median, replicate.

### Fallback for cache miss / cold worker

If the "preferred" worker is unhealthy, over-loaded, or down, route to the next-best match (second-longest cached prefix among workers) or fall back to least-loaded. Document the fallback order in the router config.

### Metrics to publish

- Router hit-rate per prefix bucket.
- Per-worker request rate (watch for hotspots).
- Per-worker cache-hit rate (watch for shards that have the wrong prefix in cache).

Dynamo (NVIDIA; see [`resources.md`](resources.md)) advertises cache-aware routing as a first-class feature. SGLang ships a router. For other setups, this lives in your gateway or in a LiteLLM-style router you own.

## Defending the config

An ops review of "we turned on prefix caching and set TTL to 60 minutes" should be able to answer:

1. **What is the hit rate on the real workload?** With a graph covering a week.
2. **What is the TTFT delta between hit and miss?** As a histogram, not a mean.
3. **What fraction of the KV pool does the cache occupy?** Is this crowding live sequences?
4. **What happens on scale-up?** Is there a cold-cache cliff? Does it cross SLO?
5. **What is the invalidation story?** If the system prompt or a hot doc is updated, how long before the stale entries age out?
6. **What is the per-tenant isolation story?** (Chapter 06.)

If any of these does not have a one-paragraph answer, the config is not production-ready.

## When to turn prefix caching *off*

- Hit rate < 20% sustained over a week. Pure overhead; reclaim the KV budget for live sequences.
- Workload is dominated by uncacheable traffic (per-request-unique prompts) and the cache is causing admission pressure.
- A security review (chapter 06) has not completed and the workload has strong per-tenant data boundaries.
- An engine upgrade has introduced a prefix-cache regression. Roll off until fixed; this has happened in multiple engines at different versions.

## Common misreadings

- **"Prefix caching is a free TTFT win."** Only above a hit-rate threshold and only if the pool can afford the retention cost. Below ~30% it is net-neutral; below ~15% it is net-negative because the cache evicts blocks that could serve live sequences.
- **"TTFT improvement is proportional to hit rate."** Approximately — but prefill scales with prompt length, not with request count. A 90% hit rate on a workload with 100-token prompts saves far less wall-clock than a 60% hit rate on a workload with 2 000-token prompts. The right metric is prefill-tokens-saved, not hits.
- **"Longer TTL is always better."** Longer TTL retains more cache, which can starve live sequences in the KV pool. The optimum is the shortest TTL that catches your conversation / session distribution.
- **"Cache-aware routing is for mega-scale fleets."** Any fleet of two or more workers behind an LB benefits. The marginal implementation cost is a hash function and a worker-id lookup.
- **"Dynamic timestamps in the system prompt are fine; they're small."** They are small *and* they poison every request's prefix. Zero hit rate on anything after that timestamp.

## Summary

- **Prefix caching's deployment** is a flag plus a fleet-routing choice. The flag enables the cache; cache-aware routing is what makes the fleet-level benefit real.
- **Measure three axes**: cache-hit rate (per-request and per-block), TTFT delta between hit and miss, cache-pool occupancy. Publish all three.
- **Chat**: stable, block-aligned, head-of-prompt system prompt; sticky routing per conversation; TTL eviction for conversation branches.
- **RAG**: deterministic doc-ordering; block-aligned concatenation; stratified hit-rate per doc-id; version-in-prefix for invalidation; pin the hot set where supported.
- **The cold-cache cliff** is the one operational surprise. Mitigate with cache-aware routing's grace-period logic or with explicit pre-warming on scale-up.
- **Turn prefix caching off** when hit rate is sustainedly low or the cache is driving admission pressure.
- **The algorithm is kernel-side**; the deployment is yours — a defensible config, a measured hit-rate, a documented invalidation story, a per-tenant data boundary (chapter 06).

Chapter 04 extends prefix caching beyond HBM — into host DRAM and NVMe — with tiered KV stores (LMCache, Mooncake). The policy decisions get richer (which tier a prefix lives in, when it is promoted) and the cost signature becomes a thing the operator chooses rather than discovers.
