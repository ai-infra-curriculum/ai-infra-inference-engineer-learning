# Chapter 06 — Cross-Tenant KV Sharing Safely

> Prefix caching gives you a performance win. Prefix caching between tenants gives you a data-boundary problem. The two are the same mechanism; the operator decides which one to run.

## Motivation

Chapters 02–04 built up prefix caching as a performance feature: amortise prefill, retain KV past a request's lifetime, serve the next request faster. Chapter 03 noted — in one line — that there is a per-tenant isolation question. This chapter is that question.

The mechanism that lets tenant A's cached system prompt accelerate tenant B's request is exactly the mechanism that lets tenant A's cached *conversation turn* leak into tenant B's. On most workloads this is fine; on some it is a disclosure incident. The operator's job is to know which workload is which, to configure the engine accordingly, and to be able to defend the decision in a security review.

This is a multi-tenant serving problem, not a kernel problem. The algorithms are chapter 02's. The policy is this chapter's. The security-depth hand-off is to [`ai-infra-security-learning`](https://github.com/ai-infra-curriculum/ai-infra-security-learning).

<!-- needs-research: tenant-namespacing primitives in prefix caching vary by engine. vLLM has evolving support; SGLang and TRT-LLM have different models. Verify the exact flag / API your engine version offers before relying on it. -->

## The sharing opportunity

Multi-tenant inference fleets commonly see:

- **Shared system prompts across tenants on the same product.** A chat product's system prompt is the same for every tenant. The 2 000-token prefix is cacheable fleet-wide.
- **Shared retrieved documents across tenants in a shared RAG corpus.** All tenants querying the same encyclopedia hit the same doc prefixes.
- **Shared few-shot exemplars across templates.** A templated workflow has stable exemplars.

In each of these cases, cross-tenant sharing is a 2× to 10× memory-amortisation win. Turning it off is a non-trivial cost.

## The leakage risk

The sharing mechanism matches on *token sequences*. Any two requests whose first `k` tokens are byte-identical (after tokenisation) will share the first `k` tokens' KV. That is the whole mechanism. There is no built-in notion of "this cached prefix belongs to tenant A" unless the operator introduces one.

Three concrete leakage shapes:

### 1. Conversation-state cross-contamination

Chapter 03 recommended caching conversation history per session with TTL. If the conversation cache is *not* namespaced by tenant / session:

- Tenant A completes a turn; the engine caches the full history prefix (system prompt + A's full conversation).
- Tenant B sends a request whose prompt happens to byte-match A's cached prefix for the first `k` tokens.
- The engine reuses A's KV for those `k` tokens of B's prefill.

**Is this a leak?** The KV is not the plaintext; the attention output does not emit A's tokens. But two risks remain:

- The *existence* of a matching cached prefix is observable via timing (B's TTFT is much faster when a match exists → inference-side-channel attack).
- If the cached prefix extends into tenant-A-specific generated content, and that content happens to collide with tenant B's prompt tokens, B's prefill reads KV that was conditioned on A's content. The *generation* that follows may be biased toward A's distribution. Subtle, workload-dependent, hard to detect without targeted testing.

The timing-side-channel risk is the easier one to argue about; the generation-contamination risk is harder but real for sensitive workloads.

### 2. PII in a shared prefix

If a product renders user PII into the system prompt (e.g. "The user's name is Alice; her account tier is Gold; today is 2026-11-03"), and the system prompt is cached:

- The cache now holds Alice's PII in its KV.
- Any tenant whose prompt matches Alice's PII prefix hits the cache.
- Even without a direct leak, the engine's cache-hit metric exposes Alice's PII shape ("a prefix matching this pattern exists for the Alice tenant").

The fix is design-time: PII does not go into the shared system prompt. Chapter 03's "move dynamic fields to the tail" rule is a *performance* recommendation; it is also a *security* recommendation.

### 3. Prefix-probe disclosure

A malicious tenant can craft a prompt specifically to probe the cache: `"System: You are a helpful assistant. [guess-at-tenant-A-system-prompt-fragment]"`. By measuring the TTFT delta, the attacker learns which fragments are in cache — effectively a side-channel oracle on other tenants' system prompts.

This is the classic cache-timing attack rendered in the KV domain. Mitigations:

- Namespace by tenant (below). Removes the shared surface.
- Rate-limit or disable prefix-cache metrics in client-facing responses. (Though the TTFT itself is observable.)
- Normalise TTFT within a configured window to blunt the side-channel. Expensive.

For high-security workloads (healthcare, finance, legal), the mitigation is usually **turn off cross-tenant sharing entirely** — chapter 03's "turn it off" option, now driven by security rather than hit-rate.

## Namespacing — the primary mitigation

The right primitive is a **tenant namespace** that participates in the cache-match key. Instead of matching on `token_sequence`, match on `(tenant_id, token_sequence)`. Only requests from the same tenant can hit each other's cache.

### Where the namespace comes from

Three options, in order of robustness:

1. **Server-side from auth context.** The API gateway attaches `tenant_id` to the request; the engine passes it through to the cache lookup. The tenant cannot forge it. This is the right default.
2. **Client-provided cache key.** Some engines expose a client-settable cache key (verify your version). Useful for sub-tenant sharding (per-user within a tenant). Only use when the server has already validated the client's claim about who they are.
3. **Hashed into the prompt.** A per-tenant salt prepended to the prompt (`[TENANT_42] <system prompt>...`). The cheapest hack; works on engines without explicit namespace support. Costs one block of cache per tenant.

Option 1 is the production default. Option 3 is a fallback where the engine does not yet support option 1. Option 2 is a precision tool.

### What a namespace guarantees

- Tenants cannot read each other's cached KV directly. (Which was never possible — the KV is not plaintext.)
- Tenants cannot *hit* each other's cached prefixes. The side-channel oracle collapses.
- Tenants cannot *populate* each other's cached prefixes. No cross-tenant eviction pressure or cache poisoning.

### What a namespace costs

- **Cache de-amortisation.** If 10 tenants share the same system prompt, each pays for its own cached copy. Memory cost is `10 ×` what shared caching gave you.
- **Headroom loss.** The product-shared system prompt that was ~0.1% of pool with sharing becomes ~1% of pool with 10-way isolation. For a 1 000-tenant fleet with a product-shared system prompt, this is a budget hit.

Operators resolve this with a **selective sharing policy** — see below.

## Selective sharing — the realistic policy

Few production deployments are either "share everything" or "share nothing." The practical policy decides, per prefix, whether sharing is safe:

### Public prefixes (shared across tenants)

- Product-shared system prompts that are not customer-specific.
- Public-corpus RAG documents (encyclopedia, product documentation).
- Standard few-shot exemplars.

Treat these as **public cache entries** with no tenant namespace. The engine caches them once; all tenants hit them. The memory win is real; the security surface is bounded by "whatever this public prefix would have been visible to anyway."

### Private prefixes (per-tenant)

- Conversation state (per-session, per-tenant).
- Tenant-specific system prompts (customised per customer).
- Private RAG documents (per-tenant knowledge base).

Treat these with a **tenant namespace**. Only intra-tenant sharing applies.

### Private-and-redacted prefixes

- Prompts that *could* contain PII in some requests but not others.

Policy: either (a) treat as private-only (no cross-request sharing even within a tenant — expensive, rarely necessary), or (b) ensure the PII-containing portion is at the tail of the prompt where it cannot be block-aligned and cached. The latter is the standard pattern.

### The configuration pattern

Expose a per-request flag or metadata field:

```
cache_scope: "public" | "tenant" | "none"
tenant_id:   <string from auth, required if cache_scope == "tenant">
```

Default to `tenant` for safety; let the application code opt into `public` for the system prompt and known-shared prefixes.

## Opt-out — the data-boundary off switch

Some tenants will require that their traffic *never* participate in cross-request caching. Reasons:

- **Regulatory**: healthcare (HIPAA-adjacent), finance (fiduciary), legal (privilege).
- **Contractual**: the SaaS contract commits to data isolation at a level stronger than tenant namespacing.
- **Discovery / litigation**: a tenant under legal hold needs their traffic treated as if it never shared infrastructure.

The engine must support **per-tenant cache opt-out**:

- Requests from an opted-out tenant run with prefix caching disabled for their own traffic.
- Their requests do not populate the cache; their requests do not hit the cache.
- Their requests pay full prefill. TTFT regression is expected and must be documented in the opt-out tenant's contract.

Mechanism: a per-tenant policy flag in the API gateway. On opted-out requests, pass a flag to the engine that disables prefix-cache lookup and insert for this request. Supported directly in some engines; otherwise implemented by setting the request's cache-scope to `none`.

**Audit requirement**: the opt-out must be auditable. A tenant should be able to verify (through a separate attestation API, through logging access, or through contractual audit rights) that their traffic was in fact not cached.

## Auditing a shared cache

For a cache that *does* permit sharing, the audit discipline:

### Logs

- Per-request: what cache_scope was used, what matched (public, tenant, none), how many blocks hit.
- Per-cache-eviction: when a public cached prefix was evicted and why.
- Per-cache-insertion: what tenant inserted (for tenant-scoped entries) and what the token-count was. Do *not* log the prefix tokens themselves — those may be PII.

### Metrics

- **Per-scope hit rate**: `cache_hits_public`, `cache_hits_tenant`, both labelled by tenant where appropriate (watch cardinality).
- **Cross-tenant hit rate**: should be zero when tenant-scope is in use. If non-zero, that is a bug.
- **Opt-out tenant hit rate**: should be zero. If non-zero, the opt-out is broken.

### Chaos tests

Twice-yearly (or whatever cadence security requires):

1. Send a probe request as tenant A with a crafted "guess" at tenant B's prefix. Measure TTFT. If significantly lower than TTFT on a known-cold prefix, the namespace is leaking.
2. Send a probe request as an opted-out tenant. Confirm `cache_hits = 0` on the request's trace.
3. Flush the cache. Confirm the hit rate drops to zero and TTFT regresses as expected.

These are the "did we actually ship what we think we shipped" tests. Add them to the periodic security drill.

## The threat model as a one-page artefact

For a security review, prepare this document:

### Trust boundaries

- Who can submit requests? (Authenticated tenants; what trust do we extend to tenant-claimed metadata?)
- Who sees the KV cache? (The engine process; no out-of-engine access.)
- Who sees cache metrics? (Internal operators; is any cache metric exposed to tenants?)

### Shared surfaces

- The product's shared system prompt (public).
- The public RAG corpus (public).
- Everything else (per-tenant namespace).

### Attack surfaces

- Timing side-channel on cross-tenant cached prefixes (mitigated by namespacing).
- Prefix-probe disclosure (mitigated by namespacing + rate limits).
- Model-level leakage from KV conditioning on another tenant's content (mitigated by not sharing conversation-state across tenants; test with chaos probes).

### Opt-out contract

- Which tenants are opted out. How enforced. How audited.

### What is deferred to security peer

Deep adversarial analysis of the timing side-channel, formal information-flow analysis, and policy design for high-regulated verticals lives at [`ai-infra-security-learning`](https://github.com/ai-infra-curriculum/ai-infra-security-learning). This chapter covers the serving-side mechanics; the security peer covers the adversarial depth.

## When to opt out of sharing entirely

- **The product is single-tenant.** No cross-tenant risk; share freely within the tenant.
- **Regulated workload** (healthcare, finance, legal). Per-tenant opt-out, probably default-on.
- **High-adversarial-risk tenant** (public-facing product with incentive for attackers to probe other tenants). Default-opt-out until the threat model is reviewed.
- **New tenant onboarding with incomplete contract clarity.** Opt-out by default; opt in when the data-boundary conversation is complete.
- **An audit found cross-tenant hits in logs**. Full rollback; re-enable only after root-cause and namespace repair.

## Interaction with other module 104 features

- **Tiered KV stores (chapter 04)**: namespaces must propagate into the tiered store. A DRAM-tier cache entry inserted by tenant A must carry the namespace so an NVMe-tier lookup by tenant B does not match. Verify that LMCache / Mooncake / whatever store you use supports tenant tags.
- **Prefix caching deployment (chapter 03)**: the chapter-03 "always-share" default is appropriate for the public prefix; it is wrong for private prefixes. The default policy must be `tenant`, not `public`.
- **Long-context budgeting (chapter 05)**: tenant-namespacing reduces effective cache hit rate; your capacity planning should assume lower per-worker amortisation than a shared cache would produce.
- **Scheduler preemption (module 103, chapter 04)**: an opted-out tenant's requests still consume KV blocks; preemption policies apply equally. Opt-out does not grant priority.

## Common misreadings

- **"The KV is not plaintext, so there is no leakage."** Correct that KV ≠ plaintext. Incorrect that this closes the risk — timing side-channels and generation-contamination are both real.
- **"Namespacing kills the performance win."** It kills *cross-tenant* amortisation. Intra-tenant amortisation (within one tenant's chat sessions, within one tenant's RAG corpus) is often where most of the gain is. Measure before giving up.
- **"We can detect leakage by watching the cache."** Only if the audit metrics are in place. By default, most engines do not emit cross-tenant-hit metrics; you have to configure them.
- **"Opt-out is a one-line flag."** It is a flag plus logging, audit evidence, and (in some contracts) a periodic attestation. The flag is the easiest part.
- **"The security team will design the policy."** They design the *threat model*. You design the *mechanism and the operational enforcement*. The two conversations are necessary; neither suffices without the other.

## Summary

- Prefix caching's **data-boundary question** is the central multi-tenant serving concern. The mechanism that amortises prefill is the same one that creates the cross-tenant surface.
- Three leakage shapes to defend against: conversation-state cross-contamination, PII in a shared prefix, prefix-probe disclosure (timing side-channel).
- **Namespacing by tenant** is the primary mitigation. Default to tenant-scoped caching; opt into public sharing only for well-understood public prefixes (product system prompts, public RAG docs).
- **Selective sharing** (public vs. tenant vs. none) is the realistic policy — pure-share and pure-isolate are both wrong for most workloads.
- **Per-tenant opt-out** is the data-boundary off switch. Must be audit-able (logs + metrics + chaos probes).
- **Namespaces must propagate** into tiered stores and across the serving stack. One missing hop collapses the protection.
- **The threat model is a one-page document** for security review. The serving-side mechanics live here; adversarial-depth analysis lives at the security peer.

This chapter closes module 104. The module as a whole gives you the KV cache as a runtime object: the block pool (chapter 01), the shared index (chapter 02), the deployment (chapter 03), the tiered store (chapter 04), the budget (chapter 05), the sharing contract (chapter 06). Carry this stack into module 105 (quantisation, where KV precision becomes a choice), module 107 (distributed topologies, where KV crosses workers), and module 110 (observability, where these metrics become SLOs).
