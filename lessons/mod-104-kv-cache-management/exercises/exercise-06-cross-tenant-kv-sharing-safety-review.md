# Exercise 06 — Cross-Tenant KV-Sharing Safety Review

**Time:** 3 hours · **Output:** one threat-model doc, one per-tenant namespace design, one controlled "attempted leak" test, one opt-out demonstration, one audit cheatsheet

## Goal

Make [Chapter 06](../06-cross-tenant-kv-sharing-safely.md) concrete by **authoring a threat model** for a multi-tenant inference deployment, **designing a per-tenant namespace scheme**, **running a controlled attempted-leak test** that proves the namespace actually isolates, **demonstrating a working opt-out** for an "opted-out tenant," and **publishing an audit cheatsheet** another operator could execute to re-verify safety on demand.

## Problem statement

Your fleet serves two product lines: a public chat product and a regulated-vertical chat product (healthcare / finance / legal — pick one for the exercise). The public product can share cache across tenants aggressively. The regulated product must be isolated per tenant and must be able to prove, on demand, that cross-tenant sharing did not occur. The same engine serves both.

This exercise walks the serving-side mechanics of making that true and auditable. The deep adversarial analysis (full side-channel modeling, policy for the regulated vertical, legal sign-off) is handed off to the security peer; your scope is "operate the engine so the mechanics hold."

## Prerequisites

- A working engine with prefix caching on (from exercise 02 or 03).
- The ability to send requests tagged with tenant identity (via an auth header, URL path, or metadata field; use a dummy auth gateway if needed).
- (If your engine supports it) the per-request cache-key / tenant-namespace API. Check your engine's docs.

## Requirements

Deliver five artefacts.

### Artefact A — Threat model document (markdown, 1–2 pages)

Author a short threat model covering, for your deployment:

1. **Trust boundaries.** Who is authenticated, by what mechanism, with what level of assurance. Where does the `tenant_id` come from — gateway-stamped (trusted) or client-claimed (not trusted)?
2. **Shared surfaces.** Which prefixes are legitimately shared across tenants (product system prompt, public RAG docs). Which must not be shared.
3. **Attack surfaces.** Three concrete attacks: (a) timing side-channel on cross-tenant prefix, (b) PII leak via shared conversation prefix, (c) prefix-probe disclosure. For each, describe the attack and the mitigation your deployment uses.
4. **Residual risk.** Risks that namespacing does not close (e.g. aggregate-usage inference across tenants). Note explicitly which ones are deferred to the security peer.
5. **Audit obligations.** What the deployment commits to logging and what external audit evidence is required.

Written for a security reviewer — not an engineer — as the primary reader.

### Artefact B — Per-tenant namespace design

Design the mechanism:

1. **Where does `tenant_id` come from?** Specify the field (header, JWT claim, metadata); specify the gateway behaviour that stamps it.
2. **How does the engine use it?** Three options from Chapter 06:
   - Server-side cache-key (if engine supports a per-request namespace).
   - Prepended salt in the prompt (fallback where native support is absent).
   - Per-tenant cache namespace via sidecar / separate engine processes.

Pick one and specify the implementation path end-to-end. If you choose the salt-based fallback, specify the salt format (e.g. `[TENANT_42]`) and the length (block-aligned if possible, to avoid wasting a partial block).

3. **Which prefixes stay public?** The product system prompt. Public RAG docs. List them.

4. **Which prefixes become tenant-scoped?** Conversation state, tenant-specific system prompts, private RAG docs.

5. **What happens on a tenant-id mismatch?** If the gateway drops or malforms the tenant header, what does the engine do? (Defensive answer: treat as a new tenant with a unique id, which means a cache miss. Fail safe, not fail performant.)

Submit as a design document with sequence diagrams or short pseudocode for the gateway-to-engine flow.

### Artefact C — Controlled attempted-leak test

Run a controlled experiment to verify your namespace works. Setup:

1. Configure three tenants: `tenant_A`, `tenant_B`, `tenant_C`.
2. Instrument the engine to log per-request `(request_id, tenant_id, cache_hit_blocks)`.

**Experiment 1 — Within-tenant hits should work.**

- `tenant_A` sends 20 requests with the same `tenant_A`-specific 500-token prefix.
- Expected: request 1 misses; requests 2–20 cache-hit the prefix.
- Pass criteria: ≥ 90% hit rate on requests 2–20.

**Experiment 2 — Cross-tenant hits should not work.**

- `tenant_A` sends 20 requests with its specific prefix (populates the tenant-A cache).
- `tenant_B` sends 20 requests with a byte-identical 500-token prefix (i.e., same tokens as `tenant_A`, but under `tenant_B`'s identity).
- Expected: all 20 of `tenant_B`'s requests miss the cache (because namespace separates them). After `tenant_B`'s first miss, `tenant_B`-tagged requests 2–20 hit `tenant_B`'s cache.
- Pass criteria: zero hits on cross-tenant lookups. First `tenant_B` request misses even though the token prefix is identical to `tenant_A`'s.

**Experiment 3 — Public prefix should hit across tenants.**

- Confirm the shared public prefix (product system prompt) is cached once and hits all three tenants.
- Pass criteria: all three tenants hit the public cache after the first request has warmed it.

Document the raw results, the hit counts per experiment, and a short narrative confirming or refuting the pass criteria. If Experiment 2 shows unexpected hits, that is a **bug** — root-cause it before claiming the namespace works.

### Artefact D — Opt-out demonstration

Mark `tenant_C` as fully opted out of prefix caching.

Setup:

1. Mechanism: either a per-tenant policy flag that the gateway honours, or a separate engine instance for opted-out traffic. Document which you chose.
2. Verification instrumentation: for every `tenant_C` request, log `cache_lookup_called` (should be false) and `cache_insert_called` (should be false).

Experiment:

- `tenant_C` sends 20 requests with varied prefixes.
- Expected: zero cache lookups, zero cache inserts, every request pays full prefill.
- Pass criteria: `cache_lookup_called = false` for all 20; TTFT is consistent with full-prefill cost (not with a tier-0 hit).

Additionally, confirm `tenant_C`'s traffic does NOT accidentally populate the cache for other tenants:

- Immediately after `tenant_C`'s 20 requests, send one request from `tenant_A` with a prefix that matches `tenant_C`'s. If it hits, the opt-out is leaking into the shared cache.
- Pass criteria: `tenant_A`'s probe misses.

### Artefact E — Audit cheatsheet (one page)

A single-page document another operator could run on demand (quarterly, after a config change, in response to a tenant request) to verify safety. Minimum contents:

1. **Pre-check**: which flags should be set, which metrics should be scraping, where to find the current tenant namespace policy.
2. **Within-tenant hit test**: a short script or set of curl commands equivalent to Experiment 1. Expected result.
3. **Cross-tenant isolation test**: Experiment 2 equivalent. Expected result.
4. **Opt-out verification test**: Experiment 3 equivalent for opted-out tenant.
5. **Cross-tenant hit metric check**: if the engine exposes a cross-tenant-hit metric, how to scrape and interpret. If not, how to compute from logs.
6. **Red flags**: a list of signatures that indicate a leak (unexpected hits, missing namespace stamps on logged requests, opt-out hits).
7. **Escalation path**: who to page if any check fails.

## Starter guidance

- **If your engine does not expose a native per-request namespace**, the salt-based fallback is practical. Prepend a short deterministic salt (`[TENANT_${id}]\n`) to the prompt before tokenisation. The salt occupies ~5–15 tokens; pad to a block boundary if you care about not fragmenting the first block.
- **Instrumentation can be local**. Add print statements or counters in the cache-lookup function; don't try to formalise this as a production metric yet.
- **The three tenants can be synthetic**. You don't need real auth; use a header `X-Tenant-Id: A|B|C` and have a thin middleware that honours it.
- **For the opt-out mechanism**, the gateway-mediated path is cleaner than engine-side policy: the gateway reads the tenant's policy, and for opted-out tenants, it adds a `?no_cache=1` query param (or equivalent) that the engine honours by setting `cache_scope=none` on the request.
- **If you can't make the opt-out work in-engine**, the fallback is a dedicated engine instance for opted-out traffic. Route by gateway. Document the operational cost (double the capacity for the regulated vertical).
- **Experiment 2 is the one that matters**. If your namespace scheme fails here, nothing else in the exercise matters. Fix before moving on.

## Fallbacks for constrained hardware

The experiments are low-throughput by design. Any GPU, or CPU-only, is sufficient. Hit-rate verification does not need production hardware.

## Acceptance criteria

1. **Artefact A** is readable by a non-engineer security reviewer. Covers trust boundaries, shared surfaces, three attacks, residual risk, and audit obligations.
2. **Artefact B** has a specified `tenant_id` provenance, a specified namespace mechanism, and a specified failure behaviour on malformed tenant metadata.
3. **Artefact C**'s three experiments all pass. Experiment 2's cross-tenant isolation is proven (zero hits across tenants on identical prefixes).
4. **Artefact D** demonstrates `tenant_C`'s requests do not populate or consult the cache; a subsequent cross-tenant probe does not hit.
5. **Artefact E** is executable — another operator, following only the cheatsheet, could run the three tests in under 30 minutes.
6. All five artefacts live in version control.

## Common failure modes

- **Namespace in the wrong layer.** If you stamp `tenant_id` after tokenisation but prefix caching matches on token sequences, your namespace never participates in the match. Confirm the stamp reaches the engine's cache-key function.
- **Public prefix accidentally scoped per-tenant.** Over-isolation. The product system prompt should remain cross-tenant-hit; if your namespace scheme accidentally scoped it, you lose the amortisation win.
- **Opt-out leaks via co-located engine state.** If `tenant_C` and other tenants share the same engine, make sure `tenant_C`'s requests don't trigger insert calls that pollute the shared cache. Verify via log instrumentation.
- **Experiment 2 produces hits.** Namespace is broken. Possible causes: salt not actually tokenising to distinct ids (unlikely for a `[TENANT_42]` style salt), namespace stripped by middleware, engine's cache-match function ignoring the namespace. Trace back from the hit metric.
- **Threat model missing residual risks.** The timing side-channel is not fully closed by namespacing (an attacker in the same tenant can still probe). Note this; defer deeper analysis to the security peer.

## Keep it

Artefact A (threat model) is a living document — refresh on every onboarded tenant class, every engine upgrade, every cache-policy change. Artefact E (audit cheatsheet) is the runbook you execute quarterly and in response to any tenant-raised concern. Both pin to the team wiki.
