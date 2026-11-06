# Chapter 06 — Boundary to `llm-application-developer` — Server-Side vs. Client-Side Routing

> "LiteLLM" is on both sides of the fence. "OpenAI SDK" is on both sides of the fence. The question is not which library is on which side — the question is which responsibilities. This chapter writes the fence.

## Motivation

The AI Engineering family's [`llm-application-developer`](https://github.com/ai-engineering-curriculum/llm-application-developer-learning) track (consumer peer, level 25) teaches application teams to call model APIs. They cover the OpenAI SDK, LiteLLM-as-a-library, Anthropic / Google / Mistral SDKs, retry-and-timeout patterns inside an application process, token-budget tracking from the product's perspective, and the UX decisions that attach to model behaviour.

This module teaches the serving platform on the other side of the API. Chapter 03 built the cost-aware router that *is* the model API from the application's perspective. There are two live confusions this chapter addresses:

1. **The same library appears on both sides.** An application team can `pip install litellm` and call `litellm.completion(...)` from their code; the platform team can also deploy `litellm --config ...` as a proxy service in front of the fleet. Same project, two very different responsibilities. The two teams have to know they are not doing the same job.
2. **Responsibilities drift across the boundary.** When a model fails over, who retries? When a tenant's quota is near, who shows the UX degradation? When a response arrives degraded, who decides to accept vs. retry? Each of these has a correct side; most platforms get at least one of them wrong first time.

By the end of this chapter you can:

- State what the server-side router owns and what the application-side SDK owns, by responsibility.
- Trace a request through the two sides and name which concerns live where.
- Decide, for a new feature, which side it belongs on.
- Negotiate the interface — what headers the server publishes downward, what metadata the client sends upward, what guarantees each side makes.
- Avoid the common anti-patterns (app-level retry around a server that already retries; server-level product UX decisions; mixed accounting that double-counts cost).

The peer's exact scope is owned by the AI Engineering family; see the "Out of scope" clause and the mod-106 boundary line in [`CURRICULUM.md`](../../CURRICULUM.md) for the authoritative handoff statement. Treat the responsibilities enumerated here as structural, not chapter-pinned.

## The two sides

| Concern | Server-side router (this module) | Application-side SDK (`llm-application-developer`) |
|---|---|---|
| **Model selection** | Resolve alias → physical model; apply tenant allow-list | Pick an alias based on product intent |
| **Backend selection** | Pick a backend; load-balance across self-host / providers | — (never sees the backend) |
| **Fallback (platform)** | On backend failure, try the next backend | — |
| **Fallback (product)** | — | On degraded response, show a UX ("please try again") or retry with a different alias |
| **Retry** | Transient-5xx retry inside a request's envelope | Application-level retry at the user action level (e.g. the user retries) |
| **Timeout** | Per-backend timeout; total admission-to-response timeout | Per-user-action timeout at the UI |
| **Rate-limit** | Per-tenant quota enforced; respond 429 with retry-after | Respect retry-after; back off; show quota exhausted to user |
| **Cost accounting** | Server is the system of record for request-level cost | Application tracks product-level cost (per user, per feature) |
| **Observability** | Per-tenant / per-model / per-backend metrics; audit log | Per-user / per-feature product metrics |
| **Streaming** | Pipe SSE from backend to client; parse `usage` from final chunk for accounting | Consume SSE in the application; handle disconnects |
| **Authentication** | Validate tenant API key; map to tenant ID; apply tenant policy | Store and manage the key; rotate; keep out of client-side code |
| **Model-swap awareness** | Transparent to application (module 105 chapter 06 handles) | — |
| **Tenant quota awareness** | Enforce quota | May read quota via a `/v1/quota` endpoint to pre-empt obvious overshoots |
| **Multi-tenancy** | Isolation contract (chapter 05) | — (one tenant per application typically) |
| **PII redaction at log** | Platform-side redaction at write-time | Application may add its own redaction; does not rely on platform's |

The boundary is sharper than the table suggests on a few concerns worth calling out.

## Where LiteLLM shows up on both sides

### Server-side — LiteLLM as a proxy

The platform team can run LiteLLM as a proxy server:

```
litellm --config config.yaml --port 8000
```

Config shape (chapter 03 reference):

```yaml
model_list:
  - model_name: chat-small
    litellm_params:
      model: openai/llama-3.1-8b-instruct-fp8
      api_base: http://vllm-small.svc.cluster.local:8000/v1
```

This is a **server** that implements the chapter-03 router's shape — provider fallback, rate-limiting, cost tracking, OpenAI-compatible endpoint upward.

### Application-side — LiteLLM as a library

The application team can import LiteLLM in their code:

```python
from litellm import completion

response = completion(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "..."}],
    fallbacks=["claude-3-5-sonnet", "gpt-4o"],
)
```

This is a **client-side** convenience: standard interface across providers, in-process fallback, in-process retry. If the application is talking directly to vendor APIs (not through a platform router), the library handles the vendor shape differences.

### The confusion

A platform that runs the LiteLLM proxy *and* has an application team using the LiteLLM library will often see:

- The application does `fallbacks=["chat-small", "chat-large"]` inside its code, which is **duplicating** what the server-side router does.
- Both sides implement retry; a transient failure causes 2 server-side retries × 3 application-side retries = 6 attempts to the backend before the user sees an error.
- Both sides track cost; billing double-counts or disagrees.

The fix is to decide, per platform:

- **Full server-side.** The application uses LiteLLM-the-library (or the OpenAI SDK) as a dumb client; all fallback, retry, routing lives on the server. Set the client's retry to 0 or 1; set the client's fallbacks to empty; let the server decide.
- **Full client-side.** The platform exposes raw backend endpoints with API keys; the application picks and routes. Server-side router is thin or absent.
- **Hybrid, with explicit split.** The server handles *platform* fallback (self-host → provider); the client handles *product* fallback (`chat-small` cheap first → `chat-large` if the UX requires richer response). The two sides' fallback lists are disjoint.

**The hybrid is the common healthy state.** Both sides have fallbacks, but they are different fallbacks. The server does not know about the application's product fallback; the application does not know about the server's platform fallback. The two compose cleanly as long as the responsibilities are named.

## Request / response contract

The interface between the two sides is the OpenAI-compatible API plus headers. The server publishes **downward**:

```
HTTP/1.1 200 OK
Content-Type: text/event-stream       (or application/json for non-streaming)
x-request-id: req_abc123
x-model-alias: chat-small
x-physical-model: llama-3.1-8b-instruct-fp8
x-backend: self-host-us-east
x-degraded-from: null                  (or e.g. "chat-large" if the server fell back)
x-tenant: customer-7f3a
x-input-tokens: 128
x-output-tokens: 342
x-cost-usd: 0.000146
x-quota-remaining-tokens: 19_000_000
x-rate-limit-remaining-rps: 42
```

The client reads these to:

- Log the physical model + backend for debugging an outcome. (Keep model alias stable in the product; log physical model for ops.)
- Detect degradation and decide whether to accept or retry.
- Observe quota drift and pre-empt rate-limit (optional).

The client sends **upward**:

```
POST /v1/chat/completions
Authorization: Bearer sk-tenant-...
x-user-id: user_123                   (for application-side observability — the server ignores or forwards)
x-request-source: web|mobile|batch    (for backend selection hints — optional)
x-max-cost-usd: 0.01                  (per-request budget cap)
x-tags: feature=summariser,surface=web
```

The server uses the headers for routing hints and observability; the body stays OpenAI-compatible.

### Pricing and quota semantics

The server computes cost per request from a pricing table. The pricing table is **server-owned**: physical models change (chapter 03's alias indirection); provider prices change; the alias stays stable. The application should **not** compute cost from the alias — it should read `x-cost-usd` from the response.

The quota is also server-owned. The application reads `x-quota-remaining-tokens` for informational purposes only — the authoritative state is the server's.

### Streaming nuances

For SSE responses, the `usage` object arrives in the final chunk (OpenAI-compatible). The server:

- Pipes the stream to the client as it arrives.
- Parses the final chunk's `usage` to compute cost and update accounting.
- Emits trailer-style headers if the HTTP layer supports them, or includes `x-input-tokens` / `x-output-tokens` in a terminal SSE event the client can read.

The application:

- Consumes the stream; renders tokens as they arrive.
- Catches disconnects (user navigates away); sends an explicit cancel if the API supports it (OpenAI-compatible API supports `AbortController` in most SDKs).

The server's cost accounting must still complete even if the client disconnects — the generation ran, the fleet consumed capacity, cost accrued. **Server-side cost accounting is independent of client-side stream termination.**

## Deciding where a feature belongs

A test for which side owns a new feature:

1. **Does the feature depend on tenant-level state?** (Quota, data boundary, isolation class.) → **Server.**
2. **Does the feature depend on the physical fleet shape?** (Backend selection, provider fallback, scheduler priority, adapter residency.) → **Server.**
3. **Does the feature depend on user-level or product-level state?** (Per-user quota, UX degradation, product-side retry on user frustration signal.) → **Application.**
4. **Would a different application team want the same feature with different values?** → **Server configuration surface** (knobs the server exposes to configure; not hard-coded in server logic).
5. **Would the application want to override the server's choice for a specific request?** → **Request-level header the server respects** (e.g. `x-prefer-backend` as an advisory hint).

Examples:

- "Cheap-model-first with escalation on confidence drop" — server (platform-level cost policy). Expose a flag on the request for the application to disable (`x-no-escalation: 1`) if they already know they want expensive.
- "Retry on 429 with exponential backoff" — server for transient capacity; application for product-level retry (the user's "retry" button).
- "If the response contains PII, scrub it before showing" — application (product-level requirement). The server redacts logs; the application controls what the user sees.
- "Cache responses for identical prompts" — application if the application defines "identical" (prompt dedup inside the app). Server for prefix sharing on identical prefixes (module 104).
- "Audit log of all requests from a specific user" — application (per-user audit); the server emits per-tenant records; the application correlates by `x-user-id`.
- "Daily cost cap per user" — application. The server's cap is per tenant; the application subdivides into per-user caps.

## Common anti-patterns

### Anti-pattern 1 — double retry

Application retries on 5xx; server retries on 5xx; backend sees 2 × 3 = 6 attempts for one user request. Observable as: backend request count = N × client request count, where N is the product of retry depths. Fix: pick one side to retry; set the other side's retry to 0.

### Anti-pattern 2 — application-side provider fallback bypassing server policy

Application reads an environment variable with provider URLs and falls through them. Server-side policy (data boundary, tenant allow-list, cost tracking) is bypassed. Fix: application must go through the server for all requests; the server is the only path to backends. If the application needs provider failover, the server implements it.

### Anti-pattern 3 — server-side product UX decisions

The server decides "this response has low confidence; return an error so the application shows a 'try again' button." But the application wanted to show the user the response anyway. The server is making a product decision. Fix: server reports the confidence signal in a header; application decides the UX.

### Anti-pattern 4 — pricing math duplicated and drifting

Application has a cost estimator computed from `input_tokens × price` where `price` is hard-coded. Server changes a provider's price; application's estimate drifts. Fix: application reads cost from `x-cost-usd`.

### Anti-pattern 5 — server returns raw provider errors

A Fireworks 500 response is passed through verbatim as a Fireworks-shaped error. The application has to understand Fireworks' error shape (and OpenAI's, and Together's, and Anthropic's). Fix: server translates all backend errors to a stable server-shape before returning. The application sees one error shape.

### Anti-pattern 6 — application-level per-request-cost cap drift

Application sends `x-max-cost-usd: 0.01`; server respects it; but when the server falls back to an expensive provider, the fallback crosses the cap. The server returned a response that violates the cap, or did it round up / round down? Document the semantics: server **must not** execute a request where estimated cost already exceeds the cap; if a fallback would cross the cap, server returns the error from the previous fallback instead.

## The model-registry handshake (lightly)

The server-side router has a model registry (chapter 03). The application has a view of "what aliases are available." The handshake is a `/v1/models` endpoint the server publishes; the application queries it to populate selectors or validate alias names before sending a request.

```
GET /v1/models
→ {
    "data": [
      {"id": "chat-small", "object": "model", "tier": "A",
       "max_context_tokens": 8192, "deprecated": false},
      {"id": "chat-large", "object": "model", "tier": "A",
       "max_context_tokens": 32768, "deprecated": false},
      {"id": "legacy-chat", "object": "model", "tier": "B",
       "deprecated": true, "deprecation_sunset": "2026-04-01"}
    ]
  }
```

The application can gate its code on `deprecated: true` and begin migration before the sunset. The server publishes the sunset date through this endpoint; the application reads it.

## Composing with other chapters

- **Chapter 01 (multi-LoRA):** the application selects an adapter via the `model` field (alias) or an `adapter_id` field (depending on the server's convention). Which style the server supports is a chapter-01 config choice; the application code is the same shape either way.
- **Chapter 02 (co-location):** transparent to the application. The server routes to whichever process holds the model; the application never sees MIG vs. engine-ensemble.
- **Chapter 03 (router):** the primary interface this chapter negotiates.
- **Chapter 04 (adapter loader):** transparent to the application. If adapter cold-start would blow the latency SLO, the server may route to a provider that has it warm; the application does not need to know.
- **Chapter 05 (isolation):** tenant policy is on the server. The application maps users to tenant via its own user → tenant model; the server doesn't need to know about users.

## Summary

- The server-side router (this module) owns platform-level concerns: tenant policy, backend selection, platform fallback, cost accounting, data boundary.
- The application-side SDK (`llm-application-developer`) owns product-level concerns: user-level state, product UX, per-user retry, product-level feature gating.
- LiteLLM, OpenAI SDK, and others appear on both sides — same project, different responsibilities.
- The interface is the OpenAI-compatible endpoint plus explicit headers in both directions (server publishes model / backend / cost / quota downward; application sends budget / tags upward).
- Decide feature ownership by asking: does it depend on tenant state (server) or user state (application)? Does it depend on fleet shape (server) or product UX (application)?
- Common anti-patterns are double retry, application-side provider fallback bypass, server-side product decisions, duplicated pricing math, raw-provider error pass-through, and ambiguous cost-cap semantics on fallback.
- The model-registry handshake (`/v1/models` with deprecation metadata) is how the server and application coordinate model-availability changes.
