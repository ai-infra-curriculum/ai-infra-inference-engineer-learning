# Chapter 06 — Request-Shape Discipline: Retry, Idempotency, Rate-Limit, Auth, Tenant Isolation

> The engine is a shared resource. The ingress is where the request becomes real — identified, authenticated, rate-limited, correlated to a trace, routed to a tenant's slice. A request that reaches the engine without a tenant tag is a request without a quota. A streamed response retried on `503` without an idempotency key is a double charge. A `429` without `Retry-After` is a client that hammers you. This chapter writes the ingress contract.

## Motivation

Three recurring inference incidents trace back to ingress discipline, not engine behaviour:

- **The retry storm.** A transient `502` fires during a traffic spike. Every client retries. The retries land on the same unhealthy pod. The retry storm doubles the inbound traffic. The fleet's admission control (module 103) is working correctly, but admission-reject rate is 90 % and the TTFT SLO is breaching across every tenant.
- **The double charge.** A streaming response times out at 59 seconds (the proxy's idle timeout). The client retries. Both requests complete; both bill tokens. The client's audit log shows two charges for one logical call. The product team opens a critical bug.
- **The noisy tenant.** A new tenant onboards and submits a 500-request burst for a benchmark. The fleet's queue depth climbs; TTFT regresses for every tenant. The fleet does not have a per-tenant quota. The on-call engineer manually blocks the IP and files an incident.

Each is an ingress discipline failure. Each is one middleware away from correct behaviour. This chapter writes the middleware.

By the end of this chapter you can:

- Implement `Idempotency-Key` handling for safe retry on streaming responses.
- Author retry semantics for the server (`Retry-After`, `503` vs. `429`, graceful degradation).
- Emit standards-track rate-limit headers (`RateLimit`, `RateLimit-Policy`) and the legacy `X-RateLimit-*` set.
- Propagate W3C `traceparent` plus vendor `x-request-id` across the ingress and into the engine.
- Author the auth model (API key, OAuth 2.1 bearer, short-lived STS) at the ingress.
- Enforce tenant isolation before the engine consumes KV budget.

## The ingress responsibilities, in order

The ingress sits between the client's HTTP connection and the engine's request admission. Its responsibilities, in the order they execute:

1. **TLS termination** — handled by the LB / reverse proxy; the ingress sees plain HTTP internally.
2. **Request identification** — assign or propagate `x-request-id`; parse `traceparent`.
3. **Authentication** — parse the `Authorization` header; map to internal identity (tenant, principal, scopes).
4. **Idempotency** — if the request carries `Idempotency-Key`, de-duplicate against prior calls.
5. **Rate limiting** — enforce per-identity quotas; set rate-limit headers; reject with `429` if exceeded.
6. **Tenant isolation** — enforce the tenant's quota of KV / concurrency / tokens-per-minute; back-pressure if nearing the limit.
7. **Request shape validation** — JSON Schema check against the endpoint; reject with `400` on malformed.
8. **Admission control hand-off** — forward to the engine with the trace context and tenant tag; propagate the engine's back-pressure.
9. **Response-side discipline** — stream through, propagate response headers, cancel the engine on client disconnect.
10. **Audit logging** — record identity, request shape, response summary, latency, outcome.

Each step is a middleware. The order is enforced — tenant isolation after auth, rate limit after auth, admission hand-off after rate limit. Chapter 02 covered steps 9 and 10's streaming behaviour; this chapter covers 1-8.

## Request identification — `x-request-id` and `traceparent`

Every request gets an ID. The ID appears in every log line, every metric sample's exemplar, every error response's body, and every downstream engine call. The vendor convention is `x-request-id`; the W3C-standard trace propagation is `traceparent` (defined in [W3C Trace Context](https://www.w3.org/TR/trace-context/)).

Discipline:

- **Accept `x-request-id` if the client provides one.** Validate (UUID-ish, length-bounded). Generate one if absent.
- **Accept `traceparent` if the client provides one.** Validate per the W3C format (`00-<trace-id>-<parent-span-id>-<flags>`). Generate a fresh trace if absent.
- **Emit the IDs in response headers.** The response carries `x-request-id` and `traceresponse` (W3C's response-side header). A client's error-report screenshot now contains the ID the on-call engineer can grep.
- **Pass both to the engine.** The engine's own spans (chapter 03 of module 110) carry the same trace ID; the trace tree is contiguous.

A sketch of the middleware:

```python
import uuid
from fastapi import FastAPI, Request
from opentelemetry import trace

app = FastAPI()

@app.middleware("http")
async def identify(request: Request, call_next):
    rid = request.headers.get("x-request-id") or f"req_{uuid.uuid4().hex[:16]}"
    tp = request.headers.get("traceparent")
    # Validate; attach to request state; propagate into logs and spans.
    request.state.request_id = rid
    request.state.traceparent = tp
    with trace.get_tracer(__name__).start_as_current_span("ingress") as span:
        span.set_attribute("request.id", rid)
        response = await call_next(request)
    response.headers["x-request-id"] = rid
    return response
```

The spec is tight; follow it. The engineering payoff is enormous — every incident triage starts with grep by request ID.

## Authentication — three models

Three auth models the ingress may serve:

### API key

A long-lived secret the client presents as `Authorization: Bearer sk-<key>`. The ingress looks the key up in a per-tenant table (stored in a database, cached in memory with a TTL), extracts tenant / scopes / rate limits, and attaches to the request state.

```python
async def authenticate_api_key(request: Request) -> Identity:
    header = request.headers.get("Authorization", "")
    if not header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    key = header[len("Bearer "):]
    tenant = await lookup_tenant(key)  # cached DB lookup
    if tenant is None:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return Identity(
        tenant_id=tenant.id,
        subject=tenant.account,
        scopes=tenant.scopes,
        rate_limits=tenant.rate_limits,
    )
```

Discipline:

- **Hash before storage.** Store a hash of the key in the database; compare the hash on lookup. A database dump that leaks the hash is not a database dump that leaks the key.
- **Cache with a TTL, not forever.** A revoked key needs to stop working within the TTL (seconds to minutes).
- **Rotate.** Make rotation easy (two active keys per account at a time); the application-layer peer handles the rotation flow.

### OAuth 2.1 bearer

A short-lived JWT the client obtains from an authorization server. The ingress validates the JWT (signature, expiration, audience, issuer, scope) and extracts identity. Covered in chapter 05; the pattern is identical for the OpenAI-compatible endpoint when it accepts OAuth tokens.

### Short-lived STS tokens

For intra-fleet calls (one service calling the inference endpoint on behalf of a user), a short-lived STS token (minutes-long credential issued by a trusted authentication service) is the right pattern. The flow: service A obtains a token with the end user's identity attestation; the inference ingress validates and extracts both the service identity and the user identity; rate-limit and audit by user, authorise by service.

Three disciplines across all three models:

- **The ingress is the auth boundary.** The engine never sees the token; it receives an internal identity header the ingress populates after validation. If the engine is reachable from outside the ingress's network namespace, that is a security bug.
- **Fail closed.** No identity → `401`. The ingress does not fall back to an anonymous identity for inference endpoints.
- **Log auth failures distinctly.** `401` and `403` are separate metric families from `5xx`; a brute-force attack lights up the auth-failure counter without affecting the service's health signals.

## Idempotency keys

The pattern (per the [IETF Idempotency-Key header draft](https://datatracker.ietf.org/doc/draft-ietf-httpapi-idempotency-key-header/) and Stripe's long-standing implementation): a client that wants to retry safely attaches a header `Idempotency-Key: <uuid>` to the request. The server, on first receipt, atomically records the key against the (tenant, endpoint) pair. On retry with the same key, the server returns the cached response of the first call rather than running the request again.

The semantics for streaming responses are subtle:

- **Non-streaming requests are easy.** First call: execute, record `(key → response)`, return response. Retry: look up key, return cached response.
- **Streaming responses have a race.** First call starts streaming; second call arrives before the stream closes. Options: (a) replay from the start if the first stream completed; (b) return `409 Conflict` if the first stream is in-flight; (c) attach the second call to the same stream (hard — requires buffering).

The production pattern is (b):

```python
async def handle_idempotent(request: Request, body: dict) -> Response:
    key = request.headers.get("Idempotency-Key")
    if not key:
        return await handle_normal(request, body)
    # Atomic register-or-get.
    record = await idempotency_store.register_or_get(
        tenant_id=request.state.identity.tenant_id,
        endpoint=request.url.path,
        key=key,
        body_hash=sha256(body),
    )
    if record.state == "new":
        # First call for this key — execute, update record on completion.
        response = await handle_normal(request, body)
        await idempotency_store.complete(record.id, response)
        return response
    if record.state == "in_flight":
        return Response(status_code=409, content=json.dumps({
            "error": {"message": "Idempotency key in flight", "code": "idempotency_conflict"}
        }))
    if record.state == "complete":
        if record.body_hash != sha256(body):
            return Response(status_code=422, content=json.dumps({
                "error": {"message": "Idempotency key reused with different body", "code": "idempotency_mismatch"}
            }))
        return record.response
```

Four correctness properties:

- **The key is scoped to (tenant, endpoint, key).** Two tenants with the same key don't collide. The same key on `/v1/chat/completions` and `/v1/embeddings` are separate; usually what the caller wants.
- **The body hash is checked.** A client that reuses a key with a different body is confused; return `422` rather than silently returning the wrong cached response.
- **The record has a TTL.** 24 hours is the Stripe convention and the IETF draft's recommendation; verify current. After TTL expiry, the key is reusable.
- **The in-flight state is tracked.** A key whose first call hasn't completed is in-flight; a retry gets `409`. The client should wait and poll, not immediately re-retry.

Idempotency applies to writes. Chat completions are read-ish from the engine's perspective (the engine's state doesn't change), but every call bills tokens; the client treats them as writes. The ingress should implement idempotency for all inference endpoints.

## Retry semantics on the server side

The server does not retry its own responses (that is the client's job). The server's responsibility is to emit retry-compatible status codes and headers so the client knows how to retry.

Status codes and semantics:

| Status | Meaning | Retry safe without `Idempotency-Key`? | `Retry-After` required? |
|---|---|---|---|
| `429 Too Many Requests` | Rate limit exceeded. | No — if the request executed before rate-limit check (shouldn't, but double-check) | Yes |
| `503 Service Unavailable` | Transient server-side issue. Server may be degraded or admission-rejecting. | No — the request may have partially executed | Yes |
| `502 Bad Gateway` | Upstream engine unreachable. | No — same reason | Optional |
| `504 Gateway Timeout` | Engine took too long. | **No** — the engine may still be running the request | Optional |
| `500 Internal Server Error` | Unspecified. | No. Treat as a bug; don't auto-retry. | — |

Three disciplines:

- **Emit `Retry-After` on `429` and `503`.** `Retry-After: 5` (seconds) or `Retry-After: Fri, 03 Nov 2026 12:00:00 GMT` (HTTP-date). The client's SDK respects this. Without it, clients implement exponential backoff that may be far from optimal.
- **`504` is dangerous to retry.** The engine may still be running the request. A naive retry runs it again, doubling cost. Clients should either use idempotency keys or not retry `504`. Document this in the API docs.
- **Streaming responses that fail mid-stream.** The HTTP status is already `200`. The failure is signalled by a late event in the stream (`data: {"error":{...}}\n\n`) and then connection close. The client retries at the application level — the request isn't idempotent-key-safe to retry (part of it executed); the client either accepts partial output or sends a new request.

## Rate limit headers

Standards-track: [IETF RateLimit header fields draft](https://datatracker.ietf.org/doc/draft-ietf-httpapi-ratelimit-headers/). Two headers:

- `RateLimit: limit=100, remaining=42, reset=30` — the client sees it has 42 units left in the current window, resetting in 30 seconds.
- `RateLimit-Policy: 100;w=60` — the policy is 100 units per 60-second window.

Legacy / vendor set (dominant in production as of writing):

- `X-RateLimit-Limit: 100`
- `X-RateLimit-Remaining: 42`
- `X-RateLimit-Reset: 1762128030` (Unix epoch or seconds-until)

Discipline:

- **Emit both the standards-track and the vendor set during the transition.** Many clients only know the vendor set; some newer clients prefer the standards set.
- **Rate limit by work, not by request.** A chat completion that generates 2000 tokens costs more than one that generates 50; a per-request limit misses this. Many providers use multiple limits in parallel: requests per minute, tokens per minute, tokens per day. Emit one header set per limit (with distinct `policy` strings).
- **Set `Retry-After` on `429`.** The IETF draft recommends it; clients rely on it; the two systems compose.

A worked response under rate limit:

```http
HTTP/1.1 429 Too Many Requests
Content-Type: application/json
RateLimit: limit=100, remaining=0, reset=30
RateLimit-Policy: 100;w=60
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 30
Retry-After: 30
x-request-id: req_abc123

{"error": {"message": "Rate limit exceeded", "type": "rate_limit_error", "code": "requests_per_minute"}}
```

## Tenant isolation

The engine's KV budget is shared across tenants. Without isolation, one noisy tenant degrades every other tenant's TTFT. Isolation mechanisms:

- **Per-tenant admission quota.** The ingress tracks each tenant's in-flight request count. When a tenant exceeds its quota, further requests are rejected with `429` before reaching the engine. The quota is a function of tenant tier (free: 1 concurrent, paid: 10, enterprise: 100+).
- **Per-tenant token-per-minute quota.** A tenant that submits 10 requests of 10k output tokens each is as much pressure as 100 requests of 1k each. The quota is `tokens_per_minute` with an estimated-tokens charge at admission (based on `max_tokens` or a per-model default) and a reconciliation on completion.
- **Per-tenant model allowlist.** Tenants may be restricted to specific models (not every tenant can run the 70B flagship). Enforce at admission; reject with `403` if the model is not in the tenant's allowlist.
- **Priority classes.** Within the engine's queue, requests from a premium tenant get earlier scheduling (chapter on priorities in module 103). The ingress tags; the engine schedules.

Three disciplines:

- **Enforce quotas before the engine sees the request.** A rejected request should not consume KV, should not count against admission-reject (which signals real admission pressure), should not surface in the engine's traces as a request. Reject with `429` at the ingress.
- **Isolate the quota counters per tenant in shared storage.** Redis or a similar in-memory cache; the counters are read-modify-write hundreds of times per second. Use atomic ops (`INCRBY` + `EXPIRE`).
- **Log every quota rejection with the tenant tag and the quota that was hit.** The dashboard shows which tenant breached which quota; the on-call engineer knows whether to grow the quota, reach out, or escalate.

## Request shape validation

Before admission, validate the request body against the endpoint's schema:

- Required fields present (`model`, `messages` for chat).
- Enum values valid (`role: "system" | "user" | "assistant" | "tool"`).
- Numeric bounds respected (`temperature >= 0 && <= 2`, `max_tokens >= 1`).
- Tool-call shapes consistent (every `tool_message` has a `tool_call_id`; the IDs correlate).
- Context length under the model's limit (compute a cheap token estimate; `tiktoken`-shape tokenizer).

Reject with `400 Bad Request` and a specific error:

```json
{"error": {
  "message": "max_tokens must be a positive integer",
  "type": "invalid_request_error",
  "param": "max_tokens",
  "code": "invalid_type"
}}
```

Three disciplines:

- **Validate before auth-check.** A malformed request from an unauthenticated client gets `400`, not `401`. Mild privacy win and clear semantics for debugging.
  - Alternative: validate after auth, so unauthenticated probes don't learn the schema. Both patterns are deployed; the public-facing OpenAI reference pattern is validate after auth. Pick one, document.
- **Validate once.** The engine will validate again (it has to — defensive); don't rely on the engine, but don't double-validate the fields the engine will.
- **Keep error messages actionable.** `"max_tokens must be a positive integer"` beats `"invalid request"`.

## Admission hand-off and back-pressure

With auth, idempotency, rate-limit, tenant-isolation, and shape-validation passed, the request is admitted to the engine. The engine has its own admission control (module 103):

- Queue depth under cap → immediate admission.
- Queue depth at cap → either (a) the engine rejects immediately with `503`, or (b) the ingress holds the request briefly and retries engine admission (short, bounded, under a second).
- Engine returns `503` → ingress propagates to client with `Retry-After` and the appropriate rate-limit header.

Three disciplines:

- **Don't hold a request long at the ingress.** The ingress is not the queue; the engine is. If the engine admission is rejecting, the client should know (`503`) rather than wait on a long ingress-side timeout. Hold for < 1 second; then reject.
- **Set the engine's trace context.** The ingress's `traceparent` becomes the parent of the engine's span; the engine's internal spans are children. The trace tree is contiguous.
- **Propagate the engine's back-pressure signal.** If the engine is admission-rejecting, the ingress should see that signal (via a status code or a sideband metric) and begin rejecting proactively rather than waiting for every request to fail.

## Audit logging

Every request produces an audit record:

- `request_id`, `traceparent`, `tenant_id`, `principal`.
- Endpoint, model, `Idempotency-Key` (if any), request body hash.
- Response status, response body size, latency.
- Auth outcome, rate-limit outcome, tenant-quota outcome.

Discipline:

- **Not the full body.** Message bodies are often sensitive (user PII, business data). Log a hash and a size; expose the body in logs only in a sampled debug mode with retention controls.
- **Structured.** JSON lines, one record per request. Chapter 03 of module 110 teaches the sampling policy.
- **Retained for incident triage.** 30-90 days for infrastructure debugging; the retention policy itself is a compliance / legal decision.

## Three failure modes

### The retry storm after a transient failure

A degraded pod returns `502` for 30 seconds. The 50 QPS of clients each retry; without exponential backoff and jitter, the retries arrive synchronously and double the inbound rate on the recovering pod. The pod goes down for real.

Defence: emit `Retry-After: 2-10` with jitter; recommend exponential backoff in the API docs; add a circuit breaker at the client-SDK level (not ingress's job, but publishable).

### The double charge from `504` retry

The ingress times out at 59 seconds while the engine is still running. The client retries without an idempotency key; both complete; both bill.

Defence: document the `504` retry danger in the API docs; recommend `Idempotency-Key` on all production clients; set the ingress timeout greater than the engine's own timeout so the ingress doesn't surface a `504` while the engine is still live.

### The noisy tenant degrades everyone

A tenant submits a benchmark burst without warning; queue depth climbs; every tenant's TTFT regresses.

Defence: per-tenant admission quota at the ingress; aggressive onboarding review for enterprise tenants; a per-tenant rate-limit policy that's conservative by default and relaxed on request.

## Summary

- The ingress is where identity, quota, and trace context get set — in that order — before the engine sees the request.
- `x-request-id` plus W3C `traceparent` propagate through every log, metric, and span.
- Three auth models: API key, OAuth 2.1 bearer, short-lived STS. Each has a middleware; each extracts internal identity; the engine never sees the token.
- `Idempotency-Key` is the client's retry safety; the server records (tenant, endpoint, key, body-hash) and returns cached response on retry.
- Retry semantics are server-emitted: `Retry-After` on `429`/`503`, document the `504` danger, treat streaming failures as a different retry class.
- Rate-limit headers: emit both the IETF `RateLimit` set and the legacy `X-RateLimit-*` set; rate limit by work (tokens-per-minute) not just by requests.
- Tenant isolation is enforced at admission, before the engine consumes KV; per-tenant quotas, model allowlists, priority classes.
- Audit log every request with structured fields; hash bodies; retain for incident triage.
- Three failure modes: retry storm, double charge from `504`, noisy tenant. Each is one middleware away from being correct.
