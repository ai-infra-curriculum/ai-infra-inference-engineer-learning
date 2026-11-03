# Chapter 03 — Cost-Aware Request Router

> A router that only picks a model is a load balancer. A router that picks a model, a backend, a budget, a fallback, and a tenant policy — all per request, all under an SLO — is a cost-aware router. It is the control plane of a modern inference platform.

## Motivation

The platform now serves several models (chapter 02), several thousand LoRA adapters (chapter 01), and traffic from multiple tenants with different cost ceilings. Three realities force a router between the clients and the fleet:

1. **Not every request needs the expensive model.** A 70B answers a tool-use intent classification in a hundred times the cost of what a 7B can do. If 90 % of traffic can be routed cheap-first, the fleet-level dollar savings dominate every other optimisation.
2. **No single backend handles every workload.** Burst traffic exceeds self-hosted capacity; some requests need a vendor-only model (frontier reasoning); some tenants prefer a specific provider for compliance reasons. The router has to route across self-host → Fireworks → Together → Baseten → provider APIs.
3. **Multiple tenants need per-tenant policy.** Tenant A has a $500/day cap; tenant B has rate-limits on a specific model; tenant C must never leave the self-hosted fleet. The router is where that is enforced.

By the end of this chapter you can:

- Enumerate the five decisions a router makes per request and name the knob for each.
- Design a cheap-model-first policy with a defensible escalation signal (confidence drop, structured-output-schema failure, explicit fallback).
- Enforce per-request token budget and per-tenant quota at admission.
- Compose self-hosted and external providers behind one OpenAI-compatible endpoint, with provider fallback on failure.
- Draw the boundary to the application-side SDK ([`llm-application-developer`](https://github.com/ai-engineering-curriculum/llm-application-developer-learning), peer, level 25) — this chapter owns the server, that peer owns the client.

<!-- needs-research: LiteLLM, OpenRouter, and other router projects evolve rapidly. Verify the specific config shapes, fallback semantics, and provider plugin availability against the pinned version of whichever router implementation you adopt. -->

## The five decisions

A cost-aware router makes five decisions per request:

1. **Which model?** The user asked for `gpt-oss-mega` (a logical alias); the router may map that to `llama-3.1-70b-instruct-fp8` on self-host, or to the vendor's `gpt-5` via provider API, depending on policy.
2. **Which backend?** Even once the *model* is picked, there may be several backends serving it — self-host primary, self-host failover, Fireworks, Together, provider direct. Which does this request go to?
3. **Which fallback?** If the primary backend fails (5xx, timeout, rate-limit), what is the next backend to try? And after that?
4. **Which budget?** Does this request have a token budget beyond which it should be truncated or declined? Does the tenant have a daily cost cap it is approaching?
5. **Which tenant policy?** What per-tenant knobs apply — rate limits, allowed-models list, data-boundary rules (self-host only), isolation class (chapter 05)?

A router that only does decisions 1 and 2 is a load balancer. All five are the full cost-aware shape.

## Decision 1 — Which model

### Logical vs. physical model names

Clients should ask for a **logical name** (`chat-small`, `chat-large`, `code-expert`, `summariser`) rather than a physical name (`llama-3.1-8b-instruct-fp8-v3`). The mapping is in router config:

```yaml
models:
  - alias: "chat-small"
    primary:
      backend: "self-host"
      physical_model: "llama-3.1-8b-instruct-fp8"
      base_url: "http://vllm-small.svc.cluster.local:8000/v1"
    fallback:
      - backend: "fireworks"
        physical_model: "accounts/fireworks/models/llama-v3p1-8b-instruct"
      - backend: "together"
        physical_model: "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
    cost_per_1k_input_tokens: 0.00018     # self-host cost at target utilisation
    cost_per_1k_output_tokens: 0.00036
    quality_tier: "A"
```

The alias decouples the fleet's physical shape from the client's contract. The physical model can swap (module 105 chapter 06) without the client re-coding; the fallback can change without a client release.

### Cheap-model-first

For workloads where quality can be approximated — intent classification, simple summarisation, straightforward RAG answers — route to a cheaper model first and only escalate on signal.

Three escalation signals:

- **Confidence drop.** The cheap model returns a logprob / top-k distribution that is too flat (entropy above a threshold); the router escalates to the expensive model. Requires logprob access from the cheap model's response (vLLM / SGLang / most providers support this).
- **Structured-output schema failure.** The cheap model's response does not match the required JSON schema (even after constrained decoding). Router escalates. See module 111 for the structured-output surface this depends on.
- **Explicit fallback request.** The client (or an orchestration layer) tags the request `fallback_on: ["low_confidence", "schema_fail"]`. Router escalates on exactly those conditions.

Example router pseudocode:

```python
async def route(request):
    # Decision 1 + 5: which model, which tenant policy
    tenant_policy = tenant_registry.get(request.tenant_id)
    requested_alias = request.model               # "chat-small"

    if requested_alias not in tenant_policy.allowed_models:
        return deny(reason="model not in tenant allow-list")

    # Try cheap model first if request allows.
    if request.allow_cheap_first:
        cheap_response = await call_backend(
            model_registry["chat-small"], request)
        if cheap_response.ok and _confident(cheap_response):
            return cheap_response
        if cheap_response.ok and _schema_valid(cheap_response, request.schema):
            return cheap_response
        # Escalate.
        request = request.with_escalation_tag(reason="cheap_low_confidence")

    # Default or escalated path.
    return await call_with_fallbacks(
        model_registry[request.target_alias], request)
```

Measurable outcome: an **escalation rate**. In production the escalation rate is one of the router's primary SLIs — too low and the cheap model is undercalibrated; too high and the cheap-first policy is net-negative. Target ranges depend on workload; typical shops aim for 10–30 % escalation and reassess if either direction drifts.

### Model-tier quality

The router config declares a **quality tier** per alias (`A`, `B`, `C`, or similar). The router's SLA with tenants is in tiers; the fleet's physical shape maps aliases to tiers. If the self-hosted `chat-large` goes down, the router can fall back to a provider-hosted equivalent in the same tier — maintaining the SLA — but will not silently downgrade a tier-A request to tier-C.

## Decision 2 — Which backend

The physical model may have multiple backends:

- **Self-host primary** (fast, cheap, controlled). The default.
- **Self-host failover** (same engine, different fleet / region). On primary failure.
- **External provider** (Fireworks, Together, Baseten, Anyscale, Modal, provider APIs). On capacity overflow or regional failure.

Backend selection is a routing-weights decision:

```yaml
"chat-small":
  routing_weights:
    - backend: "self-host-us-east"
      weight: 70
      max_rps: 500
      max_concurrent: 100
    - backend: "self-host-us-west"
      weight: 20
      max_rps: 200
    - backend: "fireworks"
      weight: 10
      max_rps: 1000              # burst capacity
```

The router routes new requests across backends by weight, respecting per-backend concurrency limits. When a backend approaches its concurrency limit, the router shifts load to the others.

### Capacity-aware routing

A smarter router reads **real-time capacity signals** from each self-hosted backend (via module 110 metrics):

- Queue depth (how many requests waiting).
- In-flight requests.
- TTFT P99 (shows degraded backend early).
- GPU memory pressure / preemption rate.

A backend that is approaching queue-depth limits is *de-prioritised*; a backend that is comfortable is *preferred*. The router does not need to be a scheduler, but it needs to know when to stop sending to a hot backend.

<!-- needs-research: in-tree router implementations in vLLM / SGLang / Dynamo expose capacity metrics via different endpoints. The specific metric names ("vllm:num_requests_waiting", "sglang:queue_depth", etc.) rotate; verify against the pinned engine. -->

### Provider adapters

Each external provider has its own API shape. Three strategies:

1. **OpenAI-compatible everywhere.** Many providers (Fireworks, Together, Groq, DeepInfra) expose an OpenAI-compatible API. The router targets `/v1/chat/completions` on each backend; swapping is just a URL + API key change.
2. **Provider-specific adapter.** For non-OpenAI shapes (older Anthropic Claude, Google Vertex, Bedrock), implement a per-provider adapter that translates request / response.
3. **Use a router library.** Projects like LiteLLM (see [`resources.md`](resources.md)) already implement provider adapters; the router can embed LiteLLM as a dependency rather than re-authoring provider shapes. **Important boundary:** LiteLLM-as-a-library-in-your-router is this chapter; LiteLLM-as-the-client-SDK-in-the-application is the application-developer peer. Same project, two usage sides.

## Decision 3 — Which fallback

Fallback is the sequence of backends to try when the primary fails. Three failure classes drive different fallback behaviours:

- **Transient (5xx, timeout, connection reset).** Retry with jitter on the same backend first; after N retries, try the next backend in the fallback list.
- **Rate-limit (429, provider quota exhausted).** Skip retry on this backend; move to next in fallback list; mark the backend as "cooldown" for a window.
- **Model-specific (model unavailable, context too long).** Try the next backend immediately; if all backends fail, surface a specific error to the client.

Example fallback semantics:

```yaml
"chat-large":
  primary: "self-host-us-east"
  fallbacks:
    - backend: "self-host-us-west"
      max_retries: 1
    - backend: "fireworks"
      max_retries: 2
    - backend: "together"
      max_retries: 2
  final_fallback:
    backend: "openai-direct"
    model: "gpt-4o"              # explicit tier downshift
    tag_response: "degraded_from=chat-large"
```

The `final_fallback` is the "we must return *something*" option — it is slower, more expensive, or a different model, but it keeps the request alive. The response should carry a header that advertises the degradation (see Decision 5 — tagging).

### Circuit breakers

For sustained backend failure, the router opens a **circuit breaker**: it stops sending requests to the backend for a cooldown window, half-opens to probe, and closes when the backend is healthy again. Standard pattern; the implementation is the usual circuit-breaker shape (threshold on failure rate over window, cooldown exponential, half-open probe at 1 % traffic).

## Decision 4 — Which budget

### Per-request budget

A request can carry a **max_tokens** (output cap) and may also carry a **max_cost** (dollars). The router enforces both:

- Compute input cost from the token count × the model's input price.
- Decline if `input_cost > request.max_cost`.
- Clamp `max_tokens` to the lower of `request.max_tokens` and `(request.max_cost - input_cost) / output_price_per_token`.

This is a cheap guard against runaway cost on a single request.

### Per-tenant quota

Each tenant has:

- **Daily token quota** (input + output tokens over 24 h).
- **Daily cost cap** (dollars over 24 h).
- **Concurrent-request quota** (max in-flight requests from this tenant).
- **RPS quota** (requests per second, with burst).

The router checks these at admission. Options for the enforcement implementation:

- **Local counter.** For a single-replica router, in-memory counters are fine.
- **Redis-backed counters with Lua for atomicity.** For multi-replica routers, a shared counter with atomic increment-and-check.
- **Token-bucket per tenant.** For RPS, a token-bucket at the gateway (e.g. Envoy / NGINX / a dedicated rate-limit service) is standard.

The budget / quota check runs at admission; if the request is admitted, the router can estimate the output cost pessimistically (assume max_tokens) and reserve budget. On completion, settle the reservation against actual tokens used.

### Cost accounting

Every response the router handles must be **accounted** — tenant ID, model, backend, input tokens, output tokens, computed cost. This feeds module 110's cost-per-1M-tokens dashboards and the tenant's billing. Where accounting happens:

- Response middleware in the router captures the usage object (OpenAI-compatible responses include `usage.prompt_tokens` and `usage.completion_tokens`).
- Pricing comes from the router config (self-host cost is computed from fleet-level amortisation; external-provider cost from the provider's rate card).
- Rows go to an append-only log (Kafka / Kinesis / Pub/Sub) for the billing pipeline.

The router is the system of record for request-level cost. If accounting is wrong here, every downstream cost signal is wrong.

## Decision 5 — Which tenant policy

The router is the first point in the stack that knows the tenant. The policy applied at admission:

- **Allowed models.** A tier-B tenant may not request `chat-large`. A tenant under a specific contract may be restricted to self-hosted models (data-boundary — see chapter 05).
- **Isolation class.** Which fleet / partition does this tenant land on? If chapter-05's isolation contract designates tenant A as "silver" with a dedicated fleet, the router routes to that fleet.
- **Rate / cost limits.** Per Decision 4.
- **Priority.** For admission queueing when the backend is near capacity, tenant tier determines priority.
- **Response tagging.** The router injects headers (`x-tenant-id`, `x-route-decision`, `x-backend`, `x-estimated-cost`, `x-fallback-chain`) for observability. These must be filtered before client response if tenant isolation requires it.

### The tenant registry

The source of tenant policy. Shape:

```yaml
tenants:
  - id: "customer-7f3a"
    tier: "gold"
    allowed_models: ["chat-small", "chat-large", "summariser"]
    daily_token_quota: 20_000_000
    daily_cost_cap_usd: 500
    rps: 50
    concurrent: 20
    isolation_class: "gold-pool"       # chapter 05
    data_boundary: "self-host-only"    # no external providers
    allow_prefix_sharing: false        # module 104 boundary
    sla_tier_promise: "A"              # which quality tier is guaranteed
```

A well-run platform has this registry version-controlled, change-reviewed, and signed-off by the ops team per tenant.

## The OpenAI-compatible surface the router presents

The router's upward-facing API is **OpenAI-compatible** (chapter 11-shaped in module 111). From the client's perspective:

```
POST /v1/chat/completions
{
  "model": "chat-small",         // alias
  "messages": [...],
  "max_tokens": 500,
  ...
}
```

The router's response adds observability headers but keeps the body shape OpenAI-compatible:

```
HTTP/1.1 200 OK
x-request-id: req_abc123
x-model-alias: chat-small
x-physical-model: llama-3.1-8b-instruct-fp8
x-backend: self-host-us-east
x-tenant: customer-7f3a
x-escalation-chain: cheap_first,escalated=false
x-input-tokens: 128
x-output-tokens: 342
x-cost-usd: 0.000146

{ "id": "chatcmpl-...", "choices": [...], "usage": {...} }
```

Headers downward give the client just enough to log the routing decision without exposing the fleet's physical structure.

## The LiteLLM shape — a reference point

[LiteLLM](https://github.com/BerriAI/litellm) (see [`resources.md`](resources.md)) is an open-source project with two usage modes:

- **As a Python library** inside an application (`litellm.completion(model="gpt-4o", ...)`). This is the application-developer peer's territory.
- **As a proxy server** (`litellm --config config.yaml` → OpenAI-compatible endpoint with routing / fallback / cost tracking built in). This is the server-side router.

The LiteLLM proxy config shape is a useful reference even when authoring a from-scratch router:

```yaml
model_list:
  - model_name: chat-small
    litellm_params:
      model: openai/llama-3.1-8b-instruct-fp8
      api_base: http://vllm-small.svc.cluster.local:8000/v1
      api_key: dummy
  - model_name: chat-small
    litellm_params:
      model: fireworks_ai/accounts/fireworks/models/llama-v3p1-8b-instruct
      api_key: os.environ/FIREWORKS_API_KEY

router_settings:
  routing_strategy: "latency-based-routing"
  fallbacks:
    - { "chat-small": ["chat-small-fireworks"] }

general_settings:
  master_key: sk-...
  database_url: postgresql://...
```

<!-- needs-research: LiteLLM's config schema has evolved; the above is a structural reference. For a production deployment, consult the LiteLLM project docs and release notes for the exact field spellings and feature set. -->

The decision whether to adopt LiteLLM-as-proxy, embed it as a library in a custom router, or author from scratch depends on operational preferences; the **shape** of the config is a reasonable template either way.

## Common failure modes

### Escalation storm

The cheap model's confidence signal is miscalibrated; 95 % of requests escalate. Fleet cost explodes. Fix: monitor escalation rate per model alias as a first-class SLI; alert on sudden drift; recalibrate the confidence threshold.

### Fallback cascade timeout

Primary fails after 10 s; failover fails after 10 s; third fallback fails after 10 s; total client wait is 30 s. The client has already given up. Fix: bound total latency across the fallback chain; set per-backend timeouts aggressively (TTFT budget of 2–3 s each, not 10 s each); abort the chain when total elapsed > client timeout.

### Silent provider downgrade

A `chat-large` request falls all the way down the chain to `openai-direct → gpt-4o-mini` (a tier-C model). The client gets a response but has no idea it is a degraded output. Fix: tag the response (`x-degraded-from: chat-large`); the application-side peer at `llm-application-developer` can decide whether to accept.

### Tenant quota bypass via alias

Tenant A's quota is on alias `chat-large`; A learns to request `chat-large-fireworks` directly to bypass. Fix: enforce quota on the resolved physical model class, not the alias string; validate the client-provided alias against the tenant's allow-list before resolution.

### Hot provider API key

One API key for all tenants hitting Fireworks. One tenant's traffic exhausts the key's rate limit; everyone 429s. Fix: per-tenant provider keys where possible; otherwise per-tenant RPS caps at the router that keep aggregated traffic under the provider's single-key limit.

### Lost cost accounting on streaming

Streaming responses don't surface `usage` until the final SSE chunk. If the router buffers responses, it has the data; if it passes through, it must parse SSE to extract. Easy to miss. Fix: ensure the router's response middleware parses the final usage event and emits the accounting record before closing the client stream.

### Rate-limit header pass-through

Provider returns `x-ratelimit-remaining: 10`; router passes it through to the client, exposing the fleet's shared provider key state. Minor but a leak. Fix: strip provider-side rate-limit headers; replace with router-computed tenant-side headers.

## The boundary to `llm-application-developer`

The server-side router (this chapter) owns:

- The admission policy and tenant registry.
- Backend selection, fallback, circuit breaker.
- Server-side cost accounting and billing feed.
- The OpenAI-compatible API surface presented upward.

The application-developer peer owns:

- The client-side SDK (OpenAI SDK, LiteLLM-as-library, custom HTTP).
- Retry semantics inside the application process.
- Application-side timeout, cancellation, request cancellation.
- Token-budget awareness from the application's product perspective.
- The product-level failure handling ("if the model gives low confidence, show the user X").

The interface between the two is the OpenAI-compatible endpoint plus the observability headers this chapter publishes. Each side has retry and timeout; both must be aware of each other's budgets, but they do not share internals. If an application wants to implement its own cheap-first → expensive fallback inside the application code (bypassing the server router entirely), that is their choice — but then they take on the responsibilities the server was handling (cost accounting, provider fallback, quota).

A healthy platform runs **both**: the server router owns platform-level policy (quota, data boundary, provider failover); the application-side SDK handles application-level concerns (per-user quota, UX degradation, product retry). Don't put product-level retry on the server; don't put platform-level quota in the app.

## Summary

- The cost-aware router makes five decisions per request: which model, which backend, which fallback, which budget, which tenant policy.
- Cheap-model-first with escalation on confidence drop or schema failure is the dominant cost-saving pattern. Monitor the escalation rate as a first-class SLI.
- Backend routing is weighted + capacity-aware; the router reads real-time backend signals and shifts load off hot backends.
- Fallback is a sequence of backends with per-backend retry policies and a circuit breaker. Bound the total latency of the fallback chain, not just per-backend.
- Per-tenant quota, cost cap, allowed-models, and isolation class are enforced at admission from a version-controlled tenant registry.
- The router presents an OpenAI-compatible upward surface; downward it adapts to each backend's API shape (OpenAI-compatible everywhere is the easy case; provider adapters cover the rest; LiteLLM is a usable library or proxy for this).
- The boundary to the application-developer peer: this chapter owns the server-side router; that peer owns the client-side SDK. The interface is the OpenAI-compatible endpoint plus observability headers.
