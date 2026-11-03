# Exercise 03 — Cost-Aware Router Authoring with LiteLLM Shape

**Time:** 4 hours · **Output:** one router config, one running router in front of your fleet, one escalation / budget / tenant-quota implementation, one measurement table, one failure-drill log

## Goal

Make [Chapter 03](../03-cost-aware-request-router.md) concrete by **standing up a cost-aware router in front of your fleet** that composes self-hosted and at least one external provider, implementing **cheap-model-first with escalation, per-request budget, per-tenant quota, and provider fallback**. By the end you can produce a cost-per-1k-tokens table that includes the escalation-rate axis and defend the router's policy against measured workload.

## Problem statement

Your platform owes two guarantees: tenants get the quality tier they signed up for, and the fleet's cost per inference stays within budget. The router is where both are enforced. You will stand up a router (either from scratch, or LiteLLM-as-proxy, or an equivalent), configure it against a real fleet (at least one self-hosted model and at least one external provider), drive mixed workload through it, and publish the measured cost / quality / escalation picture.

## Prerequisites

- At least one self-hosted model running (reuse your exercise-01 deployment or spin up a small `vllm serve`).
- API access to **at least one external provider** that offers OpenAI-compatible chat completions. Options:
  - **Fireworks AI** — https://fireworks.ai (free credits for new accounts).
  - **Together AI** — https://together.ai (free credits).
  - **Groq** — https://groq.com (free tier).
  - **DeepInfra**, **Baseten**, **Anyscale** — various free / low-cost tiers.
  - **OpenAI / Anthropic / Google** — paid but low cost at exercise volumes; honour your organisation's spend policy.
- A pinned router implementation. Options:
  - **LiteLLM** (`pip install litellm; litellm --config config.yaml`). The reference implementation; see [`resources.md`](../resources.md).
  - **Custom FastAPI router** following the chapter-03 shape.
  - **OpenRouter-style** gateway if your org already has one.
- Chapter 03 of this module read.

## Requirements

Deliver five artefacts.

### Artefact A — Router config

Author the router config. For LiteLLM-as-proxy, the shape is:

```yaml
model_list:
  - model_name: chat-small
    litellm_params:
      model: openai/<your-self-hosted-model>
      api_base: http://localhost:8000/v1
      api_key: dummy
  - model_name: chat-small
    litellm_params:
      model: fireworks_ai/accounts/fireworks/models/llama-v3p1-8b-instruct
      api_key: os.environ/FIREWORKS_API_KEY
  - model_name: chat-large
    litellm_params:
      model: openai/<your-self-hosted-large-model>    # if available
      ...
  - model_name: chat-large
    litellm_params:
      model: fireworks_ai/accounts/fireworks/models/llama-v3p1-70b-instruct
      ...

router_settings:
  routing_strategy: "latency-based-routing"  # or weighted/least-busy
  fallbacks:
    - { "chat-small": ["chat-small-fireworks"] }
    - { "chat-large": ["chat-large-fireworks"] }
  model_group_alias:
    chat-small-fireworks:
      model_name: chat-small-fireworks-only

general_settings:
  master_key: sk-dev-...
  database_url: postgresql://...        # or sqlite for exercise
```

<!-- The exact LiteLLM schema evolves; adapt to your pinned version. The structural shape is what matters for the exercise. -->

Minimum required capabilities regardless of implementation:

- At least two backends per model alias (self-host + one external).
- Fallback chain configured.
- Rate-limit / RPS configurable per tenant.
- Logging of each request's cost (input / output tokens × price).

Commit the config to version control.

### Artefact B — The running router

Launch the router and verify it serves OpenAI-compatible traffic:

```
curl http://localhost:4000/v1/chat/completions \
    -H "Authorization: Bearer <master_key>" \
    -H "Content-Type: application/json" \
    -d '{
      "model": "chat-small",
      "messages": [{"role":"user","content":"hello"}]
    }'
```

Document:

- The exact launch command and config path.
- Startup log confirming all backends registered.
- Health check against each alias.
- An end-to-end request through the self-host path and through the fallback path (how to force fallback — e.g. stop the self-host backend and verify the request still succeeds via the provider).

### Artefact C — Policy implementation

Add three policies on top of the base router (either via LiteLLM config / plugins or in a custom wrapper):

**Policy 1 — Cheap-first with escalation.**

Implement:

- A request flagged `x-allow-cheap-first: 1` goes to `chat-small` first.
- If the response's token-level logprob entropy (or structured-output validation) suggests low confidence, re-run against `chat-large`.
- Both responses' token counts and costs are accounted; the final response goes to the client.

Signal choice:

- **Simplest**: structured-output JSON schema failure. Send a request requesting JSON; if `chat-small`'s output does not parse against the schema, escalate.
- **Harder**: confidence from logprobs. Request logprobs (`logprobs: True` + `top_logprobs: 5`); compute entropy of the token distribution for the first few tokens; escalate if above a threshold.

Pick whichever is tractable. Document the signal and the threshold.

**Policy 2 — Per-request budget.**

Implement `x-max-cost-usd` header on the request:

- Compute pre-request estimated input cost from input tokens × pricing.
- Reject with 402 if input cost exceeds max.
- Clamp `max_tokens` so that estimated total cost fits within max.
- Enforce on fallback as well — do not pick a fallback whose cost exceeds the cap.

**Policy 3 — Per-tenant quota.**

Three tenants with different quotas:

```yaml
tenants:
  - id: "tenant-A"
    tier: "gold"
    daily_token_quota: 100_000
    rps: 10
    allowed_models: ["chat-small", "chat-large"]
  - id: "tenant-B"
    tier: "silver"
    daily_token_quota: 50_000
    rps: 5
    allowed_models: ["chat-small"]
  - id: "tenant-C"
    tier: "bronze"
    daily_token_quota: 10_000
    rps: 1
    allowed_models: ["chat-small"]
```

Enforce:

- Daily token quota (sum of input + output tokens over 24 h).
- Per-tenant RPS (token-bucket).
- Allowed-models (reject if the alias is not on the tenant's list).

Each enforcement must emit an audit record (log line to a file or table).

### Artefact D — Measurement table

Design and run a 30-minute workload:

- Three tenants (A, B, C) sending traffic simultaneously.
- A mix of request types:
  - 60 % simple intent-classification prompts (should pass with `chat-small`).
  - 20 % harder reasoning prompts (likely to escalate to `chat-large`).
  - 20 % structured-output requests (JSON schema).
- Vary request rate per tenant to stress their RPS cap.

Record per tenant:

- Total requests sent / admitted / rejected.
- Rejection reason breakdown (RPS, quota, allowed-models).
- Escalation rate (requests that escalated / requests that could have escalated).
- Per-request cost; aggregate cost; cost-per-1k-tokens.
- Backend distribution (self-host vs. provider).
- Fallback incidence (requests that went to the fallback chain).

Publish as a table.

Expected shape:

- Tenant A hits its RPS cap when it exceeds 10 RPS; requests over the cap are 429'd.
- Tenant B is bounded to `chat-small`; a request specifying `chat-large` is rejected with the allowed-models error.
- Escalation rate between 10 % and 30 % for the typical workload split.
- Cost per 1k tokens on `chat-small` is dramatically lower than on `chat-large`; the escalation rate drives most of the cost delta.

### Artefact E — Failure drills

Run three deliberate failure scenarios and document the router's behaviour:

**Drill 1 — Self-host backend down.** Stop the self-host backend mid-workload. Observe:

- Does the router fall through to the fallback provider within one request?
- What is the latency impact at fallback?
- When the self-host comes back, does the router re-route to it?

**Drill 2 — Provider rate-limit.** Deliberately exhaust the provider's rate limit (send requests faster than the provider allows). Observe:

- Does the router see 429 and route elsewhere?
- Is the circuit breaker (if implemented) tripped?
- When does traffic resume to the rate-limited backend?

**Drill 3 — Cost cap enforcement.** Send a request with `x-max-cost-usd: 0.0001` to a long-context prompt. Observe:

- Is the request rejected before being sent to a backend?
- Is the error clear to the client?

Document each drill with the trigger, the observable behaviour, and whether it matched expectation.

## Starter guidance

- **Start with one self-host + one provider; add more later.** The simplest two-backend fallback is enough to prove the router works.
- **Use a sandboxed provider API key with a hard spend cap.** Many providers let you set a cap in the account console — set it to $5 to avoid runaway costs on a bug.
- **Verify pricing in your router config matches the provider's actual billing.** A pricing drift here is where the "cost accounting diverges" failure mode starts.
- **For escalation, pick the simplest signal first.** JSON-schema failure is a clear binary. Logprob entropy is more interesting but harder to calibrate.
- **For tenant quotas, start with in-memory counters.** Scale to Redis if you need multi-replica; for exercise purposes an in-memory counter is fine.
- **Log every admission decision.** The audit trail is as valuable as the measurement for debugging.
- **Use the LiteLLM project's docs for the reference config shape.** Even if you write custom routing, the LiteLLM proxy's `model_list` and `router_settings` are a battle-tested shape.

## Fallbacks for constrained environments

- **No external provider access / organisation prohibits it.** Simulate a second backend with a second self-hosted vLLM at a different port and a mock-higher cost in the config. The routing logic exercises the same.
- **No self-hosted model available.** Use two different providers (Fireworks + Together) and route between them.
- **No multi-tenant workload generator.** Three Python processes with different headers simulate three tenants cleanly.

## Acceptance criteria

1. **Artefact A** is a committed router config with ≥ 2 backends per alias, fallback chain, and tenant block.
2. **Artefact B** has a running router with health checks on each backend and a documented launch command.
3. **Artefact C** implements cheap-first-with-escalation, per-request budget, and per-tenant quota, each with an audit log.
4. **Artefact D** has the per-tenant workload results with cost, escalation rate, rejection breakdown, and backend distribution.
5. **Artefact E** has three failure drills with observed behaviour vs. expectation; any gap is called out.
6. All artefacts live in version control; the master key / API keys are **not** in version control (use secret references).

## Common failure modes

- **API keys in config files committed to git.** Rotate immediately. Use env refs (`os.environ/FIREWORKS_API_KEY`).
- **Pricing table drift.** Provider changes its price; your router keeps old price; cost accounting silently drifts. Verify pricing at the start of the exercise.
- **Cost cap applied after request completion.** You charged the cap-breach anyway. Enforce *before* backend call.
- **Fallback latency unbounded.** The chain can wait 30 s before giving up. Set per-backend timeouts to a few seconds.
- **Rate-limit bucket per router instance, not per tenant.** Multi-replica routers must share state (Redis or similar).
- **Streaming `usage` lost.** The router buffers responses and never sees the final SSE chunk; cost accounting is zero. Parse the final event.
- **Allowed-models check on alias string, not resolved model class.** Tenant creates an alias collision; policy bypass. Enforce on resolved physical model class.
- **Escalation metric mis-defined.** Report both escalation rate (as `escalated / total_requests`) and escalation hit (as `escalated_and_improved / escalated`).

## Keep it

The router config, policy wrappers, and workload harness are reusable across exercises 04 and 05 and compose with module 110's observability stack (per-tenant / per-model / per-backend metrics). The failure-drill log becomes the beginning of the router's runbook (module 113).
