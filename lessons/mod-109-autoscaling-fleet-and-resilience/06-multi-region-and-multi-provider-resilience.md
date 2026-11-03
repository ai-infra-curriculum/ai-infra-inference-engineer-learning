# Chapter 06 — Multi-Region and Multi-Provider Resilience: Global LB, Provider Fallback, Cost-vs-Latency Routing

> One region's spot pool is now a well-drained, chapter-05-disciplined asset. One region's on-demand pool is still, at the end of the day, one region — one blast radius for a cloud event, one price curve, one set of per-provider rate limits. The last ring of resilience is multi-region (same provider, different region) and multi-provider (self-hosted + managed, or self-hosted + external vendor). This chapter builds the serving-edge patterns.

## Motivation

A platform team's fleet is in `us-east-1`. Chapter 01–05 are all shipped. Three events over six months stress the one-region shape:

1. **A cloud outage in `us-east-1`.** For 3 hours the whole region is degraded; the fleet's SLO is 0 % availability. The team's customers see downtime that has nothing to do with inference quality — it's substrate failure the inference tier has no answer for.
2. **A spot capacity event across all AZs in `us-east-1`.** On-demand p5 instances are also back-ordered. Chapter-05 drains are clean but the scale-out fails; the warm pool is saturated. SLO breaks for 90 minutes.
3. **A new tenant with EU data-residency requirements.** Their requests must not leave the EU; the fleet is in the US. The customer threatens to switch providers.

All three events are *not* internal-to-the-replica problems. They are above the replica — at the fleet-topology layer. Chapter 06's job is to raise the resilience ceiling by one more ring:

- Multiple regions in the same cloud, routed by a global LB (same provider, different region).
- Multiple providers — self-hosted + external (Fireworks / Together / Baseten / NVIDIA NIM / AWS Bedrock / Google Vertex) — routed by a LiteLLM-shape router.
- Cost-vs-latency routing — serve some traffic to a cheaper backend when the latency gain is marginal, keep the SLO-tight traffic on the fastest path.

By the end of this chapter you can:

- Deploy a weighted-DNS / Anycast / global LB pattern at inference-API granularity.
- Deploy a LiteLLM-shape provider-fallback router with per-tenant policy.
- Reason about cost-vs-latency routing — the latency tax, the price curve, the per-provider rate limits.
- Compose with chapters 02 (per-region scalers), 04 (per-region warm pools), 05 (regional spot preemption as a routing signal).

<!-- needs-research: specific external-provider feature parity (Fireworks, Together, Baseten, NVIDIA NIM, AWS Bedrock, Google Vertex) rotates fast. The LiteLLM project tracks a maintained adapter surface; always verify current provider support and quirks against the LiteLLM docs and each provider's current pricing / SLA. -->

## The three resilience rings

A quick taxonomy of what each ring buys and costs:

| Ring | Blast-radius guard | Mechanism | Latency tax |
|---|---|---|---|
| **Same-region on-demand** | None (one AZ event wipes it) | Chapter 01 | ~0 |
| **Multi-AZ on-demand** | AZ-level event | Chapter 01's node pools across AZs | ~0 (intra-region) |
| **Multi-AZ with spot diversity** | AZ or spot-type event | Chapter 05 | ~0 (clean drain) |
| **Multi-region same cloud** | Regional event | Chapter 06 Section 1 | 10s–100s of ms (cross-region) |
| **Multi-provider** | Provider event, provider capacity, provider quota | Chapter 06 Section 2 | tens of ms (nearest provider) to hundreds (far provider) |

A production fleet's SLO sets the required ring. 99.9 % SLO can usually live inside one region; 99.99 %+ generally requires multi-region or multi-provider.

## Section 1 — Multi-region, same cloud

### The two-region primer

Two regions, one on each coast (`us-east-1` and `us-west-2` for example). The fleet in each is a full chapter-01–05 stack: GPU operator, HPA on queue depth, warm pool, spot reactive, drain protocol.

The question is how traffic routes between them:

- **Active-active (both serve traffic).** Each region serves ~50 % of users; the LB routes by geography (nearest-wins) or by weighted DNS. Failover is "if health check fails on region A, shift the weight to region B."
- **Active-passive (one serves, one is standby).** Region A takes all traffic; region B has a warm pool but no reactive traffic. Failover is "flip the DNS." Cheaper (warm-pool-only in region B) but with longer recovery (reactive layer in region B needs to scale up).

For most inference fleets, **active-active** is the right shape because:

- The warm-pool and reactive capacity needs to be sized for failover anyway; active-active uses it at steady state rather than letting it sit cold.
- Geography-based routing lowers the baseline latency for the user — a user in `us-west-2`'s hinterland sees the local region.
- Failover is a weight shift, not a cold region warm-up.

### The global LB options

Three mechanisms:

#### Option 1 — weighted / latency / geo DNS

Classic shape: multi-A-record / Route 53 health checks / Cloudflare DNS. The LB is the DNS layer; clients resolve the fastest region's hostname.

- Pro: cheap, simple, works everywhere.
- Pro: integrates with health checks (`AWS Route 53 health check` / `Cloudflare health check`) — on failure, the DNS stops returning the failed region's IP.
- Con: DNS caching. The TTL on an A record is 30–60 seconds in practice (clients + resolvers cache). On a region failure, up to 60 seconds of client requests go to the failed region before the resolver re-queries.
- Con: no per-request routing. Each client sticks to the resolved region for its DNS cache lifetime.

#### Option 2 — Anycast + regional backends

The LB has a single Anycast IP that BGP-routes to the nearest region. Classic shape: Google Cloud Load Balancer (global), Cloudflare Load Balancing, Fastly, AWS Global Accelerator.

- Pro: no DNS caching concern — the IP is the same everywhere; BGP routes to the nearest healthy region.
- Pro: sub-second failover — the LB health check in the global tier can shift traffic in seconds.
- Con: more expensive; some clouds charge for the global tier.
- Con: inference APIs are typically long-lived SSE streams; some global LBs have limits on connection duration (verify).

#### Option 3 — client-side router / service mesh

Chapter's third pattern: a router process in front of the inference APIs that chooses a region per request. Shape: an envoy / NGINX / HAProxy / custom router that holds live latency / health signals and routes.

- Pro: per-request intelligence — route to the region with lowest current queue depth, highest cache hit, etc. (reads chapter-02 metrics).
- Pro: application-visible — can carry tenant context, API-key quotas, retry logic.
- Con: must be deployed and operated itself (chicken-and-egg: where does the router live if the regions are the failover plan?).
- Con: more latency than a straight LB.

A common hybrid: **Anycast global LB for the baseline failover; a client-aware router at the API gateway for per-tenant / per-model routing.**

### Streaming considerations

SSE and WebSocket connections persist for the whole generation. A client that connected to a region cannot cheaply move mid-generation. Three implications:

- **Failover during a generation is lossy.** A regional failure mid-generation means the client's SSE stream drops. The client must retry — the retry can hit the other region successfully.
- **Long-polling SLAs matter.** If the LB has a 60-second idle timeout and a generation is longer than that, the LB cuts the stream mid-generation. Verify LB timeouts; set them well above the P99 generation time.
- **Session affinity vs. load-based routing.** For a long-running conversation / session, sticky routing to one region keeps the prefix cache warm (chapter 104). For stateless one-shot requests, round-robin or latency-based routing wins. Tag long-session routes with a cookie / header to enforce stickiness.

### Cross-region data transfer cost

A request from a user in `eu-west-1` to a replica in `us-east-1` pays:

- The user → cloud ingress (free in most cases).
- Any cross-region replication of model weights, cache, telemetry (cost; chapter 110 covers).
- Egress on the response (small — a few tokens).

The egress is small per-request; the model-store replication is the budget line. Weights duplicated per region cost per-region HBM equivalent in storage. Chapter 04's weight-pre-stage decision applies per region.

### Capacity sizing across regions

Rule of thumb: each region should be sized to absorb **its own baseline + a share of the other region's traffic on failover**:

```
capacity(region_A) ≥ baseline(region_A) + k × baseline(region_B)
```

Where `k` is the failover scenario probability — if full failover is realistic, `k = 1.0`; if partial failover (one of two zones in B) is more realistic, `k = 0.5`.

Combined with chapter 04's warm-pool sizing: the warm pool in each region should cover the full-failover peak, not just baseline.

### A worked example — two-region active-active via Anycast

```
Global:
  Cloudflare Load Balancing pool: inference-api
  Members:
    - us-east-1-api.example.com   (ALB in us-east-1)
    - us-west-2-api.example.com   (ALB in us-west-2)
  Steering: dynamic (lowest-RTT per client)
  Health check: GET /health, 10s interval, 2 fail threshold

Per region:
  Chapter 01 substrate: GPU Operator, DCGM, Karpenter
  Chapter 02 scaler: HPA on queue depth + TTFT P99
  Chapter 04 warm pool: N_warm = 3 on-demand
  Chapter 05 reactive: 0..15 on spot with diversity
  SLO: TTFT P99 < 2s intra-region
```

Failover scenario: `us-east-1` ALB health check fails for 20 s. Cloudflare removes it from the pool; all client resolves route to `us-west-2`. The user's effective TTFT rises by cross-region RTT (~60–80 ms coast-to-coast) and the SLO stays within budget *because* `us-west-2` was sized for the failover case.

<!-- needs-research: Cloudflare Load Balancing specifics on steering and failover behaviour. AWS Global Accelerator and Google Cloud External Load Balancer are alternatives; the specific feature matrix (SSE support, long-timeout, health-check cadence) varies by provider. Verify against current docs before adopting. -->

## Section 2 — Multi-provider: the LiteLLM-shape router

Multi-provider isn't a resilience curiosity any more — it is operationally normal. The three motivations:

1. **Reliability.** Self-hosted + an external provider is a cheap insurance against your fleet failing or your capacity exhausting.
2. **Cost / price arbitrage.** External providers at scale often price below in-house cost at certain token / rate thresholds; some workloads are better served there.
3. **Feature coverage.** External providers may offer models you don't want to host (DeepSeek-R1, GPT-5-class models, Claude-class models, proprietary vendor models).

### LiteLLM — the pattern

LiteLLM (<https://github.com/BerriAI/litellm>) is the canonical shape — a proxy that speaks OpenAI-compatible on the client side and talks to many provider APIs on the backend side (OpenAI, Anthropic, Google, AWS Bedrock, Azure OpenAI, Fireworks, Together, Baseten, Groq, Modal, self-hosted vLLM / SGLang / TRT-LLM / Triton, etc.).

A client sends an OpenAI-shape request; LiteLLM:

1. Picks the backend (based on model name, policy, health, cost).
2. Translates the request to the backend's native format.
3. Streams the response back in OpenAI shape.
4. Records metrics (per-provider latency, token counts, costs, errors).

```yaml
# LiteLLM proxy config (approximate shape; verify against pinned release)
model_list:
  - model_name: llama-3.1-70b
    litellm_params:
      model: openai/meta-llama/Meta-Llama-3.1-70B-Instruct
      api_base: http://vllm-llama31-70b.inference:80/v1
      api_key: os.environ/VLLM_KEY
      rpm: 2000                       # per-provider rate budget
      tpm: 500000
  - model_name: llama-3.1-70b-fireworks
    litellm_params:
      model: fireworks_ai/accounts/fireworks/models/llama-v3p1-70b-instruct
      api_key: os.environ/FIREWORKS_KEY
      rpm: 600

router_settings:
  routing_strategy: usage-based-routing-v2
  fallbacks:
    - llama-3.1-70b: [llama-3.1-70b-fireworks]   # self-hosted fails → external
  context_window_fallbacks: []
  cooldown_time: 60
  num_retries: 2
  retry_after: 1
```

The config knobs are policy; let me unpack a few:

- **`fallbacks`** — on failure of the primary, LiteLLM routes to the backup. Error codes that trigger fallback include 5xx, rate-limit, provider-timeout.
- **`cooldown_time`** — after a provider errors enough, LiteLLM "cools it down" for N seconds (circuit-breaker shape). Prevents retry storms from re-hitting a broken backend.
- **`rpm` / `tpm`** — per-provider request/token-per-minute budgets. LiteLLM load-balances across backends respecting each one's budget.
- **`routing_strategy`** — `simple-shuffle`, `least-busy`, `usage-based-routing-v2`, `latency-based-routing` are among the options; each has trade-offs.

<!-- needs-research: LiteLLM's config shape and the specific router-strategy names rotate per release. Verify against <https://docs.litellm.ai/> and the pinned release. -->

### Where to deploy the router

- **In-cluster, behind the global LB.** The LB routes to the nearest router instance; the router routes to backends. Latency: one extra hop (~1 ms); operational control: high.
- **Client-side.** Each client library embeds its own LiteLLM. No proxy, no extra hop. Operational cost: every client is a config/policy point.
- **Managed by an external router (not self-hosted LiteLLM).** Some vendors (OpenRouter, Portkey, Helicone) offer managed routing. Trade-off: less control, less engineering.

Most self-hosted inference teams run LiteLLM in-cluster — the control / observability / policy-flexibility outweighs the one-hop latency.

### Streaming and provider shape differences

The OpenAI-compatible API is a *compatibility surface*; providers differ in subtle ways:

- **Tool-calling / function-calling** semantics vary (JSON Schema format, parallel tool-call support, tool_choice behaviour).
- **Structured output** mode (JSON schema enforcement) is not uniform.
- **Streaming shape** — SSE chunk shape is the same; `logprobs`, `usage` in the stream, and the final chunk's trailing data vary.
- **Reasoning / thinking tokens** — some providers emit a reasoning channel; others don't. Not uniform.

LiteLLM normalises many of these; expect edge cases on advanced features. Module 111 covers the API surface at depth; this chapter notes the integration cost.

## Section 3 — Cost-vs-latency routing

Cheaper backends come with latency / reliability trade-offs. Two common patterns:

### Pattern 1 — tier-by-SLO class

Route traffic by SLO class, not by average:

- **Tight SLO (TTFT P99 < 1 s, SLA-backed):** self-hosted on-demand, no fallback except for incidents.
- **Standard SLO (TTFT P99 < 3 s):** self-hosted spot-tolerant with external-provider fallback for availability.
- **Loose SLO (batch, async, "best-effort"):** cheaper backends (external vendor at lower cost) as primary; self-hosted as spill-over.

The policy is per-tenant / per-API-key / per-model; LiteLLM's `router_group` + tag-based routing supports it.

### Pattern 2 — speculative route with latency-budget cutoff

For some workloads, you'll see two valid backends with different latencies — cheaper external at 400 ms TTFT, self-hosted at 150 ms. The policy "always use the self-hosted for lower latency" over-commits on self-hosted capacity. The policy "always use the external for cost" sacrifices latency.

A middle path: route to the cheaper external *if the latency budget allows*. On traffic that's at a 2-second SLO ceiling, 400 ms is fine; on a 500 ms SLO, 400 ms is cutting it close.

### Pattern 3 — regional and provider-aware

Combine the two: route to the lowest-cost provider that is both regionally close enough and reliable enough. The matrix:

```
tenant   region   model              primary          fallback             budget
alpha    us-east  llama-3.1-70b     self-us-east      fireworks            standard
alpha    us-west  llama-3.1-70b     self-us-west      self-us-east         standard
alpha    eu       llama-3.1-70b     self-eu-west      fireworks-eu        standard
beta     us-east  llama-3.1-70b     fireworks         self-us-east         tight
```

LiteLLM's config composes model + API-key-based routing with this matrix.

## Data-residency and tenant isolation

Multi-provider routing introduces a data-exit question: does Tenant A's data cross into Provider X's region / cloud / jurisdiction?

- **Data-residency contracts.** Some tenants require strict region-boundedness; the router must not route cross-region for them. LiteLLM's per-key / per-tag routing supports it.
- **PII and compliance.** External provider's data handling (training-on-data, logging, retention) may violate your tenant contract; the router must know which tenants can and cannot be routed externally.
- **Audit trail.** Every routing decision should be logged with tenant + chosen provider + response-level metadata (not response body unless legally allowed); SOC 2 / ISO / HIPAA audits may require it.

The security peer (`ai-infra-security`, level 35) owns the deep adversarial / compliance surface; chapter 06 notes the integration points.

## Alerts the multi-region / multi-provider pattern needs

### Alert 1 — region imbalance

```promql
abs(
  sum(rate(inference_requests_total{region="us-east-1"}[5m]))
  -
  sum(rate(inference_requests_total{region="us-west-2"}[5m]))
) / sum(rate(inference_requests_total[5m])) > 0.3
```

A sustained >30 % imbalance signals the LB health check or geo-steering is unhealthy. Chapter 05 compose: a regional spot event may be the cause.

### Alert 2 — fallback rate above expected

```promql
sum(rate(litellm_fallback_requests_total[5m])) / sum(rate(litellm_requests_total[5m])) > 0.05
```

More than 5 % of traffic falling back to secondary providers is a sign the primary is struggling. If sustained, scale primary up or investigate.

### Alert 3 — provider error-budget burn

Per-provider error rate exceeding its SLA — the external provider is misbehaving; the circuit breaker is cooling it down; some tenants are impacted. Compose with chapter 110's SLO methodology.

### Alert 4 — cost deviation

If routing policy claims "cheap primary / self-hosted fallback" but actual billing shows self-hosted is doing 80 % of the work, the primary is throwing errors enough to spill over. Compose with chapter 110's cost-per-token tracking.

## Common deployment errors

### Error 1 — DNS failover tested only in theory

The team configures a Route 53 weighted record set but has never shifted the weight. The weight-shift mechanism has unknown bugs; the first real failover fails. Fix: a quarterly chaos drill where one region is proactively drained.

### Error 2 — LiteLLM router without quotas

A tenant's traffic spike exhausts the entire external-provider budget; the fallback is unavailable for all other tenants. Fix: per-tenant rate limits enforced at the router.

### Error 3 — fallback with no cooldown

The primary flaps; the router retries without a cooldown; the primary never recovers from the retry storm. Fix: circuit breaker with `cooldown_time`.

### Error 4 — session affinity broken by failover

Mid-conversation routing shifts regions; prefix cache lost; TTFT doubles for the next request in the session. Fix: sticky routing by session token; failover only on health check failure.

### Error 5 — streaming cut by LB timeout

The LB has a 60-second idle timeout; a long generation hits it and the stream dies. Fix: set LB timeouts above P99.9 generation time (minutes, not seconds).

### Error 6 — forgetting the regional weight-store

Model weights pulled fresh from `us-east-1` to `us-west-2` on failover; weight-pull adds minutes; the SLO cold-start compounds the regional failover. Fix: pre-stage weights in each region's object store (chapter 04 compose).

### Error 7 — external provider's data policy breach

Tenant A's requests routed to external provider with "we train on data" by default; tenant contract violated; incident. Fix: audit each provider's data policy; configure "no training" explicitly; verify.

### Error 8 — multi-region without cross-region observability

Each region's Prometheus is independent; the fleet-level dashboard is a manual sum. Debugging cross-region patterns is painful. Fix: Thanos / Mimir / cross-region Prometheus federation (chapter 110).

## Composing with chapters 02, 04, 05

- **Chapter 02's HPA** runs per region. Each region's scaler acts on its region's metrics; the global LB spreads traffic between regions. The two don't talk directly; they talk via "the LB's health check sees unready → traffic shifts."
- **Chapter 04's warm pool** sits per region, sized for its region's baseline *plus* the failover share. Chapter 06 is why warm pools are sized larger than they'd otherwise need.
- **Chapter 05's drain** triggers at the pod level; the global LB sees the pod unhealthy and routes new requests elsewhere (in-region if available, cross-region if the whole region is draining).
- **Chapter 03's sharing** is per-replica and inside the pod; chapter 06 is above all of it. The sharing mode choice doesn't change the chapter-06 pattern.

## The boundary to the level-20 peer and the team-lead track

This chapter does not teach:

- **DNS fundamentals**, BGP, Anycast at the networking layer. Level 20 / networking generalist.
- **General multi-region replication for a stateless service.** Level 20.
- **Vendor-management business discipline** — negotiating with providers, business continuity planning. Team-lead-track (L40) / chapter 114 of this track.

This chapter *does* teach: inference-specific multi-region routing with streaming semantics, the LiteLLM-shape router pattern for provider fallback, the cost-vs-latency routing policy shape, and the alert set that catches a mis-tuned router before it causes an SLO breach.

## Summary

- The last ring of resilience is multi-region (same cloud) + multi-provider (self-hosted + external). SLOs at 99.99 % and above generally require it.
- Three global-LB options: **weighted DNS** (cheap, slow failover), **Anycast / global LB** (fast failover, more expensive), **client-side router** (per-request intelligence, operational cost). Hybrids are common.
- Streaming inference APIs constrain the LB — long-lived SSE connections, idle timeouts, mid-generation failover is lossy. Size timeouts and plan for client-retry.
- Capacity per region: baseline + failover-share. Warm pool per region accordingly.
- **LiteLLM-shape router** for multi-provider — OpenAI-compatible front, many providers back, per-tenant policy, per-provider rate budget, circuit-breaker fallback.
- Cost-vs-latency routing is per-SLO-class (tight / standard / loose) and per-tenant; the router expresses the policy.
- Data-residency / compliance / tenant isolation shape the routing policy; security peer owns the depth.
- Alerts on region imbalance, fallback rate, provider error budget, and cost deviation.
- Compose with chapters 02 / 04 / 05: scaler per region, warm pool per region, drain per pod, cross-region failover as the next ring.
- Generic DNS / networking is level-20's; vendor-management business is team-lead's; this chapter owns the inference-shaped routing pattern.
