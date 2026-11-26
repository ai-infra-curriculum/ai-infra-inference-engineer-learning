# Exercise 06 — Multi-Region Provider-Fallback Router

**Time:** 4 hours · **Output:** one LiteLLM-shape router in front of two backends (two regions, or self-hosted + external), one fallback + circuit-breaker policy, one region-failure injection drill, one cost-vs-latency routing table, one cross-region observability wiring, one `<!-- needs-escalation -->` list

## Goal

Make [Chapter 06](../06-multi-region-and-multi-provider-resilience.md) concrete by **deploying a LiteLLM-shape router in front of two backends** — two self-hosted regions, or one self-hosted + one external provider — authoring the fallback policy with per-provider rate limits and circuit-breaker cooldowns, injecting a region failure, and confirming the SLO survives the failover. The deliverable is the last resilience ring — the thing that catches the requests exercises 01–05 can't.

## Problem statement

Exercises 01–05 built one region's fleet into a well-drained, well-scaled, well-observed thing. Chapter 06's opening events still happen: a regional outage, a region-wide spot capacity event, an EU-data-residency tenant whose requests must not leave the EU. The one-region fleet cannot answer any of them. The exercise is where the second region (or the external-provider fallback) lands, the router that makes the two look like one to the client comes up, and the drill proves the failover actually works.

## Prerequisites

- Chapter 06 of this module read. Chapters 02, 04, 05 skimmed (per-region scalers / warm pools / drains compose underneath).
- **Exercises 01–05 complete** — at least one region has the full stack. Ideally two regions, each with their own exercises 01–04; exercise 05 optional but useful.
- Two inference backends addressable from where the router will run:
  - **Option A (two self-hosted regions):** your exercise-01 fleet in region A (e.g. `us-east-1`) and region B (e.g. `us-west-2`), each exposing an OpenAI-compatible endpoint.
  - **Option B (self-hosted + external provider):** your exercise-01 fleet in one region plus an external provider with an API key (Fireworks AI, Together AI, Baseten, AWS Bedrock, Google Vertex AI, OpenAI, Anthropic, Groq, etc. — LiteLLM supports many; verify the specific adapter against <https://docs.litellm.ai/docs/providers>).
  - Option A is operationally closer to chapter 06; Option B is cheaper to stand up for the exercise. Either is acceptable.
- A Kubernetes namespace to run the router in (self-hosted) or host-based deployment of your choice.
- Credentials / API keys for the external provider (if Option B), stored as Kubernetes Secrets.
- A global LB option available — Cloudflare Load Balancing, AWS Global Accelerator, GCP Cloud Load Balancing, or Route 53 weighted DNS. One of these is enough for Deliverable D; the choice depends on what your platform already uses.
- Load driver from exercises 02 / 04 (same shape; the drill reuses it).

## Requirements

Six deliverables.

### Deliverable A — the router install + two-backend config

Install LiteLLM (<https://github.com/BerriAI/litellm>) as a proxy in your cluster. The canonical path is the LiteLLM Helm chart or the Docker image behind a Deployment + Service. Config shape (verify against the pinned LiteLLM release's docs):

```yaml
# litellm-config.yaml  (ConfigMap-mounted)
model_list:
  - model_name: llama-3.1-70b                       # client-facing model name
    litellm_params:
      model: openai/meta-llama/Meta-Llama-3.1-70B-Instruct
      api_base: http://vllm-llama31-70b.us-east-1.inference.svc:80/v1
      api_key: os.environ/VLLM_KEY_EAST
      rpm: 2000
      tpm: 500000

  - model_name: llama-3.1-70b
    litellm_params:
      model: openai/meta-llama/Meta-Llama-3.1-70B-Instruct
      api_base: http://vllm-llama31-70b.us-west-2.inference.svc:80/v1    # or an external provider URL
      api_key: os.environ/VLLM_KEY_WEST                                   # or FIREWORKS_KEY / ...
      rpm: 1500
      tpm: 400000

router_settings:
  routing_strategy: latency-based-routing         # or usage-based-routing-v2, simple-shuffle, ...
  fallbacks:
    - llama-3.1-70b: ["llama-3.1-70b"]            # same name; LiteLLM round-robins across entries
  cooldown_time: 60                               # cool a flapping backend for 60 s
  num_retries: 2
  retry_after: 1
  context_window_fallbacks: []
```

Attach:

- LiteLLM install command + manifests (Deployment, Service, ConfigMap, Secrets).
- The config file with placeholder keys (do NOT commit real keys).
- A smoke test: `curl http://litellm.inference.svc/v1/chat/completions -H "Authorization: Bearer <key>"` with a simple prompt; show the response.
- Evidence both backends are healthy from the router's perspective — LiteLLM exposes a `/health` endpoint (<https://docs.litellm.ai/docs/proxy/health>); show both backends green.

### Deliverable B — the fallback + circuit-breaker + rate-limit policy

Expand the router config with chapter 06 Section 2's shape:

- **Fallbacks** — primary and secondary per `model_name`. If primary 5xx or rate-limits, fall back to secondary.
- **Circuit-breaker cooldown** — after N errors in a window, cool the backend for `cooldown_time` seconds. Prevents the chapter-06 Error 3 scenario.
- **Per-backend `rpm` / `tpm`** — explicit per-provider budgets so one tenant can't drain the fallback.
- **Per-tenant policy (if LiteLLM team-based routing is configured)** — some tenants can only use self-hosted; others can spill to external. Chapter 06 Section 3 shape.

Document the policy choices:

```
# Router Policy — <fleet_name>

## Backends
| Backend                       | rpm  | tpm    | SLO tier | Notes |
|-------------------------------|------|--------|----------|-------|
| self-hosted us-east-1 (primary)| 2000 | 500000 | tight    | chapter-02 scaler per region |
| self-hosted us-west-2 (fallback)| 1500 | 400000 | tight    | <or external provider> |

## Fallback policy
- 5xx or provider timeout on primary → secondary attempted (<cooldown_time>s cooldown after N errors in a window)
- Rate-limit (429) on primary → secondary attempted
- Model-not-found on primary → no fallback (configuration error; surface to client)

## Per-tenant overrides (if applicable)
| Tenant / API-key tag | Allowed backends               | Reason |
|----------------------|--------------------------------|--------|
| eu-tenant            | self-hosted eu-west-1 only     | data-residency contract |
| alpha                | both self-hosted regions + Fireworks | standard SLO class |
| batch                | Fireworks primary, self-hosted fallback | loose SLO; cheaper |

## Observability
- Per-provider latency / error rate: LiteLLM Prometheus metrics (chapter-06 Alert 2).
- Per-tenant fallback rate: alert when > 5%.
- Cost-per-token per backend: compose with module 110 exercise 04.
```

### Deliverable C — the global LB wiring (or documented skip)

Set up a global LB in front of the router (or in front of the two regional LiteLLM instances if you are running one per region):

**Option 1 — Anycast / global LB (Cloudflare Load Balancing, AWS Global Accelerator, GCP Cloud LB).** Pool the two regional endpoints; configure latency-based or geo-based steering; set health checks against `/health`. Chapter 06 Section 1 walks the shape.

**Option 2 — Weighted DNS (Route 53 weighted records, Cloudflare DNS failover).** Each region has a DNS record; the global record has weighted entries. On regional failure, update the weight (or the health check removes the record).

**Option 3 — Skip the global LB.** If you only have one region's full stack (common in an exercise environment), run the router in that region and rely on LiteLLM's backend fallback for resilience. Document the skip explicitly, including the limitation (regional LB outage is unmitigated).

Attach:

- The global LB configuration (DNS record set, pool manifest, Anycast IP, or Terraform module).
- A health check verification: curl both regional endpoints; confirm global LB reaches each.
- A latency measurement: a request from your workstation through the global LB → which region did it hit? (Check response headers or add a `X-Region` header at each regional LB.)

### Deliverable D — the region-failure injection drill

Simulate a region failure and observe the behaviour. Options:

**Option 1 — real failure injection.** Scale the primary region's Deployment to 0 or cordon its nodes; the router's health check fails; traffic shifts.

**Option 2 — router-level simulation.** Mark the primary `model_name` entry as `rpm: 1` (effectively a rate-limit-at-once); the router cools it down after 1–2 requests; subsequent traffic routes to fallback.

**Option 3 — iptables injection on the router's side.** Block the primary's endpoint IP at the router pod level; `curl` from inside the router pod fails; the router sees it as unhealthy.

Script the drill:

1. **Baseline** — route sustained load from the exercise-02 load driver through the router. Confirm steady-state behaviour — most requests to primary, some to fallback (if `latency-based-routing` or `simple-shuffle`), TTFT P99 under SLO.
2. **Inject** — apply one of the three options above against the primary backend.
3. **Observe** for 2–5 minutes:
   - **Fallback rate** — LiteLLM's `litellm_fallback_requests_total` (or equivalent) spikes.
   - **Latency** — if the fallback is cross-region, TTFT rises by the RTT (30–100 ms coast-to-coast; higher for cross-provider). The SLO should still hold.
   - **Circuit breaker** — after N errors, primary is cooled; traffic goes wholly to fallback until cooldown expires.
   - **Recovery** — restore primary; after cooldown, traffic returns.

Record:

```
# Region-Failure Drill — <date>

## Setup
- Primary: <backend 1>
- Fallback: <backend 2>
- Baseline load: <RPS>
- Injection method: <scale-to-0 | iptables | router-sim>

## Timeline
t+0.0 s     Injection applied
t+<a> s     Primary health check failed (router's view)
t+<b> s     Primary cooled (circuit breaker tripped)
t+<c> s     All traffic routing to fallback
t+<d> s     Fallback latency delta: +<ms> (vs. primary baseline)
t+<e> s     Fallback 5xx rate: <percent>
t+<f> s     Injection removed
t+<g> s     Cooldown expired, primary re-attempted
t+<h> s     Primary healthy; traffic re-balances

## SLO observation
- TTFT P99 during failover: <value> (SLO: <target>) — PASS/FAIL
- 5xx rate: <value> (SLO: <target>) — PASS/FAIL
- Any client-visible errors (not 5xx-translated-to-retry-success)? <y/n>
```

Attach the LiteLLM metrics screenshot and the regional LB view across the drill window.

### Deliverable E — the cost-vs-latency routing table + cross-region observability

**Part 1 — cost-vs-latency routing table.** Instantiate chapter 06 Section 3's routing matrix for your fleet:

```
| Tenant  | Region    | Model         | Primary               | Fallback             | SLO class | Cost note |
|---------|-----------|---------------|-----------------------|----------------------|-----------|-----------|
| alpha   | us-east   | llama-3.1-70b | self-us-east          | self-us-west          | tight     | $X/1M     |
| alpha   | us-west   | llama-3.1-70b | self-us-west          | self-us-east          | tight     | $X/1M     |
| beta    | us-east   | llama-3.1-70b | fireworks             | self-us-east          | standard  | $Y/1M     |
| batch   | any       | llama-3.1-70b | fireworks (nightly)   | self-us-east          | loose     | $Y/1M     |
| eu-team | eu-west   | llama-3.1-70b | self-eu-west          | fireworks-eu         | tight (residency) | $Z/1M |
```

Reference the policy file's team-based routing feature (LiteLLM's `team_id` / `model_group` routing); link to the LiteLLM docs for the version you pinned.

**Part 2 — cross-region observability.** Chapter 06 Error 8 flags the "each region's Prometheus is independent" failure. Pick one of:

- **Thanos / Mimir / Prometheus federation.** A long-term store scrapes both regions' Prometheuses; one Grafana queries both.
- **A single central Prometheus with remote scrape.** Each regional engine `remote_write`s to a central instance.
- **LiteLLM's own observability tier** — the router's metrics live in one place; per-region metrics live separately. Acceptable short-term.

Attach the chosen approach's wiring — remote-write config, Thanos query layer, or a federation scrape. Show a Grafana panel that combines both regions' TTFT P99 on one chart.

### Deliverable F — the alert pack + `<!-- needs-escalation -->` list

Implement chapter 06's four alerts:

1. **Region imbalance** — chapter 06 Alert 1 PromQL. > 30 % sustained imbalance over 5 min → investigate LB or geo-steering.
2. **Fallback rate above expected** — chapter 06 Alert 2. > 5 % fallback rate over 5 min → investigate primary.
3. **Provider error-budget burn** — per-provider error rate exceeding its SLA → cool / investigate.
4. **Cost deviation** — expected cheap-primary vs. actual billing drift. Compose with module 110 exercise 04.

Attach the `PrometheusRule` manifest and one simulated firing (easiest: inject the same failure as Deliverable D and confirm Alert 2 fires).

Then produce `<!-- needs-escalation -->`:

- "SSE streaming semantics through our global LB — the chapter flags a Cloudflare 524 at 100 s on long streams; our LB's actual idle timeout and keepalive requirements." → cloud LB docs; test with a long-generation request.
- "External provider data-policy audit. Which tenants can route externally, which can't, per our contracts." → `ai-infra-security` level 35; legal; chapter 06 data-residency section.
- "Per-provider rate budgets. Our chosen `rpm` / `tpm` is a guess; actual provider SLA is contract-specific." → provider contract; procurement.
- "Router HA — one router instance is a single point of failure. Multi-replica + LB in front of the router." → chapter 06 "Where to deploy the router" discussion.
- "Observability unification cost. Thanos / Mimir are non-trivial installs; single-central-Prom adds remote-write bandwidth. Our platform's existing observability tier shapes the choice." → module 110 chapter 03; platform team.
- "Cross-region weight sync / model-store replication. Chapter 04 pre-stage per region requires coordinated weight updates; a drift where region A has model v2 and region B has v1 breaks fallback semantics." → module 105 model-ops; exercise 04 weight-pre-stage pattern.

## Starter guidance

- **Pick Option A or B for the backends, not both at once.** The chapter's content applies to both; the exercise is about the drill, not a full comparison matrix. Document the choice; move on.
- **Verify LiteLLM's current config schema.** The shape above is current as of the chapter; config fields rotate between releases. Cross-check against <https://docs.litellm.ai/docs/proxy/configs> and <https://docs.litellm.ai/docs/routing> for the pinned version.
- **Don't store real API keys in a ConfigMap.** Use a Secret. Reference via `os.environ/KEY_NAME` — LiteLLM reads from the proxy pod's env vars.
- **Start with `simple-shuffle` routing; upgrade later.** Latency-based routing needs latency samples; usage-based needs historical data; neither is useful day-one. `simple-shuffle` is deterministic enough to exercise fallback behaviour.
- **The drill (Deliverable D) is the acceptance test.** Without an injected failure, the router config is hypothesis; the drill is data. Make time.
- **Don't test with real customer traffic.** Even with the fallback working, injected failures produce some latency variance. Run on a staging deployment or an internal-tenant API key.
- **Record the cross-region RTT baseline.** When fallback kicks in, TTFT rises by the round-trip. If you measured it ahead of time, the drill is interpretable; if you didn't, you can't tell whether a 180 ms TTFT is "the fallback working fine" or "something else is wrong."
- **Observability wiring is often the longest step.** Thanos / Mimir installs take hours. If you're blocked here, document the intended shape in Deliverable E and ship the exercise with LiteLLM-only metrics; come back when the platform tier is ready.

## Fallbacks for constrained environments

- **Only one region's fleet.** Option B (self-hosted + external provider) works with one region; the external provider is the "second region" for the exercise. The multi-region failover drill (Deliverable D) still holds — "primary fails" vs "fallback picks up" is the same shape.
- **No external provider API keys.** Stand up a second small-model self-hosted backend in the same region; the two-backend router shape is testable even if the regional-failover analogue is weaker. Document the gap.
- **Platform team won't let you install LiteLLM in-cluster.** Run LiteLLM on a workstation or a sandbox VM; the exercise is portable. Note the production limitation.
- **Global LB unavailable.** Deliverable C's skip is acceptable; the router's own fallback is the resilience layer. Document that a regional LB outage is unmitigated.
- **Cross-region observability unavailable.** LiteLLM's metrics are one source of truth for the router's view; regional Prometheuses tell the backend story. A dual-dashboard story is acceptable short-term.
- **Shared / regulated cluster that forbids third-party proxies.** Wrap your own minimal router — a FastAPI service that does OpenAI-compatible → backend-of-choice routing with fallback and cooldown. The exercise's intent is the policy, not the specific binary; document the substitution.

## Acceptance criteria

1. **Deliverable A** stands up LiteLLM with two backends configured and healthy; a smoke-test request through the router succeeds against both backends.
2. **Deliverable B** names the fallback + circuit-breaker + rate-limit policy in a document that references the LiteLLM config fields and chapter-06 reasoning.
3. **Deliverable C** has a global LB wired (or a documented skip); health checks verify both regional endpoints; a latency / region-detection measurement is attached.
4. **Deliverable D** shows an injected region-failure drill with a timeline, SLO observation (PASS/FAIL), and a dashboard screenshot; the circuit breaker's cooldown is observable.
5. **Deliverable E** has a cost-vs-latency routing table instantiated for the fleet's tenants/classes, and a cross-region observability wiring (or a documented plan).
6. **Deliverable F** has the four-alert `PrometheusRule` manifest and one firing; the `<!-- needs-escalation -->` list names at least six items with references.
7. The drill is reproducible: anyone with the same router config and injection method should see the same timeline and SLO observation.

## Common failure modes

- **DNS failover tested only in theory.** Chapter 06 Error 1. The Route 53 record set exists; no one has ever shifted the weight. Deliverable D's drill addresses this; without it, failover is hope.
- **Router without per-tenant quotas.** Chapter 06 Error 2. One tenant exhausts the external-provider budget; all other tenants lose their fallback. Deliverable B's per-provider rate limits cover this partially; per-tenant policy is the complete fix.
- **Fallback without cooldown.** Chapter 06 Error 3. The primary flaps; the router retries without backoff; the primary never recovers from the retry storm. Verify `cooldown_time` is non-zero in Deliverable B.
- **Session affinity broken by failover.** Chapter 06 Error 4. A conversation mid-stream shifts to the other region; prefix cache lost; next request's TTFT doubles. Prefix-cache-heavy workloads need sticky routing — chapter 06 Section 1 discusses.
- **LB cuts stream mid-generation.** Chapter 06 Error 5. A long generation is longer than the LB's idle timeout. Verify timeouts in Deliverable C; set well above P99.9 generation time.
- **Model weights drift between regions.** Chapter 06 Error 6. Region A on v2; region B on v1. Failover response differs from primary response for the same prompt. Chapter 04's pre-stage per region must be coordinated across regions.
- **External provider training-on-data breach.** Chapter 06 Error 7. Tenant-A's data goes to an external provider with default training; contract violation; incident. Deliverable B's per-tenant policy must flag this; Deliverable E's EU-residency row is the tested case.
- **Multi-region without cross-region observability.** Chapter 06 Error 8. Deliverable E Part 2 addresses it; without it, incident diagnosis crosses tabs.
- **Smoke test only, no drill.** The failover is untested. Deliverable D is the test.
- **Drill without return-to-primary.** The cooldown is the second half of the loop. A drill that confirms failover but never confirms recovery hides half the behaviour.

## Keep it

The LiteLLM router (Deliverable A), the fallback policy (Deliverable B), the global LB wiring (Deliverable C), the region-failure drill record (Deliverable D), the routing table (Deliverable E), and the alert pack (Deliverable F) become the fleet's resilience-at-the-edge artefacts. This exercise closes the module — exercises 01–05 made one region's fleet correct; exercise 06 makes two regions (or self-hosted + external) correct. Module 110's SLO methodology consumes the LiteLLM metrics from Deliverable F; module 111's inference-API discipline formalises the OpenAI-compat surface the router presents; module 113's canary-and-rollout disciplines compose with the per-tenant policy from Deliverable E; module 114's build-vs-buy decisions feed off the cost-vs-latency table. When a new region or a new provider lands, re-run Deliverables A–D with the new backend; the policy document from Deliverable B is the thing that stays stable.
