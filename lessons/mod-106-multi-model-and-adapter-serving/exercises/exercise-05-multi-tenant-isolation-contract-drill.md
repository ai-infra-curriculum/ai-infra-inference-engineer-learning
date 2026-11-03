# Exercise 05 — Multi-Tenant Isolation Contract Drill

**Time:** 4 hours · **Output:** one signed contract (per tenant), one noisy-neighbour scenario, one isolation measurement, one prefix-sharing audit, one audit log

## Goal

Make [Chapter 05](../05-multi-tenant-isolation-contract.md) concrete by **authoring a contract for at least two tenants with different SLOs**, **simulating a noisy-neighbour scenario**, **demonstrating that the isolation holds** (noisy tenant does not blow quiet tenant's P99), **running a prefix-sharing audit** against the module-104 boundary, and **producing a signed contract plus audit log**. By the end you can show, with measurements, that the contract you wrote is actually enforced.

## Problem statement

Your platform has two tenants: a gold-tier tenant with a tight TTFT SLO, and a silver-tier tenant with a looser SLO and a higher burst profile. Both are served from the same fleet; both need isolation. You will author the contract, enforce it, and prove the enforcement works under adversarial load.

## Prerequisites

- The router from exercise 03 (or an equivalent) with tenant-level admission control.
- The multi-LoRA / multi-model fleet from exercises 01 and 02 (or a minimal single-model fleet acceptable).
- A workload generator that can simulate two concurrent tenants.
- Chapter 05 of this module read.

## Requirements

Deliver five artefacts.

### Artefact A — Two tenant contracts

Author one contract document per tenant, each covering all four chapter-05 axes:

**Tenant "gold-01" — a customer-facing product with tight latency.**

```yaml
tenant_id: "gold-01"
tier: "gold"

slo_contract:
  ttft_p50_ms_target: 300
  ttft_p99_ms_target: 1_000
  tpot_p50_ms_target: 25
  availability_target: 0.999
  slo_scope: ["chat-small", "chat-large"]
  slo_promise_tier: "A"

rate_contract:
  rps: 50
  rps_burst: 100
  concurrent_requests: 20
  daily_token_quota: 10_000_000
  daily_cost_cap_usd: 500

data_boundary:
  isolation_class: "shared-pool"
  allow_prefix_sharing: false            # cross-tenant isolation on
  allow_adapter_sharing: false
  log_retention_days: 7
  log_pii_redaction: "aggressive"
  allowed_regions: ["us-east"]
  compliance_scopes: ["SOC2"]

operational_contract:
  change_notification_hours: 48
  incident_notification_minutes: 15
  rollback_sla_minutes: 10
```

**Tenant "silver-02" — a batch workload with looser latency but high burst.**

```yaml
tenant_id: "silver-02"
tier: "silver"

slo_contract:
  ttft_p50_ms_target: 1_000
  ttft_p99_ms_target: 5_000
  availability_target: 0.99
  slo_scope: ["chat-small"]
  slo_promise_tier: "B"

rate_contract:
  rps: 20
  rps_burst: 500                         # high burst
  concurrent_requests: 100
  daily_token_quota: 50_000_000
  daily_cost_cap_usd: 1_000

data_boundary:
  isolation_class: "shared-pool"
  allow_prefix_sharing: false
  log_retention_days: 30
  log_pii_redaction: "basic"

operational_contract:
  change_notification_hours: 24
  incident_notification_minutes: 60
```

Both contracts committed to version control; the shape is enforced by a JSON Schema (or Pydantic model, or equivalent).

Each contract carries a **sign-off** block (names + timestamps) representing the two sides agreeing — platform team and tenant representative. For the exercise, the sign-off is a `signed_by` field with your name + a timestamp; in production, it is a wet-signed document or a verified Git commit.

### Artefact B — Enforcement implementation

Wire the contracts into the router's admission layer:

- Both tenants' rate limits enforced via token-bucket (per-tenant RPS + burst).
- Daily token quota enforced via a counter (in-memory for exercise; Redis for production).
- Allowed-models enforced on request admission.
- `allow_prefix_sharing: false` enforced in the engine (if the engine supports per-tenant KV namespace in your pinned version; otherwise document the gap).
- Isolation class routes gold-01 and silver-02 to the same shared pool (demonstrate the shared-pool contention), with gold-01 at higher scheduler priority.

Document each enforcement with:

- Where it lives (router middleware, engine flag, log pipeline).
- A unit test that exercises the enforcement with a known-good and known-bad input.

### Artefact C — Noisy-neighbour scenario

Design a workload that stresses isolation:

- **Gold-01's baseline:** steady 20 RPS, 60-second measurement window. Record TTFT P50 / P99 and TPOT.
- **Silver-02's burst:** ramp from 20 RPS to 500 RPS over 30 seconds, hold 500 RPS for 2 minutes, ramp back. Overlapping with gold-01's steady state.

Expected: silver-02 saturates its own rate limit at 500 (if the burst limit of 500 is reached) OR is admitted up to the limit and the overflow is 429'd. Gold-01's TTFT should remain within its SLO.

Measure during the burst:

- Gold-01's TTFT P50 and P99.
- Silver-02's TTFT P50 and admitted RPS.
- Scheduler queue depth.
- Per-tenant admit / reject counts.

Compare gold-01's TTFT during the burst vs. during baseline. The isolation contract holds if the degradation is within tolerance (e.g. P99 within 1.5× baseline).

If isolation fails:

- Was it admission (silver got through more than its contract allowed)?
- Was it scheduler-level (admitted silver requests starved gold)?
- Was it some other coupling (shared KV cache evicting gold's prefixes, shared adapter warm set thrashing)?

Record the failure mode; the write-up addresses what to tighten.

### Artefact D — Prefix-sharing audit

Run a prefix-sharing audit against the module-104 boundary.

Setup:

- Pick a system prompt both tenants might plausibly use (a common Markdown header, a generic instruction preamble).
- Send a prompt from gold-01 using that system prompt, with a tenant-specific user turn that would produce a measurably different response.
- Immediately send a request from silver-02 using the same system prompt but a different user turn.
- Inspect:
  - Does the engine report a prefix-cache hit for silver-02's request? (If so, that is a cross-tenant hit — a leak if `allow_prefix_sharing: false`.)
  - Do silver-02's response logprobs show any signal of gold-01's prior content? (Hard to measure exactly; a controlled probe is: use a tenant-specific marker string in gold-01's user turn and look for it to leak into silver-02's response. This is the test the security peer would strengthen; for the exercise it's an informal check.)

Expected: with `allow_prefix_sharing: false` for both tenants and per-tenant KV namespace enforced, silver-02's request is a cache miss. If not, the data-boundary enforcement has a hole.

Document:

- The exact probe (prompts, expected observations).
- The engine's cache-hit metric for both requests.
- Whether the enforcement held.
- If it held via per-tenant namespace, show the namespace in the engine's log; if via disabling prefix caching entirely, show the flag.

### Artefact E — Audit log and sign-off

Compile the audit evidence:

- **Tenant registry history.** The two contracts committed to Git, with reviewer sign-off (even if solo, record who reviewed and when).
- **Enforcement test records.** The unit tests from Artefact B passed, with timestamps.
- **Noisy-neighbour run.** The full measurement log from Artefact C with timestamps.
- **Prefix-sharing audit.** The audit record from Artefact D.
- **Policy violation count.** Any time an enforcement check rejected during the run — tabulate: tenant, reason, timestamp.

Produce a one-page handoff document that says: "At <date> for <scope>, the contract held / did not hold on <evidence>. Open items are <list>."

If the contract did not hold on one of the axes, the document says so plainly — the honest audit is more valuable than a passing audit.

## Starter guidance

- **Pick realistic SLOs, not aspirational ones.** The exercise is more interesting if gold-01's SLO is actually achievable on your fleet; otherwise everything looks like isolation failure when really it's over-promising.
- **The two contracts should differ in multiple axes.** Different RPS, different quotas, different data-boundary (one with aggressive redaction, one basic) — exercise the per-tenant plumbing.
- **Burst limits exist to protect, not to allow unlimited bursts.** Silver-02's 500-RPS burst is the enforcement test; 500 should actually be reached, then capped.
- **Measure gold-01's TTFT at the P99 under burst.** P50 is easy to protect; P99 shows real isolation.
- **For the prefix audit, use a tenant-specific marker.** "The customer's internal project name is Rhombus" injected into gold-01's prompt, then watch silver-02 for "Rhombus" leakage.
- **Pre-commit the test outcomes you expect** (expected pass / fail) and compare to actual. This is scientific method; it keeps you honest when the result is unexpected.

## Fallbacks for constrained environments

- **Engine does not support per-tenant KV namespace.** Document the gap in the data-boundary enforcement; demonstrate the gap exists; propose the ticket to the engine upstream or author a wrapper that scopes the KV cache by request header before admission.
- **No multi-replica router.** Use in-memory counters; call out that production needs Redis.
- **No signing infrastructure.** Use a plain-text `signed_by` block for the exercise; in production, use the module-105 chapter-05 signing path extended to contracts.
- **Workload generator cannot sustain 500 RPS.** Scale the burst to what the generator can produce (and note the fleet's actual rate limit should still enforce).

## Acceptance criteria

1. **Artefact A** has two committed contracts covering all four axes with signatures.
2. **Artefact B** has each contract field wired to a specific enforcement point with a unit test.
3. **Artefact C** has the noisy-neighbour run with gold's P50/P99 during baseline and during burst; isolation outcome stated.
4. **Artefact D** has a prefix-sharing probe with the result (hit or miss, cross-tenant or not), including engine-level metric evidence.
5. **Artefact E** has the signed audit document, policy-violation count, and open items.
6. All artefacts live in version control.

## Common failure modes

- **Rate-limit enforced too late in the pipeline.** The engine still consumes capacity on the request before the router rejects. Enforce at the first router hop.
- **Burst allowance consumed and never refilled.** Token-bucket math bug. Verify the bucket actually refills.
- **Shared KV cache leaks across tenants despite flag.** The flag is cosmetic; the engine's actual implementation did not plumb it through. Verify by probing, not by reading config.
- **Operational contract not enforced.** `change_notification_hours: 48` is in the contract; the deployment framework doesn't honour it. Operational isolation often breaks here.
- **Audit log drops records under load.** Pipeline backlogs; records lost. Verify by inducing load and checking record count.
- **Priority inversion.** Gold's request waits on a silver request's work (shared tokenizer thread, shared file read). Scheduler priority alone does not fix all contention.
- **One-tenant workload leaks into other-tenant's logs.** Logging middleware doesn't scope cleanly. Verify log scope by probing.

## Keep it

The contract template, enforcement scaffolding, noisy-neighbour harness, and audit document template are reusable for every future tenant onboarding and feed module 110's per-tenant SLO dashboards and module 113's tenant-aware rollout policy. The failure modes feed the inference-incident runbook library.
