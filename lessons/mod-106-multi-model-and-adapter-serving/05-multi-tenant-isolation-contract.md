# Chapter 05 — Multi-Tenant Isolation Contract

> "Multi-tenant" is not a boolean property of a system. It is a contract — written down, signed off, measured, and reviewed when the contract bends. This chapter is how to write the contract and the enforcement.

## Motivation

An inference platform that serves two teams has one tenancy problem. One that serves two hundred customers under different contracts has a dozen. The questions a platform must answer:

- If tenant A sends a traffic spike, does tenant B's P99 latency survive?
- If tenant A hits its quota, does tenant B still get its full RPS?
- Can tenant A's prompt bleed into tenant B's response via prefix caching?
- If tenant A's workload runs an adapter containing proprietary IP, can tenant B see it?
- When tenant A's workload degrades, which specific control did the ops team pull — and did they violate anyone else's SLO doing it?

The multi-tenant isolation contract is where these answers are *written down* and the enforcement is where they are *measured*. "We're multi-tenant" without a contract is marketing; with a contract, it is engineering.

By the end of this chapter you can:

- Enumerate the four axes of a tenant isolation contract (SLO, token-rate, data boundary, operational).
- Author per-tenant SLO targets and defend them against the shared fleet's capacity.
- Design noisy-neighbour mitigation at admission (priority classes, fair-share scheduling, token-bucket) and at execution (preemption rules, per-tenant iteration budget).
- Compose the data-boundary clause with module 104's cross-tenant KV sharing policy — including the opt-out and audit requirements.
- Author the audit trail that proves the contract holds.

The contract shape authored here is deliberately implementation-agnostic — concrete fleet topologies (per-tenant dedicated fleets, shared pools with admission control, or hybrid) vary by platform. Adapt the enforcement loci to your fleet.

## The four axes

A tenant isolation contract covers four axes. A contract that only addresses one is incomplete.

### Axis 1 — SLO contract

What latency / availability does the tenant get? What is the measurement window and the error-budget semantics?

Minimum fields per tenant:

```yaml
slo_contract:
  ttft_p50_ms_target: 300
  ttft_p99_ms_target: 1_200
  tpot_p50_ms_target: 25
  tpot_p99_ms_target: 60
  availability_target: 0.999        # over 30-day rolling window
  slo_measurement_window: "30d"
  error_budget_policy: "burn_rate_alert_at_2x"
  slo_scope: "chat-small, chat-large"  # which model aliases
  slo_promise_tier: "A"                # quality tier promised (see chapter 03)
```

The SLO is **model-conditional** — a tenant's SLO on `chat-small` differs from their SLO on `chat-large`. The scope field names which aliases the contract covers.

**Important:** SLOs are over *achieved* latency on *admitted* requests, not on requests the admission layer declined. A rate-limited tenant that gets 429s is not having its latency SLO violated; they are experiencing their rate-limit contract. Distinguish the two.

### Axis 2 — Token-rate contract

What is the tenant allowed to consume, measured in tokens / requests / dollars?

```yaml
rate_contract:
  rps: 50                      # steady requests per second
  rps_burst: 200               # token-bucket depth
  concurrent_requests: 20
  input_tokens_per_second: 50_000
  output_tokens_per_second: 20_000
  daily_token_quota: 20_000_000
  daily_cost_cap_usd: 500
  per_request_max_output_tokens: 4_096
  per_request_max_context_tokens: 32_000
```

Enforced at admission (chapter 03). Note the asymmetry: input-token rate is typically much higher than output-token rate, because output is where the compute actually lives.

### Axis 3 — Data-boundary contract

What can and cannot cross between tenants? Composed with module 104's cross-tenant KV sharing chapter.

```yaml
data_boundary:
  isolation_class: "shared-pool"       # dedicated | shared-pool | shared-fleet
  allow_prefix_sharing: false          # module 104 cross-tenant KV sharing
  allow_adapter_sharing: false         # tenant's adapters not visible to others
  log_retention_days: 30
  log_pii_redaction: "aggressive"      # none | basic | aggressive
  prompt_retention: "none"             # none | debug_sample | full
  allowed_regions: ["us-east", "us-west"]
  disallowed_providers: ["vendor_X"]   # chapter 03 fallback restrictions
  data_residency: "US"
  compliance_scopes: ["SOC2", "HIPAA-covered-partial"]
```

Each field is enforceable in a specific place:

- `allow_prefix_sharing: false` → the KV cache keyed by tenant (module 104 chapter 06).
- `allow_adapter_sharing: false` → the adapter registry (chapter 01) tags each adapter with its tenant; the router rejects requests that reference an adapter the requesting tenant does not own.
- `allowed_regions` → the router (chapter 03) does not route to backends in other regions.
- `disallowed_providers` → chapter 03 fallback skip list.
- `prompt_retention` → logging middleware drops / samples / retains prompts per the setting.

### Axis 4 — Operational contract

What does the ops team promise about change management, incident communication, and lifecycle?

```yaml
operational_contract:
  change_notification_hours: 48       # breaking changes announced in advance
  incident_notification_minutes: 15   # SEV1 within 15 min
  maintenance_windows: "sat 02:00-04:00 UTC"
  deprecation_notice_days: 90         # model deprecation notice
  rollback_sla_minutes: 10            # how fast a bad rollout can be undone
  support_contact: "ops-team@...; pagerduty:ops-infra-inference"
  on_call_response_minutes: 15
```

Operational isolation is often the one that bites first in a real incident — a tenant finds out their SLO is violated when their SEV1 goes unanswered for an hour because the on-call ops person was handling another tenant's incident.

## Per-tenant SLO targets — how to pick them

The SLO is not aspirational; it is a commitment the fleet must defend. Pick numbers you can defend from capacity.

### Working backwards from capacity

Fleet capacity (requests the fleet can handle at target TTFT P99) is set by module 103's scheduler + module 104's KV budget + chapter 02's model co-location. A given fleet shape has an admission limit beyond which the P99 degrades.

The sum of all tenants' RPS contracts must fit within the fleet's admission limit, *with headroom for burst*:

```
sum(tenant_RPS_contract × fleet_share) ≤ fleet_admission_limit × (1 - burst_reserve)
```

If you sell more capacity than you have, the first tenant whose SLO is violated is the signal that the contracts need re-pricing.

### Tier SLOs by tenant class

A pragmatic structure:

- **Gold tenants:** TTFT P99 ≤ 1 s, dedicated capacity, their quota is reserved.
- **Silver tenants:** TTFT P99 ≤ 2 s, shared pool, best-effort beyond quota.
- **Bronze tenants:** TTFT P99 ≤ 5 s, lowest priority, bursts via rate-limit shaping.

The tier structure maps to the admission-priority structure (next section).

### SLO as a measured object, not a promise

The platform's measurement infrastructure (module 110) must emit per-tenant TTFT, TPOT, E2E, availability. If the fleet's P99 is good but tenant X's P99 is bad, that is a tenant-specific signal — adapter cold-start, routing issue, provider-fallback latency. Only per-tenant visibility surfaces this.

## Noisy-neighbour mitigation

The core failure to prevent: tenant A's workload degrades tenant B's P99. Three layers of mitigation.

### Layer 1 — Admission priority

The router (chapter 03) implements per-tenant admission with **priority classes**:

```python
def admit(request):
    tenant = tenant_registry[request.tenant_id]
    priority = tenant.priority_class       # gold=0, silver=1, bronze=2

    if backend_queue_depth(target_backend) > thresholds[priority]:
        if priority == 0:
            # Gold: pass to backend, backend scheduler will prioritise.
            return admit_with_priority(request, priority)
        elif priority == 1:
            # Silver: shed load if backend is hot.
            return reject(429, backoff_hint=10_s)
        else:
            # Bronze: aggressively shed.
            return reject(429, backoff_hint=60_s)

    return admit_with_priority(request, priority)
```

Admission is the strongest noisy-neighbour mitigation: a bronze tenant's spike never reaches the backend when gold tenants are near capacity.

### Layer 2 — Scheduler-level priority

The engine's scheduler (module 103) can accept priority on each request. Where the scheduler supports it:

- Gold requests get preferred scheduling (admitted into the running batch sooner).
- Silver and bronze take fair-share of what gold leaves.
- Preemption prefers lower-priority requests (preempt bronze before silver before gold).

vLLM, SGLang, and TRT-LLM each have different levels of priority support — some have explicit priority queues; some only have FIFO. vLLM exposes `--scheduling-policy priority` plus a per-request integer `priority` field (lower = earlier; under KV pressure the lowest-priority running request is preempted) ([vLLM scheduler config](https://docs.vllm.ai/en/stable/configuration/engine_args/)). SGLang exposes `--enable-priority-scheduling`, `--schedule-policy priority`, and a per-request `priority` field with configurable preemption threshold ([SGLang server args](https://docs.sglang.io/docs/advanced_features/server_arguments)). TRT-LLM's inflight batcher accepts a priority on each request through its C++ / Python runtime API. Flags and defaults rotate between releases — **verify your engine's support surface before promising priority-level SLOs.**

### Layer 3 — Fair-share and token-bucket

Even within a priority class, one tenant with a sustained workload should not drown another in the same class. Fair-share scheduling at the admission layer:

- Per-tenant token-bucket (RPS + burst depth).
- Per-tenant concurrent-request cap.
- Per-tenant input-tokens-per-second cap (an aggressive-RAG tenant can push 100× the token volume of a chat tenant at the same RPS).

Deficit-round-robin or weighted-fair-queueing (the classical networking primitives) are the right shapes. Standard implementations exist (Envoy ratelimit filters, local token-bucket libraries); the choice is operational, not conceptual.

### Dedicated-fleet escape hatch

For gold tenants with the strictest SLOs, the mitigation of last resort is a **dedicated fleet** — their requests do not share workers with any other tenant. The isolation is physical, not scheduling-level. The cost is higher (lower utilisation on a dedicated fleet).

The pattern: silver and bronze share a large shared pool; gold runs on dedicated-small pools; the router (chapter 03) uses the tenant's `isolation_class` to pick.

## Data-boundary enforcement

The data-boundary axis composes with module 104's cross-tenant KV sharing. Three concrete controls.

### Control 1 — KV namespace per tenant

The engine's prefix cache is keyed by `(tenant_id, prefix_hash)` rather than just `prefix_hash`. Two tenants with identical system prompts never share a KV block. This is a direct cost — the shared system-prompt KV benefit is lost. But it is also the cleanest isolation.

Module 104 chapter 06 covers how specific engines implement this: both vLLM and SGLang implement the opt-out through a per-request `cache_salt` string that is mixed into the prefix-cache hash so entries with different salts never share a block ([vLLM security guide](https://docs.vllm.ai/en/stable/usage/security/), [SGLang radix cache](https://docs.sglang.io/docs/advanced_features/cache)). The vLLM docs explicitly caveat that this is a timing-side-channel mitigation, not a hard tenant boundary within a shared process — for strict isolation, run a dedicated instance per tenant. SGLang has a known gap where `cache_salt` is ignored on LMCache external tiers ([sgl-project/sglang#37160](https://github.com/sgl-project/sglang/issues/37160)). **Verify current support — and salt plumbing through any cache tier you depend on — against your pinned engine.**

### Control 2 — Adapter visibility

Each adapter in the registry (chapter 01) is tagged with its owning tenant. The router validates `request.adapter_id`'s tenant matches `request.tenant_id` before accepting. A tenant cannot reference another tenant's adapter even if they know the ID.

**Edge case:** a *platform-owned* adapter (shared by all tenants — e.g., a safety classifier) is tagged `tenant: platform` and is visible to all. The policy encodes which tenant IDs can see `platform`.

### Control 3 — Logging and redaction

Prompts and completions are logged (for debugging, auditing, model improvement). Each log row carries the tenant ID; retention and redaction follow the tenant's contract.

Three retention classes:

- **None:** prompt and completion dropped at response time; only non-content metadata (token counts, timing) retained.
- **Debug sample:** 0.1 % random sample retained for 7 days; otherwise dropped.
- **Full:** retained with PII-redaction per the contract's redaction level, for the contract's retention window.

PII redaction operates at ingress or at log-write time. Standard PII patterns (email, phone, SSN, credit-card) are detected and masked. **Model-generated completions can also contain PII**; the redaction must apply to completions, not just prompts.

The data-boundary contract gates each of these controls, and the enforcement is at the router (admission-time stamp of tenant), the engine (prefix-cache namespace), the adapter registry (visibility), and the logging pipeline (retention and redaction).

## The audit trail

A contract that is not auditable is not a contract. The audit trail must support three questions:

1. **Did the contract hold at a point in time?** For any 5-minute window, can you demonstrate that tenant X's SLO was met, their quota was enforced, their data boundary was respected?
2. **Was the contract violated — and when, where, and why?** When tenant X's TTFT P99 breached, what was the fleet state (concurrent tenants, their workloads), what admission decisions were made, what backend was serving?
3. **Who changed what?** Contract edits (tenant promoted from bronze to silver, data boundary tightened) must be versioned and attributed. The registry is the system of record; it must be in git with reviewers.

### Required audit artefacts

- **Tenant registry history.** Every change to the registry is a commit with author, timestamp, reason.
- **Per-tenant metrics (module 110).** Rolling counters and histograms per tenant for TTFT, TPOT, E2E, availability, rate-limit rejections, cost, adapter-cold-start incidence.
- **Admission decisions.** Log of every admission (admitted, delayed, rejected) with timestamp, tenant, model, backend, reason. Sampled or full depending on volume.
- **Policy violations.** Any time an enforcement check rejects (cross-tenant adapter reference, cross-tenant prefix cache hit at a tenant with sharing disabled, data-residency violation attempt), log a violation record and alert.
- **Change control.** Every router config, every tenant registry change, every engine config change touches a Git repository with CI validation and reviewer sign-off.

### A worked example — tenant SEV1

Tenant X alerts the ops team: "Our TTFT P99 doubled over the last 30 minutes."

The ops team walks the audit trail:

1. Pull tenant X's TTFT P99 timeseries. Confirm the breach timing.
2. Pull the admission log for tenant X. Confirm they were admitted (not rate-limited).
3. Pull the backend assignment log for tenant X's requests. See which backends served them.
4. For those backends, pull overall load. See what other tenants were active. Did a noisy neighbour spike?
5. If a specific tenant was identified: inspect their traffic signature. Is it against contract? Should their rate-limit catch it?
6. If the fleet was under-provisioned: inspect the admission policy. Should bronze have been shed?
7. If the engine's scheduler was behind: inspect scheduler metrics (module 103). Is the preemption rate elevated? Is this an engine issue the module-110 SLO burn-rate alert should have caught earlier?

Each step consumes a specific audit artefact. The point of writing them down is that the SEV1 can be investigated in minutes, not days.

## Common failure modes

### Contract written, never enforced

The tenant registry has a `daily_cost_cap_usd: 500` field; the enforcement never shipped. Tenant blows the cap by 10× before anyone notices. Fix: every contract field must have a tested enforcement path or be explicitly marked `advisory_only`.

### Priority inversion through adapter cold-start

A gold tenant's request waits on an adapter cold-start triggered by a bronze tenant's request a second earlier. Gold's TTFT suffers for bronze's work. Fix: priority-aware adapter loading (pre-empt bronze's cold-start when gold arrives); priority-aware eviction (don't evict gold-pinned adapters for bronze).

### Prefix cache leak via common system prompt

Two tenants share nothing except a Markdown system prompt they both happen to use. If prefix sharing is on, tenant B's response to the system prompt may be served from tenant A's KV. The attack is subtle (requires sampling at low temperature + probing specific logits) but documented. Fix: per-tenant KV namespace by default; opt-in to sharing within a tenant pool (not across pools).

### Logging retention violation

Tenant's contract says 7 days; the pipeline retains 30 by default. On an audit, violation surfaces. Fix: per-tenant retention gate at the storage pipeline, enforced at write-time, not scrubbed later.

### Dedicated fleet idling

A gold tenant has a dedicated fleet with 10× their steady traffic; the fleet idles most of the day. Cost blows out. Fix: autoscale the dedicated fleet (module 109), scale-to-near-zero if the contract allows, with cold-start accounting on scale-up.

### Operational isolation break on an incident

Ops is dealing with tenant A's SEV1; tenant B's SEV1 queues behind. B's SLO on incident-response is violated. Fix: on-call rotation sizing against incident load; a backup on-call during known-busy windows; the contract's `on_call_response_minutes` must be achievable in the organisation's incident-response capacity, not aspirational.

### SLO measured on degraded traffic

Tenant X is being heavily rate-limited; 50 % of their requests are 429s. The P99 TTFT on the 50 % that succeed is fine — the SLO shows green. The tenant perceives red. Fix: distinguish admitted-request SLO from availability SLO; alert on availability (admit rate) as well as latency.

## Composing with other modules

- **Module 103 (scheduler) — admission priority:** the scheduler respects per-request priority tags set by this chapter's admission layer. Without engine support, priority is advisory.
- **Module 104 (KV cache) — data boundary:** `allow_prefix_sharing` is the chapter-05-to-104 handshake.
- **Chapter 03 (router) — enforcement locus:** every contract field is enforced somewhere the router controls (admission, backend selection, response middleware).
- **Chapter 04 (adapter loader) — pin set and residency class:** gold tenants' critical adapters are pinned; silver's are hot; bronze's can live cold.
- **Module 110 (observability) — audit trail:** per-tenant metrics and the SLO dashboards that read them.
- **Module 113 (rollouts) — change notification:** operational contract's `change_notification_hours` means the deployment framework respects a quiet window per tenant.

## The boundary to the security peer

This chapter authors the **mechanism** of multi-tenant isolation. It does not author adversarial defence:

- Prompt injection designed to extract another tenant's data from a shared cache → security peer at [`ai-infra-security`](https://github.com/ai-infra-curriculum/ai-infra-security-learning).
- Model-weight exfiltration via adapter upload → security peer.
- Timing side-channel attacks against the KV cache for cross-tenant inference → security peer.
- Compliance frameworks (SOC2 / HIPAA / FedRAMP / EU AI Act) — the data-residency and audit requirements are shaped by the compliance peers; this chapter implements the mechanisms they require.

Rule of thumb: *this chapter writes the policy and enforces it mechanically. The security peer stresses the enforcement adversarially and the compliance peers set the policy requirements. The three disciplines meet at the contract document.*

## Summary

- Multi-tenant isolation is a contract across four axes: SLO, token-rate, data-boundary, operational. Each axis has specific enforcement loci.
- SLO targets are picked by working backwards from fleet capacity; tiers (gold / silver / bronze) map to admission priority and dedicated-vs-shared pool choices.
- Noisy-neighbour mitigation is layered: admission priority (shed low-priority on hot fleet), scheduler priority (where supported), fair-share and token-bucket at admission, dedicated fleets as the escape hatch.
- Data-boundary enforcement is per-tenant KV namespace (composes with module 104), adapter visibility (chapter 01's registry), and logging retention / redaction (per-tenant pipeline).
- The audit trail — tenant registry history, per-tenant metrics, admission log, policy-violation log, change-control history — is what makes the contract defensible on an incident or audit.
- Common failures are contract-written-but-never-enforced, priority inversion, prefix leaks via shared system prompts, logging over-retention, dedicated-fleet idling, and operational isolation break during incidents.
- The security peer owns adversarial defence; this chapter owns the mechanism and the policy enforcement.
