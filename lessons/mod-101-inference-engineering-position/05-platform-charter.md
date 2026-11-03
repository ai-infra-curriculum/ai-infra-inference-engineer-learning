# Chapter 05 — Authoring the Inference Platform Charter

> The charter is the artefact that routes incoming requests to the right owner. Without it, your team is the queue of last resort.

## Motivation

A platform without a charter accumulates work by default. Every unowned problem lands on your on-call rotation; every slightly-ambiguous ticket gets reassigned to "the inference team." The charter is the one-page contract that turns ambiguous ownership into a routing table — the same way chapter 01 turned ambiguous role boundaries into a diagram.

A good charter is **short**, **actionable at a Slack-ticket level**, and **negotiated with the peer tracks it defers to**. A bad charter is a mission statement nobody reads.

This chapter walks through the sections of a charter you'll author in exercise 05, with the opinions that let the sections resolve real tickets.

## What a charter is for

Three audiences, three uses:

1. **The platform team itself** — a shared mental model of what the team ships, what it defers, and what it refuses. Reduces internal re-litigation of scope.
2. **Peer teams** — a reference for "whose problem is this?" conversations in incident channels. The faster peers can self-serve this answer, the fewer pings land in your DMs.
3. **Consumer teams (the AI Engineering family at level 25)** — a contract that tells application teams what the platform guarantees (SLAs, API surface, cost model, rollback discipline) and what it does *not* guarantee (prompt quality, agent reliability, business policy).

## The seven sections

A charter that works in the wild has roughly these seven sections. Pick the names you want; keep the content.

### Section 1 — Mission (one paragraph, ≤ 5 sentences)

What you run, for whom, under what constraints. Example shape (adapt to your organisation):

> The Inference Platform runs the production GPU serving fleet for all models consumed by [this organisation's AI products]. We provide an OpenAI-compatible API surface over a managed fleet of inference engines, with stated SLOs for TTFT, TPOT, and availability, and a defensible cost model per 1 M tokens. We do not design applications, train models, or decide which models are permitted; we deliver the serving plane those decisions assume.

Signal of a good mission: a peer team member reads it once and can tell whether to open a ticket with you or with someone else. If they can't, rewrite.

### Section 2 — Services we provide

The concrete deliverables, named at the level a customer would actually request. Example rows (choose yours):

- Managed OpenAI-compatible endpoints for a curated model catalogue (chat, completions, embeddings, structured output, tool-calling)
- Streaming responses with documented back-pressure and client-disconnect semantics
- Multi-adapter LoRA serving against a shared base model
- Per-tenant rate limits, token budgets, request-id propagation
- Canary / shadow / blue-green rollout of model and engine version changes with SLO-based auto-rollback
- Fleet autoscaling against inference-shaped custom metrics (queue depth, prefill-tokens-per-second)
- Observability stack specific to inference signals (TTFT / TPOT / ITL / cache-hit / preemption / cost-per-1M-tokens)

If a service doesn't appear here, you don't ship it — even if a peer thinks you do.

### Section 3 — Explicit non-services (the "does not do" list)

The most useful part of the whole document. For each row, name the owning peer so routing is one-ping-away:

| We do not provide | Owner |
|---|---|
| CUDA / fused-kernel implementation | `ai-infra-performance` (peer, L35) |
| Model training, fine-tuning, or distillation | `fine-tuning-engineer` (peer, L30, ML Engineering family) |
| Model artefact registry, lineage, or signing | `ai-infra-ml-platform` (peer, L30) |
| Application-layer eval decisions (go/no-go on model behaviour) | `ai-eval-engineer` (peer, L30, AI Engineering family) |
| Benchmark / accuracy methodology for model releases | `model-evaluation-engineer` (peer, L30, ML Engineering family) |
| Prompt design, agent-loop logic, RAG pipeline construction | AI Engineering family (consumers, L25) |
| Adversarial defence at the API edge (prompt injection, model extraction, weight supply-chain) | `ai-infra-security` (peer, L35) |
| SRE methodology, incident-command discipline, error-budget math | `ai-infra-sre` (peer, L30) |
| Generic CI/CD plumbing (GitHub Actions, Argo, Flux) | `ai-infra-mlops` (peer, L25) |
| Kubernetes / Terraform / cloud base platform | `ai-infra-engineer` (prerequisite, L20) |
| Legal / compliance / data-residency / regulatory sign-off | out of scope — AI Governance family |
| Business decisions about product scope, pricing, roadmap | out of scope — AI Startup family |

This table is the one you will link in your Slack channel topic. Peers reading it should be able to self-route 90 % of ambiguous tickets without pinging you.

### Section 4 — SLOs we commit to (and the ones we refuse)

Published SLOs are the backbone of the charter. Each one should have a target, a measurement method, and a window. Example rows (your targets are your targets):

- **Availability** — successful-response ratio over a rolling 30-day window, per endpoint
- **TTFT** — P99 for a stated prompt-length class, per model class
- **TPOT** — P99 for a stated model class
- **E2E latency** — P99 for a stated prompt + output length class, per model class
- **Cost per 1 M input tokens** / **cost per 1 M output tokens** — rolling target per model class
- **Canary rollback latency** — time from auto-detected regression to full rollback

SLOs we **explicitly refuse** (because they belong to a peer):

- Model-behaviour quality SLOs (truthfulness, hallucination rate) — eval-engineer peers
- Application-level task-completion SLOs — AI Engineering family consumers
- End-user business metrics (conversion, retention) — product team

Refusals matter. A platform that silently absorbs a peer's SLOs ends up owning problems it cannot fix.

### Section 5 — Request-shape and protocol contract

What the API expects and what it refuses, at the level a client developer can code against:

- Supported endpoints (chat/completions, completions, embeddings, …)
- Streaming semantics (SSE primary; chunk boundaries; client-disconnect handling)
- Structured-output guarantees (JSON Schema, grammar, Pydantic)
- Tool-calling protocol (OpenAI-shape; parallel tool calls; streaming during tool-call phases)
- Rate-limit and backpressure headers; retry semantics; idempotency keys
- Authentication model and tenant isolation at ingress
- Request-id propagation for distributed tracing

This section is a *contract*. Changes to it go through a deprecation window published in the charter.

### Section 6 — Change-management and rollout discipline

How changes land on the serving plane. This is the section peer teams and consumers lean on to trust the platform:

- **Model-version changes** — canary ramp against the stated SLOs, shadow-traffic diffing where feasible, blue-green for large models with dual-load constraints
- **Engine-version changes** (vLLM / SGLang / TRT-LLM upgrades) — canary + SLO gate; always with a prepared rollback runbook
- **Hardware/firmware changes** (driver, DCGM, GPU operator) — maintenance window + canary
- **Deprecation of a served model** — published grace period, consumer migration guide, decommission timeline
- **Emergency rollback** — who can trigger it and in how long (minutes, not hours)

### Section 7 — Peer interface contracts

A row per peer boundary with (a) what they hand you, (b) what you hand them, (c) escalation path. Example:

- `ai-infra-performance` — hands you: engine releases with new kernels, performance regression analyses. You hand them: production repros of kernel-level regressions, SLO comparisons across engine versions. Escalation: joint on-call bridge during regressions.
- `ai-infra-ml-platform` — hands you: signed model artefacts with resource-profile metadata. You hand them: deployment feedback (did the artefact serve under SLO), post-deployment resource profiles. Escalation: registry-side owner.
- `ai-infra-sre` — hands you: SRE methodology, incident templates, error-budget framework. You hand them: inference-signal SLIs plugged into the common framework. Escalation: shared incident commander on cross-cutting outages.
- `ai-infra-security` — hands you: WAF rules, weight-signing policy, enclave requirements. You hand them: traffic shape, admission logs, attack-surface diagrams. Escalation: joint on-call bridge during security incidents touching serving.
- `ai-eval-engineer` — hands you: a release-gate verdict (eval pass/fail with evidence). You hand them: the serving-level canary that respects the verdict. Escalation: shared release channel.
- AI Engineering family consumers — hand you: traffic shape, latency budget, model-mix requirements. You hand them: endpoints with documented SLAs, cost telemetry, incident status. Escalation: a shared on-call rota for consumer-impacting incidents.

## Anti-patterns to catch early

Charters fail in predictable ways. Watch for:

- **Mission creep by Slack** — "we'll just help with the prompt this once" turns into informal ownership of prompt engineering. Keep the non-service list sharp.
- **Hidden ownership of adjacent problems** — "we'll monitor eval metrics for them" turns into owning model-behaviour SLOs. Refuse the SLOs you cannot fix.
- **Silent SLA inflation** — a consumer in a design doc writes "we assume 50 ms P99 TTFT" that was never agreed. The charter is where the real numbers live; the consumer's design doc isn't.
- **One-way deferral** — the charter says "we defer kernel work to performance peer" but the escalation path isn't wired. A deferral without an escalation path is a wish, not a contract.
- **Monolith charter** — one 20-page document nobody reads. Keep each section to the length of a Slack-pinned post.

## When to update it

A charter is a living document; the cadence matters:

- **On every peer-boundary shift** — a peer team splits, merges, re-charters, or changes owner → re-read the row for that peer and update.
- **On every incident post-mortem that touches ownership** — if the incident revealed that an ownership edge is unclear, the fix is a charter edit, not just a runbook edit.
- **On every quarterly SLO review** — the SLO section is the one most likely to drift.
- **On every model-platform generation jump** — Blackwell lands, Dynamo P/D-disaggregation goes GA, FP4 becomes a serving default → the services and non-services lists need a sweep.

## Summary

- The charter routes work. It is a short, actionable, peer-negotiated contract — not a mission statement.
- The seven sections (mission, services, non-services, SLOs, protocol contract, change management, peer interfaces) each resolve a distinct kind of ambiguous ticket.
- The non-services table is the most load-bearing part: it names the owner for every ambiguous problem class.
- SLO refusals are as important as SLO commitments. A platform cannot own metrics it cannot fix.
- Charters decay. Treat them as a quarterly artefact, not a founding document.

With the charter authored, you've got the context the rest of this curriculum assumes. Module 102 opens the first engine's source.
