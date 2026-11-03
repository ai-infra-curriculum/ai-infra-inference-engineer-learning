# Exercise 05 — Platform Charter Authoring

**Time:** 2 hours · **Output:** a complete inference platform charter against a stated organisation shape

## Goal

Author the charter artefact described in [Chapter 05](../05-platform-charter.md). Ship the version you would actually pin to your team's channel topic.

## Problem statement

You've been handed the inference platform team at a hypothetical organisation (or your real one, if you have it). Three ambiguous tickets arrived in week one:

1. A product team asks you to "own" response-quality SLOs for their chatbot ("just TTFT isn't enough — we also care about accuracy").
2. A security team asks whether the inference team or security team owns the WAF rules that reject prompt-injection patterns.
3. A finance team asks for a cost-per-conversation breakdown and would like you to produce it.

A charter would resolve all three without a week of meetings. Write it.

## Requirements

Deliver one artefact — a charter document (Markdown, Notion, Confluence, Google Doc — exportable) that contains the seven sections below. Keep each section as short as it can be while still routing a real ticket.

### Target organisation shape

If you don't have a real one, use this synthetic shape:

- One ML Engineering team shipping ~15 open-weight models per year (fine-tunes, distillations, periodic registry drops).
- Three AI Engineering product teams building chatbots, agents, and a RAG product.
- One SRE org that is explicitly separate from this platform.
- One security org with inference-edge responsibilities.
- Two cloud regions, mixed Hopper and Blackwell fleet, roadmapped to add a managed-vendor fall-back for overflow.
- A board / finance review that asks for cost per 1 M tokens and per business unit.

Adapt the sections below to this shape — or to your real org.

### Required sections

1. **Mission** — one paragraph, ≤ 5 sentences. What you run, for whom, under what constraints. A peer reading this should know whether to open a ticket with you.
2. **Services we provide** — bullet list of concrete deliverables at the level a customer would request. Minimum coverage: managed OpenAI-compatible endpoints for a curated model catalogue, streaming, multi-adapter LoRA, per-tenant rate limits, canary / shadow / blue-green rollouts, autoscaling against inference-shaped metrics, observability stack specific to inference signals.
3. **Non-services** (the "does not do" list) — a table with one row per class of problem you do not own, naming the owner. Must include at least the rows for the six peer boundaries from Chapter 01 (`ai-infra-performance`, `ai-infra-ml-platform`, `ai-infra-sre`, `ai-infra-mlops`, `ai-infra-security`, AI Engineering family consumers) plus the eval boundaries and the "legal / compliance / regulatory sign-off" and "product scope / pricing / roadmap" out-of-scope rows.
4. **SLOs we commit to** — a table with one row per SLO. Columns: SLO, target, measurement method, window. Minimum coverage: availability, TTFT P99, TPOT P99, E2E P99, cost per 1 M input / output tokens, canary rollback latency. Plus a short **SLOs we refuse** sub-section naming the model-behaviour SLOs the platform explicitly does *not* own (and the peer that does).
5. **Request-shape / protocol contract** — a short section listing what the API expects: supported endpoints, streaming semantics, structured-output guarantees, tool-calling protocol, rate-limit / backpressure / retry / idempotency / auth / tenant-isolation / request-id-propagation. Not a full OpenAPI spec — the shape of the shape.
6. **Change-management and rollout discipline** — a short section on how model-version, engine-version, hardware/firmware changes land. Must name canary strategy, SLO-based rollback, blue-green for large models, model-deprecation grace period, and who can trigger emergency rollback in how long.
7. **Peer interface contracts** — a table with one row per peer boundary. Columns: Peer, They hand us, We hand them, Escalation path. Minimum coverage: `ai-infra-performance`, `ai-infra-ml-platform`, `ai-infra-sre`, `ai-infra-security`, `ai-eval-engineer`, AI Engineering family consumers.

### Three ticket-resolution appendices

At the end of the charter, resolve the three ambiguous week-one tickets from the problem statement:

- For each ticket, state the owning role, cite the section of the charter that produces the routing call, and write the one-sentence response you would send back.
- If the charter *doesn't yet* resolve a ticket, that's a gap — write the charter edit (new row, new section) that would close it, and then apply the edit.

## Starter guidance

- Length target: the whole charter should fit in ≤ 2 pages of Markdown. If sections grow, cut adjectives before you cut rows.
- Negotiate the non-services table against peers (even in simulation). The table is only useful if peers would agree with their rows.
- SLO refusals are harder to write than SLO commitments. Spend time on the "SLOs we refuse" sub-section — this is where the charter actually earns its keep.
- Avoid aspirational language ("we will", "we aim to"). Write only what is actually true today and the explicit deprecation path for things you intend to change.

## Acceptance criteria

1. All seven sections are present.
2. The non-services table resolves at least two of the three week-one tickets without further edits.
3. The SLO table has concrete targets (numbers, not placeholders) and a measurement method per row.
4. The "SLOs we refuse" sub-section explicitly declines at least one SLO class that peers might otherwise try to assign to the platform.
5. The peer interface contracts table has a non-empty escalation path for every row.
6. The three ticket-resolution appendices each name an owner and cite a specific section.
7. A sibling-team peer who hasn't read the module can read the charter in ≤ 5 minutes and self-route a ticket from their inbox against it.

## Keep it

Pin the charter. Re-read it quarterly; re-write it after any incident post-mortem that revealed an ownership gap; re-write the SLO section every SLO review; re-write the services / non-services table after every major engine or hardware generation jump. Module 114 will come back to this and extend it into a cross-functional program document — don't try to pre-write 114 here.
