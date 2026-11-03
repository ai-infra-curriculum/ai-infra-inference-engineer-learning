# Exercise 04 — Performance Peer Interface Contract

**Time:** 4 hours · **Output:** a signed interface contract with the performance peer, including ticket template, response SLA table, return-shape taxonomy, kernel-regression ledger, a worked example, and three filed or dry-run tickets.

## Goal

Make [Chapter 04](../04-performance-peer-interface.md) concrete. Draft the written contract that lets the inference engineer and the performance peer move at full speed without stepping on each other. The exercise is complete when both tracks could operate against the contract tomorrow — ticket templates in place, SLAs agreed, ledger initialised, escalation path documented.

## Problem statement

You own the inference platform; a peer (real or simulated, on the performance / kernel-craft track) owns the kernel source files and the deep performance work. You have shared surface across engine configuration, attention backend selection, quantisation deployment, speculative decoding, and distributed-inference topologies.

Draft and (dry-)sign the interface contract that covers: what each side owns, what the shared 20 % is, how tickets get filed, what response times are expected, what return shapes count as complete, how known kernel-level regressions get tracked, and how P0 incidents get co-owned.

## Prerequisites

- Chapter 04 read; chapter 05 skimmed (same contract shape for other peers).
- Module 102 (engine internals) and module 107 (distributed inference) completed — this exercise references both as the surface the interface covers.
- Module 113 chapters 03 and 04 read — the runbook set and the incident-RCA methodology the interface references.
- Access to your organisation's issue tracker (or a plausible stand-in: a `tickets/` directory in your ops repo, a GitHub repo issues tab, a Linear project).

## Requirements

Seven deliverables.

### Deliverable A — the boundary statement

A one-page document that answers:

- What does the inference-platform role own (the production-serving system — engine configuration, KV allocator parameters as a runtime object, fleet shape, SLO contract, deploy rituals)?
- What does the performance peer own (kernel source files, fused-kernel implementations, attention-backend selection logic *inside* the engine, quantisation algorithms, Nsight / SASS profiling, torch.compile pass pipelines, kernel-level regression benchmarks)?
- What is the shared 20 % (which engine flag to flip on a regression, quantisation deployment, speculative decoding, distributed-inference topology)?

End with the one-sentence boundary rule from chapter 04.

### Deliverable B — the ticket template

The ticket template filed from this track to the performance peer. Chapter 04 lists the required fields: `title`, `workload_class`, `axis`, `slo_impact`, `engine_and_version`, `model_and_version`, `hardware_class`, `topology`, `reproduction`, `expected_return`, `priority`, `sla`. Plus the optional but recommended: `metric_overlays`, `trace_sample`, `prior_runbook`, `related_regressions`. Plus the required-before-filing `rule_outs_performed` checklist.

Deliver both the template (as a reusable Markdown / JSON / issue-tracker template) and a one-sentence prose explanation of each field.

### Deliverable C — the response SLA table

A negotiated SLA table by priority: P0 / P1 / P2 / P3. Each row has ack time, opinion-returned time, resolution target. Mapped to SLO burn rate from module 110. Include an explicit note on how the SLA relates to your organisation's existing severity map (which it probably already has; this table extends, does not replace).

### Deliverable D — the return-shape taxonomy

The five return shapes from chapter 04 (diagnostic opinion, kernel patch, engine config suggestion, upstream engine ticket, "not a kernel issue"). For each, one paragraph on:

- What the shape looks like concretely (what artefacts the performance peer delivers).
- What the inference engineer does on receipt.
- What the ledger entry records (Deliverable E's shape).

Plus one paragraph on the reverse direction: what reverse-tickets look like (trace capture, canary slot, SLO data slice, fleet-shape change proposal) and how this track responds.

### Deliverable E — the kernel-regression ledger

Initialise the ledger. Required fields per chapter 04: `id`, `title`, `first_seen`, `status`, `workaround`, `upstream_ticket`, `fix_commit`, `regression_fixture`, `owner`, `last_reviewed`. Populate with at least three entries — either from real history if available, or plausible ones derived from module 102 / 107 / 113 chapter-03 runbook-named failure modes.

State the quarterly review cadence and where it slots into chapter 01's cadence.

### Deliverable F — a worked example

Author one full ticket cycle from first symptom through ledger entry. Chapter 04's worked example (FlashAttention variant regression) is the template; yours should use a plausible symptom from *your* fleet (even if simulated).

The worked example includes:

- The initial on-call symptom (axis-set alert from module 110).
- The runbook step that pointed at kernel territory.
- The `rule_outs_performed` list that was exercised.
- The filed ticket (fully populated from Deliverable B).
- The performance-peer response (one of the five return shapes from Deliverable D).
- The follow-up deploy on the inference side.
- The ledger entry (fully populated from Deliverable E).

### Deliverable G — three live or dry-run tickets

Pick three realistic tickets your team would file (two from this role to the peer; one reverse from the peer to this role). File them in the real issue tracker (if a peer is available) or create them in a `tickets/` directory as mock issues (if simulated). Each uses the full template from Deliverable B.

If a real peer is involved, record the response SLAs they met or missed; if simulated, write the expected responses and note the simulation.

## Starter guidance

- **The ticket template matters more than the tool.** A Markdown file that has all the fields is more useful than a Jira project that has none of them.
- **The `rule_outs_performed` list is the whole point.** It is the mechanism that keeps non-kernel tickets off the peer's queue; invest in making it a checklist, not freeform text.
- **SLA negotiation is a conversation.** If you can, have the negotiation with a real peer. If not, write the SLA you believe is fair and note it would be negotiated.
- **Ledger entries that are plausible-hypothetical are fine.** The ledger discipline is what matters, not the authenticity of specific regressions.
- **The worked example is the best teaching tool.** Spend more time here than on any other deliverable; a reader who can't follow the worked example won't use the contract.

## Fallbacks for constrained environments

- **No performance peer:** write the contract as if addressing one; mark explicitly that sign-off is a TODO; have another engineer (or yourself after a week) review as the stand-in.
- **No issue tracker:** use a `tickets/` directory in your repo with one Markdown file per ticket.
- **No real incident to draw the worked example from:** construct a plausible one using the module 113 chapter 03 runbook top-ten as the symptom catalog.

## Acceptance criteria

1. **Deliverable A** has a clear boundary statement, lists what each side owns, and identifies the shared 20 %.
2. **Deliverable B** has all required and recommended fields, each explained; the `rule_outs_performed` checklist is explicit.
3. **Deliverable C** has P0–P3 rows, each with ack / opinion / resolution times.
4. **Deliverable D** covers all five return shapes plus reverse tickets.
5. **Deliverable E** has the ledger with at least three entries; all required fields populated; review cadence stated.
6. **Deliverable F** walks through one complete cycle end to end with every artefact referenced.
7. **Deliverable G** has three filed (or clearly-dry-run) tickets using the template; one of them is a reverse ticket.
8. The contract is signed (real or dry) by both sides' leads.

## Common failure modes

- **Boundary statement in abstract.** Chapter 04's one-sentence rule is specific; your version is "they do kernels, we do serving" — not specific enough. Name artefacts.
- **Ticket template without `rule_outs_performed`.** The peer receives tickets that are KV-cache issues. The first "not a kernel issue" return costs a day. Fix.
- **SLA table with no pager mapping.** "P0 is 1 hour" means nothing if the on-call doesn't know to escalate at 1 hour. Map the SLA to actual paging behaviour.
- **Ledger with one entry.** The ledger gets maintenance only as it is used. Seed with three realistic entries so the maintenance habit sticks.
- **Worked example incomplete.** A worked example that stops at "filed the ticket" is half the lesson. Carry it through to the ledger entry.

## Keep it

The contract, the ledger, the worked example, the tickets. Exercise 05 uses the same contract shape for four more peers. Chapter 01's quarterly cadence re-reviews the contract and the ledger together; keep both durable.
