# Exercise 01 — Role Scope and Deferral Contract

**Time:** 2 hours · **Output:** one diagram, one routing table, and one page of worked scenarios

## Goal

Make the six peer boundaries from [Chapter 01](../01-role-scope-on-the-ladder.md) concrete against *your* organisation (or an imagined one, if you don't yet have one). You will finish with a routing table you can post in a Slack channel topic.

## Problem statement

Your team runs the inference platform. Over the past two weeks, five tickets have landed in your queue, each of which *looks* like a serving problem but may not be. You need to:

1. Decide who owns each one.
2. Prove the decision from your charter.
3. Produce the deferral table and diagram that would make this decision one-ping-away for the next five tickets.

## Requirements

Deliver three artefacts.

### Artefact A — Role-scope diagram

A diagram (hand-drawn, Mermaid, Excalidraw — anything readable) that shows:

- Your role (`ai-infra-inference-engineer`, L35) at the centre.
- Each of the six peer / consumer boundaries from Chapter 01, labelled with the role name, its level, and its family.
- An arrow for each boundary labelled with the single most load-bearing deliverable that crosses it (e.g., "signed artefact + resource profile" from `ai-infra-ml-platform`, "WAF rules + weight-signing policy" from `ai-infra-security`).
- A visual mark on the arrow or boundary indicating whether it is a *defer-sideways*, *defer-up*, *defer-down*, or *consumer-of-ours* relationship.

### Artefact B — Deferral table

A table with one row per peer / consumer track. Columns:

| Peer | Level | Family | They own | We own | Routing rule (one sentence) | Escalation path |
|---|---|---|---|---|---|---|

Cover at minimum the six boundaries from Chapter 01 (`ai-infra-performance`, `ai-infra-ml-platform`, `ai-infra-sre`, `ai-infra-mlops`, `ai-infra-security`, AI Engineering family consumers) plus one row for the eval peer boundary (`ai-eval-engineer` at L30 for application eval; `model-evaluation-engineer` at L30 for benchmark methodology).

### Artefact C — Five worked routing scenarios

Pick (or adapt) five of the following tickets and resolve each one in ≤ 100 words:

1. "P99 TTFT doubled overnight on the chat endpoint. Monday's model swap landed on Sunday."
2. "A customer reports JSON-mode responses are no longer valid JSON for ~1 % of requests."
3. "An external security researcher demonstrated a prompt-injection chain that caused the model to leak another user's prefix cache."
4. "A new 70B model arrived from the ML team; the artefact's revision string doesn't match any registry lineage."
5. "The agent framework team wants to add a long-polling WebSocket mode because SSE 'is unreliable on their side.'"
6. "A new quantization recipe landed from research; the engine release notes say it should give FP8-equivalent accuracy at 1.3× throughput — can we deploy it?"
7. "A finance review wants cost-per-million-tokens broken down by business unit for the last quarter."
8. "Oncall just paged on a NCCL hang spanning two racks of H200s."

For each: name the owning role, cite the deferral row that supports the call, state the one piece of *information* the inference team needs to hand over to route it, and state the one piece of information the inference team needs *back* to validate the fix.

## Starter guidance

- Start from Chapter 01's six-boundary diagram and `CURRICULUM.md`'s ownership rule.
- When a scenario is ambiguous (routing could be a serving bug OR a kernel bug), *split* the ticket into the parts your team owns and the parts you'll re-route — this is the realistic answer in production.
- "Both" is almost always wrong. If a scenario really is co-owned, write down the *primary* owner and the *supporting* role explicitly.
- Keep the deferral table rows at tweet-length. Operators will read this during an incident, not at a desk.

## Acceptance criteria

Pass criteria for a self-review (or peer review inside your team):

1. The diagram shows all six boundaries and labels each with a concrete deliverable, not a vibe (not "collaboration," not "alignment").
2. The deferral table resolves at least 4 of the 5 chosen scenarios in Artefact C without re-opening the chapter.
3. Each scenario names *one* primary owner. If more than one is listed, there is an explicit note on why co-ownership is correct and what the split is.
4. The routing rule in every row of the deferral table is a *rule* (shape: "if the deliverable is X, route to Y"), not a description.
5. A peer teammate from a sibling track reads the table in under two minutes and agrees with the row for their track — or disagrees, and the disagreement is the real information you've generated.

## Keep it

Pin the deferral table to your team's Slack channel topic. Revisit it after module 102 (you'll want to add rows for engine-specific debugging) and after module 114 (you'll want to add vendor / build-vs-buy routing).
