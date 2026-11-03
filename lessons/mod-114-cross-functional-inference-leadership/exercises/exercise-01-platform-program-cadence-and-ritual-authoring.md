# Exercise 01 — Platform Program Cadence and Ritual Authoring

**Time:** 5 hours · **Output:** the four signed cadence artefacts for an inference platform (charter; SLO contract pointer; change calendar; runbook set + on-call rotation), a dry-run of the weekly SLO review, a monthly change-calendar view, a quarterly migration roadmap with one migration plan at full detail, and the on-call paging matrix.

## Goal

Make [Chapter 01](../01-platform-program-cadence.md) concrete. Produce the written documents that let an inference platform run on a cadence instead of on heroics. The exercise is complete when another engineer could sit in your on-call rotation, read your artefacts, and run the ritual tomorrow without consulting you.

## Problem statement

Your platform (real or plausible-hypothetical) serves at least two workload classes. You will author the four cadence artefacts chapter 01 names, run (or dry-run) a weekly SLO review against them, publish a monthly change-calendar view for the next 4–6 weeks, draft a quarterly migration roadmap with one of the migrations at full detail, and document the on-call paging matrix with explicit escalation routing.

If your environment is simulated, be explicit about that in the preamble to each artefact; the shape and discipline are the deliverable, not real sponsors' signatures.

## Prerequisites

- Chapters 01–05 of this module read; chapter 06 skimmed.
- Module 110 (SLO engineering, cost per 1M tokens) completed; the dashboards and SLO definitions this exercise references exist (even if the data is synthetic).
- Module 113 chapters 01–06 read; the runbook top-ten from chapter 03 either authored or sketched (exercise 03 of module 113 covers that).
- Access to your organisation's change-calendar tool (Linear / Jira / Confluence / Notion / plain Markdown — the tool doesn't matter; the discipline does).

## Requirements

Seven deliverables.

### Deliverable A — the charter

A one-page charter for your platform:

- What the platform owns (serving stack, routing, deploy ritual, runbooks, SLOs, cost accounting).
- What it does *not* own (kernel craft, model publication, product features, application code).
- Who its consumers are (named teams, workload classes, tenant tiers).
- The SLOs it commits to (pointer to the SLO contract, not duplicated here).
- The current scope boundary (which workload classes, which clusters, which regions).

Signed by a plausible sponsor (name the role even if simulated). State the review cadence.

### Deliverable B — the SLO contract pointer

The SLO contract itself lives in module 110's deliverables; this exercise produces the *pointer document* that the program uses:

- Where the contract lives (URL, file path).
- Which SLOs are covered per workload class.
- Who the signers are on both sides (consumer-track lead + platform lead + SRE peer).
- When the contract is reviewed (quarterly cadence).
- The escalation path for an SLO breach that exceeds the error budget.

If your module-110 artefacts are incomplete, name the gap explicitly and write the pointer as if it were complete; mark TODO where the contract will be filled in.

### Deliverable C — the change calendar

A change-calendar entry template and at least five populated entries covering the next 4–6 weeks. The required fields (chapter 01):

- `change_id`, `window` (date + time + timezone + duration), `blast_radius`, `change_type`, `rollout_primitive`, `reversibility`, `on_call`, `peer_dependency`, `freeze_check`, `eval_gate`.

At least one entry is a real (or realistic) model swap, one is a scheduler-flag change, one is a hardware event (driver / firmware / node drain), one has a peer dependency that touches another track, and one has a `freeze_check: waived-with-justification` (the justification must be defensible).

The deliverable includes both the template and the populated entries.

### Deliverable D — the runbook set pointer + on-call rotation

The runbooks themselves live in module 113 chapter 03's exercise. This deliverable is the *rotation plan and the paging matrix*:

- Rotation structure (shift length, primary + secondary, rotation gap, handoff doc).
- The paging matrix (chapter 01's table shape): alert class → primary paging → secondary paging timing → peer escalation.
- The "leaving on-call" handoff-doc template.
- The frequency and shape of on-call retrospectives.

### Deliverable E — the weekly SLO review, run or dry-run

Pick a representative week. Produce the dashboard (URL or screenshot) that is the agenda. Convene the meeting (real or dry-run with teammates / self as substitute for absent peers). Write the output:

- Minutes: one paragraph per agenda item (last week's burn, incidents, capacity trend, this week's changes, action items).
- The tickets created out of the meeting, with owners and due dates.
- An honest critique: what the ritual missed; what the dashboard didn't show that it should.

If you dry-run, write the minutes as if the meeting happened and name the stand-ins for each absent peer.

### Deliverable F — the quarterly migration roadmap

Draft the roadmap shape for the next quarter:

- Migration ledger (what's in flight, stage of each, owner, next milestone). Three to five entries.
- Ranking of 3–6 candidate migrations for the next quarter, with named sponsors.
- Critical path (what blocks what; dependency graph drawn).
- Commit: the 2–3 migrations the team will do; the others explicitly backlog with a reason.

Then pick *one* committed migration and author the full migration plan using chapter 01's eight-field shape (problem, scope, target state, peer touch points, stages, rollback branch, consumer communication plan, evidence of done).

### Deliverable G — the anti-pattern self-review

A short written review (one page) addressing chapter 01's four anti-patterns:

- Hero on-call — does any one engineer carry a disproportionate load?
- Dashboard-as-ritual — do action items come out of the SLO review?
- Hidden freeze — is the change calendar actually consulted by consumer tracks?
- Migration-by-stealth — does any recent change lack a migration plan or peer sign-off?

For each, honest assessment and one proposed fix.

## Starter guidance

- **The tool matters less than the discipline.** A Markdown file in a repo is a perfectly good change calendar; a polished Confluence page that nobody consults is not. Pick the tool your team uses.
- **The dashboard is the agenda, not the deck.** If you catch yourself preparing slides for the SLO review, replace them with a dashboard panel.
- **The migration plan is one document.** Resist the urge to spread it across a Confluence tree. One file; eight headings; all signers on one page.
- **Name stand-ins honestly.** A dry-run review where you played every peer is still useful; mark it explicitly.
- **Your fleet doesn't have to be real.** A plausible hypothetical — "fleet of 40 H100 nodes serving Llama-3.1-70B-FP16 plus a long-tail of 20 smaller models via a LiteLLM router" — is enough to anchor every artefact.

## Fallbacks for constrained environments

- **No fleet of your own:** work the exercise against a plausible hypothetical derived from the open-weights vLLM / SGLang ecosystem; be explicit in the preamble.
- **No peers to review:** write the artefacts; critique them yourself using the chapter's common-failure-mode list as a rubric; note in each artefact that peer sign-off is a TODO.
- **No prior module 110 or 113 artefacts:** sketch pointers to what the artefacts would say if authored; the discipline of pointer + handoff is what matters.

## Acceptance criteria

1. **Deliverable A** is a one-page charter; signed (or clearly marked simulated); names a review cadence.
2. **Deliverable B** names the SLO contract location, the signers, the review cadence, and the escalation path.
3. **Deliverable C** includes the template plus at least five populated entries that exercise the required fields; at least one freeze-check waiver with a defensible justification.
4. **Deliverable D** specifies the rotation structure, the paging matrix across all alert classes in chapter 01, and the handoff-doc template.
5. **Deliverable E** has the dashboard as the agenda, the meeting minutes in the five-item structure, the tickets-with-owners that came out, and an honest critique.
6. **Deliverable F** includes a migration ledger, a ranked candidate list, a critical path, a committed 2–3 migrations, and one full migration plan with all eight fields.
7. **Deliverable G** addresses each of the four anti-patterns with honest assessment and a fix.
8. The artefacts are consultable — another engineer could read them cold and run the next week's cadence.

## Common failure modes

- **Charter sprawl.** A three-page charter dilutes the one sentence that matters: what we own and what we don't. Keep it tight.
- **Change calendar without peer dependency.** Entries all have `peer_dependency: none`; first real peer-crossing change will surprise someone. Audit.
- **SLO review as status update.** No tickets came out of the meeting. Fix the ritual, not the template.
- **Migration plan without rollback branch.** The plan commits to the move; it doesn't name where rollback becomes expensive. Add the branch.
- **Paging matrix with everyone paged.** Every alert pages all four peers. The first alert will teach the team to silence the pager. One primary; others escalate on a clock.

## Keep it

The charter, the SLO-contract pointer, the change calendar template + entries, the rotation + paging matrix, the SLO-review minutes, the migration roadmap, the one migration plan, the anti-pattern self-review. Exercise 05 of this module reuses the paging matrix. Exercise 02 of this module uses the committed migration from Deliverable F as a plausible candidate for the build-vs-buy RFC.
