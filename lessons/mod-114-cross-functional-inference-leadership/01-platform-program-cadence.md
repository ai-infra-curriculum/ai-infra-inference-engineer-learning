# Chapter 01 — Running the Inference Platform Program: Cadence, SLO Review, On-Call, Change Calendar, Migration Planning

> Modules 102–113 built the serving stack and the deploy discipline that surrounds it. This chapter builds the *program* those artefacts live inside — the weekly ritual that keeps an inference platform aligned with its peers, its consumers, and its SLOs across a quarter.

## Motivation

A healthy inference platform looks the same as a healthy database platform or a healthy identity platform: there is a cadence you can draw on a calendar, an SLO review you can walk a director through, an on-call rotation that is written down, a change calendar nobody argues about, and a migration plan for every model, engine, hardware class, and dependency currently in flight. An unhealthy one looks like: heroics, a different incident every Friday, a build that nobody can reproduce, and a model-deprecation surprise that blocks the consumer team's quarter.

The difference between the two is not talent; it is *program discipline*. By level 35 the inference engineer does not just build the serving stack — they own the cadence of change on it. This chapter is the vocabulary for that cadence.

By the end of this chapter you can:

- Design a weekly / monthly / quarterly cadence for an inference platform team that integrates with the peer tracks' cadences.
- Author an SLO review ritual that reads the module-110 axis set against the module-113 deploy events and the module-109 capacity trend on one page.
- Author an inference-aware on-call rotation that respects the GPU-fleet failure modes (not just HTTP 5xx).
- Author a change calendar that gates deploys against the consumer tracks' release windows and the peer tracks' incident load.
- Author a migration plan that slots a specific change (engine upgrade, hardware swap, quantisation cutover, API-version bump) against the cadence without surprising any peer.

## The four artefacts that define the program

An inference-platform program at team scope is defined by four written artefacts. If any one is missing or stale, the program is one incident away from chaos.

| Artefact | What it says | Who signs | Cadence it drives |
|---|---|---|---|
| **Charter** | What the platform owns, what it does *not* own, who its consumers are, what SLOs it commits to. | Platform lead + sponsor (director / VP). | Quarterly review; no routine deploy depends on it. |
| **SLO + error-budget contract** | The latency / availability / cost / semantic-correctness SLOs per workload class (built in module 110). | Platform lead + consumer-track leads + SRE peer. | Weekly SLO review (the central ritual of this chapter). |
| **Change calendar** | What is deploying when, who is on call for it, which freezes apply. | Platform on-call + change-management partner. | Weekly (next 1–2 weeks fine-grained) and quarterly (the migration roadmap). |
| **Runbook set + on-call rotation** | The inference-flavoured runbooks from module 113 chapter 03, plus the rotation schedule and the paging matrix. | Platform lead + on-call engineers. | Updated after every incident (module 113 chapter 04's post-mortem feeds this). |

None of these are new ideas from an SRE perspective. The adaptation for inference is *what they contain*:

- The charter names the model landscape the platform serves (open weights it hosts, routed providers it brokers, custom checkpoints it swaps through), not just "the inference API."
- The SLO contract has tokens-per-second and semantic-equivalence terms, not just HTTP latency.
- The change calendar's "deploy events" include *model swaps*, *adapter promotions*, *quantisation cutovers*, *engine upgrades*, and *hardware migrations*, each with its own blast radius.
- The on-call rotation is sized for a surface that includes `NCCL hang`, `GPU ECC`, `cache-hit cliff`, and `silent accuracy regression` — failure modes a generic web on-call shift does not train for.

## Weekly cadence — the SLO review ritual

The weekly SLO review is the central ritual. If you pick one thing to institutionalise in the first quarter of owning a platform, pick this one. It takes 45–60 minutes, and it produces the input the rest of the week's decisions consume.

The agenda (fixed; variation causes drift):

1. **Last week's SLO burn.** For each workload class and each SLO (TTFT P95, TPOT P95, ITL P99, cache-hit, cost-per-1M-tokens, error-rate, semantic-equivalence), show the burn-rate chart from module 110 overlaid with the deploy events from module 113 chapter 02's change calendar. One chart per class is enough.
2. **Incidents since last week.** For every incident (including auto-rollbacks from module 113 chapter 01), name the runbook that ran, the time-to-mitigate, the RCA status, and the fixture back-feed (module 113 chapter 04).
3. **Capacity trend.** From module 109, where the fleet is against its headroom budget — GPU utilisation at peak, warm-pool depth, spot interruption rate, scale-up lead time. One number per cluster, with the trend over four weeks.
4. **This week's planned changes.** From the change calendar: what deploys when, what peer is involved, what blocks or freezes apply. A one-line risk score per change.
5. **Open action items.** A standing list with owners and due dates; items roll forward until closed or formally dropped.

Three disciplines make the ritual load-bearing instead of theatre:

- **The dashboard is the agenda.** The SLO review dashboard is one URL; the agenda above matches its panel order; nobody takes "pre-reads" or pastes screenshots into slides. If the dashboard doesn't say it, it doesn't exist.
- **Every burn or incident lands in a ticket.** If an SLO breached or almost-breached and no ticket came out of the meeting, the ritual is already decorative.
- **The attendee list is explicit.** Platform lead and on-call engineer always. SRE peer (module 113 boundary) rotates. Consumer-track representative every 2–4 weeks. Sponsor every month. Eval peers (module 113 chapter 05) when semantic-equivalence SLOs are on the agenda. A four-person meeting is fine; a twelve-person meeting is a project-manager meeting, not an SLO review.

### What the SLO review is not

- It is not an incident review. Incident reviews happen per incident (module 113 chapter 04's post-mortem ritual); the weekly review names them, does not replay them.
- It is not a capacity planning meeting. Capacity planning happens quarterly (below); the weekly review surfaces the trend, not the plan.
- It is not a product roadmap. Product roadmap happens with the consumer-track leads on their cadence; the weekly review honours the roadmap's deadlines when they appear on the change calendar.

## Monthly cadence — the change-calendar review

Once a month, zoom out from "this week's deploys" to the next 4–6 weeks. The questions:

- **Freezes.** When do the consumer tracks have release windows that preclude non-critical platform changes? (Example: a consumer team on a monthly release cut typically freezes the week of the cut.) When do the ML-platform peer's retraining pipelines land a new checkpoint that the platform must accept on day X? When does the hardware vendor's driver cadence force a reboot?
- **Overlapping change risk.** Two independently-safe changes can create an unsafe product (engine upgrade *and* quantisation cutover on the same workload class in the same week doubles the diagnostic surface; an incident becomes un-bisectable). The monthly review names and spreads these.
- **Blackout conventions.** Many teams run a convention like "no non-critical change deploys on Fridays after 14:00" or "no cross-time-zone cutover on a Monday morning." Name the conventions and defend the exceptions.
- **Dependency-driven pulls.** A security peer's planned certificate rotation, an ML-platform peer's model-registry schema change, an SRE peer's cluster-upgrade window — each creates a dependency on the inference platform's calendar. The monthly review resolves the dependency graph.

The output: a shared calendar (Confluence / Linear / Notion / a `change-calendar.md` in the ops repo — the artefact is what matters, not the tool) with the next 4–6 weeks' known changes, their owners, their blast radius, and their reversibility window.

## Quarterly cadence — the migration roadmap

The quarterly cadence is about *multi-week moves*: migrations that take longer than a single deploy and have to be scheduled against the consumer tracks' quarterly plans. The typical surface:

- **Engine upgrades across major versions.** A vLLM / SGLang / TRT-LLM major-version jump with a scheduler change (chunked prefill becomes default, KV allocator flag renamed, etc.) is a multi-week migration, not a single deploy. Chapter 02 builds the RFC shape.
- **Hardware migrations.** H100 → H200, add Blackwell capacity, retire an older SKU. Multi-week because supply, networking, firmware, and driver cadence all participate.
- **Quantisation cutovers.** FP16 → FP8 for a flagship model. The shadow-roll (module 113 chapter 01), the eval-gate rework (module 113 chapter 05), and the canary ramp together are typically two–four weeks.
- **API-surface evolution.** A new endpoint, a deprecation (module 113 chapter 06), a tool-calling-protocol upgrade (module 111). Multi-week because the consumer tracks must migrate too.
- **Vendor / build-vs-buy transitions.** Moving a workload class from self-hosted to a vendor, or the reverse. Chapters 02 and 03 build the RFC and the vendor eval.

The quarterly ritual:

1. **Review the migration ledger.** What migrations are in flight; which stage each is in (RFC / pilot / rollout / stabilisation / retired); the owner; the next milestone.
2. **Rank the next quarter's candidate migrations.** From SLO pressure, cost pressure, hardware refresh pressure, consumer-track feedback. A short-list of 3–6 candidates with named sponsors.
3. **Draw the critical path.** Which migrations must complete before which others can start (e.g. engine upgrade must land before the FP8 cutover; the ML-platform peer's model-registry schema must land before the API-version bump).
4. **Commit to a quarter.** Pick 2–3 big migrations to commit to; the rest explicitly stay on the backlog with a reason.
5. **Draft the first RFC.** For the top-ranked migration, draft the chapter-02 RFC skeleton and name the author.

## On-call rotation for an inference platform

Inference on-call is *not* generic web on-call. The runbook top-ten from module 113 chapter 03 is the proof — half of the failure modes do not appear in a stateless web service. Design the rotation around that reality.

### Sizing the rotation

A generic rule of thumb for a team of N engineers is "N-1 or N-2 on-call shifts covered so the primary isn't exhausted." For inference, two additional rules:

- **Inference incidents are more diagnostically expensive than a 500-error spike.** A cache-hit cliff or a silent accuracy regression can take hours to diagnose even with the runbook in hand. Rotation length should let the engineer trade off incident response and project work in a week-long block; shift-switching mid-incident is a tax on time-to-mitigate.
- **Inference-platform secondary engineers shadow the primary during the first week on-call.** The runbook set is the training material (module 113 chapter 03); live incidents are the exam. Rotate new engineers into secondary for a full cycle before primary.

A workable baseline for a team of 6–8 engineers:

- Week-long primary shift.
- Separate secondary who shadows and takes handoffs.
- Rotation gap of at least two weeks between primary shifts for the same engineer.
- A written "leaving on-call" doc that the departing primary writes before the handover: open incidents, open freezes, open dependencies, open action items.

### The paging matrix

A paging matrix names *which alert pages whom*. For inference:

| Alert class | Primary paging | Secondary paging | Peer escalation |
|---|---|---|---|
| Latency SLO burn (TTFT / TPOT / ITL) | Platform primary | Platform secondary at 15 min | Performance peer (L35) if NCCL or kernel-attribution |
| Availability / error-rate | Platform primary | SRE peer at 10 min | SRE peer leads incident command |
| Cost SLO burn (cost-per-1M-tokens ceiling) | Platform primary | Finance partner (async, not pageable) | None |
| Semantic-equivalence SLO (eval regression on live) | Platform primary | AI-eval peer / ML-eval peer | Eval peers co-own mitigation |
| Model-deprecation window expiry | Platform primary | Consumer-track lead | Consumer-track lead signs the extension |
| GPU hardware (DCGM, ECC, driver XID) | Platform primary | Infrastructure peer / cloud TAM | Hardware vendor escalation path |
| NCCL hang / collective comms failure | Platform primary | Performance peer (L35) | Performance peer co-owns root cause |
| Security alert at inference edge | Security peer (L35) | Platform primary (support) | Security peer leads |

Three rules:

- **Alerts page one human first.** If three humans page simultaneously on every incident, every incident becomes a committee decision. One primary; others escalate on a clock.
- **Escalation is time-bound.** "Page secondary at 15 min" means the alerting system auto-escalates, not that the primary asks politely.
- **The matrix is reviewed after every escalation.** If the peer was paged and they shouldn't have been — or wasn't paged and should have been — the matrix is wrong. Update it in the post-mortem.

## The change calendar — what it actually tracks

The change calendar is the shared document (or tool) that says *what is deploying when and under whose discipline*. For inference, it tracks more axes than a web service's calendar:

| Field | Values | Why it's here |
|---|---|---|
| `change_id` | short slug | unique lookup; references the RFC / ticket |
| `window` | date + time + timezone + duration | respects consumer-track freezes |
| `blast_radius` | fleet / workload-class / tenant / adapter | so overlapping changes can be spread |
| `change_type` | engine version / model swap / quantisation / scheduler flag / routing / hardware / API | determines the rollout primitive (module 113 chapter 01) |
| `rollout_primitive` | canary / shadow / blue-green / drain-and-swap / hard cutover | which module-113 pattern |
| `reversibility` | seconds / minutes / hours / permanent-after-N-days | how fast module 113's rollback can run |
| `on_call` | primary name for the window | so the pager routes correctly |
| `peer_dependency` | performance / ML-platform / eval / SRE / security / consumer-track / none | so peers know they're in the critical path |
| `freeze_check` | passed / waived-with-justification | so consumer-track release windows are honoured |
| `eval_gate` | pending / green / waived | so semantic-equivalence changes have evidence (module 113 chapter 05) |

The value of the calendar is *not* the tool; it is the discipline of filling every field *before* the change is approved. A change with an empty `peer_dependency` field is a change that will surprise a peer the day of the deploy.

## Migration planning across the peer ladder

The "migration" in the module title has a specific meaning: a multi-week move that touches the consumer tracks and at least one peer. A migration plan is distinct from a deploy; it is a mini-program of its own, with a start, a middle, and an end.

The structure of a migration plan:

1. **Problem statement.** What forces the migration (SLO pressure, cost, hardware refresh, vendor policy change, consumer-track requirement). One paragraph.
2. **Scope.** Which workload classes, which models, which clusters, which tenants are in and out.
3. **Target state.** What the platform looks like at the end; what SLOs it commits to in the new state.
4. **Peer touch points.** Which peers have deliverables on the critical path; what they deliver by when.
5. **Stages.** Pre-work → pilot → ramp → stabilisation → retirement. Each with a gate.
6. **Rollback branch.** At what stage rollback is still cheap; at what stage it becomes a second migration.
7. **Consumer communication plan.** Who tells the consumer tracks what, by when, through which channel. Chapter 06 of module 113 is the deprecation special-case; migrations in general need the same discipline.
8. **Evidence of done.** The SLO data, the eval data, the cost data that will be true when the migration is called complete.

A worked example (illustrative): "Migrate the general-chat workload class from Llama-3.1-70B FP16 to Llama-3.1-70B FP8."

- Problem: 40 % of peak-hour GPU budget on this workload; FP8 benchmark shows ~1.9× decode throughput at eval-threshold accuracy.
- Scope: chat workload class only; long-context remains FP16; code-completion stays on the dedicated model.
- Target state: FP8 serving Llama-3.1-70B chat; TTFT SLO unchanged; cost-per-1M-output-tokens SLO tightened by 40 %; semantic-equivalence SLO adds a new axis (reasoning-benchmark delta ≤ 1.5 %).
- Peer touch points: ML-platform (publishes the FP8 artefact), eval peers (run the capability and application eval), performance peer (reviews the FlashAttention variant selection for FP8 long-context), SRE (reviews the SLO contract update).
- Stages: pre-flight eval (2 weeks) → shadow roll (1 week, module 113 chapter 01) → canary ramp (1 week) → hold-at-100 (1 week) → FP16 retired (capacity reclaimed to the warm pool, module 109).
- Rollback branch: through the shadow-and-canary stages, flip back by routing weight (module 113 chapter 01). Once FP16 replicas are drained and capacity reclaimed, rollback is a full re-provisioning — treat as a second migration.
- Consumer communication plan: announcement at pre-flight; reminder at shadow; `X-Model-Version` header carries the new version from canary onward; retention metrics shared weekly with the product partner for four weeks.
- Evidence of done: SLO dashboard four-week stable in new state; cost-per-1M-tokens number achieved; eval delta inside tolerance; no new regression fixtures from the canary period.

The migration plan is one document, is signed before work starts, and is updated every week until the migration is retired. Chapter 02 then takes the same discipline into its RFC form for build-vs-buy decisions that are structurally similar but scoped larger.

## How the program plugs into the peer tracks

Every chapter in modules 102–113 names a peer boundary. The program chapter is where those boundaries become scheduling facts:

- **Performance peer (L35, kernel craft).** The weekly SLO review flags any NCCL / latency-decomposition anomaly to the performance peer; the monthly change calendar coordinates engine-upgrade timings with their kernel-regression gates. Chapter 04 formalises the interface.
- **ML-platform peer (L30, model handoff).** The weekly SLO review covers the eval-gate status for pending model handovers; the monthly change calendar co-ordinates the model-registry schema changes. Chapter 05 formalises.
- **Eval peers (L30, AI-eval and model-evaluation).** The weekly SLO review has a semantic-equivalence section; the monthly covers the fixture-back-feed status (module 113 chapter 04 produces, chapter 05 consumes). Chapter 05 formalises.
- **SRE peer (L30, methodology).** The SRE peer attends the SLO review on a rotation; the on-call rotation and the incident-command ownership for availability incidents sit with them. Chapter 05 formalises.
- **Security peer (L35, inference-edge).** The weekly surfaces any security alert at the edge; the monthly coordinates certificate rotations, model-signing cadence, and any inference-edge posture change. Chapter 05 formalises.
- **Consumer tracks (L25, AI Engineering family).** The monthly covers their release windows; the quarterly reviews their roadmap pulls on the platform. Module 113 chapter 06 handles the deprecation special case.

## Anti-patterns

Four failure modes recur in young inference programs. Name them so you can catch them in a peer review of your own cadence.

- **Hero on-call.** One engineer owns every incident because they're the one who wrote the engine config. No rotation; no shadow; no runbook. The platform's reliability is bounded by that engineer's calendar.
- **Dashboard-as-ritual.** The SLO review happens every week, nobody reads the dashboard, and no ticket ever comes out. The meeting still exists; the discipline doesn't.
- **Hidden freeze.** A consumer team is in a release window that nobody on the platform knows about; a routine deploy lands the day of their cut; both teams spend a day sorting out whose fault it was. The change calendar's freeze-check field exists to prevent this.
- **Migration-by-stealth.** A quantisation cutover or an engine upgrade happens as "just another deploy" without a migration plan, without a peer sign-off, without the consumer-track announcement. The incident that follows is both avoidable and expensive to recover from.

## Summary

- An inference-platform program at team scope is defined by four artefacts: charter, SLO contract, change calendar, runbook set + on-call rotation. Each must be written, signed, and current.
- The weekly SLO review is the central ritual. Burn chart, incidents, capacity trend, this week's changes, action items. Attendees are small; the dashboard is the agenda; every item lands in a ticket or dies.
- The monthly change-calendar review spreads overlapping-change risk, respects consumer-track freezes, resolves peer dependencies, and makes blackout conventions explicit.
- The quarterly migration roadmap picks the 2–3 multi-week moves the team will commit to, the critical path between them, and drafts the first RFC.
- The on-call rotation is sized for inference failure modes (not web failure modes); the paging matrix is explicit about who pages for which alert class and who escalates when.
- The change calendar tracks more axes than a web service's calendar — rollout primitive, reversibility, peer dependency, eval-gate status. Fill every field before the change is approved.
- Migration plans are mini-programs with eight fields (problem, scope, target state, peer touch points, stages, rollback branch, consumer communication plan, evidence-of-done). One document, signed before work starts, updated weekly.
- Chapter 02 picks up with the RFC form for the biggest class of migration — build-vs-buy decisions. Chapter 03 operationalises the vendor-eval scorecard that the build-vs-buy RFC references. Chapters 04 and 05 formalise the peer interfaces this chapter's cadence coordinates with.
