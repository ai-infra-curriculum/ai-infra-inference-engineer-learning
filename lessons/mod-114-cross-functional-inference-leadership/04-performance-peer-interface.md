# Chapter 04 — Interface Contract with the Performance Peer: Tickets, Returns, Regression Tracking

> The single most important peer boundary in this curriculum is with `ai-infra-performance` at L35. Both roles share the same market signal and work on the same stack; neither can do the other's job well. The interface is the written contract that lets the two tracks move at full speed without stepping on each other.

## Motivation

A production-serving incident fires. A FlashAttention variant chosen for FP8 long-context batches regresses ITL-P99 by 40 % on the chat workload class. The inference engineer bisects it to a specific kernel-selection path in the engine; the performance peer owns the kernel. What happens next is the whole chapter.

Done badly: the inference engineer files a Slack message to the performance peer, "hey, this attention kernel is slow, can you look"; the performance peer responds "we need a reproduction"; two days pass; nobody owns the ticket; the SLO burns through its budget; eventually a VP escalates and both teams discover they were waiting on each other.

Done well: the inference engineer files the ticket against a documented template with the exact reproduction, the exact axis set from module 110, the exact engine version, the SLO impact, and the expected return shape. The performance peer picks it off a queue that both tracks know is the queue. A regression-tracking record captures the kernel-path, the fix, and the regression fixture. The next time a similar regression appears, the inference engineer notices because they have been watching the regression ledger.

That written interface is the artefact. By the end of this chapter you can:

- Draw the responsibility boundary with the performance peer precisely — what this role files to them, what they return.
- Author the ticket template (what fields; what evidence; who signs each).
- Author the kernel-regression ledger — the shared record of known regressions the two tracks consult and update together.
- Author the counter-contract — what the performance peer files *to* this track, and how.
- Avoid the five failure modes that make a peer interface theatre.

Chapter 05 then extends the pattern to the other peer tracks (ML platform, eval, SRE, security). This chapter specialises on the performance boundary because it is the densest.

## The boundary — one sentence, then the details

**Rule.** If the primary deliverable is a *CUDA / Triton / CUTLASS kernel source file*, that is the performance peer. If the primary deliverable is a *production serving system that calls kernels*, that is this role.

The one-sentence rule resolves 80 % of cases. The 20 % — scheduler decisions, torch.compile dispatch, attention-variant selection, kernel-fusion policy — are the hard ones; the ticket template is where they land.

### What this role owns (inference-platform side)

- Engine configuration (vLLM / SGLang / TRT-LLM / Triton / Dynamo args, scheduler knobs, admission policy).
- KV allocator parameters as a runtime object (module 104).
- Fleet shape (TP / PP / EP topology for a given model + fleet + traffic shape; module 107).
- The SLO contract and the module-110 axis set.
- Deploy rituals, rollbacks, runbooks (module 113).
- All the above from the program-level perspective (modules 102–114).

### What the performance peer owns (kernel-craft side)

- CUDA / Triton / CUTLASS kernel source files.
- Fused-kernel implementations (FlashAttention variants, MoE expert dispatch kernels, custom GEMM epilogues).
- Attention-backend selection logic *inside the engine* (which variant for which shape).
- Quantisation algorithms (AWQ, GPTQ, SmoothQuant, FP8 / FP4 calibration) at design depth.
- Nsight / SASS-level profiling.
- torch.compile pass pipelines, TensorRT builder-config tuning.
- Kernel-level regression benchmarks.

### The 20 % where it is genuinely shared

- **Which engine flag to flip on a regression.** The flag exists in the engine; selecting it under live traffic is this role; the *implementation* of what it dispatches to is the performance peer. The ticket template routes the "I think this flag is wrong" to the performance peer with reproduction evidence.
- **Quantisation deployment.** This role deploys FP8; the peer designs the FP8 recipe. The accuracy-regression gate (module 105) is shared with the ML-eval peer at L30; the kernel-level FP8 variant lives at the performance peer.
- **Speculative decoding.** This role operates draft-target pairs (module 108) and monitors acceptance rate; the peer implements the kernels and may inform the draft-model structural choice.
- **Distributed inference.** This role picks the topology (TP within node, PP across nodes, EP for MoE; module 107); the peer implements the NCCL / RCCL kernels and informs the topology choice via collective-cost analysis.

For each shared area, the ticket template below forces an owner for every ticket.

## The ticket template — what to file, what to include

Both tracks agree on one ticket template. It lives in whichever issue tracker (Linear, Jira, GitHub Issues) the organisation uses. The template has required and optional fields.

### Required fields

- **`title`** — one line. "ITL-P99 regression on FP8 long-context chat on vLLM 0.9.x" is a title; "attention is slow" is not.
- **`workload_class`** — chat, agent, long-context-chat, code, embedding, etc.
- **`axis`** — the axis from module 110 that is affected: TTFT, TPOT, ITL, cache-hit, preemption, NCCL-time, GPU-util, etc.
- **`slo_impact`** — the SLO being burned: "ITL-P99 SLO 300 ms, measured 420 ms over the last 4 hours, burn rate 6× on the 30-day window."
- **`engine_and_version`** — vLLM 0.9.3, SGLang 0.4.1, TRT-LLM 0.17.0, etc. Include commit hash if from source.
- **`model_and_version`** — exact artefact identifier: Llama-3.1-70B-Instruct FP8 revision X.
- **`hardware_class`** — H100-SXM5-80GB × 8 node, H200-SXM6-141GB, MI300X, TPU v5p, etc.
- **`topology`** — TP=8, PP=1, EP=none, etc.
- **`reproduction`** — the minimal reproduction. Preferred: a self-contained benchmark script or a captured trace (module 110 chapter 03) that the performance peer can run locally. "I saw it in prod" is not a reproduction.
- **`expected_return`** — one of: diagnostic opinion, kernel patch, engine config suggestion, upstream engine ticket, "not a kernel issue."
- **`priority`** — follows the organisation's severity map; mapped from the SLO burn rate.
- **`sla`** — the agreed response time (see below).

### Optional but strongly recommended

- **`metric_overlays`** — screenshot or Grafana URL of the affected axis, overlaid with deploy events.
- **`trace_sample`** — one or two captured traces (OpenTelemetry, torch profiler, Nsight) at the regressing path.
- **`prior_runbook`** — if the module-113 runbook was consulted, which one, with what step reached.
- **`related_regressions`** — the ledger entries (see below) that may be related.

### Required *before* the ticket is filed

Discipline the inference engineer runs first — these are the cheap diagnostics that avoid filing a non-kernel ticket to the kernel team:

- Did a config change land? Check the chapter-01 change calendar.
- Is this a load-pattern change? Compare the traffic shape week over week.
- Is cache-hit normal? If cache-hit cliffed, this is a KV / routing problem, not a kernel problem (module 104's chapter).
- Is this a specific node? If one replica is slow, this is a hardware / topology issue (module 109's runbook), not a kernel issue.
- Does the regression track a specific engine version? If so, the engine's own release-notes are the first stop (module 102's engine debug surface).

Only after those rule-outs does the ticket go to the performance peer. The template has a `rule_outs_performed` field with checkboxes to document what was already eliminated.

## Response SLAs — the ticket queue, not the ad-hoc ping

The interface is only load-bearing if the ticket queue has a response SLA both tracks can trust. Representative SLAs (adapt to your organisation's severity map):

| Ticket priority | SLO-burn trigger | Performance-peer response SLA | Resolution SLA |
|---|---|---|---|
| P0 | Active SLO breach, auto-rollback fired, consumer impact | 1 hour (ack) | best-effort; incident ownership shared |
| P1 | SLO burning fast; ~1-week budget consumed | 1 business day (ack); 1 business day (opinion) | 1 week |
| P2 | SLO burning slowly; multi-week budget consumed | 2 business days (ack); 1 week (opinion) | 2–4 weeks |
| P3 | Planned optimisation, no active burn | 1 week (ack) | best-effort |

Discipline notes:

- **The SLA is negotiated.** Both tracks sign the table; it is reviewed quarterly alongside the chapter-01 cadence.
- **P0 does not mean "page the performance peer from Slack."** It means file the ticket at P0 and the paging system routes. If the organisation has a shared incident channel, the ticket number lives there.
- **The queue is public to both tracks.** Both leads can see the whole queue; both can re-prioritise with a documented reason.

## Return shape — what the performance peer delivers

A ticket is complete when the performance peer returns one of a short list of shapes:

- **Diagnostic opinion.** "The regression is attributable to attention-variant selection X on shape Y. The fix is upstream; I've filed an engine ticket." The inference engineer then applies a workaround (engine flag, pinning to a prior version) and tracks the upstream fix.
- **Kernel patch.** The peer authored a kernel patch and landed it in the engine (or in a fork the fleet runs). The inference engineer validates on canary (module 113 chapter 01) and promotes.
- **Engine config suggestion.** "Set `attention_backend=flash_attn_v3` for this shape class; the default picks a wrong variant below a threshold." The inference engineer applies via engine config and validates.
- **Upstream engine ticket.** "This is a vLLM upstream bug; I've filed it; here's the ticket URL and the ETA." The inference engineer tracks the upstream.
- **"Not a kernel issue."** The ticket is returned with evidence pointing to KV, scheduler, admission, routing, or hardware. The inference engineer re-routes to the right module.

Every return shape has a *regression ledger entry* (next section). A ticket closed without a ledger entry is a ticket whose knowledge the organisation loses.

## The kernel-regression ledger

The ledger is the shared, cross-track record of known kernel-level regressions and their state. It is the single most load-bearing artefact of this chapter — without it, the same regression is diagnosed three times in three quarters and every time feels novel.

### Entry fields

- **`id`** — short slug.
- **`title`** — one line.
- **`first_seen`** — date + engine + hardware + model.
- **`status`** — open-investigating / open-workaround / fixed-upstream / fixed-in-fleet / won't-fix.
- **`workaround`** — the engine config or pinning that mitigates (if any).
- **`upstream_ticket`** — URL to the engine project's issue if one exists.
- **`fix_commit`** — commit SHA when the fix lands (upstream or fork).
- **`regression_fixture`** — pointer to the benchmark / unit test / canary acceptance gate that would detect recurrence. Module 113 chapter 04's fixture back-feed feeds this.
- **`owner`** — the engineer on each side responsible for tracking.
- **`last_reviewed`** — date of the last re-check (quarterly minimum).

### The ledger review

Once a quarter, both leads walk the ledger:

- Fixed-upstream entries: has the fleet moved to the fixing version? If yes, close.
- Open-workaround entries: is the workaround still active? Any evidence the workaround is causing secondary damage?
- Open-investigating entries: still a priority? Should they become won't-fix with a documented rationale?
- New entries since last review: validated reproductions; reasonable priorities.

The ledger is the chapter-01 quarterly cadence's "performance peer" slot in action.

### Why the ledger matters

Without it, every new incident that looks like a known regression starts from zero. With it, the on-call engineer's chapter-03 runbook from module 113 can reference the ledger entry and reach "pin to version X, workaround Y" in minutes instead of hours.

## Reverse-direction tickets — what the performance peer files to this role

The interface is bidirectional. The peer sometimes needs this role to provide evidence or a change.

### Typical reverse tickets

- **Production trace capture.** The peer is diagnosing a kernel-selection logic and needs a representative trace from real traffic (this role controls the traffic; the peer does not). The ticket from the peer: "please capture a TTFT-heavy trace on the chat workload class at peak hour on cluster X with engine version Y; attach torch profiler + Nsight if feasible."
- **Canary slot for a candidate kernel.** The peer has a kernel patch ready and needs a canary slot in the deploy calendar. The ticket from the peer: "please schedule a canary for engine build Z containing kernel patch W on the chat workload class; see module 113 chapter 01 for the ramp."
- **SLO data for a regression characterisation.** The peer needs a specific slice of module-110 axis data at a specific time window for post-hoc analysis.
- **Fleet-shape change.** The peer finds that a topology (TP=4 instead of TP=8) would avoid a particular kernel-selection path on the current hardware; the ticket from the peer proposes the topology change for evaluation.

The reverse tickets have the same template and the same SLA structure (the roles swap). Neither side is the sole requester.

## The shared on-call escalation — for P0 incidents

When a P0 fires with a candidate performance-peer root cause, the incident is co-owned. Module 113 chapter 04's incident-command methodology applies; the specific add-ons:

- **Incident channel has a named performance-peer engineer on the hour.** Not a group alias. A human the inference engineer can talk to.
- **The RCA artefact's "kernel" section is co-authored.** The chapter-04-of-113 RCA page has a section that only the performance peer can fully write; they contribute inline.
- **The post-mortem's action items split by track.** Items of kernel nature → performance peer's backlog; items of config / runbook / deploy nature → this track's backlog. Neither side carries the other's action items.
- **The ledger entry opens inside the incident.** The incident commander ensures the ledger entry is created before the incident is closed, with status open-investigating or open-workaround.

## Common failure modes in the peer interface

- **Ad-hoc ping as the default channel.** Tickets never get filed; the queue is invisible; priority is "who asked louder." Fix: no non-trivial request without a ticket; the Slack ping is a reminder of the ticket number, not a substitute.
- **No rule-outs performed.** The performance peer receives a ticket that is a KV or scheduler problem labelled "kernel." They spend a day returning it. Fix: the `rule_outs_performed` field in the template is required.
- **Ledger not maintained.** Every incident feels novel. Fix: the ledger is a required post-ticket step; the chapter-01 quarterly cadence re-reviews it.
- **One-way interface.** This track files to the peer; the peer never files back; neither side feels responsible for the other's data needs. Fix: reverse tickets are a first-class pattern, not a courtesy.
- **No shared on-call pattern.** A P0 is diagnosed via Slack while nobody writes anything down. Fix: incident command methodology (module 113 chapter 04) extends to the peer; the RCA has a co-authored section.

## Worked example — a FlashAttention variant regression

A sketch of the full cycle (shape only — not a specific vendor or incident):

1. **Monday 14:00.** The chat workload class's ITL-P99 chart (module 110) starts drifting above 300 ms. The on-call primary (chapter-01 rotation) checks the change calendar — an engine version upgrade landed at 10:00. The runbook for "ITL regression" (module 113 chapter 03) says "check attention-backend selection first."
2. **Monday 14:30.** The runbook's bisection points at FP8-long-context batches on this engine version. Cache-hit, scheduler, admission, and hardware have been ruled out (the `rule_outs_performed` list).
3. **Monday 14:45.** Ticket filed at P1 to the performance peer, template filled, trace attached.
4. **Monday 15:00.** Auto-rollback (module 113 chapter 01) kicks on the burn-rate alert; traffic returns to the prior engine version; SLO recovers. Incident becomes a P1 investigation, not a P0.
5. **Tuesday 11:00.** Performance peer acks the ticket (within P1 SLA). Opinion: attention-backend selection path picks `flash_attn_v3_fp8` for shapes it shouldn't.
6. **Tuesday 16:00.** Engine config suggestion returned: `attention_backend=flash_attn_v2_fp8` for FP8 long-context batches until upstream fix.
7. **Wednesday.** Inference engineer canaries the config change on the engine version from step 2; validates; promotes. Rollback to prior engine version is no longer blocking.
8. **Wednesday evening.** Ledger entry created: id `fa-v3-fp8-longctx-regression`, status `open-workaround`, workaround documented, upstream ticket URL recorded, regression fixture added (a canary acceptance test against a long-context FP8 fixture that would catch recurrence).
9. **The next quarter's ledger review.** Upstream ticket closed; fleet moves to the fixing version; ledger entry moves to `fixed-upstream`.

Every step references an artefact from this chapter and one from modules 110 / 113 / 01-of-114. The interface held.

## Summary

- The boundary with the performance peer: production serving systems live here; CUDA / Triton / CUTLASS kernels live there. The 20 % of shared surface routes via the ticket template.
- The ticket template has required fields (title, workload class, axis, SLO impact, engine version, model version, hardware class, topology, reproduction, expected return, priority, SLA) and a required `rule_outs_performed` list that shuts down non-kernel tickets early.
- Response SLAs are negotiated, public, and reviewed quarterly against the chapter-01 cadence.
- Return shapes are one of: diagnostic opinion, kernel patch, engine config suggestion, upstream engine ticket, "not a kernel issue." Every return produces a ledger entry.
- The kernel-regression ledger is the chapter's most load-bearing artefact. Entries have status, workaround, upstream ticket, fix commit, regression fixture. Ledger is reviewed quarterly.
- Reverse tickets (peer → this role) are first-class, not optional. Representative asks: trace capture, canary slot, SLO data, fleet-shape change.
- P0 incidents are co-owned via module 113 chapter 04's incident-command methodology, with a named peer engineer on the channel, a co-authored RCA kernel section, and a ledger entry opened before close.
- Chapter 05 extends the same contract pattern to the ML-platform, eval, SRE, and security peers — different artefacts, same discipline.
