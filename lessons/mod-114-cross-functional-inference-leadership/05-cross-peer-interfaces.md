# Chapter 05 — Interface Contracts with the ML Platform, Eval, SRE, and Security Peers

> Chapter 04 built the interface with the performance peer as the archetype. This chapter extends the pattern to the other four peers — ML platform (L30), eval (L30, two peers), SRE (L30), security (L35). Each has its own artefacts, its own tempo, its own failure modes.

## Motivation

Each peer-boundary failure has its own recognisable shape. A model lands from the ML-platform peer with a broken `preprocessor_config.json` field; the inference engineer scrambles the day of the promo window. An eval regression fires on a live SLO; the on-call engineer and the eval peer each think the other owns mitigation; the SLO burns through its budget. An availability incident becomes a two-team shouting match because neither side agreed on who runs incident command when the signal is tokens and the resource is GPU. A security incident at the inference edge has an engineer who can patch and a peer who can approve, neither signed off on who does which.

All four failures are caused by the same thing: a peer interface that was *implicit* when it needed to be *written*. This chapter writes them.

By the end of this chapter you can:

- Author the interface contract with the ML-platform peer for model handoff.
- Author the eval-gate contract with the AI-eval and model-evaluation peers (chapter 05 of module 113 built part of this; this chapter situates it in the full program context).
- Author the SRE interface for incident ownership, SLO methodology, and error-budget accounting on an inference signal.
- Author the inference-edge security interface — what the security peer owns, what this role owns, how the two coordinate on prompt injection, model extraction, and model-weight supply chain.
- Avoid the classic failure modes for each interface.

## The contract shape — common across all four

Every peer interface in this chapter follows the same five-part shape (the one chapter 04 built for the performance peer, generalised):

1. **Boundary statement.** One sentence that names what each side owns; a short list of the shared 20 %.
2. **Mutual deliverables.** The named artefacts that cross the boundary — what, when, in what format.
3. **Ticket / request template.** The shape of a non-trivial request from one side to the other.
4. **Response SLA.** The agreed response time by priority; reviewed on the chapter-01 quarterly cadence.
5. **Shared incident / escalation path.** How a P0 that touches both sides is co-owned.

Each section below applies this frame to one peer.

## ML-platform peer (L30, AI Infrastructure family) — the model handoff

The ML-platform peer ships the artefacts this role serves. Their primary deliverable is the model registry, the training-to-serving handoff, and the self-service ML platform that other teams use to publish models. Our primary deliverable is the serving system that consumes what they publish.

### Boundary statement

The ML-platform peer owns the **model registry, model-artifact lineage, and the training-to-serving handoff pipeline**. This role owns **what happens from the moment the artifact enters the serving stack to the moment the response leaves the fleet**. The 20 % shared surface:

- **Artifact format and metadata.** They produce; we consume. The contract says *exactly* what metadata is required and in what format (chapter 05 of module 105 named this at depth; this chapter situates it program-wide).
- **Promotion decision.** They pass the capability-eval gate; we run the deploy gate (module 113 chapters 01 and 05). The promotion has two signatures.
- **Deprecation decision.** We signal usage decay (module 113 chapter 06); they act on the decision to stop publishing a model line.

### Mutual deliverables

| Deliverable | Direction | Format | When |
|---|---|---|---|
| Model artefact (weights + tokenizer + config + preprocessor configs if applicable) | ML-platform → this role | safetensors + `config.json` + `tokenizer.json` + `generation_config.json` + any preprocessor configs; signed; immutable; versioned | At model publication |
| Model metadata (lineage, training data summary, eval results, licence, intended use) | ML-platform → this role | structured record in the registry (registry schema signed by both sides) | At model publication |
| Quantisation artefact (if separate from base model) | ML-platform → this role | engine-specific format (TRT engine, AWQ, GPTQ, etc.) with a reference to the base; eval results for the quantised variant | At quantisation publication |
| Serving configuration (recommended engine args) | ML-platform → this role | a `serving_recipe.yaml` with starting engine args, topology suggestion, expected cost profile | At model publication |
| Deploy-event feedback (what engine version, what SLO achieved, what cost) | this role → ML-platform | a per-model feedback record posted to the registry | After promotion; updated weekly |
| Deprecation notice (usage falling below threshold, candidate for retirement) | this role → ML-platform | a deprecation proposal (module 113 chapter 06's shape) | Quarterly scan of usage |
| Model-regression evidence (an artefact does not pass the deploy gate) | this role → ML-platform | a one-page RCA citing module 113 chapter 04's post-mortem shape | Per failed promotion |

### Ticket / request template

The archetype request is "new model version, please promote." The template:

- Model name + version + revision.
- Registry record pointer.
- Target workload classes (chat, agent, long-context, code, etc.).
- Expected rollout primitive (canary, shadow, blue-green; from module 113 chapter 01).
- Pre-flight eval status (AI-eval peer and model-evaluation peer scores; chapter 05 of module 113).
- Target promote-by date.
- Rollback contact (which engineer).

### Response SLA

| Priority | Trigger | ML-platform response |
|---|---|---|
| P0 | Production model breaks because of a registry data issue | ack 1 hour; resolve same day |
| P1 | New model version blocked on a metadata gap | ack 1 business day; resolve within 3 business days |
| P2 | Deprecation proposal under review | ack 1 week; resolve within 2–4 weeks |

### Shared incident / escalation path

When a live incident attributes to a model-artifact issue (bad quantisation, corrupted tokenizer, missing preprocessor config), both tracks are on the incident channel. The inference engineer runs mitigation (rollback — module 113 chapter 01); the ML-platform peer owns the "why did a bad artefact pass our side's gate" portion of the RCA. The post-mortem action items split on that axis.

### Common failure modes

- **Metadata drift.** A new field appears in the registry; the serving side doesn't know about it; a promotion silently drops it. Fix: the registry schema is signed by both sides; schema changes go through the chapter-04-style ticket.
- **"Just push the weights."** The handoff bypasses the metadata pipeline in a hurry; lineage is lost; a later incident can't trace which training run produced the model. Fix: no promotion without a complete registry record.
- **Deprecation deadlock.** This role proposes deprecation; the ML-platform peer waits for consumer sign-off; the consumer tracks don't know until the warning fires. Fix: the module 113 chapter 06 deprecation runbook is the shared source of truth.

## Eval peers (L30, two peers across two families) — the eval-gate contract

The eval peers produce the evidence that gates model promotion. There are two of them — intentionally — and their work is not the same.

- **`model-evaluation-engineer`** (peer, L30, ML Engineering family) — benchmark / capability evals. The test is "does the model know things." Scores on MMLU, GSM8K, HumanEval, HELM, long-context benchmarks, task-specific capability suites, and the custom benchmarks the organisation maintains.
- **`ai-eval-engineer`** (peer, L30, AI Engineering family) — application-layer evals. The test is "does the model behave right in our product." Rubric-based LLM-as-judge scores, A/B outcomes on retention / conversion / satisfaction, safety / refusal / jailbreak evaluations against the organisation's policy, task-specific application rubrics the product team maintains.

Module 113 chapter 05 built the eval-gate contract end-to-end. This section gives the program-scope summary and the division of ownership this chapter needs; details defer there.

### Boundary statement

The eval peers own **what eval evidence exists and what it means**. This role owns **what the eval evidence gates on the serving side**. The 20 % shared surface:

- **Threshold setting.** The eval peers propose thresholds; this role accepts them against the product's SLO and the chapter-02 TCO. Thresholds are signed together.
- **Fixture back-feed.** Module 113 chapter 04's post-mortem produces regression fixtures; this role writes them; the eval peers integrate them into the eval set. Fixture provenance is tracked on both sides.
- **Live-eval regression ownership.** A live eval regression (an application rubric score drops in production) is a shared incident — this role mitigates (rollback, model swap), the AI-eval peer leads the root cause.

### Mutual deliverables

| Deliverable | Direction | Format | When |
|---|---|---|---|
| Benchmark eval results per model version | model-evaluation → this role | structured record (benchmark name, score, bootstrap CI, threshold, pass/fail) | At model publication |
| Application eval results per model version | ai-eval → this role | structured record (rubric name, score, threshold, pass/fail; per-workload-class) | At model publication |
| Pre-flight eval sign-off | both peers → this role | a signed record attached to the deploy ticket | At canary step 0 |
| Live-eval regression alert (production rubric drops) | ai-eval → this role | a paging ticket | On trigger |
| Fixture back-feed (post-mortem regression) | this role → eval peers | a test case in the agreed format | Per post-mortem |
| Eval-gate contract itself | shared | the signed contract from module 113 chapter 05 | Quarterly review |

### Response SLA

| Priority | Trigger | Eval-peer response |
|---|---|---|
| P0 | Live rubric regression on production; auto-rollback fired | ack 1 hour; engineer on the incident channel |
| P1 | Pre-flight eval blocks a deploy; question whether threshold is right | ack 1 business day; threshold re-review within 2 business days |
| P2 | Fixture back-feed integration | ack 1 week; integrate within 2 weeks |

### Shared incident / escalation path

A live-eval regression is the archetype. The inference engineer runs mitigation (module 113 chapter 01's auto-rollback, or a routing weight shift if automated rollback can't fire); the AI-eval peer owns root cause — which rubric regressed, what inputs caused it, is it a model issue or a prompt / data drift. The RCA has an "eval-regression characterisation" section the eval peer writes.

### Common failure modes

- **Threshold is a vendor's number.** The eval-peer threshold is a benchmark-paper number copied over; your product's needs are different. Fix: threshold re-set per product per workload class.
- **Fixture back-feed lost.** A post-mortem produces a regression fixture; nobody files it; the next similar regression is a surprise. Fix: the fixture back-feed is a required post-mortem step; this chapter's ledger tracks whether every post-mortem has a back-feed entry.
- **One eval peer, not two.** The team has only one peer (AI-eval, say) and the capability eval is implicit. Fix: name the gap; either hire/assign the second peer, or explicitly roll both roles into the one peer's charter with a written caveat that this is a known deficiency.

## SRE peer (L30, AI Infrastructure family) — methodology and incident command

The SRE peer provides the reliability methodology this role specialises for inference signals. Module 113 chapter 04 and module 110 chapter 07 both name the boundary; this chapter integrates it program-wide.

### Boundary statement

The SRE peer owns **SRE methodology primitives** — the definition of SLI / SLO / error budget, multi-window multi-burn-rate alert design, incident command role structure, blameless post-mortem framing, action-item discipline, on-call rotation design patterns. This role owns **the inference-specific application of that methodology** — the token-granularity SLOs, the inference axis set, the inference runbooks, the inference-flavoured RCA (modules 110 and 113).

The 20 % shared surface:

- **Availability incident command.** For an outage that is a straightforward availability incident (fleet-wide, consumer tracks can't reach the API), the SRE peer leads incident command. For an inference-flavoured incident (semantic regression, cache-hit cliff, preemption storm), this role leads.
- **SLO contract review.** SLO definitions and thresholds are owned by this role; the methodology under them is signed with the SRE peer.
- **On-call rotation design.** This role chooses the rotation; the SRE peer reviews it as a reliability pattern.

### Mutual deliverables

| Deliverable | Direction | Format | When |
|---|---|---|---|
| SLO contract | this role → SRE peer | the signed contract from module 110 | Quarterly review |
| Burn-rate alert definitions | this role → SRE peer | the Prometheus rules; reviewed for burn-rate math | At alert authoring and annual |
| Incident response methodology | SRE peer → this role | the organisation's incident-command playbook | Reference |
| Blameless post-mortem template | SRE peer → this role | the organisation's post-mortem shape; adapted for inference artefacts (module 113 chapter 04) | Reference |
| Availability-incident command | SRE peer runs | the SRE peer's incident-command process | Per availability incident |
| Inference-flavoured incident command | this role runs | module 113 chapter 04's process | Per inference incident |
| On-call rotation plan | this role → SRE peer | the chapter-01 rotation; reviewed quarterly | Quarterly |

### Response SLA

| Priority | Trigger | SRE-peer response |
|---|---|---|
| P0 | Availability incident (fleet-wide) | immediate; SRE peer runs incident command |
| P1 | SLO contract change request; burn-rate ratio question | ack 1 business day; resolve within 1 week |
| P2 | On-call rotation review; methodology consultation | ack 1 week; resolve within 2–4 weeks |

### Shared incident / escalation path

A fleet-wide availability incident defaults to SRE-peer incident command; this role provides the inference-specific triage and mitigation. An inference-flavoured incident (semantic regression, cache-hit cliff, NCCL hang, silent accuracy regression) defaults to this role's incident command; the SRE peer is on the channel for methodology support. The call about *which* mode applies is made in the first ten minutes of the incident by whichever on-call engineer is the first to see it.

### Common failure modes

- **Methodology drift.** This role reinvents an SLO or an alert rule in a way that doesn't follow the SRE peer's methodology. Fix: the SRE peer is a signer on the SLO contract and the alert definitions.
- **Incident-command confusion.** Both tracks think the other is leading; neither takes the commander role. Fix: the incident template at page time names the default commander based on alert class; the matrix lives in chapter 01's paging matrix.
- **Methodology-only RCA.** An RCA that quotes SRE methodology without the inference-specific axis-set bisection; or the inverse. Fix: the module 113 chapter 04 RCA template integrates both; both leads sign.

## Security peer (L35, AI Infrastructure family) — inference-edge security

The security peer owns adversarial-defence depth at the inference edge. The boundary is real: both roles sit at L35 in the same family, and the inference-edge surface has specialisations that neither role can handle alone.

### Boundary statement

The security peer owns **adversarial defence research and primitives at the inference edge** — prompt injection detection depth, model-extraction attacks, query-rate control against extraction, model-weight supply-chain signing design, jailbreak-resistance evaluation, red-team methodology. This role owns **production serving of a fleet that includes security controls** — the prompt-injection detector deployed in the request pipeline, the rate-limiter the security peer specified, the signed-weights loader the security peer designed, the DLP integration at the edge, the audit logs, the data-boundary artefacts (chapter 03's data boundary dimension consumes these).

The 20 % shared surface:

- **Policy evaluation at serving.** The security peer sets policy; this role operates the enforcement. "Block requests matching pattern X" is their decision; where in the pipeline it runs and what latency it adds is ours.
- **Weight-signing and supply chain.** They design the trust chain; we operate the loader that verifies.
- **Incident response for edge attacks.** Jailbreak storm, prompt-injection-at-scale, extraction attempts — they lead investigation; we operate the mitigations (rate-limit changes, model-side filter deployment, routing changes).

### Mutual deliverables

| Deliverable | Direction | Format | When |
|---|---|---|---|
| Edge-security policy | security peer → this role | signed policy document | Quarterly review |
| Prompt-injection detector | security peer → this role | a service or library + a threshold recommendation | At publication; updated per model class |
| Signed-weight manifest | security peer → this role | signing root + verification protocol | Per release |
| Rate-limit tier recommendations (anti-extraction) | security peer → this role | per-tier RPS / TPS / daily token caps | Quarterly; per model class |
| Serving telemetry (request patterns indicative of attacks) | this role → security peer | agreed-upon log slice; privacy-respecting | Continuous |
| Red-team evidence (jailbreak coverage, policy compliance) | security peer → this role | before major model promotion | Per promotion |
| Edge-incident runbook | shared | combines both tracks' primitives | Quarterly review |

### Response SLA

| Priority | Trigger | Security-peer response |
|---|---|---|
| P0 | Active attack (extraction, jailbreak storm, data-exfil pattern) | ack immediate; incident co-command |
| P1 | New model's red-team evidence needed before promotion | ack 1 business day; resolve within 1 week (depends on model surface) |
| P2 | Policy revision; detector-threshold tuning | ack 1 week; resolve within 2–4 weeks |

### Shared incident / escalation path

An edge-security incident defaults to the security peer's incident command; this role operates the mitigations. The RCA has a "edge-attack characterisation" section the security peer writes, and a "mitigation deployment" section this role writes. Action items split on the axis.

### Common failure modes

- **Detector in theory, not in production.** The security peer ships a prompt-injection detector; this role never deploys it because the latency cost was not budgeted. Fix: the deliverable includes a latency budget; the deploy ticket treats the detector like any other change (module 113 chapter 01).
- **Signed weights optional.** The weight-signing protocol exists but the loader in production is still the unsigned path because signing is seen as slow. Fix: signing is a pre-production requirement, not a nice-to-have; the chapter-04 model-registry artefact includes the signature.
- **Rate-limit tiers chosen by product only.** Anti-extraction tiers exist but are set by product for conversion, not by security peer for extraction defence. Fix: tiers are co-set; the security peer's input is per model class.

## The cross-peer meeting cadence

The chapter-01 weekly SLO review has four peer tracks on rotation. The rotation (rough shape; adapt to the organisation):

- **Week 1.** SRE peer attends. Focus: availability SLOs, error-budget status, on-call rotation hygiene.
- **Week 2.** Eval peers attend. Focus: semantic-equivalence SLO status, live-eval regressions, fixture back-feed status.
- **Week 3.** Performance peer attends. Focus: kernel-regression ledger review, engine-upgrade coordination.
- **Week 4.** ML-platform peer attends; security peer attends (in alternating months with the performance peer). Focus: model-handoff pipeline, deprecation ledger; or edge-security posture and detector performance.

The quarterly cadence has all five peer leads present for the migration-roadmap review.

## Common cross-peer failure modes

- **Peer interface is implicit.** Everyone thinks they know what the other side owns; nobody has written it down; the first incident exposes the gap. Fix: every peer interface is a signed, dated document. Quarterly review.
- **Interface written but not consulted.** The document exists on a wiki; nobody has read it in six months; it has drifted. Fix: the chapter-01 quarterly cadence re-reads each peer contract; outdated clauses are updated.
- **No reverse direction.** This role files to peers; peers do not file back. Fix: reverse tickets are expected (chapter 04 named this for the performance peer; same for the others).
- **Peer-lead turnover.** A peer-track lead changes; the contract is inherited without re-signature; the new lead doesn't know what they signed. Fix: on turnover, the new lead re-signs within the first quarter.
- **Methodology-SRE conflation.** This role treats the SRE peer as "the on-call peer"; the real contract is methodology. Fix: the boundary statement is specific — methodology primitives, not operation.

## Summary

- Every peer interface uses the same five-part shape: boundary statement, mutual deliverables, ticket template, response SLA, shared incident path.
- ML-platform peer: handoff contract on artefact format, metadata, lineage, signing; deliverable table names what crosses which way; deprecation is bidirectional.
- Eval peers (two): capability eval from model-evaluation (L30, ML Engineering); application eval from ai-eval (L30, AI Engineering). Thresholds co-set; fixture back-feed is bidirectional; live-eval regression is a shared incident with named mitigation and named RCA ownership.
- SRE peer: methodology primitives (SLI / SLO / burn-rate / incident command / post-mortem) in; inference-specific application (axis set, runbooks, inference RCA) out; availability-incident command defaults to the SRE peer, inference-incident command defaults to this role.
- Security peer: policy and primitives (detectors, signing, rate-limit tiers, red-team evidence) in; production operation of the controls out; edge-incident command defaults to the security peer.
- Weekly SLO review rotates one peer per week; the quarterly review has all five peer leads for the migration roadmap.
- Common cross-peer failure modes: implicit interfaces, drift, no reverse direction, lead-turnover blind spots, methodology conflation. The chapter-01 quarterly cadence prevents all five.
- Chapter 06 steps out of the peer interface to the engineer's own trajectory — what senior / staff / architect / principal looks like after this track, and what certifications and open-source contribution plug into which jumps.
