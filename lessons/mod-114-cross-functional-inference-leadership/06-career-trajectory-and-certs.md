# Chapter 06 — Career Trajectory, Certifications, and Open-Source Contribution

> Modules 101–113 and chapters 01–05 of this module built the inference-engineer craft at L35. This chapter zooms out to the ladder above and beside it — what senior / staff / architect / principal look like next, what certifications matter in enterprise hiring, and how upstream contribution to the open-source engines positions the engineer for the next step.

## Motivation

By L35 an engineer is a production-serving specialist with team-scope program ownership. The question that follows — "what do I want next" — has several plausible answers. Four defensible trajectories: deeper specialism (senior IC, principal engineer), broader scope (staff → architect), different depth (cross-training to the peer craft at L35), or people (team lead → manager). Each has a different investment profile, a different visibility pattern, and a different set of load-bearing credentials.

This chapter is not career advice; it is a map. The engineer makes the choice; the chapter names what is on the map so the choice is informed.

By the end of this chapter you can:

- Place the inference-engineer L35 role on the broader ladder — what L40 and L50 look like in the same family and adjacent families.
- Decide which certifications are load-bearing for enterprise hiring in inference infrastructure (NVIDIA DLI, Kubernetes CKA / CKAD, cloud-provider ML cert) and which are decorative.
- Plan an open-source contribution path through vLLM / SGLang / TensorRT-LLM / Dynamo that compounds craft and visibility.
- Decide whether the next move is specialism (senior → staff), scope (staff → architect), craft-cross (into the performance peer at L35), or people (team lead).

## The ladder — a map, not a destination

Role titles vary by organisation. The curriculum uses a level-based ladder (L10 → L50) to normalise. The relevant ladder for an L35 inference engineer:

- **L35 — this role.** Production-serving specialist with team-scope program ownership. Owns the inference platform program at team scope. Peer to performance (L35) and security (L35) in the same family.
- **L40 — next up, two tracks diverge.**
  - [`ai-infra-architect`](https://github.com/ai-infra-curriculum/ai-infra-architect-learning) — cross-stack architecture at team-of-teams scope. The IC path that stays close to the systems.
  - [`ai-infra-team-lead`](https://github.com/ai-infra-curriculum/ai-infra-team-lead-learning) — people management of an inference-platform team. The management path.
- **L45 — senior architecture.**
  - [`ai-infra-senior-architect`](https://github.com/ai-infra-curriculum/ai-infra-senior-architect-learning) — multi-year infrastructure strategy. Still IC, but at organisation scope.
- **L50 — principal IC / principal architect.**
  - [`ai-infra-principal-engineer`](https://github.com/ai-infra-curriculum/ai-infra-principal-engineer-learning) — principal IC; cross-organisation technical direction.
  - [`ai-infra-principal-architect`](https://github.com/ai-infra-curriculum/ai-infra-principal-architect-learning) — principal architecture; cross-organisation strategy.

Two side-moves that are common at this level:

- **Cross-training to the performance peer (L35).** Kernel-authoring craft. A ~6–18 month investment to be productive; it opens the kernel side of the shared surface the chapter 04 interface describes.
- **Cross-training to inference-edge security (L35).** Specialisation in adversarial defence at the inference edge. Less common as a move from inference engineer because the domain crossover is less than the performance peer's, but a plausible path for someone interested in the security side.

Deciding by taste:

- Enjoyed modules 102 / 107 and the performance-peer boundary conversation? → the performance peer craft is plausible.
- Enjoyed module 114 chapters 01 / 02 (program, RFC)? → architecture or team-lead is plausible.
- Enjoyed module 113 chapter 04 (incident command, RCA, post-mortem)? → SRE peer specialisation at L30→L35 is plausible.
- Enjoyed the cross-peer coordination (module 114 chapter 05)? → architect or team-lead is plausible.
- No strong taste for any of these — stay at L35 and go deeper. An L35 IC with eight years of inference-engineering operations against the flagship workloads of a well-known product is a *very* strong market position.

## Senior engineer / staff — the specialism-deeper path

Not every next step is a level change. Within L35, the trajectory from "capable of owning the platform at team scope" to "the person the organisation trusts to own the biggest platform transitions" is a 2–4 year growth curve. The markers that compound:

- **Portfolio of multi-week migrations landed.** Chapter 01 names the migration plan; chapter 02 the RFC. A portfolio of three to five of these, signed off and shipped, is the single strongest signal of staff-level IC craft in this role.
- **Post-mortems authored and acted on.** Module 113 chapter 04's post-mortem artefact. A set of ten to twenty, with the fixture back-feeds visible in the eval set, demonstrates incident-command maturity.
- **Cross-peer sign-offs collected.** The chapter 04 / 05 interface contracts have signers. A senior engineer has signed many of them, and has been the counter-signer on peer-track artefacts in return.
- **Upstream contributions to the engines.** See the open-source section below.
- **Written artefacts the organisation uses.** A runbook set, an RFC template, an SLO contract, a vendor scorecard — authored by this engineer, reused by successors, maintained on a cadence.

The management-side analogue — team lead / engineering manager — is covered in the L40 team-lead track and is a different investment. The specialism-deeper path is an IC commitment.

## Staff → architect — the scope-wider path

The architect transition is from "one inference-platform program" to "the inference discipline across several programs or a whole organisation." The markers:

- **Multi-program coordination.** The cross-peer interfaces in chapter 05 are multiplied across several platform teams; the architect writes the standards those platform teams consume.
- **Technology-direction decisions.** The chapter-02 build-vs-buy RFCs are now organisation-wide: "should we have one self-host platform or several"; "should we move the long tail to a vendor"; "what is our hardware direction for the next 24 months."
- **Peer-track alignment at the next level.** Where the L35 peer interface is with the team-scope performance peer, the architect interface is with the architect-scope performance / ML-platform / SRE / security peers.
- **Sponsor conversations.** Executives who sign RFCs rather than team leads who commit sprints.

The architect transition typically requires an organisation large enough to have multiple inference platforms. In a smaller organisation, the staff IC path stays L35 and grows into principal (see below) rather than architect.

## Principal — the organisation-direction path

L50 principal IC and principal architect are distinct roles — the principal engineer is still writing load-bearing code and systems; the principal architect is almost entirely proposal, standard, and decision-making work. Both require:

- **Industry presence.** Not optional. Conference talks (OSDI, MLSys, NeurIPS systems track, PyTorch Conference, KubeCon, GTC); upstream maintainer status in one of the engines; published writing (blog, book, reports) that frames the field. Principal is a *recognised* seniority, not just a company-internal one.
- **Decade of track-recognised work.** Shipping the engines, not just serving them.
- **Succession in your wake.** Several L35s or L40s who learned the craft from you and are now running their own platforms.

Principal is the next-decade conversation for an engineer entering L35; the first career-chapter decision is between specialism-deeper, scope-wider, craft-cross, or people.

## Certifications — what is load-bearing, what is decorative

Certifications matter in enterprise hiring exactly to the extent that they let a hiring manager or a procurement team check a box. Some are load-bearing — they open doors or make a candidate cost less to approve. Some are decorative — they signal interest but do not clear a bar. The accurate view at L35 for inference engineering in 2026:

### Load-bearing

- **Kubernetes CKA (Certified Kubernetes Administrator).** The platform runs on Kubernetes in almost every enterprise inference team; CKA is the credential that proves day-two operation. Publisher: CNCF / Linux Foundation. <!-- needs-research: confirm the current CKA curriculum and exam format at <https://training.linuxfoundation.org/certification/certified-kubernetes-administrator-cka/>; revise the summary if the curriculum materially shifts. -->
- **Kubernetes CKAD (Certified Kubernetes Application Developer).** The companion credential for building workloads on Kubernetes. Less strictly required than CKA for a platform role, but valued where the inference engineer is building the serving stack rather than operating it.
- **Cloud-provider ML certification** — one of:
  - **AWS Certified Machine Learning — Specialty** (consumer of SageMaker / Bedrock / Inferentia). <!-- needs-research: AWS cert names and exam content revise annually — check <https://aws.amazon.com/certification/> at refresh time. -->
  - **Google Cloud Professional Machine Learning Engineer** (Vertex AI, TPU). <!-- needs-research: <https://cloud.google.com/learn/certification/machine-learning-engineer>. -->
  - **Microsoft Certified: Azure AI Engineer Associate** (Azure OpenAI Service, Azure ML). <!-- needs-research: <https://learn.microsoft.com/en-us/certifications/azure-ai-engineer/>. -->
  - Pick the one aligned with your organisation's primary cloud; a second is nice-to-have, not load-bearing.
- **NVIDIA-DLI (Deep Learning Institute) certifications** — specifically the ones that touch inference deployment:
  - Instructor-led or online courses on triton-inference-server, TensorRT-LLM, and performance optimisation. NVIDIA issues completion certificates and some have formal exam versions. <!-- needs-research: NVIDIA's DLI catalog is at <https://www.nvidia.com/en-us/training/> and the specific inference-aligned tracks rotate; confirm the inventory at refresh time. -->
  - Not strictly a certification in the CKA sense, but recognised in enterprise hiring where NVIDIA stack expertise is valued (which is most places).

### Supporting (not load-bearing, useful)

- **Kubernetes CKS (Certified Kubernetes Security Specialist).** Only useful if security is part of the role scope or a near-term direction.
- **Cloud-provider architect certifications** (AWS Solutions Architect Professional, Google Professional Cloud Architect, Azure Solutions Architect Expert). Supports the L40 architect transition path; less important for L35 IC.
- **CNCF-ecosystem certifications** (Prometheus Certified Associate, Istio certified — where available). Each is useful if the related technology is the primary observability or service-mesh stack.

### Decorative (interest signals, not credentials)

- **Vendor-specific certifications for inference APIs** (an "Anthropic Claude Certified Developer" badge or similar) — these typically signal product familiarity, not infrastructure depth. Useful on a resume for breadth; not load-bearing for an infrastructure role.
- **Machine-learning foundational certifications** that do not touch serving (fundamentals, data-science). Appropriate for an earlier career level; not specifically load-bearing at L35.

### How to sequence investments

- **Year 1 of entering the role**: whichever cloud-provider ML cert matches your organisation; CKA if Kubernetes operations are a significant part of your day.
- **Year 2**: Add NVIDIA DLI inference-aligned courses; they land close to the daily craft.
- **Year 3+**: Follow the shape of the career decision (architect → architect-level cloud certs; principal → de-emphasise certifications and emphasise visibility).

Note: a certification is not a substitute for the artefacts in this track. A hiring manager shown a signed RFC, a vendor scorecard, a runbook set, and a certification will value the artefacts above the certification. Certifications let the resume cross a filter; artefacts win the role.

## Open-source contribution — the long-game investment

Contribution to the open-source engines (vLLM, SGLang, TensorRT-LLM, NVIDIA Dynamo, Triton Inference Server, llama.cpp, MLC-LLM) is the most durable visibility investment available to an inference engineer. The craft is public; contributions are public; a maintainer-level presence on one of the engines is a career-lifetime credential.

### The contribution ladder, inference-engine flavoured

- **Level 1 — bug-fix PR against an engine.** A single well-authored PR against a reproducible bug. Entry-level. Teaches the engine's contribution process, the review culture, the test-infrastructure requirements. Realistic in a weekend if the bug is small.
- **Level 2 — small-feature PR.** A new engine arg, a small performance improvement in a non-kernel path, a bug-fix in a scheduler module, a runbook-visibility improvement. More invested; demonstrates ownership of a surface.
- **Level 3 — recurring contributor.** A streak of PRs across multiple surfaces. Review comments on others' PRs. Attendance at the project's community meetings. Named in the project's sustaining-contributor lists where they exist.
- **Level 4 — committer / maintainer.** Review and merge rights. Owning a module or a subsystem. The project's commit-bit holder for an area.
- **Level 5 — core / architecture.** Setting direction. Writing or shepherding RFCs. The project's technical steering.

Most inference engineers who contribute reach L2–L3 and stop, which is fine. L4 is a significant investment and a signal; L5 is a career-shaping move.

### Which engine to pick

- **vLLM** — the largest open-source inference-engine community in 2026; highest-impact for visibility; also the busiest, so a specific niche inside it (scheduler, KV cache, speculative decoding, structured output) is where ownership is reachable.
- **SGLang** — smaller community, faster-moving in some areas (RadixAttention tuning, structured generation), more approachable for first-time contribution with less competition for review attention.
- **TensorRT-LLM** — NVIDIA-stewarded; contribution is possible and welcomed; the review bar is high and the direction is NVIDIA-set. Good path if your organisation is deep on NVIDIA.
- **NVIDIA Dynamo** — newer; smaller community; active area for disaggregated prefill-decode work.
- **Triton Inference Server** — long-running NVIDIA project; approachable for contributions around backends and HTTP handling.
- **llama.cpp / MLC-LLM** — different audience (edge, on-device); same craft discipline; different certification value in enterprise hiring.

A workable pattern: pick one engine whose community you already read, make three to five small PRs there before deciding whether to specialise further. Switching engines is cheap early and expensive late.

### Compounding craft and visibility

Open-source contribution compounds in a way certifications do not:

- Every PR against an engine teaches the production engineer that engine's internals at a depth no documentation covers.
- Review comments on others' PRs build peer-track relationships that outlast any one organisation's tenure.
- A public commit history is a portable artefact; a certification lives on an employer's reimbursement line.
- When the engine you contribute to gains traction in enterprise adoption, your specific expertise becomes more valuable — visibility and expertise reinforce each other.

The investment is slow — a first PR reviewed in a few weeks, a steady contribution pattern over a year, maintainer-level trust over several — but the compounding is substantial.

## The track-wide reading list, re-framed for the next step

The track's `resources.md` files (for this module and the earlier modules) contain the primary / citable references for the craft. For the career-trajectory step, three categories are additional:

- **Industry venues.** OSDI (serving-systems papers), MLSys (ML systems), NeurIPS systems track, PyTorch Conference, KubeCon, GTC, Ray Summit. Attend when possible; read proceedings when not. The staff → principal trajectory runs through these venues.
- **Engine project community meetings.** vLLM has public office hours; SGLang has a Discord; TensorRT-LLM has NVIDIA-hosted calls. Attending a community meeting once a quarter is a cheap, high-information commitment.
- **Writing.** A technical blog that tracks your own work — a runbook authored, an RFC lessons-learned, a migration retrospective — is both professional development and a hiring artefact. The staff → principal step depends on writing almost as much as on code.

## A decision aid — a one-page template for your own plan

For your own use, a one-page plan (fill it in once a quarter):

- **Current state.** Role level; years at level; primary systems owned; recent artefacts shipped.
- **Target state at horizon.** 12-month, 24-month, 60-month. One role level per horizon; one dominant artefact to ship per horizon.
- **Specialism, scope, craft-cross, or people.** Pick one for the near horizon; note the others as later options.
- **Certifications queued.** Which, by when, with what investment. Not a long list.
- **Open-source engine commitment.** Which engine; what niche; three to five realistic contributions for the next 12 months.
- **Visibility steps.** One venue to attend; one piece to write; one community meeting to join.
- **Mentors / sponsors.** Who you are learning from; who is advocating for your next step inside your organisation.

Review the plan every three months. Rewrite it once a year. The plan is a working document, not a commitment contract.

## Common anti-patterns in career planning

- **Certification shopping as the career plan.** The resume has eight certs; the portfolio of shipped artefacts is empty. Fix: one load-bearing cert, then invest in artefacts.
- **Open-source as a demo project.** A single flashy repo that never gets contributed to upstream engines. Fix: contribute to an engine others use; stop building parallel toys.
- **Scope-change without a sponsor.** Requesting a staff promotion based on the current scope rather than a widened one. Fix: get sponsor agreement on the widening before expecting the promotion.
- **People-path as the escape from craft.** The engineer bored of technical work moves to management without actually enjoying people work. Fix: honest self-assessment of what energises you; the team-lead track is a different craft, not a step away from a craft.
- **Principal-as-destination.** Aiming for "principal" as a title while under-investing in the visibility and succession work that the role actually requires. Fix: principal is a recognised seniority; you earn it by consistently producing recognisable work, not by asking for the title.

## Summary

- The L35 role has four plausible next steps: specialism-deeper (senior / staff IC), scope-wider (staff → architect at L40 → senior architect at L45 → principal at L50), craft-cross (performance peer or security peer at L35), or people (team lead at L40). Each is a different investment and visibility profile.
- Within L35, staff-level maturity comes from a portfolio of migrations landed, post-mortems acted on, cross-peer sign-offs collected, upstream contributions made, and written artefacts other engineers reuse. A senior L35 IC in this role is an extremely strong market position; moving up is a choice, not an obligation.
- Load-bearing certifications for enterprise hiring: Kubernetes CKA (and often CKAD); one cloud-provider ML certification matching your organisation's primary cloud; NVIDIA DLI inference-aligned courses. Supporting (not load-bearing): CKS, cloud architect certs, CNCF-ecosystem certs. Decorative: vendor API badges, ML foundational certs (at this level).
- Certifications let a resume cross a filter; signed artefacts (RFCs, vendor scorecards, runbooks) win the role.
- Open-source contribution to the engines (vLLM, SGLang, TRT-LLM, Dynamo, Triton) is the most durable visibility investment. The contribution ladder is bug-fix → small-feature → recurring contributor → committer → core. Most reach L2–L3; L4 is a signal; L5 is career-shaping.
- Pick one engine whose community you already read; make three to five small PRs before specialising. Switching engines is cheap early, expensive late.
- Visibility compounds through industry venues (OSDI, MLSys, GTC, KubeCon), engine community meetings, and writing. Principal is recognised, not requested.
- Keep a one-page career plan, review quarterly, rewrite annually. Avoid certification shopping, demo-project open-source, scope-change without a sponsor, people-path-as-escape, and principal-as-destination.
- This closes the module and the track. The next module — if one exists for you — is the L40 track or the peer craft at L35; the real "next module" is the production work in which this module's artefacts live.
