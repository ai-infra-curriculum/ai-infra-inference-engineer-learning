# AI Infrastructure · AI Inference Engineer — Learning Repository

<!-- aicg:site-banner -->
> 🎓 Part of the free, open-source **AI Career Curriculum** ecosystem — [Infrastructure](https://github.com/ai-infra-curriculum) · [ML Engineering](https://github.com/ml-engineering-curriculum) · [AI Engineering](https://github.com/ai-engineering-curriculum) · [Governance](https://github.com/ai-governance-curriculum) · [Startup](https://github.com/ai-startup-curriculum) · [Workforce](https://github.com/ai-workforce-curriculum).
<!-- /aicg:site-banner -->

<!-- aicg:sponsor -->
> 💜 **[Sponsor this curriculum](https://github.com/sponsors/ai-infra-curriculum)** — sponsorships keep the whole open-source AI Career Curriculum free and moving.
<!-- /aicg:sponsor -->

**Role level:** 35 (production-serving specialist — AI Infrastructure family)

Owns the production operation of inference engines in a GPU fleet: inference-engine configuration and operation (vLLM, SGLang, TensorRT-LLM, NVIDIA Dynamo), request scheduling + continuous batching + chunked prefill, KV-cache management as a runtime concern, multi-model / multi-tenant / multi-adapter serving, distributed inference topologies and disaggregated prefill-decode, speculative-decoding deployment, GPU-fleet autoscaling and resilience, inference-specific observability and SLO engineering, OpenAI-compatible API design and streaming / structured-output / tool-calling protocols, multimodal and emerging-workload serving, production safety / rollouts / inference incident response, and the cross-functional inference-platform program at team scope.

> **Status**: planned. This bootstrap packet lands the requirements catalog and the module / project plan authored from [`.aicg/curriculum-plan.json`](.aicg/curriculum-plan.json). Lessons and projects will be drafted by subsequent autonomous content cycles. Live posting evidence was deferred (WebSearch / WebFetch was not granted in the bootstrap session) — see [`JOB_REQUIREMENTS.md`](JOB_REQUIREMENTS.md) `Status`.

## What you get

Planned commitment: **214 hours across 14 modules** + **120 hours across 3 projects** = **~334 hours**.

- **Module 101** — the inference engineer's position on the infra ladder, engine capability map (vLLM / SGLang / TensorRT-LLM / Dynamo / Triton), inference vocabulary (TTFT / TPOT / ITL / cache-hit / preemption / goodput), GPU hardware landscape at operator-depth.
- **Module 102** — inference-engine internals and operational fluency: source-reading depth of vLLM / SGLang / TensorRT-LLM, failure-mode triage, deprecated-vs-maintained landscape (TGI archived in 2026).
- **Module 103** — request scheduling and continuous batching: iteration-level scheduling, chunked prefill (SARATHI-shape), preemption policy, SLO-driven scheduler tuning.
- **Module 104** — KV-cache management as a runtime concern: PagedAttention block-table tuning, RadixAttention prefix caching, LMCache / Mooncake tiered KV stores, long-context budgeting, cross-tenant sharing safety.
- **Module 105** — quantization deployment and model-format operations: FP8 / FP4 / INT4 / AWQ / GPTQ / SmoothQuant serving, safetensors / GGUF / TRT-engine conversion, accuracy-regression gates.
- **Module 106** — multi-model, multi-tenant, and multi-adapter serving: S-LoRA / Punica multi-LoRA, MIG / MPS / DRA GPU sharing, cost-aware router across self-hosted + external providers.
- **Module 107** — distributed inference topologies and disaggregated serving: TP / PP / EP as deployment topologies, DistServe / Splitwise / NVIDIA Dynamo, cross-node KV transfer, MoE expert parallel deployment.
- **Module 108** — speculative-decoding deployment and operation: draft-target, Medusa, EAGLE; acceptance-rate monitoring and draft-model lifecycle.
- **Module 109** — GPU fleet autoscaling and resilience: NVIDIA GPU Operator, DCGM, HPA on inference custom metrics, Karpenter, warm pools, spot preemption, multi-region failover.
- **Module 110** — inference observability, SLO engineering, and cost per inference: TTFT / TPOT / ITL / cache-hit / preemption axes, Prometheus / Grafana / OTel, cost per 1M tokens, throughput-vs-latency Pareto.
- **Module 111** — inference API design and protocols: OpenAI-compatible API, SSE / WebSocket streaming, structured output (Outlines / XGrammar), tool calling, Model Context Protocol (MCP) server-side.
- **Module 112** — multimodal, diffusion, and emerging-workload serving: VLM, image / video / 3D diffusion, streaming ASR / TTS, reasoning-model serving, agentic workloads, edge compare-point.
- **Module 113** — production safety, rollouts, and incident response for inference: canary / shadow / blue-green, response-diffing, inference runbooks, incident RCA, model-deprecation runbook.
- **Module 114** — cross-functional inference-platform leadership: build-vs-buy RFC, vendor eval, peer-interface contracts, career trajectory.
- **Three projects** — a production single-model inference platform on vLLM or SGLang (40 h); a multi-model, multi-adapter fleet with cost-aware router (40 h); a disaggregated prefill-decode serving system at multi-node scale (40 h).

## Level ladder — where this track fits

- **level 10** — [`ai-infra-junior-engineer-learning`](https://github.com/ai-infra-curriculum/ai-infra-junior-engineer-learning) (assumed engineering-craft prerequisites)
- **level 20** — [`ai-infra-engineer-learning`](https://github.com/ai-infra-curriculum/ai-infra-engineer-learning) (assumed K8s / cloud base)
- **level 25** — [`ai-infra-mlops-learning`](https://github.com/ai-infra-curriculum/ai-infra-mlops-learning) (peer, MLOps automation this track plugs into)
- **level 30 (peers)** — [`ai-infra-ml-platform-learning`](https://github.com/ai-infra-curriculum/ai-infra-ml-platform-learning) (ML platform / registry handoff), [`ai-infra-sre-learning`](https://github.com/ai-infra-curriculum/ai-infra-sre-learning) (SRE methodology), [`ai-eval-engineer-learning`](https://github.com/ai-engineering-curriculum/ai-eval-engineer-learning) (eval release-gate), [`fine-tuning-engineer-learning`](https://github.com/ml-engineering-curriculum/fine-tuning-engineer-learning), [`model-evaluation-engineer-learning`](https://github.com/ml-engineering-curriculum/model-evaluation-engineer-learning)
- **level 35 (this + peers)** — **`ai-infra-inference-engineer-learning`** (this track — production serving operation), [`ai-infra-performance-learning`](https://github.com/ai-infra-curriculum/ai-infra-performance-learning) (peer — kernel-authoring craft; **the single most important peer boundary**), [`ai-infra-security-learning`](https://github.com/ai-infra-curriculum/ai-infra-security-learning) (peer — inference-edge security)
- **level 40** — [`ai-infra-architect-learning`](https://github.com/ai-infra-curriculum/ai-infra-architect-learning) (cross-stack architecture), [`ai-infra-team-lead-learning`](https://github.com/ai-infra-curriculum/ai-infra-team-lead-learning) (people management)
- **level 45** — [`ai-infra-senior-architect-learning`](https://github.com/ai-infra-curriculum/ai-infra-senior-architect-learning) (multi-year strategy)
- **level 50** — [`ai-infra-principal-engineer-learning`](https://github.com/ai-infra-curriculum/ai-infra-principal-engineer-learning) + [`ai-infra-principal-architect-learning`](https://github.com/ai-infra-curriculum/ai-infra-principal-architect-learning) (principal scope)

See [`CURRICULUM.md`](CURRICULUM.md) for the ownership rule this track applies and [`PREREQUISITES.md`](PREREQUISITES.md) for the assumed lower-level curriculum.

## Layout

```
ai-infra-inference-engineer-learning/
├── lessons/mod-XXX-*/        modules with lectures, exercises, labs, quizzes
├── projects/project-XXX-*/   multi-module capstones
├── CURRICULUM.md             role-level coverage map
├── PREREQUISITES.md          assumed entry skills
├── VERSIONS.md               release history
├── JOB_REQUIREMENTS.md       requirements catalog with cited authoritative references
├── .aicg/job-requirements.json    machine-readable requirements & references
├── .aicg/curriculum-plan.json     machine-readable module / exercise / project plan
└── README.md                 this file
```

## Paired Solutions Repo

[`ai-infra-inference-engineer-solutions`](https://github.com/AI-Infra-Curriculum/ai-infra-inference-engineer-solutions) carries the reference implementations.

---

<!-- aicg:maintained-by -->
Maintained by [VeriSwarm.ai](https://veriswarm.ai)
