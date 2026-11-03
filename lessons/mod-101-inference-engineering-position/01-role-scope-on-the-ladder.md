# Chapter 01 — The Inference Engineer's Position on the Ladder

> You run inference engines in production. Everyone around you touches the model from a different side. This chapter draws the lines.

## Motivation

An inference platform team is surrounded by other roles that all look like they could own a given problem. A model-swap outage *could* be a serving bug, *could* be a bad quantization, *could* be a kernel regression, *could* be a routing / autoscaling bug, *could* be an eval failure that never gated the release, *could* be a prompt-injection incident. If the inference team tries to own all of it, the team becomes the queue of last resort. If it owns too little, outages fall through the cracks between it and the peer tracks.

This module exists to fix that routing problem *before* you touch an engine, so that in modules 102–114 you know whether a given problem is yours or someone else's.

The AI Infrastructure family levels used in this chapter are the ones defined in [`CURRICULUM.md`](../../CURRICULUM.md) and the sibling-track READMEs. Numbers like "level 35" are the family-wide seniority grid, not years of experience.

## The one-line charter

**You own the production operation of inference engines.** You keep a GPU fleet serving a stated set of models at a stated SLO (latency, availability, correctness) at a defensible cost. You do not own the kernels those engines dispatch to, the models those engines serve, the applications that call those engines, or the policy that decides whether a given model may be served at all.

Everything else in this chapter is a corollary of that line.

## The peer boundaries, left to right

The six boundaries you must be able to defend to a new hire in week one:

### 1. `ai-infra-performance` — peer, level 35, **the single most important boundary**

Share market signal, split ownership. The performance peer owns the GPU / CUDA / kernel-authoring craft; you own the production serving operation of what that peer and the ML-platform peer produce.

| Owned by `ai-infra-performance` | Owned here |
|---|---|
| CUDA / CUTLASS / CuTe / Triton (OpenAI) kernel source | vLLM / SGLang / TRT-LLM engine configuration and operation |
| Flash Attention / fused-kernel implementation | Deploying the engine that *calls* Flash Attention |
| Nsight Compute → SASS profiling, warp-level stalls | DCGM / Prometheus fleet metrics, TTFT / TPOT dashboards |
| Model-compression algorithm design (new AWQ / GPTQ variant) | Choosing a quantization recipe and gating its accuracy regression |
| CUTLASS GEMM epilogue, FP8 scaling-factor math | Picking FP8 vs. FP16 for a given model + fleet |

Routing rule: **primary deliverable is a `.cu` / `.cuh` / `.py` kernel file → performance peer. Primary deliverable is a running serving system → you.** When the two overlap (e.g., Flash Attention 3 lands with a new kernel contract), the performance peer ships the kernel and lands it in the engine source; you ship the engine upgrade in the fleet, the SLO comparison, and the rollback plan.

### 2. `ai-infra-ml-platform` — peer, level 30, the training–serving handoff

The ML-platform peer runs the model registry, the artifact store, the lineage graph, and the self-service pipeline that turns a training job into a signed artifact. You consume what that peer ships.

| Owned by `ai-infra-ml-platform` | Owned here |
|---|---|
| Model registry, artifact store, lineage, signing | Pulling the artifact into a serving fleet, loading it into vLLM / SGLang / TRT-LLM |
| Training-cluster scheduling, experiment store | Serving-cluster capacity, warm-pool sizing, canary routing |
| "Train → artifact → promote" pipeline | "Artifact → engine build → canary → fleet" pipeline |

Routing rule: anything upstream of the registry (training, fine-tuning, artifact signing) is the ML-platform peer's. Anything from the registry pull onward (engine build, warm-up, routing, SLO) is yours.

### 3. `ai-infra-sre` — peer, level 30, SRE methodology

The SRE peer owns the SRE *methodology* — the error-budget math, the on-call rotation design, the incident reporting discipline, the post-mortem template. You apply that methodology to inference signals. You do not re-derive it.

| Owned by `ai-infra-sre` | Owned here |
|---|---|
| Error-budget math, SLO vs. SLI vs. SLA definitions | TTFT / TPOT / cache-hit as inference-specific SLIs |
| Incident command, severity grid, post-mortem template | Writing the TTFT-regression post-mortem using that template |
| On-call rotation discipline | Running the inference-specific on-call shift |

Routing rule: if the question is "how do we do SRE here" → SRE peer. If the question is "what are the right numbers to put inside the SRE template for an inference workload" → you.

### 4. `ai-infra-mlops` — peer, level 25, CI/CD automation

The MLOps peer owns the generic CI/CD and automation plumbing. You plug the inference-platform specifics into that plumbing. Think "GitHub Actions, Argo, Flux" (them) vs. "vLLM Helm chart, canary weight ramp, rollback runbook" (you).

Routing rule: generic pipeline tooling → MLOps peer. Inference-specific gates, canary strategies, model-swap runbooks → you.

### 5. `ai-infra-security` — peer, level 35, inference-edge security

Share the level, split the attack surface. The security peer owns deep adversarial defence at the inference edge; you own the serving plane the security peer protects.

| Owned by `ai-infra-security` | Owned here |
|---|---|
| Prompt-injection defence at the API gateway | OpenAI-compatible API surface, request-shape validation |
| Model-extraction defence (query-rate shaping, watermarking) | Per-tenant rate limits as *capacity* controls |
| Model-weight supply-chain signing, SBOM | Verifying the signed weight before loading into the engine |
| Confidential-compute / enclave attestation | Running inside the enclave the security peer specifies |

Routing rule: *adversarial* threat → security peer. *Multi-tenant isolation as a capacity / noisy-neighbour concern* → you. In practice you share the API gateway: the security peer writes the WAF rules, you write the engine-side admission logic.

### 6. The AI Engineering family — consumers, level 25

`llm-application-developer`, `rag-engineer`, `applied-ai-engineer` sit at level 25 in a different family. They are your **customers**, not your peers. They write the application that calls your inference API. You deliver them an endpoint; they deliver you traffic.

Routing rule: application code (prompt templates, agent loops, RAG retrieval, tool-schema design) → AI Engineering family. The inference endpoint, its SLA, its rate limits, its streaming semantics, its structured-output enforcement → you.

The peer at level 30 in that same family — `ai-eval-engineer` — is the one you share a release gate with. They own application-layer eval; you own the serving-level canary that uses the eval verdict as a go/no-go. (The post-training eval peer, `model-evaluation-engineer`, lives in the ML Engineering family at level 30 and owns benchmark methodology.)

## The up-and-down ladder

You defer **up** to:

- `ai-infra-architect` (level 40) — cross-stack architecture at team-of-teams scope. If the question spans inference + training + ML platform + security, the architect runs the RFC; you contribute the inference piece.
- `ai-infra-team-lead` (level 40) — people management for the inference platform team.
- `ai-infra-senior-architect` (level 45), `ai-infra-principal-engineer` / `ai-infra-principal-architect` (level 50) — multi-year strategy and organisation-scope decisions.

You inherit **down** from:

- `ai-infra-engineer` (level 20) — the K8s / Terraform / cloud / Linux base. You do not re-teach Kubernetes in this track; you assume it and extend it with GPU-specific primitives.
- `ai-infra-junior-engineer` (level 10) — engineering-craft prerequisites (Python, Linux, Git, basic distributed systems).

The sibling senior-IC role, `ai-infra-senior-engineer` (prerequisite track, level 30-ish), is the generalist you were before you specialised; this role is one of the specialisations beyond that.

## Two concrete routing scenarios

### Scenario A: "P99 TTFT doubled after the Monday model swap"

- Is it an engine config change? → **yours**
- Is it a kernel regression that landed in the engine upstream? → performance peer owns the fix; you own the rollback and the fleet exposure
- Is it a quantization accuracy issue showing up as speculative-decoding acceptance drop? → you own the serving regression; `model-evaluation-engineer` (peer, level 30) owns the accuracy diagnosis
- Is it a routing / autoscaling bug that packed the wrong requests onto the wrong rank? → **yours**
- Is it a bad artifact pulled from the registry (wrong revision)? → ML-platform peer; you own the version-pin fix

### Scenario B: "A customer says the agent 'went off the rails' on a long tool-use chain"

- Is it a prompt / agent-loop bug? → AI Engineering family (consumer); not yours
- Is it a tool-call protocol mis-serialisation? → yours (API surface, module 111)
- Is it a KV-cache / prefix-sharing leak across tenants? → yours (module 104, cross-tenant safety) + security peer (adversarial depth)
- Is it a safety-policy decision ("should we serve this at all")? → AI Governance family, out of scope for this role

Both scenarios share a pattern: *the first move is routing, not debugging*. If you cannot name the owner in under 30 seconds, your charter is not sharp enough.

## Diagram: the five boundaries from above

```
                           ┌───────────────────────────────┐
                           │  AI Engineering family (L25)  │
                           │  consumers of your API        │
                           └──────────────┬────────────────┘
                                          │   endpoints, SLAs, telemetry
                                          ▼
       ai-infra-security  ──── adversarial defence   ┐
       (peer, L35)                                   │
                                                     │
       ai-infra-performance ── kernel craft          │
       (peer, L35)                                   ├──►  YOU
                                                     │     ai-infra-inference-engineer (L35)
       ai-infra-ml-platform ── registry handoff      │     production operation of inference engines
       (peer, L30)                                   │
                                                     │
       ai-infra-sre ────────── SRE methodology       │
       (peer, L30)                                   │
                                                     │
       ai-infra-mlops ──────── CI/CD plumbing        │
       (peer, L25)                                   ┘

       ai-infra-engineer (L20)       ──── K8s / cloud base
       ai-infra-junior-engineer (L10) ──── engineering craft
```

The arrows point *into* you. Each one is a contract, not a vibe: an artifact, a metric, a decision, a signed deliverable.

## Summary

- Your charter is one line: production operation of inference engines, under SLO, at defensible cost.
- The six peer boundaries are performance (kernel craft), ML platform (registry handoff), SRE (methodology), MLOps (CI/CD), security (adversarial defence), and the AI Engineering family (consumers of your API).
- The single most important boundary is with `ai-infra-performance` at the same level: same market signal, different artifacts (`.cu` kernel vs. running serving system).
- Routing comes before debugging. If you cannot name the owner of an incoming problem, you cannot run the platform.

The rest of this module gives you the vocabulary and the hardware map to make those routing calls fluent. Then modules 102–114 take each box inside your boundary and go deep.
