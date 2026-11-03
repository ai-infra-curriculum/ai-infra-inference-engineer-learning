# Job Requirements — AI Inference Engineer

**Role level:** 35 (production-serving specialist — AI Infrastructure family)
**Track:** `ai-infra-inference-engineer-learning`
**Research window:** 2026-08-05 → 2026-11-03 (last 90 days)
**Today:** 2026-11-03
**Live postings sampled this cycle:** 0 (bootstrap scaffold — see Status below)

This file documents the requirements catalog used to seed the AI Inference Engineer curriculum. Raw normalised data lives in [`.aicg/job-requirements.json`](.aicg/job-requirements.json); the planned curriculum lives in [`.aicg/curriculum-plan.json`](.aicg/curriculum-plan.json).

## Status — bootstrap session, live postings deferred

The 2026-11 bootstrap cycle was authored in a session where `WebSearch` and `WebFetch` are permission-gated and have not been granted — delegated subagents inherit the same state. Rather than invent posting evidence, the `postings[]` array in [`.aicg/job-requirements.json`](.aicg/job-requirements.json) is **intentionally empty** and `research_status.phase` is set to `"scaffold"`. Every entry in `requirement_themes[]` carries `"frequency": "needs-research"`.

The curriculum structure below is grounded in two sources that are stable without live-posting evidence:

1. The **sibling [`ai-infra-performance-learning`](../ai-infra-performance-learning/) packet** (level 35, peer specialist in the same family), whose 2026-10 autonomous cycle sampled **63 in-window postings** across the overlapping GPU / inference-engine market. The named-engine frequency (vLLM / SGLang / TensorRT-LLM / Dynamo at 0.73), continuous-batching, PagedAttention, KV-cache management, multimodal serving, and disaggregated prefill-decode signals carry over to this role, with the inference-engineer market slanting toward the serving-platform operation of those engines rather than the kernel-authoring craft.
2. **Authoritative public references** that publish what the role is hired against — engine docs, foundational papers, standards, and vendor references. Every `requirement_themes[]` entry cites at least one such reference. Live-posting evidence is additive and will be backfilled on the first autonomous research cycle without restructuring the schema.

The autonomous research loop should backfill `postings[]` against the sampling convention documented in [`.aicg/job-requirements.json`](.aicg/job-requirements.json) `research_status.needs_research_note`, then recompute frequencies against the realised `n`.

## Grounding references

- **Open-source inference engines (operational depth)** — [vLLM](https://docs.vllm.ai/), [SGLang](https://docs.sglang.ai/), [TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM/), [NVIDIA Dynamo](https://developer.nvidia.com/dynamo), [NVIDIA Triton Inference Server](https://docs.nvidia.com/deeplearning/triton-inference-server/), [Ray Serve](https://docs.ray.io/en/latest/serve/index.html) + [Ray LLM](https://docs.ray.io/en/latest/serve/llm/index.html), [llama.cpp](https://github.com/ggml-org/llama.cpp), [MLC-LLM](https://llm.mlc.ai/), [TGI (archived maintenance mode)](https://github.com/huggingface/text-generation-inference).
- **Foundational serving papers** — [PagedAttention / vLLM (Kwon et al. 2023)](https://arxiv.org/abs/2309.06180), [SGLang / RadixAttention (Zheng et al. 2024)](https://arxiv.org/abs/2312.07104), [SARATHI / chunked prefill (Agrawal et al.)](https://arxiv.org/abs/2308.16369), [DistServe (Zhong et al. 2024)](https://arxiv.org/abs/2401.09670), [Splitwise (Patel et al. 2024)](https://arxiv.org/abs/2311.18677), [Mooncake (Qin et al. — Moonshot AI)](https://arxiv.org/abs/2407.00079), [S-LoRA (Sheng et al. 2023)](https://arxiv.org/abs/2311.03285), [Punica (Chen et al. 2023)](https://arxiv.org/abs/2310.18547), [Speculative Decoding (Leviathan et al. 2023)](https://arxiv.org/abs/2211.17192), [Medusa (Cai et al. 2024)](https://arxiv.org/abs/2401.10774), [EAGLE (Li et al. 2024)](https://arxiv.org/abs/2401.15077), [FlashAttention (Dao et al. 2022)](https://arxiv.org/abs/2205.14135), [Megatron-LM (Shoeybi et al.)](https://arxiv.org/abs/1909.08053).
- **Quantization / model-format references** — [AWQ (Lin et al.)](https://arxiv.org/abs/2306.00978), [GPTQ (Frantar et al.)](https://arxiv.org/abs/2210.17323), [SmoothQuant (Xiao et al.)](https://arxiv.org/abs/2211.10438), [FP8 Formats whitepaper (NVIDIA / Arm / Intel)](https://arxiv.org/abs/2209.05433), [safetensors](https://huggingface.co/docs/safetensors/index), [GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md), [Hugging Face Hub](https://huggingface.co/docs/hub/index).
- **GPU runtime and Kubernetes GPU infrastructure** — [NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/), [CUDA Graphs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cuda-graphs), [torch.compile](https://pytorch.org/docs/stable/torch.compiler.html), [NCCL](https://docs.nvidia.com/deeplearning/nccl/), [NVIDIA DCGM](https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/index.html), [NVIDIA GPU Operator for Kubernetes](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/), [NVIDIA MIG](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/), [NVIDIA MPS](https://docs.nvidia.com/deploy/mps/index.html), [NVIDIA Blackwell architecture](https://resources.nvidia.com/en-us-blackwell-architecture), [NVIDIA NIM microservices](https://docs.nvidia.com/nim/), [Kubernetes Dynamic Resource Allocation](https://kubernetes.io/docs/concepts/scheduling-eviction/dynamic-resource-allocation/), [Karpenter](https://karpenter.sh/), [Knative Serving](https://knative.dev/docs/serving/).
- **Inference API and protocol references** — [OpenAI API](https://platform.openai.com/docs/api-reference) + [OpenAI structured outputs](https://platform.openai.com/docs/guides/structured-outputs) + [OpenAI production best practices](https://platform.openai.com/docs/guides/production-best-practices), [Model Context Protocol (MCP)](https://modelcontextprotocol.io/), [HTML5 Server-Sent Events](https://html.spec.whatwg.org/multipage/server-sent-events.html), [JSON Schema](https://json-schema.org/), [Outlines](https://dottxt-ai.github.io/outlines/), [XGrammar](https://xgrammar.mlc.ai/), [FastAPI](https://fastapi.tiangolo.com/), [Pydantic](https://docs.pydantic.dev/latest/), [LiteLLM](https://docs.litellm.ai/).
- **SRE / observability / load-test** — [Google SRE Book — SLOs](https://sre.google/sre-book/service-level-objectives/), [Prometheus](https://prometheus.io/docs/introduction/overview/), [Grafana](https://grafana.com/docs/grafana/latest/), [OpenTelemetry](https://opentelemetry.io/docs/), [k6](https://k6.io/docs/), [Locust](https://docs.locust.io/en/stable/).

Every `requirement_themes[]` entry below cites at least one of these references, and every entry is shaped so posting-frequency evidence can be added underneath it without restructuring the schema.

## Methodology

1. Sourced the canonical task domains for an AI Inference Engineer (level 35, AI Infrastructure family) from the public references catalogued in `.aicg/job-requirements.json → authoritative_references`.
2. Cross-referenced the sibling [`ai-infra-performance-learning`](../ai-infra-performance-learning/.aicg/job-requirements.json) 2026-10 cycle (63 in-window postings) for the overlapping market shape — the two tracks are peer specialists in the same family and the market signal for them overlaps substantially.
3. Mapped each task domain to (a) the role on our level ladder that should own it primarily and (b) the curriculum module that will cover it.
4. Applied the **ownership rule** — assign primary coverage to the lowest-level role that genuinely requires the skill; higher-level tracks link back rather than duplicate. In particular:
   - *Sideways at the SAME level* to `ai-infra-performance` (peer, level 35, same family) — GPU / CUDA / kernel-authoring / profiling / compression at kernel-authoring quality. **This is the single most important peer boundary in the catalog.** The two tracks share market signal but divide ownership by primary deliverable: a serving system (this role) vs. a kernel source file (the performance peer).
   - *Sideways at the same level* to `ai-infra-security` (peer, level 35) — adversarial-defence at the inference edge, model-extraction defence, model-weight signing. This role implements the serving-side hooks; the depth stays with the security peer.
   - *Down* to `ai-infra-ml-platform` (peer, level 30, same family) — self-service ML platform, model registry, training-serving handoff.
   - *Down* to `ai-infra-sre` (peer, level 30, same family) — SRE methodology (incident response, error budgets, DR).
   - *Down* to `ai-eval-engineer` (peer, level 30, AI Engineering family) — application-layer eval release-gate.
   - *Down* to `ai-infra-mlops` (peer, level 25, same family) — MLOps automation (CI/CD, model-registry integration).
   - *Down* to `ai-infra-engineer` (level 20) — K8s / Terraform / cloud / Docker base.
   - *Down* to `ai-infra-junior-engineer` (level 10) — engineering-craft prerequisites.
   - *Up* to `ai-infra-architect` (level 40), `ai-infra-team-lead` (level 40) — cross-stack architecture and people management.
   - *Up* to `ai-infra-senior-architect` (level 45), `ai-infra-principal-engineer` / `ai-infra-principal-architect` (level 50) — organisation-scope architecture and strategy.
   - *Out of scope* to the AI Engineering family (`llm-application-developer`, `rag-engineer`, `applied-ai-engineer` at level 25) — prompt engineering, agent design, product analytics — those tracks consume this role's inference API; this role does not deliver the application.
   - *Out of scope* to legal / compliance — licensing, data residency, regulator sign-off.
5. Flagged every requirement whose frequency is `"needs-research"` so the next cycle can backfill posting evidence and promote / demote accordingly.

## Requirement themes → curriculum ownership

The table below lists each requirement theme, its posting frequency this cycle, its planned owner per the level hierarchy, and the curriculum coverage path.

| # | Theme | Freq | Owner role | Coverage |
|---|---|---|---|---|
| 1 | Operational fluency with named inference engines (vLLM, SGLang, TensorRT-LLM, NVIDIA Dynamo, Triton Inference Server; TGI as historical compare-point) | <!-- needs-research --> | `ai-infra-inference-engineer` (this) | [`mod-102-inference-engine-internals`](lessons/mod-102-inference-engine-internals) |
| 2 | Continuous / dynamic batching + request-scheduler internals (admission control, token-level scheduling, chunked prefill, preemption, priority) | <!-- needs-research --> | `ai-infra-inference-engineer` | [`mod-103-request-scheduling-and-batching`](lessons/mod-103-request-scheduling-and-batching) |
| 3 | KV-cache management as a runtime concern — PagedAttention, RadixAttention prefix caching, copy-on-write, fragmentation, tiered KV stores (LMCache / Mooncake) | <!-- needs-research --> | `ai-infra-inference-engineer` | [`mod-104-kv-cache-management`](lessons/mod-104-kv-cache-management) |
| 4 | Deploying quantized models (FP8, FP4 / NVFP4 / MXFP4 on Blackwell, INT8, INT4, AWQ, GPTQ, SmoothQuant) with accuracy-regression gates and format conversion (safetensors / GGUF / TRT-engines) | <!-- needs-research --> | `ai-infra-inference-engineer` | [`mod-105-quantization-and-model-ops`](lessons/mod-105-quantization-and-model-ops) |
| 5 | Multi-model / multi-tenant / multi-adapter serving — S-LoRA / Punica multi-LoRA, cost-aware router, model co-location, dynamic weight loading, cross-provider failover | <!-- needs-research --> | `ai-infra-inference-engineer` | [`mod-106-multi-model-and-adapter-serving`](lessons/mod-106-multi-model-and-adapter-serving) |
| 6 | Distributed inference topologies — TP / PP / EP as deployment topologies, disaggregated prefill-decode (DistServe / Splitwise / Dynamo), KV router, cross-node KV transfer | <!-- needs-research --> | `ai-infra-inference-engineer` | [`mod-107-distributed-inference-topologies`](lessons/mod-107-distributed-inference-topologies) |
| 7 | Speculative-decoding deployment — draft-target, Medusa, EAGLE — acceptance-rate monitoring, draft-model lifecycle, when speculative hurts | <!-- needs-research --> | `ai-infra-inference-engineer` | [`mod-108-speculative-decoding-ops`](lessons/mod-108-speculative-decoding-ops) |
| 8 | GPU fleet autoscaling and resilience — K8s HPA on custom inference metrics, Karpenter / Cluster Autoscaler, DRA / MIG / MPS GPU sharing, warm pools, spot resilience, scale-to-zero | <!-- needs-research --> | `ai-infra-inference-engineer` | [`mod-109-autoscaling-fleet-and-resilience`](lessons/mod-109-autoscaling-fleet-and-resilience) |
| 9 | Inference-specific observability + SLO engineering — TTFT / TPOT / ITL / E2E / cache-hit / preemption, DCGM / Prometheus / Grafana / OTel, cost per 1M tokens, throughput-vs-latency Pareto | <!-- needs-research --> | `ai-infra-inference-engineer` | [`mod-110-observability-slo-and-cost`](lessons/mod-110-observability-slo-and-cost) |
| 10 | Inference API design + protocols — OpenAI-compatible API, streaming (SSE / WebSocket), structured output / constrained decoding, tool calling, MCP server-side implications, retry / idempotency / backpressure | <!-- needs-research --> | `ai-infra-inference-engineer` | [`mod-111-inference-api-and-protocols`](lessons/mod-111-inference-api-and-protocols) |
| 11 | Multimodal / diffusion / speech / video / VLM / reasoning-model serving | <!-- needs-research --> | `ai-infra-inference-engineer` | [`mod-112-multimodal-and-emerging-serving`](lessons/mod-112-multimodal-and-emerging-serving) |
| 12 | Production safety + rollouts — canary / shadow traffic for inference, response-diff correctness, blue-green model swap, inference-specific incident response and runbooks | <!-- needs-research --> | `ai-infra-inference-engineer` | [`mod-113-production-safety-and-rollouts`](lessons/mod-113-production-safety-and-rollouts) |
| 13 | Cross-functional inference-platform leadership at team scope — build-vs-buy, vendor eval, RFC authoring, SLO ritual, on-call shape, peer interfaces | <!-- needs-research --> | `ai-infra-inference-engineer` | [`mod-114-cross-functional-inference-leadership`](lessons/mod-114-cross-functional-inference-leadership) |
| 14 | Position-on-the-ladder vocabulary and engine capability map | <!-- needs-research --> | `ai-infra-inference-engineer` | [`mod-101-inference-engineering-position`](lessons/mod-101-inference-engineering-position) |
| 15 | GPU / CUDA / kernel-authoring / CUTLASS / CuTe DSL / Nsight-Compute-to-SASS depth | <!-- needs-research --> (peer) | `ai-infra-performance` (level 35, same family) | Linked out — [`ai-infra-performance-learning`](../ai-infra-performance-learning/) |
| 16 | MLOps / model registry / deployment pipelines (CI/CD for ML) | <!-- needs-research --> (peer) | `ai-infra-mlops` (level 25) | Linked out — [`ai-infra-mlops-learning`](../ai-infra-mlops-learning/) |
| 17 | Kubernetes / Terraform / cloud (AWS / GCP / Azure) base skills | <!-- needs-research --> (lower) | `ai-infra-engineer` (level 20) | Listed in [`PREREQUISITES.md`](PREREQUISITES.md); GPU-specific extensions covered here in mod-109 |
| 18 | Inference-edge security — prompt-injection defence at the API gateway, PII / jailbreak classifier sidecars, model-extraction query-rate control, model-weight supply-chain signing | <!-- needs-research --> (peer) | `ai-infra-security` (level 35) | Linked out — [`ai-infra-security-solutions`](../ai-infra-security-solutions/) + mod-111 ingress hooks |
| 19 | Application / model-eval release-gate contract | <!-- needs-research --> (peer) | `ai-eval-engineer` (level 30) | Linked out — mod-113 wires the contract |
| 20 | Model-artifact handoff contract (safetensors / GGUF / TRT-engine, metadata, lineage, licence) | <!-- needs-research --> (peer) | `ai-infra-ml-platform` (level 30) | Linked out — mod-105 defines the serving-side promotion contract |
| 21 | SRE methodology — incident response, error budgets, DR, chaos engineering | <!-- needs-research --> (peer) | `ai-infra-sre` (level 30) | Linked out — mod-110 and mod-113 adopt the methodology for inference signals |
| 22 | Cross-platform infrastructure architecture + organisation-scope strategy | <!-- needs-research --> (next-up) | `ai-infra-architect` (level 40) + above | Linked out — mod-114 feeds RFCs upward |

## Posting evidence

**0 in-window postings sampled** for the 2026-11 bootstrap cycle — `WebSearch` and `WebFetch` are permission-gated in this session. See [`.aicg/job-requirements.json`](.aicg/job-requirements.json) `research_status` for the sampling convention the next autonomous cycle should apply.

Expect title fragmentation: `AI Inference Engineer` / `Inference Platform Engineer` / `Member of Technical Staff, Inference` / `Model Serving Engineer` / `LLM Serving Engineer` / `Software Engineer, Inference` / `Infrastructure Engineer, Model Serving` / `ML Serving Platform Engineer` all map to this role. The sibling [`ai-infra-performance-learning`](../ai-infra-performance-learning/.aicg/job-requirements.json) 2026-10 cycle's 63 postings are a strong co-sample — many postings legitimately cross both roles. The deciding filter is **primary deliverable**: a production serving system / scheduler / router / fleet operator → this role; a CUDA / CUTLASS / Triton kernel source file → the performance peer.

## Ownership map — quick reference for next cycle

When backfilling postings, use this ownership decision to keep the curriculum from drifting into peer territory:

- **AI Infra Junior Engineer** (level 10) — engineering-craft prerequisites.
- **AI Infra Engineer** (level 20) — Kubernetes, Terraform, cloud (AWS / GCP / Azure), Docker, CI/CD base, observability base. **This inference engineer consumes those skills and adds GPU-specific extensions in mod-109.**
- **AI Infra MLOps** (peer, level 25) — MLOps automation (model registry, CI/CD for ML, deployment pipelines). **This inference engineer plugs rollout and rollback contracts into that peer's pipelines.**
- **AI Infra ML Platform** (peer, level 30) — self-service ML platform, model registry, training-serving handoff.
- **AI Infra SRE** (peer, level 30) — SRE methodology. **This inference engineer applies that methodology to serving incidents.**
- **AI Eval Engineer** (peer, level 30, AI Engineering family) — application-layer evaluation. **This inference engineer consumes the eval-gate contract.**
- **AI Inference Engineer** (this track, level 35) — production operation of inference engines: scheduler + batching + KV cache runtime + multi-model routing + autoscaling + SLO + cost + API protocols + rollouts.
- **AI Infra Performance Engineer** (peer, level 35, same family) — GPU / CUDA / kernel-authoring / profiling / compression craft. **This is the single most important peer boundary.** Route by primary deliverable (serving system vs. kernel source).
- **AI Infra Security** (peer, level 35) — inference-edge adversarial-defence and model-weight supply chain.
- **AI Infra Architect** (level 40) / **Team Lead** (level 40) — cross-platform architecture and people management. **This inference engineer feeds platform-RFC authorship upward.**
- **AI Infra Senior Architect** (level 45) / **Principal Engineer / Principal Architect** (level 50) — organisation-scope architecture and strategy. Linked out.

## Salary evidence

No salary data captured this cycle — see [`.aicg/job-requirements.json`](.aicg/job-requirements.json) `salary_summary`. The sibling `ai-infra-performance-learning` 2026-10 cycle summarised the overlapping market as spanning roughly US$180K–US$500K base + heavy equity for the kernel-authoring side of the market, with named frontier labs (Anthropic, xAI, OpenAI) at the upper tail. The inference-engineer market overlaps substantially — expect a comparable range for named-engine production-serving roles.

**Do not publish an aggregate until this packet samples ≥n=15 postings with salary present and recomputes the distribution independently.** <!-- needs-research: salary distribution for AI Inference Engineer -->
