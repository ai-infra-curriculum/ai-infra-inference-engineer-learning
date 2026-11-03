# Prerequisites — AI Inference Engineer track

**Role level:** 35 (production-serving specialist, AI Infrastructure family)
**Track:** `ai-infra-inference-engineer-learning`

This track is a level-35 production-serving specialty. It assumes the learner arrives with the K8s / cloud / CI/CD base muscle-memoried, MLOps automation familiarity in hand, and enough transformer-model literacy to debug a serving outage without a hand-held primer. Fundamentals are **not** re-taught here — the modules step directly onto inference-engine internals, request scheduling, KV-cache runtime management, quantization deployment, multi-model fleets, and SLO engineering.

## Assumed lower-level curriculum

Complete or be practically fluent in the equivalent of these lower-level tracks before starting:

- **[`ai-infra-junior-engineer-learning`](https://github.com/ai-infra-curriculum/ai-infra-junior-engineer-learning) — level 10** — engineering-craft prerequisites (Linux, Git, Python packaging, HTTP APIs, networking primitives, Docker basics, unit testing, logging). If you cannot ship a small Python service and instrument it, start here.
- **[`ai-infra-engineer-learning`](https://github.com/ai-infra-curriculum/ai-infra-engineer-learning) — level 20** — generalist AI infrastructure: Kubernetes (deployments, services, ingress, HPA, PodDisruptionBudget), Terraform (cloud provider modules, state), AWS / GCP / Azure base (VPC / IAM / networking / managed services), container runtime (Docker, containerd), CI/CD (GitHub Actions / GitLab CI), observability base (Prometheus / Grafana / OpenTelemetry at generalist depth), cost reporting primitives. If you cannot stand up a stateless web service on Kubernetes with Terraform and point Grafana at its metrics, start here.
- **[`ai-infra-mlops-learning`](https://github.com/ai-infra-curriculum/ai-infra-mlops-learning) — level 25** — MLOps automation (model registry, CI/CD for ML, deployment pipelines, dataset / feature-store wiring, experiment tracking at platform level). This role *plugs rollouts and rollback contracts into* that peer's pipelines — you do not need to build the pipeline from scratch, but you must be able to read, modify, and operate one.

## Assumed working knowledge

Even without holding the equivalent role formally, you should be practically fluent in:

- **Python** — comfortable with packaging (pyproject.toml, uv / poetry, virtualenvs), unit testing (pytest), typed code (mypy / pyright), FastAPI / async request handling, and CI (GitHub Actions / GitLab CI).
- **Kubernetes at operator depth** — deployments, services, ingress, ConfigMap / Secret, HPA / VPA, StatefulSet, PodDisruptionBudget, taints / tolerations, resource requests / limits, probes, headless services, service mesh basics. GPU-specific extensions (NVIDIA GPU Operator, DCGM-exporter, DRA, MIG, MPS) are taught here in mod-109.
- **Linux + networking** — comfortable with `/proc`, `strace`, `tcpdump`, iptables / nftables at reading-depth, DNS resolution, TLS termination, L4 vs. L7 load balancing. GPU driver / NUMA / PCIe / NVLink / RDMA topology is taught here at operator depth.
- **PyTorch + Hugging Face Transformers** — enough to load a model, run inference, understand `.safetensors` layout, swap in a LoRA adapter. You should be able to read `transformers` modeling files for a given architecture (attention layers, RoPE, LayerNorm) at reading-depth — kernel-authoring depth lives at the performance peer.
- **HTTP APIs + LLM providers** — comfortable calling OpenAI / Anthropic / Google / open-source (vLLM / SGLang / TensorRT-LLM) inference APIs from Python; comfortable with streaming (SSE / WebSocket) and tool-calling at the client level. Server-side implementation is taught here in mod-111.
- **Observability** — comfortable reading Prometheus metrics, writing PromQL at a working level, authoring Grafana dashboards, reading OpenTelemetry traces. Inference-specific instrumentation (TTFT / TPOT / ITL decomposition, DCGM metrics, scheduler-metric export) is taught here in mod-110.
- **Transformer model literacy** — comfortable with the attention mechanism, multi-head vs. multi-query vs. grouped-query attention, KV cache concept, token-level autoregressive decoding, prefix / prompt / completion phases, tokenizer basics. Deep architecture authoring is NOT required.
- **Load testing** — comfortable running a load test with any tool (k6, Locust, wrk, hey) and interpreting P50 / P95 / P99 / P99.9 tails.
- **Reading the primary literature** — comfortable opening a paper on arXiv (PagedAttention, RadixAttention, DistServe, Mooncake, S-LoRA, speculative decoding) and translating method → runnable deployment within a working day.

## Not assumed, taught here

You do **not** need prior experience with:

- Any specific inference engine — vLLM, SGLang, TensorRT-LLM, NVIDIA Dynamo, Triton Inference Server. Mods 101 / 102 cover the operator-depth landscape and internals.
- Continuous batching + chunked prefill mechanics. Mod-103 covers the scheduler loop end-to-end.
- PagedAttention / RadixAttention runtime management, tiered KV stores (LMCache / Mooncake), prefix-cache tuning. Mod-104 covers KV cache as a runtime object.
- Quantization deployment formats (FP8 / FP4 / INT4 / AWQ / GPTQ / SmoothQuant) or model-format conversion (safetensors / GGUF / TRT-engines). Mod-105 covers deployment-depth; algorithm-design depth is linked out to the performance peer.
- Multi-LoRA serving at scale (S-LoRA / Punica), multi-model co-location, cost-aware routing. Mod-106 covers the working suite.
- Tensor / pipeline / expert parallelism as *deployment topologies*; disaggregated prefill-decode (DistServe / Splitwise / Dynamo); NCCL / NVLink / RDMA at operator depth. Mod-107 covers topology selection + deployment; kernel-level implementation is at the performance peer.
- Speculative-decoding deployment (draft-target, Medusa, EAGLE), acceptance-rate monitoring, draft-model lifecycle. Mod-108 covers.
- GPU fleet autoscaling (HPA on inference custom metrics, Karpenter for GPU bin-packing), GPU sharing (MIG / MPS / DRA), warm pools, spot resilience, multi-region failover. Mod-109 covers the GPU-specific extensions.
- Inference-specific observability (TTFT / TPOT / ITL / cache-hit / preemption axes), SLO engineering at per-token granularity, cost per 1M tokens accounting, throughput-vs-latency Pareto. Mod-110 covers.
- OpenAI-compatible API server-side implementation, SSE / WebSocket streaming at production quality, structured output / constrained decoding (Outlines / XGrammar), server-side tool-calling, MCP server-side exposure. Mod-111 covers.
- Multimodal / diffusion / speech / video / VLM / reasoning-model serving. Mod-112 covers.
- Canary / shadow / blue-green rollouts specific to inference, response-diffing correctness, inference-specific incident response and runbooks. Mod-113 covers.
- Inference-platform RFC authoring, build-vs-buy framing, vendor eval, cross-peer interface contracts. Mod-114 covers.

## Recommended reading before starting

- [vLLM documentation](https://docs.vllm.ai/) — the "Deploying" and "Serving" sections, front-to-back once.
- [PagedAttention / vLLM paper (Kwon et al. 2023)](https://arxiv.org/abs/2309.06180) — the design paper for the dominant open-source inference-engine architecture.
- [SGLang paper + RadixAttention post](https://arxiv.org/abs/2312.07104) — the design paper for RadixAttention and structured generation.
- [TensorRT-LLM getting started](https://nvidia.github.io/TensorRT-LLM/) — the engine-build workflow end-to-end.
- [Google SRE Book — SLOs chapter](https://sre.google/sre-book/service-level-objectives/) — SLO engineering vocabulary adapted here to inference-specific metrics.
- [OpenAI API reference](https://platform.openai.com/docs/api-reference) + [OpenAI structured outputs guide](https://platform.openai.com/docs/guides/structured-outputs) — the de-facto industry API surface.
- A recent public engine release note (e.g., a vLLM / SGLang / TensorRT-LLM release blog) — read it as an operator would, tracing each new feature to the operational concern it changes.

## Not required, useful to have

- Prior work with a managed inference provider (Fireworks, Together, Baseten, Modal, Replicate, Anyscale, Groq, Cerebras) — a working feel is useful before the build-vs-buy framing in mod-114.
- Prior work with a service mesh (Istio / Linkerd) — not required, but the ingress / routing patterns in mod-111 and mod-106 compose cleanly with a mesh.
- Prior work with a cost-reporting / FinOps platform (Vantage, OpenCost, Kubecost, Vanta FinOps) — mod-110 touches cost-per-request engineering.
- Prior work with a load-testing platform at scale (k6 Cloud, Grafana k6, Locust distributed) — the project-101 / project-102 load-tests scale naturally into one.
- Certifications: Kubernetes CKA / CKAD, NVIDIA DLI (Deep Learning Institute) GPU courses, and cloud-provider ML certifications are treated as strong signals in enterprise hiring but not required.

## Not in scope for this track (linked out)

- **GPU / CUDA / kernel-authoring / profiling / compression depth at kernel-authoring quality** — owned by [`ai-infra-performance-learning`](https://github.com/ai-infra-curriculum/ai-infra-performance-learning) (peer, level 35, same family). **This is the single most important peer boundary.** Route by primary deliverable: a production serving system → this track; a CUDA / CUTLASS / Triton kernel source file → the performance peer.
- **Inference-edge adversarial-defence depth** — prompt-injection at the API gateway, PII / jailbreak classifier sidecars, model-extraction query-rate control, model-weight supply-chain signing — owned by [`ai-infra-security-learning`](https://github.com/ai-infra-curriculum/ai-infra-security-learning) (peer, level 35). This track implements the serving-side hooks only.
- **Self-service ML platform + model registry depth** — owned by [`ai-infra-ml-platform-learning`](https://github.com/ai-infra-curriculum/ai-infra-ml-platform-learning) (peer, level 30). This track consumes model artifacts over a defined handoff contract.
- **SRE methodology depth** — incident response, error budgets, DR, chaos engineering — owned by [`ai-infra-sre-learning`](https://github.com/ai-infra-curriculum/ai-infra-sre-learning) (peer, level 30). This track applies the methodology to serving incidents.
- **Application-layer eval release-gate depth** — owned by [`ai-eval-engineer-learning`](https://github.com/ai-engineering-curriculum/ai-eval-engineer-learning) (peer, level 30, AI Engineering family). This track consumes the eval-gate contract.
- **Benchmark / statistical / calibration methodology depth** — owned by [`model-evaluation-engineer-learning`](https://github.com/ml-engineering-curriculum/model-evaluation-engineer-learning) (peer, level 30, ML Engineering family). This track consumes the methodology.
- **Post-training / fine-tuning stack depth** — owned by [`fine-tuning-engineer-learning`](https://github.com/ml-engineering-curriculum/fine-tuning-engineer-learning) (peer, level 30). This track consumes LoRA / full-weight artifacts.
- **MLOps automation (CI/CD pipeline authoring)** — owned by [`ai-infra-mlops-learning`](https://github.com/ai-infra-curriculum/ai-infra-mlops-learning) (peer, level 25). This track plugs rollouts into those pipelines.
- **General K8s / Terraform / cloud base** — owned by [`ai-infra-engineer-learning`](https://github.com/ai-infra-curriculum/ai-infra-engineer-learning) (level 20). This track covers GPU-specific extensions only.
- **Cross-stack infra architecture at team-of-teams scope** — owned by [`ai-infra-architect-learning`](https://github.com/ai-infra-curriculum/ai-infra-architect-learning) (next-up, level 40). This track feeds platform RFCs upward.
- **People management for an inference platform team** — owned by [`ai-infra-team-lead-learning`](https://github.com/ai-infra-curriculum/ai-infra-team-lead-learning) (next-up, level 40). This track stops at technical-leadership scope.
- **Prompt engineering, agent design, product analytics** — owned by the AI Engineering family (`llm-application-developer`, `rag-engineer`, `applied-ai-engineer` at level 25). Those tracks consume this track's inference API.
- **Legal / compliance** — licensing, data residency, regulator sign-off — out of scope; routed to counsel.
