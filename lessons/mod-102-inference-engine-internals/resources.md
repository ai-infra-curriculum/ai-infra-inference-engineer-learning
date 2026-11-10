# Resources — Module 102

Primary / citable references for the topics covered in this module. Prefer official repos, release notes, and peer-reviewed / arXiv papers to blog posts; where only a blog post exists for a current product claim, note it and verify against the official docs on the next refresh.

Many flag names, metric names, and defaults in the chapters are tagged with `<!-- needs-research: ... -->` or caveated as "verify against the version your fleet runs." These resources are the sources of truth to verify against.

## vLLM

- Project home — <https://github.com/vllm-project/vllm>
- Documentation — <https://docs.vllm.ai/>
- Release notes — <https://github.com/vllm-project/vllm/releases>
- Design docs / internals (in-tree) — <https://docs.vllm.ai/en/latest/design/>
- PagedAttention paper: Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention" — <https://arxiv.org/abs/2309.06180>
- OpenAI-compatible server docs — <https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html>
- Engine args reference — `vllm serve --help` (authoritative; mirrored in the docs)
- Metrics reference — <https://docs.vllm.ai/en/latest/serving/metrics.html>
- Prefix caching docs — the "automatic prefix caching" section of the docs (verify current path)
- Speculative decoding docs — the speculative-decoding guide (verify current path)
- LoRA docs — the LoRA adapters guide (verify current path)

## SGLang

- Project home — <https://github.com/sgl-project/sglang>
- Documentation — <https://docs.sglang.ai/>
- Release notes — <https://github.com/sgl-project/sglang/releases>
- RadixAttention paper: Zheng et al., "SGLang: Efficient Execution of Structured Language Model Programs" — <https://arxiv.org/abs/2312.07104>
- Server launch and args reference — the server-arguments section of the SGLang docs
- DP attention documentation — <https://docs.sglang.ai/> (search for "data parallelism attention")
- Structured output / grammar backend docs — <https://docs.sglang.ai/> (search for "structured output")

## TensorRT-LLM

- Project home — <https://github.com/NVIDIA/TensorRT-LLM>
- Documentation — <https://nvidia.github.io/TensorRT-LLM/>
- Release notes — <https://github.com/NVIDIA/TensorRT-LLM/releases>
- `trtllm-build` reference — the command-line reference in the docs
- Model-specific convert scripts — the `examples/` directory in the TRT-LLM repo
- Quantization and the TensorRT Model Optimizer — <https://github.com/NVIDIA/TensorRT-Model-Optimizer>
- NVIDIA developer blog on TensorRT-LLM — <https://developer.nvidia.com/blog/tag/tensorrt-llm/>

## Triton Inference Server + TRT-LLM backend

- Triton Inference Server project home — <https://github.com/triton-inference-server/server>
- Triton documentation — <https://docs.nvidia.com/deeplearning/triton-inference-server/>
- `tensorrtllm_backend` — <https://github.com/triton-inference-server/tensorrtllm_backend>
- Model configuration docs — <https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html>
- NVIDIA NGC container registry (for paired TRT-LLM + Triton images) — <https://catalog.ngc.nvidia.com/>

## NVIDIA Dynamo

- Project home — <https://github.com/ai-dynamo/dynamo>
- Documentation — <https://docs.nvidia.com/dynamo/>
- Dynamo-related NVIDIA developer blog posts — <https://developer.nvidia.com/blog/tag/dynamo/>
- NIXL (NVIDIA Inference Xfer Library, KV transport) — reference from the Dynamo docs

## Hugging Face TGI and the maintained alternatives

- TGI project home — <https://github.com/huggingface/text-generation-inference>
- TGI documentation — <https://huggingface.co/docs/text-generation-inference/>
- Hugging Face Inference Endpoints — <https://huggingface.co/docs/inference-endpoints/>
- Text-Generation-WebUI — <https://github.com/oobabooga/text-generation-webui>
- The `huggingface/text-generation-inference` GitHub repository was archived on 2026-03-21 and is in maintenance mode (minor bug fixes, documentation, and lightweight maintenance only, per the repo README). Hugging Face's README now points new deployments to vLLM, SGLang, and — for local / on-device use — llama.cpp or MLX. Verify the exact posture against <https://github.com/huggingface/text-generation-inference> on each content refresh.

## Scheduling, batching, and KV-cache primitives (research references)

- **Continuous batching (Orca):** Yu et al., "Orca: A Distributed Serving System for Transformer-Based Generative Models" — OSDI 2022 — <https://www.usenix.org/conference/osdi22/presentation/yu>
- **PagedAttention / vLLM:** Kwon et al. (see vLLM section)
- **RadixAttention / SGLang:** Zheng et al. (see SGLang section)
- **Chunked prefill (SARATHI):** Agrawal et al., "SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills" — <https://arxiv.org/abs/2308.16369>
- **SARATHI-Serve:** Agrawal et al., "Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve" — OSDI 2024 — <https://www.usenix.org/conference/osdi24/presentation/agrawal>
- **Disaggregated prefill-decode (DistServe):** Zhong et al., "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving" — OSDI 2024 — <https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin>
- **Splitwise:** Patel et al., "Splitwise: Efficient Generative LLM Inference Using Phase Splitting" — ISCA 2024 — <https://arxiv.org/abs/2311.18677>

## Attention and compile stack

- **Flash Attention 2:** Dao, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning" — <https://arxiv.org/abs/2307.08691>
- **Flash Attention 3:** Shah et al., "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision" — <https://arxiv.org/abs/2407.08608>
- **FlashInfer:** project home — <https://github.com/flashinfer-ai/flashinfer>; paper — <https://arxiv.org/abs/2501.01005> (verify current citation)
- **CUDA graphs:** NVIDIA CUDA documentation — <https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cuda-graphs>
- **`torch.compile`:** PyTorch documentation — <https://pytorch.org/docs/stable/torch.compiler.html>

## Structured output / constrained decoding

- **XGrammar:** <https://github.com/mlc-ai/xgrammar>
- **Outlines:** <https://github.com/dottxt-ai/outlines>
- Both have docs linked from their respective repos; both are the backing libraries for several engines' structured-output modes.

## NCCL, collective communication, and distributed debugging

- NCCL project home — <https://github.com/NVIDIA/nccl>
- NCCL documentation — <https://docs.nvidia.com/deeplearning/nccl/>
- PyTorch distributed documentation — <https://pytorch.org/docs/stable/distributed.html>
- NCCL troubleshooting / watchdog configuration — see the "troubleshooting" and "env variables" sections of the NCCL docs
- RCCL (AMD equivalent of NCCL) — <https://github.com/ROCm/rccl>

## GPU monitoring and per-rank diagnostics

- NVIDIA DCGM — <https://docs.nvidia.com/datacenter/dcgm/>
- DCGM Exporter for Prometheus — <https://github.com/NVIDIA/dcgm-exporter>
- `nvidia-smi` reference — part of the NVIDIA driver install; see <https://docs.nvidia.com/deploy/nvidia-smi/>
- AMD equivalents: `rocm-smi`, `amd-smi` — see <https://rocm.docs.amd.com/>

## Observability and metrics

- Prometheus — <https://prometheus.io/docs/>
- Grafana — <https://grafana.com/docs/>
- OpenTelemetry — <https://opentelemetry.io/docs/>
- Server-Sent Events (SSE) — WHATWG HTML Living Standard — <https://html.spec.whatwg.org/multipage/server-sent-events.html>

## Peer-track and out-of-scope references

- **`ai-infra-performance` (kernel-authoring peer, L35)** — <https://github.com/ai-infra-curriculum/ai-infra-performance-learning>
- **`ai-infra-ml-platform` (registry / artifact peer, L30)** — <https://github.com/ai-infra-curriculum/ai-infra-ml-platform-learning>
- **`ai-infra-sre` (SRE-methodology peer, L30)** — <https://github.com/ai-infra-curriculum/ai-infra-sre-learning>
- **`ai-infra-engineer` (base-platform prerequisite, L20)** — <https://github.com/ai-infra-curriculum/ai-infra-engineer-learning>
- **SRE methodology** (deferred to `ai-infra-sre`) — Google SRE book — <https://sre.google/books/>

## Prior-module anchors

- Module 101 — [`CURRICULUM.md`](../../CURRICULUM.md), [role-boundary chapter](../mod-101-inference-engineering-position/01-role-scope-on-the-ladder.md), [capability-map chapter](../mod-101-inference-engineering-position/02-engine-capability-map.md), [vocabulary chapter](../mod-101-inference-engineering-position/03-inference-vocabulary.md), [GPU-fleet chapter](../mod-101-inference-engineering-position/04-gpu-fleet-landscape.md), [platform-charter chapter](../mod-101-inference-engineering-position/05-platform-charter.md).

## Notes on provenance

- Every vendor-product URL above should be re-verified on each content refresh; vendor documentation reorganises frequently and link rot is normal.
- Every arXiv link is a *preprint*; many of these also appear in OSDI / ISCA / MLSys / NeurIPS proceedings — the conference versions are preferred for citation if available.
- Where a chapter claim is marked `<!-- needs-research: ... -->`, treat it as a planning assumption and consult the primary source listed here before acting on it. The deprecation state of TGI and the Blackwell / Ironwood roadmaps are the specific items most likely to shift between content refreshes.
