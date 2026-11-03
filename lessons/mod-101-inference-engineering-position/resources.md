# Resources — Module 101

Primary / citable references for the topics covered in this module. Prefer official docs and peer-reviewed / arXiv papers to blog posts; where only a blog post exists for a current product claim, note it and verify against the official docs on the next refresh.

## Role-boundary / curriculum context

- **`CURRICULUM.md`** — the role's ownership rule and peer-track map. [../../CURRICULUM.md](../../CURRICULUM.md).
- **`JOB_REQUIREMENTS.md`** — the requirements-to-coverage map that this module's objectives roll up to. [../../JOB_REQUIREMENTS.md](../../JOB_REQUIREMENTS.md).
- **Sibling tracks (ownership boundaries):**
  - `ai-infra-performance-learning` — <https://github.com/ai-infra-curriculum/ai-infra-performance-learning>
  - `ai-infra-ml-platform-learning` — <https://github.com/ai-infra-curriculum/ai-infra-ml-platform-learning>
  - `ai-infra-sre-learning` — <https://github.com/ai-infra-curriculum/ai-infra-sre-learning>
  - `ai-infra-mlops-learning` — <https://github.com/ai-infra-curriculum/ai-infra-mlops-learning>
  - `ai-infra-security-learning` — <https://github.com/ai-infra-curriculum/ai-infra-security-learning>
  - AI Engineering family: `llm-application-developer-learning`, `rag-engineer-learning`, `applied-ai-engineer-learning` under <https://github.com/ai-engineering-curriculum>

## Open-source inference engines (primary sources)

### vLLM

- Project home — <https://github.com/vllm-project/vllm>
- Documentation — <https://docs.vllm.ai/>
- PagedAttention paper: Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention" — <https://arxiv.org/abs/2309.06180>
- Release notes — <https://github.com/vllm-project/vllm/releases>

### SGLang

- Project home — <https://github.com/sgl-project/sglang>
- Documentation — <https://docs.sglang.ai/>
- RadixAttention paper: Zheng et al., "SGLang: Efficient Execution of Structured Language Model Programs" — <https://arxiv.org/abs/2312.07104>

### TensorRT-LLM

- Project home — <https://github.com/NVIDIA/TensorRT-LLM>
- Documentation — <https://nvidia.github.io/TensorRT-LLM/>
- NVIDIA developer blog on TensorRT-LLM — <https://developer.nvidia.com/blog/tag/tensorrt-llm/>

### NVIDIA Dynamo

- Project home — <https://github.com/ai-dynamo/dynamo>
- Documentation — <https://docs.nvidia.com/dynamo/>
- Dynamo overview on NVIDIA developer blog — <https://developer.nvidia.com/blog/tag/dynamo/>

### Triton Inference Server

- Project home — <https://github.com/triton-inference-server/server>
- Documentation — <https://docs.nvidia.com/deeplearning/triton-inference-server/>

### llama.cpp

- Project home — <https://github.com/ggml-org/llama.cpp>
- GGUF format spec — <https://github.com/ggml-org/ggml/blob/master/docs/gguf.md>

### MLC-LLM

- Project home — <https://github.com/mlc-ai/mlc-llm>
- Documentation — <https://llm.mlc.ai/docs/>

### TGI (historical compare-point)

- Project home — <https://github.com/huggingface/text-generation-inference>
- Documentation — <https://huggingface.co/docs/text-generation-inference/>
- <!-- needs-research: lock the current maintenance status of TGI against Hugging Face's most recent announcement; CURRICULUM.md references a 2026 archival signal — verify the exact date and current recommended-serving guidance. -->

## Commercial compare-points (vendor-primary sources)

- NVIDIA NIM — <https://docs.nvidia.com/nim/> and <https://build.nvidia.com/>
- Fireworks AI — <https://docs.fireworks.ai/>
- Together AI — <https://docs.together.ai/>
- Baseten — <https://docs.baseten.co/>
- Modal — <https://modal.com/docs>
- Groq — <https://console.groq.com/docs>
- Cerebras — <https://inference-docs.cerebras.ai/>

## Scheduling, batching, and KV-cache research

- **Continuous batching (Orca):** Yu et al., "Orca: A Distributed Serving System for Transformer-Based Generative Models" — OSDI 2022 — <https://www.usenix.org/conference/osdi22/presentation/yu>
- **PagedAttention / vLLM:** Kwon et al. (above)
- **RadixAttention / SGLang:** Zheng et al. (above)
- **Chunked prefill (SARATHI):** Agrawal et al., "SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills" — <https://arxiv.org/abs/2308.16369>
- **SARATHI-Serve** — Agrawal et al., "Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve" — OSDI 2024 — <https://www.usenix.org/conference/osdi24/presentation/agrawal>
- **Disaggregated prefill–decode (DistServe):** Zhong et al., "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving" — OSDI 2024 — <https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin>
- **Splitwise:** Patel et al., "Splitwise: Efficient Generative LLM Inference Using Phase Splitting" — ISCA 2024 — <https://arxiv.org/abs/2311.18677>

## Speculative decoding research

- **Draft–target speculative decoding:** Leviathan et al., "Fast Inference from Transformers via Speculative Decoding" — <https://arxiv.org/abs/2211.17192>
- **Medusa:** Cai et al., "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads" — <https://arxiv.org/abs/2401.10774>
- **EAGLE:** Li et al., "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty" — <https://arxiv.org/abs/2401.15077>
- **Lookahead decoding:** Fu et al., "Break the Sequential Dependency of LLM Inference Using Lookahead Decoding" — <https://arxiv.org/abs/2402.02057>

## Multi-adapter / LoRA serving

- **S-LoRA:** Sheng et al., "S-LoRA: Serving Thousands of Concurrent LoRA Adapters" — <https://arxiv.org/abs/2311.03285>
- **Punica:** Chen et al., "Punica: Multi-Tenant LoRA Serving" — <https://arxiv.org/abs/2310.18547>

## Structured output / constrained decoding

- **Outlines:** <https://github.com/dottxt-ai/outlines>
- **XGrammar:** <https://github.com/mlc-ai/xgrammar>
- Both have docs linked from their respective repos; both are the backing libraries for several engines' structured-output modes.

## OpenAI API surface and streaming

- OpenAI API reference — <https://platform.openai.com/docs/api-reference>
- Server-Sent Events (SSE) — WHATWG HTML Living Standard — <https://html.spec.whatwg.org/multipage/server-sent-events.html>
- Model Context Protocol (MCP) — <https://modelcontextprotocol.io/>

## GPU hardware landscape (vendor-primary)

### NVIDIA

- Hopper architecture whitepaper — NVIDIA — <https://resources.nvidia.com/en-us-tensor-core/gtc22-whitepaper-hopper>
- H100 / H200 product pages — <https://www.nvidia.com/en-us/data-center/h100/>, <https://www.nvidia.com/en-us/data-center/h200/>
- Blackwell architecture overview — <https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/>
- GB200 NVL72 — <https://www.nvidia.com/en-us/data-center/gb200-nvl72/>
- Rubin platform (announcements) — NVIDIA GTC pages (verify current URL per announcement)
- NVLink / NVSwitch — <https://www.nvidia.com/en-us/data-center/nvlink/>
- Transformer Engine (FP8 library) — <https://github.com/NVIDIA/TransformerEngine>
- Data Center GPU Manager (DCGM) — <https://docs.nvidia.com/datacenter/dcgm/>

### AMD

- Instinct MI300X product page — <https://www.amd.com/en/products/accelerators/instinct/mi300.html>
- Instinct MI325X product page — <https://www.amd.com/en/products/accelerators/instinct/mi300/mi325x.html>
- ROCm documentation — <https://rocm.docs.amd.com/>
- RCCL (collective comms) — <https://github.com/ROCm/rccl>

### AWS

- Trainium / Inferentia product pages — <https://aws.amazon.com/machine-learning/trainium/>, <https://aws.amazon.com/machine-learning/inferentia/>
- AWS Neuron SDK — <https://awsdocs-neuron.readthedocs-hosted.com/>

### Google

- Cloud TPU product pages — <https://cloud.google.com/tpu>
- TPU v5p / v5e / v6 (Trillium / Ironwood) announcements — <https://cloud.google.com/blog/products/compute> (verify current URLs per announcement)
- JAX — <https://docs.jax.dev/> ; Pallas (TPU kernel) — <https://docs.jax.dev/en/latest/pallas/>

## Peer-track + out-of-scope references

- **SRE methodology (deferred to `ai-infra-sre`):** "Site Reliability Engineering" (Google) — <https://sre.google/books/>
- **MoE serving architectures:** DeepSeek-V3 Technical Report — <https://arxiv.org/abs/2412.19437> ; Mixtral of Experts — <https://arxiv.org/abs/2401.04088>
- **Model Context Protocol server-side** (deferred into module 111) — <https://modelcontextprotocol.io/>

## Notes on provenance

- Verify every vendor-product URL on each content refresh; vendor documentation reorganises frequently and link rot is normal.
- Every arXiv link above is a *preprint*; many of these also appear in OSDI / ISCA / NeurIPS / MLSys proceedings — the conference versions are preferred for citation if available.
- Where a claim in a chapter is marked `<!-- needs-research: ... -->`, treat the chapter claim as a planning assumption and consult the primary source listed here before acting on it.
