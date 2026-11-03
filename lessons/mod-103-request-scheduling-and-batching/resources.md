# Resources — Module 103

Primary / citable references for the topics covered in this module. Prefer official repos, release notes, and peer-reviewed / arXiv papers to blog posts; where only a blog post exists for a current product claim, note it and verify against official docs on the next refresh.

Many flag names, metric names, and defaults in the chapters are tagged with `<!-- needs-research: ... -->` or caveated as "verify against the version your fleet runs." These resources are the sources of truth to verify against.

## Research — scheduling, continuous batching, chunked prefill

- **Orca (continuous / iteration-level batching):** Yu et al., "Orca: A Distributed Serving System for Transformer-Based Generative Models" — OSDI 2022 — <https://www.usenix.org/conference/osdi22/presentation/yu>
- **PagedAttention / vLLM:** Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention" — SOSP 2023 — <https://arxiv.org/abs/2309.06180>
- **Chunked prefill (SARATHI):** Agrawal et al., "SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills" — <https://arxiv.org/abs/2308.16369>
- **SARATHI-Serve:** Agrawal et al., "Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve" — OSDI 2024 — <https://www.usenix.org/conference/osdi24/presentation/agrawal>
- **Disaggregated prefill-decode (DistServe):** Zhong et al., "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving" — OSDI 2024 — <https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin>
- **Splitwise:** Patel et al., "Splitwise: Efficient Generative LLM Inference Using Phase Splitting" — ISCA 2024 — <https://arxiv.org/abs/2311.18677>
- **RadixAttention / SGLang:** Zheng et al., "SGLang: Efficient Execution of Structured Language Model Programs" — <https://arxiv.org/abs/2312.07104>
- **FlexGen (useful for the swap-vs-recompute intuition):** Sheng et al., "FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU" — ICML 2023 — <https://arxiv.org/abs/2303.06865>

## vLLM — scheduler, chunked prefill, metrics

- Project home — <https://github.com/vllm-project/vllm>
- Documentation — <https://docs.vllm.ai/>
- Release notes — <https://github.com/vllm-project/vllm/releases>
- Engine args reference — `vllm serve --help` (authoritative; mirrored in the docs)
- Metrics reference — <https://docs.vllm.ai/en/latest/serving/metrics.html>
- Prefix caching docs — the "automatic prefix caching" section of the docs (verify current path)
- Chunked prefill docs — the chunked-prefill section of the docs (verify current path)
- Scheduler source — in-tree under `vllm/core/scheduler.py` (verify path for your release — the scheduler module has been refactored across versions)
- Benchmark serving script — `benchmarks/benchmark_serving.py` in-tree

## SGLang — scheduler, chunked context, metrics

- Project home — <https://github.com/sgl-project/sglang>
- Documentation — <https://docs.sglang.ai/>
- Release notes — <https://github.com/sgl-project/sglang/releases>
- Server launch and args reference — the server-arguments section of the SGLang docs
- RadixAttention documentation — <https://docs.sglang.ai/> (search for "radix attention")
- Chunked prefill knob — the server-args reference (search for `chunked-prefill`)
- Scheduler source — in-tree under `python/sglang/srt/managers/scheduler.py` (verify path for your release)

## TensorRT-LLM, Triton, in-flight batching

- TensorRT-LLM project home — <https://github.com/NVIDIA/TensorRT-LLM>
- TensorRT-LLM documentation — <https://nvidia.github.io/TensorRT-LLM/>
- Release notes — <https://github.com/NVIDIA/TensorRT-LLM/releases>
- In-flight batching / chunked context — the "advanced" or "in-flight batching" section of the TRT-LLM docs
- Triton Inference Server project home — <https://github.com/triton-inference-server/server>
- Triton documentation — <https://docs.nvidia.com/deeplearning/triton-inference-server/>
- `tensorrtllm_backend` — <https://github.com/triton-inference-server/tensorrtllm_backend>
- Backend `config.pbtxt` parameters — the model-configuration docs: <https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html>

## NVIDIA Dynamo (scheduler at the fleet level)

- Project home — <https://github.com/ai-dynamo/dynamo>
- Documentation — <https://docs.nvidia.com/dynamo/>
- Dynamo-related NVIDIA developer blog posts — <https://developer.nvidia.com/blog/tag/dynamo/>

## Prefill / decode disaggregation (deferred to mod-107, included for cross-reference)

- See the DistServe and Splitwise papers above.
- NVIDIA Dynamo documentation on disaggregated serving (see Dynamo section).
- NIXL (NVIDIA Inference Xfer Library, KV transport) — referenced from the Dynamo docs.

## Attention backends (chunked-prefill-aware)

- **Flash Attention 2:** Dao, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning" — <https://arxiv.org/abs/2307.08691>
- **Flash Attention 3:** Shah et al., "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision" — <https://arxiv.org/abs/2407.08608>
- **FlashInfer:** project home — <https://github.com/flashinfer-ai/flashinfer>; paper — <https://arxiv.org/abs/2501.01005> (verify current citation).
- CUDA graphs (relevant to iteration-time stability): NVIDIA CUDA documentation — <https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cuda-graphs>

## Observability — Prometheus / Grafana / Alertmanager

- Prometheus documentation — <https://prometheus.io/docs/>
- Prometheus histogram and `histogram_quantile` reference — <https://prometheus.io/docs/practices/histograms/>
- Alertmanager documentation — <https://prometheus.io/docs/alerting/latest/alertmanager/>
- Grafana documentation — <https://grafana.com/docs/>
- OpenTelemetry documentation — <https://opentelemetry.io/docs/>
- SRE workbook — alerting on SLOs: <https://sre.google/workbook/alerting-on-slos/>
- SRE book — service level objectives: <https://sre.google/sre-book/service-level-objectives/>

## GPU observability

- NVIDIA DCGM — <https://docs.nvidia.com/datacenter/dcgm/>
- DCGM Exporter for Prometheus — <https://github.com/NVIDIA/dcgm-exporter>
- `nvidia-smi` reference — part of the NVIDIA driver install; see <https://docs.nvidia.com/deploy/nvidia-smi/>
- AMD equivalents: `rocm-smi`, `amd-smi` — see <https://rocm.docs.amd.com/>

## Load generators and benchmarking

- vLLM `benchmark_serving.py` — in-tree under `benchmarks/` (verify path per release)
- SGLang benchmark utilities — in-tree under `benchmark/` (verify path)
- TensorRT-LLM `benchmarks/` — in-tree
- Locust (general-purpose HTTP load generator) — <https://locust.io/>
- `wrk` (HTTP benchmarking) — <https://github.com/wg/wrk>
- Server-Sent Events (SSE) spec (relevant to streaming benchmarks) — WHATWG HTML Living Standard — <https://html.spec.whatwg.org/multipage/server-sent-events.html>

## Example compose / wiring references for Prometheus + Grafana

- Prometheus Operator (Kubernetes) — <https://github.com/prometheus-operator/prometheus-operator>
- Grafana dashboards for vLLM — community dashboards exist in the Grafana dashboard registry; verify freshness before trusting panel definitions.

## Peer-track and out-of-scope references

- **`ai-infra-performance` (kernel-authoring peer, L35)** — <https://github.com/ai-infra-curriculum/ai-infra-performance-learning> — kernel-level ownership (Flash Attention variants, mixed prefill-decode attention kernels, FP8/FP4 GEMM epilogues).
- **`ai-infra-sre` (SRE-methodology peer, L30)** — <https://github.com/ai-infra-curriculum/ai-infra-sre-learning> — SLO methodology, error budgets, incident response.
- **`ai-infra-ml-platform` (registry / artifact peer, L30)** — <https://github.com/ai-infra-curriculum/ai-infra-ml-platform-learning>.
- **Google SRE book** (SLO methodology deferred to `ai-infra-sre`) — <https://sre.google/books/>

## Prior-module anchors

- Module 101 — [`CURRICULUM.md`](../../CURRICULUM.md), [vocabulary chapter](../mod-101-inference-engineering-position/03-inference-vocabulary.md).
- Module 102 — [vLLM internals chapter](../mod-102-inference-engine-internals/01-vllm-internals-and-operation.md), [SGLang internals chapter](../mod-102-inference-engine-internals/02-sglang-internals-and-operation.md), [TRT-LLM + Triton chapter](../mod-102-inference-engine-internals/03-trt-llm-and-triton-and-dynamo.md), [failure modes chapter](../mod-102-inference-engine-internals/04-engine-failure-modes-and-triage.md).

## Downstream-module anchors (preview; may be stubs)

- Module 104 — [`mod-104-kv-cache-management`](../mod-104-kv-cache-management/README.md) — PagedAttention and RadixAttention at operator depth; prefix caching; tiered KV stores. The block allocator the scheduler calls into every iteration.
- Module 109 — [`mod-109-autoscaling-fleet-and-resilience`](../mod-109-autoscaling-fleet-and-resilience/README.md) — HPA / KEDA on inference-specific custom metrics (queue depth, prefill-tokens-per-second). The fleet-level knob the scheduler metrics feed.
- Module 110 — [`mod-110-observability-slo-and-cost`](../mod-110-observability-slo-and-cost/README.md) — SLO methodology, burn-rate alerts, cost-per-token accounting. The methodology this module's instrumentation feeds into.
- Module 113 — [`mod-113-production-safety-and-rollouts`](../mod-113-production-safety-and-rollouts/README.md) — canary rollouts, response-diff, auto-rollback — the deploy discipline downstream of scheduler-config changes.

## Notes on provenance

- Every vendor-product URL above should be re-verified on each content refresh; vendor documentation reorganises frequently and link rot is normal.
- Every arXiv link is a *preprint*; many of these also appear in OSDI / ISCA / MLSys / SOSP proceedings — the conference versions are preferred for citation where available.
- Where a chapter claim is marked `<!-- needs-research: ... -->`, treat it as a planning assumption and consult the primary source listed here before acting on it. Scheduler flag defaults and metric spellings are the specific items that drift most across releases.
