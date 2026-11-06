# Chapter 01 — Multi-LoRA Serving with S-LoRA and Punica

> One base model in HBM, a thousand MiB-scale adapters on the shelf, every request in the batch potentially using a different adapter. The abstraction that makes this plausible is what you need to learn to configure, not re-implement.

## Motivation

A product team ships twenty tenants, each with their own fine-tuned LoRA on top of the same 7B base model. A naive deployment loads twenty copies of the 7B — forty GiB of HBM gone before you've served a request. A slightly less naive one serves each tenant from its own worker — the fleet multiplies by twenty, and the smallest tenants' GPUs sit idle. The right deployment serves all twenty tenants from one fleet of workers, with one copy of the base model per worker and twenty adapter sidecars that can be *batched together* in a single engine iteration — tenant A's request and tenant F's request go through the same `base_model.forward()` and diverge only in the LoRA GEMM that adds the per-request `(B · A · x) · α / r` delta.

That is S-LoRA / Punica. The operator-level question is not *how* the batched LoRA GEMM is implemented (that is kernel craft, deferred to the performance peer), but *how to configure the engine* so a thousand adapters can coexist, how to feed the adapter registry, how to size HBM for the warm set, and how to debug a drop-off in batched-LoRA throughput when a batch suddenly contains eight different adapter ranks.

By the end of this chapter you can:

- Draw the batched-LoRA forward pass on a whiteboard (base GEMM + per-request LoRA GEMM, both batched in a single kernel launch).
- Enumerate the adapter-registry schema (ID, base model, rank, alpha, target modules, path, SHA, licence).
- Size the per-adapter memory footprint in bytes and the fleet's HBM budget against a target warm-set size.
- Launch vLLM and SGLang with multi-LoRA enabled and verify the engine is batching heterogeneous adapters rather than serializing them.
- Recognise the kernel-level signatures (ragged-rank penalty, warm-up stall on first use) and route them correctly between this module and the performance peer.

The specific flag set and per-request field names below are drawn from current vLLM ([docs.vllm.ai LoRA guide](https://docs.vllm.ai/en/latest/features/lora/)) and SGLang ([docs.sglang.io LoRA guide](https://docs.sglang.io/docs/advanced_features/lora)); defaults and rank constraints still rotate between engine releases, so pin to the version your fleet runs.

## The structural picture — base-plus-adapter forward pass

A LoRA (Hu et al., 2021 — see [`resources.md`](resources.md)) adds a low-rank update to one or more weight matrices of a base model. For an attention or MLP projection `W ∈ ℝ^{d_out × d_in}`, LoRA parameterises an update as `ΔW = B · A` where `B ∈ ℝ^{d_out × r}` and `A ∈ ℝ^{r × d_in}` with `r ≪ min(d_out, d_in)`. The forward pass becomes:

```
y = W x + α / r · B (A x)
```

The `α / r` factor is the LoRA scaling (sometimes written `α`). For a 7B model with rank-16 LoRAs on the attention Q, K, V, O projections and the MLP gate / up / down projections, each adapter adds roughly:

```
params_per_adapter ≈ num_target_modules · 2 · r · d_model
                    (for a 7B with d_model = 4096, r = 16, 7 target modules per layer, 32 layers)
                   ≈ 32 · 7 · 2 · 16 · 4096 ≈ 29 M parameters
                   ≈ 58 MiB in FP16, ≈ 29 MiB in FP8 / INT8 adapter storage
```

That is one to two orders of magnitude smaller than the base model (7B in BF16 is ~14 GiB). Many such adapters fit on one worker. The *batching* question — can a single engine iteration apply tenant A's adapter to request 1, tenant F's to request 2, and the base-only path to request 3 — is what S-LoRA and Punica solve.

### What the engine does per iteration

For each decode (or prefill) step across a batch of `N` requests:

1. **Base-model forward pass.** `base_model.forward(X_batch) → H_batch` runs the full 7B GEMM-and-attention stack over the concatenated batch. No per-request branching.
2. **Per-request LoRA GEMM.** For each layer that has LoRA target modules, the engine performs a *batched* `B_i (A_i x_i)` for every request `i` that is using adapter `i`. The batching here is where the two abstractions matter.

Three implementations:

- **Punica** (Chen et al., 2023 — see [`resources.md`](resources.md)) introduced the **SGMV** (Segmented Gather Matrix-Vector multiplication) kernel. A single CUDA kernel takes the whole batch and a per-request adapter-index tensor, gathers each request's `A_i` and `B_i`, and performs the LoRA update inline. Requests using the same adapter are grouped (segmented) for throughput; requests using different adapters coexist in the same kernel launch.
- **S-LoRA** (Sheng et al., 2023 — see [`resources.md`](resources.md)) extended this with a *unified paging abstraction* for adapters: adapters live in a paged pool on the GPU (analogous to the KV cache's PagedAttention), the engine lazy-loads them from host memory on demand, and the batched kernel can handle **heterogeneous ranks** in the same batch. S-LoRA also demonstrated serving thousands of adapters per worker in the research setting.
- **vLLM / SGLang integration.** vLLM ships a Punica-shape batched LoRA path (the kernel is `punica` under `csrc/`); SGLang ships its own batched LoRA. TRT-LLM has multi-LoRA support via a separate converter. Each engine exposes its own flag set and limits.

The operator's rule: the kernel is the engine's problem; the registry, residency, and budgeting are yours.

## The adapter registry

The engine needs to know, for each request, which adapter to apply. The registry is the data structure that lets the request specify an adapter by ID. A minimal registry entry:

```yaml
adapter_id: "customer-7f3a-support-tone-v3"      # the ID the request references
base_model: "meta-llama/Llama-3.1-8B-Instruct"   # must match the serving fleet's base
base_model_revision: "<HF revision SHA>"         # pinned
adapter_path: "s3://ml-adapters/customer-7f3a/support-tone/v3/"   # or hf://, or local
adapter_sha256: "<SHA of adapter.safetensors>"
rank: 16
alpha: 32                                        # LoRA α (not α / r — stored directly)
target_modules: ["q_proj", "k_proj", "v_proj", "o_proj",
                 "gate_proj", "up_proj", "down_proj"]
dtype: "float16"                                 # adapter weights' storage dtype
licence: "<customer-contract-ref>"
tenant: "customer-7f3a"
status: "active"                                 # active | staging | deprecated | revoked
loaded_version: "v3"                             # for in-flight-swap atomicity
```

The engine consumes this registry two ways:

- **Startup.** It is given a list of adapters to pre-load (the pinned warm set). It validates each adapter against the base model's architecture (number of layers, `d_model`, target-module names must match the base's), reserves HBM for them, and loads them.
- **Per request.** The request header carries `adapter_id`; the engine looks it up in the registry, verifies the adapter is loaded (or lazy-loads it; see chapter 04), and runs the batched LoRA path for that request.

### Request shape

vLLM's OpenAI-compatible path accepts the adapter by passing the loaded LoRA name in the request's `model` field — the engine registers each adapter under its own served-model name alongside the base ([vLLM LoRA docs](https://docs.vllm.ai/en/latest/features/lora/)). SGLang accepts a per-request `lora_path` field on `/generate`, with one entry per batched sequence (`null` for base-only) ([SGLang LoRA docs](https://docs.sglang.io/docs/advanced_features/lora)). TRT-LLM uses a separate adapter-id parameter in the inflight-batcher. Three different shapes; one semantic.

### Rank, α, target modules — what each means

- **Rank `r`.** The dimensionality of the low-rank update. Common values: 4, 8, 16, 32, 64. Larger rank = more adapter capacity = more parameters per adapter.
- **Scaling `α`.** Scales the LoRA contribution. The effective scale applied at inference is `α / r` (per Hu et al.). A common convention is to set `α = 2r`, giving an effective scale of 2; another is `α = r`, giving a scale of 1. The adapter card specifies which.
- **Target modules.** Which weight matrices the LoRA attaches to. For a decoder-only transformer, typical targets are the attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`) and the MLP projections (`gate_proj`, `up_proj`, `down_proj`). Some adapters target only Q/V; some target everything including the embedding and LM head (rarely).

The **operator-visible consequence**: a batch containing adapters with different ranks or different target-module sets is harder for the batched kernel to pack efficiently. S-LoRA's heterogeneous-rank support is specifically what addresses this; Punica originally required homogeneous rank per batch. The engine's docs will state its current support surface. If your fleet's adapters are all rank-16 with the same target modules, the batched path is straightforward; if they are a zoo of ranks and targets, you'll pay a packing cost (see "ragged-rank penalty" below).

## Memory footprint — per adapter and warm set

The byte-math composes directly with the module-104 KV budget. Per-adapter bytes:

```
bytes_per_adapter = sum over target modules of (
    (B_i size) + (A_i size)
)

For a target module with input dim d_in and output dim d_out:
    A_i is r × d_in  →  r · d_in · dtype_bytes
    B_i is d_out × r → d_out · r · dtype_bytes

For a 7B (d_model = 4096, 32 layers) with LoRA on
Q,K,V,O (all 4096 → 4096) and gate,up,down (4096 → 11008, 11008 → 4096 for down):

    attention target modules: 4 × (4096 + 4096) · r · bytes = 4 · 8192 · r · bytes
    MLP targets:              3 × (4096 + 11008) · r · bytes = 3 · 15104 · r · bytes
    per-layer total:          (4 · 8192 + 3 · 15104) · r · bytes
                            = (32768 + 45312) · r · bytes = 78080 · r · bytes
    per-adapter (32 layers):  32 · 78080 · r · bytes = 2 498 560 · r · bytes

At r = 16, FP16 (2 bytes/elem):  2 498 560 · 16 · 2 ≈ 76 MiB per adapter.
At r = 16, FP8  (1 byte /elem):  2 498 560 · 16 · 1 ≈ 38 MiB per adapter.
```

These numbers assume the target-module set listed above. Adapters that touch fewer modules (e.g., Q/V only) are proportionally smaller; always sanity-check against the adapter card and the actual safetensors file size before planning a warm set.

The HBM budget for the warm set is then:

```
HBM_warm_set = base_model_bytes
             + KV_pool_bytes               (from module 104)
             + N_warm · bytes_per_adapter
             + reserves + activations
```

For a 7B BF16 base (~14 GiB) on an 80 GiB H100 with a 40 GiB KV pool and 76 MiB adapters, the warm set bounds at:

```
N_warm ≤ (80 - 14 - 40 - 10) / 0.076 ≈ 200 adapters per worker (BF16 adapter storage).
```

With FP8 adapter storage (half the per-adapter bytes), ~400 adapters per worker. Chapter 04 operates this budget — how to pin the critical adapters, how to let the LRU set roll, how to cost-account the cold-start.

## Launching the engine with multi-LoRA

### vLLM

```
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --enable-lora \
    --max-loras 32 \
    --max-lora-rank 64 \
    --max-cpu-loras 256 \
    --lora-modules customer-7f3a-support=s3://.../customer-7f3a/support/v3 \
                   customer-a21-code=s3://.../customer-a21/code/v2 \
                   ... \
    --max-model-len 8192
```

Key flags (verify spellings against your pinned version):

- `--enable-lora` turns on the Punica path.
- `--max-loras` caps the number of LoRAs the batched kernel can serve in one iteration (not the total number loaded). Larger values trade batched-kernel overhead for parallel-adapter throughput.
- `--max-lora-rank` is the maximum rank any served adapter can have. The engine allocates adapter slots sized for this rank — oversizing wastes HBM, undersizing rejects adapters at load.
- `--max-cpu-loras` is the number of adapters cached on host DRAM for lazy load into HBM. Chapter 04 operates this.
- `--lora-modules` registers startup adapters by `name=path` pairs.

For runtime loading, vLLM exposes `POST /v1/load_lora_adapter` (body: `lora_name`, `lora_path`) and `POST /v1/unload_lora_adapter` (body: `lora_name`), gated behind `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`; the vLLM docs explicitly warn against enabling this in untrusted/production environments ([vLLM LoRA — dynamic serving](https://docs.vllm.ai/en/latest/features/lora/)).

### SGLang

```
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --lora-paths lora_support=... lora_code=... \
    --max-loras-per-batch 16
```

SGLang also exposes `--enable-lora`, `--max-lora-rank`, `--max-loaded-loras` (shelf cap), and dynamic `POST /load_lora_adapter` / `POST /unload_lora_adapter` endpoints; per-request selection is a `lora_path` list on `/generate` with one entry per sequence ([SGLang LoRA docs](https://docs.sglang.io/docs/advanced_features/lora), [SGLang server args](https://docs.sglang.io/docs/advanced_features/server_arguments)).

### TRT-LLM

TRT-LLM requires converting the base model with LoRA plugins enabled at build time, then loading adapters at runtime via the inflight-batcher adapter table. See the TRT-LLM LoRA documentation; the build flag is typically `--lora_plugin fp16` (or `fp8` for FP8 adapters) and the serving-side uses the LoRA cache manager. More operational friction than vLLM / SGLang; verify current status for your TRT-LLM version.

### Health check — did the engine actually batch?

The question that matters is not "did the adapter load" (easy to verify) but "is the batched LoRA path running efficiently." Instrumentation to check:

- **Per-iteration counter: `num_requests` vs. `num_adapters_in_batch`.** If the ratio is 1:1 for every batch, every request is using a different adapter — potentially fine, but check that the batched kernel is doing the segmented work rather than falling back to a per-request path. The engine's metrics (vLLM: look for `vllm:lora_request_info` or similar; SGLang: equivalent) tell you.
- **Per-iteration latency with adapters vs. without.** Compare a request with `adapter_id=None` (base model only) and a request with an adapter attached. The adapter-attached iteration should cost on the order of `α · d_model · r` FLOPs more; measurable but small. If it costs 2–3× more, the batched path is probably not firing — route to the performance peer with the specific kernel timing.

## Common failure modes

### Rank-mismatch rejection at load

The adapter card declares `rank = 32`; the engine was launched with `--max-lora-rank 16`. The adapter is rejected at load. Fix: launch with `--max-lora-rank` at least as large as the largest adapter in your registry; or filter the registry.

### Target-module mismatch

The adapter targets `up_proj`, `down_proj`, `gate_proj`, `q_proj`, `k_proj`, `v_proj`, `o_proj`; the base model's module names are `w1`, `w2`, `w3`, `wq`, `wk`, `wv`, `wo` (older Llama naming). The adapter cannot bind. Fix: produce a target-module alias map, or re-train the adapter against the canonical names. This is more common than it should be; different fine-tuning toolchains use different module names.

### Base-model mismatch

The adapter was trained on `Llama-3-8B-Instruct` (3.0); the fleet serves `Llama-3.1-8B-Instruct` (3.1). The adapter loads but produces garbage. The registry's `base_model` and `base_model_revision` fields catch this at load time if you enforce them. **Enforce them.**

### Ragged-rank penalty

A batch contains four requests at rank 8, four at rank 32, four at rank 64. The segmented matmul must handle three different inner dimensions in the same kernel; the kernel pads to the max rank (rank 64), wasting work on the rank-8 requests. Observable as reduced throughput at mixed-rank batches vs. homogeneous-rank batches. Both vLLM's Punica-derived kernels ([vLLM Punica wrapper](https://docs.vllm.ai/en/latest/api/vllm/lora/punica_wrapper/punica_gpu/)) and SGLang's LoRA path accept heterogeneous ranks in one batch, but every adapter's rank must be ≤ the startup-time `--max-lora-rank` and the kernel pads to that cap — so a fleet with wildly ragged ranks still pays a packing tax. Mitigation: tier your adapters by rank and admit mixed-rank batches only when the throughput budget allows; profile the mixed-rank path on your engine version.

### Warm-up stall on first use

The adapter is loaded but has not been seen by a CUDA graph. First inference with the adapter triggers graph recompile; TTFT for that request spikes. Mitigation: warm up every pinned adapter at startup with a synthetic forward pass. The engine may already do this; verify.

### Loading an adapter that targets layers not loaded

An adapter was trained with target modules that include `embed_tokens` (the embedding table). The engine's LoRA path may not support LoRA on the embedding. Load fails or silently drops that target. Fix: enforce "attention + MLP projections only" at the fine-tuning side (adapter-training peer: fine-tuning-engineer); verify at the serving side.

## The boundary to the performance peer

This chapter does not teach:

- How Punica's SGMV kernel computes a segmented matrix-vector product with per-request indices in one warp-level reduction.
- How S-LoRA's unified paging interleaves adapter blocks with KV blocks in a shared HBM pool.
- How the engine's attention backend composes with the LoRA GEMM (fused epilogue vs. separate kernel vs. graph fusion).
- How the batched kernel degrades at mixed ranks and what the cutoff rank for degradation is.

Those are kernel-authoring questions. They live at [`ai-infra-performance`](https://github.com/ai-infra-curriculum/ai-infra-performance-learning). If your drill lands on "the Punica kernel underperforms on this GPU at this batch shape," you ship the trace, the batch shape, and the measurement to the performance peer; you do not open `.cu` yourself.

Where this chapter does send you to the performance peer:

- A multi-LoRA batch shows disproportionate TPOT regression (adapter-attached decode is 3× base decode, not 1.05× base decode).
- The engine falls back to a per-request LoRA path (not a batched kernel) under some batch shapes.
- Adapter loading triggers a CUDA-graph rebuild storm (graph recompile cost dominates).
- The engine's `nvprof` / Nsight timeline shows the LoRA GEMM is memory-bandwidth-bound where theory says it should be compute-bound (or vice versa).

The hand-off bundle: trace timeline, batch shape (number of requests per adapter, rank distribution, target-module set), engine version, GPU model, and the base-vs-adapter latency numbers.

## Summary

- A LoRA is a low-rank update `ΔW = B · A` that composes with a base-model weight at inference. S-LoRA and Punica are the abstractions that batch per-request LoRA updates in a single engine iteration.
- The adapter registry is the data structure the engine consumes to resolve `adapter_id` to a loaded adapter. Minimum fields: ID, base model + revision, path + SHA, rank, α, target modules, dtype, licence, tenant, status.
- Per-adapter memory footprint is on the order of 10–100 MiB per adapter for a 7B base at rank 16. Many adapters fit on one worker; the warm-set size is chapter-04's budget.
- vLLM, SGLang, and TRT-LLM each expose multi-LoRA with different flag spellings — verify against the pinned engine version.
- The common failure modes are rank mismatch, target-module mismatch, base-model mismatch, ragged-rank batching penalty, and warm-up stalls. All are configuration / registry issues, not kernel issues.
- Kernel-level questions (batched-GEMM efficiency, ragged-rank cost, graph fusion) route to the performance peer.
