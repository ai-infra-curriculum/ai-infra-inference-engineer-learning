# Module 104 — KV-Cache Management as a Runtime Concern

> **Role:** AI Inference Engineer · **Level:** L35 (AI Infrastructure family) · **Hours:** 16

Module 103 treated the KV allocator as a dependency the scheduler calls into. This module opens that dependency and operates it. The KV cache is the single largest moving piece of state in an LLM serving system — bigger than weights for long-context workloads, more fragmented than weights can ever be, and the one object whose management decisions directly bend TTFT, throughput, and cost per token. By the end of this module you can decide a block size, a prefix-cache policy, a tiered-store topology, and a cross-tenant sharing contract — each defended with measurements rather than defaults.

By the end of it you can:

- Walk **PagedAttention** at operator depth — block tables, block-size tuning, copy-on-write semantics, allocator fragmentation, block-swap-to-CPU, block-release discipline. Point at each in the vLLM / SGLang / TRT-LLM source.
- Walk **RadixAttention** at operator depth — radix-tree prefix sharing, cache-aware routing (sticky routing to the shard that holds the prefix), cache-eviction strategy, TTL vs. LRU for conversation state.
- Deploy **prefix caching** for chat (system prompts + conversation prefixes) and RAG (retrieved-context prefixes). Measure cache-hit rate, prefix-cache-hit benefit on TTFT, and the cost of a cache miss.
- Deploy **tiered KV-cache stores** — host-DRAM + NVMe + remote object store — using LMCache / Mooncake-style abstractions. Reason about the TTFT / cost trade-off at each tier.
- Reason about **long-context KV footprint** — bytes per token at each precision, head-dim fan-out, memory headroom budgeting. Author a per-request KV budget.
- Deploy **cross-request KV-cache sharing** safely — per-tenant isolation, PII leakage risk from prefix sharing, when to opt out, how to audit a leak.
- Cite the **boundary to the performance peer** — the KV allocator *algorithms* and attention-kernel contracts live at [`ai-infra-performance`](https://github.com/ai-infra-curriculum/ai-infra-performance-learning); the KV cache as a *runtime object managed in production* is owned here.

> **Prerequisites:** [Module 101](../mod-101-inference-engineering-position/README.md) (vocabulary — TTFT / TPOT / cache-hit / prefix caching), [Module 102](../mod-102-inference-engine-internals/README.md) (five-layer engine anatomy; where the block manager sits in vLLM / SGLang / TRT-LLM), and [Module 103](../mod-103-request-scheduling-and-batching/README.md) (how the scheduler calls `allocate` / `append_slot` / `swap_out` / `swap_in` / `free` every iteration).

## Boundary against `ai-infra-performance`

The KV cache has two sides. One side is the **algorithm** — how PagedAttention lays out blocks for the attention kernel to consume, how the attention backend reads a logical block table and gathers physical blocks, how FlashAttention's variable-length path interprets the layout. That side is kernel craft; it lives at the performance peer. If your diagnosis lands on "the attention kernel is gathering blocks slower than the specialist's benchmark suggests," the fix is a `.cu` / `.cuh` / Triton-DSL / CUTLASS edit — route it to the performance peer.

The other side is the **runtime object**: the block pool as a budget, the radix tree as a cache the fleet shares, the tiered store as an operations problem, the per-tenant sharing policy as a data-boundary. That side is this module. The deliverable here is a config, a policy, a dashboard, and a runbook — not a kernel edit.

Rule of thumb: *if the fix is to the block allocator's data structure or to the kernel that reads it, that is the performance peer. If the fix is to a block-size flag, a prefix-cache TTL, a tiered-store policy, or a sharing contract, that is this module.*

## Boundary against module 103

Module 103 owns the scheduler loop. The scheduler calls into this module's allocator every iteration. If a dashboard shows the scheduler is preempting too often but the KV pool is nowhere near full, that is a 103 problem. If the KV pool is thrashing — prefix cache is cold, blocks are fragmenting, swap is thrashing between DRAM and NVMe — that is a 104 problem. In practice the two interact: this module's decisions change the budget the 103 scheduler has to work with, and the 103 scheduler's admission pressure changes which 104 policies hurt or help.

## Boundary against module 110

This module instruments the KV cache — hit rate, swap rate, block-eviction rate, bytes-in-use per tenant. Module 110 owns the SLO methodology those metrics feed into (burn-rate alerts, cost-per-token accounting). The axes live here; the alerting discipline lives there.

## Chapters

1. [**PagedAttention at Operator Depth**](01-paged-attention-at-operator-depth.md) — virtual-vs-physical block tables; the block manager's `allocate` / `append_slot` / `free` / `swap_out` / `swap_in` API; block-size tuning (why 16 is default, when 8 or 32 wins); copy-on-write for beam-search / parallel-sampling; allocator fragmentation and the internal-fragmentation cost of partial blocks; swap-to-CPU as a preemption helper; the block-release discipline that keeps the pool healthy.
2. [**RadixAttention and Prefix Sharing at Operator Depth**](02-radix-attention-and-prefix-sharing.md) — the radix-tree index of prefix KV; the lookup / insert / match-prefix API; cache-aware routing (why requests sharing a system prompt should land on the same shard); eviction policy (LRU as default, TTL as the conversation-state primitive); the SGLang-style cache-aware router that external gateways emulate.
3. [**Deploying Prefix Caching for Chat and RAG**](03-prefix-caching-for-chat-and-rag.md) — the operator's recipe: system-prompt prefix, conversation prefix, retrieved-context prefix; how to measure the cache-hit rate as an axis (hits / lookups) and the prefix-cache-hit benefit (TTFT saved per hit); the cost signature of a cold miss; defending the knob against a workload shape.
4. [**Tiered KV Stores with LMCache / Mooncake**](04-tiered-kv-stores-lmcache-mooncake.md) — the three tiers (HBM, host DRAM, NVMe or remote object store); the LMCache / Mooncake abstraction over them; the TTFT-vs-cost trade-off at each tier; eviction and promotion between tiers; the operational signature of a tiered store (which SLOs it buys and which it does not).
5. [**Long-Context KV Footprint Budgeting**](05-long-context-kv-footprint-budgeting.md) — bytes per token = `2 × layers × kv_heads × head_dim × dtype_bytes`; the head-dim and GQA fan-out effect; computing the budget for a model + context + batch size; headroom for prefix cache, swap space, and a safety margin; how to decide when a workload *cannot* fit on a given SKU.
6. [**Cross-Tenant KV Sharing Safely**](06-cross-tenant-kv-sharing-safely.md) — the sharing opportunity (common system prompts, templated RAG prefixes) vs. the leakage risk (PII bleed through a shared prefix); per-tenant namespaces; opt-out and opt-in semantics; auditing a shared cache; the data-boundary conversation with the security peer.

## Exercises

The exercises carry each chapter into a hands-on drill you keep as a reusable artefact for the rest of the track:

1. [**PagedAttention Block-Table Walkthrough**](exercises/exercise-01-paged-attention-block-table-walkthrough.md) — locate the block manager and block table in your engine's source; trace one request's block allocations across prefill, decode, and release; demonstrate copy-on-write under parallel sampling; sketch allocator fragmentation on a worksheet.
2. [**RadixAttention Prefix-Cache Tour**](exercises/exercise-02-radix-attention-prefix-cache-tour.md) — walk the SGLang radix cache (or the vLLM automatic-prefix-cache data structure) in source; prove prefix sharing by sending repeated prefixes and observing hit metrics; demonstrate cache-aware routing between workers.
3. [**Prefix-Cache Hit-Rate Measurement on Chat and RAG**](exercises/exercise-03-prefix-cache-hit-rate-measurement-on-chat-and-rag.md) — generate a chat workload (shared system prompt + varied user turns) and a RAG workload (varied queries + a stable retrieved-doc prefix on hot documents); measure hit-rate, TTFT saved per hit, cost of a cold miss; defend a prefix-cache policy against the workload.
4. [**Tiered KV Store with LMCache or Mooncake**](exercises/exercise-04-tiered-kv-store-with-lmcache-or-mooncake.md) — stand up a two- or three-tier KV store (HBM + DRAM, or HBM + DRAM + NVMe); replay a trace; measure which tier each lookup lands in and the TTFT at each tier; author a cost / TTFT trade-off table.
5. [**Long-Context KV Footprint Budget**](exercises/exercise-05-long-context-kv-footprint-budgeting.md) — compute the per-token KV footprint for three real models in BF16, FP8, and INT4; derive the max concurrent-sequence count at a stated context length on a stated SKU; verify against the engine's reported block count; publish a "fits / does not fit" table.
6. [**Cross-Tenant KV-Sharing Safety Review**](exercises/exercise-06-cross-tenant-kv-sharing-safety-review.md) — audit a shared-prefix configuration for leakage risk; design a per-tenant namespace policy; demonstrate the policy with a controlled "attempted leak" test (two tenants sending overlapping-looking prefixes); document the opt-out path.

## Resources

See [`resources.md`](resources.md) for primary / citable references — the PagedAttention and RadixAttention papers, the LMCache and Mooncake references, the vLLM / SGLang / TRT-LLM block-manager and prefix-cache docs, and the capacity / byte-math references the exercises build on.

## Where this fits in the curriculum

- **Before:** [Module 103 — Request Scheduling, Continuous Batching, and Chunked Prefill](../mod-103-request-scheduling-and-batching/README.md). The scheduler is the primary caller of the KV allocator this module operates.
- **After:** [`mod-105-quantization-and-model-ops`](../mod-105-quantization-and-model-ops/README.md) (KV precision and FP8 / FP4 KV cache; the quant choice feeds back into byte-math); [`mod-107-distributed-inference-topologies`](../mod-107-distributed-inference-topologies/README.md) (KV transfer across prefill / decode workers; cross-node KV routing); [`mod-110-observability-slo-and-cost`](../mod-110-observability-slo-and-cost/README.md) (SLO methodology on top of this module's KV metrics).
- **Peer-track hand-off:** when a KV-management drill lands on an attention-kernel finding (gather kernel slower than expected, Flash Attention variable-length path missing a case, CUTLASS-level block-copy regression), the hand-off is to [`ai-infra-performance-learning`](https://github.com/ai-infra-curriculum/ai-infra-performance-learning). This module teaches you to *recognise and route* that hand-off, not to execute it.
- **Security peer hand-off:** when a cross-tenant KV sharing audit (chapter 06, exercise 06) surfaces a data-boundary question — "is a shared prefix across tenant A and tenant B a leak?" — the methodology lives with [`ai-infra-security-learning`](https://github.com/ai-infra-curriculum/ai-infra-security-learning). This module teaches you the serving-side mechanics and when to escalate.
