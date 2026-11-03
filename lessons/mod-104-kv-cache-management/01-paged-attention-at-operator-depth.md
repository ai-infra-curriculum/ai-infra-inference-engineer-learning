# Chapter 01 — PagedAttention at Operator Depth

> The KV cache looks like a single matrix in the textbook and like a pool of 16-token blocks in production. The gap between those two mental models is where every KV-related outage lives.

## Motivation

In module 102 you met the block manager as "one of the five layers." In module 103 the scheduler called `allocate`, `append_slot`, and `swap_out` on it every iteration. This chapter cracks it open. By the end you can name the data structure the block manager actually maintains, the API it exposes to the scheduler, the tuning knob (block size) and what it modulates, the hidden cost (internal fragmentation and copy-on-write), and the operational signature (allocation failures, swap thrash, block-release regressions).

PagedAttention — introduced in the vLLM paper (Kwon et al., SOSP 2023; see [`resources.md`](resources.md)) — is now the default KV layout across production engines. The *algorithm* is kernel-side; the specialist at [`ai-infra-performance-learning`](https://github.com/ai-infra-curriculum/ai-infra-performance-learning) owns its implementation. The *runtime object* — the block pool as a budget, the allocator as a service — is this chapter's subject.

<!-- needs-research: block-manager class and function names in vLLM / SGLang / TRT-LLM rotate between releases. Verify against the exact engine version your fleet runs before quoting them in a runbook. -->

## The problem PagedAttention solves

Before PagedAttention, a serving engine typically pre-allocated a contiguous KV buffer per sequence sized for the sequence's **maximum** possible length. Two consequences:

1. **Internal fragmentation.** A sequence that generated 50 tokens but had a 2048-token max-length reservation wasted (2048 − 50) × per-token-bytes of KV. On a 70B model that is tens of megabytes of HBM per request.
2. **External fragmentation.** Freed buffers were variable-sized; the pool fragmented over time, so even if total free bytes were sufficient, no contiguous span was large enough for the next request. Rejection or restart.

Together, these effects produced KV utilisation in the 20–40% range — the pool looked three-quarters free yet refused to admit new requests.

PagedAttention borrows the operating-system idea: carve KV memory into **fixed-size blocks** (typically 16 tokens each), hand them out one at a time, and use an **indirection table** (the "block table") per sequence to map logical token positions to physical blocks. The attention kernel reads the block table to gather KV from scattered physical blocks. The result: internal fragmentation is bounded by one block per sequence, external fragmentation is eliminated (all blocks are the same size), and effective KV utilisation moves to 90%+.

## The data structures the operator must know

### The block pool

A flat array (or free-list / free-set) of physical KV blocks on the GPU. Each physical block holds `block_size × 2 × num_kv_heads × head_dim × dtype_bytes` bytes — the K and V tensors for `block_size` tokens of one layer. The pool is sized once at engine start from the HBM budget left after weights, activations, and runtime overhead; this is the number vLLM reports as `num_gpu_blocks` on startup.

Example, 70B-class model, BF16 KV, `block_size=16`, `head_dim=128`, `num_kv_heads=8` (GQA), 80 layers:

```
block_bytes = 16 (tokens) × 2 (K, V) × 8 (kv_heads) × 128 (head_dim) × 2 (bf16)
            = 65_536 bytes per layer per block
            × 80 layers
            = 5_242_880 bytes ≈ 5 MiB per block (across the full stack)
```

If the pool has 10_000 blocks that is ~50 GiB of HBM dedicated to KV. The chapter-05 budgeting method is the arithmetic that justifies the "10_000 blocks" number from the HBM you actually have.

(Chapter 05 does the full byte-math, including MQA / GQA fan-out and FP8 KV. Chapter 01 is the mechanics.)

### The block table (per sequence)

A **logical-to-physical** map: logical token index `i` lives in physical block `block_table[i // block_size]` at offset `i % block_size`. Stored as a small integer array per sequence (one entry per 16 tokens). Grows by one entry each time the sequence fills a block and needs a new one.

The attention kernel takes the block table as an input tensor and does a scatter / gather across the physical pool. **This is the point of contact between the operator's abstraction and the kernel specialist's abstraction**: you choose how many blocks the pool has and when they are released; the kernel specialist implements the gather.

### The free-block registry

Which physical blocks are available. Implementations vary — a free-list, a free-set, a bitmap. The operator cares about three operations:

- `pop()` — hand out a free block (returns an index into the pool).
- `push(block_idx)` — return a block to the free pool.
- `size()` — how many blocks are free *right now*.

The `size()` metric is what you alert on. A pool that is 95% full for sustained periods means admission pressure is pushing the scheduler into preemption; a pool that is 100% full means admission is being refused.

### The reference-count table

Each physical block has a reference count. On single-sequence prefill / decode the ref-count is 1 (the sequence holds it). On copy-on-write (parallel sampling, beam search) the ref-count can be > 1. On `free()` the ref-count drops; the block returns to the free registry when it hits zero.

## The API the scheduler calls

The block manager's API has, at the operator's resolution, five calls:

### `allocate(sequence, num_prompt_tokens)`

Called when the scheduler admits a new sequence. Grants it enough blocks to hold the prompt KV. Fails if the free registry has too few blocks.

Internally: compute `ceil(num_prompt_tokens / block_size)` block count, pop that many blocks from the free registry, build the sequence's block table.

### `append_slot(sequence)`

Called each decode step after the sequence generates a token. If the current trailing block has room, nothing happens (the token lands in the next free slot of the trailing block, which is a no-op at the allocator level — the kernel does the write). If the trailing block is full, pops a new block from the free registry and appends to the block table.

The hot path: most decode steps are no-ops at the allocator; only every `block_size`-th decode step triggers a new block allocation.

### `free(sequence)`

Called when a sequence finishes or is preempted-with-release. Pushes all of the sequence's physical blocks back to the free registry. Decrements ref-counts if the sequence shared blocks (copy-on-write — see below).

The subtle operational issue: `free` must be called *promptly*. A delayed free is a block that cannot be reused, which translates directly into admission pressure. Operational symptom: the pool is reported as 90% used but the engine logs show no running sequences at that level of parallelism — a sign of leaked blocks (bugs do exist; cross-reference recent engine release notes).

### `swap_out(sequence, cpu_pool)`

Called by the preemption path (module 103, chapter 04). Copies the sequence's physical blocks to CPU memory, frees the GPU blocks, and keeps the mapping so the sequence can be swapped back later. Returns the number of GPU blocks freed.

Costs:
- PCIe transfer of all the sequence's KV, in megabytes per sequence.
- The sequence's slot in the running queue is released; it goes to a "swapped" state.

### `swap_in(sequence, gpu_pool)`

Called when the scheduler resumes a swapped-out sequence. Pops GPU blocks, copies KV back from CPU, rebuilds the block table. Costs PCIe transfer in the same direction.

### Operational note: the API is not necessarily named this way

The exact method names vary by engine, by release, and by refactor. vLLM's block manager has had multiple generations (`BlockSpaceManagerV1`, `V2`, newer unified manager; verify your release). SGLang's token-attention / radix cache code lays out blocks differently again. TRT-LLM's batch manager is C++. The *operations* this chapter names — allocate on admit, append on decode, free on finish, swap on preempt — are universal. The specific symbol you grep for rotates.

## Block size — the primary tuning knob

Block size is the one dial PagedAttention gives you. Default is **16 tokens** in vLLM and most derivatives. The trade-off:

### Smaller blocks (e.g. 8)

- **Lower internal fragmentation.** A sequence that generates 10 tokens wastes at most `block_size − 10 = 6` tokens of a block, not 22. On high-concurrency workloads with short outputs this is non-trivial.
- **More block-table entries per sequence.** The attention kernel reads a longer indirection array. Launch overhead and kernel cost scale (slightly) with block count.
- **More allocator pressure.** Every ~8 decode steps triggers an `append_slot` that pops from the free registry. More Python-side bookkeeping (vLLM) or more C++ list ops (TRT-LLM).

### Larger blocks (e.g. 32 or 128)

- **Lower allocator overhead.** The free registry is touched less often.
- **Higher internal fragmentation.** Short sequences waste up to `block_size − 1` tokens. On an average 100-token response with `block_size=128`, every sequence wastes ~28 tokens of its last block = ~22% waste on that block.
- **Simpler block table.** Fewer entries, faster to scan in the attention kernel's gather.

### Choosing

The default is almost always correct. Reasons to consider changing:

- **Very short outputs** (chat completions with ~20-token average response): consider `block_size=8` to reduce internal fragmentation.
- **Very long outputs with narrow concurrency** (one-off reasoning traces): consider larger blocks to reduce allocator traffic.
- **Attention backend requirements.** Some attention kernels have a preferred or required block size (e.g. FlashAttention variants have constraints; verify against the backend docs). The kernel's requirement wins; this is a hand-off to the performance peer.

Measure before changing. A block-size change is a Pareto-sweep conversation, not a vibe.

## Copy-on-write — the subtle operational feature

Parallel sampling and beam search ask one engine to generate N outputs from one prompt. The prompt KV is identical for all N — it would waste N× HBM to copy it.

PagedAttention handles this with **reference counting on physical blocks**:

1. On admit, the parent sequence allocates its prompt blocks with ref-count = 1.
2. When N sibling sequences fork off, their block tables point to the *same* physical blocks. The ref-counts rise to N.
3. On the first decode step of a sibling, it tries to write to the trailing shared block. The write would corrupt the siblings. The allocator detects the shared block (ref-count > 1), **copies** the block's current contents to a fresh physical block, decrements the shared block's ref-count, and the sibling's block table now points to the private copy. "Copy-on-write."
4. From that point each sibling has its own trailing block; prior blocks remain shared until a similar event or until all siblings release them.

Operational consequences:

- **Memory is proportional to divergence, not to N.** For a 2000-token prompt and N=4 beams generating 100 tokens each, you pay ~2000 tokens of KV once (the shared prompt) + up to ~100 tokens × 4 (the diverging decode suffixes), not 4 × 2100.
- **The copy itself has a cost.** It is a block-sized memcpy on the GPU. On the first decode step of each sibling, there is a one-time per-shared-trailing-block cost. Usually invisible; occasionally visible in iteration-time jitter.
- **Reference counting must be audited.** Bugs where a shared block's ref-count never drops to zero are rare but real; symptom is a KV leak over time. Watch pool usage during long-running fleet uptime.
- **Beam search is less common now.** Most production serving uses greedy / temperature sampling without beams. Parallel sampling (one prompt, N samples) is still common and uses the same mechanism.

## Allocator fragmentation — the thing PagedAttention solved and the ghost that still haunts

The headline claim is "PagedAttention eliminates external fragmentation." That is accurate: because all blocks are the same size, the free pool can never fail to satisfy an `allocate(k blocks)` call as long as `free_size() ≥ k`. There is no "I have 100 free blocks but no 50-contiguous span" problem.

Internal fragmentation does remain, and the operator must understand its shape:

- Every sequence has, on average, half a block's worth of waste in its trailing block (uniform distribution of lengths within a block).
- Across N concurrent sequences, that is `N × block_size / 2` tokens of KV wasted. For `block_size=16` and `N=256` that is 2048 tokens of KV — small, but real.
- Workloads with many short sequences have higher relative fragmentation than workloads with few long sequences.
- The chapter-05 budget calculation treats this as a flat 50% block-fill fudge on the trailing block.

A separate kind of fragmentation — **temporal fragmentation** — is worth naming. If sequences arrive, allocate blocks, release them, and new sequences allocate different blocks, there is no fragmentation problem in the strict sense. But if the free registry's `pop()` policy is "give the oldest free block," the physical layout of a sequence's KV can drift across the pool over its lifetime. Impact on attention-kernel efficiency depends on the backend; most modern backends are agnostic. If the backend you run is sensitive, that is a hand-off to the performance peer.

## Block-swap-to-CPU — the preemption helper

Chapter 04 of module 103 covered swap-vs-recompute at the policy level. Here is the mechanism from the allocator's side.

### The CPU KV pool

A second block pool, allocated in host DRAM at engine start, sized by `--swap-space` (vLLM; verify equivalent for your engine). Same block layout as the GPU pool. Typical production value: 4–16 GiB per worker.

### Swap-out

On preemption, the block manager:

1. Picks the victim sequence (per the eviction policy — longest, lowest priority, oldest, whatever module 103 configured).
2. For each physical GPU block the sequence owns, pops a free CPU block, DMA-copies GPU → CPU, pushes the GPU block to the free registry, records the CPU block in the sequence's "swapped" block table.
3. Marks the sequence as swapped-out in the scheduler.

The cost is dominated by PCIe bandwidth. For a 70B model with 5 MiB per block and a sequence occupying ~200 blocks (3200 tokens of context), that is ~1 GiB to transfer — on PCIe Gen4 ~5–10 GiB/s, so ~100–200 ms. Measurably long compared to one decode iteration (~10–50 ms).

### Swap-in

The reverse. Called when the scheduler decides to resume the sequence (admission on next iteration if GPU blocks are free). Same cost signature as swap-out.

### Operational signatures

- **Steady swap-rate under capacity pressure** = scheduler is correctly using swap as its pressure-relief valve. Not an alert on its own.
- **Swap thrash** (same sequences swapping in and out repeatedly within seconds) = the scheduler is admitting sequences it cannot sustain; raise `--max-num-seqs` cap or lower per-request KV expectations. Alert.
- **Zero swap but high `--max-num-seqs` cap** = you are under-utilising the swap space. Could be a sign that the recompute path is being taken; verify.
- **Swap-in failures / stalls** = CPU pool is also full. The sequence has nowhere to go. Alert and investigate `--swap-space` sizing.

## Block-release discipline — the operational hygiene

A healthy KV pool is one where blocks are released as soon as the sequence that owns them is done with them. In practice this means:

### Release on sequence finish

The moment a sequence hits a stop condition (EOS, max tokens, user cancel, timeout), `free()` is called. Promptly. Operational test: launch a long-running probe at high concurrency, periodically kill all clients, and watch the pool-usage curve. It should fall sharply, not plateau.

### Release on cancellation

Client disconnect, deadline exceeded, request-timeout. All must trigger `free()`. The chapter in module 103 on preemption treats this; the data point to verify here is that the free actually happens on the KV side.

Common bug (historical, mostly fixed in modern vLLM / SGLang but worth checking per release): client disconnects at the HTTP layer; the engine's finish logic doesn't fire because it is waiting for a decode step to complete; the KV sits allocated for the full stop-condition window. Signature: `num_requests_running = 0`, pool usage still elevated for seconds-to-tens-of-seconds. The fix is engine-side; your job is to detect it.

### Release on OOM / exception

Any error during the forward pass that aborts a sequence must still free its KV. Modern engines handle this; verify in a chaos drill.

### Release on streaming completion

A streaming request that has emitted its last token but whose HTTP response is still being flushed to the client should release KV immediately. Do not wait for the HTTP close.

### The audit

Every N minutes in production, a healthy fleet shows pool usage proportional to `num_requests_running × avg_sequence_length / num_gpu_blocks × block_size`. Deviations are either a leak (pool rising with no new load) or a cold cache being populated (pool rising on first warm-up). The dashboard that overlays `num_requests_running` with `pool_usage_bytes` is the one that spots leaks.

## Where to find the block manager in source

### vLLM

- Block manager — in `vllm/core/` (older) or the newer unified allocator (verify path for your release). Look for a class whose constructor takes `num_gpu_blocks` and `num_cpu_blocks`.
- Block table — look for `BlockTable` / `block_tables` members on sequence / sequence-group objects.
- Allocate / free / swap entry points — grep for `allocate`, `free`, `swap_out`, `swap_in` methods on the block manager.
- Attention backends read the block table via the attention metadata; see `vllm/attention/`.

### SGLang

- Token-attention / radix cache — `python/sglang/srt/managers/` and `python/sglang/srt/mem_cache/` (verify paths). SGLang's layout is radix-tree-first; chapter 02 goes deep.
- The block / page / token-level allocator interacts with the radix cache and the scheduler.

### TensorRT-LLM (via Triton `tensorrtllm_backend`)

- The KV cache manager is C++ in-tree in the TRT-LLM repo. In the Python backend config (`config.pbtxt`) the operator-facing knobs are things like `max_tokens_in_paged_kv_cache`, `kv_cache_free_gpu_mem_fraction`.
- TRT-LLM has multiple KV cache implementations (contiguous and paged); verify which your build uses.

Walk your engine's source once with this chapter's five-call API as the map. If you cannot point at the allocator in under a minute, exercise 01 fixes that.

## Common misreadings

- **"Blocks are just a performance detail."** No — block size changes internal-fragmentation cost by a factor-of-2 and allocator overhead by a factor of (block_size_old / block_size_new). It is a tuning knob with operational consequences, not a kernel implementation detail.
- **"PagedAttention eliminates fragmentation."** External fragmentation, yes. Internal fragmentation remains, bounded by one block per sequence.
- **"Swap is free preemption."** No — swap costs real PCIe time and competes with request-serving over the PCIe bus. Each swap round-trip is 100 ms-order on today's hardware. Chapter 03 of module 103 is the policy; chapter 01 of this module is the mechanism.
- **"Copy-on-write is a beam-search thing."** It is also parallel-sampling-N > 1 (one prompt, N samples) and prefix-sharing-triggered writes to a shared block (next chapter). Still a hot path on any workload with those features.
- **"If I raise the block count I get more throughput."** Only if the pool was the binding constraint. If decode is already memory-bandwidth-bound at the current batch size, a bigger KV pool buys nothing — until you also raise `--max-num-seqs`, which chapter 02 of module 103 is the arithmetic for.

## Summary

- The KV cache is organised as a **flat pool of fixed-size blocks**; each sequence has a **block table** mapping logical token positions to physical blocks.
- The allocator's **five operations** — `allocate`, `append_slot`, `free`, `swap_out`, `swap_in` — are the surface the scheduler drives every iteration.
- **Block size** (default 16) is the primary tuning knob. Smaller → less fragmentation, more overhead. Larger → more fragmentation, less overhead. The default is almost always correct.
- **Copy-on-write** on reference-counted blocks is how parallel sampling and beam search avoid N× KV blow-up. The copy itself has a one-time per-divergence cost.
- **External fragmentation is solved by PagedAttention.** Internal fragmentation remains, bounded by one block per sequence.
- **Swap-to-CPU** is the mechanism behind the module-103 preemption policy. PCIe-bound, 100-ms-order per round-trip at today's bandwidths.
- **Block-release discipline** is the operational hygiene that distinguishes a healthy fleet from a leaky one. Audit with `pool_usage` vs. `num_requests_running` overlays.
- **The algorithm is kernel craft** — owned at the performance peer. **The runtime object is yours** — the pool as a budget, the allocator as a service with a dashboard.

Chapter 02 moves from PagedAttention's block pool to RadixAttention's prefix tree — the other half of the KV-management picture, where prefix sharing turns the pool from a per-request allocator into a fleet-shared cache.
