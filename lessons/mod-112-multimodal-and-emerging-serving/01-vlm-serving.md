# Chapter 01 — VLM Serving: Image Preprocessing, Visual-Token KV, Per-Image Budget, Variable-Image-Count Batching

> A VLM request looks like a chat request with pictures. The arithmetic underneath is different: a single 1024×1024 image can expand to several thousand *input* tokens before a single character of the user's text is tokenized, and those tokens have to be cached, batched, and billed the same way text tokens are. This chapter is where the engineer who already understands text serving learns the four new seams.

## Motivation

By 2026 the "chat with pictures" shape is a default product requirement. The product team ships a feature — upload a screenshot, upload a PDF page, drop a photo of a receipt — and the inference fleet has to serve the resulting request without regressing TTFT on text-only traffic sharing the same replica. Four things that were simple in text serving become seams:

- **The request's input is now bimodal and large.** A text request's input is a token stream in the hundreds-to-thousands. A VLM request's input is text *plus* images, and each image expands to a number of visual tokens driven by the vision encoder (patch grid × channels after projection). An iPhone-quality image at a high-detail setting can be worth 1500+ visual tokens on current open-weights VLMs — multiplied by the number of images the user attached. The prefill time, the KV footprint, and the admission decision all shift.
- **A preprocessing stage appears in front of the engine.** Text requests arrive tokenized by the engine itself. VLM requests arrive with image URLs or base64 blobs that must be fetched, decoded, normalised, resized, and patchified *before* the engine sees them. Where that work runs (ingress process, dedicated preprocess worker, inside the engine) changes the TTFT budget and the fault isolation.
- **The prefix cache's unit changes.** Text prefix caches hash on prompt-prefix token IDs; a VLM prefix cache has to include a hash of each image's post-encoded visual tokens (or an identifier the client provides) if a chat with multiple turns re-sends the same image. Getting this wrong turns a `cache_hit = 99 %` claim into a `cache_hit = 0 %` reality on image-heavy traffic.
- **The batcher sees requests with wildly variable input lengths.** Text requests cluster around a dominant length distribution. VLM requests do not — a text-only turn is short, a one-image turn is medium, a twelve-image turn is enormous. The scheduler's padding, admission, and preempt policy must survive that distribution.

By the end of this chapter you can:

- Draw the VLM request flow from HTTPS ingress through image-preprocess to engine prefill to decode, with the extra stages named.
- Compute the visual-token count for a given model + image + detail setting from the model's documented encoder parameters.
- Decide where image-preprocess runs (ingress vs. engine) based on your fleet's fault-isolation and throughput goals.
- Configure the visual-token prefix cache so that *re-sending the same image in a later turn is a cache hit*.
- Set the per-image token budget and the per-request image-count budget that bound a worst-case request.
- Choose a batching strategy that stays stable under mixed 0 / 1 / 3 / 12-image traffic.

## The VLM request flow

A VLM chat request lands at the ingress with the OpenAI-compatible multimodal content-part shape from module 111 chapter 01:

```json
{
  "model": "qwen2.5-vl-7b-instruct",
  "messages": [
    {"role": "user", "content": [
      {"type": "text", "text": "What's in this screenshot?"},
      {"type": "image_url", "image_url": {"url": "https://cdn.example.com/screens/42.png", "detail": "high"}}
    ]}
  ],
  "stream": true,
  "max_tokens": 512
}
```

The stages the engine does *not* do, that the serving stack must:

1. **Fetch.** If the content part is an `image_url` with an HTTPS URL, fetch the bytes. If it is a `data:` URL (base64 inline), decode. Enforce a per-image size cap (e.g. 10 MB) and a per-request total-bytes cap.
2. **Decode.** Decode the image bytes to a pixel tensor (PNG, JPEG, WEBP, GIF, HEIC — the set depends on your Pillow / libvips / `image` crate build).
3. **Normalise and resize.** Convert to the model's expected colour space, resize to the encoder's expected input size (or the detail-driven tiled layout — some VLMs split a high-detail image into multiple tiles plus a thumbnail).
4. **Patchify and encode.** Run the vision encoder (ViT or similar) to produce visual tokens — a sequence of embedding vectors of length `N` that the LLM backbone will see interleaved with text tokens.
5. **Project and inject.** Project the visual embeddings into the LLM's embedding dimension (a trained MLP or Q-Former) and splice them into the input sequence at the position where the `<image>` placeholder sits.
6. **Prefill and decode as usual.** From here on it is a text-serving request with a longer input sequence.

Stages 1–3 are classic image-processing work. Stages 4–5 are forward passes on the vision tower. Stage 6 is the engine you already know. The seam this chapter cares about is *who runs 1–5 and how 1–5 feed stage 6's KV cache*.

## Visual-token expansion: the number that dominates everything else

For a given VLM, each image contributes a number of tokens to the context window. Call it `N_visual`. For an open-weights VLM, `N_visual` is a function of the vision encoder's patch grid, the detail setting, and whether the image was tiled.

A worked example — the shape most 2026 open-weights VLMs follow (verify numbers against your model's config before billing on them):

- Encoder patch size: 14×14 pixels.
- Encoder native input size: 448×448 pixels → `(448 / 14) × (448 / 14)` = `32 × 32` = 1024 patches per tile.
- Detail: `low` → single 448×448 downsample; `high` → the image is tiled into K tiles of 448×448 plus a 448×448 thumbnail, so `(K + 1) × 1024` patches.
- Some encoders collapse adjacent 2×2 patch groups (`pixel_shuffle`) before projection — divides the count by 4.

<!-- needs-research: the exact tiling layout and `pixel_shuffle` behaviour varies per model family (LLaVA-OneVision, Qwen2-VL / Qwen2.5-VL, Pixtral, InternVL, Molmo). Confirm `N_visual` against the model's `preprocessor_config.json` and the encoder's `image_processor` logic before quoting a number on a billing dashboard. -->

The lesson: **a VLM request has an input length you cannot bound at the API edge from `len(messages[].content.text)` alone**. The ingress has to run image-preprocess (or at least an image-metadata parse) to even know whether the request will fit in the model's context window. The admission-decision chapter (module 103) that assumed `prefill_tokens = sum(tokenize(message.content))` has a new term.

## Where image-preprocess runs

Three deployment shapes for stages 1–5 of the flow above:

| Deployment | Fetch / decode / resize | Vision encoder forward | Pro | Con |
|---|---|---|---|---|
| **In-engine** | engine process | engine process | one process to deploy; the engine's prefix cache has native visibility | engine CPU time competes with scheduler loop; a slow image fetch stalls the engine's main loop |
| **Preprocess sidecar** | sidecar process on same host | engine process | isolates fetch / decode from the engine's main loop | one more process; shared-memory handoff for the pixel tensor to avoid a serialisation cost |
| **Preprocess service** | dedicated service (CPU pool) | engine process | horizontal scaling of fetch / decode independent of GPU replicas | network hop in the TTFT budget; backpressure plumbing between services |

A fourth variant — running the vision encoder on CPU in a sidecar and only shipping projected embeddings to the engine — exists on some multi-modal engines but imposes a serialization and transport cost that usually makes it worse than running the full encoder on the GPU. Avoid unless the vision tower is genuinely small and the GPU is scarce.

Default for a greenfield fleet: **preprocess sidecar on the same host**. Fetch / decode / resize is CPU-bound and I/O-bound work that does not belong on the GPU replica's main thread; the vision encoder forward belongs on the GPU alongside the LLM. The sidecar is a thin CPU process that answers "give me the pixel tensor ready for the vision encoder" and returns bytes over local shared memory or a Unix domain socket.

The engines that bundle this for you — vLLM's multimodal support, SGLang's VLM support, TRT-LLM's multimodal pathways — do so with the in-engine deployment above. That is fine for small / medium traffic; a high-image-rate fleet usually ends up with the sidecar.

## The visual-token KV cache

The LLM backbone's KV cache is where visual tokens live after the vision tower has turned them into embeddings. Three questions decide whether it works correctly on VLM traffic:

### Does the engine's prefix cache include visual tokens?

A text prefix cache hashes on the input token sequence (module 104 chapter 03). A VLM extends the token sequence with the projected visual embeddings. If the engine's prefix-cache key is only the text tokens, two conversations with the same screenshot upload different images get the *same* prefix-cache hit — a correctness bug. If the engine's prefix-cache key hashes the entire concatenated embedding sequence (text tokens + visual embeddings as float tensors), the hash is expensive to compute and the cache hit rate on image-reuse drops to near-zero because floating-point embeddings are rarely bitwise-identical across runs.

The correct approach: **cache-key visual tokens by the content-addressable hash of the raw image bytes (or a stable identifier the client provides)**, not by the floating-point embedding. The engine then treats "image hash H at position P with M visual tokens" as a prefix segment, same as the text tokens.

Example cache key for the engine (illustrative — the exact representation varies per engine):

```
prefix_key = [
    text_token(1234), text_token(56),           # system / user prefix text
    image_segment(hash="sha256:abcd...", N=1024),  # 1024-visual-token segment
    text_token(789),  text_token(42)            # the trailing "What's in this?" text
]
```

The engine's prefix-cache lookup matches this prefix against a previous request's key; if the previous request also had the same image hash at the same position with the same text prefix, the KV blocks for the vision-encoder-produced tokens and the following text are reusable.

### Where does the image hash come from?

Three options:

- **Content hash of the raw image bytes.** Robust, slow to compute (SHA-256 on a 5 MB JPEG).
- **Content hash of the resized / normalised pixel tensor.** Stabler across trivial encoding changes; still a tensor-hash cost.
- **Client-provided identifier.** Fastest; requires client cooperation (an `image_url` field with a stable `id` or `cache_key` extension); fragile if clients don't send it correctly.

Default: content hash of raw bytes, computed in the preprocess sidecar in parallel with the fetch / decode / resize work. The hash is in the TTFT budget but overlaps with other preprocess work.

### Does the engine support multi-image prefix caching?

A chat with three images in message `i` and the same three images reused in message `i+1` is a common pattern (think an iterative design review where the user attaches the same mockups over several turns). The engine must cache each image's visual-token segment separately, keyed by the image hash, so that a later turn that reorders or adds images can still reuse the ones that match.

Engine-level support here is uneven as of 2026; several open-weights VLM serving stacks treat the first image's visual tokens as "part of the prompt" and lose the cache on any image-list change. <!-- needs-research: confirm current multi-image prefix-cache support in vLLM / SGLang / TRT-LLM for the VLM families they list; cite the specific flag or config that enables it, and the fallback behaviour when it is off. -->

## The per-image token budget

A single high-detail image on a current open-weights VLM can be worth 1500+ visual tokens. A 32k-context model can therefore fit roughly a dozen high-detail images *before any text* in the request. A user who uploads 32 photos of their vacation is a context-overflow bug waiting to happen.

The ingress must enforce two budgets:

- **Per-image token budget** — e.g. `max_visual_tokens_per_image = 1600`. Images that would exceed it get down-sampled (detail → `low`, or an explicit resize to a smaller tile count). Reject with a 413 if even the lowest-detail path exceeds the budget.
- **Per-request image-count budget** — e.g. `max_images_per_request = 16`. Reject with a 400 above the budget.

Both budgets are *admission-time* checks, before the request consumes a KV slot. The ingress can compute them from the image metadata without running the full encoder forward:

```python
def estimate_visual_tokens(image_meta, detail, encoder_cfg):
    # image_meta: {width, height, format}
    # detail:     "low" | "high" | "auto"
    tile_size = encoder_cfg.tile_size      # e.g. 448
    patch_size = encoder_cfg.patch_size    # e.g. 14
    patches_per_tile = (tile_size // patch_size) ** 2  # e.g. 1024
    if detail == "low":
        tiles = 1
    else:
        # Model-specific tiling rule; see each VLM's processor config.
        tiles_w = ceil(image_meta.width  / tile_size)
        tiles_h = ceil(image_meta.height / tile_size)
        tiles   = tiles_w * tiles_h + 1    # +1 for thumbnail
    tokens = tiles * patches_per_tile
    if encoder_cfg.pixel_shuffle_ratio:
        tokens //= encoder_cfg.pixel_shuffle_ratio ** 2
    return tokens
```

<!-- needs-research: the tiling rule, thumbnail inclusion, and `pixel_shuffle` ratio differ per VLM family (LLaVA-OneVision, Qwen2.5-VL, Pixtral, InternVL, Molmo). Replace this stub with the specific model's documented formula before pricing a request. -->

The number returned feeds the admission control from module 103: a 12-image request with 1600 visual tokens per image is a 19200-visual-token prefill request before text, and the scheduler should treat it accordingly.

## Batching across variable image counts

The scheduler assumption that most text serving engines were designed under — requests have similar prefill-token counts, so continuous batching across them is efficient — bends under VLM traffic. A fleet that serves a mix of:

- 60 % text-only turns (≈ 300 input tokens)
- 25 % one-image turns (≈ 1600 visual tokens + ≈ 300 text tokens ≈ 1900)
- 10 % three-image turns (≈ 5100 tokens)
- 5 % twelve-image turns (≈ 19500 tokens)

sees a 65× prefill-length spread. Three strategies to keep the batcher stable:

- **Chunked prefill (SARATHI-Serve style).** Chop any single prefill into fixed-size chunks (e.g. 2048 tokens) and interleave with decode. The 12-image request now takes ~10 chunks instead of one giant forward pass; the batch stays close to balanced. This is the default for modern engines and should be enabled on VLM fleets.
- **Admission-side classification.** Classify incoming requests into prefill-length bands (short / medium / long / huge) and admit them against different KV budgets. The 12-image request can't preempt a long-running text request; the text request can't be starved by a stream of huge VLM requests.
- **Per-replica specialisation.** Route VLM requests to replicas tuned for them (larger `max-num-batched-tokens`, possibly a smaller `max-num-seqs`), and route text-only traffic to replicas tuned the other way. This costs more fleet complexity; chapter 06 of module 106 covered the pattern.

A sign the batcher is unhealthy: TTFT P99 for text-only traffic jumps when a few 12-image requests arrive. The scheduler is spending all its decode-step budget doing prefill for the huge requests, starving the small ones.

## Three failure modes

### "A chat with a repeated image is slow on the second turn"

Symptom: TTFT on turn 2 of a conversation is as high as TTFT on turn 1 even though the image hasn't changed.

Common cause: the engine's prefix cache is text-token-only and doesn't hash on image identity, so the visual-token segment is re-encoded every turn. The vision encoder forward is doing 100–500 ms of work it didn't need to do.

Fix: enable the engine's multi-image prefix-cache mode, pass the image hash through, or (if the engine doesn't support it) cache the projected visual embeddings in a sidecar keyed on image hash and short-circuit the vision encoder.

### "An image-heavy user crashes the replica"

Symptom: a single user who uploads a batch of large screenshots causes an OOM or a context-overflow abort on the replica.

Common cause: no per-image or per-request budget; the ingress accepted a 50-image request that computes to 80k visual tokens in a 32k-context model.

Fix: enforce `max_visual_tokens_per_image` and `max_images_per_request` at the ingress, reject with 400 / 413 before the request reaches the engine, and surface the budget in the API error so the client knows which knob to turn.

### "Text-only latency regressed when we enabled image uploads"

Symptom: TTFT P99 on the text-only workload class went from 400 ms to 1.2 s after the image-upload feature shipped.

Common cause: the batcher is admitting huge-prefill VLM requests on the same replica as text-only traffic without chunked prefill, so a single 12-image request stalls the decode loop for every batchmate.

Fix: enable chunked prefill, lower `max-num-batched-tokens` or tighten admission bands, or isolate VLM traffic to a dedicated replica class.

## Summary

- A VLM request is a text request with an image-preprocess stage, a vision-encoder forward, and an input length driven by `images × visual_tokens_per_image` rather than text alone.
- Image preprocessing (fetch, decode, resize, patchify) runs best in a sidecar on the GPU host; the vision encoder forward runs on the GPU with the LLM.
- The engine's prefix cache must key on image identity, not on floating-point visual embeddings, if a chat that re-sends the same image is to hit the cache.
- Per-image and per-request image-count budgets are admission-time checks that must fire before the engine sees the request.
- The scheduler must survive a 60×+ spread in prefill lengths; chunked prefill plus admission-side classification is the default pattern.
- The three failure modes — stale-prefix-cache, OOM-on-image-flood, text-latency-regression — all have fixes at the admission and prefix-cache layers; verify against the specific VLM family and engine you deploy.
