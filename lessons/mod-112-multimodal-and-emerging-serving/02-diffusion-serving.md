# Chapter 02 — Diffusion Serving: Steps, Schedulers, Denoising Batching, VAE Cost

> A text request's latency scales with output tokens. A diffusion request's latency scales with *denoising steps times step time*, and the single most powerful latency knob in the entire stack is the scheduler the operator picks. This chapter is where an inference engineer trained on text learns the diffusion vocabulary well enough to ship a serving replica and price a step.

## Motivation

Diffusion models — image, video, 3D — have a serving shape that looks nothing like a token-streaming LLM. There are no output tokens; there is a fixed-size noise tensor that gets iteratively denoised. The forward pass per step is a dense transformer or UNet / DiT on a fixed-shape latent; the number of forward passes per request is a parameter the operator sets at request time. A text engineer's intuition "longer output = longer wait" is wrong for diffusion; the right intuition is "more steps = more wait, and a better scheduler buys you fewer steps at the same quality."

Four things that were simple in text serving become seams here:

- **The latency knob is not `max_tokens`; it is the scheduler and the step count.** A 50-step DDIM at 1024×1024 can take seconds on a top-end GPU. The same model at 4 steps with an LCM (Latent Consistency Model) scheduler can take under half a second. The quality trade-off is non-obvious and model-specific; the inference engineer has to let the operator choose and must publish the per-choice latency / quality curve.
- **Batching groups along a different axis.** Continuous text batching batches *sequences* along token position. Diffusion batching groups *prompts* along a batch axis and runs the UNet / DiT step on the batched latent. All prompts in a batch must run the same number of steps to the same scheduler on the same model — batching logic that looks like legacy static batching, not continuous batching.
- **Every request has a VAE encode / decode bill.** The denoising loop runs in latent space; the input image (for img2img / inpainting / controlnet) must be VAE-encoded, and the output latent must be VAE-decoded to pixels. That is extra forward-pass work that is *not* in the step count and must be accounted for.
- **The output artefact is big and stays big.** A text response is a few KB. A 1024×1024 PNG is on the order of 1–3 MB; a 5-second 1024×576 video can be 50+ MB. Response-path storage, egress, and client transport become serving concerns, not application concerns.

By the end of this chapter you can:

- Draw the diffusion request flow from ingress through VAE-encode → denoising-loop → VAE-decode to response, with the step count and scheduler named at each stage.
- Choose a scheduler (DDIM, DPM-Solver, Euler-ancestral, LCM / consistency-distilled) against a stated latency-vs-quality trade-off.
- Configure denoising-batch grouping to amortise the UNet / DiT step across concurrent prompts.
- Account for VAE encode / decode cost in the per-request budget and the per-replica throughput plan.
- Pick the response-path pattern (blob-storage + signed URL vs. direct stream vs. progressive) for the output artefact.

## The diffusion request flow

A text-to-image diffusion request lands at the ingress with the OpenAI-compatible images endpoint shape (module 111 chapter 01):

```json
{
  "model": "sdxl-base-1.0",
  "prompt": "A quiet library at dawn, oil painting style",
  "negative_prompt": "blurry, low quality",
  "n": 1,
  "size": "1024x1024",
  "response_format": "url",
  "quality": "standard",
  "extra": {
    "scheduler": "dpm_solver_multistep",
    "steps": 25,
    "guidance_scale": 7.5,
    "seed": 42
  }
}
```

`extra` here is an engine-extension bag — the OpenAI canonical request doesn't expose scheduler or step count directly, and serving stacks vary widely on how they surface these parameters. `response_format: "url"` means the server stores the output image to blob storage and returns a signed URL; `"b64_json"` means it inlines the base64 payload in the response body.

The stages:

1. **Validate and classify.** Prompt / negative-prompt lengths within limits; image size in the supported set; scheduler and step count on the allow-list; LoRA / ControlNet attachments resolved to stored weights (module 106 chapter 02's adapter-serving work is relevant).
2. **Encode condition inputs.** Run the text encoder (CLIP / T5 / Gemma-based depending on model) on the prompt to produce the conditioning tensor. For img2img / inpainting / controlnet, also VAE-encode the input image to its latent representation.
3. **Initialise the latent.** Sample a Gaussian noise tensor of shape `(n, latent_channels, H / 8, W / 8)` seeded by `seed`.
4. **Run the denoising loop.** For each of `steps` iterations: predict the noise (one UNet / DiT forward pass), update the latent per the scheduler's integration step. Classifier-free guidance doubles the forward-pass count when `guidance_scale > 1.0` (conditional and unconditional paths).
5. **VAE-decode.** Convert the final latent tensor back to pixel space — one large forward pass through the VAE decoder.
6. **Post-process and store.** Save the pixel tensor as PNG / JPEG / WEBP, upload to object storage, mint a signed URL, return the response.

The dominant cost line is stage 4, and inside stage 4 the dominant lever is the step count × scheduler. Everything else in the pipeline is overhead on that lever.

## Scheduler choice: the latency / quality lever

A scheduler is the integration method the denoising loop uses to turn "the model's predicted noise at step `t`" into "the latent at step `t-1`." Different schedulers admit different step-count budgets at comparable quality. The four families an inference engineer should recognise:

- **DDIM (Denoising Diffusion Implicit Models).** The canonical reference. Deterministic, well-studied, needs around 50 steps on SD-class models for the quality the model was trained for. <!-- needs-research: Song, Meng, Ermon, "Denoising Diffusion Implicit Models," ICLR 2021 — arXiv:2010.02502. Verify current best-practice step counts per model family. -->
- **DPM-Solver / DPM-Solver++ / DPM-Solver-Multistep.** Higher-order ODE solvers; typically cut the step count to 20–25 at matching quality. Default in a lot of 2024–2026 deployments. <!-- needs-research: Lu et al., "DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps," NeurIPS 2022 — arXiv:2206.00927; and the follow-up "DPM-Solver++" — arXiv:2211.01095. -->
- **Euler / Euler-ancestral / Heun.** Simple first- or second-order solvers; competitive at 20–30 steps; "ancestral" variants add noise per step and break reproducibility (same seed → different image). Popular with users who want sampling diversity.
- **LCM / Consistency-distilled schedulers.** The model has been separately *distilled* into a few-step student that can produce a reasonable image in 2–8 steps. LCM-LoRA is a lightweight adapter that retrofits a Stable-Diffusion-class base. Latency drops by an order of magnitude at some cost to peak image quality. <!-- needs-research: Song et al., "Consistency Models," ICML 2023 — arXiv:2303.01469; Luo et al., "LCM-LoRA: A Universal Stable-Diffusion Acceleration Module," arXiv:2311.05556. Confirm current state-of-the-art few-step distillations and their quality-vs-latency placement. -->

The engineering decision is a Pareto point on `(steps, scheduler) → (latency, quality)`. Chapter 05 of module 110 (throughput-vs-latency Pareto) is the pattern; the axes here are different but the method is the same:

```
Scheduler              | Typical good step count | Rel. latency | Quality floor
-----------------------|-------------------------|--------------|-----------------
DDIM                   | 50                      | 1.00         | reference
DPM-Solver++ multistep | 25                      | 0.50         | ~= reference
Euler                  | 30                      | 0.60         | ~= reference
Euler-ancestral        | 30                      | 0.60         | non-deterministic
LCM (distilled)        | 4                       | 0.08         | below reference, acceptable for product
```

(Numbers above are illustrative; measure against your deployed model on your hardware. The acceptance criterion in exercise 02 is the Pareto report you produce from a real sweep.)

A serving stack should expose scheduler and step count as per-request parameters (with an allow-list and a default) rather than hard-code one choice. Product teams will have different quality tolerances for a thumbnail generator (short step count, LCM distilled) and a final-render tool (high step count, DPM-Solver).

## Classifier-free guidance doubles the step cost

Most text-to-image diffusion models were trained with classifier-free guidance: at inference time, the UNet is run *twice* per step — once with the text conditioning (conditional path) and once with an unconditional / empty prompt — and the two predictions are linearly combined by `guidance_scale`. That doubles the GPU work per step compared to the no-guidance path.

Some newer model variants (SD3, FLUX.1, latent-consistency distillations) adjust or skip this; some inference-time optimisations (perturbed attention guidance, dynamic guidance) change it per step. The operational implication: when you benchmark step time, measure it on the *actual guidance path* in use, not the raw UNet forward. The two differ by 2×.

## Denoising batching: group prompts, not sequences

Text continuous batching slots short requests into the free attention slots of longer-running requests (module 103 chapter 02). Diffusion batching does not work that way. The denoising loop for a batch of `B` prompts does `B` UNet forward passes in parallel on a `(B, C, H, W)` latent tensor; all `B` prompts must:

- Use the same model weights.
- Use the same scheduler and the same step count.
- Share the same resolution (or be padded to one).
- Be ready to start the next step simultaneously.

That means the batcher is closer to classic ML-serving dynamic batching: wait up to a tight time window, group requests with compatible parameters, dispatch a batch, run the entire denoising loop, split outputs back to per-request response streams.

A practical dispatch loop:

```python
# pseudo-code — illustrative, not an engine's actual scheduler
def dispatch_loop():
    pending = []
    deadline = None
    while True:
        req = queue.get(timeout=remaining(deadline))
        if req is None:
            # Deadline hit; dispatch what we have.
            dispatch(pending)
            pending, deadline = [], None
        else:
            if compatible(req, pending):
                pending.append(req)
                if deadline is None:
                    deadline = now() + MAX_BATCH_WAIT  # e.g. 50 ms
                if len(pending) == MAX_BATCH_SIZE:
                    dispatch(pending)
                    pending, deadline = [], None
            else:
                # Incompatible (different scheduler, step count, size); flush and start new.
                dispatch(pending)
                pending, deadline = [req], now() + MAX_BATCH_WAIT
```

The tuning knobs:

- **`MAX_BATCH_WAIT`** — how long to wait for a batchmate before dispatching. Tight (10–50 ms) keeps TTFT low under light load; loose (200+ ms) raises throughput under heavy load at the cost of per-request latency.
- **`MAX_BATCH_SIZE`** — the GPU's VRAM ceiling on a `(B, C, H, W)` forward. SDXL at 1024×1024 can typically batch 4–8 on an 80 GB GPU; a few-step LCM can batch more because the whole loop fits in cache.
- **Compatibility class** — group by `(model, scheduler, step_count, size)`. A dispatch with mixed sizes needs padding or a per-request mask that most engines don't support.

The implication for admission control: if your fleet serves two models of different sizes with different default step counts, one model's load can starve the other's batching window. Isolate per model where possible.

## The VAE encode / decode cost line

The VAE is the autoencoder that moves between pixel space and the model's latent space. It appears in:

- **img2img and inpainting:** input image → VAE encoder → latent → denoising loop.
- **Every request:** final latent → VAE decoder → output image.
- **ControlNet / IP-Adapter:** condition image → preprocess → optionally VAE encode → condition features.

The VAE decoder forward pass for a 1024×1024 image is on the order of 100–300 ms on current GPUs — a meaningful fraction of a short-step diffusion request. Two optimisations worth knowing:

- **Tiled VAE.** The decoder is run in overlapping tiles rather than on the whole image at once. Trades a small amount of boundary-stitching artefact for a much lower peak VRAM; essential for very high resolutions.
- **TAESD / tiny VAE.** A much smaller distilled decoder used for previews. Useful if the client wants a 128×128 preview while the main decode completes, or for intermediate step visualisations.

When you bench a diffusion endpoint, split latency into `(encode) + (steps × step_time) + (decode)` so that you can see which stage dominates. On a 4-step LCM run, the VAE decode may be a larger slice than all the denoising steps combined.

## Video and 3D diffusion: everything above plus more axes

Video diffusion (think AnimateDiff-style temporal U-Nets, Stable Video Diffusion, Wan / Hunyuan video models in 2024–2026) extends the latent to a time axis: `(B, C, T, H, W)`. The practical seams:

- **VRAM scales with T.** A 25-frame 1024×576 video with 50 steps is 25× the per-step memory of a single image. Peak VRAM, not step count, is usually the first constraint.
- **Temporal attention is expensive.** Many video UNets add a cross-frame attention layer that costs quadratic in T. Doubling T more than doubles the step time.
- **The output artefact is large.** A 5-second 1080p video is tens to hundreds of MB. Always encode to a compressed video codec (H.264, H.265, AV1) on the server before egressing; do not ship raw frames.
- **Scheduler and step count defaults differ.** A model trained on an LCM-style distillation may converge in 4 video steps; a classical video-diffusion model may want 50.

3D diffusion (text-to-3D via Score Distillation Sampling, Gaussian Splatting diffusion, mesh diffusion) is a research-grade serving shape in 2026 — the outputs are structured artefacts (point clouds, Gaussian splats, meshes) and the pipelines mix diffusion with optimisation loops that can take minutes per request. If you serve one of these, expect:

- Request-level timeouts in minutes, not seconds.
- Asynchronous response patterns (job submission → status polling / webhook → result retrieval) rather than request-response.
- A per-request cost that is a meaningful fraction of a GPU-hour and should be billed per completion rather than per request.

<!-- needs-research: the video and 3D diffusion state of the art evolves fast. Cite the specific models and pipelines (Stable Video Diffusion, Wan, Hunyuan video, DreamFusion / SDS, Instant3D, Gaussian Splatting diffusion) in the module's `resources.md` and verify current scheduler / step defaults before quoting numbers. -->

## The response path: blob storage vs. inline

Three response-path patterns:

- **Signed URL to blob storage.** The server stores the image / video to S3 / GCS / Azure Blob and returns a signed URL that expires. Preferred for larger-than-a-few-KB outputs; separates serving-replica network from egress traffic. Adds a storage dependency and an object-lifecycle cost.
- **Inline base64.** The server returns the output as a base64 string in the JSON body. Fine for small (<100 KB) outputs on a trusted-client workflow; blows up the response size by 33 %.
- **Progressive / streamed.** For video or long diffusion runs, stream intermediate previews as the loop progresses (every K steps, emit the current VAE-decode or a TAESD preview). Keeps the user engaged; costs extra VAE decodes per request and SSE / WebSocket framing on the server.

A general-purpose image endpoint should default to signed URL plus an `expires_in` on the response. An inpainting / editor endpoint used inside a design tool may want inline base64 for fewer network round-trips. A video endpoint should always use signed URL or a streaming protocol.

## Three failure modes

### "The endpoint is fast when I test it, slow in production"

Common cause: tests ran at `batch_size = 1` and `guidance_scale = 1.0` (no classifier-free guidance). Production traffic runs at `guidance_scale = 7.5` (doubling step time) and the batcher dispatches at `batch_size = 1` because the request mix doesn't line up on scheduler and size. Per-request latency is 2× what the bench said.

Fix: bench with the actual guidance-scale default and with realistic batching. Expose the compatibility-class mix in observability so you can see whether the batcher is finding mates.

### "Different users get wildly different quality"

Common cause: the API exposes `scheduler` and `steps` without an allow-list, and users or an LLM-generated client are picking low step counts or non-deterministic schedulers thinking "fewer is better."

Fix: publish the allow-list and the default; the per-model Pareto report from exercise 02 is the artefact that justifies the defaults. If the product needs a "fast" and a "high quality" tier, surface them as named presets rather than raw scheduler / step knobs.

### "VRAM OOM on the first high-res request after deploy"

Common cause: the VAE decoder's peak VRAM for a 1024×1024 image is larger than the headroom you reserved; the first request that hits the big decoder path hits an OOM.

Fix: enable tiled VAE, pre-warm the decoder at startup with a dummy forward, and set the replica's VRAM budget to account for peak decoder memory, not steady-state denoising memory.

## Summary

- A diffusion request's latency is dominated by `steps × step_time`; the scheduler sets the step-count budget at a given quality.
- Scheduler families on the Pareto curve: DDIM (reference, 50 steps), DPM-Solver (25 steps), Euler (30 steps), LCM / consistency-distilled (2–8 steps, lower quality floor).
- Classifier-free guidance doubles step time; measure on the real guidance path, not the raw UNet forward.
- Diffusion batching groups prompts on a `(B, C, H, W)` axis with compatible `(model, scheduler, steps, size)`; batch wait and batch size are the primary tuning knobs.
- VAE encode / decode are a separate cost line and can dominate a few-step LCM run; split the bench by stage.
- Video and 3D extend the shape (time axis, large artefacts, multi-minute latencies) and usually need asynchronous response patterns.
- The response path should default to signed URL in blob storage for anything above ~100 KB; inline base64 and progressive streams are special-case choices.
