# Exercise 03 — Streaming ASR and TTS Walkthrough

**Time:** 5 hours · **Output:** one streaming ASR endpoint, one streaming TTS endpoint, one working voice-turn client against both, one barge-in verification report, one observability-panel set

## Goal

Make [Chapter 03](../03-streaming-speech.md) concrete by **standing up streaming ASR and streaming TTS endpoints, driving them from a client over WebSocket, verifying partial-hypothesis emission and time-to-first-audio-frame, implementing barge-in cancellation, and publishing the observability panels an on-call engineer would read during a voice incident**.

## Problem statement

Chapter 03 names the full-duplex transport, the chunking / VAD discipline, the TTFA SLO, and the barge-in state machine. This exercise stands up all four against real runtimes and proves they behave correctly under scripted adversarial conditions.

You will serve a Whisper-class ASR with streaming decode, a modern TTS model with streamed audio emission, a WebSocket control channel that wires both plus the LLM in between, and a browser-shaped client that drives the whole loop. You will then implement barge-in and stress the cancellation path.

## Prerequisites

- Chapter 03 of this module read.
- Chapter 02 of module 111 (streaming SSE / WebSocket) read — the chunk-discipline and cancellation machinery are reused here.
- Python 3.11+, `websockets` (server) or `fastapi` with WebSocket support, `aiortc` if you want WebRTC on the server side.
- A streaming ASR runtime. In 2026, reasonable choices: `faster-whisper` (CTranslate2-based Whisper, which streams via chunking), `whisper.cpp` (GGML-based, CPU + GPU), NVIDIA NeMo Parakeet / Canary (Parakeet TDT is commonly cited as a low-latency production option, Canary as higher-accuracy batch), or a hosted provider's ASR streaming endpoint. The landscape moved meaningfully across 2024–2026; re-check vendor positioning against your target latency before committing.
- A TTS runtime. Reasonable 2026 choices: Coqui XTTS (if still maintained — verify), StyleTTS 2, Fish-Speech, OuteTTS, or a hosted provider's streaming TTS endpoint.
- Access to an LLM endpoint for the middle stage — any OpenAI-compatible chat endpoint you deployed earlier in the curriculum.
- A microphone-capable client: a browser with `MediaStream` is fine; a Python script using `sounddevice` is fine; a smartphone with a WebRTC demo app is fine.

## Requirements

Six deliverables.

### Deliverable A — the streaming ASR endpoint

A WebSocket endpoint at `/v1/audio/stream` that:

- Accepts a `session.start` control frame with `{sr, format, codec}` metadata.
- Accepts a sequence of binary audio chunks (PCM 16 kHz mono as a sensible default).
- Emits typed JSON events: `asr.partial`, `asr.stable`, `asr.final`.
- Implements a VAD-driven endpoint detector that triggers `asr.final` on silence-after-speech.

Document the ASR runtime, the model, the chunk size, the hop size, and the stability-threshold for promoting partials to stable.

Verify:

- A 10-second audio recording of a single sentence produces at least 3 partial events, at least one stable event, and exactly one final event.
- Partials emerge within 500 ms of corresponding audio chunk arrival (measure: wall-clock delta between "server received chunk ending at audio time T" and "server emitted partial that contains the word at audio time T").

### Deliverable B — the streaming TTS endpoint

A WebSocket endpoint at `/v1/audio/speech/stream` that:

- Accepts a `tts.start` control frame with `{voice, format, codec}`.
- Accepts a stream of text chunks on the same WebSocket.
- Emits binary audio frames in a chosen codec (Opus at 24 kHz is a reasonable default for voice assistant use; PCM 16-bit 24 kHz if you want to skip codec setup).
- Supports `tts.cancel` to abort synthesis in flight.

Document the TTS runtime, the voice model, the chunk-to-synth threshold (do you wait for a full clause, N words, N characters?), and the vocoder choice.

Measure time-to-first-audio-frame (TTFA):

- Input: a fixed 50-word text stream pushed in at a simulated LLM token rate (e.g. 40 tokens / second).
- TTFA: wall-clock from "first text chunk arrived" to "first audio frame emitted by the server."
- Target: < 500 ms for an interactive voice assistant. Report P50 and P95 over 50 runs.

### Deliverable C — the voice-turn demo client

A client that:

- Captures microphone audio (browser `MediaStream` or Python `sounddevice`).
- Sends it to the ASR endpoint over WebSocket.
- When `asr.final` fires, forwards the transcript to an LLM `/v1/chat/completions` endpoint with streaming.
- Pipes the LLM's streamed tokens into the TTS endpoint.
- Plays the TTS audio frames through the speaker.

End-to-end acceptance: speaking a short sentence produces a spoken reply within a reasonable latency on your hardware. The point is not the reply's quality (that depends on the LLM); the point is the *plumbing works and the streaming paths are connected*.

### Deliverable D — the barge-in implementation

Implement the full-duplex state machine from chapter 03:

- Server keeps the audio-in path open while TTS is playing.
- A `voice_activity_detected` event during the `SPEAKING` state triggers `tts.cancel` plus `llm.cancel` (via the LLM backend's abort hook — module 111 chapter 02's client-disconnect pattern).
- The client stops playing the server's TTS audio on `tts.cancel`.
- The server transitions to `LISTENING` and the ASR resumes.

Verify:

- End-to-end barge-in test: start a long reply, speak over it, measure wall-clock from "user starts speaking" to "server stopped emitting TTS frames." Target: under 300 ms.
- Server-side KV release: the LLM's KV slot for the cancelled generation is freed within one decode step (poll the engine's `/metrics` kv_cache_usage to confirm).
- Audio-in is being received during TTS playback (verify in the server log that chunks arrive while in `SPEAKING` state).

### Deliverable E — the observability panel set

Author Grafana (or equivalent) panels for a voice-session dashboard:

1. **TTFA histogram** (TTS).
2. **ASR partial-to-stable latency** histogram.
3. **ASR WER proxy** — stability-change rate (how often a stable event's text is later contradicted by a later stable). Not a true WER but a cheap signal.
4. **Barge-in latency** histogram.
5. **TTS buffer-underrun events** per minute (count of client-reported gaps).
6. **Concurrent sessions** gauge.
7. **Session abandonment rate** (sessions where the user disconnected mid-reply).

Render against a short test run; capture screenshots and / or dashboard JSON.

### Deliverable F — the adversarial-condition stress

Run three adversarial conditions and document behaviour:

1. **Packet loss on the audio-in stream.** Drop 5 % of chunks (toxiproxy, iptables, or a client-side simulator). ASR partials should degrade gracefully; the stream should not drop; the server should not pile queue.
2. **Slow LLM.** Simulate a 2-second TTFT on the LLM (add a sleep or point at a slow endpoint). Verify TTFA for TTS only starts counting from "LLM first token," not from "user finished speaking" — your TTFA metric should be LLM-TTFT-independent, or if it is dependent, document that clearly.
3. **Mid-TTS network blip on the audio-out stream.** The client's WebSocket drops for 2 seconds and reconnects. The server should either (a) detect the disconnect and abort the LLM / TTS (chapter 02 of module 111) or (b) hold state and let the client resume. Pick one, document which, and verify it works.

## Starter guidance

- **Use WebSocket binary frames for audio, text frames for control.** This simplifies parsing. If you prefer multiplexed JSON with base64-audio, you can — but WebSocket binary is cheaper on the wire.
- **Separate text and audio WebSockets if the client language is awkward with mixed binary / text.** Browsers are fine; some older mobile SDKs struggle.
- **Pin the audio sample rate.** 16 kHz mono PCM into ASR, 24 kHz Opus out of TTS, is a reasonable default. Mismatches cause either silent corruption (ASR hears garbage) or loud glitches (TTS plays at the wrong pitch).
- **Enable echo cancellation on the browser.** `getUserMedia({audio: {echoCancellation: true, noiseSuppression: true}})`. Without it, barge-in tests are impossible — the mic hears the speaker.
- **Timestamp everything on the server.** TTFA, partial-to-stable latency, and barge-in latency all require timestamps at precise points. Use `time.monotonic()` and log in ms.
- **Keep VAD state close to the audio demux.** Running VAD on the raw bytes before the ASR forward is a tiny CPU cost and keeps endpoint detection responsive even when the ASR runtime is slow.

## Fallbacks for constrained environments

- **No GPU.** `whisper.cpp` on CPU runs small Whisper models fine for this exercise; latencies will be higher than production but the plumbing is identical.
- **No TTS available.** For deliverable B, use a stub TTS that emits a sine wave at 440 Hz for each input character; it isn't speech but it preserves the streaming semantics for the exercise.
- **No real microphone client.** Pre-record an audio file, stream it through the client in real-time (one chunk per 100 ms) to simulate a live mic.
- **No barge-in client.** For deliverable D, script the test: start playback, after 2 seconds inject a synthetic VAD trigger from the client, measure server-side cancel.
- **No full Grafana stack.** The observability panels can be plotted from the server's structured logs in a Jupyter notebook — the deliverable is the measurements and the panel definitions, not the specific dashboard tool.

## Acceptance criteria

1. **Deliverable A** ASR endpoint produces partials, stables, and finals; latencies in the chapter-03 envelope.
2. **Deliverable B** TTS endpoint emits streamed audio; TTFA P95 < 1 s on your hardware.
3. **Deliverable C** voice-turn demo works end-to-end: speak, get a spoken reply, under a reasonable total latency.
4. **Deliverable D** barge-in cancellation fires within 300 ms; LLM KV is freed; audio-in flows during SPEAKING.
5. **Deliverable E** six-to-seven panels rendered; dashboard reflects behaviour under a live session.
6. **Deliverable F** three adversarial conditions exercised; documented behaviour (not necessarily all "clean" — a known degradation with a mitigation is acceptable).

## Common failure modes

- **Server-side VAD lagging the audio.** If VAD runs on a delayed thread, endpoint detection fires late, the user feels the turn "hang." Run VAD in the audio-in callback, not via a batch pipeline.
- **TTFA inflated by framework buffering.** Module 111 chapter 02's SSE flush discipline applies to WebSocket too; check for proxy buffering, `fastapi`'s default middlewares, and any ASGI server's write batching.
- **Barge-in cancels TTS but not LLM.** The TTS stops; the LLM keeps generating; token cost continues; KV slot held. Wire both cancels; verify the LLM's abort hook actually fires.
- **ASR partials in the wrong language.** Whisper's language detection on short chunks is lossy. Force the language in the ASR request (`language="en"`) when you know it; your partials will stop flipping between en and nl.
- **Audio codec negotiation silently falls back.** The client requests Opus 24 kHz; the server supports only Opus 48 kHz; the result is garbled audio. Pin codec and sample rate on both ends.

## Keep it

The ASR and TTS endpoints, the voice-turn demo client, the barge-in implementation, the observability panel set, and the adversarial-condition stress report are the fleet's voice-serving artefact. Exercise 04 (reasoning-model serving) borrows the cancellation plumbing for operator-abort flows; exercise 05 (agentic workload) borrows the long-session state model.
