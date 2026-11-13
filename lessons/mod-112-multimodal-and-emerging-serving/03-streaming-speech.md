# Chapter 03 — Streaming Speech Serving: ASR, TTS, Chunking, Barge-In

> SSE streams tokens one way, from server to client. Speech streams bytes *both* ways over seconds — the user talks, the server transcribes while they talk; the model speaks back, the server synthesises while they listen; the user interrupts, the server stops speaking and starts listening. The HTTP shape from module 111 does not express this; this chapter does.

## Motivation

Speech is where the chat-serving HTTP discipline from module 111 has to bend. Three things change at once:

- **The transport is full-duplex, not request-response.** A voice interaction has both an audio-in stream (the user) and an audio-out stream (the model), running concurrently, over the same session. SSE is one-way; plain HTTP streaming responses are one-way. The transport is WebSocket (RFC 6455) or WebRTC — and the server has to manage both directions simultaneously.
- **The time budget is real-time, not interactive.** A text chat client is happy with 200 ms TTFT because the user is reading. A voice client is unhappy with 400 ms time-to-first-audio-frame because the user is listening to silence. The SLO vocabulary shifts: first-audio-frame latency, audio-render drift, end-of-utterance detection latency, barge-in cancellation latency.
- **The workload has three concurrent model paths, not one.** A voice turn runs ASR (speech → text), an LLM (text in, text out), and TTS (text → speech). Each is a model with its own replica class, its own batching, its own SLO. The serving platform orchestrates all three without letting one's latency spill into the others' budget.

By the end of this chapter you can:

- Draw a voice-turn pipeline with ASR, LLM, and TTS as separate serving replicas and the audio framing between them.
- Choose a chunking strategy for streaming ASR that balances transcription quality against partial-hypothesis latency.
- Configure a streaming TTS endpoint that emits first audio frame under a stated latency SLO.
- Implement barge-in: the server stops TTS mid-utterance when user audio is detected, and promotes the ASR path.
- Pick the wire protocol (WebSocket vs. WebRTC vs. HTTP-chunked) for your product's constraints.

## The voice-turn pipeline

A simple one-turn voice loop:

```
User audio in →→ [ASR replica] →→ text →→ [LLM replica] →→ tokens →→ [TTS replica] →→ Audio out
                   (streaming)              (streaming)              (streaming)
```

The product usually wants *all three* to be streaming — ASR emits partial hypotheses so the LLM can start drafting before the user stops speaking; the LLM streams tokens into the TTS so audio starts playing before the LLM finishes; the TTS emits audio frames as they're synthesised so the user hears the first word under 400 ms of the model deciding to speak.

Compared to text chat, three wiring differences:

- **Three replicas, not one.** Each model is a different family (an acoustic model + CTC / seq2seq decoder; an LLM; a mel-spectrogram-and-vocoder TTS). Each has its own batching rules (module 106 chapter 02's multi-model serving pattern is the baseline).
- **An end-of-utterance detector sits in front of or inside the ASR.** The pipeline has to know when the user *stopped* speaking so the LLM can commit to a final response (rather than keep redrafting on partial hypotheses).
- **The audio framing on the wire is codec-specific.** PCM at 16 kHz 16-bit mono is the baseline interchange format; Opus-in-WebRTC is common for the user→server direction; Opus or MP3 chunks for the server→user direction. The framing is explicit, not transport-implicit.

## Transport choice: WebSocket vs. WebRTC vs. HTTP

Three options:

| Transport | Direction | Pro | Con |
|---|---|---|---|
| **WebSocket (RFC 6455)** | full-duplex, long-lived | one TCP connection; ordered binary frames; good browser / mobile SDK support | TCP-based, so a dropped packet retransmits and head-of-lines the next frame — bad under mobile network jitter |
| **WebRTC (DataChannel + audio track)** | full-duplex, UDP-based | packet-loss resilient (Opus has FEC); sub-100 ms round-trip possible on good networks; browser-native audio capture | SDP / ICE negotiation overhead; TURN servers for NAT traversal; heavier operational surface |
| **HTTP long-poll / chunked** | one-way each, paired | works through every proxy on earth; falls back cleanly | two parallel streams to correlate; not real-time enough for most voice UX |

Default by product shape:

- **Browser-based voice assistant** → WebRTC for audio capture, WebSocket for text-side control signalling. The browser's `MediaRecorder` / `MediaStreamTrack` plus `RTCPeerConnection` is the native path.
- **Mobile app with native audio stack** → WebSocket, with the app doing its own packet-loss concealment on the audio stream.
- **Server-to-server** (e.g., a telephony gateway relaying a call) → WebSocket or gRPC streaming; pick the one your gateway supports natively.

Chapter 02 of module 111 drew the SSE discipline; this chapter inherits most of it (flush per frame, don't buffer, propagate cancellation) and adds the duplex choreography.

## Streaming ASR: chunking vs. latency vs. WER

Modern streaming ASR (Whisper-class, Conformer / Zipformer-class, RNN-T) runs the model on short audio chunks as they arrive, emitting *partial hypotheses* that may be revised when more audio is seen. Three knobs:

- **Chunk size.** How much audio feeds each forward pass. Small chunks (200–500 ms) emit partials quickly but have less acoustic context per decision; large chunks (1–2 s) are more accurate but add latency before the first partial appears. 500 ms is a typical default for Conformer-class streaming; Whisper is not natively streaming — the common pattern is a sliding-window inference with 30 s context and periodic re-decoding of the last N seconds.
- **Left-context / right-context.** How much past and future audio the chunk's decoder attends to. More right-context = better accuracy, higher latency. Streaming-trained models (Zipformer, RNN-T) usually have right-context 0 or tiny; non-streaming models forced into streaming via sliding-window have effective right-context equal to the lookahead you give them.
- **Final / stable-partial threshold.** Partial hypotheses are tentative; the serving layer commits them to "stable" once a confidence threshold holds across several chunks. Clients display partials in a lighter style and solidify them on stable events.

A representative streaming ASR event stream to the client:

```
{"type": "partial",  "text": "what is the weather"}
{"type": "partial",  "text": "what is the weather in"}
{"type": "partial",  "text": "what is the weather in Sea"}
{"type": "stable",   "text": "what is the weather in", "endpoint": 1270}   // ms into utterance
{"type": "partial",  "text": "Seattle today"}
{"type": "stable",   "text": "Seattle today", "endpoint": 1920}
{"type": "final",    "text": "what is the weather in Seattle today"}
```

`final` is the server's commitment that the user finished the utterance. The commit is usually triggered by a silence-after-speech heuristic (voice activity detection, or VAD) rather than a fixed timer.

For the serving platform, the operational implications:

- **Chunk size × hop size sets admission control.** A 500 ms chunk with a 100 ms hop is five overlapping forwards per second of audio — the per-user forward-pass rate is a linear function of concurrent sessions, not request count. Admission control should think in audio-seconds-in-flight, not requests-in-flight.
- **VAD lives in the serving layer, not the model.** The ASR model emits text; the serving layer emits the endpoint events. Keep the VAD state close to the audio demux so endpoint detection runs on raw audio, not the ASR output.
- **The ASR model's own batching is the limiting throughput factor.** Whisper-class inference can batch across concurrent sessions by padding chunks to a common length. `faster-whisper` on CTranslate2 and `WhisperX` / `whisper.cpp` ports have different batching ergonomics, and the NeMo Parakeet / Canary family is a separate runtime path with its own batching surface. Pick the runtime first against your workload, then tune its batch window; batching characteristics and throughput ceilings shift release to release, so re-measure rather than quote from a blog.

## Streaming TTS: time-to-first-audio-frame

The streaming TTS pattern:

1. **Input arrives as streamed text tokens from the LLM.** Don't wait for the full response — start synthesising as soon as the first clause is available.
2. **Tokenize into TTS-input units.** Usually phonemes or sub-word tokens for neural TTS; graphemes with text normalization for end-to-end models.
3. **Run the acoustic model.** For a transformer-TTS: encode the text, autoregressively predict mel-spectrogram frames. For a diffusion-TTS: run a diffusion loop on mel-spectrograms.
4. **Run the vocoder.** HiFi-GAN / BigVGAN / RingFormer / Vocos-class vocoder turns the mel-spectrogram into raw audio samples.
5. **Encode and frame for the wire.** Convert raw PCM to Opus / MP3 / AAC in short frames (20 ms Opus is the WebRTC default); emit on the audio-out stream.

The primary SLO is **time-to-first-audio-frame (TTFA)** — the wall-clock delay between "the LLM emitted the first text token" and "the user's speaker played the first synthesised byte." TTFA has four additive components:

- Text accumulation (wait for a synthesizable chunk, often one clause or 2–3 words).
- Acoustic-model forward for that chunk.
- Vocoder forward for the mel output.
- Framing + network + client-side playback latency.

A target TTFA under 400 ms is aggressive but achievable on modern GPUs; 1 s is conservative. Chapter 05 of module 110 (Pareto) applies here — a smaller acoustic model and a lighter vocoder trade quality for TTFA.

**Audio-render drift** is the secondary SLO: once playback starts, the server must deliver audio frames at least as fast as the client plays them. If the server falls behind, the client runs out of audio and the user hears a glitch. A healthy pipeline keeps a small buffer (100–300 ms) on the client; monitoring here is "buffer-underrun events per hour," not "P99 latency."

## Barge-in: the full-duplex state machine

Barge-in is the behaviour that makes a voice interface feel like a conversation rather than a walkie-talkie: the user starts speaking while the server is still playing TTS; the server stops the TTS mid-sentence and starts listening. The server-side state machine:

```
   [IDLE]
     │ user audio detected (VAD)
     ▼
   [LISTENING]
     │ end-of-utterance (VAD + silence timer)
     ▼
   [THINKING]              // LLM running; audio-in still being monitored
     │ LLM emits first clause
     ▼
   [SPEAKING]              // TTS streaming; audio-in still being monitored
     │                     // ← this is where barge-in may interrupt
     │ user audio detected while SPEAKING
     ▼
   [CANCELLING]            // stop TTS, cancel LLM if still generating, promote to LISTENING
     │
     ▼
   [LISTENING]
```

Three places the implementation is easy to get wrong:

- **Audio-in must keep flowing during `SPEAKING`.** Many naive implementations open the mic only in `LISTENING`; barge-in then can't fire because the server isn't seeing audio during TTS. The audio-in path must be always-on through the session.
- **Echo cancellation must run on the client.** The user's microphone picks up the TTS output played back through the speaker. Without echo cancellation, the ASR transcribes the model's own voice. Browser WebRTC does this natively (`echoCancellation: true`); native mobile apps use the platform's AEC.
- **Cancel semantics must reach every active replica.** When barge-in fires: the TTS replica must drop its remaining synthesis work (and free its queue slot), the LLM replica must abort any in-flight generation (module 111 chapter 02's client-disconnect handling applies), and the ingress must transition the state cleanly. Any of the three missing causes "ghost audio" (TTS keeps speaking for 500 ms after the user interrupts) or token leaks (LLM keeps burning KV cache).

## The chunked wire format (illustrative)

A representative WebSocket message schema a serving stack might use — not a standard, pattern-level:

```
C→S {"type": "session.start", "sr": 16000, "format": "pcm_s16le", "codec": "raw", ...}
C→S <binary audio chunk>                             // 20 ms PCM
C→S <binary audio chunk>
...
S→C {"type": "asr.partial", "text": "hello", "ts": 180}
S→C {"type": "asr.stable",  "text": "hello",   "endpoint": 420}
S→C {"type": "asr.final",   "text": "hello, what's the weather"}
S→C {"type": "llm.start",   "request_id": "req_abc"}
S→C {"type": "llm.delta",   "text": "The weather in"}
S→C {"type": "tts.start",   "voice": "alloy"}
S→C <binary audio chunk>                             // 20 ms Opus
S→C <binary audio chunk>
...
C→S <binary audio chunk>                             // user starts talking
S→C {"type": "tts.cancel",  "reason": "barge_in"}
S→C {"type": "llm.cancel",  "reason": "barge_in"}
S→C {"type": "asr.partial", "text": "wait"}
```

The key framing properties: text events are JSON on the control channel; audio is binary-framed on a dedicated audio channel (second WebSocket on some stacks, a WebRTC data track on others). Multiplexing both on one WebSocket as discriminated messages works too and is simpler to operate at the cost of coupling text-control throughput to audio throughput.

## Protocol references

- **WebSocket** — RFC 6455, <https://datatracker.ietf.org/doc/html/rfc6455>.
- **WebRTC** — W3C WebRTC specification, <https://www.w3.org/TR/webrtc/>; the IETF RTCWEB cluster of RFCs (JSEP, ICE, DTLS, SRTP).
- **Opus audio codec** — RFC 6716, <https://datatracker.ietf.org/doc/html/rfc6716>.
- **Media capture (browser)** — W3C Media Capture and Streams, <https://www.w3.org/TR/mediacapture-streams/>.

## Three failure modes

### "The assistant keeps talking after the user interrupts"

Common cause: the TTS cancel path is wired but the audio-playback buffer on the client is 500 ms deep and the server's cancel event doesn't arrive before the buffered frames play out.

Fix: send an explicit `tts.cancel` event the client acts on by flushing its playback buffer, not just stopping the frame stream. Combine with a smaller client buffer (100–200 ms) and server-side VAD that fires fast on interruption.

### "Partial hypotheses jump around and look nervous"

Common cause: the serving layer is emitting every ASR partial without a stability filter. The ASR model is doing its job, but the UI sees "wat", "wait", "want", "what" within 200 ms.

Fix: suppress partials that churn (same prefix revised more than N times in a short window); only emit stable events to the UI; keep raw partials on a debug channel.

### "Latency is fine on wifi, terrible on cellular"

Common cause: WebSocket over TCP head-of-lines when any audio frame retransmits — a single dropped packet freezes the audio stream for the RTT.

Fix: move audio to WebRTC (UDP-based, packet-loss resilient via Opus FEC). Keep the control channel on WebSocket. The TURN server for NAT traversal is the operational cost of this decision.

## Summary

- Speech serving is full-duplex: audio-in and audio-out run concurrently over the same session; SSE and request-response HTTP do not express this.
- Three replicas participate in a voice turn — ASR, LLM, TTS — each with its own batching, SLOs, and failure modes; chapter 02 of module 106 is the baseline.
- Streaming ASR trades chunk size against partial-hypothesis latency and WER; VAD and endpoint detection live in the serving layer.
- Streaming TTS has time-to-first-audio-frame as its primary SLO; audio-render drift is the secondary SLO measured as buffer-underrun events.
- Barge-in requires an always-on audio-in path, client-side echo cancellation, and cancel semantics that propagate to all three replicas simultaneously.
- WebRTC for audio under mobile / packet-loss conditions; WebSocket for everything else; HTTP-chunked only when the ingress / proxy stack forces it.
