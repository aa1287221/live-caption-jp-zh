# Selectable Gemini / local translation with Gemini parity

## Approved intent

Goal set by the user on 2026-09-26: meet the upstream owner's requirements while adding the
local-LLM features proposed in this fork; the local LLM must be wired in properly and must not
produce results below what the Gemini API path produces.

Upstream review of hhggffddssqjp-coder/live-caption-jp-zh#1 (Riva version, 9/22) raised:
translation quality depends on Gemini's context chaining (previous sentence, ASR correction,
editorial polishing), long sentences could be truncated, the WSL/Ontime setup is a burden, and
Windows capture/playback was not exercised. hhggffddssqjp-coder/live-caption-jp-zh#2 fixes a
duplicate `DelayedAudioPlayer.stop()` that left the audio feeder thread running.

## Decisions

1. **Gemini stays the default and keeps its behaviour.** Prompts, `temperature=0.3`, rate-limit
   skip in live captions, 60-line reconstruction batches, 40-line numbered offline batches,
   3 attempts with 20 s waits and 4.5 s pacing are unchanged. The only structural changes are
   moving the batch loops into `_polish_with_gemini` / `_translate_cues_gemini` and extracting
   prompt builders so the local path can share them. The Gemini SDK is imported lazily.
2. **The engine is selectable**: startup dialog option ⑥ (remembered in `last_settings.json`),
   `TRANSLATION_BACKEND`, or `transcribe_audio_file.py --backend`. The translator is built and
   verified before the source application's audio route is changed.
3. **Transport** (`local_llm.py`, standard library only, never starts or downloads anything):
   - `instruct`: `POST /v1/chat/completions` with the unchanged system/user messages, so the
     server applies the model's own chat template; `temperature=0.3`, `seed=1234`,
     `chat_template_kwargs.enable_thinking=false`, optional `model` for multi-model servers.
   - `riva`: `POST /completion` with Ontime's turn markers and greedy sampling (the 9/26 contract).
   - `auto` (default) reads `GET /props`: Riva-Translate templates or model names → `riva`.
   - `ensure_ready` waits on `/health` (503 while loading), falls back to `/v1/models`.
   - Errors carry `kind`: `context` (HTTP 400 `exceed_context_size_error` or prompt truncation)
     and `truncated` (`stop_type=limit`, `finish_reason=length`) are `too_large`; `connection`
     and `timeout` are `unreachable`. Prompts and model output are never included.
4. **Parity**: live, reconstruction and numbered-batch prompts come from the same builders
   (`_context_prompts`, `_rebuild_user_prompt`, `_offline_user_prompt`, `_REBUILD_SYSTEM_PROMPT`,
   `_OFFLINE_SYSTEM_PROMPT`). Integration tests assert byte-identical prompts and identical
   offline outputs for both engines.
5. **Guard rails for the local path** (`local_translation.py`), all additive:
   - live: 15 s timeout, no sleeps; an empty/echoed/untranslated/runaway reply is retried once
     with the context-free prompt; a live circuit breaker pauses 30 s after 3 unreachable calls;
   - batches: 20 lines with the 8 preceding lines as context (Gemini uses 60/40 lines with 2);
     truncation, context overflow, malformed output and timeouts split the batch in half down
     to a single-line fallback; numbered gaps are re-requested; nothing is silently dropped;
   - output budget scales with the batch and never exceeds half the server context;
   - 3 consecutive transient failures abort the workflow; reconstruction then keeps the WAV.
6. **Translation-only models** (`riva` profile): a translation model cannot follow the Gemini
   instructions (the 9/20 probe returned only the first line of labelled context), so this
   profile uses Ontime's plain `Translate this into Traditional Chinese: {text}` prompt with an
   empty system turn, masks glossary terms and restores configured targets, and splits input
   longer than 120 characters. This deliberately deviates from the 9/26 spec's "never drop
   context" rule; the startup log and the reconstruction header state that this profile does not
   use context, correction or polishing. Gemini parity requires an instruction model.
7. **Non-inferiority check**: `compare_translation_backends.py` runs identical samples through
   both engines with the application's own `Translator` (live or offline mode), reports problem
   rates, glossary hits, latency and chrF similarity, optionally has Gemini judge each pair blind,
   and exits non-zero when the local engine is worse than Gemini beyond configurable margins
   (problem rate +2 %, glossary hit rate −2 %, judge score ≥ 0.45).
8. **Removed**: the Ontime relay client (`ontime_riva.py`, WSL auto-start) and its tests.

## Verification

- `python -m unittest discover -s tests`: transport, engine, application integration (Gemini
  unchanged, prompt parity, guard rails, wiring) and harness tests, without GPU/audio/network.
- A real llama-server built from llama.cpp `4b1a27f` served a deterministic test GGUF; the
  application paths (startup, live caption, numbered offline batches, reconstruction, harness)
  ran end to end in `instruct` and auto-detected `riva` modes, and a 512-token context server
  produced real `exceed_context_size_error` responses that were split without losing lines.
  `tests/test_llama_server_live.py` repeats the contract checks against any running server.
- Not verified here: translation quality of real models (use the harness), live Gemini calls,
  and Windows capture/playback.
