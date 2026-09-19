# Local translation with feature parity

## Approved intent

Replace Gemini inference with a local language model while preserving the existing application functions. The user approved implementation on 2026-09-20. Existing entrypoint names remain valid for launcher, shortcut, and offline-transcription compatibility.

## Source findings and decision

`live_caption_gemini.py:1218` implements `Translator`; `translate` and `translate_with_context` construct system instructions, glossary hints, and previous-sentence context. `_chat(system_prompt, user_prompt)` is also called by `rebuild_transcript_from_full_audio` for bilingual reconstruction and by `transcribe_audio_file.py` for numbered batches. These are instruction-following workloads, not bare text translation.

Ontime's `tools/relay/llama_runtime.py:452` demonstrates a local llama.cpp server reached through standard-library HTTP with explicit timeouts. Its `tools/relay/translate_api.py` accepts text and language fields (plus a beam-size hint), caps batches at 32 texts, and returns DeepL-shaped results. It has no implementation of this application's glossary, contextual instruction, or editorial contracts. `tools/relay/nmt.py` explicitly constructs Riva-specific translation templates for `/completion`, rather than using the server's chat template.

Adopt Ontime's local HTTP inference architecture, using a **general instruction-following model** behind llama.cpp's OpenAI-compatible `/v1/chat/completions`. Do not route existing prompts to Riva's `/v1/translate` or imply that translation-specific Riva reproduces arbitrary editorial instructions. Keep the local service external to this application; do not start/stop Ontime, download models, or alter its repository. A model swap preserves the application's feature contracts, not identical linguistic quality or latency.

## Global constraints

- Python 3.10 or newer; local transport uses only the standard library.
- Preserve `live_caption_gemini.py`, `launcher_gemini.py`, and `transcribe_audio_file.py` entrypoints.
- No Google SDK import, Gemini API key lookup, or automatic cloud fallback in the migrated path.
- Preserve translation prompts, glossary hints, context boundaries, repetition cleanup, batch mapping, subtitle timing, and output file formats.
- Preserve Whisper, capture, playback, recording, pause/resume, and prerecorded cue behavior.
- New Python blocks use tabs; new comments and docstrings use English. Do not reindent unrelated existing blocks.
- User-facing Chinese text uses Taiwan Traditional Chinese.
- Do not modify other translator variants, delete files, change shared services, or create a PR.

## Transport contract

New `local_llm.py` owns `LocalChatClient` and `LocalLLMError`. `LocalChatClient(base_url=None, model=None, timeout=None)` reads `LOCAL_LLM_BASE_URL` (default `http://127.0.0.1:8080/v1`), `LOCAL_LLM_MODEL` (default `local-model`), and `LOCAL_LLM_TIMEOUT` (default `30` seconds). Explicit constructor values override environment values. Accept a base URL with or without a trailing `/v1`; normalize once to exactly `/v1/chat/completions`. Require an HTTP(S) URL with a hostname and a finite positive timeout. Reject relay `/v1/translate` and completion-path URLs rather than appending a confusing suffix.

`chat(system_prompt: str, user_prompt: str) -> str` sends one nonstreaming request with `model`, the original two role messages, `temperature=0.3`, and `stream=False`. It returns stripped `choices[0].message.content`. Empty content, nonstring content, missing fields, invalid JSON, HTTP failures, and connection/timeout errors raise `LocalLLMError`. A length-truncated response (`finish_reason == "length"`) is a failure, not a complete transcript. No retries or sleeps occur inside transport. Errors must include an actionable category without printing prompts. Bypass proxy environment variables for local inference and reject redirects so an endpoint cannot silently move requests elsewhere.

`Translator` owns existing presentation behavior: call `client.chat`, apply `_collapse_repetition`, print an actionable local-service error and return `""` on `LocalLLMError`. Existing offline callers retain their retry policy; live requests remain a single attempt. Invalid client configuration fails clearly on construction.

## Verification and limitations

Transport tests run on Linux without Windows/audio/GPU imports. Integration tests exercise real application translation methods and offline callers with mocked hardware imports, a fake Whisper model, and a fake local client. Preserve actual original prompt strings and output mappings, not merely method presence. Avoid testing recording cleanup by deleting an audio fixture: mock `Path.unlink` for that integration scenario.

A running general-purpose local model enables a separate live smoke test for ordinary translation, glossary/context instructions, numbered output, and bilingual reconstruction. If no compatible model is running, report that limitation explicitly. Passing mocked tests establishes wiring and feature contracts, not translation quality. Full audiovisual validation remains a Windows task.
