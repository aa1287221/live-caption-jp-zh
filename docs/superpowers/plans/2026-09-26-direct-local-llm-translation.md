# Direct local LLM translation implementation plan

> **Status:** partly superseded by `2026-09-26-dual-backend-gemini-parity.md`.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Gemini API inference with direct local llama-server `/completion` calls while preserving the repository's existing translation workflows.

**Architecture:** Keep the application-facing `Translator._chat(system_prompt, user_prompt)` boundary. Implement a standard-library `LocalLLMClient` that frames those prompts with Ontime-Translator's llama.cpp turn markers and posts directly to `/completion` using the measured Riva local sampling contract. The application never imports or starts Ontime-Translator.

**Spec:** `docs/superpowers/specs/2026-09-26-direct-local-llm-translation-design.md`

## Global constraints

- Use Python 3.10+ standard-library HTTP only for the local client.
- Preserve entry-point filenames and existing public translation/cue/output interfaces.
- Preserve prompt construction, glossary/context data, batch numbering, retry policy, timing, queues, recording cleanup, and file formats.
- No Gemini SDK, API key lookup, cloud fallback, Ontime import, relay call, WSL command, or `/v1/translate` endpoint.
- Do not delete existing wrong-direction files in this task.
- New Python blocks use tabs and code comments use English.

## Task 1: Implement and test direct completion transport

**Files:** `local_llm.py`, `tests/test_local_llm.py`.

- [ ] Add validated configuration from `LOCAL_LLM_BASE_URL`, `LOCAL_LLM_TIMEOUT`, and `LOCAL_LLM_MAX_TOKENS`; default to `http://127.0.0.1:8766`, 120 seconds, and 256 tokens.
- [ ] Add `LocalLLMError` and a client with `complete(prompt)` plus a compatibility `chat(system_prompt, user_prompt)` method.
- [ ] Build the exact `<s>System` / `<s>User` / `<s>Assistant` framing and send `POST /completion` with `stream=false`, `temperature=0`, `top_p=1`, `top_k=0`, `repeat_penalty=1`, `seed=1234`, `n_predict=256`, and both stop strings.
- [ ] Bypass proxies and reject redirects. Validate HTTP errors, connection/timeout errors, invalid JSON, malformed response shape, empty text, and length truncation without logging request content.
- [ ] Write tests for non-ASCII prompt fidelity, exact endpoint/payload, constructor validation, timeout propagation, redirect rejection, malformed responses, and one-request/no-retry behavior.
- [ ] Run `python -m unittest discover -s tests -p test_local_llm.py -v`, `python3 -m py_compile local_llm.py`, and `git diff --check`.
- [ ] Review and commit only task files.

## Task 2: Replace the Gemini provider boundary

**Files:** `live_caption_gemini.py`, `transcribe_audio_file.py`, `launcher_gemini.py`.

- [ ] Replace Gemini imports, key/model setup, and provider status text with the local client; preserve the existing method signatures and prompt-building logic.
- [ ] Make `_chat` call `LocalLLMClient.chat`, retain repetition cleanup, and convert `LocalLLMError` to the existing empty-result/error path.
- [ ] Ensure offline reconstruction and numbered batch translation use the same client without bypassing their current retry behavior.
- [ ] Confirm no runtime import or string reference points to `ontime_riva`, `ONTIME_RELAY_URL`, WSL startup, or `/v1/translate`.
- [ ] Run focused integration tests with audio/Whisper modules mocked and verify context, glossary, batch mapping, timestamps, queue tuples, and output files.
- [ ] Review and commit only task files.

## Task 3: Correct setup documentation and regression coverage

**Files:** `README.md`, `requirements.txt`, `tests/test_local_translation_integration.py`.

- [ ] Document that the user starts a compatible local llama-server independently, with its loaded local model; show only `LOCAL_LLM_BASE_URL`, `LOCAL_LLM_TIMEOUT`, and `LOCAL_LLM_MAX_TOKENS` configuration.
- [ ] Remove Ontime project paths, relay startup, WSL commands, and `/v1/translate` instructions from the migrated setup section.
- [ ] Explain that the historical `gemini` filename remains for compatibility, that model quality/latency can differ, and that localhost reachability and shared VRAM affect operation.
- [ ] Add regression tests proving construction needs no Google package/key and that local transport errors preserve existing caller behavior.
- [ ] Run the full test suite, compile changed Python files, `git diff --check`, and inspect the final diff for unrelated subsystem changes.
- [ ] Commit only documentation/tests and report unavailable live-model or Windows audiovisual checks.

## Task 4: Independent acceptance audit

- [ ] Compare the complete branch diff against the spec and this plan.
- [ ] Confirm the direct request path is `/completion`, never `/v1/translate` or `/v1/chat/completions`.
- [ ] Confirm no Ontime sibling path, WSL startup, relay dependency, or model download was introduced.
- [ ] Confirm existing wrong-direction files remain present but are unreachable from the migrated runtime.
- [ ] Run final scoped tests and hand off test results, commit SHAs, and any live-environment limitations.
