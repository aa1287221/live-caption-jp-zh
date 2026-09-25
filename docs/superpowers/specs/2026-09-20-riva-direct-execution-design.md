# Direct execution with Ontime Riva

> **Status:** superseded by `2026-09-26-dual-backend-gemini-parity-design.md`. The Ontime relay client and WSL auto-start were removed; Riva-Translate is used directly through llama-server (`LOCAL_LLM_MODE=riva`).

Status: approved by the user on 2026-09-20. This follows the instruction-model migration at `2e811b3` and supersedes its requirement to provision another model.

## Outcome and accepted tradeoff

Use the existing Ontime Riva installation to execute Japanese-to-Traditional-Chinese translation immediately. Preserve live capture, Whisper, subtitle timestamps, playback, recording, cue JSON, bilingual logs, transcript filenames, and glossary terminology. The user explicitly accepted loss of previous-sentence semantic context, ASR text correction by the translator, and editorial polishing. Full-audio Whisper retranscription remains available, but its raw Japanese is preserved and each corresponding Chinese translation is formatted deterministically.

## Grounded protocol

Primary references are `/project/Ontime-Translator/tools/relay/translate_api.py`, `nmt.py`, `server.py`, `run.sh`, and `/project/Ontime-Translator/start-relay.sh`.

- Default service origin: `http://127.0.0.1:8765`.
- Readiness: `GET /health`, requiring `nmt.loaded is True`, `nmt.state == "ready"`, and no `nmt.error`.
- Translate: `POST /v1/translate` with `{"text": ["こんにちは。"], "source_lang": "JA", "target_lang": "zh-TW"}`. The exact `zh-TW` target is supported by `nmt_text.py` and confirmed by a live probe.
- Response: `translations` contains one ordered object per input, each with `text` and `ontime.verdict`, `ontime.reasons`, `ontime.guard`.
- Requests contain at most 32 strings; each string is at most 1200 characters. Split longer text at Japanese sentence boundaries, then bounded character boundaries, and reassemble one result per original input in order.
- `reject` is an error even under HTTP 200. `input:truncated` is an error regardless of verdict. Accept `warn` only when not truncated, and report its reasons. A nonempty substantive input yielding `skipped` or empty output must be explicit failure. Whitespace-only input bypasses the service.
- HTTP 400/422, malformed payloads, mismatched counts, missing verdicts, and glossary integrity failures are permanent failures. Surface connection errors, 429, 503, and timeouts with actionable details. Transport does not secretly retry.

## Service lifecycle

Create a focused `ontime_riva.py` module for configuration, transport, input splitting, glossary restoration, and readiness/startup. Retain `local_llm.py` and its tests as an unused compatibility module; do not delete files.

Configuration uses `ONTIME_RELAY_URL`, `ONTIME_REPO_PATH`, `ONTIME_WSL_DISTRO`, `ONTIME_AUTO_START`, `ONTIME_START_TIMEOUT`, and `ONTIME_TRANSLATE_TIMEOUT`; defaults are `http://127.0.0.1:8765`, `/project/Ontime-Translator`, `Ubuntu-26.04`, `1`, `180`, and `120` respectively. Validate booleans, positive finite deadlines, and origins before launching anything. Only auto-start a local loopback target; a configured remote service is checked but never launched locally.

Reuse a ready relay without starting or stopping it. Wait on an already-starting relay. When no relay is reachable and auto-start is enabled, invoke the existing launcher once with `--no-preload --port 8765` (use the configured URL port). On Windows, use `wsl.exe -d Ubuntu-26.04 -e bash /project/Ontime-Translator/start-relay.sh --no-preload --port 8765`; on Linux, execute the same script directly with Bash. Use argument arrays, not interpolated shell commands. Existing scripts and assets must exist; never download models or dependencies automatically. Persist service output in an application log and include a bounded tail plus its path on startup failure. Windows Python can open the durable log in this repository and pass its handle as `Popen` stdout/stderr while launching WSL with direct argv. Detach the process appropriately and leave a reused or started relay running after the caption application exits. Do not kill a service occupying the port. Disabled/failed NMT and malformed health responses produce clear errors rather than starting a competing instance.

## Translation behavior

`Translator.translate_with_context()` remains callable for compatibility but translates only `current_ja`; no prompt strings are sent to Riva. Its name does not justify claims of contextual translation in UI or documentation. `Translator.translate_batch()` returns exactly one output string per original Japanese input. Live errors are logged and become `（翻譯失敗，保留日文原文）{source}` without retry or sleep. Never expose a rejected candidate or silently return an empty translation for substantive source text. A punctuation-only skipped item may preserve source punctuation.

Glossary handling masks matched source terms longest-first with collision-resistant markers, translates, then restores exact configured targets. Preserve repeated occurrences, overlaps, and source text already containing marker-like strings. Never split inside a marker. Check that every expected marker survives with its exact multiplicity; missing, duplicated, or unexpected markers fail explicitly. Smoke-test this strategy on the installed Riva before claiming working terminology support.

Offline callers batch at 32 lines, preserve source order and cue timestamps, and use at most three caller-level attempts for transient failures. Permanent guard/glossary errors are not retried. Failed items become `（翻譯失敗，保留日文原文）{source}`; a request-level failure produces that explicit fallback for each input in the affected request. Log degraded completion clearly and never claim those fallback strings were translated. Full-audio output writes raw JA then ZH with a blank line between pairs. Remove claims that Riva edits ASR or considers the whole conversation. Preserve existing successful-reconstruction WAV cleanup semantics; tests mock `wav_path.unlink()` and execution does not delete unrelated files.

## Existing live probe evidence

Main-session probes reported that `__GLOSSARY_0__` survived Riva translation and could be restored, while private-use Unicode markers were rejected. Labeled concatenated context returned only the first line, supporting the current-sentence-only design. The numeral guard correctly rejected a model output that changed 三億 to 三十億. These are measured spot checks, not guarantees across arbitrary inputs; marker multiplicity and guard validation remain mandatory.

## Verification and completion

Automated verification covers the actual HTTP transport against a loopback fake server, mixed item verdicts, truncation, cardinality, long input, glossary integrity, mocked Windows/Linux startup, startup reuse/failure, and integration with 33+ source lines. Existing local instruction-client tests remain passing. Live verification must start/reuse the installed relay and translate Japanese through the new client, including glossary and multiple inputs. Independent verification assesses behavior from the user workflow, not only the implementation diff. Windows capture/playback is reported as unverified if no Windows execution is available; Linux protocol success must not be described as a Windows end-to-end run.
