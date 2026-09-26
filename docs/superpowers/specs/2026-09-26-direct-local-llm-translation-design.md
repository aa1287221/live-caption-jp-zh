# Direct local LLM translation replacement

> **Status:** partly superseded by `2026-09-26-dual-backend-gemini-parity-design.md`. The `/completion` contract below is kept as `LOCAL_LLM_MODE=riva`; Gemini is kept as the default engine instead of being replaced.

## Approved intent

Replace the Gemini inference boundary in this repository with the local LLM
calling method used by Ontime-Translator. The application must call a local
`llama-server` directly; it must not import, start, stop, or otherwise depend
on the sibling Ontime-Translator project.

The existing caption, batch transcription, glossary, context, timing, queue,
and output workflows remain in place. Entry-point filenames remain unchanged
for shortcut and import compatibility. This change replaces the inference
transport and provider configuration only.

## Non-goals and dependency boundary

- Do not call Ontime-Translator code, its relay, or `/v1/translate`.
- Do not execute WSL commands or discover a sibling project path.
- Do not download or manage a model from this repository.
- Do not change Whisper, capture, playback, recording cleanup, subtitle
  timing, queue tuples, batch mapping, or output formats.
- Do not delete existing wrong-direction files in this task. They are not part
  of the runtime after the migration; a separate, explicitly approved cleanup
  can remove them later.

The only external prerequisite is a user-started local inference service with
the loaded local model. The service URL is configured by this application,
not inferred from an Ontime installation.

> **Superseded (start/stop only):** the "user-started" clause above is superseded by
> `2026-09-26-managed-llama-server-design.md`, which has the app start/stop a
> `llama-server` it spawned itself when nothing answers the configured base URL. An
> already-running service is still always used as-is and never touched.

## Local inference contract

`local_llm.py` owns a standard-library HTTP client. Its primary operation is
`complete(prompt: str) -> str`, which sends one non-streaming `POST` request to
`{LOCAL_LLM_BASE_URL}/completion`.

The request JSON is:

```json
{
  "prompt": "<fully framed prompt>",
  "stream": false,
  "temperature": 0.0,
  "top_p": 1.0,
  "top_k": 0,
  "repeat_penalty": 1.0,
  "seed": 1234,
  "n_predict": 256,
  "stop": ["</s>", "<s>"]
}
```

These values reproduce Ontime-Translator's direct llama.cpp completion
method. The client must use an explicit finite positive timeout, bypass proxy
environment variables for localhost inference, reject redirects, and never
retry internally. The client must reject malformed JSON, missing or
non-string `content`, empty output, and an explicit length truncation. Errors
are raised as `LocalLLMError` without logging prompts or model output.

Default configuration:

| Environment variable | Default | Meaning |
| --- | --- | --- |
| `LOCAL_LLM_BASE_URL` | `http://127.0.0.1:8766` | llama-server root URL; `/completion` is appended |
| `LOCAL_LLM_TIMEOUT` | `120` | Per-request timeout in seconds |
| `LOCAL_LLM_MAX_TOKENS` | `256` | `n_predict`; must be a positive integer |

The loaded GGUF/model alias is a server concern. No `LOCAL_LLM_MODEL` field is
required by the direct completion API, and the client must not pretend to
select a model that the server has already loaded.

## Prompt framing

The client preserves the existing application-facing `chat(system_prompt,
user_prompt)` boundary so translation callers do not change. The adapter turns
those two strings into the llama.cpp prompt format used by Ontime:

```text
<s>System
{system_prompt}</s>
<s>User
{user_prompt}</s>
<s>Assistant
```

For a plain Japanese-to-Traditional-Chinese request, the exact Ontime prompt
shape is:

```text
<s>System
</s>
<s>User
Translate this into Traditional Chinese: {source_text}</s>
<s>Assistant
```

The adapter must not silently drop the caller's context or glossary text. If a
model context or output budget is insufficient, it raises a bounded transport
error and leaves the existing caller retry/fallback behavior responsible for
the workflow result. It must not truncate source text to force a request to
fit.

The client strips only leading/trailing whitespace from the returned assistant
text. Turn markers are controlled by the `stop` list; callers retain their
existing repetition cleanup and numbered-output parsing.

## File ownership

| File | Responsibility |
| --- | --- |
| `local_llm.py` | Direct `/completion` transport, prompt framing, configuration, response validation, `LocalLLMError` |
| `live_caption_gemini.py` | Replace Gemini construction and `_chat` provider boundary; preserve translation prompts and workflow logic |
| `transcribe_audio_file.py` | Use the same local client through its existing translation caller; preserve offline retry and output behavior |
| `launcher_gemini.py` | Update provider-facing status/help text only; retain filename and launch behavior |
| `tests/test_local_llm.py` | Hardware-independent HTTP, framing, sampling, validation, timeout, redirect, and error tests |
| `tests/test_local_translation_integration.py` | Real translation methods/callers with the local client mocked; verify feature contracts |
| `README.md` | Direct llama-server setup, environment variables, Windows reachability, model/VRAM limits, and smoke test |
| `requirements.txt` | Remove Gemini-specific dependency wording only; do not add an Ontime dependency |

Existing `ontime_riva.py`, Riva-focused tests, and historical planning files
are intentionally not deleted in this task. They must not be imported or
referenced by the migrated runtime or setup documentation.

## Acceptance criteria

- The migrated runtime has no Google SDK import, Gemini key lookup, cloud
  fallback, Ontime import, WSL startup, or `/v1/translate` call.
- All existing translation entry points reach the same local client and keep
  their public function signatures, prompt construction, glossary/context
  behavior, retry policy, subtitle timing, queue tuple shape, and output files.
- Unit tests assert the exact `/completion` path, prompt framing, sampling
  payload, UTF-8 handling, timeout, redirect rejection, malformed responses,
  empty output, and truncation handling.
- Integration tests exercise live-caption and offline reconstruction paths
  with a fake local client and verify no unrelated audio or timing behavior
  changed.
- README instructions require only a user-started local llama-server and a
  loaded compatible model; no Ontime path or relay setup appears.
- A real smoke test is reported only when a compatible local service is
  available. Mocked tests demonstrate wiring and feature contracts, not
  translation quality.
