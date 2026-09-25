# Local Translation Implementation Plan

> **Status:** superseded by `2026-09-26-dual-backend-gemini-parity.md`.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Gemini with local instruction-following inference without losing any translation-dependent application feature.

**Architecture:** Keep original prompts and callers, replace the provider transport behind `Translator._chat` with a standard-library local chat client. Follow Ontime's local llama.cpp HTTP architecture, but use a general-purpose instruction model because its Riva translation relay cannot honor this application's editorial and context contracts.

**Tech Stack:** Python 3.10+, urllib, json, unittest; external llama.cpp-compatible local service.

**Spec:** `docs/superpowers/specs/2026-09-20-local-translation-design.md`

## Global Constraints

- Python 3.10 or newer; local transport uses only the standard library.
- Preserve `live_caption_gemini.py`, `launcher_gemini.py`, and `transcribe_audio_file.py` entrypoints.
- No Google SDK import, Gemini API key lookup, or automatic cloud fallback in the migrated path.
- Preserve translation prompts, glossary hints, context boundaries, repetition cleanup, batch mapping, subtitle timing, and output file formats.
- Preserve Whisper, capture, playback, recording, pause/resume, and prerecorded cue behavior.
- New Python blocks use tabs; new comments and docstrings use English. Do not reindent unrelated existing blocks.
- User-facing Chinese text uses Taiwan Traditional Chinese.
- Do not modify other translator variants, delete files, change shared services, or create a PR.

## File responsibilities

| File | Responsibility |
| --- | --- |
| `local_llm.py` (new) | Config validation, request/response transport, typed failure |
| `tests/test_local_llm.py` (new) | Hardware-independent transport contract |
| `live_caption_gemini.py` | Replace provider initialization and `_chat` only; correct provider documentation |
| `launcher_gemini.py` | Correct user-visible local-provider label |
| `transcribe_audio_file.py` | Correct configuration documentation; preserve existing batch caller |
| `tests/test_local_translation_integration.py` (new) | Real translation methods and offline output contracts with hardware mocked |
| `README.md` | Local service setup, endpoint/model/timeout configuration, feature and platform limits |

### Task 1: Add independently testable local chat transport

**Files:** Create `local_llm.py`, `tests/test_local_llm.py`.

**Interfaces:**

- Produces `LocalLLMError(RuntimeError)`.
- Produces `LocalChatClient(base_url: str | None = None, model: str | None = None, timeout: float | None = None)`.
- Produces `LocalChatClient.chat(self, system_prompt: str, user_prompt: str) -> str` and readable `base_url`, `model`, `timeout` attributes.
- Consumes environment values defined in the spec; no application/hardware imports.

- [ ] **Step 1: Write failing request contract tests.** Use `unittest` and a mocked urllib opener; a fake response must implement the context manager protocol. Cover UTF-8 request fidelity and these response/transport cases: successful response, invalid JSON, empty choices, null/nonstring/empty content, `finish_reason=length`, HTTP 400/404/429/503, refusal, timeout. Assert one request per chat, explicit timeout, and no sleeps. Test base URLs with and without `/v1`, invalid scheme/host/path, empty model, and nonpositive/nonfinite timeout. Core success assertions:

```python
payload = json.loads(request.data.decode("utf-8"))
self.assertEqual(request.full_url, "http://127.0.0.1:8080/v1/chat/completions")
self.assertEqual(payload["messages"], [
	{"role": "system", "content": "只翻譯目前句子"},
	{"role": "user", "content": "前一句：田中です。\n目前：よろしく。"},
])
self.assertEqual(payload["model"], "local-model")
self.assertEqual(payload["temperature"], 0.3)
self.assertIs(payload["stream"], False)
```

- [ ] **Step 2: Run the red tests.** `python -m unittest discover -s tests -p test_local_llm.py -v` should fail because `local_llm` does not exist yet.

- [ ] **Step 3: Implement configuration and transport.** Constructor reads explicit argument first, environment second, defaults third. Validate with `urllib.parse.urlsplit` and `math.isfinite`. Normalize root URLs to `/v1` and preserve supplied `/v1`. Build an opener with `ProxyHandler({})` and an HTTPRedirectHandler subclass whose `redirect_request` returns `None`. The request core is:

```python
payload = {
	"model": self.model,
	"messages": [
		{"role": "system", "content": system_prompt},
		{"role": "user", "content": user_prompt},
	],
	"temperature": 0.3,
	"stream": False,
}
request = urllib.request.Request(
	self.base_url + "/chat/completions",
	data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
	headers={"Content-Type": "application/json"},
	method="POST",
)
```

Use the opener with `timeout=self.timeout`. Validate response containers and first message before returning stripped text. Catch HTTPError before URLError; raise LocalLLMError with HTTP status or connection/JSON/shape/truncation category and endpoint guidance. Never log raw request/response bodies. Do not auto-discover models or silently fall back to another provider. Do not truncate user prompts to fit a model.

- [ ] **Step 4: Verify red tests now pass, plus a real loopback HTTP smoke test.** A stdlib `ThreadingHTTPServer` fixture returns chat JSON and captures payload; verify the public `chat` method traverses actual HTTP, preserves non-ASCII text, rejects a redirect, and ignores configured proxy environment variables. Local socket restrictions require the normal sandbox escalation process, not fabricated success.

- [ ] **Step 5: Review and commit task 1.** Inspect `git diff --check`, then stage only these task files and commit `feat: add local instruction model chat transport`. Parent session handles repository permissions and review gates.

### Task 2: Wire existing workflows and prove feature-contract parity

**Files:** Modify `live_caption_gemini.py`, `launcher_gemini.py`, `transcribe_audio_file.py`, `README.md`; create `tests/test_local_translation_integration.py`.

**Interfaces:**

- Consumes `LocalChatClient` and `LocalLLMError` from task 1.
- Preserves `Translator()`, `_chat(system_prompt, user_prompt)`, `translate(text, glossary=None)`, and `translate_with_context(prev_ja, current_ja, glossary=None)` signatures.
- Preserves `rebuild_transcript_from_full_audio(...)`, `transcribe_audio_file(...)`, and cue queue tuple/output file contracts.

- [ ] **Step 1: Write failing integration tests against actual production methods.** Mock Windows/audio/GPU modules in `sys.modules` before importing `live_caption_gemini`; do not fake the translator module itself. For actual offline calls patch Whisper and use a temporary output directory. Mock `Path.unlink` in reconstruction to prevent recording deletion. Mock sleep to avoid retry delays. Capture `LocalChatClient.chat` calls. Concrete live behavior:

```python
translator = app.Translator()
translator.client.chat.return_value = "  請多指教。  "
self.assertEqual(translator.translate_with_context(
	"田中です。", "よろしく。", {"田中": "田中先生"}
), "請多指教。")
system, user = translator.client.chat.call_args.args
self.assertIn(app.glossary_hint({"田中": "田中先生"}), system)
self.assertIn("田中です。", user)
self.assertIn("よろしく。", user)
self.assertIn("不用翻譯", user)
```

Existing `glossary_hint` preserves glossary keys and does not render mapped values; retain this behavior rather than changing glossary policy. Also test empty inputs produce zero HTTP calls, duplicate-output cleanup remains active, LocalLLMError produces `""`, and construction works with no Google module or key. Test the following offline scenarios using fake ASR segments with deterministic timestamps:

1. Forty-one numbered lines span two batches; first response is 40 numbered translations, second one translation. Every cue retains its exact start/end/ja and correctly matched zh; second batch prompt retains context and glossary. Parser still fills a missing numbered translation with empty text rather than shifting other lines.
2. Sixty-one lines of full-audio reconstruction span two batches; return bilingual outputs, assert both preserved in the Markdown, second prompt includes prior context, system prompt includes glossary and bilingual-editing instructions.
3. First offline request fails then succeeds; caller retries while transport does not. All-failure reconstruction returns None and does not delete audio.
4. `push_cues_to_queues` emits unchanged `(start + playback_start_time + DISPLAY_DELAY_SEC, ja, zh)` tuples to both queues.

- [ ] **Step 2: Run red integration tests.** `python -m unittest discover -s tests -p test_local_translation_integration.py -v`; expected failure is old Google initialization or missing local-client wiring, not an unrelated hardware import failure.

- [ ] **Step 3: Replace only the provider boundary.** Remove Google imports and key/model constants. Import the new transport. Replace Translator constructor, `_load_api_key`, `_chat`, and `_print_available_models` with the local implementation; keep actual translation prompt-building methods verbatim. Whole replacement methods use tabs to avoid mixing indentation inside a block.

```python
def __init__(self):
	self.client = LocalChatClient()
	print(f"翻譯將使用本機模型：{self.client.model}（{self.client.base_url}）")

def _chat(self, system_prompt: str, user_prompt: str) -> str:
	try:
		return _collapse_repetition(self.client.chat(system_prompt, user_prompt))
	except LocalLLMError as exc:
		print(f"本機模型翻譯失敗：{exc}")
		return ""
```

Update obsolete Gemini/free-quota messaging in touched paths and docstrings. Preserve existing offline retry counts, batch sizes, context handling, and sleeps in this bounded migration; performance tuning is separate. Do not change Whisper or recording cleanup logic. Keep script filenames to preserve existing shortcuts/imports.

- [ ] **Step 4: Document concrete setup and limitations.** Add a clearly labeled local replacement section to README without implying the other existing Ollama variants changed. Show the external service contract and Windows PowerShell configuration:

```powershell
$env:LOCAL_LLM_BASE_URL = "http://127.0.0.1:8080/v1"
$env:LOCAL_LLM_MODEL = "local-model"
$env:LOCAL_LLM_TIMEOUT = "30"
python live_caption_gemini.py
python transcribe_audio_file.py recording.wav --title "節目名稱"
```

Explain that `LOCAL_LLM_MODEL` must match the server model/alias, a general instruction-following model is required, no Gemini key or Google SDK is required, and the old filename is a compatibility name. Explain localhost reachability from the **Windows Python process**, shared VRAM with Whisper, timeout tuning, long offline prompt context requirements, and that Riva `/v1/translate` is not the configured endpoint. If giving llama-server startup flags, verify against the installed binary's help before documenting them. Do not pick/download a new model silently.

- [ ] **Step 5: Run final scoped verification and review.** Run `python -m unittest discover -s tests -v`, compile changed Python modules, `git diff --check`, and inspect diffs for any changes to prompts, timestamps, capture/playback or other variant files. Confirm no remaining Google imports/key reads in the migrated runtime. Run a real model smoke test only if a compatible configured server exists; report exact endpoint/model and tested behaviors, or explicitly mark live-model/Windows audiovisual checks unavailable. Do not claim translation quality equivalence based on fake responses.

- [ ] **Step 6: Commit and hand off.** Stage only task-2 files, commit `feat: replace Gemini inference with local model across caption workflows`, and hand parent the diff, test results, limitations, and any unfulfilled acceptance criteria. Parent performs an independent final acceptance audit against spec and both commits before reporting completion.

## Risks and acceptance criteria

- Riva-only installation cannot satisfy the generic prompt contract: require a separate compatible instruction server, never discard context to make its API fit.
- Small local models may mishandle numbering or editing: existing parsers remain; report quality separately from transport correctness.
- Long batches can exceed context or generation limits: preserve source text, surface failures, reject known truncated responses; configure sufficient server context rather than silently dropping lines.
- Windows hardware code cannot execute in Linux: mock only environmental dependencies and exercise real translation/caller logic; reserve full device testing for Windows.
- Acceptance: all four inference consumers use local HTTP; prompt/context/glossary contracts and file mappings pass tests; startup needs no Google SDK/key; configuration and limitations are documented; untouched application subsystems remain unchanged.
