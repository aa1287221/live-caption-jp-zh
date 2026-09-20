# Ontime Riva Direct Execution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run all existing translation workflows using the installed Ontime Riva service without provisioning another model.

**Architecture:** Add a focused standard-library Riva transport/lifecycle module. Adapt existing translation callers to typed ordered results and deterministic bilingual formatting, preserving the Windows media workflow and existing artifact formats.

**Tech Stack:** Python 3, urllib, subprocess, unittest, Ontime relay, existing Whisper and Windows GUI/audio dependencies.

**Spec:** `docs/superpowers/specs/2026-09-20-riva-direct-execution-design.md`

## Global Constraints

- Default service origin: `http://127.0.0.1:8765`.
- Requests contain at most 32 strings; each string is at most 1200 characters.
- Reuse installed assets; never download models or dependencies automatically.
- New Python indentation uses tabs; new comments use English; user-visible Chinese uses Taiwan Traditional Chinese.
- Retain `local_llm.py`; no file deletion, PR creation, or remote shared-system changes.
- User accepted loss of previous-sentence semantic context, ASR text correction by the translator, and editorial polishing.
- Main session owns progress tracking. Workers do not create/update task-tool entries. Each implementation task uses a fresh worker, followed by main review and a code-reviewer/Python review gate.

## Task 1: Riva transport, glossary, and startup lifecycle

**Files:**
- Create `ontime_riva.py`.
- Create `tests/test_ontime_riva.py`.
- Read `/project/Ontime-Translator/tools/relay/{translate_api,nmt,nmt_text,server,relay_cli}.py` and `/project/Ontime-Translator/{start-relay.sh,tools/relay/run.sh}`.
- Preserve `local_llm.py` and `tests/test_local_llm.py`.

**Interfaces produced:**

```python
class RivaError(RuntimeError):
	def __init__(self, message: str, *, retryable: bool = False):
		super().__init__(message)
		self.retryable = retryable

class OntimeRivaClient:
	def __init__(self, base_url: str | None = None, timeout: float | None = None): ...
	def ensure_ready(self) -> None: ...
	def translate(self, text: str, glossary: dict | None = None) -> str: ...
	def translate_batch(self, texts: list[str], glossary: dict | None = None) -> list[str]: ...
```

The signatures above are interface notation. `base_url` is the service origin and `timeout` is translation timeout; configuration defaults/environment fields are defined by the spec. Construction validates configuration without requiring a running service; callers explicitly invoke `ensure_ready()`.

- [ ] **Step 1: Add failing HTTP contract tests using `ThreadingHTTPServer`.** Run the real urllib path; capture request JSON and return DeepL-shaped objects. One successful fixture is:

```python
payload = {
	"translations": [{
		"text": "你好。",
		"ontime": {"verdict": "ok", "reasons": [], "guard": {}},
	}],
}
self.assertEqual(client.translate("こんにちは。"), "你好。")
self.assertEqual(request_body["text"], ["こんにちは。"])
self.assertEqual(request_body["source_lang"], "JA")
self.assertEqual(request_body["target_lang"], "zh-TW")
```

Cover empty input without HTTP; 33 inputs generating 32+1 requests; response cardinality mismatch; malformed JSON; empty/missing text; HTTP 422; HTTP 200 with one rejected item; warning with `input:truncated`; 429/503 retryable errors; explicit timeouts; disabled proxies and redirects. Verify no automatic retry by counting server requests.

- [ ] **Step 2: Add lifecycle and glossary failures before implementation.** Mock platform/process calls and `/health`; readiness requires all three NMT fields. Ready service must not spawn; starting service must wait; disabled/failed/malformed service must fail without spawn; absent service launches once and waits until deadline. Test Windows argv and Linux argv, custom port/path including spaces, remote URL never auto-launches, missing launcher, process failure, deadline timeout, and bounded log-tail diagnostics. Test >1200-character Japanese input, unpunctuated text, split-marker integrity, repeated glossary terms, overlapping terms, literal marker collision, lost marker, and duplicate marker.

```python
ready = {"nmt": {"loaded": True, "state": "ready", "error": None}}
not_ready = {"nmt": {"loaded": True, "state": "starting", "error": None}}
windows_command = [
	"wsl.exe", "-d", "Ubuntu-26.04", "-e", "bash",
	"/project/Ontime-Translator/start-relay.sh", "--no-preload", "--port", "8765",
]
```

- [ ] **Step 3: Run new tests red.** `python3 -m unittest discover -s tests -p 'test_ontime_riva.py' -v`; expect missing module/API or assertions exposing the absent behavior.

- [ ] **Step 4: Implement transport and glossary.** Validate origin, finite positive timeouts, JSON structure, item count, string types, and item verdict metadata. Send `source_lang="JA"` and `target_lang="zh-TW"`, confirmed against source and a live probe. Reject truncation reasons even when verdict is `warn`. Accept other warnings with an explicit diagnostic. Preserve per-item order and report original input index on failure. Segment at Japanese sentence delimiters, fall back to bounded character chunks, flatten in order, send groups of 32, then reassemble by original index. Mask glossary longest-first using generated markers; budget masked length, never cut a marker, and compare exact expected/restored marker counts before returning. Do not modify raw Japanese strings used by callers.

```python
for index, item in enumerate(body["translations"]):
	meta = item["ontime"]
	if meta["verdict"] == "reject" or "input:truncated" in meta["reasons"]:
		raise RivaError(f"第 {index + 1} 句被拒絕或截斷：{meta['reasons']}")
```

- [ ] **Step 5: Implement lifecycle.** Probe before launching; only a connection-unavailable local service permits auto-start. Probe failures from an existing HTTP server must not be treated as an absent port. Use subprocess argument arrays and detached process settings appropriate to the platform, redirect stdout/stderr to a durable application log, and never terminate relay on client exit. On Windows, open that log in this repository with Python and pass the file handle as `Popen` stdout/stderr alongside direct WSL argv. Poll with a monotonic deadline; startup/translation readiness errors include URL, stage, log path, and a bounded tail. Read only NMT readiness, because `--no-preload` intentionally avoids preloading ASR. Respect `ONTIME_AUTO_START=0` and remote URL settings.

- [ ] **Step 6: Run the new suite and preserved transport suite.** `python3 -W error::ResourceWarning -m unittest discover -s tests -p 'test_ontime_riva.py' -v` and `python3 -m unittest discover -s tests -p 'test_local_llm.py' -v`. Close server sockets, response handles, and log handles. Main session reviews diff before the integration task.

- [ ] **Step 7: Commit scoped files after review.** `git add ontime_riva.py tests/test_ontime_riva.py` then `git commit -m "feat: connect and start existing Ontime Riva translation service"`.

## Task 2: Integrate all workflows and document direct execution

**Files:**
- Modify `live_caption_gemini.py`: import/configuration comments, `Translator`, `rebuild_transcript_from_full_audio`, and related user-facing help text.
- Modify `transcribe_audio_file.py`: batch translation and failure handling.
- Modify `tests/test_local_translation_integration.py`.
- Modify `README.md`; modify `install.bat` only if it still claims an instruction-model prerequisite.

**Interfaces consumed:** `OntimeRivaClient`, `RivaError.retryable`, `ensure_ready()`, `translate()`, `translate_batch()` from Task 1.

**Interfaces produced:** existing `Translator.translate(text, glossary=None)` and `translate_with_context(prev_ja, current_ja, glossary=None)` continue returning strings; `Translator.translate_batch(texts, glossary=None)` returns ordered strings with explicit source-preserving failure markers after caller retry policy. The lower-level Riva client raises `RivaError`; the application layer handles visible fallback.

- [ ] **Step 1: Adapt failing integration tests to the approved backend.** Mock only media/Whisper dependencies and the Riva client; preserve tests for cue timestamps, filenames, transcript layout, and empty input. Confirm startup calls `ensure_ready()` exactly once per Translator, playback-only cue mode does not initialize translation, and context compatibility sends only the current source sentence.

```python
translator.client.translate.return_value = "請多指教。"
self.assertEqual(translator.translate_with_context("前一句", "よろしく。"), "請多指教。")
translator.client.translate.assert_called_once_with("よろしく。", None)
```

Add 33+ segment tests proving a 32-line batch followed by the remainder, exact JA/ZH pairing, unchanged source text/start/end times, and source-visible fallback instead of silent missing lines when one batch fails. Retryable errors get at most three attempts; permanent guard/glossary failures fall back immediately. Mock `Path.unlink` to preserve test assets while verifying existing successful-reconstruction cleanup is still called.

- [ ] **Step 2: Run integration tests red.** `python3 -m unittest discover -s tests -p 'test_local_translation_integration.py' -v`; expected failures refer to the old LocalChatClient/prompt workflow.

- [ ] **Step 3: Replace Translator implementation.** Constructor instantiates Riva and runs readiness before starting translation-dependent capture/ASR work. Single-sentence wrapper logs `RivaError` and returns `（翻譯失敗，保留日文原文）{source}` without retry or sleep. Batch wrapper makes at most three attempts for retryable failures, then returns one explicit source-preserving fallback per failed input. Permanent guard/glossary failures use fallback immediately. Keep `translate_with_context` as a documented current-sentence compatibility wrapper; bypass empty source without HTTP. Retire `_chat` call sites without deleting the old transport file.

```python
def translate_with_context(self, prev_ja: str, current_ja: str, glossary: dict | None = None) -> str:
	return self.translate(current_ja, glossary)

def translate_batch(self, texts: list[str], glossary: dict | None = None) -> list[str]:
	for attempt in range(3):
		try:
			return self.client.translate_batch(texts, glossary)
		except RivaError as exc:
			print(f"翻譯失敗：{exc}")
			if not exc.retryable or attempt == 2:
				return [f"（翻譯失敗，保留日文原文）{text}" for text in texts]
			time.sleep(20)
```

- [ ] **Step 4: Replace both offline prompt pipelines.** Set `_POLISH_CHUNK_LINES` and `_CHUNK_LINES` to 32. Translate raw Japanese arrays directly; deterministic full-audio rendering uses paired lines, while cue conversion assigns each translated string to its original cue. Retry only in the Translator batch wrapper; do not stack retries in its callers or sleep after the final failure. Log any source-preserving fallback as degraded output. Preserve existing successful-reconstruction `wav_path.unlink()` behavior and mock it in tests. Preserve the `_notebooklm_style.md` filename for compatibility but describe it as retranscribed bilingual output, without promises of correction or editorial quality.

```python
translations = translator.translate_batch(batch, glossary)
rendered = "\n\n".join(f"{ja}\n{zh}" for ja, zh in zip(batch, translations))
```

- [ ] **Step 5: Update operator instructions and claims.** README documents defaults, environment fields, Windows WSL distro/path overrides, direct launcher command, persisted service/log lifecycle, troubleshooting occupied port/model readiness, standalone client smoke test, and the accepted semantic differences. Explain original Whisper Windows installation remains required for capture; Riva is already installed and is auto-started. Remove active instructions requiring a chat model, API key, or `/v1/chat/completions`. Explain existing recording cleanup and the exact source-preserving failure marker.

```bash
python3 -c 'from ontime_riva import OntimeRivaClient; c=OntimeRivaClient(); c.ensure_ready(); print(c.translate("今日はいい天気ですね。"))'
```

- [ ] **Step 6: Run full tests and review.** `python3 -W error::ResourceWarning -m unittest discover -s tests -v`; `python3 -m py_compile ontime_riva.py live_caption_gemini.py transcribe_audio_file.py`; `git diff --check`. Review active source/docs for chat-backend dependencies and unsupported claims. Independently review code quality and Python behavior before committing the task.

- [ ] **Step 7: Commit scoped integration changes.** `git add live_caption_gemini.py transcribe_audio_file.py tests/test_local_translation_integration.py README.md` then `git commit -m "feat: run caption and offline translation directly with Riva"`. Include `install.bat` only if changed.

## Independent validation and delivery gate

- [ ] A fresh independent verifier reads the spec and user workflow, then checks client protocol, process reuse/failure, 33+ source lines, source-visible failures, unchanged recording cleanup with mocked deletion, glossary integrity, and raw Japanese/timestamp preservation.
- [ ] Main session starts/reuses installed Riva through the new client. Record health NMT fields, actual model identifier, requests and results for ordinary Japanese, repeated glossary names, and a multi-input batch. Include a >1200-character input to confirm reassembly or explicitly record model guard rejection; never call rejected output success.
- [ ] If markers fail on the real model, fix the marker strategy and rerun both marker integrity tests and live glossary probes before delivery. Do not drop glossary protection or silently accept loss.
- [ ] Record real elapsed times only when measured; do not make unmeasured performance claims. Persist concise verification evidence in the plan or delivery report.
- [ ] A read-only final audit checks the spec against final source and commits. Report Windows capture/playback verification separately from Linux client tests. Do not claim a live audio end-to-end run without executing that workflow.
- [ ] Main session checks branch, diff stat, and commit log; no PR. Prior upstream push was rejected for permissions, so local commits are sufficient unless an authorized writable destination is already available.
