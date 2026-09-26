# Managed local llama-server — design

Date: 2026-09-26
Status: approved by the repo collaborator (session request: "整合進整個 repo 做自動執行")

## Contract change

Previous contract (`2026-09-20-local-translation-design.md`): the local service is external;
the app never starts or stops it.

New contract: **when the local backend is selected and nothing answers at `LOCAL_LLM_BASE_URL`,
the app starts a user-configured `llama-server.exe`, waits for it through the existing readiness
path, and stops it on exit — only if the app itself started it. The app never downloads
anything.** Downloading the server binary and a model is a separate, explicit script
(`setup_local_llm.py`). The "never touch Ontime / other services" half of the old contract still
holds: an already-running server (Ontime, Ollama, WSL, user-started) is used as-is and never
stopped.

## Why

A caption session with the local backend needs a manually started server every time, and after a
reboot it is simply gone. Measured on the target machine (RTX 5080 16 GB, 64 GB RAM), the chosen
setup (Gemma 4 26B-A4B QAT q4_0 + Whisper large-v3) peaks at 12.4 GB VRAM and translates at
~33 tok/s, so running the server as a child of the app is practical.

## Configuration

Resolution order for every key: environment variable → `local_llm_server.json` (program folder,
gitignored) → default. The JSON file exists so the double-click exe works without env vars (same
reason `gemini_api_key.txt` exists).

| Key (JSON) | Env var | Default | Meaning |
|---|---|---|---|
| `server_exe` | `LOCAL_LLM_SERVER_EXE` | `tools/llama/llama-server.exe` if it exists | llama-server binary |
| `model_path` | `LOCAL_LLM_MODEL_PATH` | none | GGUF to load; **no model → auto-start disabled** |
| `server_args` | `LOCAL_LLM_SERVER_ARGS` | `-c 8192 -np 1 -ngl 99 --jinja --reasoning off` | extra args; string (shlex/Windows split) or JSON list |
| `startup_wait` | `LOCAL_LLM_SPAWN_WAIT` | `300` | seconds to wait for a server the app spawned |

Host and port are derived from `LOCAL_LLM_BASE_URL` (default `http://127.0.0.1:8766`) and passed
as `--host/--port`; auto-start is refused for non-loopback base URLs. Relative paths resolve
against the program folder. Machine-specific tuning such as `--n-cpu-moe 26` is user-supplied via
`server_args`, never a default.

## Lifecycle rules

1. Local backend not selected → nothing happens.
2. Health probe to the base URL succeeds (or reports "loading", 503) → use it, do not spawn,
   never stop it.
3. Unreachable and auto-start not configured (no model path) → today's behaviour and error
   message, unchanged, plus one line saying auto-start is available (see README).
4. Unreachable and configured → validate exe and model exist (clear Chinese error naming the
   missing path otherwise), spawn with stdout/stderr to `logs/llama-server.log`, then go through
   the existing `ensure_ready`/`on_wait` path with `startup_wait` as the budget.
5. Spawned child exits before becoming ready → fail fast with the exit code and the last lines of
   the log, do not keep waiting for the full budget.
6. App exit (normal, exception, window close) → terminate the child if and only if the app
   spawned it; wait briefly, then kill. On Windows the child is also put in a Job Object with
   `KILL_ON_JOB_CLOSE` so a crashed app cannot leave an orphan holding VRAM.
7. Readiness-wait messages reuse the existing log lines ("等待本機模型服務就緒…").

`local_llm.py` and the new module stay standard-library only (`subprocess`, `ctypes`, `shlex`,
`json`, `atexit`).

## Integration points

- New module `llama_server.py`: config resolution + `ManagedLlamaServer` (start / wait / stop).
- `Translator.__init__` local branch (`live_caption_gemini.py`): ensure the server before
  `LocalTranslationEngine.prepare`; register the stop. `transcribe_audio_file.py` and
  `compare_translation_backends.py` get the same behaviour through `Translator`.
- GUI label for the local backend: drop "需先自行啟動"; say it starts automatically when set up.
- Whisper model becomes configurable: env `WHISPER_MODEL`, CUDA default changes from
  `large-v3-turbo` to `large-v3` (measured: turbo hallucinated repeated fragments and stock
  phrases on a noisy stream clip where large-v3 did not; p50 0.32 s per utterance), CPU default
  stays `small`.
- `glossary.json` becomes untracked (gitignored). `load_glossary()` already regenerates it from
  `_DEFAULT_GLOSSARY` when missing, so behaviour is unchanged for a fresh clone, and a user's own
  list no longer conflicts with `git pull`.

## setup_local_llm.py

Explicit, run once by the user. Downloads a pinned llama.cpp Windows CUDA release (server zip +
cudart zip, CUDA 13.4 by default, `--cuda 12.4` option) into `tools/llama/`, downloads the default
model (`google/gemma-4-26B-A4B-it-qat-q4_0-gguf`, file `gemma-4-26B_q4_0-it.gguf`) into `models/`,
verifies sha256 for everything it downloads, skips files that already verify, and writes
`local_llm_server.json`. Options: `--model-url/--model-sha256` for a different model,
`--server-args` to store tuning. No download happens anywhere else.

## Tests (offline, red first)

Fake server = small Python script started as the "exe" that serves `/health` on the given port
(optionally 503 for N polls, or exits immediately). Paths to cover:

| # | Situation | Expected |
|---|---|---|
| 1 | server already reachable | no spawn, not stopped on exit |
| 2 | unreachable, no model configured | unchanged error + auto-start hint, no spawn |
| 3 | unreachable, exe path missing | error names the exe path, no spawn |
| 4 | unreachable, model path missing | error names the model path, no spawn |
| 5 | configured, spawn → 503 → ready | ready, host/port/model/args passed correctly |
| 6 | spawned child exits early | fails fast with exit code + log tail, well before the budget |
| 7 | stop after spawn | child terminated; stop is idempotent |
| 8 | config precedence | env beats JSON beats default; relative paths resolve to program folder |
| 9 | non-loopback base URL | auto-start refused with a clear message |
| 10 | `server_args` as string and as list | both produce the same argv |

Pre-existing Windows-only failure `test_unreachable_server_is_reported_without_waiting_when_no_budget`
(`'timeout' != 'connection'`) is out of scope and must stay visible, not masked.

## Out of scope

Auto-downloading from inside the app, managing non-llama.cpp servers, GPU/VRAM auto-tuning.
