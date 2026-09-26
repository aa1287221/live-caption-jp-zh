# Selectable Gemini / local translation implementation plan

**Spec:** `docs/superpowers/specs/2026-09-26-dual-backend-gemini-parity-design.md`

## Global constraints

- Gemini behaviour stays as upstream shipped it; shared prompt builders, no prompt edits.
- The local path sends the same prompts and produces the same files; extra behaviour is additive.
- Standard-library HTTP only; the application never starts, stops or downloads a model.
- New Python blocks use tabs and English comments; user-facing text is Taiwan Traditional Chinese.

## Tasks

- [x] Merge hhggffddssqjp-coder/live-caption-jp-zh#2 (duplicate `stop()` fix, requirements).
- [x] Transport: chat and Riva modes, `/props` auto-detection, `/health` readiness with loading
      wait, error kinds for truncation/context/unreachable, per-call budget and timeout.
- [x] Restore the Gemini path; lazy SDK import; `Translator(backend)`; dialog option ⑥,
      `TRANSLATION_BACKEND`, `--backend`; translator verified before audio rerouting.
- [x] Local engine: live repair and circuit breaker; splitting batches on truncation, context
      overflow, malformed output and timeouts; numbered gap filling; wider context for smaller
      batches; Riva profile with glossary masking and segmentation.
- [x] Tests: transport contract, engine behaviour, application integration including Gemini
      regression and byte-identical prompt parity; opt-in real-server contract tests.
- [x] Comparison harness with report, JSON output and non-inferiority gate.
- [x] Remove the Ontime relay client; README for both engines, model choice and verification.

## Verification commands

```bash
python -m unittest discover -s tests
python -m py_compile *.py
LLAMA_SERVER_URL=http://127.0.0.1:8766 python -m unittest discover -s tests -p test_llama_server_live.py -v
python compare_translation_backends.py samples.txt --judge
```

Windows capture/playback and real-model translation quality must be checked on the target
machine; the harness report is the evidence for "not lower than Gemini".
