# Groq cloud backend — update v1

## Goal

A third transcription quality tier, **Cloud**, that runs Whisper large-v3-turbo
on Groq's hosted API instead of on-device. Motivated by a cost/speed
comparison for Czech transcription (see chat): Groq benchmarked ~7-8x faster
than this Mac's local MLX large-v3 on real Czech call audio, for pennies per
hour of audio. This app is otherwise fully offline — Cloud is the one
intentional, opt-in exception.

## Phases

1. **Benchmark harness** (done, this session) — `testing/models_config.py`
   already had Groq backends wired for benchmarking; fixed a stale
   `LOCAL_TRANSCRIBER_ROOT` path (assumed a `local-transcriber/` subfolder that
   no longer exists post-rename) that was blocking `run_benchmark.py` entirely.
   Added the local Czech-finetune as a benchmark backend too, for a fair
   three-way comparison. Confirmed via `rate_server.py` UI.

2. **Engine backend** (done) — `src/transcription.py`: new `groq:` model-id
   prefix, dispatched in `transcribe_mp3_segments` alongside the existing
   Parakeet/Whisper branches. Uses `response_format="verbose_json"` (not the
   default) since the whole dialogue/multi-track pipeline depends on
   per-segment timestamps. One model wired to the catalog: `large-v3-turbo`
   (not plain `large-v3` — it didn't show the hallucinated-repeat-loop large-v3
   produced on a quiet opening in testing).

3. **Secret loading** (done) — `app/__init__.py` now loads `.env` into
   `os.environ` at import time (no new dependency — the file is flat
   `KEY=VALUE` lines). This runs before the engine subprocess is ever spawned,
   so `GROQ_API_KEY` is inherited by it (multiprocessing's `spawn` context
   inherits the parent's env at `Process.start()`).

4. **UI** (done) — third `Cloud` button in the existing `.qseg` segmented
   control (`app/web/index.html`), spec'd by a dedicated UX pass (see
   `wiki.md`). `app/bridge.py`'s `QUALITY_TO_MODEL`/`MODEL_TO_QUALITY` extended
   to three entries; `get_settings()` never restores into `Cloud` at launch —
   it's opt-in every session.

5. **Verify** — launch the real app, pick Cloud, transcribe a real file,
   confirm a correct `.md` transcript with the same shape as the local
   backends produce.

## Status

Phases 1-4 done. Phase 5 (live verification) in progress.
