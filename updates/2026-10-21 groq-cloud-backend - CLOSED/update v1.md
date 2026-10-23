# Groq cloud backend — update v1 (CLOSED)

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

5. **Verify** (done) — launched the real app, confirmed a correct `.md`
   transcript with the same shape as the local backends produce.

6. **Second Cloud backend: ElevenLabs Scribe v2** (done) — Groq's Czech
   accuracy turned out mediocre in real use (see chat: hallucinated
   fansub-credit boilerplate, "Titulky vytvořil..."). A cloud-model-research
   spike (this session) found ElevenLabs Scribe v2 scores 3-5% WER on
   Czech (FLEURS/Common Voice) vs. Whisper-class scoring far worse. Added
   as a second Cloud provider *alongside* Groq, not instead of it (user's
   call — cost/speed vs. accuracy stay both one click away):
   - `src/transcription.py`: `elevenlabs:` model-id prefix,
     `_transcribe_with_elevenlabs`, and `_coalesce_elevenlabs_words` (Scribe
     returns only a flat per-word stream, not Whisper-shaped segments — see
     `wiki.md`). Unit-tested in `tests/test_transcription_elevenlabs.py`.
   - `src/catalog.py` / `app/bridge.py`: Cloud now resolves to one of two
     models via an independently-persisted `cloud_provider` setting
     (`groq` default), decoupled from the Cloud-is-opt-in gate on `quality`.
   - `app/web/index.html`: a small `.qsub` provider toggle (Groq /
     ElevenLabs), spec'd by a dedicated UX pass, visible only when Cloud is
     selected; `#qeta`'s "sends audio online" disclosure now covers both
     providers unconditionally.
   - `testing/models_config.py`: mirrored `_make_groq_backend` with
     `_make_elevenlabs_backend` for head-to-head benchmarking.
   - Needs `ELEVENLABS_API_KEY` in `.env` - added by the user and live-verified
     this session (see below).

7. **Live verification of ElevenLabs** (done) — real `ELEVENLABS_API_KEY`
   added to `.env`; confirmed `.env` loading picks it up in-process. Ran
   `_transcribe_with_elevenlabs` directly against a real 60s Czech/Slovak call
   sample (`testing/videos/martin_track3_60s.mp3`): one HTTP call completed in
   ~7.5s, returned 15 word-coalesced segments with correct sentence/pause
   splitting and accurate Czech/Slovak text (no hallucinated boilerplate).
   `elevenlabs` SDK's `speech_to_text.convert` signature and
   `SpeechToTextChunkResponseModel.words` shape confirmed to match the code's
   assumptions. App launched standalone (`python -m app.main`) with no
   startup errors; GUI click-through (Cloud → ElevenLabs provider toggle →
   real file) left for the user to confirm visually, since this environment
   has no native-window screenshot/click access.

8. **UI simplification: drop the provider toggle** (done) — the Groq/
   ElevenLabs sub-picker made the quality step five buttons wide and didn't
   fit the detect panel (user feedback: "too many buttons, they don't fit
   in"). ElevenLabs already won on Czech accuracy (see phase 6), so Cloud now
   resolves straight to it; the `.qsub` Groq/ElevenLabs row and
   `cloud_provider` setting are gone. Quality picker is now exactly three
   buttons: **Fast** (Parakeet), **Slow** (Czech-finetuned large-v3, renamed
   from "Best" to read as the tradeoff it is), **Cloud** (ElevenLabs). Groq
   support stays in `src/transcription.py` and the testing harness for
   benchmarking - only the app's UI stopped exposing it.

## Status

**CLOSED (2026-10-23).** All 8 phases done; 150 tests pass. User confirmed
the simplified three-button picker works. Groq remains available at the
engine/benchmark level, just not surfaced in the app UI.
