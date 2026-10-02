# Update: Transcription Quality Research — v1

**Status: OPEN — build complete, pending real-app verify + commit (2026-10-01)**

## Goal
Find the best *free, local, reasonably-fast* transcription setup for **Czech**
sales calls — judged on the **overall result** (not just the raw ASR engine, but
the final readable transcript, optionally LLM-polished). Research the landscape,
design tests on real call audio, rate the output quality, and improve the engine.

Hardware ceiling: **Apple M3, 16 GB RAM, ~14 GB free disk.** Everything must run
offline and free.

## Test data
7-minute clips taken from the middle of real Czech sales calls in
`My Call Analysis/`, reduced to the tracks we care about via the app's own
`track_separation` (per-speaker). Set:
- `tomas` (3-track .mov) → clean remote speaker + isolated local speaker
- `marek` (3-track .mov) → same
- `vaclav` ("difficult", 3-track .mov) → same
- `roleplay` (1-track .mp4) → single mixed mono

Clean track = separation role `keep`; isolated track = role `isolate` (mic minus
the other voice — the subtraction residual). Audio kept in scratchpad (large);
spike scripts in `spike/`.

## Roadmap (phased)
1. **Research** — landscape of local ASR for Czech + local LLM polishing +
   pipeline best practices. ✅ DONE (see `wiki.md`).
2. **Phase A — ASR shootout** on clean + isolated `tomas` tracks across all
   candidate engines/configs; objective metrics + LLM-judge rating. ✅ DONE.
3. **Rating** — two native-Czech LLM judges (clean track, isolated track),
   reference-free, anonymized candidates. ✅ DONE.
4. **Czech-finetune + VAD** experiment (faster-whisper + Silero). ⏳ IN PROGRESS.
5. **Phase B** — confirm winner; settle the isolated-track architecture. ✅ DONE
   (VAD + Czech finetune wins both track types; see `wiki.md` §4).
6. **Phase C** — LLM polishing (Gemma 3 12B via mlx-lm). ✅ DONE (clean win, no
   hallucination; integrated as an optional default-ON pass).
7. **Phase D** — integrate winning engine (+ optional polish) into `src/` + app,
   tests, verify in the real app. ✅ DONE — see "Integration delivered" below.

## Integration delivered
- `src/transcription.py`: Czech-finetune id resolver + Silero-VAD-chunked Whisper
  with tuned decode params and timestamp remapping (`use_vad`, default on).
- `src/polish.py`: Gemma 3 12B conservative Czech cleanup (temp 0, length
  tripwire). `src/catalog.py`: new default model/language + polish constants.
- `app/engine.py`: transcribe/polish request dispatch. `app/worker.py`: polish
  pass + always keeps the raw transcript. `src/md_writer.py`: `.raw.md` sibling +
  `polished` frontmatter flag. `app/ui.py`: per-file "Polish (LLM cleanup)" toggle.
- `scripts/convert_czech_model.py` (+ vendored `mlx_whisper_convert.py`): one-time
  Czech→MLX conversion. `requirements.txt`: +mlx-lm, +faster-whisper.
- Tests: `tests/test_transcription_vad.py`, `tests/test_polish.py`, extended
  `tests/test_md_writer.py`. Full suite green (77 passed at integration time).

## Key results so far (see `wiki.md` for detail + quotes)
- **Whisper wins clean tracks, Parakeet wins isolated/residual tracks.** Clean:
  large-v3-turbo (tuned) > large-v3 >> Parakeet (Parakeet garbles clean speech).
  Isolated: Parakeet (zero hallucination) >> Whisper (injects boilerplate) >>
  small/medium (catastrophic loops, unusable).
- **The current pipeline's #1 quality bug**: transcribing the subtraction
  residual (`isolate` role) with Whisper small/medium produces catastrophic
  looping/hallucination (`rep` 0.70–0.85). The app default is `whisper-small`.
- **Tuned decode params matter**: `condition_on_previous_text=False` + short
  temperature ladder cut looping; but on the residual track they don't fully
  fix Whisper — only a large model or Parakeet does.
- **The Czech `initial_prompt` BACKFIRED** — its text leaked into the output as a
  fabricated opening summary. Do not use a content-ful initial_prompt.
- **LLM polish (Gemma 3 12B QAT 4-bit) is a clear win**: converts raw ASR lines
  into cased, punctuated, paragraphed Czech, length ratio ~0.95–0.99 (no
  hallucination), and conservatively keeps ASR errors it can't safely fix.

## Final recommendation (shipped)
**Czech-finetuned Whisper large-v3 (MLX) + Silero VAD + tuned decode + optional
Gemma 3 12B polish.** It won both track types in blind judging, so no hybrid is
needed. Parakeet TDT v3 kept as a selectable fast mode. Polish is an optional,
default-ON cleanup that never discards the raw transcript. (User decisions:
offer both engines / polish optional default-on / full build.)
