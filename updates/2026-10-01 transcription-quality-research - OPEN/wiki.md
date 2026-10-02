# Wiki — Transcription Quality Research (durable findings)

## 1. Research landscape (2025–2026), for M3 / 16 GB, Czech

### Local ASR for Czech — accuracy ranking
Source-backed WER (test set in parens; only comparable within a set):
- **Canary-1b-v2** (NVIDIA) — 7.86% FLEURS-cs — best general, but runs via NeMo
  (PyTorch/MPS), heavier to set up on Apple Silicon.
- **mikr/whisper-large-v3-czech-cv13** (Czech finetune) — 7.89% CommonVoice-13 —
  best *practical* Czech model; runs on Whisper runtimes. (3.2 GB fp16.)
- **Parakeet TDT 0.6b v3** (NVIDIA, multilingual v3 incl. Czech) — 11.01%
  FLEURS-cs — ≈ vanilla large-v3, but **no silence-hallucination** (CTC/TDT, not
  autoregressive) and dramatically faster; ~2 GB RAM. Weaker punctuation/casing,
  worse on CS↔EN code-switching.
- **Whisper large-v3** — 11.33% FLEURS-cs. **large-v3-turbo** slightly worse but
  faster (pruned decoder).
- distil-whisper / Canary-Qwen / Parakeet v2 = **English-only** (not for Czech).
  Voxtral = Czech not officially supported.

### Runtimes on Apple Silicon (speed)
mlx-whisper (1.02, fastest Whisper) ≈ insanely-fast-whisper (1.13) > whisper.cpp
(1.23) >> faster-whisper/CTranslate2 (6.96 — **CPU-only on Mac**, no Metal).
Parakeet via parakeet-mlx ≈ 20–50× realtime; Parakeet CoreML/ANE even faster
(measured here at **127× realtime** on the clean track). RAM: large-v3 ~10 GB
headroom needed, medium ~5 GB, Parakeet ~2 GB.

### Pipeline best practices (adopted/decided)
- `condition_on_previous_text=False` — biggest single anti-hallucination flag.
- VAD (Silero) pre-filter — removes silence Whisper hallucinates on; faster-whisper
  has it built in (`vad_filter=True`).
- Short temperature ladder `(0.0, 0.2, 0.4)` — drop the high-temp retries that
  invent text. Keep default compression/logprob/no_speech thresholds.
- **Keep per-track transcription; do NOT add diarization** — separate tracks give
  0% attribution error; diarizing a mix would *regress*.
- `initial_prompt` for Czech domain terms — **rejected**: it leaked into output as
  hallucinated content in our test. Skip, or keep extremely minimal.
- Forced alignment / pyannote diarization — **skip** for this project.

### Local LLM polishing for Czech
- Best fit M3/16 GB via **mlx-lm**: **Gemma 3 12B (4-bit, QAT)** primary (strong
  multilingual Czech, ~6.7 GB), Mistral-Nemo 12B / Qwen3 8B alternatives.
- Conservative "correct-only, never add/summarize" prompt, temperature 0, chunked,
  with a length-ratio tripwire. (`spike/polish_transcript.py`.)
- `kredor/punctuate-all` (XLM-R, incl. Czech) is a deterministic punctuation-only
  fallback but can't remove fillers / fix ASR errors — not a full substitute.

## 2. Experimental findings (this project's own audio)

### Objective metrics (tomas call, 7-min clips) — `rep`=repeated-5gram frac, `comp`=compression ratio
Clean remote track (`spk2`):
| config | words | rep | comp | RTF |
|---|---|---|---|---|
| small_cs_default | 948 | 0.003 | 2.09 | 0.04 |
| medium_cs_tuned | 688 | 0.064 | 2.12 | 0.09 |
| large-v3 tuned | 885 | 0.001 | 2.04 | 0.24 |
| turbo tuned | 862 | 0.005 | 2.03 | 0.14 |
| parakeet mlx | 735 | 0.001 | 1.96 | 0.04 |
| parakeet coreml | 904 | 0.001 | 2.10 | **0.01 (127×)** |

Isolated local track (`spk1`, subtraction residual):
| config | words | rep | comp | note |
|---|---|---|---|---|
| small_cs_default | 1211 | **0.84** | 9.78 | catastrophic loop |
| medium_cs_default | 418 | **0.85** | 9.83 | catastrophic loop |
| medium_cs_tuned | 528 | **0.70** | 4.83 | still looping |
| large-v3 tuned | 205 | 0.16 | 2.06 | "..." for silence + some boilerplate |
| turbo tuned | 268 | 0.13 | 2.38 | similar |
| parakeet mlx | 137 | 0.008 | 1.61 | **clean, no halluc** |
| parakeet coreml | 187 | 0.000 | 1.68 | **clean, no halluc** |

### LLM-judge verdicts (native-Czech, reference-free, anonymized)
**Clean track:** large-v3-turbo (tuned) is best — acc 8 / fluency 9 / completeness
9; nails hard idioms + code-switching others miss (`v uvozovkách silnýho chlapa`).
large-v3 close 2nd. **Parakeet mediocre on clean** (6/6: garbles, even inverts a
meaning). medium/small poor→unusable (loops). The large-v3 **+prompt** variant was
*worse* than plain (prompt leaked as fake summary).

**Isolated track:** **Parakeet best** (CoreML then MLX) — hallucination-freedom
10/10, captures the real (short) speech coherently. Whisper large-v3/turbo
borderline — fluent where real, but inject `Titulky vytvořil JohnyX.`×7 and
`Ahoj.`×14. medium/small/medium-tuned unusable (loops like `si si si…`×150,
`že se vám závodí`×hundreds).

**Conclusion:** different engines win different track types →
**per-role engine selection** (Whisper for clean `keep`, Parakeet for `isolate`).

### LLM polish (Gemma 3 12B QAT 4-bit, clean large-v3 transcript)
Length ratios 0.95–0.99 (no inflation). Produced cased, punctuated, paragraphed
Czech; removed fillers; **kept** ASR errors it couldn't safely resolve
(`pro řidiče`, `Učitelům`) rather than inventing — the desired conservative
behavior. Big readability win; cannot recover true ASR word errors (so ASR
accuracy still dominates final quality).

## 3. Decisions
- Default app model was `whisper-small`, which **loops catastrophically** on the
  isolate-role residual tracks → must change.
- No content-ful `initial_prompt`.
- Polish is a net positive and safe when prompted conservatively + tripwired.

## 4. Resolution — the winner (final A/B judges)
Focused 4-way blind A/B (Czech-finetune+VAD vs turbo vs large-v3 vs Parakeet):
- **Clean track:** Czech-finetune+VAD is the clear #1 (complete, loop-free, most
  faithful on hard spots). Parakeet #2 (clean but garbles words); plain
  large-v3/turbo show occasional loops/dropped passages/hallucinated questions.
- **Isolated track:** Czech-finetune+VAD #1 (beats Parakeet on completeness +
  diacritics); plain large-v3/turbo WITHOUT VAD are unusable (boilerplate loops).

→ **Single engine wins both: Czech-finetuned large-v3 + Silero VAD.** Confirmed it
runs fast enough once converted to **MLX**: measured **0.24x RTF on the clean
track, 0.07x on the isolated track (silence skipped)** on M3 - and clean (no
`titulky`/`www.` hallucination) on both. Parakeet kept as the selectable fast
mode.

## 5. How it was built (integration)
- **Model**: `mikr/whisper-large-v3-czech-cv13` converted to MLX via
  `scripts/convert_czech_model.py` → `~/.cache/local-transcriber/models/...`.
  Catalog id `czech-large-v3-mlx` (`src.transcription.CZECH_FT_ID`); new default.
- **VAD**: Silero (from faster-whisper's bundled `get_speech_timestamps`, no new
  model download) in `transcription._transcribe_whisper_vad` - groups speech into
  ≤28 s chunks, transcribes each with tuned params, offsets timestamps back to the
  original timeline (so dialogue blocking still works). Whisper-only; Parakeet
  needs none.
- **Tuned decode** (`WHISPER_DECODE_OPTS`): `condition_on_previous_text=False`,
  temperature `(0.0,0.2,0.4)`, `no_speech_threshold=0.6`, NO initial_prompt.
- **Polish**: `src/polish.py` (Gemma 3 12B QAT 4-bit via mlx-lm), conservative
  Czech prompt, temp 0, per-chunk length tripwire → falls back to raw on
  ballooning output. Runs in the engine subprocess after ASR (models not
  co-resident). Optional toggle, default ON; raw transcript always kept as a
  `.raw.md` sibling.
- **Measured LLM-polish safety**: length ratios 0.78-0.99 across tracks; keeps
  ASR errors it can't safely fix instead of inventing.
