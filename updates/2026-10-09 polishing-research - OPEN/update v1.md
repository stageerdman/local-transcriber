# Update: Polishing Research — v1

**Status: OPEN — research complete; winner found, recommendation written; app
change deferred to a future update (2026-10-09)**

## TL;DR
Best polishing approach = **minimal-edit Czech prompt + moderate small-batch
(~5-turn) context + single pass** (`g_batch_small`). It reads as well as
whole-document polishing but drifts less, and **beats the current app prompt on
both readability and faithfulness at no extra cost**. No finalist fabricated any
price/claim (verified by direct read). See `wiki.md` §4 for the full verdict and
the `src/polish.py` recommendation (not applied — app untouched as asked).

## Goal
Find the transcript-polishing approach — **prompt × context-granularity ×
algorithm** — that produces the most readable Czech transcript **without
hallucination**, measured on the real Marek Pavel P2 sales call. Research the
state of the art first, design a reference-free testing harness, let AI judges
rate each variant (counting mistakes + invented content), and recommend the
winner. **Do not touch `src/`/`app/`** — this is an isolated spike.

## Test data
`data/marek_p2_raw.md` — a 126-block, 2-speaker (Obchodník/Klient), ~16-min
Czech sales call, raw `czech-large-v3-mlx` ASR output with heavy recognition
errors, Slovak-ish code-switching, filler, and inconsistent punctuation. Copied
verbatim from the user's Downloads.

## Roadmap (phased)
1. **Research SOTA** — LLM ASR post-editing + reference-free LLM-judging /
   hallucination detection. ✅ DONE (two web-research agents → `wiki.md` §1–2,
   URLs in `spike/sources.md`).
2. **Build harness** — `spike/lab.py` (parse/run/metrics), `spike/strategies.py`
   (prompt×granularity×algorithm building blocks), `spike/judge.py` (two-pass
   source-grounded judge + deterministic hard gate), `spike/run.py`
   (staged tournament + full-transcript final, cached/resumable). ✅ DONE.
3. **Tournament (excerpt)** — sweep 4 axes (granularity, prompt, algorithm,
   model), each holding the others at the evidence-predicted optimum. ✅ DONE →
   `results/scorecard.md`. Surfaced that the LLM judge is fooled by fluent
   paraphrase → deterministic drift + direct reading needed.
4. **Final** — 3 finalists + current-app baseline on the FULL 126-block
   transcript, Opus judge N=3 + direct read of the risk blocks. ✅ DONE →
   `results/scorecard_final.md`, `wiki.md` §4.
5. **Recommendation** — best approach + how it should change `src/polish.py`
   (a SEPARATE future update; this spike only recommends). ✅ DONE → `wiki.md` §4e.

## Method notes / honest limitations
- Reference-free: no gold transcript, so all faithfulness is *source-grounded*
  (polished vs raw ASR). LLM claim-counts are noisy → N-run median + spread, plus
  deterministic number/novel-word gates that don't depend on the LLM.
- **Self-preference risk**: generators and judge are both Claude (no non-Claude
  model available via `claude -p` offline). Mitigated by the deterministic gate +
  a stronger judge (Opus) than most generators (Sonnet); flagged, not solved.
- **Greedy staging** ignores cross-axis interactions (documented trade-off for
  tractability — full cartesian would be ~90 configs × N judges).
- `claude -p` exposes no temperature/seed → voting is a cross-model ensemble, and
  a reasoning on/off axis couldn't be toggled cleanly (dropped; evidence predicts
  reasoning would over-edit anyway).
- Confidence-gated correction (the one useful GER idea) needs ASR logprobs the
  finished `.md` doesn't carry → out of scope; noted for a future engine-level
  update.
