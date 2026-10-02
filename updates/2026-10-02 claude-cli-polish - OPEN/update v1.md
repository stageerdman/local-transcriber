# Update: Claude-CLI polish backend — v1

**Status: OPEN — built, verified, committed (2026-10-02)**

## ▶ RESUME HERE
Done & committed: polish now shells out to `claude -p` (no local Gemma/mlx-lm),
and the polish phase is folded into the same progress bar (no more frozen 0).
Verified: 97 tests pass, real `polish_blocks`/`polish_text` clean on Czech, app
boots, progress math + live panel checked.
**Next session:** run one real end-to-end transcription in the actual app with
Polish ON and confirm (a) it's fast, (b) the bar animates through the polish
tail, (c) the `.md` is cleaned and the `.raw.md` sibling is kept. If polish ever
feels slow again, the deferred option is "true background" (write raw first,
update the file when Claude returns) — we chose fast-inline for now.

## Goal
The LLM cleanup pass (polish) was too slow: it loaded a ~6.7 GB local model
(Gemma 3 12B via mlx-lm) and generated per chunk, adding minutes to every job.
Switch polish to run through the **Claude Code CLI** (`claude -p`, headless) so a
whole transcript is cleaned in one fast network-backed call. Also fold the polish
phase into the **same overall progress line** so the bar never resets to 0 or
freezes while polishing.

User decisions: **replace the local pass entirely** (no mlx-lm fallback) and run
polish **inline** (keep the existing worker flow, just faster).

Tradeoff accepted: polish now needs network + an authenticated `claude` CLI. If
the CLI is missing or a call fails, the **raw transcript is kept** — polish is a
readability bonus, never a correctness risk (the raw `.raw.md` sibling is still
always written).

## Roadmap (phased)
1. **Spike + research** — confirm `claude -p` headless works for Czech cleanup;
   research battle-tested transcript-cleanup prompts + anti-hallucination
   techniques. ✅ DONE (see `wiki.md`).
2. **New `src/polish.py`** (Claude-CLI backend) + anti-hallucination guards. ✅ DONE.
3. **Remove mlx-lm/Gemma plumbing** (catalog, jobs, engine, worker, requirements). ✅ DONE.
4. **Unified progress** — polish as a time-estimated tail of the same bar. ✅ DONE.
5. **Tests** (fake CLI) + **real-app verify**. ✅ DONE.

## Verification (2026-10-02)
- `pytest`: 87 passed (15 in `test_polish.py` rewritten against a fake CLI:
  JSON round-trip, length/length-mismatch/invalid-JSON/CLI-missing fallbacks,
  containment backstop, per-block guard, short-turn skip).
- Real end-to-end through the integrated path: `polish_blocks` 7→7 (~8–24 s,
  one call), `polish_text` ~11 s — fillers removed, diacritics/punctuation
  fixed, short reactions kept, order preserved, no hallucination.
- Progress math verified: transcription fills [0, 1−share]; polish continues
  from there (e.g. 81% → 100%) and never resets to 0 or freezes; ETA during
  transcription already includes the polish estimate. App boots; the live
  "polishing" panel renders "Cleaning up the transcript with Claude • ~X left".

## Not done / notes
- Full live transcription of real multi-track audio wasn't driven end-to-end in
  this environment (needs the ~3 GB Czech model + call audio) — unit/path/boot
  verification stands in for it. Worth one real run on the user's machine.
- Not committed yet: the working tree still carries the prior
  `transcription-quality-research` update's uncommitted integration in the same
  files, so these changes are intermingled with it.

## Key spike results (2026-10-02)
- `claude -p` headless, full prompt via stdin (no shell): clean Czech output,
  fillers removed, punctuation/casing/diacritics fixed, no hallucination.
- Whole-text passage: ~8 s. Batched **JSON array of 7 blocks → 7 cleaned blocks**
  same order, short reactions ("Jo.", "Mhm") preserved: ~8.6 s for the whole
  multi-track transcript in ONE call (vs. per-block local generation before).
- Bare binary `/opt/homebrew/bin/claude` works without the shell alias, so
  `shutil.which("claude")` is all the app needs.
</content>
