# Mic-bleed removal for multi-track recordings — update v1 (OPEN)

## Goal
Multi-track recordings often have **bleed** (a.k.a. crosstalk / spill): when
several people are recorded on separate mics in one room, each mic also picks up
the *other* people — a quieter, slightly-delayed copy of their voice. Concrete
case the user hit: **Person 2's track = Person 2 + a faint copy of Person 3**,
while Person 3's track is clean.

Today we transcribe each track independently and merge. So a bleed copy of
Person 3's words gets **transcribed twice** — once (correctly) on Person 3's
track, once as "bleed" on Person 2's track — and the bleed copy is **mislabeled**
as Person 2 speaking (or flagged as an `(interrupting)` overlap). The transcript
is polluted with ghost lines attributed to the wrong speaker.

**Goal:** detect directional bleed between time-aligned tracks and remove it
*before* transcription, so each utterance is transcribed once and attributed to
the one person who actually said it — safely, by default, and without ever
touching the user's source files.

## Feasibility (validated before committing — see `wiki.md`)
- **Tracks are streams inside one container** → inherently **time-aligned** at
  `t=0`, same duration. This is the easy case for bleed removal (we have a clean
  reference for the interfering speaker; no sync guesswork).
- **Detection is a solved problem:** directional, pairwise lagged
  cross-correlation. If track A carries a quieter, delayed copy of track B's
  signal, B bleeds into A. Gives a per-pair map + per-region confidence.
- **numpy 2.4 + scipy 1.17 are already in the venv** (transitive via
  mlx-whisper) and **ffmpeg 8.1.1** is present. So detection + gating need
  **zero new heavy dependencies** — PCM decode via ffmpeg `-f f32le` → numpy,
  everything else is `scipy.signal`. `padasip`/`adaptfilt` are **not** needed.
- Reference approach: Auphonic's multitrack "mic bleed remover" solves exactly
  this scenario for podcasts/interviews.

**Verdict: technically feasible → proceeding.**

## The one big scope decision: gating, not subtraction (v1)
Two removal strategies exist:
1. **Gating** — mute a track in the regions where another track clearly
   dominates *and* this track is confidently just a delayed copy. Cheap, and
   **can't damage the real speaker** (in those regions the track carries nothing
   true by definition).
2. **Adaptive subtraction (NLMS)** — estimate the acoustic path and subtract the
   bleed waveform. Cleaner-sounding audio, but can rarely harm the true speaker
   during simultaneous talk ("double-talk").

**We ship gating only in v1.** The deliverable is a *transcript*, not mastered
audio — we don't need clean-sounding audio, we need Whisper to not transcribe
the echo, and gating achieves that. Subtraction's only advantage (fidelity
during double-talk) buys nothing for a transcript and adds real risk +
complexity. It doesn't earn its place. (Deferred as an optional later phase
*only* if real-world testing shows gating leaves transcribable leftovers.)

## Safety model — the core of the design
Detection is fallible, and this feature **modifies audio by default**, so trust
comes from *behavior*, not a confidence number the user can't interpret. Three
tiers, decided per contaminated track:
- **Confident → auto-clean** (gate the bleed regions, on by default).
- **Uncertain → ask** (leave audio untouched, offer to clean).
- **Too much overlap to gate safely → refuse** (say so plainly; don't offer a
  control we believe would delete real speech).
- **No bleed → show nothing.** Silence is the honest signal; no "all clear"
  banner.

The **source file is never rewritten.** Gating is a mute-mask applied to the
derived audio fed to the transcriber. "Undo" = untick and re-transcribe —
genuinely reversible, zero risk.

## UX (synthesized from two UX-expert agents — full design in `wiki.md`)
- **One global toggle** at the top of the Audio-tracks panel,
  **shown only when bleed is detected**, **on by default**. No per-pair matrix,
  no strength dial (both cut — see `wiki.md`).
- **Plain-language disclosure**, reusing the panel's existing `(~Other)`
  duplicate grammar. Summary line: *"Some mics also picked up the other people
  in the room. Cleaned up automatically before transcribing."*
- **Per-track disclosure + opt-out folded into one element:** the contaminated
  row gets a muted sub-label *"also picks up Person 3"*; clicking it (or a
  right-click `Don't clean this track`) toggles cleaning for that track.
- **Auto-picked strength, never exposed.** User sees one toggle, not
  gating-vs-subtraction.
- **Honest cost feedback:** a distinct preprocessing status
  (*"Removing mic bleed from {n} tracks…"*) before the normal position-in-
  recording progress, reusing `progress_detail_var`.
- **Provenance:** the output `.md` ends with a quiet line when cleaning was
  applied, e.g. `_Mic-bleed cleanup applied to: Person 2, Room laptop._`
- **Trust surface (Phase 5):** an optional collapsed "Review" with an A/B
  listen — the cleaned preview is just a mute-mask over the *existing*
  `TrackPlayer`, so it reuses `src/playback.py` with no second audio pipeline.

One open wording call (Phase 4): toggle label **"Remove mic bleed between
tracks"** vs an even-plainer phrasing. Default to the former + subtext; worth a
quick gut-check, doesn't block engineering.

## ▶ RESUME HERE (paused 2026-09-23)

**State:** the manual-control feature is BUILT, tested (63 pass), and the app
launches clean. All of the user's explicit asks work end-to-end (show tracks,
include/exclude, name→label, per-track language, per-track RNNoise filter,
subtract one track from another). Working tree clean; everything committed &
pushed.

**Pending user request (last message, truncated):** "Improve UI, let's drop
the …" — the user wants **UI improvements** and to **drop something** (the
sentence was cut off). **First action next session: ask the user what to drop
and which UI improvements they want** before changing the panel.

**Known-open polish (designed by the UX pros, not yet built):**
- A/B **"Cleaned · original"** preview button so the user can *hear* a subtraction
  result before transcribing (cheap: `render_transcription_wav` → temp wav →
  existing `TrackPlayer`).
- One-time **"two people in one track?"** discovery hint on the panel.
- **Recent-names** suggestion list in the name entry (from `history_db`).
- **Reset** button (revert per-track names/language/noise/subtraction for a job).
- Ensure per-track drawer controls disable while a job transcribes (render uses
  `job.status == "ready"`, but the panel isn't force re-rendered on status
  change — verify/attach).

**Not yet verified:** a full real transcription through the new pipeline in the
actual app (each piece is unit-tested; a real 25-min call end-to-end hasn't been
run). Suggested test: Miroslav call → name track 3 "Miroslav", track 1 "Me", on
"Me" set Remove voice → Miroslav, uncheck the duplicate, transcribe; expect a
clean 2-speaker transcript.

**Key files:** engine `src/track_separation.py`; backend `app/worker.py`
(`_transcribe_multi_track`) + `app/jobs.py` (Job per-track fields); UI
`app/ui.py` (`_render_tracks_section`, `_build_track_detail`, and the
`_on_track_*` / `_track_*` helpers). Spikes/analysis in this folder's `spike/`.
RNNoise model: `scripts/fetch_rnnoise_model.sh` (gitignored weights).

## Phased roadmap

### Phase 0 — Detection spike (isolated, in this folder) — GATE ✅ GO
Ran in `spike/` (synth.py / detect.py / cases.py / sweep.py). Windowed
normalized cross-correlation (250 ms / 125 ms hop, ±30 ms lags) recovers
direction + lag reliably. **Verdict: GO.**
- [x] Synthetic bleed recovers true direction + lag. The user's exact 3-track
      case (P3→P2) detected correctly: only P2,P3 flagged, dir P3→P2, exact 6 ms
      lag; both clean pairs (P1,P2 / P1,P3) dismissed.
- [x] Swept α × τ × overlap. Clean pairs: **0/60 false positives** (max rho
      0.117). Zero direction errors for τ ≥ 5 ms at every α. Known safe blind
      spots: τ ≤ 2 ms → direction unresolvable → *refuse* (not mislabeled);
      α < 0.15 (~16 dB down) → below floor → not gated (too faint to matter).
- [x] Thresholds picked (in `wiki.md`): RHO_CLEAN 0.12, RHO_STRONG 0.22,
      CONSIST_MIN 0.60, SYMMETRY_MUTUAL 0.30, zero-lag band 2 ms.
- Caveat carried to Phase 1: thresholds fit on synthetic reverb; **re-fit rho/
  symmetry on a real recording** before shipping (heavy room reverb spreads the
  peak and may lower true-bleed rho). Mechanism is sound; scalars need real data.

## ⟳ Plan revision (2026-09-23) — real task is **speaker isolation by subtraction**
Testing on the user's real recording (see `wiki.md`) showed the actual problem is
**not** acoustic mic-bleed but a **"contained-voice" mix**: OBS routes the
computer audio (remote speaker, clean) into its own track **and** into the mic
track (unity gain, zero lag), so:
- one track (**C**) = a clean single voice,
- another (**M**) = C **+** the local speaker,
- a third (**s2**) ≈ M (duplicate mix).

The fix that "does it right" is: **keep C as one speaker, isolate `M − C` as the
other, drop the duplicate.** This subsumes the user's "fix duplicates AND finish
bleed" ask. The Phase-0 acoustic-bleed detector/gating is retained as a secondary
regime (no real recording needs it yet), but the **primary** path below is
subtraction. Phases 1–6 are re-scoped accordingly; Phase 0 stays valid (its
lagged-xcorr direction test is reused by the classifier).

### Phase 1 — Track-relationship classifier `src/track_separation.py`
- [ ] `decode_track_pcm(source, stream_index, channel_index, rate)` — ffmpeg
      `-f f32le` → numpy mono (alignment guaranteed by shared container).
- [ ] `classify_tracks(tracks) -> Plan` per pair, into:
      **duplicate** (symmetric ~unity containment, e.g. s1≈s2),
      **contained** (C's active windows ⊆ M's + `M−g·C` decorrelates ⇒ isolate
      `M−C`), **acoustic-bleed** (delayed copy ⇒ gate; reuse `spike/detect.py`),
      or **independent**. Emit a recording-level plan: clean-speaker tracks,
      derived (subtracted) tracks, dropped duplicates.
- [ ] Signals: active-set containment + scalar-subtraction residual/decorrelation
      (validated in `spike/subtract_probe.py`) + lagged-xcorr direction.
- [ ] Tests (synthetic): contained→isolate, duplicate→drop-one,
      independent→keep-both, acoustic-bleed→gate.

### Phase 2 — Isolation + residual cleanup
- [ ] `isolate(M, C) -> residual` — scalar `g=<M,C>/<C,C>` subtraction (fallback
      to short-FIR only if scalar leaves the ref correlated).
- [ ] **Noise-gate / VAD the residual** (zero out inactive windows) so Whisper
      doesn't hallucinate filler on the near-silent gaps — the gotcha found in
      testing. Reuse the active-window mask.
- [ ] Render isolated/clean speaker tracks to temp wav for the engine (keeps the
      "engine takes a file path" contract). **Source never touched.**
- [ ] Tests: residual isolates the unique speaker; gate kills silence; clean
      track passes through; source untouched.

### Phase 3 — Worker integration `app/worker.py`
- [ ] Refactor `_transcribe_multi_track`: **decode-all → classify → isolate/gate/
      drop-duplicates → transcribe each resulting speaker → merge with labels.**
      (Fixes the current "transcribes the same audio 3×" bug for this file.)
- [ ] Honor a per-job enable flag + per-track opt-out.
- [ ] Emit an honest preprocessing status (e.g. `Separating speakers…`) via
      `progress_detail_var`.
- [ ] `md_writer`: provenance footer noting derived/dropped tracks.
- [ ] Tests: synthetic contained-voice fixture → 2 clean, correctly-labeled
      speakers, no duplicated lines.

### Phase 4 — UI `app/ui.py` (UX-expert-designed surface, re-scoped)
- [ ] `Job`: enable flag, per-track exclusions, derived relationship plan
      (populated in the same async probe that fills `duplicate_of`).
- [ ] `settings.json`: persist the default enable flag.
- [ ] Panel shows the detected plan in plain language, e.g.
      *"Track 3 — one voice (kept). Track 1 — has Track 3 mixed in; we'll isolate
      the other voice. Track 2 — same as Track 1 (skipped)."* + names.
- [ ] Toggle (default on when a plan is found) + per-track opt-out + speaker
      naming; keyboard-reachable, contrast-safe, disabled while transcribing.

### Phase 5 — Verify on the **real** recording — CLOSE-gate
- [ ] Run the app on the Miroslav call: expect **2 speakers** (host + client),
      clean, no triple-transcription, host questions correctly attributed.
      Keep all outputs local (sensitive client data; public repo).

### Phase 6 — Close / lessons
- [ ] Capture lessons in `wiki.md`; flip folder `OPEN` → `CLOSED`.

### Deferred
- Acoustic-bleed **gating** path (Phase-0 detector) — kept but idle until a real
  in-room multi-mic recording needs it.
- Short-FIR/NLMS subtraction — only if a real recording shows non-unity / delayed
  containment that scalar subtraction can't clean.

## Status
- 2026-09-23: Update opened. Feasibility validated (time-aligned streams;
  numpy/scipy/ffmpeg already present). Two UX-expert designs synthesized into
  the design above.
- 2026-09-23: **Phase 0 spike done → GO.** Directional detection reliable
  (0/60 clean false positives; user's 3-track case nailed). Thresholds set.
- 2026-09-23: **Real-recording analysis → task re-scoped.** Verified true
  structure: s3 = clean voice, s1 = s3 + local speaker (unity gain, zero lag),
  s2 ≈ s1 dup. `s1 − s3` (g=0.999) cleanly isolates the other speaker (proven by
  transcribing the residual). Built `src/track_separation.py` (classify + isolate
  + gate + render, 9 tests).
- 2026-09-23: **RNNoise denoise added** (`denoise_pcm` via ffmpeg `arnndn`; model
  fetched, gitignored; 11 tests). User confirmed isolated+denoised audio "quite
  good".
- 2026-09-23: **Tested classifier on 5 real recordings → auto-classification NOT
  reliable** (2 of 5 wrong; Michal K genuinely ambiguous — see `wiki.md` table).
  **PIVOT (user-directed): drop auto-classification, give the user manual control
  with great UX** — per track: include/exclude, name, language, noise filter,
  and declare subtractions. Engine primitives stay as the backend.
- 2026-09-23: **Manual-control feature BUILT.** Two UX-pro designs synthesized &
  user-approved (noise default ON). Shipped:
  - Engine: `render_transcription_wav` (subtract refs → RNNoise → gate).
  - Backend: Job per-track fields + `_transcribe_multi_track` rewrite (per-track
    name→label, language override, noise filter, subtraction). Wiring test.
  - UI: multi-track panel with play · include · **editable name** · **▾ detail
    drawer** (language / reduce-noise / **remove-a-voice** / channel) · waveform ·
    read-only status summary; header count. Verified via headless render smoke
    test + clean app launch. 63 tests pass.
  - **All of the user's explicit asks are functional:** show tracks, include/
    exclude, name, per-track language, per-track noise filter, subtract one track
    from another.
  - **Polish still open (UX-pro niceties, not requested essentials):** A/B
    "Cleaned · original" preview of a subtraction, one-time discovery hint,
    recent-names suggestions, Reset button. **Next: verify end-to-end on a real
    recording in the app, then add polish / close.**
