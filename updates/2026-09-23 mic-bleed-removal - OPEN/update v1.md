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

### Phase 1 — Detection module `src/crosstalk.py`
- [ ] `decode_track_pcm(source, stream_index, channel_index, target_rate)` —
      ffmpeg `-f f32le` → numpy mono (downsampled; alignment guaranteed by
      shared container).
- [ ] `detect_bleed(tracks) -> BleedMap` — directional pairwise map
      `{contaminated_index: [(source_index, confidence, tier)]}` + per-region
      gating mask, at a coarse hop.
- [ ] Reuse/extend `audio_tracks.correlation`; keep the module isolatable with a
      small explicit interface.
- [ ] Tests: known synthetic bleed → correct direction/tier; clean tracks →
      empty map; symmetric heavy overlap → `refuse` tier.

### Phase 2 — Gating application
- [ ] `apply_gate(pcm, mask) -> pcm` and a writer that renders the cleaned track
      to a temp wav/mp3 for the transcriber (keeps the existing
      "engine takes a file path" contract).
- [ ] Tests: mask muting is exact; **source file untouched**; unaffected tracks
      pass through byte-for-byte-equivalent.

### Phase 3 — Worker integration `app/worker.py`
- [ ] Refactor `_transcribe_multi_track`: **extract-all → detect → gate →
      transcribe** (today it extracts+transcribes one track at a time).
- [ ] Honor `job.remove_crosstalk` + `job.crosstalk_exclude`.
- [ ] Emit `Removing mic bleed from {n} tracks…` via `progress_detail_var`.
- [ ] `md_writer`: append the provenance footer when cleaning was applied.
- [ ] Tests: worker path with a synthetic bleedy fixture → single, correctly-
      attributed transcript (no ghost lines).

### Phase 4 — UI `app/ui.py` (UX-expert-designed surface)
- [ ] `Job`: add `remove_crosstalk: bool`, `crosstalk_exclude: set[int]`,
      derived `bleed_sources` (populated in the same async probe that fills
      `duplicate_of`).
- [ ] `settings.json`: add `remove_crosstalk: true` (per-track exclusions stay
      per-Job — track indices don't transfer across files).
- [ ] Global toggle (shown only when `bleed_sources` non-empty) + subtext.
- [ ] Contaminated-row sub-label + click/context-menu opt-out.
- [ ] The four disclosure states: auto-cleaned / ask / refuse / none.
- [ ] Keyboard-reachable, contrast-safe (state via text not color), disabled
      while transcribing — consistent with existing track controls.

### Phase 5 — Trust surface: A/B listen (optional, cheap)
- [ ] Collapsed "Review": play cleaned (mask-ducked) vs `▶ Hear original` on the
      existing `TrackPlayer`; optionally draw muted spans faintly on the
      waveform.

### Phase 6 — Verify on the real app — CLOSE
- [ ] Run the actual app on a genuinely bleedy multi-track recording; confirm
      ghost lines are gone and attribution is correct end-to-end (not just green
      tests).
- [ ] Capture lessons in `wiki.md`; flip folder `OPEN` → `CLOSED`.

### Deferred (only if Phase 6 shows gating is insufficient)
- NLMS adaptive-subtraction escalation per-pair, behind the same single toggle
  (auto-picked strength). Kept out of scope unless real recordings demand it.

## Status
- 2026-09-23: Update opened. Feasibility validated (time-aligned streams;
  numpy/scipy/ffmpeg already present). Two UX-expert designs synthesized into
  the design above.
- 2026-09-23: **Phase 0 spike done → GO.** Directional detection reliable
  (0/60 clean false positives; user's 3-track case nailed). Thresholds set.
- 2026-09-23: **Real-recording reality check (see `wiki.md`).** Ran the detector
  on the user's real OBS sales call: all 3 tracks are near-identical full mixes
  (zero-lag corr 0.95–1.00, dual-mono), i.e. **no acoustic bleed present** — this
  is the duplicate-tracks case, not the bleed case. Detector correctly refuses
  everywhere (good fail-safe). **Blocked on Phases 2–6 until we confirm the real
  recording setup and get an actual in-room multi-mic file to tune against.**
  Awaiting user input on their recording setup.
