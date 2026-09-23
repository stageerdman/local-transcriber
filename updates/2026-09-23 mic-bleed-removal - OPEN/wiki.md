# Mic-bleed removal — wiki (durable decisions & research)

## The problem, precisely
"Bleed" (= crosstalk = spill = leakage): in a multi-mic recording made in one
acoustic space, each mic captures the target speaker **plus** an attenuated,
delayed, mildly-reverberated copy of every other speaker. In our data the tracks
are audio **streams inside one container** (Zoom per-participant recording,
multi-track OBS), so they share `t=0` and duration — **inherently time-aligned**.
That alignment is what makes clean removal tractable: for a contaminated track we
already possess a near-clean reference of the interfering voice (that person's
own track).

Failure it causes today: each track is transcribed independently and merged in
`dialogue.merge_segments_into_blocks`. A bleed copy of speaker B on speaker A's
track is transcribed a second time and attributed to A (or flagged
`interrupting`). Result: duplicated, misattributed "ghost" lines.

## Research — approaches & tooling
- **Directional lagged cross-correlation (detection).** For each ordered pair
  (A receives from B?), cross-correlate A against B over small lags. A strong
  peak at a positive lag with A's copy quieter than B ⇒ B bleeds into A. This is
  directional and gives per-region confidence. `audio_tracks.correlation()`
  (Pearson) already exists as a starting point; scipy adds lagged/FFT
  correlation.
- **Gating / ducking (removal, chosen for v1).** Mute the contaminated track in
  regions where the reference dominates and the contaminated signal is
  confidently just a delayed copy. Cheap; provably can't remove real content in
  those regions. This is essentially Auphonic's multitrack mic-bleed approach
  ("know when/which track a speaker is active, remove the correlated signal from
  the others").
- **Adaptive-filter subtraction (NLMS/RLS) (deferred).** Estimate the
  acoustic-path FIR from reference→contaminated and subtract the predicted bleed.
  Cleaner audio; known failure mode: attenuates the true speaker during
  double-talk. Libraries exist (`padasip`, `adaptfilt`) but an NLMS is ~20 lines
  of numpy — **not adopting for v1** (see scope decision below).

### Environment facts (checked 2026-09-23)
- `.venv` already has **numpy 2.4.6** and **scipy 1.17.1** (transitive via
  mlx-whisper). **ffmpeg 8.1.1** on PATH. → detection + gating add **no new
  heavy deps**. PCM access = ffmpeg `-f f32le` piped to numpy.
- Codebase has **no** existing numpy/soundfile/librosa usage — all audio is
  ffmpeg subprocess today. The new `src/crosstalk.py` introduces the first
  in-memory PCM path; keep it isolated behind a small interface.

### Sources
- Auphonic mic-bleed remover: https://auphonic.com/blog/2025/10/08/mic-bleed-remover/
- Auphonic multitrack algorithms: https://auphonic.com/help/algorithms/multitrack.html
- adaptfilt (Python NLMS/RLS): https://github.com/Wramberg/adaptfilt
- Sonix on crosstalk/mic-bleed removal: https://sonix.ai/articles/how-to-remove-crosstalk-and-mic-bleed-from-audio-for-free

## Codebase integration facts (from the code map)
- Tracks: `src/audio_tracks.py` — `AudioStream(index, channels, title)`;
  `probe_audio_streams`, `extract_track_to_mp3`, `extract_channel_to_mp3`,
  `compute_volume_envelope` (150-bucket RMS @8kHz), `correlation`.
- Transcription: `src/transcription.py` — takes **mp3 paths**, one pass/track
  (`TranscribedSegment`). Backends: mlx-whisper, parakeet-mlx.
- Merge: `src/dialogue.py` — `Segment`→`Block`; overlap only *labeled*
  (`interrupting`), never suppressed. This is the gap bleed removal fills
  upstream (acoustic), not here (text).
- Insertion point: `app/worker.py` `TranscriptionWorker._transcribe_multi_track`
  (~L253-301). Currently extract-one→transcribe-one in a loop; must become
  **extract-all → detect → gate → transcribe** for cross-track processing.
- Job: `app/jobs.py` `@dataclass Job` already carries derived per-recording maps
  (`duplicate_of`, `channel_pairs_identical`, `selected_track_indices`,
  `selected_channel_by_track`, `track_envelopes`) — mirror these for
  `bleed_sources` / `remove_crosstalk` / `crosstalk_exclude`.
- Settings: `app/ui.py` `settings.json` (~L69-80) holds only model + language.
- Existing precedent to reuse: `_maybe_analyze_tracks` (`app/ui.py` ~L928-967)
  already auto-detects **whole-track duplicates** via envelope correlation and
  auto-deselects them, rendering `Name (~Other)`. Bleed is the *partial* sibling
  of this; reuse the pattern and the grammar.
- Player: `src/playback.py` `TrackPlayer` (ffplay, seek/scrub) — reuse for the
  Phase-5 A/B cleaned-vs-original listen via a mute-mask (no new pipeline).

## UX synthesis (two UX-expert agents) — decisions
Both agents converged strongly; recorded here are the reconciled decisions.

**Agreed (both agents):**
- One **global** toggle, **default ON**, **shown only when bleed is detected**.
- **No per-pair / N×N directional matrix** — unreadable, redoes the app's
  detection by hand, ~0% of users can fill it correctly. Cut.
- **No strength dial** (gating vs subtraction) — an engineering risk trade-off
  the user can't evaluate; auto-picked, hidden. Cut.
- **Source never touched;** undo = untick + re-run (non-destructive).
- Reuse the existing `(~Other)` duplicate grammar for disclosure.
- Honest **processing-time** feedback; detection rides the existing probe pass.
- Per-track **opt-out** ("don't clean this one") folded into the contaminated
  row (sub-label + context menu), not a checkbox on every row.
- `remove_crosstalk` in `settings.json` (default true); per-track exclusions are
  **per-Job only** (track indices don't transfer across recordings).
- Cut: separate cleanup screen/wizard, "all clear" banner, confidence
  percentage / dB / lag-ms in primary UI.

**Reconciled divergences:**
- **NLMS subtraction:** Designer A said cut it from v1; Designer B said
  auto-escalate to it per-pair. → **Cut from v1** (A wins): transcript ≠ mastered
  audio; gating suffices and can't damage the real speaker. Subtraction deferred,
  optional, only if real testing demands it.
- **Trust preview:** A designed a collapsed "Review" A/B listen (mute-mask over
  the existing player — cheap); B deferred any preview (thought it needed a
  second pipeline). → A's technical read is right (mask over `TrackPlayer` is
  cheap). Keep it, but as **Phase 5** so v1's safety rests on the three-tier
  behavior first.
- **Terminology:** A avoids "bleed"/"crosstalk", prefers a plain sentence +
  "echo"; B argues "mic bleed" is the term non-engineers recognize and names the
  cause. → **Lead with the plain sentence** everywhere ("Some mics also picked
  up the other people in the room"); use **"mic bleed"** as the short noun /
  toggle label, defined inline on first use. Avoid the word "crosstalk" in the
  UI. Revisit the exact toggle wording in Phase 4.

**Three-tier safety behavior (from Designer A, adopted as core):**
- Confident → **auto-clean** (on by default).
- Uncertain → **ask** (untouched by default; offer to clean).
- Too much mutual overlap to gate safely → **refuse** (state it; no dangerous
  control offered).
- No bleed → **render nothing**.

### Microcopy (proposed defaults)
- Summary: `Some mics also picked up the other people in the room. Cleaned up
  automatically before transcribing.` + `Review ▸`
- Toggle: `Remove mic bleed between tracks`
- Toggle subtext: `Cleans tracks where another mic was picked up. Adds a little
  processing time.`
- Contaminated sub-label: `also picks up Person 3` / opted-out: `(left as-is)`
- Context menu: `Don't clean this track` ⇄ `Clean this track`
- Uncertain: `Person 2's mic might be picking up Person 3 — we're not sure.`
- Refuse: `Person 2's mic overlaps too much with the others to clean safely —
  left as-is.`
- Preprocessing status: `Removing mic bleed from {n} tracks…`
- Transcript footer: `_Mic-bleed cleanup applied to: Person 2, Room laptop._`

## Open questions / risks
- **Detection reliability is the whole feature's gate** (Phase 0). If we can't
  detect direction + confidence robustly, safe-by-default is impossible → stop.
- Reverb/room response makes the acoustic path more than a pure delay; gating
  (region-based) is robust to this, which is another reason it beats subtraction
  for v1.
- Channel selection interacts with detection — detect on the same channel the
  user will transcribe (respect `selected_channel_by_track`).
