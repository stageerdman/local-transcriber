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

## Phase 0 spike results (2026-09-23) — GO
Detector = per-window (250 ms, 125 ms hop) normalized cross-correlation over
±30 ms lags; aggregate `rho`, dominant `lag` (sign ⇒ direction),
`lag_consistency`, `side_symmetry`; plus a per-window gating mask. Code in
`spike/` (synth.py, detect.py, cases.py, sweep.py) — self-contained, no new deps.

Measured:
- Clean pair rho ≈ 0.047; **0/60 false positives** across random voice pairs,
  overlap 0.4–0.95, max rho 0.117.
- B→A α=0.30 τ=8 ms: rho 0.296, correct direction, 60% windows gated.
- 3-track (P3→P2, user's case): only P2,P3 flagged (rho 0.326, dir P3→P2,
  consistency 1.00, lag 6 ms); P1,P2 & P1,P3 dismissed.
- Zero direction errors for τ ≥ 5 ms at every α.
- Blind spots (both fail *safe*): τ ≤ 2 ms → refuse (direction unresolvable);
  α < 0.15 (~16 dB down) → below clean floor, not gated.
- Mutual-bleed guardrail: single-direction symmetry ≈ 0.00 vs any real mutual
  ≥ 0.45 → mutual cut set to 0.30; equal 0.5/0.5 mutual correctly refuses,
  strongly-asymmetric (0.4/0.1) still auto-cleans dominant direction.

**Chosen thresholds** (high confidence on clean/strong/mutual boundaries; the
0.12–0.22 "ask" band is the genuinely fuzzy zone, routed to a human):
```
RHO_CLEAN       = 0.12   # below -> clean (max clean pair measured 0.117)
RHO_STRONG      = 0.22   # + consistent lag -> confident bleed
CONSIST_MIN     = 0.60   # >=60% active windows agree on lag sign+value
SYMMETRY_MUTUAL = 0.30   # + strong rho -> mutual -> refuse
ZERO_LAG_BAND   = 2 ms   # |lag| within -> direction unresolvable -> refuse
```
Tiers: auto-clean = `rho<0.12` OR (`rho>=0.22` & consistency>=0.60 & resolvable);
ask = correlated 0.12–0.22 or low consistency; refuse = mutual OR zero-lag.

**Carry to Phase 1:** thresholds fit on synthetic FIR reverb — **re-fit
rho/symmetry on a real recording** before shipping (heavier real-room reverb
spreads the correlation peak, may lower true-bleed rho). Mechanism is sound; the
scalars are data-dependent.

## Real-recording reality check (2026-09-23) — IMPORTANT
Ran the detector against a real user recording ("Sales Call - Miroslav.mov", OBS,
3 audio streams, 25 min). Harness: `spike/real_data.py`, `spike/channel_probe.py`.

**Finding: this recording does NOT contain the mic-bleed pattern the feature
targets.** Structure measured:
- Each stream is **dual-mono** (L channel == R channel, exactly identical).
- All three streams are **near-identical full mixes**: zero-lag correlation
  0.95–1.00 between every stream pair, **dominant lag = 0 ms everywhere**, across
  windows at 60 s / 300 s / 1200 s. RMS differs slightly (stream 2 ~3–5 dB
  quieter) — gain/processing differences, not different speakers.
- The detector **correctly refuses in every window** (zero-lag ⇒ direction
  unresolvable ⇒ refuse). It did not manufacture a false directional bleed. Good
  fail-safe behavior confirmed on real data.

**Interpretation:** this is a *remote* sales call recorded via OBS where every
track carries essentially the same program mix — there are **no isolated
per-speaker microphones**, so there is nothing to subtract. Acoustic bleed
requires multiple *open mics in one physical room*; a remote call routed through
OBS doesn't produce it (and this OBS setup didn't save isolated sources either).
This is the **whole-track (near-)duplicate** case — which the app's existing
`_maybe_analyze_tracks` duplicate detection is the right mechanism for, not this
feature. (Note: the existing detector uses 150-bucket envelope correlation with a
0.98/0.995 threshold; at 0.95–0.99 these tracks sit right at its boundary — worth
a look separately.)

**Consequence for this update:** cannot re-fit bleed thresholds on this file
(no bleed present). Phase 1 detector code is still valid and setup-independent,
and its refusal on this file is already the correct outcome. But before investing
in Phases 2–6, we need to confirm the user's real recording setup actually
produces isolated per-mic tracks with bleed (in-room multi-mic), and get such a
file to tune against. Flagged to the user.

## DEFINITIVE structure of the real recording (2026-09-23)
After the user clarified the setup (one mic that also captures the computer's
audio; a separate clean computer-audio track), I verified the true structure
empirically with `spike/subtract_probe.py` (active-set + subtraction) and
`spike/verify_speakers.py` (transcribe clean track, mix track, and residual).

**Confirmed structure (privacy note: the call is a real client's sensitive
health consultation — NO transcript content is recorded here or committed; the
repo is public):**
- **s3 = one speaker only** (the "person in the computer" / remote party),
  clean. Active in ~64% of windows; whenever s3 is active, s1/s2 are too
  (containment = 1.00), and s1/s2 are additionally active in the ~23% of windows
  that are the *other* speaker's solo turns.
- **s1 = s3 + the other speaker** (the host mic, which also carries the computer
  audio). `s2 ≈ s1` (near-duplicate full mix).
- **s1 − g·s3 with g = 0.999 isolates the other speaker**, residual decorrelated
  from s3 (corr 0.00), −12 dB. Transcribing the residual yields the host's
  utterances that appear **nowhere** in s3 — proving clean separation. The
  relationship is **digital, unity-gain, zero-lag** (OBS mixed desktop audio into
  the mic track), so a scalar subtraction is exact; adaptive filtering is NOT
  needed here (and my quick Wiener FIR did *worse* than scalar — expected).

**So the real task ≠ acoustic mic-bleed.** It is **speaker isolation by reference
subtraction** in a "contained-voice" recording:
- One track (C) is a clean single voice.
- Another track (M) contains C plus a second voice, at ~unity gain / zero lag.
- Output two clean speakers: **C as-is**, and **M − C** for the other; drop the
  duplicate mix (s2).

**Gotcha found:** the residual (M−C) is mostly silence during C's solo turns, and
Whisper **hallucinates** filler ("a a a a") on the near-silent gaps. So the
isolation step must be followed by a **noise gate / VAD** on the residual before
transcription (or transcribe only its active regions). This is essential to "do
it right."

**Detection signal for this regime** (distinct from the acoustic-bleed tiers):
- zero-lag dominant + very high rho (≥ ~0.9) + strong active-set **containment**
  (C's active windows ⊆ M's, and M has extra solo windows) ⇒ "C is contained in
  M": isolate.
- If two tracks are contained in *each other* symmetrically at ~unity (s1≈s2)
  ⇒ duplicates: keep one.

## End-to-end test on real audio (2026-09-23) → better architecture found
Built `src/track_separation.py` (classify + isolate + gate + render) and ran it
end-to-end on the real call. Results:
- **Classification: perfect** on the full 25 min — keep(s3=remote), isolate(s1 −
  s3 = local), drop(s2 duplicate). 2 speakers out.
- **Isolation math: works** — the residual transcript contains the host's own
  questions (absent from s3), i.e. the two speakers ARE separated.
- **BUT transcribing the subtraction *residual* is low quality:** Whisper
  **degenerate-loops / hallucinates** ("...top of the hill ×9", "Závodný výstav
  ×20") on the rough, quiet residual, and auto-detects the wrong language.
  Forcing language + `condition_on_previous_text=False` + normalization helped
  but did NOT eliminate the looping. ASR on a subtracted residual is just poor
  audio.

**Key realization — don't transcribe the residual; transcribe only clean audio:**
- The **mix track s1 is good-quality audio** (the real recording), not a residual.
- During the **client's silent windows, s1 ≈ the host alone, clean.** (We already
  compute per-window activity for both tracks.)
- So: **client** = transcribe s3 (clean); **host** = transcribe s1, but only in
  windows where s3 (client) is inactive — there s1 is clean host audio. The
  acoustic subtraction's real job is **VAD/attribution** (who's active when), not
  producing audio to feed Whisper.
- Overlap (both talking) is the minority in turn-taking calls; handle as a
  fallback (attribute to the louder-in-residual speaker, or mark overlapping).

This reframes Phase 2/3: use subtraction + per-track VAD to build a **speaker
activity timeline**, then transcribe each speaker from the **cleanest available
source** for their active regions, and merge by timestamp. Avoids residual-ASR
entirely for the common case. The `isolate()`/`noise_gate()` code stays useful
for the VAD signal and for genuine-overlap fallback.

## Reliability test across 5 real recordings (2026-09-23) → PIVOT to manual control
Tested `classify_tracks` on 5 real OBS sales calls. Every one has 3 stereo tracks
where **T3 is a low-activity (~0.3) clean remote/client track** and **T1/T2 are
high-activity (~0.75) full-conversation mixes** — but the mixes are balanced
differently per recording, so their correlation swings wildly (0.70–0.98):

| recording   | corr(T1,T2) | corr(T1,T3) | auto-plan result        | correct? |
|-------------|-------------|-------------|-------------------------|----------|
| Miroslav    | 0.98        | 0.99        | keep3 / isolate1 / dup2 | ✅ (2 spk) |
| Frantisek   | 0.98        | 0.42        | keep1 / dup2 / keep3    | ✅ (2 spk) |
| Michal Sebo | 0.95        | 0.67        | keep1 / dup2 / keep3    | ✅ (2 spk) |
| Marek       | 0.81        | 0.78        | keep3 / isolate1 / keep2| ✗ (3 spk) |
| Michal K    | 0.70        | 0.99        | keep3 / isolate1 / keep2| ✗ (3 spk, ambiguous) |

Michal K is genuinely ambiguous even under careful analysis: T1 ≈ client
(corr 0.99, 17.7 dB subtractable) yet also has host-solo activity; corr(T1−T3,
T2−T3)=0.24 (the two mixes' residuals are NOT the same signal). No simple,
safe universal rule separates "mix that reduces to the host" from "track that is
≈ a copy of the client" across all five.

**Decision (user-directed): drop auto-classification; give the user manual
control with great UX.** The user can tell who's who by listening. This is the
robust answer — it sidesteps the unsolved classification problem entirely.

`classify_tracks` is demoted to an optional *suggestion* (may still power a
"suggested setup" hint later), NOT the source of truth. The engine primitives
(`isolate`, `denoise_pcm`, `noise_gate`, `render_speaker_wav`) are exactly what
the manual controls drive — nothing wasted.

### Manual-control feature set (user-specified)
Per track, the user can: include/exclude it, **name** it (→ speaker label),
set its **language**, toggle a **noise filter** (RNNoise), and declare a
**subtraction** ("remove Track B's voice from Track A"). Two UX-pro agents are
designing the panel + the subtraction interaction.

### RNNoise denoise — DONE
`denoise_pcm` uses ffmpeg `arnndn` (RNNoise, model `bd.rnnn`), no new Python dep,
model fetched via `scripts/fetch_rnnoise_model.sh` (gitignored, no-ops if absent).
User confirmed the isolated+denoised audio quality is "quite good".

## Synthesized manual-control UX (2026-09-23, user-approved) — BUILD
Two UX pros (panel + subtraction) reconciled. Approved by user; **noise filter
default ON** (user override of the pro's "off" recommendation).

- **Collapsed row:** `▶ play/scrub · ☑ include · click-to-edit name ▾ · waveform ·
  right-edge read-only status summary`. Summary tokens: language code, `Clean`
  (denoise on), `− {Name}` (a voice removed), `~{Name}` (suspected duplicate).
- **`▾` detail drawer** (native `pack(after=row)` like `expand_frame`): Language,
  Reduce background noise, Remove-a-voice.
- **Naming:** inline click/F2 to edit; default `Person N` muted → solid when set;
  names become transcript speaker labels; "recent names" suggestions from
  `history_db`.
- **Per-track language:** first item `Same as app ({effective})` = inherit
  (live link to global); concrete choice overrides that track only.
- **Noise filter:** label `Reduce background noise` (no "RNNoise"); **default ON**.
- **Remove a voice (subtraction):** quiet affordance → popover "Remove a voice
  from '{name}'" listing the *other* tracks, each with ▶ hear; applies on click;
  set-state chip `⊘ {Name}'s voice removed ✕`; row preview gains **Cleaned ·
  original A/B** (plays `render_speaker_wav`, cached). Direction reads as object
  of a sentence — no math words. One-time panel discovery hint. Edge cases:
  self excluded from its own picker; reciprocal loop disabled with reason;
  unnamed → "Track N" fallback + focus its name field; rare multi-subtract via
  `+ another` (maps to `reference_indices` list).
- **Duplicates:** suspected dup pre-unchecked, `~{Name}` hint, fully reversible.
- **Persistence:** names / include / language / noise / subtraction are **per-Job**
  (never leak across files); only preferences (noise default, drawer state,
  global language) persist to settings.json.
- **A11y:** keyboard loop, text-not-color state, panel read-only during
  transcription, per-track load-error state isolated.
- **Cut:** volume sliders, waveform zoom/timeline, solo/mute, color swatches,
  auto-classification UI, wizard/onboarding, bulk apply, free-text tags, drag
  gestures, confirm step, arithmetic vocabulary.

## Open questions / risks
- **Detection reliability is the whole feature's gate** (Phase 0). If we can't
  detect direction + confidence robustly, safe-by-default is impossible → stop.
- Reverb/room response makes the acoustic path more than a pure delay; gating
  (region-based) is robust to this, which is another reason it beats subtraction
  for v1.
- Channel selection interacts with detection — detect on the same channel the
  user will transcribe (respect `selected_channel_by_track`).
