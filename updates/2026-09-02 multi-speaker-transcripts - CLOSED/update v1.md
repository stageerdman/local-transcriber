# Multi-speaker transcripts (multi-track dialogue blocks)

Status: **done — 2026-09-02.** Covered by automated tests (32 passing,
`pytest` from `local-transcriber/`). Real UI click-through with actual audio
wasn't done this session (see Verification).

## Goal

Upgrade Local Transcriber so a transcript reads like a dialogue instead of
one wall of text, **when the recording actually gives us the means to know
who's speaking**:

- When a recording has **separate audio tracks per participant** (e.g.
  Zoom's "record a separate audio file for each participant", or any
  container with >1 audio stream), detect that automatically and transcribe
  each track separately — the track is ground truth for who's speaking, no
  guessing involved. Break the merged result into pause-delimited,
  speaker-labeled blocks:

  ```
  **Person 1** [00:03]
  Hey, so I wanted to go over the proposal from last week.

  **Person 2** [00:11] (interrupting)
  Actually, before that, can we talk about the timeline?
  ```

  Overlapping speech (one person cutting the other off) shows up as an
  actual `(interrupting)` block, not a clean turn-by-turn alternation.

- A recording with only **one mixed-down track** is transcribed as plain
  text, same as the app always did. **No speaker detection and no
  pause-based blocking on single-track audio** — see History below for why.

## History

**v1 (superseded)** also tried to guess speakers on single-track audio using
pitch/spectral-centroid heuristics clustered with a small k-means
(`src/speaker_heuristics.py`), gated by a cluster-quality check so it would
fall back to one speaker rather than mislabel a monologue. It worked on
clean turn-taking but, as expected and confirmed by its own tests, couldn't
reliably separate genuinely simultaneous overlapping speech on one mixed
waveform. The user decided this wasn't worth it: **removed entirely**.
Single-track files now just get a plain-text transcript, exactly like
before this feature existed. `src/speaker_heuristics.py` and its tests were
deleted; `numpy` was dropped from `requirements.txt` since nothing imports
it directly anymore; the now-redundant `transcribe_mp3()` (flat-text, no
timestamps) was also removed from `src/transcription.py` since the worker
gets everything it needs from `transcribe_mp3_segments()` and joins segments
itself for the single-track case.

## Decisions

- Pause-based blocking and speaker labels are **multi-track only**. Single
  track = plain text, full stop.
- `job.status` keeps its existing canonical values (`converting`,
  `transcribing`, etc.) so UI logic (`ACTIVE_STATUSES`/`CANCELABLE_STATUSES`/
  `TERMINAL_STATUSES`) doesn't break; `job.status_detail` carries the
  human-readable "track 2/2 · Person 2" extra info during multi-track jobs.
- `speakers:`/`diarization:` frontmatter fields are only written for
  multi-track output (`diarization: multitrack`); single-track `.md` files
  have no such fields, same as before this feature.

## Implementation

- `src/audio_tracks.py` — ffprobe-based detection of multiple audio streams
  in a source file (`probe_audio_streams`/`safe_probe_audio_streams`),
  ffmpeg extraction of each stream to its own mp3
  (`extract_track_to_mp3`), and speaker naming from a stream's `title` tag
  if present, else "Person N" (`track_speaker_names`).
- `src/dialogue.py` — `Segment`/`Block` dataclasses,
  `merge_segments_into_blocks` (pause-based merging + interruption
  flagging), `render_blocks_markdown` (dialogue-format rendering). Used only
  on the multi-track path.
- `src/transcription.py` — `transcribe_mp3_segments()` returns Whisper's
  per-segment timestamps (needed for both the multi-track merge and the
  single-track plain-text join).
- `app/engine.py` — returns segments (not flat text) from the transcription
  child process.
- `app/worker.py` — `_process()` probes track count; multi-track loops
  per-track transcription via `_transcribe_multi_track()`, merges, and
  renders dialogue markdown; single-track transcribes once and joins
  segment text into a plain paragraph, same shape of output as before this
  feature. Multi-track jobs get their time estimate scaled by track count.
- `app/jobs.py` / `app/ui.py` — `Job.status_detail` shows per-track progress
  without touching the canonical status strings the UI matches on.

## Verification

Automated (`local-transcriber/tests/`, 32 tests):
- `test_dialogue.py` (11) — pause merging, speaker-change block breaks,
  interruption flagging, timestamp formatting, single- vs multi-speaker
  rendering. Pure logic, no audio.
- `test_audio_tracks.py` (6) — real ffmpeg-generated 2-audio-stream fixtures
  (not mocked): stream count detection, `title` tag → speaker name,
  per-track extraction produces correctly-sized independent mp3s.
- `test_worker_integration.py` (1) — the real multi-track extraction +
  dialogue-merge path against a real ffmpeg-generated 2-stream fixture,
  confirming an overlapping second track is flagged `(interrupting)`
  end-to-end.
- `test_md_writer.py` — frontmatter `speakers:`/`diarization:` fields
  present when given, absent when not (i.e. absent for single-track output).

Not done this session: no click-through of the real macOS app UI with a real
multi-track recording (would need an actual sample file and a downloaded
Whisper model) — worth doing once a real test recording is available,
particularly to sanity-check the multi-track ETA scaling and the
`status_detail` UI text against a live run.

## Known limitations / future work (not done, not blocking)

- No participant-name detection from multi-track file metadata beyond
  reading the `title` stream tag ffmpeg happens to preserve — falls back to
  "Person 1", "Person 2", ... in track order.
- No manual "rename Person 1 -> Alex" UI yet — would be a nice follow-up.
- Single-track recordings with multiple people get no speaker separation at
  all, by design (see History) — if that's ever revisited, real diarization
  (pyannote.audio or similar) is the likely path, not another heuristic.
