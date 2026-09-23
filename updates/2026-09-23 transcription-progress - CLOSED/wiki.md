# Wiki — transcription progress

Durable decisions and gotchas from this update.

## Progress hooks per backend
- **Parakeet (`parakeet-mlx`)**: `model.transcribe(..., chunk_callback=fn)`.
  `fn(current_samples, total_samples)` is called **before** each chunk is
  processed, and only when the file is long enough to be chunked
  (`audio_length > chunk_duration`). Short files return in one pass → no
  callback → the UI keeps the time-estimate bar. Fraction = current / total.
- **mlx-whisper**: `transcribe()` has **no** progress/callback parameter (it
  runs one blocking call with an internal tqdm). No clean hook. Real progress
  would require splitting the audio ourselves.

## The one UX invariant
Timecode ("12:30 / 41:05") is shown **only** when progress is real. The
estimate fallback shows "~ Estimated N min left" and never a timecode. So the
presence of a timecode on screen always means a true position — teach it once.

## Data model
- `Job.transcribe_fraction` — overall bar value. Multi-track: `(done + f)/N`.
- `Job.transcribe_position_fraction` — position within the current track/call,
  used for the timecode. Equal to `transcribe_fraction` for a single track.
- Both reset to `None` at the start of `_process` and on UI reset, so a re-run
  starts clean and a short (un-chunked) file correctly falls back to estimate.

## ETA
In real-progress mode the ETA is derived live from actual throughput
(`elapsed/fraction - elapsed`), not the static pre-run estimate — it self-
corrects as the run proceeds. Early updates (tiny fraction) can be jumpy; that's
accepted.

## Not a speed problem
Investigation showed Parakeet is already the fastest model and cached; imports
are sub-second. The complaint was fundamentally missing/false feedback, not CPU
time. Honest progress addresses the "feels slow" perception directly.
