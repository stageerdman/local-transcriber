# Real transcription progress — update v1 (CLOSED)

## Goal
A user transcribing long recordings couldn't tell "where in the call am I."
The progress bar was driven purely by a time **estimate** (elapsed ÷ predicted
total), so it read as stuck/slow and never reflected the true position in the
audio. Give an honest, movable progress signal.

## What we found
- The default/most-used backend is **Parakeet TDT v3** — already the fastest
  model and already cached locally; library imports are ~0.2s. So there was no
  real "loading" bottleneck to fix — the pain was **feedback**, not raw speed.
- `parakeet-mlx`'s `transcribe()` exposes a `chunk_callback(current_samples,
  total_samples)` fired once per chunk. That's a true position-in-audio hook.
- `mlx-whisper`'s `transcribe()` exposes **no** progress hook (single blocking
  call). Whisper jobs therefore keep the time-estimate fallback.

## What we built (single phase, done)
1. `src/transcription.py` — `transcribe_mp3_segments(..., progress_callback)`;
   the Parakeet path maps `chunk_callback` samples → a fraction in [0,1].
2. `app/engine.py` — forwards `("progress", job_id, fraction)` over the
   response queue during transcription.
3. `app/worker.py` — `_transcribe_via_engine(..., on_progress)`; single-track
   sets the fraction directly, multi-track maps it to `(done + frac)/N` for the
   overall bar while keeping the within-track fraction for the timecode.
4. `app/jobs.py` — `transcribe_fraction` (overall, bar) +
   `transcribe_position_fraction` (within current track, timecode).
5. `app/ui.py` — `_progress_view()` unifies real-vs-estimate for both the row
   and the top panel. Real progress shows a **timecode** ("12:30 / 41:05") as
   the hero + live ETA; the estimate fallback is marked "~ Estimated … left"
   and shows **no** timecode.

## UX decisions (from a UX-expert agent)
- **Timecode present = real position; "~"/"Estimated" = a guess.** One rule
  makes every screen self-explanatory.
- One overall bar, never per-track bars. Ship the raw chunk jumps (honest)
  rather than faking smoothness.
- "Loading model" reworded so it doesn't imply a download every time and
  doesn't read as frozen.

## Verified
- 50 tests pass (4 new: format helpers + Parakeet callback→fraction mapping).
- Real Parakeet run on a 150s clip: `chunk_callback` fired 0.80 → 1.00,
  monotonic, reached 1.0; model load + transcribe in 6.5s.
- Headless UI render of every state (loading / estimate fallback / real single
  / real multi-track) produced the expected labels.

## Known limitation / possible follow-ups
- **Whisper backend has no real progress** — stays on the estimate bar. Real
  progress there would require chunking the audio ourselves (quality tradeoff
  at boundaries); deferred. Tracked in `issues.txt`.
- Parakeet chunking is 120s (matches its CLI default, chosen to avoid a Metal
  OOM on ~40-min files). Smaller chunks would give finer/ more frequent bar
  updates and are *more* memory-safe, at some extra overlap recompute — a knob
  if the jumps feel too coarse.
