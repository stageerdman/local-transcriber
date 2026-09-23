# Scrubbable track preview — update v1 (CLOSED)

## Goal
The per-track preview (▶ on each channel) was unusable for real work: it
**re-encoded the whole track to MP3 before any sound played** (slow to start on
long recordings) and used `afplay`, which **can't seek** - so you couldn't jump
around / "scroll" the audio while listening. Make preview start instantly and
let the user seek anywhere in a track.

## Approach (validated before building)
- Switched the player from `afplay` to **`ffplay`** (already present via the
  Homebrew ffmpeg the app requires). ffplay plays and **seeks straight from the
  source** (`-ss`), selecting the track (`-ast a:N`) and, if needed, a single
  channel (`-af pan`). **No extraction, no temp files → effectively instant
  start**, and seeking is just relaunching at a new offset.
- Confirmed empirically: `-ast a:N` selects the Nth audio stream; `-nodisp`
  runs headless; `-ss` seeks; real position advances correctly.

## What we built
1. `src/playback.py` - `TrackPlayer` now drives ffplay:
   `play(source, audio_index, channel_index, start_seconds, on_finish)`,
   `stop()`, `is_playing()`, `position()`. Single-owner (a new play/seek stops
   the previous), on_finish only on natural end.
2. `app/ui.py` - the waveform is now an interactive seek bar:
   - **Click** to seek + play from there; **drag** to scrub (playhead + time
     follow live, audio (re)starts on release so ffplay isn't thrashed).
   - Moving **playhead** with a grab handle; played portion drawn in the accent
     color; **"3:42 / 12:05" timecode** tucked in the corner.
   - The ▶/■ button is transport (stop / resume from playhead); the waveform is
     position. Each track keeps its **own playhead**, so you can hop between
     tracks and resume each where you left off.
   - Keyboard: Left/Right = ±5s, Space = play/stop; focus ring; `hand2` cursor;
     everything disabled (and playheads cleared) while a job transcribes.

## UX decisions (from a UX-expert agent)
- Timecode is the "where am I" anchor; one overall playhead; click = play-from-
  here (no separate move-without-play gesture); scrub seeks on release.
- Keep both button and waveform - distinct jobs (transport vs position).
- Deviation: the agent suggested *resetting* an inactive track's playhead; we
  chose to **persist per-track playheads** instead, because we implemented
  real resume (▶ continues from the playhead) and jumping between tracks and
  resuming each is more useful. Playheads are cleared on natural end / job
  start / reset.

## Position tracking - the one gotcha
ffplay's status timestamp is **absolute for some containers, relative for
others** (mp3 reported absolute, a .mov looked relative), so parsing it
double-counted the seek offset. Dropped stderr parsing entirely; `position()`
is now **wall-clock from the seek point** - predictable, container-agnostic,
and accurate to within ffplay's small startup latency (fine for a scrubber).

## Verified
- 51 tests pass (playback tests rewritten for ffplay + seek + wall-clock).
- Real ffplay run: seek to 60s, position advanced 61.2 → 62.2s, stop clean,
  on_finish suppressed on stop.
- Headless UI drive: click-seek launches `-ss/-ast`, playhead advances, stop
  keeps playhead, resume seeks to the kept playhead, supersede keeps the other
  track's playhead, transcribe-start clears + disables.

## Removed
- The extract-to-MP3 preview path (`extract_track_to_mp3`/`extract_channel_to_mp3`
  are still used by the worker for transcription; only the *preview* extraction
  and its temp-dir/cache/cleanup were removed).
