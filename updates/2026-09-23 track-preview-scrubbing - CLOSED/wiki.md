# Wiki — track preview / scrubbing

## Why ffplay (not afplay / AVFoundation / sounddevice)
- `afplay` can't seek and has no position feedback → dead end for a scrubber.
- `ffplay` ships with the Homebrew ffmpeg the app already requires → **zero new
  dependencies**, plays + seeks straight from the source container.
- AVFoundation (pyobjc) or sounddevice would give sample-accurate seeking but
  add a dependency; not worth it for a preview.

## ffplay invocation cheatsheet
- Headless: `-nodisp` (no window), `-autoexit` (quit at end), `-loglevel quiet`.
- Seek: `-ss <sec>` **before** `-i` (fast input seek; accurate enough).
- Track select: `-ast a:N` = the Nth **audio** stream (0-based among audio
  streams = its position in `job.tracks`). NOT the absolute stream index, and
  `i:` matches by stream *id*, not index - both confusing, avoid.
- Channel isolate: `-af "pan=mono|c0=c{ch}"`.

## Position tracking gotcha (important)
ffplay's periodic status line ("`12.34 M-A: ...`") is **absolute for some
containers and relative for others** - an mp3 seeked to 60s reported ~60.9,
while a short .mov seeked to 1s looked ~0. So `start + parsed` double-counts.
Do NOT parse it. `TrackPlayer.position()` uses wall clock from the seek point;
it can lead the audio by ffplay's startup latency (~0.2-0.4s), negligible here.

## Playhead model
- `self._playhead[(job_id, stream_index)]` = fraction [0,1], persisted per
  track. Persisting (not resetting) lets ▶ resume and lets the user hop between
  tracks and resume each.
- Cleared on: natural end (`_on_playback_finished`), job leaving "ready"
  (`_clear_job_playheads`), and job reset.
- Only one track is audible; `_start_playback` stops the previous first.

## Drawing
`_draw_envelope` paints the muted envelope, overpaints the played portion
(left of the playhead) in the accent color, draws the playhead line + top cap,
and writes the corner timecode. Waveform canvas is `WAVEFORM_HEIGHT` tall and
focusable; disabled state drops focus + cursor.

## Interaction
Click/drag both go through press→drag→release: press stops audio + marks the
scrub key, drag moves the playhead/time only, release calls `_start_playback`
at the final fraction. This makes a plain click = "play from here" and a drag =
"scrub, commit on release" with the same code, and avoids restarting ffplay on
every mouse-move.
