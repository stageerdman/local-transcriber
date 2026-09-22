# Local Transcriber

A local Mac app for transcribing sales calls (and any other audio/video) entirely offline.

Open it from Finder/Spotlight, then either **drag and drop** a file/folder onto the window or use
"Add File.../Add Folder..." — files are added to a resizable list, each with its own model picker
and a ▶ button, so nothing transcribes until you say so. As soon as a file is added its estimated
transcription time shows up automatically (probed in the background), before you ever click ▶.
Pick a model per file and a language (global, in the toolbar), click ▶, and it converts,
transcribes locally with `mlx-whisper` on Apple Silicon/Metal, and writes a clean `.md` transcript
to **`~/Downloads`**. While it runs the button becomes ■ (click it to cancel immediately, no
partial output written) and you get a live progress bar with an estimated time remaining. Once a
file is done, stopped, or failed, its button becomes ↻, which clears it back to a fresh, unstarted
state so you can run it again — with a different model if you like. It tracks every transcription
in a local history and uses that history to predict how long the next one will take — the more you
use it, the more accurate the estimate gets.

No OpenAI API, no cloud service, and no external server are used.

## Multi-Speaker Transcripts

If the source file has **more than one audio track** - e.g. Zoom's "record a
separate audio file for each participant" - the app detects that
automatically (via `ffprobe`) and transcribes each track separately, so
who-said-what comes from the recording itself rather than a guess. The
transcript is then broken into pause-delimited, speaker-labeled blocks:

```markdown
**Person 1** [00:03]
Hey, so I wanted to go over the proposal from last week.

**Person 2** [00:11] (interrupting)
Actually, before that, can we talk about the timeline?
```

`(interrupting)` marks a block that started before the previous speaker's
block ended - i.e. real overlapping/talked-over speech, not just fast
turn-taking. This is reliable for multi-track sources since each mic is an
independent signal.

A source with just **one mixed-down track** is transcribed as plain text,
same as always - there's no reliable way to tell speakers apart from a
single waveform, so the app doesn't try or guess.

See `updates/multi-speaker-transcripts.md` for the design write-up, and
`src/dialogue.py` / `src/audio_tracks.py` for the implementation.

## Supported Files

- `.mp3`
- `.m4a`
- `.wav`
- `.mp4`
- `.mov`
- `.mkv`
- `.webm`

## Models & Languages

Every file in the list has its own model picker, so a batch can mix models freely — pick a quick
`Tiny` pass for one file and `Large-v3` for another, side by side. The picker is locked once a file
starts and unlocks again after Reset. Language is chosen once in the toolbar and applies to files
as they're added.

| Label | Repo | Notes |
|---|---|---|
| Tiny | `mlx-community/whisper-tiny-mlx` | fastest, lowest accuracy |
| Base | `mlx-community/whisper-base-mlx` | |
| Small (default) | `mlx-community/whisper-small-mlx` | good balance for English |
| Medium | `mlx-community/whisper-medium-mlx` | recommended for Czech and other non-English audio |
| Large-v3 | `mlx-community/whisper-large-v3-mlx` | best accuracy, slowest |
| Large-v3-turbo | `mlx-community/whisper-large-v3-turbo` | fast + accurate |

Language dropdown: Auto-detect, English, Czech, Slovak, German. The first transcription with a
model you haven't used before downloads/caches it locally via Hugging Face — that download time
is one-off and not representative of normal transcription speed (see the note on estimates
below).

## Output

Every transcribed file gets a Markdown transcript written to **`~/Downloads`**, regardless of
where the source file lives:

```text
~/Movies/Alex/call.mp4   (source, anywhere)
~/Downloads/call.md      <- written here
```

The `.md` file has a small frontmatter block followed by the plain-text transcript:

```markdown
---
source: /Users/you/Movies/Alex/call.mp4
model: mlx-community/whisper-small-mlx
language: en
transcribed_at: 2026-07-31T10:00:00
elapsed_seconds: 612
---

<transcript text>
```

For a multi-track source, two extra fields are added: `speakers` (how many
tracks/speakers were found) and `diarization: multitrack`. Single-track
transcripts don't get these fields - see Multi-Speaker Transcripts above.

If a `.md` with that name already exists in Downloads (e.g. two same-named source files), it's
written as `call (2).md`, `call (3).md`, etc.

Audio conversion to mp3 (for non-mp3 inputs) happens in a temporary directory and is cleaned up
after each transcription — it's an internal step, not an output.

## Start / Stop / Reset

Adding a file (via drag-and-drop or the pickers) never starts it automatically — it appears in the
list with a ▶ button so you control exactly when transcription begins and with which model. Each
file's button reflects its own state:

- **▶ Start** — ready, not yet running. Click to begin.
- **■ Stop** — converting, loading the model, or transcribing. Click to cancel; the in-progress
  transcription process is killed immediately (no `.md` is written, and the run is recorded in
  History as "stopped"). Only one file transcribes at a time, so a file that's merely waiting its
  turn also shows ■ and can be cancelled before it starts.
- **↻ Reset** — finished (done, error, or stopped). Click to clear the file back to a fresh state
  so you can transcribe it again, e.g. with a different model.

Transcription itself runs in a separate background process specifically so Stop can kill it
outright — Whisper's decode step has no built-in cancellation hook. That process is reused across
files (so a loaded model stays warm) and only gets torn down and restarted when you actually stop
a job.

## Progress and Time Estimates

As soon as a file is added, the app probes its audio duration in the background and shows an
estimated total time **before you click ▶** — based on the average speed of past successful
transcriptions with that file's model (falling back to the same model regardless of language, then
to a rough built-in default if you've never used that model before). Changing a file's model while
it's still idle instantly recalculates the estimate. Every completed transcription is recorded, so
the estimate keeps improving the more you use the app.

Once you click ▶ you get:
- A progress bar and "~Xm Ys remaining" readout for the active file, updated live.
- A per-row progress bar and elapsed time in the list for every running file.
- A distinct "Loading model" phase before transcribing starts — this is when a model you haven't
  used before gets downloaded from Hugging Face, which can take a while and isn't representative
  of normal transcription speed; later runs of the same model are fast and pull the average back
  down.

## Window Layout

The file list is a resizable table: widen the window and the file name column grows to use the
extra space, while the model/status/progress columns stay a fixed, readable width. It also
scrolls (mouse wheel/trackpad) once you've added more files than fit on screen.

## History

The **History** tab lists every past transcription: date, file, model, language, audio duration,
processing time, and status. It's backed by a local SQLite database at:

```text
~/Library/Application Support/LocalTranscriber/history.db
```

Your last-used model/language selection is remembered in:

```text
~/Library/Application Support/LocalTranscriber/settings.json
```

## Mac Setup

macOS's own Python (Xcode Command Line Tools) links against an ancient Tcl/Tk 8.5, which is known
to render Tkinter windows blank/broken on modern macOS. Use a Homebrew Python built with Tk 8.6+
instead:

```bash
brew install ffmpeg python-tk@3.11
/opt/homebrew/bin/python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

You can sanity-check the Tk version any time with:

```bash
python3 -c "import tkinter; print(tkinter.TkVersion)"   # must be >= 8.6
```

The first transcription with a given model may take longer because `mlx-whisper` downloads the
model files (shown as a "Loading model" step — see Progress above).

## Running the App

**As an installed app (recommended):**

```bash
./scripts/build_app.sh
```

This runs the test suite, then builds and installs `LocalTranscriber.app` (icon from `logo.jpg`)
into `/Applications` (override the destination with `APP_INSTALL_DIR=~/Applications
./scripts/build_app.sh`). Launch it from Finder or Spotlight like any other app — no Terminal
window. Re-run this script any time you change the code; the launcher always runs the current
code in this project directory, so most of the time you don't even need to rebuild, but
re-running keeps the version stamp current. The bundle is staged in a temp directory during the
build and never left behind in the project, so re-running never produces a second, stray app for
Spotlight to find.

**For development, without installing:**

```bash
source .venv/bin/activate
python3 -m app.main
```

Or double-click `run_transcriber.command`, which activates `.venv` and starts the app the same
way, keeping a Terminal window open so you can see logs.

## Runtime Behavior

- Drag and drop one or more files, or a whole folder, onto the drop zone (or the queue list) to
  add them to the list — folders are scanned recursively for supported files. "Add File..."/"Add
  Folder..." open native pickers as an alternative. Nothing transcribes until you click **Start**
  on a file (see Start / Stop / Reset above).
- Checks that `ffmpeg` is installed and `mlx-whisper` is importable in the background so the
  window is never blocked from appearing; shows a banner if either is missing (rather than
  exiting).
- Processes started files **one at a time** in the background (local Metal transcription is
  GPU-bound, so parallel jobs wouldn't be faster) while the UI stays responsive.
- Continues to the next file if one fails; the failure and its error are recorded in History.
- Writes clean text only. Multi-track sources get pause-based, speaker-labeled, timestamped
  blocks (see Multi-Speaker Transcripts above); single-track sources get plain text, without
  timestamps, same as before.

## Architecture

```text
app/
  main.py       entry point: builds the (drag-and-drop-capable) Tk window, starts the worker
  ui.py          MainWindow: Transcribe tab (drop zone, add files/folder, model+language,
                 live progress bar + ETA, per-file Start/Stop/Reset queue) and History tab
  worker.py       background thread: probe duration -> estimate -> detect audio tracks ->
                  multi-track: (per track: convert -> run job in the engine process,
                  killable on Stop) -> merge into dialogue blocks; single-track: convert
                  -> run job in the engine process -> join into plain text -> write .md
                  -> record history
  engine.py        long-lived child process (multiprocessing): loads/caches the model and
                    runs mlx-whisper transcription, returning per-segment timestamps;
                    killed and respawned fresh on Stop
  jobs.py          Job / WorkerEvent dataclasses passed between the UI and the worker thread;
                   Job carries a threading.Event used to request a stop
src/
  converter.py      ffmpeg mp3 conversion
  media_finder.py     recursive supported-file discovery
  file_pickers.py      native file/folder picker dialogs
  transcription.py      mlx-whisper wrapper (+ explicit model preload/download step),
                         incl. transcribe_mp3_segments() for per-segment timestamps
  audio_tracks.py         ffprobe multi-audio-track detection + ffmpeg per-track extraction
  dialogue.py               pause-based block merging, interruption detection, and
                             dialogue-format markdown rendering (multi-track transcripts only)
  duration.py             ffprobe-based audio/video duration probing
  history_db.py            SQLite history storage + time estimation
  md_writer.py              writes the .md transcript to ~/Downloads
  catalog.py                 model/language choices + default speed-estimate table
scripts/
  build_app.sh                builds and installs the .app bundle
```

## Running Automated Tests

```bash
source .venv/bin/activate
pytest
```

Covers media discovery, `.md` writing (including filename collisions), duration probing, and the
history/estimation logic (including that estimates improve once real history exists).
