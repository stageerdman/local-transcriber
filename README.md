# Local Transcriber

A local Mac app for transcribing sales calls (and any other audio/video) entirely offline.

Open it from Finder/Spotlight, add a file or a whole folder, pick a model and language, and it
converts, transcribes locally with `mlx-whisper` on Apple Silicon/Metal, and writes a clean
`.md` transcript **next to each source file**. It tracks every transcription in a local history
and uses that history to predict how long the next one will take — the more you use it, the
more accurate the estimate gets.

No OpenAI API, no cloud service, and no external server are used.

## Supported Files

- `.mp3`
- `.m4a`
- `.wav`
- `.mp4`
- `.mov`
- `.mkv`
- `.webm`

## Models & Languages

Pick a model per run from the dropdown:

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

Each transcribed file gets a Markdown transcript written next to it:

```text
SelectedFolder/Alex/call.mp4
SelectedFolder/Alex/call.md      <- written here
```

The `.md` file has a small frontmatter block followed by the plain-text transcript:

```markdown
---
source: call.mp4
model: mlx-community/whisper-small-mlx
language: en
transcribed_at: 2026-07-31T10:00:00
elapsed_seconds: 612
---

<transcript text>
```

If a `.md` with that name already exists (e.g. two same-named files with different extensions in
one folder), it's written as `call (2).md`, `call (3).md`, etc.

Audio conversion to mp3 (for non-mp3 inputs) happens in a temporary directory and is cleaned up
after each transcription — it's an internal step, not an output.

## Time Estimates

Before each file starts transcribing, the app shows an estimated time based on the file's audio
duration and the average speed of past successful transcriptions with the same model (falling
back to the same model regardless of language, then to a rough built-in default if you've never
used that model before). Every completed transcription is recorded, so the estimate keeps
improving the more you use the app. The first run of a brand-new model will look slow in history
because it includes the one-time download — later runs pull the average back down.

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

From the `local-transcriber` folder:

```bash
brew install ffmpeg
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The first transcription with a given model may take longer because `mlx-whisper` downloads the
model files.

## Running the App

**As an installed app (recommended):**

```bash
./scripts/build_app.sh
```

This runs the test suite, then builds and installs `LocalTranscriber.app` into `/Applications`
(override the destination with `APP_INSTALL_DIR=~/Applications ./scripts/build_app.sh`). Launch
it from Finder or Spotlight like any other app — no Terminal window. Re-run this script any time
you change the code; the launcher always runs the current code in this project directory, so most
of the time you don't even need to rebuild, but re-running keeps the version stamp current.

**For development, without installing:**

```bash
source .venv/bin/activate
python3 -m app.main
```

Or double-click `run_transcriber.command`, which activates `.venv` and starts the app the same
way, keeping a Terminal window open so you can see logs.

## Runtime Behavior

- Opens a native macOS file/folder picker — "Add File..." for one or more files, "Add Folder..."
  to recursively queue every supported file inside it.
- Checks that `ffmpeg` is installed and `mlx-whisper` is importable, showing a banner if either is
  missing (rather than exiting).
- Processes queued files **one at a time** in the background (local Metal transcription is
  GPU-bound, so parallel jobs wouldn't be faster) while the UI stays responsive.
- Continues to the next file if one fails; the failure and its error are recorded in History.
- Writes clean text only, without timestamps.

## Architecture

```text
app/
  main.py       entry point: builds the Tk window, starts the background worker
  ui.py          MainWindow: Transcribe tab (add files/folder, model+language, live queue)
                 and History tab
  worker.py       background thread: probe duration -> estimate -> convert -> transcribe
                  -> write .md -> record history
  jobs.py          Job / WorkerEvent dataclasses passed between the UI and the worker thread
src/
  converter.py      ffmpeg mp3 conversion
  media_finder.py     recursive supported-file discovery
  file_pickers.py      native file/folder picker dialogs
  transcription.py      mlx-whisper wrapper
  duration.py             ffprobe-based audio/video duration probing
  history_db.py            SQLite history storage + time estimation
  md_writer.py              writes the .md transcript next to the source file
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
