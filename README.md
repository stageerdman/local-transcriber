# Local Transcriber

A standalone local Mac transcription tool for folders of English sales calls.

It opens a Finder-style folder picker, recursively finds supported audio/video files, converts non-MP3 files to MP3 with `ffmpeg`, transcribes locally with `faster-whisper`, and writes clean text transcripts into a timestamped folder under:

```text
~/Downloads/transcripts/YYYY-MM-DD_HH-MM-SS/
```

No OpenAI API, no cloud service, no database, and no server are used.

## Supported Files

- `.mp3`
- `.m4a`
- `.wav`
- `.mp4`
- `.mov`
- `.mkv`
- `.webm`

## Output Structure

Each run creates:

```text
~/Downloads/transcripts/YYYY-MM-DD_HH-MM-SS/
  text/
  audio/
  transcription_report.json
```

Output filenames are flat and based on the relative parent folders plus the original file stem.

Example:

```text
SelectedFolder/Alex/call.mp3
```

becomes:

```text
~/Downloads/transcripts/2026-05-18_14-30-00/text/Alex - call.txt
~/Downloads/transcripts/2026-05-18_14-30-00/audio/Alex - call.mp3
```

Deeper example:

```text
SelectedFolder/Sales Calls/Alex/call.mp4
```

becomes:

```text
Sales Calls - Alex - call.txt
Sales Calls - Alex - call.mp3
```

## Mac Setup

From the `local-transcriber` folder:

```bash
brew install ffmpeg
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
chmod +x run_transcriber.command
```

The first transcription may take longer because `faster-whisper` downloads the local `medium` model files.

## Launching

Double-click:

```text
run_transcriber.command
```

The launcher activates `.venv` if it exists, starts the Python script, and keeps Terminal open at the end so you can read the result.

## Runtime Behavior

- Opens a macOS folder picker.
- Checks that `ffmpeg` is installed.
- Checks that `faster-whisper` is importable.
- Finds supported media files recursively.
- Copies existing MP3s into the run `audio/` folder.
- Converts video and non-MP3 audio to MP3 into the run `audio/` folder.
- Transcribes locally with:
  - model: `medium`
  - language: English
  - `device="cpu"`
  - `compute_type="int8"`
  - `vad_filter=True`
- Writes clean text only, without timestamps.
- Continues processing if one file fails.
- Writes `transcription_report.json` at the end.

## Manual Test Checklist

1. Put one MP3 into a test folder.
2. Launch `run_transcriber.command`.
3. Select the test folder.
4. Confirm output appears in `~/Downloads/transcripts/<timestamp>/text/`.
5. Put one MP4 into a test folder.
6. Confirm MP3 appears in `audio/` and TXT appears in `text/`.
7. Confirm `transcription_report.json` exists.

## Running Automated Tests

```bash
source .venv/bin/activate
pytest
```

The automated tests cover filename generation, recursive media discovery, and report JSON generation.
