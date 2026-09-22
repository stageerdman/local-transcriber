# Transcription benchmark

Compare transcription backends on the same sample file: how long each one
takes, and (via the rating UI) how good the output actually is.

## Setup

Uses the same Python env as `local-transcriber` (mlx-whisper, parakeet-mlx,
ffmpeg). The benchmark runner and rating server themselves are stdlib-only;
a couple of backends need extra packages:

```
cd /Users/stage/dev/transcriber/local-transcriber
source .venv/bin/activate   # or however you normally activate it
cd ../testing
pip install -r requirements-testing.txt   # silicon-asr, groq - skip if not using those backends
```

The Groq backends also need an API key:

```
export GROQ_API_KEY=...   # console.groq.com/keys
```

## 1. Add a sample

Drop a video (or audio) file into `videos/`. If there's exactly one file
there, `run_benchmark.py` picks it up automatically.

## 2. Run the benchmark

```
python run_benchmark.py                     # uses the file in videos/, all backends
python run_benchmark.py path/to/clip.mp4    # explicit file
python run_benchmark.py --list-models       # see backend ids
python run_benchmark.py --models whisper-small-mlx,parakeet-tdt-0.6b-v3
```

Audio is extracted once to `results/audio/` and reused across runs/backends
(`--force-audio` to redo it). Each backend's timing and transcript get
written into `results/results.json` and `results/transcripts/<id>.txt`.
Re-running only the backends you name leaves everything else (including
ratings you've already given) untouched.

Currently configured backends:

- **mlx-whisper** (GPU, local): tiny / base / small / medium / large-v3 /
  large-v3-turbo - same as `local-transcriber`'s own model list.
- **parakeet-mlx** (GPU, local): Parakeet TDT v3, also from the app.
- **silicon-asr** (Apple Neural Engine, local): Parakeet TDT v3, TDT v2, and
  CTC-110M, run on the ANE via CoreML instead of the GPU - same model
  family as parakeet-mlx above, different hardware path. Needs
  `pip install -r requirements-testing.txt`.
- **Groq API** (cloud): Whisper Large-v3 and Large-v3-turbo, hosted on
  Groq's LPU hardware - not a different model, just a much faster place to
  run the same weights. Needs `groq` installed and `GROQ_API_KEY` set.

See `models_config.py` to add more - a new backend just needs a
`run(audio_path, language) -> TranscribeResult` function.

## 3. Rate quality

```
python rate_server.py
```

Opens a local page listing every backend's transcript side by side (fastest
first), with a 1-5 star rating and an optional notes field per backend.
Ratings save immediately to `results/results.json` - no separate export
step.

## Layout

```
testing/
  videos/                 # drop sample media here (gitignored contents)
  results/
    audio/                # extracted mp3s, cached
    transcripts/          # one .txt per backend
    results.json          # timings + ratings + notes, source of truth
  models_config.py         # backend registry - edit to add/remove models
  run_benchmark.py          # CLI: extract audio, run backends, log speed
  rate_server.py             # local server for the rating UI
  static/rate.html            # the rating UI itself
```
