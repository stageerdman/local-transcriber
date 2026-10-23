"""Definitive who-is-who check: transcribe the clean track, the mix, and the
subtraction residual, and compare the words.

Hypothesis: s3 = one speaker only; s1 = s3 + other speaker; s1 - s3 = the other
speaker alone. If true:
  - transcript(s3)      shows speaker B only
  - transcript(s1)      shows BOTH speakers' words
  - transcript(s1 - s3) shows speaker A only (the words missing from s3)
"""
import subprocess
import sys
import numpy as np
from scipy.io import wavfile

sys.path.insert(0, "/Users/stage/dev/transcriber")
from src.transcription import transcribe_mp3_segments  # noqa: E402

SR = 16000
MODEL = "mlx-community/whisper-small-mlx"


def decode_mono(path, s, start, dur):
    cmd = ["ffmpeg", "-v", "error", "-ss", str(start), "-t", str(dur),
           "-i", path, "-map", f"0:{s}", "-ac", "1", "-ar", str(SR),
           "-f", "f32le", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).astype(np.float64)


def peak_norm(x, peak=0.7):
    m = np.max(np.abs(x)) + 1e-12
    return (x * (peak / m)).astype(np.float32)


def write_wav(path, x):
    wavfile.write(path, SR, peak_norm(x))


def transcribe(path):
    segs = transcribe_mp3_segments(MODEL, path, language=None)
    return " ".join(s.text for s in segs).strip()


def main():
    src = sys.argv[1]
    start = float(sys.argv[2]) if len(sys.argv) > 2 else 300.0
    dur = float(sys.argv[3]) if len(sys.argv) > 3 else 90.0

    s1 = decode_mono(src, 1, start, dur)
    s3 = decode_mono(src, 3, start, dur)
    n = min(len(s1), len(s3)); s1, s3 = s1[:n], s3[:n]
    g = np.dot(s1, s3) / (np.dot(s3, s3) + 1e-12)
    resid = s1 - g * s3
    print(f"window {start}-{start+dur}s, optimal g={g:.3f}")

    write_wav("_s1_mix.wav", s1)
    write_wav("_s3_clean.wav", s3)
    write_wav("_resid_s1_minus_s3.wav", resid)

    for label, path in [("s3 (clean track)", "_s3_clean.wav"),
                        ("s1 (mix track)", "_s1_mix.wav"),
                        ("s1 - s3 (residual)", "_resid_s1_minus_s3.wav")]:
        print(f"\n===== {label} =====")
        print(transcribe(path))


if __name__ == "__main__":
    main()
