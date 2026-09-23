"""Understand the real recording's channel structure.

Decodes every channel of every audio stream, reports energy, and cross-correlates
all channel pairs at zero lag (are they the same signal?) and at best lag (is one
a delayed copy of another?). Also checks exact digital identity.
"""
import subprocess
import sys
import numpy as np
from scipy.signal import correlate

SR = 16000


def decode_channel(path, stream_index, ch, start=None, dur=None, sr=SR):
    cmd = ["ffmpeg", "-v", "error"]
    if start is not None:
        cmd += ["-ss", str(start)]
    if dur is not None:
        cmd += ["-t", str(dur)]
    cmd += ["-i", path, "-map", f"0:{stream_index}",
            "-af", f"pan=mono|c0=c{ch}", "-ar", str(sr), "-f", "f32le", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).astype(np.float64)


def zero_lag_corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def best_lag(a, b, max_lag):
    a = a - a.mean(); b = b - b.mean()
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    full = correlate(a, b, mode="full", method="fft") / (na * nb + 1e-12)
    mid = len(b) - 1
    seg = full[mid - max_lag: mid + max_lag + 1]
    lags = np.arange(-max_lag, max_lag + 1)
    k = np.argmax(np.abs(seg))
    return lags[k], float(seg[k])


def rms_db(x):
    return 20 * np.log10(np.sqrt(np.mean(x * x)) + 1e-12)


def main():
    path = sys.argv[1]
    start = float(sys.argv[2]) if len(sys.argv) > 2 else 300.0
    dur = float(sys.argv[3]) if len(sys.argv) > 3 else 120.0
    streams = [1, 2, 3]

    chans = {}
    print(f"start={start} dur={dur}")
    for s in streams:
        for c in (0, 1):
            x = decode_channel(path, s, c, start, dur)
            chans[(s, c)] = x
            print(f"  {s}.{c}: rms={rms_db(x):7.1f} dBFS")

    keys = list(chans)
    max_lag = int(0.03 * SR)
    print("\nzero-lag corr / best-lag corr (lag in ms) between all channels:")
    print("        " + "  ".join(f"{s}.{c}" for (s, c) in keys))
    for ka in keys:
        row = []
        for kb in keys:
            if ka == kb:
                row.append("  --   ")
                continue
            z = zero_lag_corr(chans[ka], chans[kb])
            row.append(f"{z:+.2f}")
        print(f"  {ka[0]}.{ka[1]}   " + "   ".join(row))

    print("\nbest-lag detail for non-trivial pairs (|zero-lag corr|<0.999):")
    seen = set()
    for i, ka in enumerate(keys):
        for kb in keys[i+1:]:
            z = zero_lag_corr(chans[ka], chans[kb])
            lag, c = best_lag(chans[ka], chans[kb], max_lag)
            exact = np.array_equal(chans[ka], chans[kb])
            print(f"  {ka} vs {kb}: zero-lag={z:+.3f}  best-lag={lag/SR*1000:+.2f}ms "
                  f"corr={c:+.3f}  exact-identical={exact}")


if __name__ == "__main__":
    main()
