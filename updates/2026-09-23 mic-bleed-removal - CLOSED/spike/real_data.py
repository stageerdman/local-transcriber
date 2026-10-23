"""Run the bleed detector against a REAL multi-track recording.

Decodes each audio stream to mono 16 kHz via ffmpeg, runs analyze_pair/decide
on every unordered pair, and reports the metrics + per-pair energy so we can
re-fit the tier thresholds on real-room reverb (the spike caveat).

Usage:
    python real_data.py "<path to .mov>" [start_sec] [dur_sec]
"""
import subprocess
import sys
import numpy as np

import detect

SR = 16000


def probe_audio_indices(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=index", "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True).stdout
    return [int(x) for x in out.split() if x.strip()]


def decode_mono(path, stream_index, start=None, dur=None, sr=SR):
    cmd = ["ffmpeg", "-v", "error"]
    if start is not None:
        cmd += ["-ss", str(start)]
    if dur is not None:
        cmd += ["-t", str(dur)]
    cmd += ["-i", path, "-map", f"0:{stream_index}",
            "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).astype(np.float64)


def rms_db(x):
    r = np.sqrt(np.mean(x * x)) + 1e-12
    return 20 * np.log10(r)


def main():
    path = sys.argv[1]
    start = float(sys.argv[2]) if len(sys.argv) > 2 else None
    dur = float(sys.argv[3]) if len(sys.argv) > 3 else None

    idxs = probe_audio_indices(path)
    print(f"audio streams: {idxs}  (start={start}, dur={dur})")
    tracks = {}
    for i in idxs:
        x = decode_mono(path, i, start, dur)
        tracks[i] = x
        print(f"  stream {i}: {len(x)/SR:6.1f}s  rms={rms_db(x):6.1f} dBFS  "
              f"peak={20*np.log10(np.max(np.abs(x))+1e-12):6.1f} dBFS")

    print("\n=== pairwise analysis ===")
    keys = list(tracks)
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            ia, ib = keys[a], keys[b]
            m = detect.analyze_pair(tracks[ia], tracks[ib])
            verdict, tier, reason = detect.decide(m)
            # direction naming: B->A means the SECOND arg (ib) leads -> ib bleeds into ia
            dirtxt = {"B->A": f"{ib}->{ia}", "A->B": f"{ia}->{ib}"}.get(
                m["direction"], m["direction"])
            gate = detect.gating_mask(m)
            gated_frac = gate.mean() if len(gate) else 0.0
            print(f"\n  pair ({ia},{ib}):")
            print(f"    rho={m['rho']:.3f}  lag={m['lag_ms']:+.1f}ms  "
                  f"consistency={m['lag_consistency']:.2f}  "
                  f"symmetry={m.get('side_symmetry', 0):.2f}  "
                  f"n_active={m['n_active']}")
            print(f"    direction: {dirtxt}")
            print(f"    VERDICT: {verdict}  [{tier}]  -- {reason}")
            print(f"    gating mask: {gated_frac*100:.0f}% of windows would be gated")


if __name__ == "__main__":
    main()
