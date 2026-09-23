"""Empirically determine the real track structure and test voice subtraction.

User's model: one track is a clean single voice (C); another is that voice PLUS
a second voice (M = C + other). Subtracting C from M should recover the other
voice. We verify which streams play which role and whether subtraction works.

Three probes:
  1. ACTIVE-SET analysis: per 250ms window, which streams have energy? If M is
     "host+prospect" and C is "prospect only", then C's active windows are ~a
     subset of M's, and M has extra active windows (host-only turns) where C is
     silent. This reveals who-contains-whom without any subtraction.
  2. SCALAR subtraction: residual = M - g*C (g = optimal). Residual energy +
     whether the residual decorrelates from C.
  3. FIR (Wiener) subtraction in freq domain: H = S_MC / S_CC; residual =
     M - filtered(C). Captures any delay/room filtering the scalar misses.
"""
import subprocess
import sys
import numpy as np
from scipy.signal import correlate, csd, welch

SR = 16000


def decode_mono(path, s, start, dur, sr=SR):
    cmd = ["ffmpeg", "-v", "error", "-ss", str(start), "-t", str(dur),
           "-i", path, "-map", f"0:{s}", "-ac", "1", "-ar", str(sr),
           "-f", "f32le", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).astype(np.float64)


def active_windows(x, sr=SR, win_ms=250, hop_ms=125, thresh_ratio=0.1):
    win = int(win_ms * sr / 1000); hop = int(hop_ms * sr / 1000)
    ref = np.sqrt(np.mean(x * x)) + 1e-12
    acts = []
    for s in range(0, len(x) - win + 1, hop):
        w = x[s:s + win]
        acts.append(np.sqrt(np.mean(w * w)) > thresh_ratio * ref)
    return np.array(acts)


def scalar_subtract(M, C):
    g = np.dot(M, C) / (np.dot(C, C) + 1e-12)
    R = M - g * C
    return g, R


def fir_subtract(M, C, sr=SR, nfft=4096):
    f, Scc = welch(C, sr, nperseg=nfft, return_onesided=False)
    f, Smc = csd(M, C, sr, nperseg=nfft, return_onesided=False)
    H = Smc / (Scc + 1e-9 * np.max(np.abs(Scc)))
    # apply H to C via overlap: do a straightforward FFT-domain filter on the
    # whole signal (approximate but fine for an energy read)
    n = len(C)
    Cf = np.fft.fft(C, n=nfft * ((n // nfft) + 1))
    # simpler: filter in blocks
    L = nfft
    h = np.fft.ifft(H).real
    h = np.fft.fftshift(h)
    from scipy.signal import fftconvolve
    est = fftconvolve(C, h, mode="same")
    R = M - est
    return R


def edb(x):
    return 10 * np.log10(np.mean(x * x) + 1e-12)


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    path = sys.argv[1]
    start = float(sys.argv[2]) if len(sys.argv) > 2 else 300.0
    dur = float(sys.argv[3]) if len(sys.argv) > 3 else 180.0
    streams = [1, 2, 3]
    tr = {s: decode_mono(path, s, start, dur) for s in streams}

    print(f"=== ACTIVE-SET analysis (start={start} dur={dur}) ===")
    act = {s: active_windows(tr[s]) for s in streams}
    n = min(len(a) for a in act.values())
    for s in streams:
        act[s] = act[s][:n]
        print(f"  stream {s}: active in {act[s].mean()*100:5.1f}% of windows")
    print("  containment  P(col active | row active):")
    print("         " + "   ".join(f"s{c}" for c in streams))
    for a in streams:
        row = []
        for b in streams:
            if a == b:
                row.append(" -- ")
                continue
            both = (act[a] & act[b]).sum()
            row.append(f"{both/ (act[a].sum()+1e-9):.2f}")
        print(f"    s{a}   " + "   ".join(row))
    print("  (row=given active, col=also active. If s3's row shows ~1.0 under a\n"
          "   stream X, s3 is active only when X is -> s3 subset of X -> X contains s3.)")

    print("\n=== SUBTRACTION: residual M - ref, for each ordered pair ===")
    print("  looking for: residual much lower energy than M AND decorrelated from"
          " ref\n  (that means ref was cleanly contained in M).")
    for M in streams:
        for C in streams:
            if M == C:
                continue
            g, Rs = scalar_subtract(tr[M], tr[C])
            Rf = fir_subtract(tr[M], tr[C])
            drop_s = edb(Rs) - edb(tr[M])
            drop_f = edb(Rf) - edb(tr[M])
            print(f"  M=s{M} - ref=s{C}:  scalar g={g:+.2f}  "
                  f"resid {drop_s:+5.1f} dB (corr_to_ref {corr(Rs,tr[C]):+.2f})   |   "
                  f"FIR resid {drop_f:+5.1f} dB (corr_to_ref {corr(Rf,tr[C]):+.2f})")


if __name__ == "__main__":
    main()
