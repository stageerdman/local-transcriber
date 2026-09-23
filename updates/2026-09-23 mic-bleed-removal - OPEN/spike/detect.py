"""Mic-bleed detector for the spike.

Question we answer, per ORDERED pair (target A, source B): "does B bleed into A?"
i.e. does A contain a scaled, delayed copy of B?

Key physics used for DIRECTION:
    If B bleeds into A, the copy of B inside A is DELAYED relative to B's own
    track (sound travels speaker_B -> mic_A, arriving later than at mic_B).
    With corr[tau] = sum_t A[t] * B[t-tau], a positive dominant lag means A
    follows B  ->  B leads  ->  B -> A. Negative lag => A -> B.

The cross-correlation MAGNITUDE is symmetric (same shared component seen from
either track), so direction is carried by the SIGN of the lag, and its
reliability by how consistent that lag is across windows. Magnitude gives the
strength/confidence used for tiering.

Metrics per unordered pair {A,B}:
  - rho        : aggregated normalized cross-correlation coefficient at best lag
                 (fraction-of-variance-shared proxy, sqrt scale). In [0,1].
  - lag        : dominant lag in samples (sign encodes direction).
  - lag_consistency : fraction of active windows whose best lag agrees (sign +
                 within tolerance) with the global dominant lag.
  - direction  : "B->A", "A->B", or "ambiguous".
  - tier       : "auto-clean" (confident no OR clean gating), "ask", "refuse".

We compute this globally *and* per-window (a mask) to prove gating regions are
locatable.
"""

import numpy as np
from scipy.signal import correlate, fftconvolve

SR = 16000


def _norm_xcorr_fft(a, b, max_lag):
    """Normalized cross-correlation of a,b over lags [-max_lag, +max_lag].
    Returns (lags, coeffs) where coeff = <a, shift(b, tau)> / (||a|| ||b||).
    corr[tau] with tau>0 means a follows b (a[t] ~ b[t-tau])."""
    a = a - a.mean()
    b = b - b.mean()
    na = np.sqrt(np.sum(a * a)) + 1e-12
    nb = np.sqrt(np.sum(b * b)) + 1e-12
    full = correlate(a, b, mode="full", method="fft")
    mid = len(b) - 1  # index of lag 0
    lo = mid - max_lag
    hi = mid + max_lag + 1
    seg = full[lo:hi]
    lags = np.arange(-max_lag, max_lag + 1)
    return lags, seg / (na * nb)


def _active(x, win, thresh_ratio=0.15):
    """Boolean: is this window energetically active (above thresh of global rms)?"""
    return np.sqrt(np.mean(x * x)) > thresh_ratio


def analyze_pair(A, B, sr=SR, win_ms=250, hop_ms=125, max_lag_ms=30):
    """Windowed analysis of unordered pair {A,B}. Returns a dict of metrics."""
    n = min(len(A), len(B))
    A = np.asarray(A[:n], dtype=np.float64)
    B = np.asarray(B[:n], dtype=np.float64)
    win = int(win_ms * sr / 1000)
    hop = int(hop_ms * sr / 1000)
    max_lag = int(max_lag_ms * sr / 1000)

    # global activity reference
    rmsA = np.sqrt(np.mean(A * A)) + 1e-12
    rmsB = np.sqrt(np.mean(B * B)) + 1e-12

    win_rhos = []
    win_lags = []
    win_centers = []
    win_active = []
    starts = range(0, n - win + 1, hop)
    for s in starts:
        aw = A[s:s + win]
        bw = B[s:s + win]
        act = (np.sqrt(np.mean(aw * aw)) > 0.15 * rmsA) and \
              (np.sqrt(np.mean(bw * bw)) > 0.15 * rmsB)
        lags, co = _norm_xcorr_fft(aw, bw, max_lag)
        k = np.argmax(np.abs(co))
        win_rhos.append(np.abs(co[k]))
        win_lags.append(lags[k])
        win_centers.append((s + win / 2) / sr)
        win_active.append(act)

    win_rhos = np.array(win_rhos)
    win_lags = np.array(win_lags)
    win_active = np.array(win_active)
    win_centers = np.array(win_centers)

    active_mask = win_active
    if active_mask.sum() < 2:
        return dict(rho=0.0, lag=0, lag_ms=0.0, direction="none",
                    lag_consistency=0.0, tier="auto-clean", n_active=int(active_mask.sum()),
                    win_rhos=win_rhos, win_lags=win_lags, win_active=win_active,
                    win_centers=win_centers, note="insufficient joint activity")

    # Dominant lag: weight each active window's lag by its rho, take the mode
    # via a weighted histogram over the lag axis.
    a_rhos = win_rhos[active_mask]
    a_lags = win_lags[active_mask]
    lag_axis = np.arange(-max_lag, max_lag + 1)
    hist = np.zeros(len(lag_axis))
    for l, r in zip(a_lags, a_rhos):
        hist[l + max_lag] += r
    # smooth histogram a touch
    if len(hist) > 5:
        k = np.hanning(5); k /= k.sum()
        hist = fftconvolve(hist, k, mode="same")
    dom_idx = np.argmax(hist)
    dom_lag = lag_axis[dom_idx]

    # rho aggregate = median of active-window rhos whose lag sign matches dominant
    tol = max(2, int(0.002 * sr))  # 2 ms tolerance
    agree = np.abs(a_lags - dom_lag) <= tol
    if agree.sum() == 0:
        agree = np.ones_like(a_lags, dtype=bool)
    rho = float(np.median(a_rhos[agree]))
    lag_consistency = float(agree.mean())

    # detect mutual/ambiguous: strong mass at BOTH positive and negative lags
    pos_mass = hist[lag_axis > tol].sum()
    neg_mass = hist[lag_axis < -tol].sum()
    total_mass = pos_mass + neg_mass + 1e-12
    # symmetry: min/max of the two side masses (1.0 = perfectly mutual)
    side_symmetry = min(pos_mass, neg_mass) / (max(pos_mass, neg_mass) + 1e-12)

    if abs(dom_lag) <= tol:
        direction = "ambiguous-zero-lag"
    elif dom_lag > 0:
        direction = "B->A"
    else:
        direction = "A->B"

    return dict(
        rho=rho,
        lag=int(dom_lag),
        lag_ms=float(dom_lag / sr * 1000),
        direction=direction,
        lag_consistency=lag_consistency,
        side_symmetry=float(side_symmetry),
        n_active=int(active_mask.sum()),
        win_rhos=win_rhos, win_lags=win_lags, win_active=win_active,
        win_centers=win_centers,
    )


# ---- Tiering ------------------------------------------------------------
# Thresholds are set from the sweep (see sweep.py output). Documented there.
RHO_CLEAN = 0.12      # below this: no meaningful shared component -> clean
RHO_STRONG = 0.22     # above this (with consistent directional lag): confident bleed
CONSIST_MIN = 0.6     # fraction of active windows that must agree on lag
SYMMETRY_MUTUAL = 0.30 # side-mass symmetry above this = mutual/too-ambiguous
                       # (genuine single-direction bleed measures ~0.00; even
                       #  perfectly-equal mutual bleed measures >=0.45, so 0.30
                       #  is a safe cut with wide margin either side)


def decide(metrics):
    """Map metrics -> (verdict, tier, reason).

    verdict: 'no-bleed' | 'bleed <dir>' | 'mutual'
    tier: 'auto-clean' | 'ask' | 'refuse'
    """
    rho = metrics["rho"]
    cons = metrics["lag_consistency"]
    sym = metrics.get("side_symmetry", 0.0)
    direction = metrics["direction"]

    # No meaningful correlation at all -> confidently clean, nothing to gate.
    if rho < RHO_CLEAN or metrics["n_active"] < 2:
        return ("no-bleed", "auto-clean", f"rho {rho:.3f} < {RHO_CLEAN}: no shared component")

    # Strong mutual/bidirectional correlation -> can't safely gate.
    if sym > SYMMETRY_MUTUAL and rho >= RHO_STRONG:
        return ("mutual", "refuse", f"bidirectional lag mass (sym {sym:.2f}), rho {rho:.3f}: too ambiguous to gate")

    # Directional and strong and consistent -> confident bleed, auto-clean gate.
    if rho >= RHO_STRONG and cons >= CONSIST_MIN and direction in ("B->A", "A->B"):
        return (f"bleed {direction}", "auto-clean", f"rho {rho:.3f}, consistency {cons:.2f}, lag {metrics['lag_ms']:.1f} ms")

    # Zero-lag dominant but correlated -> can't resolve direction.
    if direction == "ambiguous-zero-lag":
        return ("ambiguous", "refuse", f"dominant lag ~0 (rho {rho:.3f}): direction unresolvable")

    # Everything in between -> ask.
    return (f"bleed? {direction}", "ask", f"rho {rho:.3f}, consistency {cons:.2f}: uncertain")


def gating_mask(metrics, rho_gate=None):
    """Per-window boolean mask: 'A is (in part) a delayed copy of B here'.
    A window qualifies if it is active, its best lag agrees with the dominant
    direction, and its local rho exceeds the gate threshold."""
    if rho_gate is None:
        rho_gate = RHO_CLEAN
    dom = metrics["lag"]
    tol = max(2, int(0.002 * SR))
    mask = metrics["win_active"] & \
        (np.abs(metrics["win_lags"] - dom) <= tol) & \
        (metrics["win_rhos"] >= rho_gate)
    return mask
