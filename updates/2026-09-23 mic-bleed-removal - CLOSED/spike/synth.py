"""Synthetic multi-track voice + bleed generator for the mic-bleed spike.

No external files. Produces clean "voices" (distinct spectra + speech-like
on/off envelopes) at 16 kHz, then synthesizes bleed:

    contaminated_A = clean_A + alpha * reverb(shift(clean_B, tau_samples))

All tracks share t=0, duration, and sample rate (they are streams in one
container), so no global alignment is required.
"""

import numpy as np

SR = 16000


def _rng(seed):
    return np.random.default_rng(seed)


def speech_envelope(n, sr, rng, syllable_hz=4.0, on_frac=0.55):
    """Speech-like amplitude envelope: bursts of energy (syllables) with gaps.

    Returns an envelope in [0,1] with roughly `on_frac` of the time active.
    """
    # Slow syllable modulation
    t = np.arange(n) / sr
    # random phase per call
    phase = rng.uniform(0, 2 * np.pi)
    syl = 0.5 * (1 + np.sin(2 * np.pi * syllable_hz * t + phase))
    # gate into on/off "words" at ~0.5-1.5 s scale
    word_hz = rng.uniform(0.4, 0.9)
    word = 0.5 * (1 + np.sin(2 * np.pi * word_hz * t + rng.uniform(0, 2 * np.pi)))
    gate = (word > (1 - on_frac)).astype(float)
    # smooth the gate edges to avoid clicks
    from scipy.signal import fftconvolve
    win = np.hanning(int(0.02 * sr))
    win /= win.sum()
    gate = fftconvolve(gate, win, mode="same")
    env = syl * gate
    return env


def voice(n, sr, rng, formants, syllable_hz=4.0, on_frac=0.55):
    """A distinct 'voice': summed harmonics with formant-ish shaping under a
    speech envelope. formants = list of (center_hz, bw_hz, gain)."""
    t = np.arange(n) / sr
    # a wandering fundamental
    f0 = rng.uniform(90, 180)
    f0_track = f0 * (1 + 0.05 * np.sin(2 * np.pi * 0.7 * t + rng.uniform(0, 6)))
    sig = np.zeros(n)
    # harmonic stack
    phase = np.cumsum(2 * np.pi * f0_track / sr)
    for h in range(1, 40):
        fh = h * f0
        if fh > sr / 2 - 200:
            break
        # formant shaping: gain by proximity to formant centers
        g = 0.0
        for (fc, bw, fg) in formants:
            g += fg * np.exp(-((fh - fc) ** 2) / (2 * bw ** 2))
        sig += (g / h) * np.sin(h * phase + rng.uniform(0, 2 * np.pi))
    # add a little band-limited noise for fricatives, shaped by top formant
    noise = rng.standard_normal(n)
    from scipy.signal import butter, sosfilt
    sos = butter(4, [1500 / (sr / 2), 6000 / (sr / 2)], btype="band", output="sos")
    fric = sosfilt(sos, noise) * 0.15
    sig = sig + fric
    env = speech_envelope(n, sr, rng, syllable_hz, on_frac)
    sig = sig * env
    # normalize to unit RMS (over active regions)
    rms = np.sqrt(np.mean(sig ** 2) + 1e-12)
    return (sig / rms).astype(np.float64)


# Three distinct formant profiles (speaker-like)
VOICE_A = [(600, 80, 1.0), (1000, 120, 0.7), (2400, 200, 0.4)]
VOICE_B = [(400, 70, 1.0), (1700, 150, 0.8), (2900, 250, 0.5)]
VOICE_C = [(730, 90, 1.0), (1220, 130, 0.6), (2600, 220, 0.45)]


def reverb_fir(rng, sr, decay=0.0025, taps=12):
    """Light early-reflection FIR (a few decaying taps over ~2-6 ms)."""
    h = np.zeros(int(0.006 * sr))
    h[0] = 1.0
    n_refl = taps
    idx = rng.integers(1, len(h), size=n_refl)
    amp = rng.uniform(0.05, 0.25, size=n_refl) * np.exp(-idx / (decay * sr))
    for i, a in zip(idx, amp):
        h[i] += a
    return h


def shift(x, tau):
    """Delay x by tau samples (tau >= 0), zero-padded at the front."""
    if tau == 0:
        return x.copy()
    y = np.zeros_like(x)
    y[tau:] = x[:-tau]
    return y


def bleed_into(clean_target, clean_source, alpha, tau, rng, sr, add_reverb=True):
    """target picks up source: target + alpha * reverb(shift(source, tau))."""
    from scipy.signal import fftconvolve
    s = shift(clean_source, tau)
    if add_reverb:
        h = reverb_fir(rng, sr)
        s = fftconvolve(s, h, mode="full")[: len(clean_target)]
    return clean_target + alpha * s


def make_pair(seed, alpha_ba=0.0, tau_ba=0, alpha_ab=0.0, tau_ab=0,
              dur=6.0, sr=SR, on_frac_a=0.55, on_frac_b=0.55, reverb=True):
    """Build a contaminated pair (A, B).

    alpha_ba: how much B bleeds INTO A (B->A). tau_ba: lag samples.
    alpha_ab: how much A bleeds INTO B (A->B).
    Returns (A, B, clean_A, clean_B).
    """
    rng = _rng(seed)
    n = int(dur * sr)
    ca = voice(n, sr, rng, VOICE_A, syllable_hz=4.2, on_frac=on_frac_a)
    cb = voice(n, sr, rng, VOICE_B, syllable_hz=3.6, on_frac=on_frac_b)
    A = ca.copy()
    B = cb.copy()
    if alpha_ba > 0:
        A = bleed_into(ca, cb, alpha_ba, tau_ba, rng, sr, reverb)
    if alpha_ab > 0:
        B = bleed_into(cb, ca, alpha_ab, tau_ab, rng, sr, reverb)
    # small independent mic noise floor
    A = A + 0.002 * rng.standard_normal(n)
    B = B + 0.002 * rng.standard_normal(n)
    return A, B, ca, cb


def make_triple(seed, dur=6.0, sr=SR):
    """User's exact scenario: P3 clean, P2 = P2 + bleed(P3), P1 clean.
    Returns dict of tracks p1,p2,p3 plus clean references."""
    rng = _rng(seed)
    n = int(dur * sr)
    p1c = voice(n, sr, rng, VOICE_A, syllable_hz=4.2, on_frac=0.5)
    p2c = voice(n, sr, rng, VOICE_B, syllable_hz=3.6, on_frac=0.5)
    p3c = voice(n, sr, rng, VOICE_C, syllable_hz=4.8, on_frac=0.5)
    p1 = p1c + 0.002 * rng.standard_normal(n)
    p3 = p3c + 0.002 * rng.standard_normal(n)
    # P2 picks up P3 (alpha 0.3, tau 6 ms)
    tau = int(0.006 * sr)
    p2 = bleed_into(p2c, p3c, 0.3, tau, rng, sr, add_reverb=True)
    p2 = p2 + 0.002 * rng.standard_normal(n)
    return {"p1": p1, "p2": p2, "p3": p3,
            "p1c": p1c, "p2c": p2c, "p3c": p3c, "tau_p3_into_p2": tau}
