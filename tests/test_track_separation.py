"""Tests for speaker separation on mixed / echoed multi-track recordings.

Synthetic "voices" (distinct spectra, on/off speech envelopes) let us build the
exact relationships the classifier must recognise, with no audio files.
"""
import numpy as np
import pytest

from src.track_separation import (
    SeparationPlan,
    TrackRole,
    TrackSignal,
    classify_tracks,
    isolate,
    noise_gate,
    render_speaker_wav,
    write_wav,
)

RATE = 16000


def _voice(seed: int, duration_s: float = 30.0, on_prob: float = 0.6, rate: int = RATE) -> np.ndarray:
    """A distinct band-limited 'voice' that talks in bursts (silence between)."""
    rng = np.random.default_rng(seed)
    n = int(duration_s * rate)
    # distinct timbre: a few random formant tones + fricative noise
    t = np.arange(n) / rate
    tone = np.zeros(n)
    for f in rng.uniform(120, 3000, size=4):
        tone += np.sin(2 * np.pi * f * t + rng.uniform(0, 6.28))
    tone += 0.3 * rng.standard_normal(n)
    # speech on/off envelope in ~300 ms blocks
    block = int(0.3 * rate)
    env = np.zeros(n)
    for s in range(0, n, block):
        if rng.random() < on_prob:
            env[s:s + block] = 1.0
    # smooth the envelope edges
    k = np.ones(int(0.02 * rate)) / int(0.02 * rate)
    env = np.convolve(env, k, mode="same")
    sig = tone * env
    return sig / (np.max(np.abs(sig)) + 1e-9)


def _sig(index: int, samples: np.ndarray, name: str = "") -> TrackSignal:
    return TrackSignal(index=index, name=name or f"Person {index}", samples=samples)


# --- classification ----------------------------------------------------------
def test_contained_voice_is_isolated_and_reference_kept():
    """C clean, M = C + other, M2 = M duplicate -> keep C, isolate M, drop M2."""
    a = _voice(1, on_prob=0.55)          # local speaker (talks less here)
    c = _voice(2, on_prob=0.8)           # remote speaker, clean track
    mix = a + c                          # the mic: local + remote at unity gain
    mix2 = mix.copy()                    # duplicate mix

    plan = classify_tracks([
        _sig(1, mix, "Mic"),
        _sig(2, c, "Remote"),
        _sig(3, mix2, "Mic copy"),
    ])

    remote = plan.role_for(2)
    microle = plan.role_for(1)
    dup = plan.role_for(3)

    assert remote.kind == "keep"
    assert microle.kind == "isolate"
    assert 2 in microle.reference_indices        # subtract the remote track
    assert dup.kind == "duplicate"
    assert dup.duplicate_of in (1, 3)            # duplicate of one of the mixes
    assert len(plan.speaker_roles()) == 2        # two real speakers out


def test_independent_tracks_are_all_kept():
    a = _voice(10, on_prob=0.5)
    b = _voice(11, on_prob=0.5)
    plan = classify_tracks([_sig(1, a), _sig(2, b)])
    assert all(r.kind == "keep" for r in plan.roles)
    assert plan.is_trivial()  # trivial plan == all keep


def test_pure_duplicate_pair_drops_one():
    a = _voice(20, on_prob=0.6)
    plan = classify_tracks([_sig(1, a), _sig(2, a.copy())])
    kinds = sorted(r.kind for r in plan.roles)
    assert kinds == ["duplicate", "keep"]


def test_single_track_is_kept():
    plan = classify_tracks([_sig(1, _voice(30))])
    assert plan.role_for(1).kind == "keep"
    assert plan.is_trivial()


# --- isolation ---------------------------------------------------------------
def test_isolate_recovers_the_unique_voice():
    a = _voice(40, on_prob=0.6)
    c = _voice(41, on_prob=0.7)
    mix = a + c
    recovered = isolate(mix, [c])
    # recovered should look like `a`, not like `c`
    corr_a = abs(np.corrcoef(recovered, a)[0, 1])
    corr_c = abs(np.corrcoef(recovered, c)[0, 1])
    assert corr_a > 0.9
    assert corr_c < 0.2


def test_isolate_with_no_reference_returns_mix():
    a = _voice(42)
    assert np.allclose(isolate(a, []), a)


def test_noise_gate_silences_inactive_regions():
    c = _voice(50, on_prob=0.7)
    a = _voice(51, on_prob=0.3)  # talks little -> lots of silence in residual
    residual = isolate(a + c, [c])
    gated = noise_gate(residual)
    # energy concentrates where `a` is active; total energy should drop notably
    assert np.mean(gated ** 2) < np.mean(residual ** 2)
    assert len(gated) == len(residual)


# --- rendering ---------------------------------------------------------------
def test_render_speaker_wav_writes_readable_wav(tmp_path):
    import wave

    a = _voice(60, on_prob=0.6)
    c = _voice(61, on_prob=0.7)
    signals = {1: a + c, 2: c}
    role = TrackRole(index=1, name="Mic", kind="isolate", reference_indices=(2,))
    out = tmp_path / "spk.wav"
    render_speaker_wav(role, signals, out)
    assert out.exists()
    with wave.open(str(out), "rb") as w:
        assert w.getnchannels() == 1
        assert w.getframerate() == RATE
        assert w.getnframes() > 0


def test_write_wav_roundtrips_shape(tmp_path):
    import wave

    x = np.linspace(-0.5, 0.5, RATE).astype(np.float64)
    out = tmp_path / "x.wav"
    write_wav(out, x)
    with wave.open(str(out), "rb") as w:
        assert w.getnframes() == len(x)
