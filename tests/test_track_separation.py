"""Tests for speaker separation on mixed / echoed multi-track recordings.

Synthetic "voices" (distinct spectra, on/off speech envelopes) let us build the
exact relationships the classifier must recognise, with no audio files.
"""
import shutil

import numpy as np
import pytest

from src.track_separation import (
    DEFAULT_RNNOISE_MODEL,
    SeparationPlan,
    TrackRole,
    TrackSignal,
    classify_tracks,
    denoise_pcm,
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


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or not DEFAULT_RNNOISE_MODEL.exists(),
    reason="ffmpeg or the RNNoise model is unavailable",
)
def test_denoise_preserves_length_and_reduces_overall_noise():
    rng = np.random.default_rng(0)
    speech = _voice(70, on_prob=0.6)
    noise = 0.05 * rng.standard_normal(len(speech))
    out = denoise_pcm(speech + noise, RATE)
    assert len(out) == len(speech)
    # denoising a mostly-quiet signal should not raise its overall energy
    assert np.mean(out ** 2) <= np.mean((speech + noise) ** 2) + 1e-6


def test_denoise_no_model_is_identity():
    x = _voice(71)
    assert np.allclose(denoise_pcm(x, RATE, model_path=None), x)


# --- shared-voice safety net (detect_shared_voice) ---------------------------
from src.track_separation import (  # noqa: E402
    SharedVoiceWarning,
    detect_shared_voice,
    gate_to_reference,
)


def test_shared_voice_flags_clean_source_inside_mix_as_contains():
    """s3 = clean voice; s1 = s3 + another speaker. Transcribing both would
    double s3 -> warn 'contains' (mix=s1, component=s3)."""
    # The clean component (s3) is active a SUBSET of the time; the mix (s1) is
    # also active during the other speaker's solo turns - matching the real
    # OBS "call audio + mic" structure.
    component = _voice(10, on_prob=0.45)    # clean source, talks in bursts
    other = _voice(11, on_prob=0.75)        # the mix's other, busier speaker
    mix = component + other
    warns = detect_shared_voice([_sig(1, mix), _sig(3, component)])
    assert len(warns) == 1
    w = warns[0]
    assert w.kind == "contains"
    assert w.mix_index == 1 and w.component_index == 3


def test_shared_voice_silent_on_cleanly_separated_speakers():
    """A clean source and the isolated complement (mix - source) share no voice
    -> no warning (so a correctly-configured job is never nagged)."""
    client = _voice(12, on_prob=0.8)
    coach = _voice(13, on_prob=0.5)
    mix = client + coach
    isolated_coach = isolate(mix, [client])
    signals = [_sig(3, client), _sig(1, isolated_coach)]
    assert detect_shared_voice(signals) == []


def test_shared_voice_flags_duplicate_mixes():
    """Two different mixes of the SAME two speakers (similar activity, neither a
    clean unity component) -> 'duplicate' (drop one)."""
    a = _voice(14, on_prob=0.7)
    b = _voice(15, on_prob=0.7)
    mix1 = a + b
    mix2 = 0.8 * a + 1.2 * b                 # same voices, different blend
    warns = detect_shared_voice([_sig(1, mix1), _sig(2, mix2)])
    assert len(warns) == 1
    assert warns[0].kind == "duplicate"


def test_shared_voice_silent_on_independent_speakers():
    """Two unrelated single-voice tracks share nothing -> no warning."""
    assert detect_shared_voice([_sig(1, _voice(16)), _sig(2, _voice(17))]) == []


# --- mix-relative gate (gate_to_reference) -----------------------------------
def test_gate_to_reference_zeroes_silent_isolated_stretch():
    """Where the isolated speaker is silent (residual << mix), the gate must
    drive the output to true silence so ASR/VAD don't hallucinate on it."""
    # Second half: the isolated speaker is silent but an imperfect subtraction
    # left faint leftover, while the mix stays loud (the other speaker). That
    # leftover is what Whisper hallucinates on, and the gate must kill it.
    n = 20 * RATE
    half = n // 2
    coach = _voice(22, duration_s=20.0, on_prob=0.9)[:n]
    other = _voice(23, duration_s=20.0, on_prob=0.9)[:n]
    mix = np.zeros(n)
    mix[:half] = coach[:half] + 0.3 * other[:half]   # coach really talking
    mix[half:] = other[half:]                        # only the other speaker
    residual = np.zeros(n)
    residual[:half] = coach[:half]                   # isolated coach, present
    residual[half:] = 0.03 * other[half:]            # faint uncancelled leftover
    gated = gate_to_reference(residual, mix, RATE)
    first = float(np.sum(gated[:half] ** 2))
    second = float(np.sum(gated[half:] ** 2))
    assert first > 0.3 * float(np.sum(residual[:half] ** 2))   # real speech kept
    assert second < 0.05 * float(np.sum(residual[half:] ** 2))  # leftover killed


def test_gate_to_reference_keeps_isolated_speech():
    """Where the isolated speaker really talks (residual comparable to mix), the
    gate must keep the audio."""
    coach = _voice(22, on_prob=0.9)
    mix = coach.copy()                       # no other speaker -> residual == mix
    residual = mix.copy()
    gated = gate_to_reference(residual, mix, RATE)
    assert np.mean(gated ** 2) > 0.3 * np.mean(residual ** 2)


def test_gate_to_reference_short_input_is_passthrough():
    x = np.ones(100)
    assert np.allclose(gate_to_reference(x, x, RATE), x)


# --- resolve_shared_voice (suggested fix) ------------------------------------
from src.track_separation import resolve_shared_voice  # noqa: E402


def test_resolve_common_3track_keeps_source_isolates_mix_drops_duplicate():
    # The canonical case: s1 is a mix of s3 (+other); s2 is a redundant mix.
    warns = [
        SharedVoiceWarning(mix_index=1, component_index=3, kind="contains", drop_db=5),
        SharedVoiceWarning(mix_index=2, component_index=1, kind="duplicate", drop_db=5),
        SharedVoiceWarning(mix_index=3, component_index=2, kind="duplicate", drop_db=5),
    ]
    subs, drop = resolve_shared_voice([1, 2, 3], warns)
    assert subs == {1: [3]}          # isolate s1 - s3
    assert drop == {2}               # drop the redundant mix
    assert 3 not in drop             # clean source kept


def test_resolve_never_drops_everything():
    # All mutual duplicates (e.g. identical tracks) -> keep at least one.
    warns = [
        SharedVoiceWarning(mix_index=2, component_index=1, kind="duplicate", drop_db=5),
        SharedVoiceWarning(mix_index=3, component_index=1, kind="duplicate", drop_db=5),
        SharedVoiceWarning(mix_index=3, component_index=2, kind="duplicate", drop_db=5),
    ]
    subs, drop = resolve_shared_voice([1, 2, 3], warns)
    assert len(set([1, 2, 3]) - drop) >= 1


def test_resolve_ignores_warnings_about_unselected_tracks():
    warns = [SharedVoiceWarning(mix_index=1, component_index=3, kind="contains", drop_db=5)]
    subs, drop = resolve_shared_voice([1, 2], warns)  # s3 not selected
    assert subs == {} and drop == set()
