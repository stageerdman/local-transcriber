"""Separate speakers from mixed / echoed multi-track recordings.

Some recordings route one person's audio into more than one track. The case
this module was built for (a screen-recorded call): the remote person's audio is
saved BOTH as its own clean track AND mixed into the local mic track (at unity
gain, zero delay), and a second near-identical copy of the mix exists too. So the
three tracks are really:

    C  = one clean voice (the remote person)
    M  = C + the local voice          (the mic, which also carries the call audio)
    M2 = M                            (a duplicate mix)

Transcribing all three would emit the remote person three times and never cleanly
attribute the local person. The fix is to recover one signal per *actual*
speaker:

    keep      C            -> the remote person, already clean
    isolate   M - C        -> the local person, echo removed
    drop      M2           -> duplicate, skip

This module classifies the tracks into that plan (`classify_tracks`) and renders
the resulting per-speaker audio (`render_speaker_wav`). It never modifies the
source file.

Everything works on time-aligned mono PCM. Tracks are streams inside one
container, so they already share t=0 and duration; we decode each to 16 kHz mono
(Whisper's native rate) for both analysis and rendering.

This is the one module in `src/` that decodes raw PCM into numpy - the rest of
the app treats audio purely through ffmpeg. PCM sample math (subtracting one
track from another) genuinely needs it; it's kept isolated here.
"""
from __future__ import annotations

import subprocess
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ANALYSIS_RATE = 16000  # Whisper's native rate; fine for both analysis and ASR.
_WIN_MS = 250
_HOP_MS = 125

# RNNoise speech-denoise model, applied to isolated tracks via ffmpeg's built-in
# `arnndn` filter (no extra Python dependency). RNNoise runs at 48 kHz, so
# `denoise_pcm` resamples around the filter.
DEFAULT_RNNOISE_MODEL = Path(__file__).resolve().parent.parent / "assets" / "rnnoise" / "bd.rnnn"

# --- classification thresholds (validated on real + synthetic recordings) -----
# A track is "active" in a window when its RMS there exceeds this fraction of its
# own overall RMS.
_ACTIVE_RATIO = 0.10
# "i active => j active almost always": i's activity is contained in j's.
_CONTAIN_HIGH = 0.90
# A contained reference must be active strictly less of the time than the mix it
# sits inside - the mix also carries the other speaker's solo turns. (This is the
# direction signal: the smaller-activity track is the component, the larger is
# the mix.)
_FRAC_ASYM = 0.97
# Two tracks are duplicates when each contains the other AND they are active a
# similar fraction of the time (ratio of active fractions above this).
_DUP_ACT_RATIO = 0.90
# Subtracting a contained reference out of a mix must remove real energy...
_MIN_DROP_DB = 3.0
# ...and leave a residual that no longer correlates with that reference.
_MAX_RESIDUAL_CORR = 0.35


@dataclass
class TrackSignal:
    """One track's decoded mono audio, tagged with its stream index and name."""

    index: int
    name: str
    samples: np.ndarray  # mono float, ANALYSIS_RATE
    channel: int | None = None


@dataclass(frozen=True)
class TrackRole:
    """What to do with one track when producing per-speaker transcripts.

    kind:
      "keep"      - transcribe this track as-is (a clean speaker).
      "isolate"   - transcribe this track minus `reference_indices` (recovers the
                    speaker unique to this mix).
      "duplicate" - skip; it duplicates `duplicate_of`.
    """

    index: int
    name: str
    kind: str
    reference_indices: tuple[int, ...] = ()
    duplicate_of: int | None = None


@dataclass
class SeparationPlan:
    roles: list[TrackRole] = field(default_factory=list)

    def role_for(self, index: int) -> TrackRole | None:
        return next((r for r in self.roles if r.index == index), None)

    def speaker_roles(self) -> list[TrackRole]:
        """Roles that produce a transcript (everything except duplicates)."""
        return [r for r in self.roles if r.kind != "duplicate"]

    def is_trivial(self) -> bool:
        """True when the plan changes nothing (every track kept as-is)."""
        return all(r.kind == "keep" for r in self.roles)


# --- decoding -----------------------------------------------------------------
def decode_track_pcm(
    source: Path,
    stream_index: int,
    channel_index: int | None = None,
    rate: int = ANALYSIS_RATE,
    start_s: float | None = None,
    duration_s: float | None = None,
) -> np.ndarray:
    """Decode one audio stream (optionally one channel) to mono float32 PCM.

    `start_s`/`duration_s` decode only a window (used by the shared-voice safety
    net, which needs a representative segment, not a whole 90-minute track)."""
    if channel_index is not None:
        af = f"pan=mono|c0=c{channel_index}"
    else:
        af = "aformat=channel_layouts=mono"
    command = ["ffmpeg", "-v", "error"]
    if start_s is not None:
        command += ["-ss", str(start_s)]
    if duration_s is not None:
        command += ["-t", str(duration_s)]
    command += [
        "-i", str(source),
        "-map", f"0:{stream_index}",
        "-af", af,
        "-ar", str(rate),
        "-f", "f32le", "-",
    ]
    raw = subprocess.run(command, check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32).astype(np.float64)


# --- low-level analysis -------------------------------------------------------
def _active_windows(x: np.ndarray, rate: int = ANALYSIS_RATE) -> np.ndarray:
    """Boolean per-window: is this track energetically active here?"""
    win = int(_WIN_MS * rate / 1000)
    hop = int(_HOP_MS * rate / 1000)
    ref = np.sqrt(np.mean(x * x)) + 1e-12
    if len(x) < win:
        return np.zeros(0, dtype=bool)
    starts = range(0, len(x) - win + 1, hop)
    return np.array(
        [np.sqrt(np.mean(x[s:s + win] ** 2)) > _ACTIVE_RATIO * ref for s in starts]
    )


def _containment(active_a: np.ndarray, active_b: np.ndarray) -> float:
    """P(b active | a active): how often b is also active when a is."""
    n = min(len(active_a), len(active_b))
    a, b = active_a[:n], active_b[:n]
    if a.sum() == 0:
        return 0.0
    return float((a & b).sum() / a.sum())


def _scalar_gain(mix: np.ndarray, ref: np.ndarray) -> float:
    """Least-squares scale g minimizing ||mix - g*ref||."""
    denom = float(np.dot(ref, ref)) + 1e-12
    return float(np.dot(mix, ref) / denom)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    a, b = a[:n] - np.mean(a[:n]), b[:n] - np.mean(b[:n])
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _energy_db(x: np.ndarray) -> float:
    return 10.0 * np.log10(np.mean(x * x) + 1e-12)


def _subtraction_removes(mix: np.ndarray, ref: np.ndarray) -> tuple[float, float]:
    """Subtract ref out of mix; return (energy drop in dB, residual-vs-ref corr).

    A large drop with a near-zero residual correlation means ref was genuinely
    present inside mix and has been cleanly removed.
    """
    n = min(len(mix), len(ref))
    mix, ref = mix[:n], ref[:n]
    g = _scalar_gain(mix, ref)
    residual = mix - g * ref
    drop = _energy_db(mix) - _energy_db(residual)
    return drop, abs(_corr(residual, ref))


# --- classification -----------------------------------------------------------
def classify_tracks(signals: list[TrackSignal]) -> SeparationPlan:
    """Work out, per track, whether to keep / isolate / drop it.

    Pure function of the decoded samples - no I/O - so it's easy to unit-test
    with synthetic voices.
    """
    n = len(signals)
    if n <= 1:
        return SeparationPlan([
            TrackRole(index=s.index, name=s.name, kind="keep") for s in signals
        ])

    active = [_active_windows(s.samples) for s in signals]
    frac = [float(a.mean()) if len(a) else 0.0 for a in active]

    # containment[i][j] = P(j active | i active)
    contain = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                contain[i][j] = _containment(active[i], active[j])

    # Pass 1: duplicates. i and j are duplicates when each is active almost
    # exactly when the other is, and they're active a similar fraction of the
    # time. Keep the lower-indexed one; mark the other a duplicate of it.
    duplicate_of: dict[int, int] = {}  # position -> position of the kept twin
    for i in range(n):
        for j in range(i + 1, n):
            if j in duplicate_of or i in duplicate_of:
                continue
            symmetric = contain[i][j] >= _CONTAIN_HIGH and contain[j][i] >= _CONTAIN_HIGH
            hi, lo = max(frac[i], frac[j]), min(frac[i], frac[j])
            similar_activity = lo >= _DUP_ACT_RATIO * hi if hi > 0 else False
            if symmetric and similar_activity and _corr(signals[i].samples, signals[j].samples) >= 0.9:
                duplicate_of[j] = i

    # Pass 2: containment. Among the survivors, position c is a clean component
    # of mix m when c's activity is contained in m's (but not vice versa), c is
    # active less of the time, and subtracting c out of m removes real energy
    # and decorrelates.
    survivors = [k for k in range(n) if k not in duplicate_of]
    references_for: dict[int, list[int]] = {k: [] for k in survivors}  # mix -> [refs]
    is_reference: set[int] = set()
    for m in survivors:
        for c in survivors:
            if m == c:
                continue
            asymmetric = (
                contain[c][m] >= _CONTAIN_HIGH
                and frac[c] < _FRAC_ASYM * frac[m]
            )
            if not asymmetric:
                continue
            drop, resid_corr = _subtraction_removes(signals[m].samples, signals[c].samples)
            if drop >= _MIN_DROP_DB and resid_corr <= _MAX_RESIDUAL_CORR:
                references_for[m].append(c)
                is_reference.add(c)

    # Build roles.
    roles: list[TrackRole] = []
    for pos, sig in enumerate(signals):
        if pos in duplicate_of:
            roles.append(TrackRole(
                index=sig.index, name=sig.name, kind="duplicate",
                duplicate_of=signals[duplicate_of[pos]].index,
            ))
        elif references_for.get(pos):
            refs = tuple(signals[c].index for c in references_for[pos])
            roles.append(TrackRole(
                index=sig.index, name=sig.name, kind="isolate", reference_indices=refs,
            ))
        else:
            roles.append(TrackRole(index=sig.index, name=sig.name, kind="keep"))
    return SeparationPlan(roles)


# --- shared-voice safety net --------------------------------------------------
@dataclass(frozen=True)
class SharedVoiceWarning:
    """Two selected tracks carry the same voice, so transcribing both would
    double that speaker in the output.

    kind:
      "duplicate" - the two tracks are near-identical (same content both ways);
                    transcribe one, exclude the other.
      "contains"  - `mix_index` = `component_index`'s voice + another speaker
                    (a mix); transcribe `mix_index` with `component_index`
                    subtracted out, and keep `component_index` as its own
                    speaker.
    """

    mix_index: int
    component_index: int
    kind: str
    drop_db: float


# Two tracks "share a voice" when, in the windows where both are active, their
# waveforms are substantially correlated (the same person's signal is in both).
# Cleanly-separated speakers sit near 0 here (different voices); shared-voice
# pairs sit well above it even across different mix ratios.
_SHARED_CORR = 0.30
# A track is a (near-)unity-gain component of another - i.e. the larger one is
# "this voice + someone else" - when the median subtraction gain lands here with
# a decorrelated residual. Tighter than a generic containment band so two
# different mixes of the same people don't masquerade as a clean component.
_UNITY_GAIN_LO = 0.85
_UNITY_GAIN_HI = 1.20
_DETECT_WIN_MS = 2000
_DETECT_HOP_MS = 1000
# Direction is set by activity CONTAINMENT: a clean component's active windows
# are a subset of the mix's (the mix is also active in the other speaker's solo
# turns). b is a component of a when P(a active | b active) is high but
# P(b active | a active) is not - i.e. b sits inside a. A single threshold both
# ways keeps the asymmetry test simple and robust to continuous talk.
_SUBSET_THR = 0.93


def _pairwise_stats(a: np.ndarray, b: np.ndarray, rate: int = ANALYSIS_RATE) -> dict:
    """Window two tracks and aggregate, over windows where BOTH are active:
    median waveform correlation, and the median subtraction gains each way.

    Windowing + medians make this robust to who is talking when - the single
    stable signal across a long call, unlike a single-window snapshot (which
    flips verdict depending on the moment sampled)."""
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    win = int(_DETECT_WIN_MS * rate / 1000)
    hop = int(_DETECT_HOP_MS * rate / 1000)
    ra = np.sqrt(np.mean(a * a)) + 1e-12
    rb = np.sqrt(np.mean(b * b)) + 1e-12
    corrs, gains_ab, gains_ba = [], [], []
    for s in range(0, max(1, n - win + 1), hop):
        wa, wb = a[s:s + win], b[s:s + win]
        if np.sqrt(np.mean(wa * wa)) < 0.1 * ra or np.sqrt(np.mean(wb * wb)) < 0.1 * rb:
            continue  # need both active to judge shared content
        corrs.append(abs(_corr(wa, wb)))
        gains_ab.append(_scalar_gain(wa, wb))  # a ~= g*b  -> b's gain inside a
        gains_ba.append(_scalar_gain(wb, wa))  # b ~= g*a  -> a's gain inside b
    # Activity CONTAINMENT sets DIRECTION: a mix is active whenever EITHER
    # speaker talks, so a clean component's active windows sit inside the mix's.
    # (Gain alone can't tell them apart - in the component's own turns the mix
    # equals it, so both subtraction gains look like unity.)
    act_a, act_b = _active_windows(a, rate), _active_windows(b, rate)
    p_a_given_b = _containment(act_b, act_a)  # P(a active | b active)
    p_b_given_a = _containment(act_a, act_b)  # P(b active | a active)
    if not corrs:
        return {"corr": 0.0, "g_b_in_a": 0.0, "g_a_in_b": 0.0, "resid": 1.0,
                "p_a_given_b": p_a_given_b, "p_b_given_a": p_b_given_a}
    _, resid = _subtraction_removes(a, b)
    return {
        "corr": float(np.median(corrs)),
        "g_b_in_a": float(np.median(gains_ab)),
        "g_a_in_b": float(np.median(gains_ba)),
        "resid": resid,
        "p_a_given_b": p_a_given_b,
        "p_b_given_a": p_b_given_a,
    }


def detect_shared_voice(signals: list[TrackSignal]) -> list[SharedVoiceWarning]:
    """Find pairs among the given tracks that carry the same voice.

    Pure PCM analysis (no I/O). For each pair, if their active-window waveforms
    are substantially correlated they share a voice, so transcribing both would
    double that speaker - the exact failure behind the doubled transcript. The
    direction is set by the subtraction gain: if one track is a (near-)unity
    component of the other, the larger one is a mix of "that voice + another",
    so the fix is to isolate the mix minus the component and keep the component
    ("contains"); otherwise the two are redundant takes of the same voices and
    one should be dropped ("duplicate").

    Intended to run on the tracks the user has SELECTED to transcribe, over a
    representative segment with both speakers active; it never modifies audio or
    the plan. The caller filters these against the current selection and any
    configured subtractions and surfaces the ones still unhandled.
    """
    n = len(signals)
    warnings: list[SharedVoiceWarning] = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = signals[i].samples, signals[j].samples
            st = _pairwise_stats(a, b)
            if st["corr"] < _SHARED_CORR or st["resid"] > _MAX_RESIDUAL_CORR:
                continue
            drop = float(10.0 * np.log10(st["corr"] / max(1e-3, 1 - st["corr"]) + 1e-9))
            # b is a clean component of a: b's activity sits inside a's
            # (P(a|b) high, P(b|a) lower) AND b subtracts out of a at ~unity, so
            # isolating a - b will actually cancel it.
            b_in_a = (
                st["p_a_given_b"] >= _SUBSET_THR
                and st["p_b_given_a"] < _SUBSET_THR
                and _UNITY_GAIN_LO <= st["g_b_in_a"] <= _UNITY_GAIN_HI
            )
            a_in_b = (
                st["p_b_given_a"] >= _SUBSET_THR
                and st["p_a_given_b"] < _SUBSET_THR
                and _UNITY_GAIN_LO <= st["g_a_in_b"] <= _UNITY_GAIN_HI
            )
            if b_in_a:
                warnings.append(SharedVoiceWarning(
                    mix_index=signals[i].index, component_index=signals[j].index,
                    kind="contains", drop_db=drop,
                ))
            elif a_in_b:
                warnings.append(SharedVoiceWarning(
                    mix_index=signals[j].index, component_index=signals[i].index,
                    kind="contains", drop_db=drop,
                ))
            else:
                # Similar activity (or no clean unity component) -> two takes of
                # the same voices; drop one.
                warnings.append(SharedVoiceWarning(
                    mix_index=signals[j].index, component_index=signals[i].index,
                    kind="duplicate", drop_db=drop,
                ))
    return warnings


def resolve_shared_voice(
    selected: list[int], warnings: list[SharedVoiceWarning]
) -> tuple[dict[int, list[int]], set[int]]:
    """Turn shared-voice warnings into a suggested fix over the selected tracks.

    Returns `(subtractions, drop)`:
      - `subtractions[mix]` = tracks to remove from `mix` before transcribing it
        (so a mix that carries a clean component becomes that mix's *other*
        speaker), with the component kept as its own speaker.
      - `drop` = tracks to stop transcribing (redundant takes of a voice already
        covered).

    Correct for the common "one clean source + a mix (+ a duplicate mix)" case;
    best-effort on complex multi-track files, where the user adjusts by hand.
    Never drops so much that nothing is left to transcribe.
    """
    sel = [i for i in selected]
    components = {w.component_index for w in warnings if w.kind == "contains"}
    subtractions: dict[int, list[int]] = {}
    for w in warnings:
        if w.kind == "contains" and w.mix_index in sel and w.component_index in sel:
            subtractions.setdefault(w.mix_index, [])
            if w.component_index not in subtractions[w.mix_index]:
                subtractions[w.mix_index].append(w.component_index)

    drop: set[int] = set()
    for w in warnings:
        if w.kind != "duplicate":
            continue
        a, b = w.component_index, w.mix_index
        if a not in sel or b not in sel:
            continue
        # Prefer dropping a track that is neither a kept clean component nor a
        # mix we're going to isolate; fall back to the later-numbered one.
        expendable = [t for t in (b, a) if t not in components and t not in subtractions]
        drop.add(expendable[0] if expendable else max(a, b))
    # A mix we isolate (or a kept component) must never be dropped.
    drop -= set(subtractions)
    drop -= components
    # Safety: keep at least one track to transcribe.
    if len(set(sel) - drop) == 0:
        drop.clear()
    return subtractions, drop


# --- isolation & rendering ----------------------------------------------------
def isolate(mix: np.ndarray, references: list[np.ndarray]) -> np.ndarray:
    """Remove the reference signals from the mix (least-squares), leaving the
    speaker unique to the mix.

    Solves min ||mix - R c|| over the reference columns jointly, so overlapping
    references don't double-subtract.
    """
    if not references:
        return mix.astype(np.float64)
    n = min([len(mix)] + [len(r) for r in references])
    mix = mix[:n].astype(np.float64)
    R = np.column_stack([r[:n].astype(np.float64) for r in references])
    coeffs, *_ = np.linalg.lstsq(R, mix, rcond=None)
    return mix - R @ coeffs


def noise_gate(x: np.ndarray, rate: int = ANALYSIS_RATE, ramp_ms: float = 20.0) -> np.ndarray:
    """Silence windows where the isolated speaker isn't actually talking.

    After subtraction the residual is near-silent during the other person's
    turns; Whisper hallucinates filler ("a a a a") on such near-silence, so we
    zero those stretches. Gains are ramped to avoid clicks.

    Thresholds against the signal's OWN RMS, so it only helps when the residual
    has a clear loud/quiet structure. For the subtraction path prefer
    `gate_to_reference`, which compares against the pre-subtraction mix.
    """
    active = _active_windows(x, rate)
    if len(active) == 0:
        return x
    hop = int(_HOP_MS * rate / 1000)
    gain = np.zeros(len(x))
    for i, on in enumerate(active):
        if on:
            gain[i * hop:(i + 1) * hop + hop] = 1.0  # +1 hop so windows overlap-fill
    gain = gain[:len(x)]
    # smooth the on/off edges
    ramp = max(1, int(ramp_ms * rate / 1000))
    kernel = np.ones(ramp) / ramp
    gain = np.convolve(gain, kernel, mode="same")
    gain = np.clip(gain, 0.0, 1.0)
    return x * gain


# Gate tuning (validated on real call audio - see mic-bleed update wiki).
_GATE_WIN_MS = 300
_GATE_HOP_MS = 150
# Keep a window only if the residual still holds this fraction of the mix's
# energy there (subtracting the reference removed most of it => the other
# speaker dominated => this speaker was silent).
_GATE_REL = 0.5
# ...and clears an absolute floor tied to the mix's overall level, so faint
# uncancelled residue can't leak through where the mix itself is loud.
_GATE_ABS_FRAC = 0.1
# Extend each kept region forward so word tails/soft onsets survive the gate.
_GATE_HANGOVER_MS = 250


def gate_to_reference(
    residual: np.ndarray,
    mix: np.ndarray,
    rate: int = ANALYSIS_RATE,
    ramp_ms: float = 30.0,
) -> np.ndarray:
    """Gate an isolated (mix - reference) residual against the ORIGINAL mix.

    The fix for the own-RMS blind spot in `noise_gate`: when the isolated
    speaker is silent, subtraction drives the residual far below the mix, but
    its own level is uniformly low so an own-RMS gate keeps the whole thing and
    Whisper hallucinates on it. Here a window survives only if the residual kept
    a real fraction of the mix's energy there (`_GATE_REL`) and clears an
    absolute floor (`_GATE_ABS_FRAC` of the mix RMS). Produces a hard 0/1 mask
    (with a forward hangover and short ramps) so VAD/ASR see clean silence in
    the gaps rather than low-level residue.
    """
    n = min(len(residual), len(mix))
    residual, mix = residual[:n].astype(np.float64), mix[:n].astype(np.float64)
    win = int(_GATE_WIN_MS * rate / 1000)
    hop = int(_GATE_HOP_MS * rate / 1000)
    if n < win:
        return residual
    floor = _GATE_ABS_FRAC * (np.sqrt(np.mean(mix * mix)) + 1e-12)
    starts = list(range(0, n - win + 1, hop))
    keep = np.zeros(len(starts), dtype=bool)
    for k, s in enumerate(starts):
        rr = np.sqrt(np.mean(residual[s:s + win] ** 2))
        mr = np.sqrt(np.mean(mix[s:s + win] ** 2)) + 1e-12
        keep[k] = (rr > _GATE_REL * mr) and (rr > floor)
    # Forward hangover so trailing syllables aren't clipped.
    hang = int(np.ceil(_GATE_HANGOVER_MS / _GATE_HOP_MS))
    if hang and keep.any():
        extended = keep.copy()
        for k in np.flatnonzero(keep):
            extended[k:k + hang + 1] = True
        keep = extended
    gain = np.zeros(n)
    for k, s in enumerate(starts):
        if keep[k]:
            gain[s:s + hop] = 1.0
    ramp = max(1, int(ramp_ms * rate / 1000))
    gain = np.clip(np.convolve(gain, np.ones(ramp) / ramp, mode="same"), 0.0, 1.0)
    return residual * gain


def denoise_pcm(
    samples: np.ndarray,
    rate: int = ANALYSIS_RATE,
    model_path: Path | None = DEFAULT_RNNOISE_MODEL,
) -> np.ndarray:
    """Speech-denoise a mono signal with RNNoise via ffmpeg's `arnndn` filter.

    Removes the broadband hiss / faint residue left after subtracting one track
    out of another, so the isolated voice is clean for the transcriber. RNNoise
    operates at 48 kHz, so we resample up for the filter and back down. Returns
    the input unchanged if the model file is missing (feature simply off).
    """
    if model_path is None or not Path(model_path).exists():
        return samples
    pcm = np.clip(samples, -1.0, 1.0).astype("<f4").tobytes()
    chain = f"aresample=48000,arnndn=m={model_path},aresample={rate}"
    command = [
        "ffmpeg", "-v", "error",
        "-f", "f32le", "-ar", str(rate), "-ac", "1", "-i", "-",
        "-af", chain,
        "-f", "f32le", "-ac", "1", "-",
    ]
    out = subprocess.run(command, input=pcm, capture_output=True, check=True).stdout
    denoised = np.frombuffer(out, dtype="<f4").astype(np.float64)
    # ffmpeg's resampler can shift length by a few samples; align for callers.
    if len(denoised) >= len(samples):
        return denoised[:len(samples)]
    return np.concatenate([denoised, np.zeros(len(samples) - len(denoised))])


def _peak_normalize(x: np.ndarray, peak: float = 0.9) -> np.ndarray:
    m = float(np.max(np.abs(x))) if len(x) else 0.0
    if m <= 0:
        return x
    return x * (peak / m)


def write_wav(path: Path, samples: np.ndarray, rate: int = ANALYSIS_RATE) -> None:
    """Write mono PCM to a 16-bit wav the transcription engine can read."""
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(samples, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm16.tobytes())


def render_transcription_wav(
    mix: np.ndarray,
    references: list[np.ndarray],
    output_path: Path,
    *,
    denoise: bool = True,
    rate: int = ANALYSIS_RATE,
    denoise_model: Path | None = DEFAULT_RNNOISE_MODEL,
) -> None:
    """Render one track's audio for transcription, applying the user's choices.

    `references` are other tracks whose voices should be removed from `mix`
    (empty = none). `denoise` runs RNNoise. When references are subtracted the
    residual has silent gaps that would make ASR hallucinate, so those are
    gated; a plain (un-subtracted) track is left ungated. Peak-normalized so a
    quiet result still reaches the transcriber at a healthy level.
    """
    if references:
        out = isolate(mix, references)
        if denoise:
            out = denoise_pcm(out, rate, denoise_model)
        # Gate against the pre-subtraction mix so the gaps where the other
        # speaker dominated become true silence (no hallucinated filler).
        out = gate_to_reference(out, mix, rate)
    else:
        out = denoise_pcm(mix, rate, denoise_model) if denoise else mix
    write_wav(output_path, _peak_normalize(out), rate)


def render_speaker_wav(
    role: TrackRole,
    signals_by_index: dict[int, np.ndarray],
    output_path: Path,
    rate: int = ANALYSIS_RATE,
    denoise_model: Path | None = DEFAULT_RNNOISE_MODEL,
) -> None:
    """Produce the audio for one speaker role and write it to `output_path`.

    "keep" -> the track as-is; "isolate" -> the track with its references
    subtracted out, RNNoise-denoised, and the silent gaps gated. Both are
    peak-normalized so the (often quiet) isolated voice reaches the transcriber
    at a healthy level.
    """
    mix = signals_by_index[role.index]
    if role.kind == "isolate" and role.reference_indices:
        refs = [signals_by_index[i] for i in role.reference_indices]
        # subtract the other voice(s) -> denoise the residue -> gate the gaps
        # (against the original mix, so silent stretches become true silence)
        out = gate_to_reference(denoise_pcm(isolate(mix, refs), rate, denoise_model), mix, rate)
    else:
        out = mix
    write_wav(output_path, _peak_normalize(out), rate)
