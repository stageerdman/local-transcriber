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
) -> np.ndarray:
    """Decode one audio stream (optionally one channel) to mono float32 PCM."""
    if channel_index is not None:
        af = f"pan=mono|c0=c{channel_index}"
    else:
        af = "aformat=channel_layouts=mono"
    command = [
        "ffmpeg", "-v", "error",
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


def render_speaker_wav(
    role: TrackRole,
    signals_by_index: dict[int, np.ndarray],
    output_path: Path,
) -> None:
    """Produce the audio for one speaker role and write it to `output_path`.

    "keep" -> the track as-is; "isolate" -> the track with its references
    subtracted out and the silent gaps gated. Both are peak-normalized so the
    (often quiet) isolated voice reaches the transcriber at a healthy level.
    """
    mix = signals_by_index[role.index]
    if role.kind == "isolate" and role.reference_indices:
        refs = [signals_by_index[i] for i in role.reference_indices]
        out = noise_gate(isolate(mix, refs))
    else:
        out = mix
    write_wav(output_path, _peak_normalize(out))
