"""Python↔JS bridge for the pywebview front-end.

Methods here are exposed to the web UI as `window.pywebview.api.<name>()` (each
returns a value / Promise on the JS side). The UI is pure presentation; all real
work happens here, reusing the existing engine:

  - history + settings:  src.history_db, settings.json
  - detect speakers:     src.audio_tracks + src.track_separation
  - transcribe:          app.worker.TranscriptionWorker (unchanged) - which already
                         auto-separates voices, de-duplicates bleed, and polishes.

Progress is pushed to the UI via `self._emit(event, payload)` → `window.appEvent`.
"""
from __future__ import annotations

import json
import queue
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import webview

from app.jobs import Job
from app.worker import SENTINEL, TranscriptionWorker
from src import catalog, history_db
from src.audio_tracks import safe_probe_audio_streams, track_speaker_names
from src.duration import probe_duration_seconds
from src.playback import TrackPlayer
from src import track_separation

SETTINGS_PATH = Path.home() / "Library" / "Application Support" / "LocalTranscriber" / "settings.json"

FAST_MODEL = "mlx-community/parakeet-tdt-0.6b-v3"
QUALITY_TO_MODEL = {"Best": catalog.DEFAULT_MODEL, "Fast": FAST_MODEL}
MODEL_TO_QUALITY = {catalog.DEFAULT_MODEL: "Best", FAST_MODEL: "Fast"}

_MEDIA_TYPES = ("Audio / video (*.mp3;*.wav;*.m4a;*.aac;*.flac;*.mp4;*.mov;*.m4v;*.mkv;*.webm)",)


def _load_settings() -> dict:
    try:
        return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _save_settings(settings: dict) -> None:
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(settings), encoding="utf-8")


def _fmt_duration(seconds: float | None) -> str:
    if not seconds:
        return ""
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _relative(iso: str | None) -> str:
    if not iso:
        return ""
    try:
        when = datetime.fromisoformat(iso)
    except ValueError:
        return ""
    # Lightweight relative phrasing; the exact clock lives in the row if needed.
    return when.strftime("%b %-d")


# Parse a written transcript .md into the two-voice turn list the UI renders.
_BLOCK_RE = re.compile(
    r"\*\*(?P<name>.+?)\*\* \[(?P<ts>\d{1,2}:\d{2}(?::\d{2})?)\](?P<intr> \(interrupting\))?\n(?P<text>.+?)(?=\n\n\*\*|\Z)",
    re.S,
)


def _parse_transcript(md_path: Path) -> dict:
    text = md_path.read_text(encoding="utf-8")
    body = text.split("---", 2)[2] if text.startswith("---") else text
    lanes: dict[str, str] = {}
    turns = []
    for m in _BLOCK_RE.finditer(body):
        name = m.group("name")
        if name not in lanes:
            lanes[name] = "a" if len(lanes) % 2 == 0 else "b"
        turns.append({
            "speaker": name, "lane": lanes[name], "ts": m.group("ts"),
            "text": m.group("text").strip(), "interrupting": bool(m.group("intr")),
        })
    # Single-speaker / plain transcripts have no **Name** blocks: fall back to paragraphs.
    if not turns and body.strip():
        turns = [{"speaker": "", "lane": "a", "ts": "", "text": p.strip(), "interrupting": False}
                 for p in body.strip().split("\n\n") if p.strip()]
    return {"speakers": list(lanes.keys()), "turns": turns}


class Bridge:
    def __init__(self) -> None:
        self._window = None
        self._settings = _load_settings()
        self.db = history_db.init_db()
        self.job_queue: "queue.Queue[Job | object]" = queue.Queue()
        self.event_queue: "queue.Queue[Any]" = queue.Queue()
        self.worker = TranscriptionWorker(self.job_queue, self.event_queue, self.db)
        self.worker.start()
        self._pending: dict[str, Job] = {}  # detected-but-not-yet-transcribed jobs
        self._player = TrackPlayer()
        threading.Thread(target=self._drain_events, daemon=True).start()

    def set_window(self, window) -> None:
        self._window = window

    def shutdown(self) -> None:
        self.job_queue.put(SENTINEL)

    # -- settings -------------------------------------------------------------
    def get_settings(self) -> dict:
        model = self._settings.get("model", catalog.DEFAULT_MODEL)
        return {
            "quality": MODEL_TO_QUALITY.get(model, "Best"),
            "tidy": bool(self._settings.get("polish", catalog.DEFAULT_POLISH_ENABLED)),
            "language": self._settings.get("language", catalog.DEFAULT_LANGUAGE),
        }

    def set_settings(self, patch: dict) -> dict:
        if "quality" in patch:
            self._settings["model"] = QUALITY_TO_MODEL.get(patch["quality"], catalog.DEFAULT_MODEL)
        if "tidy" in patch:
            self._settings["polish"] = bool(patch["tidy"])
        if "language" in patch:
            self._settings["language"] = patch["language"]
        _save_settings(self._settings)
        return self.get_settings()

    # -- history --------------------------------------------------------------
    def recent_calls(self) -> list[dict]:
        rows = history_db.fetch_history(self.db, limit=60)
        calls = []
        for r in rows:
            if r.status != "success" or not r.output_path or not Path(r.output_path).exists():
                continue
            calls.append({
                "id": f"h{r.id}", "name": Path(r.source_path).stem,
                "when": _relative(r.finished_at or r.started_at),
                "dur": _fmt_duration(r.audio_duration_seconds),
                "output": r.output_path, "source": r.source_path, "status": "done",
            })
        return calls

    def open_call(self, output_path: str) -> dict:
        p = Path(output_path)
        if not p.exists():
            return {"error": "That transcript file is missing."}
        data = _parse_transcript(p)
        data["name"] = p.stem
        data["output"] = output_path
        return data

    # -- add + detect ---------------------------------------------------------
    def pick_files(self) -> list[str]:
        """Open the native picker and return chosen paths (fast; no analysis)."""
        if self._window is None:
            return []
        try:
            picked = self._window.create_file_dialog(
                webview.OPEN_DIALOG, allow_multiple=True, file_types=_MEDIA_TYPES
            )
        except Exception:  # some macOS versions reject the type filter - fall back
            picked = self._window.create_file_dialog(webview.OPEN_DIALOG, allow_multiple=True)
        return [str(p) for p in (picked or [])]

    def detect(self, path: str) -> dict:
        """Probe + detect speakers for one file (slow; call after pick_files)."""
        try:
            return self._prepare(Path(path))
        except Exception as e:  # noqa: BLE001 - surface to the UI
            return {"error": str(e), "name": Path(path).stem}

    def add_files(self) -> list[dict]:  # kept for back-compat
        return [self.detect(p) for p in self.pick_files()]

    def _prepare(self, path: Path) -> dict:
        """Probe a file and detect who's on it, WITHOUT transcribing yet."""
        job = Job(
            source_path=path,
            model=self._settings.get("model", catalog.DEFAULT_MODEL),
            language=self._settings.get("language", catalog.DEFAULT_LANGUAGE),
            polish=bool(self._settings.get("polish", catalog.DEFAULT_POLISH_ENABLED)),
        )
        try:
            streams = safe_probe_audio_streams(path)
            job.tracks = streams
            job.audio_duration_seconds = probe_duration_seconds(path)
        except Exception:  # noqa: BLE001 - surface a usable error to the UI
            streams = []
        people = self._detect_people(job, streams)
        self._pending[job.id] = job
        return {
            "id": job.id, "name": path.stem, "source": str(path),
            "dur": _fmt_duration(job.audio_duration_seconds),
            "people": people, "status": "ready",
        }

    def _detect_people(self, job: Job, streams: list) -> list[dict]:
        """The resulting speakers after auto-separation (keep clean + isolate mix,
        drop duplicates) - what the worker will actually transcribe."""
        default_names = dict(zip((s.index for s in streams), track_speaker_names(streams)))
        pos_by_index = {s.index: i for i, s in enumerate(streams)}  # ffplay audio position
        if len(streams) <= 1:
            idx = streams[0].index if streams else 0
            return [{"index": idx, "audio": pos_by_index.get(idx, 0),
                     "name": default_names.get(idx, "Speaker"), "lane": "a", "ghost": None}]
        warnings = []
        try:
            start = max(0.0, (job.audio_duration_seconds or 0) * 0.35)
            seg = 90.0
            if job.audio_duration_seconds and start + seg > job.audio_duration_seconds:
                start = max(0.0, job.audio_duration_seconds - seg)
            signals = [
                track_separation.TrackSignal(
                    index=s.index, name="",
                    samples=track_separation.decode_track_pcm(job.source_path, s.index, start_s=start, duration_s=seg),
                )
                for s in streams
            ]
            warnings = track_separation.detect_shared_voice(signals)
        except Exception:  # noqa: BLE001 - detection is best-effort
            warnings = []
        selected = [s.index for s in streams]
        subs, drop = track_separation.resolve_shared_voice(selected, warnings)
        job.track_subtractions = subs or None
        kept = [s for s in streams if s.index not in drop] or streams
        # Pin the plan so the worker transcribes exactly these speakers (keep +
        # isolate), not every raw track. The worker respects a preset selection
        # and subtraction; the text backstop still runs underneath.
        job.selected_track_indices = {s.index for s in kept}
        people = []
        for n, s in enumerate(kept):
            refs = (subs or {}).get(s.index, [])
            ghost = default_names.get(refs[0]) if refs else None
            people.append({
                "index": s.index, "audio": pos_by_index.get(s.index, n),
                "name": default_names.get(s.index, f"Speaker {n+1}"),
                "lane": "a" if n % 2 == 0 else "b", "ghost": ghost,
            })
        return people

    # -- transcribe -----------------------------------------------------------
    def transcribe(self, job_id: str, options: dict | None = None) -> dict:
        job = self._pending.get(job_id)
        if job is None:
            return {"error": "That call is no longer available. Add it again."}
        options = options or {}
        names = options.get("names")  # {index: name}
        if names:
            job.track_names = {int(k): v for k, v in names.items()}
        if "quality" in options:
            job.model = QUALITY_TO_MODEL.get(options["quality"], job.model)
        if "tidy" in options:
            job.polish = bool(options["tidy"])
        job.status = "queued"
        self.job_queue.put(job)
        self._emit("job", {"id": job.id, "status": "queued", "progress": 0})
        return {"id": job.id}

    # -- worker events → UI ---------------------------------------------------
    def _drain_events(self) -> None:
        while True:
            event = self.event_queue.get()
            job = getattr(event, "job", None)
            if job is None:
                continue
            self._emit("job", self._job_payload(job))

    def _job_payload(self, job: Job) -> dict:
        progress = int(round((job.transcribe_fraction or 0) * 100))
        payload = {
            "id": job.id, "status": job.status, "progress": progress,
            "detail": job.status_detail or job.status,
        }
        payload["source"] = str(job.source_path)
        if job.status == "done" and job.output_path:
            payload["output"] = str(job.output_path)
            try:
                payload["transcript"] = _parse_transcript(Path(job.output_path))
            except Exception:  # noqa: BLE001
                pass
        if job.status == "error":
            payload["error"] = job.error
        return payload

    # -- Python → UI ----------------------------------------------------------
    def _emit(self, event: str, payload: Any = None) -> None:
        if self._window is None:
            return
        self._window.evaluate_js(f"window.appEvent && window.appEvent({event!r}, {json.dumps(payload)})")

    # -- playback + export ---------------------------------------------------
    def play_source(self, source: str, seconds: float = 0.0, audio_index: int = 0) -> dict:
        """Play one audio stream of a call from `seconds` (preview / read-along).
        Only one thing is audible at a time."""
        if not TrackPlayer.available():
            return {"error": "ffplay not found — install ffmpeg (brew install ffmpeg)."}
        try:
            self._player.play(Path(source), audio_index=int(audio_index), start_seconds=float(seconds))
            return {"ok": True}
        except Exception as e:  # noqa: BLE001
            return {"error": str(e)}

    def stop_audio(self) -> dict:
        self._player.stop()
        return {"ok": True}

    def playback_position(self) -> dict:
        """Where playback is now (seconds) and whether it's still going - polled
        by the UI to drive the progress bar, line highlight, and auto-scroll."""
        return {"seconds": self._player.position() or 0.0, "playing": self._player.is_playing()}

    def reveal(self, path: str) -> dict:
        """Reveal the written transcript in Finder."""
        import subprocess
        p = Path(path)
        if not p.exists():
            return {"error": "That file is missing."}
        subprocess.Popen(["open", "-R", str(p)])
        return {"ok": True}

    def ping(self) -> str:
        return "ok"
