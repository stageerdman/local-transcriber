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
import os
import queue
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import webview

from app.jobs import Job
from app.worker import SENTINEL, TranscriptionWorker
from src import catalog, history_db, job_store
from src.audio_tracks import safe_probe_audio_streams, track_speaker_names
from src.duration import probe_duration_seconds
from src.playback import TrackPlayer
from src import track_separation

SETTINGS_PATH = Path.home() / "Library" / "Application Support" / "LocalTranscriber" / "settings.json"

FAST_MODEL = "mlx-community/parakeet-tdt-0.6b-v3"
QUALITY_TO_MODEL = {"Best": catalog.DEFAULT_MODEL, "Fast": FAST_MODEL, "Cloud": catalog.GROQ_ID}
MODEL_TO_QUALITY = {catalog.DEFAULT_MODEL: "Best", FAST_MODEL: "Fast", catalog.GROQ_ID: "Cloud"}
# "Cloud" is opt-in only - never the restored/default tier. This app is
# offline-first; a user who never touched the quality picker (fresh install,
# or settings reset) should never be silently switched to the one tier that
# sends audio off the device.
_RESTORABLE_QUALITIES = {"Best", "Fast"}

_MEDIA_TYPES = ("Audio / video (*.mp3;*.wav;*.m4a;*.aac;*.flac;*.mp4;*.mov;*.m4v;*.mkv;*.webm)",)
_MEDIA_EXT = (".mp3", ".wav", ".m4a", ".aac", ".flac", ".mp4", ".mov", ".m4v", ".mkv", ".webm")


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
        self._pending: dict[int, Job] = {}  # jobs by id: detected, in-flight, or paused
        self._resumed: list[Job] = []  # jobs re-queued from disk on this launch
        self._player = TrackPlayer()
        threading.Thread(target=self._drain_events, daemon=True).start()
        import atexit
        atexit.register(self.shutdown)  # never leave ffplay / the engine running
        # Pick up anything interrupted by a previous quit/crash/reboot and let it
        # continue from its checkpoint. The UI learns about these via
        # `resuming_jobs()` once it's ready, then via live job events.
        self._resume_interrupted_jobs()

    def set_window(self, window) -> None:
        self._window = window

    def shutdown(self) -> None:
        try:
            self._player.stop()  # kill any ffplay preview so audio never outlives the app
        except Exception:
            pass
        self.job_queue.put(SENTINEL)

    def user_name(self) -> str:
        """The account's display name for the home greeting. macOS stores a full
        name in the gecos field (e.g. "Stage Erdman" → "Stage"); fall back to the
        login name, or "" so the UI greets without a name rather than a wrong one."""
        try:
            import pwd
            gecos = pwd.getpwuid(os.getuid()).pw_gecos.split(",")[0].strip()
            if gecos:
                return gecos.split()[0]
        except Exception:  # noqa: BLE001 - best-effort; fall through to login name
            pass
        try:
            return (os.getlogin() or "").strip()
        except Exception:  # noqa: BLE001
            return ""

    # -- settings -------------------------------------------------------------
    def get_settings(self) -> dict:
        model = self._settings.get("model", catalog.DEFAULT_MODEL)
        quality = MODEL_TO_QUALITY.get(model, "Best")
        if quality not in _RESTORABLE_QUALITIES:
            quality = "Best"  # never auto-restore into Cloud - opt-in every launch
        return {
            "quality": quality,
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

    def handle_drop(self, event) -> None:
        """DOM 'drop' handler (attached to <body> in webview_main). pywebview puts
        the real filesystem path on each file as `pywebviewFullPath`; forward the
        media ones to the UI, which runs the same detect flow as the picker."""
        try:
            files = ((event or {}).get("dataTransfer") or {}).get("files") or []
            paths = [f.get("pywebviewFullPath") for f in files
                     if f.get("pywebviewFullPath", "").lower().endswith(_MEDIA_EXT)]
            if paths:
                self._emit("dropped", {"paths": paths})
        except Exception:  # noqa: BLE001
            pass

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
        people, set_aside = self._detect_people(job, streams)
        self._pending[job.id] = job
        return {
            "id": job.id, "name": path.stem, "source": str(path),
            "dur": _fmt_duration(job.audio_duration_seconds),
            "dur_secs": float(job.audio_duration_seconds or 0.0),
            "people": people, "setaside": set_aside, "status": "ready",
        }

    def _first_speech(self, source: Path, stream_index: int) -> float:
        """Seconds to the first SUSTAINED speech (so previews skip quiet intros and
        don't land on a one-off transient like a bump on the desk). Looks for the
        first ~4s stretch that's mostly at a real talking level. 0.0 if unsure."""
        try:
            import numpy as np
            import statistics
            rate = track_separation.ANALYSIS_RATE
            pcm = track_separation.decode_track_pcm(source, stream_index, start_s=0.0, duration_s=360.0)
            win = int(0.5 * rate)
            n = len(pcm) // win
            if n < 1:
                return 0.0
            rms = [float(np.sqrt(np.mean(pcm[i * win:(i + 1) * win] ** 2))) for i in range(n)]
            peak = max(rms) or 1.0
            voiced = [r for r in rms if r > 0.04 * peak]  # ignore near-silence
            level = statistics.median(voiced) if voiced else peak  # typical talking level
            thr = 0.4 * level
            run, need = 8, 6  # 6 of 8 half-second windows active => ~3-4s of real talk
            for i in range(0, n - run):
                if sum(1 for r in rms[i:i + run] if r > thr) >= need:
                    return max(0.0, i * 0.5 - 0.3)
        except Exception:  # noqa: BLE001
            pass
        return 0.0

    def _name_for(self, stream, seq: int) -> str:
        """Embedded track title if present, else a sequential 'Speaker N' (so the
        displayed roster never has confusing gaps like 'Speaker 1, Speaker 3')."""
        return stream.title if getattr(stream, "title", None) else f"Speaker {seq}"

    def _detect_people(self, job: Job, streams: list) -> tuple[list[dict], list[dict]]:
        """(people, set_aside). `people` are the resulting speakers after auto-
        separation (keep clean + isolate mix); `set_aside` are tracks dropped as
        duplicates, offered so the user can add one back if detection was wrong."""
        pos_by_index = {s.index: i for i, s in enumerate(streams)}
        if len(streams) <= 1:
            s = streams[0] if streams else None
            idx = s.index if s else 0
            person = {"index": idx, "audio": pos_by_index.get(idx, 0),
                      "name": self._name_for(s, 1) if s else "Speaker 1",
                      "lane": "a", "ghost": None, "start": self._first_speech(job.source_path, idx) if s else 0.0}
            return ([person], [])
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
        subs, drop = track_separation.resolve_shared_voice([s.index for s in streams], warnings)
        job.track_subtractions = subs or None
        kept = [s for s in streams if s.index not in drop] or streams
        dropped = [s for s in streams if s.index in drop and s not in kept]
        job.selected_track_indices = {s.index for s in kept}
        kept_name = {s.index: self._name_for(s, n + 1) for n, s in enumerate(kept)}
        people = []
        for n, s in enumerate(kept):
            refs = (subs or {}).get(s.index, [])
            ghost = kept_name.get(refs[0]) if refs else None  # the voice removed from this mix
            people.append({
                "index": s.index, "audio": pos_by_index.get(s.index, n), "name": kept_name[s.index],
                "lane": "a" if n % 2 == 0 else "b", "ghost": ghost,
                "removes": list((subs or {}).get(s.index, [])),  # other voices removed from this one
                "start": self._first_speech(job.source_path, s.index),
            })
        set_aside = [{
            "index": s.index, "audio": pos_by_index[s.index],
            "name": s.title if getattr(s, "title", None) else "Another recording",
            "start": self._first_speech(job.source_path, s.index),
        } for s in dropped]
        return (people, set_aside)

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
        speakers = options.get("speakers")  # final set of track indices to transcribe
        if speakers:
            job.selected_track_indices = {int(x) for x in speakers}
        subs = options.get("subtractions")  # {person index: [voice indices to remove]}
        if subs is not None:
            cleaned = {int(k): [int(x) for x in v] for k, v in subs.items() if v}
            job.track_subtractions = cleaned or None
        job.status = "queued"
        self._pending[job.id] = job
        self.job_queue.put(job)
        self._emit("job", {"id": job.id, "status": "queued", "progress": 0})
        return {"id": job.id}

    def stop_job(self, job_id: int) -> dict:
        """Stop a running/queued job. Its checkpoint is kept, so it can be
        resumed (via `resume_job`, or automatically on the next launch)."""
        job = self._pending.get(job_id)
        if job is None:
            return {"error": "That job is no longer active."}
        job.stop_event.set()
        return {"ok": True}

    def resume_job(self, job_id: int) -> dict:
        """Re-queue a job that was stopped this session, continuing from its
        checkpoint. (Jobs interrupted by a quit/crash resume on their own at
        launch; this is the in-session Resume button.)"""
        job = self._pending.get(job_id)
        if job is None:
            return {"error": "That job is no longer available."}
        job.stop_event = threading.Event()  # fresh, un-set
        job.status = "queued"
        self.job_queue.put(job)
        self._emit("job", {"id": job.id, "status": "queued",
                           "progress": int(round((job.transcribe_fraction or 0) * 100))})
        return {"id": job.id}

    def discard_job(self, job_id: int) -> dict:
        """Throw away a paused job and its saved partial work for good."""
        job = self._pending.pop(job_id, None)
        if job is not None:
            job.stop_event.set()
            job_store.discard(job.key)
        return {"ok": True}

    # -- resume interrupted jobs on launch ------------------------------------
    def _resume_interrupted_jobs(self) -> None:
        try:
            resumable = job_store.list_resumable()
        except Exception:  # noqa: BLE001 - never let a bad checkpoint block startup
            resumable = []
        for cp in resumable:
            source = (cp.config or {}).get("source_path")
            if not source or not Path(source).exists():
                job_store.discard(cp.key)  # source moved/deleted -> unresumable
                continue
            try:
                job = self._job_from_checkpoint(cp)
            except Exception:  # noqa: BLE001 - skip a checkpoint we can't rebuild
                job_store.discard(cp.key)
                continue
            job.status = "queued"
            self._pending[job.id] = job
            self._resumed.append(job)
            self.job_queue.put(job)

    def _job_from_checkpoint(self, cp: "job_store.Checkpoint") -> Job:
        cfg = job_store.deserialize_config(cp.config)
        return Job(
            source_path=cfg["source_path"],
            model=cfg["model"],
            language=cfg["language"],
            polish=cfg["polish"],
            key=cp.key,  # same on-disk identity, so the worker finds this checkpoint
            audio_duration_seconds=cfg.get("audio_duration_seconds"),
            tracks=cfg.get("tracks"),
            selected_track_indices=cfg.get("selected_track_indices"),
            track_names=cfg.get("track_names"),
            track_languages=cfg.get("track_languages"),
            track_denoise=cfg.get("track_denoise"),
            track_subtractions=cfg.get("track_subtractions"),
            selected_channel_by_track=cfg.get("selected_channel_by_track"),
            separation_done=cfg.get("separation_done", False),
        )

    def resuming_jobs(self) -> list[dict]:
        """Jobs re-queued from disk this launch that are still in flight, so the
        UI can show them as working the moment it loads (live events follow)."""
        out = []
        for job in self._resumed:
            if job.status in ("done", "error", "stopped"):
                continue
            out.append({
                "id": job.id, "name": job.source_path.stem, "source": str(job.source_path),
                "status": "working", "resuming": True,
                "progress": int(round((job.transcribe_fraction or 0) * 100)),
                "detail": "Resuming…",
            })
        return out

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
        if job.status == "polishing":
            # polish emits no percentage; hand the UI an ETA so it can show a
            # "Polishing the text… about N left" tail instead of a frozen 100%.
            payload["polish_eta"] = int(job.estimated_polish_seconds or 0)
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
