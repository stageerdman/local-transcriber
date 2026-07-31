from __future__ import annotations

import json
import queue
import sqlite3
import tkinter as tk
from pathlib import Path
from tkinter import ttk

from app.jobs import Job, WorkerEvent
from src import history_db
from src.catalog import DEFAULT_LANGUAGE, DEFAULT_MODEL, LANGUAGE_CHOICES, MODEL_CHOICES
from src.converter import check_ffmpeg_installed
from src.file_pickers import pick_files, pick_folder
from src.media_finder import find_media_files
from src.transcription import ensure_mlx_whisper_available

SETTINGS_PATH = Path.home() / "Library" / "Application Support" / "LocalTranscriber" / "settings.json"

MODEL_LABEL_TO_VALUE = dict(MODEL_CHOICES)
MODEL_VALUE_TO_LABEL = {value: label for label, value in MODEL_CHOICES}
LANGUAGE_LABEL_TO_VALUE = dict(LANGUAGE_CHOICES)
LANGUAGE_VALUE_TO_LABEL = {value: label for label, value in LANGUAGE_CHOICES}

QUEUE_COLUMNS = ("file", "status", "estimated", "elapsed")
HISTORY_COLUMNS = ("date", "file", "model", "language", "duration", "elapsed", "status")


def _load_settings() -> dict:
    if SETTINGS_PATH.exists():
        try:
            return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_settings(settings: dict) -> None:
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(settings), encoding="utf-8")


def format_seconds(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    seconds = max(0, round(seconds))
    minutes, secs = divmod(seconds, 60)
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


class MainWindow:
    def __init__(
        self,
        root: tk.Tk,
        job_queue: "queue.Queue[Job | object]",
        event_queue: "queue.Queue[WorkerEvent]",
        db_conn: sqlite3.Connection,
    ) -> None:
        self.root = root
        self.job_queue = job_queue
        self.event_queue = event_queue
        self.db_conn = db_conn
        self.settings = _load_settings()

        root.title("Local Transcriber")
        root.geometry("820x520")

        self._build_banner()
        self._build_notebook()
        self._refresh_history()
        self._poll_events()

    # -- layout -----------------------------------------------------------

    def _build_banner(self) -> None:
        warnings = []
        if not check_ffmpeg_installed():
            warnings.append("ffmpeg not found on PATH (brew install ffmpeg)")
        try:
            ensure_mlx_whisper_available()
        except RuntimeError as exc:
            warnings.append(str(exc))

        if warnings:
            banner = tk.Label(
                self.root,
                text="Warning: " + " | ".join(warnings),
                fg="white",
                bg="#b3261e",
                anchor="w",
                padx=10,
                pady=4,
            )
            banner.pack(fill="x")

    def _build_notebook(self) -> None:
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True)

        transcribe_tab = ttk.Frame(notebook)
        history_tab = ttk.Frame(notebook)
        notebook.add(transcribe_tab, text="Transcribe")
        notebook.add(history_tab, text="History")
        notebook.bind("<<NotebookTabChanged>>", lambda _event: self._refresh_history())

        self._build_transcribe_tab(transcribe_tab)
        self._build_history_tab(history_tab)

    def _build_transcribe_tab(self, parent: ttk.Frame) -> None:
        controls = ttk.Frame(parent, padding=10)
        controls.pack(fill="x")

        ttk.Button(controls, text="Add File...", command=self._add_files).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(controls, text="Add Folder...", command=self._add_folder).grid(row=0, column=1, padx=(0, 16))

        ttk.Label(controls, text="Model:").grid(row=0, column=2, padx=(0, 4))
        model_labels = [label for label, _value in MODEL_CHOICES]
        default_model_label = MODEL_VALUE_TO_LABEL.get(
            self.settings.get("model", DEFAULT_MODEL), model_labels[2]
        )
        self.model_var = tk.StringVar(value=default_model_label)
        model_combo = ttk.Combobox(
            controls, textvariable=self.model_var, values=model_labels, state="readonly", width=32
        )
        model_combo.grid(row=0, column=3, padx=(0, 16))
        model_combo.bind("<<ComboboxSelected>>", lambda _event: self._persist_selection())

        ttk.Label(controls, text="Language:").grid(row=0, column=4, padx=(0, 4))
        language_labels = [label for label, _value in LANGUAGE_CHOICES]
        default_language_label = LANGUAGE_VALUE_TO_LABEL.get(
            self.settings.get("language", DEFAULT_LANGUAGE), language_labels[1]
        )
        self.language_var = tk.StringVar(value=default_language_label)
        language_combo = ttk.Combobox(
            controls, textvariable=self.language_var, values=language_labels, state="readonly", width=14
        )
        language_combo.grid(row=0, column=5)
        language_combo.bind("<<ComboboxSelected>>", lambda _event: self._persist_selection())

        tree_frame = ttk.Frame(parent, padding=(10, 0, 10, 10))
        tree_frame.pack(fill="both", expand=True)

        self.queue_tree = ttk.Treeview(tree_frame, columns=QUEUE_COLUMNS, show="headings")
        headings = {"file": "File", "status": "Status", "estimated": "Estimated", "elapsed": "Elapsed"}
        widths = {"file": 400, "status": 120, "estimated": 100, "elapsed": 100}
        for col in QUEUE_COLUMNS:
            self.queue_tree.heading(col, text=headings[col])
            self.queue_tree.column(col, width=widths[col], anchor="w")
        self.queue_tree.pack(fill="both", expand=True)

    def _build_history_tab(self, parent: ttk.Frame) -> None:
        tree_frame = ttk.Frame(parent, padding=10)
        tree_frame.pack(fill="both", expand=True)

        self.history_tree = ttk.Treeview(tree_frame, columns=HISTORY_COLUMNS, show="headings")
        headings = {
            "date": "Date",
            "file": "File",
            "model": "Model",
            "language": "Language",
            "duration": "Duration",
            "elapsed": "Elapsed",
            "status": "Status",
        }
        widths = {"date": 140, "file": 260, "model": 140, "language": 90, "duration": 90, "elapsed": 90, "status": 80}
        for col in HISTORY_COLUMNS:
            self.history_tree.heading(col, text=headings[col])
            self.history_tree.column(col, width=widths[col], anchor="w")
        self.history_tree.pack(fill="both", expand=True)

    # -- selection state ----------------------------------------------------

    def _selected_model(self) -> str:
        return MODEL_LABEL_TO_VALUE.get(self.model_var.get(), DEFAULT_MODEL)

    def _selected_language(self) -> str | None:
        return LANGUAGE_LABEL_TO_VALUE.get(self.language_var.get(), DEFAULT_LANGUAGE)

    def _persist_selection(self) -> None:
        self.settings["model"] = self._selected_model()
        self.settings["language"] = self._selected_language()
        _save_settings(self.settings)

    # -- actions ------------------------------------------------------------

    def _add_files(self) -> None:
        paths = pick_files()
        for path in paths:
            self._enqueue(path)

    def _add_folder(self) -> None:
        folder = pick_folder()
        if folder is None:
            return
        for path in find_media_files(folder):
            self._enqueue(path)

    def _enqueue(self, path: Path) -> None:
        job = Job(source_path=path, model=self._selected_model(), language=self._selected_language())
        self.queue_tree.insert(
            "", "end", iid=str(job.id), values=(str(path), job.status, "-", "-")
        )
        self.job_queue.put(job)

    # -- event loop -----------------------------------------------------------

    def _poll_events(self) -> None:
        try:
            while True:
                event = self.event_queue.get_nowait()
                self._apply_event(event.job)
        except queue.Empty:
            pass
        self.root.after(150, self._poll_events)

    def _apply_event(self, job: Job) -> None:
        iid = str(job.id)
        if not self.queue_tree.exists(iid):
            return
        self.queue_tree.item(
            iid,
            values=(
                str(job.source_path),
                job.error if job.status == "error" else job.status,
                format_seconds(job.estimated_seconds),
                format_seconds(job.elapsed_seconds),
            ),
        )
        if job.status in ("done", "error"):
            self._refresh_history()

    def _refresh_history(self) -> None:
        self.history_tree.delete(*self.history_tree.get_children())
        for row in history_db.fetch_history(self.db_conn):
            self.history_tree.insert(
                "",
                "end",
                iid=str(row.id),
                values=(
                    row.finished_at or row.started_at,
                    row.source_path,
                    row.model,
                    row.language or "auto",
                    format_seconds(row.audio_duration_seconds),
                    format_seconds(row.elapsed_seconds),
                    row.status,
                ),
            )
