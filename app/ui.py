from __future__ import annotations

import json
import queue
import sqlite3
import threading
import tkinter as tk
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from tkinter import ttk

from app.jobs import Job, WorkerEvent
from src import history_db
from src.audio_tracks import (
    AudioStream,
    compute_volume_envelope,
    correlation,
    safe_probe_audio_streams,
    track_speaker_names,
)
from src.catalog import DEFAULT_LANGUAGE, DEFAULT_MODEL, LANGUAGE_CHOICES, MODEL_CHOICES
from src.converter import check_ffmpeg_installed
from src.duration import probe_duration_seconds
from src.file_pickers import pick_files, pick_folder
from src.media_finder import SUPPORTED_EXTENSIONS, find_media_files
from src.transcription import ensure_mlx_whisper_available

try:
    from tkinterdnd2 import DND_FILES

    DND_AVAILABLE = True
except ImportError:
    DND_FILES = None
    DND_AVAILABLE = False

SETTINGS_PATH = Path.home() / "Library" / "Application Support" / "LocalTranscriber" / "settings.json"

# Short per-row model labels ("Tiny" instead of "Tiny - fastest, lowest
# accuracy") so the row's model picker stays compact.
SHORT_MODEL_CHOICES = [(label.split(" - ")[0], value) for label, value in MODEL_CHOICES]
SHORT_MODEL_LABEL_TO_VALUE = dict(SHORT_MODEL_CHOICES)
SHORT_MODEL_VALUE_TO_LABEL = {value: label for label, value in SHORT_MODEL_CHOICES}

LANGUAGE_LABEL_TO_VALUE = dict(LANGUAGE_CHOICES)
LANGUAGE_VALUE_TO_LABEL = {value: label for label, value in LANGUAGE_CHOICES}

HISTORY_COLUMNS = ("date", "file", "model", "language", "duration", "elapsed", "status")

ACTIVE_STATUSES = ("converting", "loading model", "transcribing")
CANCELABLE_STATUSES = ("queued",) + ACTIVE_STATUSES
TERMINAL_STATUSES = ("done", "error", "stopped")
TICK_MS = 400

ICON_START = "▶"
ICON_STOP = "■"
ICON_RESET = "↻"
ICON_BUSY = "…"


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


@dataclass
class RowWidgets:
    frame: ttk.Frame
    file_label: ttk.Label
    status_label: ttk.Label
    progress_bar: ttk.Progressbar
    detail_label: ttk.Label
    action_button: ttk.Button
    expand_frame: ttk.Frame
    model_var: tk.StringVar
    model_combo: ttk.Combobox
    language_var: tk.StringVar
    language_combo: ttk.Combobox
    tracks_container: ttk.Frame
    track_checkbuttons: list[ttk.Checkbutton] = field(default_factory=list)
    channel_combos: dict[int, ttk.Combobox] = field(default_factory=dict)
    # One waveform canvas per track (AudioStream.index) - shows whichever
    # channel(s) that track is currently set to use, even if it has more
    # than one underlying channel.
    track_canvases: dict[int, tk.Canvas] = field(default_factory=dict)


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
        self.jobs: dict[int, Job] = {}
        self.rows: dict[int, RowWidgets] = {}
        self.expanded: dict[int, bool] = {}
        # Match the waveform canvas background to the current ttk theme
        # instead of hardcoding a light color that clashes in dark mode.
        self.canvas_bg = ttk.Style().lookup("TFrame", "background") or root.cget("bg")

        root.title("Local Transcriber")
        root.geometry("900x600")
        root.minsize(560, 360)

        self.banner_container = tk.Frame(root)
        self.banner_container.pack(fill="x", side="top")

        self._build_notebook()
        self._refresh_history()
        self._check_dependencies_async()
        self._poll_events()
        self._tick()

    # -- startup checks -----------------------------------------------------

    def _check_dependencies_async(self) -> None:
        warnings = []
        if not check_ffmpeg_installed():
            warnings.append("ffmpeg not found on PATH (brew install ffmpeg)")
        if warnings:
            self._show_banner(warnings)

        # mlx-whisper's first import in a run can take a while (native/Metal
        # library init); do it off the GUI thread so the window paints
        # immediately instead of appearing blank while this runs.
        threading.Thread(target=self._check_mlx_whisper, daemon=True).start()

    def _check_mlx_whisper(self) -> None:
        try:
            ensure_mlx_whisper_available()
            error = None
        except RuntimeError as exc:
            error = str(exc)
        self.root.after(0, lambda: self._show_banner([error]) if error else None)

    def _show_banner(self, warnings: list[str]) -> None:
        banner = tk.Label(
            self.banner_container,
            text="Warning: " + " | ".join(warnings),
            fg="white",
            bg="#b3261e",
            anchor="w",
            padx=10,
            pady=4,
        )
        banner.pack(fill="x")

    # -- layout -----------------------------------------------------------

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

        ttk.Label(
            controls, text="(model and language are chosen per file below)", foreground="#777"
        ).grid(row=0, column=2, padx=(0, 0))

        drop_text = (
            "Drag audio or video files (or a whole folder) here to transcribe\n"
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
        if not DND_AVAILABLE:
            drop_text += "\n(drag-and-drop unavailable - install tkinterdnd2)"
        self.drop_zone = tk.Label(
            parent,
            text=drop_text,
            relief="ridge",
            borderwidth=2,
            bg="#f4f6fb",
            fg="#33415c",
            justify="center",
            pady=14,
        )
        self.drop_zone.pack(fill="x", padx=10, pady=(0, 10))
        self._register_drop_target(self.drop_zone)

        progress_frame = ttk.Frame(parent, padding=(10, 0, 10, 10))
        progress_frame.pack(fill="x")
        self.progress_title_var = tk.StringVar(value="Idle - click ▶ on a file below to start")
        ttk.Label(progress_frame, textvariable=self.progress_title_var, font=("", 11, "bold")).pack(
            anchor="w"
        )
        self.progress_bar = ttk.Progressbar(progress_frame, mode="determinate", maximum=100)
        self.progress_bar.pack(fill="x", pady=(4, 2))
        self.progress_detail_var = tk.StringVar(value="")
        ttk.Label(progress_frame, textvariable=self.progress_detail_var, foreground="#555").pack(anchor="w")

        self._build_queue_list(parent)

    # Column layout shared between the header row and every collapsed file
    # row, so widening the window gives the extra space to the file name
    # (column 0) while the rest stay a fixed, readable width. Model,
    # language, and track selection live in the expand panel instead of a
    # column, so a collapsed row stays a single glanceable line.
    def _configure_row_columns(self, frame: ttk.Frame) -> None:
        frame.grid_columnconfigure(0, weight=1, minsize=200)
        frame.grid_columnconfigure(1, weight=0, minsize=90)
        frame.grid_columnconfigure(2, weight=0, minsize=130)
        frame.grid_columnconfigure(3, weight=0, minsize=150)
        frame.grid_columnconfigure(4, weight=0, minsize=36)

    def _build_queue_list(self, parent: ttk.Frame) -> None:
        list_container = ttk.Frame(parent, padding=(10, 0, 10, 10))
        list_container.pack(fill="both", expand=True)

        header = ttk.Frame(list_container)
        header.pack(fill="x")
        self._configure_row_columns(header)
        headings = ["File", "Status", "Progress", "Estimate / Detail", ""]
        for col, text in enumerate(headings):
            ttk.Label(header, text=text, font=("", 9, "bold"), anchor="w").grid(
                row=0, column=col, sticky="w", padx=4
            )
        # Roughly matches the scrollbar width below so header labels line up
        # with the row columns.
        ttk.Frame(header, width=18).grid(row=0, column=len(headings), sticky="e")

        canvas = tk.Canvas(list_container, highlightthickness=0)
        scrollbar = ttk.Scrollbar(list_container, orient="vertical", command=canvas.yview)
        self.rows_frame = ttk.Frame(canvas)
        self.rows_frame.bind(
            "<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas_window = canvas.create_window((0, 0), window=self.rows_frame, anchor="nw")
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(canvas_window, width=e.width))
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.queue_canvas = canvas

        self._register_drop_target(canvas)
        self._register_drop_target(self.rows_frame)
        self._bind_mousewheel(canvas)
        self._bind_mousewheel(self.rows_frame)

    def _bind_mousewheel(self, widget: tk.Widget) -> None:
        widget.bind("<MouseWheel>", self._on_mousewheel)

    def _on_mousewheel(self, event) -> None:
        self.queue_canvas.yview_scroll(-1 * event.delta, "units")

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

    # -- drag and drop --------------------------------------------------------

    def _register_drop_target(self, widget: tk.Widget) -> None:
        if not DND_AVAILABLE:
            return
        try:
            widget.drop_target_register(DND_FILES)
            widget.dnd_bind("<<Drop>>", self._on_drop)
        except tk.TclError:
            # Root wasn't created as a tkinterdnd2 TkinterDnD.Tk() (e.g. the
            # tkdnd Tcl package failed to load on this platform/arch) - fail
            # soft, drag-and-drop just won't be available.
            pass

    def _on_drop(self, event) -> None:
        for raw_path in self.root.tk.splitlist(event.data):
            path = Path(raw_path).expanduser().resolve()
            if path.is_dir():
                for media_path in find_media_files(path):
                    self._add_job(media_path)
            elif path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                self._add_job(path)

    # -- selection state ----------------------------------------------------

    def _default_model(self) -> str:
        return self.settings.get("model", DEFAULT_MODEL)

    def _default_language(self) -> str | None:
        return self.settings.get("language", DEFAULT_LANGUAGE)

    def _persist_language_selection(self, language_value: str | None) -> None:
        self.settings["language"] = language_value
        _save_settings(self.settings)

    def _persist_model_selection(self, model_value: str) -> None:
        self.settings["model"] = model_value
        _save_settings(self.settings)

    # -- adding files ---------------------------------------------------------

    def _add_files(self) -> None:
        paths = pick_files()
        for path in paths:
            self._add_job(path)

    def _add_folder(self) -> None:
        folder = pick_folder()
        if folder is None:
            return
        for path in find_media_files(folder):
            self._add_job(path)

    def _add_job(self, path: Path) -> None:
        job = Job(source_path=path, model=self._default_model(), language=self._default_language())
        self.jobs[job.id] = job
        self._create_row(job)
        self._probe_estimate_async(job)

    def _file_label_text(self, job: Job, expanded: bool) -> str:
        arrow = "▾" if expanded else "▸"
        return f"{arrow} {job.source_path.name}"

    def _create_row(self, job: Job) -> None:
        row_frame = ttk.Frame(self.rows_frame, padding=(4, 6))
        row_frame.pack(fill="x")
        self._configure_row_columns(row_frame)

        file_label = ttk.Label(
            row_frame, text=self._file_label_text(job, False), anchor="w", cursor="hand2"
        )
        file_label.grid(row=0, column=0, sticky="ew", padx=4)
        file_label.bind("<Button-1>", lambda _e: self._toggle_expand(job.id))

        status_label = ttk.Label(row_frame, text=job.status, anchor="w")
        status_label.grid(row=0, column=1, sticky="w", padx=4)

        progress_bar = ttk.Progressbar(row_frame, length=120, mode="determinate", maximum=100)
        progress_bar.grid(row=0, column=2, sticky="w", padx=4)

        detail_label = ttk.Label(row_frame, text="-", anchor="w")
        detail_label.grid(row=0, column=3, sticky="w", padx=4)

        action_button = ttk.Button(
            row_frame, text=ICON_START, width=3, command=lambda: self._on_action_clicked(job.id)
        )
        action_button.grid(row=0, column=4, sticky="e", padx=4)

        ttk.Separator(self.rows_frame, orient="horizontal").pack(fill="x")

        expand_frame, model_var, model_combo, language_var, language_combo, tracks_container = (
            self._build_expand_panel(job)
        )

        self.rows[job.id] = RowWidgets(
            frame=row_frame,
            file_label=file_label,
            status_label=status_label,
            progress_bar=progress_bar,
            detail_label=detail_label,
            action_button=action_button,
            expand_frame=expand_frame,
            model_var=model_var,
            model_combo=model_combo,
            language_var=language_var,
            language_combo=language_combo,
            tracks_container=tracks_container,
        )
        self._bind_mousewheel(row_frame)
        for child in row_frame.winfo_children():
            self._bind_mousewheel(child)
        self._refresh_row(job)

    # The expand panel isn't packed here - it's built once up front and
    # shown/hidden by `_toggle_expand` so state (selected tracks, comboboxes)
    # survives collapsing the row.
    def _build_expand_panel(
        self, job: Job
    ) -> tuple[ttk.Frame, tk.StringVar, ttk.Combobox, tk.StringVar, ttk.Combobox, ttk.Frame]:
        expand_frame = ttk.Frame(self.rows_frame, padding=(24, 4, 10, 10))

        settings_row = ttk.Frame(expand_frame)
        settings_row.pack(fill="x", pady=(0, 10))

        ttk.Label(settings_row, text="Model:").pack(side="left", padx=(0, 4))
        model_var = tk.StringVar(value=SHORT_MODEL_VALUE_TO_LABEL.get(job.model, job.model))
        model_combo = ttk.Combobox(
            settings_row,
            textvariable=model_var,
            values=[label for label, _value in SHORT_MODEL_CHOICES],
            state="readonly",
            width=13,
        )
        model_combo.pack(side="left", padx=(0, 16))
        model_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_row_model_changed(job.id))

        ttk.Label(settings_row, text="Language:").pack(side="left", padx=(0, 4))
        language_var = tk.StringVar(value=LANGUAGE_VALUE_TO_LABEL.get(job.language, "Auto-detect"))
        language_combo = ttk.Combobox(
            settings_row,
            textvariable=language_var,
            values=[label for label, _value in LANGUAGE_CHOICES],
            state="readonly",
            width=11,
        )
        language_combo.pack(side="left")
        language_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_row_language_changed(job.id))

        ttk.Label(expand_frame, text="Audio tracks", font=("", 9, "bold")).pack(anchor="w")
        tracks_container = ttk.Frame(expand_frame)
        tracks_container.pack(fill="x", pady=(4, 0))

        self._bind_mousewheel(expand_frame)
        for child in (settings_row, tracks_container):
            self._bind_mousewheel(child)

        return expand_frame, model_var, model_combo, language_var, language_combo, tracks_container

    # -- expand / collapse ---------------------------------------------------

    def _toggle_expand(self, job_id: int) -> None:
        row = self.rows.get(job_id)
        job = self.jobs.get(job_id)
        if row is None or job is None:
            return
        expanded = not self.expanded.get(job_id, False)
        self.expanded[job_id] = expanded
        row.file_label.configure(text=self._file_label_text(job, expanded))
        if expanded:
            row.expand_frame.pack(fill="x", after=row.frame)
            self._ensure_tracks_loaded(job)
        else:
            row.expand_frame.pack_forget()

    # -- track selection & volume preview -------------------------------------

    def _ensure_tracks_loaded(self, job: Job) -> None:
        if job.tracks is None:
            self._render_tracks_section(job)  # shows "detecting..." placeholder
            self._probe_tracks_async(job)
        else:
            self._render_tracks_section(job)
            self._ensure_envelopes_loaded(job)

    def _probe_tracks_async(self, job: Job) -> None:
        def probe() -> None:
            streams = safe_probe_audio_streams(job.source_path)
            self.root.after(0, lambda: self._on_tracks_probed(job, streams))

        threading.Thread(target=probe, daemon=True).start()

    def _on_tracks_probed(self, job: Job, streams: list[AudioStream]) -> None:
        job.tracks = streams
        job.selected_track_indices = {s.index for s in streams}
        self._recompute_estimate(job)
        self._render_tracks_section(job)
        self._ensure_envelopes_loaded(job)

    def _render_tracks_section(self, job: Job) -> None:
        row = self.rows.get(job.id)
        if row is None:
            return
        for child in row.tracks_container.winfo_children():
            child.destroy()
        row.track_checkbuttons = []
        row.channel_combos = {}
        row.track_canvases = {}

        tracks = job.tracks
        if tracks is None:
            ttk.Label(row.tracks_container, text="Detecting audio tracks...", foreground="#777").pack(
                anchor="w"
            )
            return
        if not tracks:
            ttk.Label(
                row.tracks_container, text="No separate audio tracks detected.", foreground="#777"
            ).pack(anchor="w")
            return

        names = track_speaker_names(tracks)
        name_by_index = dict(zip((s.index for s in tracks), names))
        if job.selected_track_indices is None:
            job.selected_track_indices = {s.index for s in tracks}
        show_checkboxes = len(tracks) > 1
        identical_channel_pairs = job.channel_pairs_identical or {}
        duplicate_of = job.duplicate_of or {}

        for stream, name in zip(tracks, names):
            track_row = ttk.Frame(row.tracks_container)
            track_row.pack(fill="x", pady=1)
            self._bind_mousewheel(track_row)

            dup_source = duplicate_of.get(stream.index)
            label_text = name
            if dup_source is not None:
                label_text = f"{name} (~{name_by_index.get(dup_source, dup_source)})"

            if show_checkboxes:
                var = tk.BooleanVar(value=stream.index in job.selected_track_indices)
                checkbutton = ttk.Checkbutton(
                    track_row,
                    text=label_text,
                    variable=var,
                    command=lambda s=stream, v=var: self._on_track_toggle(job, s.index, v),
                )
                checkbutton.configure(state="normal" if job.status == "ready" else "disabled")
                checkbutton.pack(side="left", padx=(0, 6))
                row.track_checkbuttons.append(checkbutton)
            else:
                ttk.Label(track_row, text=label_text, width=14, anchor="w").pack(side="left", padx=(0, 6))

            channels_identical = stream.channels > 1 and identical_channel_pairs.get(stream.index, False)

            # A stream's channels aren't assumed to carry the same signal
            # (e.g. two mics recorded to L/R of one stream) - let the user
            # pin this track to just one of them instead of always
            # downmixing every channel together. Skipped when both channels
            # turned out to be the same signal anyway (e.g. a mono mic
            # duplicated to stereo) - picking "Ch 1" vs "Ch 2" would be a
            # meaningless choice, so don't show it.
            if stream.channels > 1 and not channels_identical:
                channel_var = tk.StringVar(value=self._channel_mode_label(job, stream))
                channel_combo = ttk.Combobox(
                    track_row,
                    textvariable=channel_var,
                    values=["Mix"] + [f"Ch {i + 1}" for i in range(stream.channels)],
                    state="readonly",
                    width=6,
                )
                channel_combo.configure(state="readonly" if job.status == "ready" else "disabled")
                channel_combo.pack(side="left", padx=(0, 6))
                channel_combo.bind(
                    "<<ComboboxSelected>>",
                    lambda _e, s=stream, v=channel_var: self._on_channel_mode_changed(job, s.index, v),
                )
                row.channel_combos[stream.index] = channel_combo

            canvas = tk.Canvas(track_row, height=18, bg=self.canvas_bg, highlightthickness=0)
            canvas.pack(side="left", fill="x", expand=True)
            canvas._envelope = self._representative_envelope(job, stream)
            canvas.bind("<Configure>", lambda _e, c=canvas: self._draw_envelope(c))
            row.track_canvases[stream.index] = canvas

    def _channel_mode_label(self, job: Job, stream: AudioStream) -> str:
        channel_index = (job.selected_channel_by_track or {}).get(stream.index)
        if channel_index is None:
            return "Mix"
        return f"Ch {channel_index + 1}"

    def _on_channel_mode_changed(self, job: Job, stream_index: int, var: tk.StringVar) -> None:
        if job.selected_channel_by_track is None:
            job.selected_channel_by_track = {}
        label = var.get()
        if label.startswith("Ch "):
            job.selected_channel_by_track[stream_index] = int(label.rsplit(" ", 1)[-1]) - 1
        else:
            job.selected_channel_by_track.pop(stream_index, None)
        self._redraw_track(job, stream_index)

    def _representative_envelope(self, job: Job, stream: AudioStream) -> list[float] | None:
        """The single series drawn for a track's waveform: the one channel
        if it's mono (or its channels are duplicates of each other), the
        chosen channel if the user pinned one, otherwise every loaded
        channel averaged together."""
        envelopes = job.track_envelopes or {}
        if stream.channels <= 1 or (job.channel_pairs_identical or {}).get(stream.index):
            return envelopes.get((stream.index, 0))
        channel_index = (job.selected_channel_by_track or {}).get(stream.index)
        if channel_index is not None:
            return envelopes.get((stream.index, channel_index))
        available = [envelopes[(stream.index, ch)] for ch in range(stream.channels) if (stream.index, ch) in envelopes]
        if not available:
            return None
        length = min(len(e) for e in available)
        if length == 0:
            return []
        return [sum(e[i] for e in available) / len(available) for i in range(length)]

    def _redraw_track(self, job: Job, stream_index: int) -> None:
        row = self.rows.get(job.id)
        stream = next((s for s in (job.tracks or []) if s.index == stream_index), None)
        if row is None or stream is None:
            return
        canvas = row.track_canvases.get(stream_index)
        if canvas is not None:
            canvas._envelope = self._representative_envelope(job, stream)
            self._draw_envelope(canvas)

    def _ensure_envelopes_loaded(self, job: Job) -> None:
        if not job.tracks:
            return
        if job.track_envelopes is None:
            job.track_envelopes = {}
        for stream in job.tracks:
            for channel_index in range(max(1, stream.channels)):
                if (stream.index, channel_index) in job.track_envelopes:
                    continue
                self._compute_envelope_async(job, stream, channel_index)

    def _compute_envelope_async(self, job: Job, stream: AudioStream, channel_index: int) -> None:
        def compute() -> None:
            try:
                envelope = compute_volume_envelope(
                    job.source_path,
                    stream.index,
                    channel_index=channel_index if stream.channels > 1 else None,
                )
            except Exception:
                envelope = []
            self.root.after(0, lambda: self._on_envelope_computed(job, stream.index, channel_index, envelope))

        threading.Thread(target=compute, daemon=True).start()

    def _on_envelope_computed(
        self, job: Job, stream_index: int, channel_index: int, envelope: list[float]
    ) -> None:
        if job.track_envelopes is None:
            job.track_envelopes = {}
        job.track_envelopes[(stream_index, channel_index)] = envelope
        self._redraw_track(job, stream_index)
        self._maybe_analyze_tracks(job)

    # Runs once, as soon as every channel's waveform has been computed:
    # flags stereo tracks whose two channels are the same signal (collapses
    # their waveform display to one row - see `_render_tracks_section`), and
    # flags/deselects tracks that are near-duplicates of an earlier track
    # (e.g. the same OBS source routed to more than one output track).
    def _maybe_analyze_tracks(self, job: Job) -> None:
        if job.duplicates_analyzed or not job.tracks or job.track_envelopes is None:
            return
        channels_needed = sum(max(1, s.channels) for s in job.tracks)
        if len(job.track_envelopes) < channels_needed:
            return
        for stream in job.tracks:
            for channel_index in range(max(1, stream.channels)):
                if (stream.index, channel_index) not in job.track_envelopes:
                    return  # a key is still missing even though the count matches

        job.duplicates_analyzed = True

        job.channel_pairs_identical = {}
        representative_envelope: dict[int, list[float]] = {}
        for stream in job.tracks:
            first_channel = job.track_envelopes.get((stream.index, 0), [])
            representative_envelope[stream.index] = first_channel
            if stream.channels > 1:
                second_channel = job.track_envelopes.get((stream.index, 1), [])
                job.channel_pairs_identical[stream.index] = correlation(first_channel, second_channel) >= 0.98

        job.duplicate_of = {}
        for i, stream in enumerate(job.tracks):
            for earlier in job.tracks[:i]:
                score = correlation(representative_envelope[stream.index], representative_envelope[earlier.index])
                if score >= 0.995:
                    job.duplicate_of[stream.index] = earlier.index
                    break

        # Default to unchecking detected duplicates - but only as the
        # initial suggestion. If the user already changed the selection by
        # hand before this analysis finished, leave it alone.
        if job.selected_track_indices == {s.index for s in job.tracks}:
            job.selected_track_indices -= set(job.duplicate_of)
            if not job.selected_track_indices:
                job.selected_track_indices = {s.index for s in job.tracks}

        self._recompute_estimate(job)
        self._render_tracks_section(job)

    def _draw_envelope(self, canvas: tk.Canvas) -> None:
        envelope = getattr(canvas, "_envelope", None)
        canvas.delete("all")
        canvas.configure(bg=self.canvas_bg)
        width = canvas.winfo_width()
        height = canvas.winfo_height()
        if width <= 1 or height <= 1:
            return
        mid = height / 2
        # Loading (None), silent, or single-sample: a flat mid-line reads as
        # "nothing here yet/nothing to see" without extra caption text.
        if not envelope or len(envelope) < 2 or max(envelope) <= 0:
            canvas.create_line(0, mid, width, mid, fill="#8a8a8a")
            return
        n = len(envelope)
        step = width / (n - 1)
        points: list[float] = []
        for i, level in enumerate(envelope):
            points.append(i * step)
            points.append(mid - level * (mid - 1))
        canvas.create_line(*points, fill="#4c6ef5", width=1.5, smooth=True)

    def _on_track_toggle(self, job: Job, stream_index: int, var: tk.BooleanVar) -> None:
        if job.selected_track_indices is None:
            job.selected_track_indices = {s.index for s in job.tracks or []}
        if var.get():
            job.selected_track_indices.add(stream_index)
        else:
            if len(job.selected_track_indices) <= 1:
                # Always keep at least one track selected.
                var.set(True)
                return
            job.selected_track_indices.discard(stream_index)
        self._recompute_estimate(job)

    # -- pre-start time estimate ----------------------------------------------

    def _probe_estimate_async(self, job: Job) -> None:
        def probe() -> None:
            try:
                duration = probe_duration_seconds(job.source_path)
            except Exception:
                duration = None
            self.root.after(0, lambda: self._on_duration_probed(job, duration))

        threading.Thread(target=probe, daemon=True).start()

    def _on_duration_probed(self, job: Job, duration: float | None) -> None:
        if job.status != "ready" or duration is None:
            return
        job.audio_duration_seconds = duration
        self._recompute_estimate(job)

    def _recompute_estimate(self, job: Job) -> None:
        if job.audio_duration_seconds is None:
            return
        estimate = history_db.estimate_seconds(
            self.db_conn, job.model, job.language, job.audio_duration_seconds
        )
        # Mirrors the worker: transcription is one full pass per selected
        # track when the container has more than one audio stream.
        if job.tracks and len(job.tracks) > 1:
            selected_count = len(job.selected_track_indices or job.tracks)
            estimate *= max(1, selected_count)
        job.estimated_seconds = estimate
        self._refresh_row(job)

    def _on_row_model_changed(self, job_id: int) -> None:
        job = self.jobs.get(job_id)
        row = self.rows.get(job_id)
        if job is None or row is None:
            return
        job.model = SHORT_MODEL_LABEL_TO_VALUE.get(row.model_var.get(), job.model)
        self._persist_model_selection(job.model)
        self._recompute_estimate(job)

    def _on_row_language_changed(self, job_id: int) -> None:
        job = self.jobs.get(job_id)
        row = self.rows.get(job_id)
        if job is None or row is None:
            return
        job.language = LANGUAGE_LABEL_TO_VALUE.get(row.language_var.get(), job.language)
        self._persist_language_selection(job.language)
        self._recompute_estimate(job)

    # -- start / stop / reset -------------------------------------------------

    def _on_action_clicked(self, job_id: int) -> None:
        job = self.jobs.get(job_id)
        if job is None:
            return
        if job.status == "ready":
            self._start_job(job)
        elif job.status in CANCELABLE_STATUSES:
            self._stop_job(job)
        elif job.status in TERMINAL_STATUSES:
            self._reset_job(job)

    def _start_job(self, job: Job) -> None:
        job.status = "queued"
        self._refresh_row(job)
        self.job_queue.put(job)

    def _stop_job(self, job: Job) -> None:
        job.stop_event.set()
        row = self.rows.get(job.id)
        if row is not None:
            row.action_button.configure(text=ICON_BUSY, state="disabled")

    def _reset_job(self, job: Job) -> None:
        job.status = "ready"
        job.stop_event = threading.Event()
        job.estimated_seconds = None
        job.started_at = None
        job.elapsed_seconds = None
        job.output_path = None
        job.error = None
        self._refresh_row(job)
        self._recompute_estimate(job)

    # -- event loop -----------------------------------------------------------

    def _poll_events(self) -> None:
        try:
            while True:
                event = self.event_queue.get_nowait()
                self.jobs[event.job.id] = event.job
                self._apply_event(event.job)
        except queue.Empty:
            pass
        self.root.after(150, self._poll_events)

    def _apply_event(self, job: Job) -> None:
        if job.id not in self.rows:
            return
        if job.status in TERMINAL_STATUSES:
            self._refresh_history()
        self._refresh_row(job)

    def _tick(self) -> None:
        active_job = next(
            (job for job in self.jobs.values() if job.status in ACTIVE_STATUSES),
            None,
        )
        for job in self.jobs.values():
            if job.status in ACTIVE_STATUSES and job.id in self.rows:
                self._refresh_row(job)

        self._update_progress_panel(active_job)
        self.root.after(TICK_MS, self._tick)

    # -- rendering --------------------------------------------------------------

    def _action_button_state(self, job: Job) -> tuple[str, str]:
        if job.status == "ready":
            return ICON_START, "normal"
        if job.status in CANCELABLE_STATUSES:
            return ICON_STOP, "normal"
        if job.status in TERMINAL_STATUSES:
            return ICON_RESET, "normal"
        return ICON_BUSY, "disabled"

    def _refresh_row(self, job: Job) -> None:
        row = self.rows.get(job.id)
        if row is None:
            return

        elapsed = None
        if job.status in ACTIVE_STATUSES and job.started_at:
            elapsed = (datetime.now() - job.started_at).total_seconds()
        elif job.elapsed_seconds is not None:
            elapsed = job.elapsed_seconds

        status_text = f"error: {job.error}" if job.status == "error" and job.error else job.status
        if job.status_detail:
            status_text = f"{status_text} ({job.status_detail})"
        row.status_label.configure(text=status_text)
        row.model_combo.configure(state="readonly" if job.status == "ready" else "disabled")
        row.language_combo.configure(state="readonly" if job.status == "ready" else "disabled")
        for checkbutton in row.track_checkbuttons:
            checkbutton.configure(state="normal" if job.status == "ready" else "disabled")
        for channel_combo in row.channel_combos.values():
            channel_combo.configure(state="readonly" if job.status == "ready" else "disabled")

        if job.status == "transcribing" and job.estimated_seconds:
            fraction = min(0.99, (elapsed or 0.0) / job.estimated_seconds)
            row.progress_bar.stop()
            row.progress_bar.configure(mode="determinate")
            row.progress_bar["value"] = fraction * 100
            remaining = max(0.0, job.estimated_seconds - (elapsed or 0.0))
            row.detail_label.configure(text=f"{round(fraction * 100)}% - ~{format_seconds(remaining)} left")
        elif job.status == "loading model":
            row.progress_bar.configure(mode="indeterminate")
            row.progress_bar.start(15)
            row.detail_label.configure(text="loading model...")
        elif job.status == "done":
            row.progress_bar.stop()
            row.progress_bar.configure(mode="determinate")
            row.progress_bar["value"] = 100
            row.detail_label.configure(text=f"done in {format_seconds(job.elapsed_seconds)}")
        elif job.status == "stopped":
            row.progress_bar.stop()
            row.progress_bar.configure(mode="determinate")
            row.progress_bar["value"] = 0
            row.detail_label.configure(text="stopped")
        elif job.status == "ready":
            row.progress_bar.stop()
            row.progress_bar.configure(mode="determinate")
            row.progress_bar["value"] = 0
            if job.estimated_seconds:
                row.detail_label.configure(text=f"~{format_seconds(job.estimated_seconds)} estimated")
            else:
                row.detail_label.configure(text="estimating..." if job.audio_duration_seconds is None else "-")
        elif job.status == "queued":
            row.progress_bar.stop()
            row.progress_bar.configure(mode="determinate")
            row.progress_bar["value"] = 0
            if job.estimated_seconds:
                row.detail_label.configure(text=f"waiting... (~{format_seconds(job.estimated_seconds)})")
            else:
                row.detail_label.configure(text="waiting...")
        else:
            row.progress_bar.stop()
            row.progress_bar.configure(mode="determinate")
            row.progress_bar["value"] = 0
            row.detail_label.configure(text=format_seconds(elapsed) if elapsed is not None else "-")

        action_text, action_state = self._action_button_state(job)
        row.action_button.configure(text=action_text, state=action_state)

    def _update_progress_panel(self, job: Job | None) -> None:
        if job is None:
            self.progress_bar.stop()
            self.progress_bar.configure(mode="determinate")
            self.progress_bar["value"] = 0
            self.progress_title_var.set("Idle - click ▶ on a file below to start")
            self.progress_detail_var.set("")
            return

        elapsed = (datetime.now() - job.started_at).total_seconds() if job.started_at else 0.0
        title = f"{job.status.capitalize()}: {job.source_path.name}"
        if job.status_detail:
            title = f"{title} ({job.status_detail})"
        self.progress_title_var.set(title)

        if job.status == "loading model":
            self.progress_bar.configure(mode="indeterminate")
            self.progress_bar.start(15)
            self.progress_detail_var.set(
                "Loading model - first use of a model downloads it and can take a while."
            )
            return

        self.progress_bar.stop()
        self.progress_bar.configure(mode="determinate")
        if job.estimated_seconds:
            fraction = min(0.99, elapsed / job.estimated_seconds)
            self.progress_bar["value"] = fraction * 100
            remaining = max(0.0, job.estimated_seconds - elapsed)
            self.progress_detail_var.set(
                f"{round(fraction * 100)}%  •  ~{format_seconds(remaining)} remaining "
                f"(est. {format_seconds(job.estimated_seconds)} total, {format_seconds(elapsed)} elapsed)"
            )
        else:
            self.progress_bar["value"] = 0
            self.progress_detail_var.set(f"{format_seconds(elapsed)} elapsed - estimating...")

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
