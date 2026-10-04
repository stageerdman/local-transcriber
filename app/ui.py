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
from src import history_db, track_separation
from src.audio_tracks import (
    AudioStream,
    compute_volume_envelope,
    correlation,
    safe_probe_audio_streams,
    track_speaker_names,
)
from src.catalog import (
    DEFAULT_LANGUAGE,
    DEFAULT_MODEL,
    DEFAULT_POLISH_ENABLED,
    LANGUAGE_CHOICES,
    MODEL_CHOICES,
    POLISH_BASE_SECONDS,
    POLISH_SHARE_MAX,
    POLISH_SHARE_MIN,
    estimate_polish_seconds,
)
from src.converter import check_ffmpeg_installed
from src.duration import probe_duration_seconds
from src.file_pickers import pick_files, pick_folder
from src.media_finder import SUPPORTED_EXTENSIONS, find_media_files
from src.playback import TrackPlayer
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

ACTIVE_STATUSES = ("converting", "loading model", "transcribing", "polishing")
CANCELABLE_STATUSES = ("queued",) + ACTIVE_STATUSES
TERMINAL_STATUSES = ("done", "error", "stopped")
TICK_MS = 400

ICON_START = "▶"
ICON_STOP = "■"
ICON_RESET = "↻"
ICON_BUSY = "…"

# Amber used only for a genuine warning (unresolved shared voice / re-created
# doubling); distinct from the gray status text and meets contrast on the panel.
WARN_FG = "#8a6500"
# Calm status palette for the people-first track panel.
OK_FG = "#2b8a3e"      # "set up / handled" confirmations
ACCENT_FG = "#4c6ef5"  # inline action links (Undo / Include)
MUTED_FG = "#868e96"   # skipped / secondary text

# Waveform / seek-bar colors (readable in both light and dark themes).
WAVEFORM_UNPLAYED = "#9aa0a6"
WAVEFORM_PLAYED = "#4c6ef5"
WAVEFORM_PLAYHEAD = "#f03e3e"
WAVEFORM_TIME = "#868e96"
WAVEFORM_HEIGHT = 24


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


def format_timecode(seconds: float | None) -> str:
    """A clock-style position, e.g. "12:30" or "1:04:05" - used to show where
    in the recording transcription currently is."""
    if seconds is None:
        return "0:00"
    total = max(0, int(round(seconds)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def format_minutes_left(seconds: float | None) -> str | None:
    """Coarse remaining-time phrase ("<1 min", "9 min") for an ETA - kept
    deliberately rounded since it's always an approximation."""
    if seconds is None or seconds < 0:
        return None
    if seconds < 45:
        return "<1 min"
    return f"{int(round(seconds / 60))} min"


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
    polish_var: tk.BooleanVar
    tracks_container: ttk.Frame
    track_checkbuttons: list[ttk.Checkbutton] = field(default_factory=list)
    channel_combos: dict[int, ttk.Combobox] = field(default_factory=dict)
    # One preview play/stop button per track (keyed by AudioStream.index).
    track_play_buttons: dict[int, ttk.Button] = field(default_factory=dict)
    # One waveform canvas per track (AudioStream.index) - shows whichever
    # channel(s) that track is currently set to use, even if it has more
    # than one underlying channel.
    track_canvases: dict[int, tk.Canvas] = field(default_factory=dict)
    # Per-track manual-control widgets (keyed by AudioStream.index).
    track_name_vars: dict[int, tk.StringVar] = field(default_factory=dict)
    track_name_entries: dict[int, ttk.Entry] = field(default_factory=dict)
    track_status_labels: dict[int, ttk.Label] = field(default_factory=dict)
    # Per-track "why" line under the row (reason + an inline include/undo link).
    track_reason_frames: dict[int, ttk.Frame] = field(default_factory=dict)
    track_detail_frames: dict[int, ttk.Frame] = field(default_factory=dict)
    tracks_header: "ttk.Label | None" = None
    # Persistent holder for the shared-voice warning banner, so it can be
    # re-rendered in place (on selection/subtraction changes) without a full
    # section rebuild.
    shared_voice_holder: "ttk.Frame | None" = None


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

        # Track preview playback: one track audible at a time, played (and
        # seeked) straight from the source via ffplay - no extraction step.
        self._player = TrackPlayer()
        # (job_id, stream_index) of the track currently playing, or None.
        self._playing: tuple[int, int] | None = None
        # (job.id, AudioStream.index) pairs whose per-track detail drawer is
        # open in the panel.
        self._track_details_open: set[tuple[int, int]] = set()
        # Pre-fix snapshot per job id, so a one-click shared-voice fix is undoable.
        self._shared_voice_undo: dict[int, tuple] = {}
        # Playhead per track as a fraction [0, 1] of its duration. Persists
        # after stop so ▶ resumes where it left off; reset to 0 when another
        # track takes over.
        self._playhead: dict[tuple[int, int], float] = {}
        # While dragging on a waveform to scrub: the track key and the fraction
        # under the cursor. Audio is (re)started on mouse release.
        self._scrub_key: tuple[int, int] | None = None
        self._scrub_fraction: float = 0.0
        self._playhead_tick()

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

    def _default_polish(self) -> bool:
        return self.settings.get("polish", DEFAULT_POLISH_ENABLED)

    def _persist_polish_selection(self, enabled: bool) -> None:
        self.settings["polish"] = enabled
        _save_settings(self.settings)

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
        job = Job(
            source_path=path,
            model=self._default_model(),
            language=self._default_language(),
            polish=self._default_polish(),
        )
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

        (expand_frame, model_var, model_combo, language_var, language_combo,
         polish_var, tracks_container) = self._build_expand_panel(job)

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
            polish_var=polish_var,
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
    ) -> tuple[ttk.Frame, tk.StringVar, ttk.Combobox, tk.StringVar, ttk.Combobox,
               tk.BooleanVar, ttk.Frame]:
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

        polish_var = tk.BooleanVar(value=job.polish)
        polish_check = ttk.Checkbutton(
            settings_row,
            text="Polish (LLM cleanup)",
            variable=polish_var,
            command=lambda: self._on_row_polish_changed(job.id),
        )
        polish_check.pack(side="left", padx=(16, 0))

        ttk.Label(expand_frame, text="Audio tracks", font=("", 9, "bold")).pack(anchor="w")
        tracks_container = ttk.Frame(expand_frame)
        tracks_container.pack(fill="x", pady=(4, 0))

        self._bind_mousewheel(expand_frame)
        for child in (settings_row, tracks_container):
            self._bind_mousewheel(child)

        return (expand_frame, model_var, model_combo, language_var, language_combo,
                polish_var, tracks_container)

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
        row.track_play_buttons = {}
        row.track_name_vars = {}
        row.track_name_entries = {}
        row.track_status_labels = {}
        row.track_reason_frames = {}
        row.track_detail_frames = {}
        row.tracks_header = None
        row.shared_voice_holder = None

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

        if job.selected_track_indices is None:
            job.selected_track_indices = {s.index for s in tracks}
        show_checkboxes = len(tracks) > 1
        editable = job.status == "ready"

        if show_checkboxes:
            header = ttk.Label(row.tracks_container, foreground="#777")
            header.pack(anchor="w", pady=(0, 3))
            row.tracks_header = header
            self._update_tracks_header(job)
            holder = ttk.Frame(row.tracks_container)
            holder.pack(fill="x")
            row.shared_voice_holder = holder
            self._render_shared_voice_banner(job)

        for stream in tracks:
            track_row = ttk.Frame(row.tracks_container)
            track_row.pack(fill="x", pady=1)
            self._bind_mousewheel(track_row)

            # Leading, fixed-position preview button so every row's play
            # target lines up regardless of whether a channel picker is shown.
            is_playing = self._playing == (job.id, stream.index)
            play_button = ttk.Button(
                track_row,
                text=ICON_STOP if is_playing else ICON_START,
                width=2,
                command=lambda s=stream: self._on_track_play(job, s),
            )
            play_enabled = job.status == "ready" and TrackPlayer.available()
            play_button.configure(state="normal" if play_enabled else "disabled")
            play_button.pack(side="left", padx=(0, 6))
            row.track_play_buttons[stream.index] = play_button

            if show_checkboxes:
                var = tk.BooleanVar(value=stream.index in job.selected_track_indices)
                checkbutton = ttk.Checkbutton(
                    track_row,
                    variable=var,
                    command=lambda s=stream, v=var: self._on_track_toggle(job, s.index, v),
                )
                checkbutton.configure(state="normal" if editable else "disabled")
                checkbutton.pack(side="left")
                row.track_checkbuttons.append(checkbutton)

            # Editable speaker name - becomes the transcript's speaker label.
            # A combobox (free-type) offers recently-used names, since the same
            # speakers ("Klient"/"Obchodník") recur across calls.
            name_var = tk.StringVar(value=self._current_track_name(job, stream))
            row.track_name_vars[stream.index] = name_var
            name_entry = ttk.Combobox(
                track_row, textvariable=name_var, width=13, values=self._recent_names()
            )
            name_entry.configure(state="normal" if editable else "disabled")
            name_entry.pack(side="left", padx=(2, 6))
            name_entry.bind("<Return>", lambda _e, s=stream, v=name_var: self._commit_track_name(job, s, v))
            name_entry.bind("<FocusOut>", lambda _e, s=stream, v=name_var: self._commit_track_name(job, s, v))
            name_entry.bind("<<ComboboxSelected>>", lambda _e, s=stream, v=name_var: self._commit_track_name(job, s, v))
            row.track_name_entries[stream.index] = name_entry

            # Detail-drawer disclosure (language / noise / remove-a-voice).
            detail_button = ttk.Button(
                track_row,
                text="▾" if (job.id, stream.index) in self._track_details_open else "▸",
                width=2,
                command=lambda s=stream: self._toggle_track_detail(job, s.index),
            )
            detail_button.pack(side="left", padx=(0, 6))

            # Right-edge read-only status summary (language / Clean / − Name / ~Dup).
            status = ttk.Label(track_row, foreground="#868e96", anchor="e")
            status.pack(side="right", padx=(6, 0))
            row.track_status_labels[stream.index] = status

            canvas = tk.Canvas(
                track_row,
                height=WAVEFORM_HEIGHT,
                bg=self.canvas_bg,
                highlightthickness=1,
                highlightbackground=self.canvas_bg,
                highlightcolor=WAVEFORM_PLAYED,
                takefocus=1,
            )
            canvas.pack(side="left", fill="x", expand=True)
            canvas._envelope = self._representative_envelope(job, stream)
            canvas.bind("<Configure>", lambda _e, c=canvas: self._draw_envelope(c))
            canvas.bind("<Button-1>", lambda e, s=stream: self._on_waveform_press(job, s, e))
            canvas.bind("<B1-Motion>", lambda e, s=stream: self._on_waveform_drag(job, s, e))
            canvas.bind("<ButtonRelease-1>", lambda e, s=stream: self._on_waveform_release(job, s, e))
            canvas.bind("<Left>", lambda _e, s=stream: self._nudge_playhead(job, s, -5.0))
            canvas.bind("<Right>", lambda _e, s=stream: self._nudge_playhead(job, s, 5.0))
            canvas.bind("<space>", lambda _e, s=stream: (self._on_track_play(job, s), "break")[1])
            row.track_canvases[stream.index] = canvas
            self._set_canvas_enabled(canvas, job.status == "ready" and TrackPlayer.available())
            self._redraw_track(job, stream.index)

            if (job.id, stream.index) in self._track_details_open:
                self._build_track_detail(job, stream)

            # "Why" line under the row (reason + an Include/Undo link), populated
            # by _update_track_status. Packed after this track_row so it sits
            # directly beneath it.
            reason = ttk.Frame(row.tracks_container)
            reason.pack(fill="x", padx=(34, 0))
            self._bind_mousewheel(reason)
            row.track_reason_frames[stream.index] = reason

            self._update_track_status(job, stream)

    # -- per-track manual controls (name / language / noise / subtraction) ----
    def _noise_default(self) -> bool:
        return bool(self.settings.get("noise_filter_default", True))

    def _track_default_name(self, job: Job, stream: AudioStream) -> str:
        tracks = job.tracks or [stream]
        names = track_speaker_names(tracks)
        return dict(zip((s.index for s in tracks), names)).get(stream.index, f"Speaker {stream.index}")

    def _current_track_name(self, job: Job, stream: AudioStream) -> str:
        override = (job.track_names or {}).get(stream.index)
        return override if override else self._track_default_name(job, stream)

    def _track_name_by_index(self, job: Job, index: int) -> str:
        stream = next((s for s in (job.tracks or []) if s.index == index), None)
        return self._current_track_name(job, stream) if stream else f"Speaker {index}"

    def _recent_names(self) -> list[str]:
        """Recently-used speaker names, most-recent first, for the name dropdown."""
        names = self.settings.get("recent_names", [])
        return names if isinstance(names, list) else []

    def _push_recent_name(self, name: str) -> None:
        """Remember a user-typed name so it's suggested on the next call."""
        default = name.startswith("Speaker ") and name[8:].isdigit()
        if not name or default:
            return
        recent = [n for n in self._recent_names() if n != name]
        recent.insert(0, name)
        self.settings["recent_names"] = recent[:12]
        _save_settings(self.settings)

    def _commit_track_name(self, job: Job, stream: AudioStream, var: tk.StringVar) -> None:
        if job.status != "ready":
            return
        name = var.get().strip()
        if not name:
            name = self._track_default_name(job, stream)
            var.set(name)
        if job.track_names is None:
            job.track_names = {}
        if job.track_names.get(stream.index) == name:
            return
        job.track_names[stream.index] = name
        self._push_recent_name(name)
        # Other rows reference this name (subtraction summaries); refresh all.
        for s in job.tracks or []:
            self._update_track_status(job, s)

    def _toggle_track_detail(self, job: Job, stream_index: int) -> None:
        key = (job.id, stream_index)
        if key in self._track_details_open:
            self._track_details_open.discard(key)
        else:
            self._track_details_open.add(key)
        self._render_tracks_section(job)

    def _build_track_detail(self, job: Job, stream: AudioStream) -> None:
        row = self.rows.get(job.id)
        if row is None:
            return
        editable = job.status == "ready"
        state = "normal" if editable else "disabled"
        readonly = "readonly" if editable else "disabled"

        detail = ttk.Frame(row.tracks_container)
        detail.pack(fill="x", padx=(34, 0), pady=(1, 5))
        self._bind_mousewheel(detail)
        row.track_detail_frames[stream.index] = detail

        # Language (first choice = inherit the file's global language).
        ttk.Label(detail, text="Language").pack(side="left")
        override = (job.track_languages or {}).get(stream.index)
        has_override = bool(job.track_languages) and stream.index in job.track_languages
        lang_label = LANGUAGE_VALUE_TO_LABEL.get(override, "Auto-detect") if has_override else "Same as file"
        lang_var = tk.StringVar(value=lang_label)
        lang_combo = ttk.Combobox(
            detail,
            textvariable=lang_var,
            values=["Same as file"] + [label for label, _v in LANGUAGE_CHOICES],
            state=readonly,
            width=14,
        )
        lang_combo.pack(side="left", padx=(4, 12))
        lang_combo.bind(
            "<<ComboboxSelected>>",
            lambda _e, v=lang_var: self._on_track_language_changed(job, stream, v),
        )

        # Noise filter.
        noise_on = (job.track_denoise or {}).get(stream.index, self._noise_default())
        noise_var = tk.BooleanVar(value=noise_on)
        noise_check = ttk.Checkbutton(
            detail,
            text="Reduce background noise",
            variable=noise_var,
            command=lambda v=noise_var: self._on_track_denoise_changed(job, stream, v),
        )
        noise_check.configure(state=state)
        noise_check.pack(side="left", padx=(0, 12))

        # Remove-a-voice (subtraction). Only meaningful with another track.
        others = [s for s in (job.tracks or []) if s.index != stream.index]
        if others:
            ttk.Label(detail, text="Remove voice").pack(side="left")
            refs = (job.track_subtractions or {}).get(stream.index, [])
            current = "Keep both voices"
            if refs:
                current = self._track_name_by_index(job, refs[0])
            subtract_var = tk.StringVar(value=current)
            subtract_combo = ttk.Combobox(
                detail,
                textvariable=subtract_var,
                values=["Keep both voices"] + [self._current_track_name(job, s) for s in others],
                state=readonly,
                width=16,
            )
            subtract_combo.pack(side="left", padx=(4, 12))
            subtract_combo.bind(
                "<<ComboboxSelected>>",
                lambda _e, v=subtract_var: self._on_track_subtract_changed(job, stream, others, v),
            )

        # Channel picker (only when a stream has >1 non-identical channel).
        identical = (job.channel_pairs_identical or {}).get(stream.index, False)
        if stream.channels > 1 and not identical:
            channel_var = tk.StringVar(value=self._channel_mode_label(job, stream))
            channel_combo = ttk.Combobox(
                detail,
                textvariable=channel_var,
                values=["Mix"] + [f"Ch {i + 1}" for i in range(stream.channels)],
                state=readonly,
                width=6,
            )
            channel_combo.pack(side="left")
            channel_combo.bind(
                "<<ComboboxSelected>>",
                lambda _e, v=channel_var: self._on_channel_mode_changed(job, stream.index, v),
            )
            row.channel_combos[stream.index] = channel_combo

    def _on_track_language_changed(self, job: Job, stream: AudioStream, var: tk.StringVar) -> None:
        if job.track_languages is None:
            job.track_languages = {}
        label = var.get()
        if label == "Same as file":
            job.track_languages.pop(stream.index, None)
        else:
            job.track_languages[stream.index] = LANGUAGE_LABEL_TO_VALUE.get(label)
        self._update_track_status(job, stream)

    def _on_track_denoise_changed(self, job: Job, stream: AudioStream, var: tk.BooleanVar) -> None:
        if job.track_denoise is None:
            job.track_denoise = {}
        job.track_denoise[stream.index] = bool(var.get())
        self._update_track_status(job, stream)

    def _on_track_subtract_changed(
        self, job: Job, stream: AudioStream, others: list[AudioStream], var: tk.StringVar
    ) -> None:
        if job.track_subtractions is None:
            job.track_subtractions = {}
        label = var.get()
        if label == "Keep both voices":
            job.track_subtractions.pop(stream.index, None)
        else:
            ref = next((s for s in others if self._current_track_name(job, s) == label), None)
            if ref is not None:
                job.track_subtractions[stream.index] = [ref.index]
        self._update_track_status(job, stream)
        self._refresh_shared_voice(job)

    # -- per-track state model (people-first) ---------------------------------
    # One enum drives the row's chip, reason line, dimming, and the header count.
    # States: "clean" (its own speaker), "isolated" (another voice removed),
    # "conflict" (still shares a voice with another INCLUDED track - the only
    # amber state), "skipped_dup" (a duplicate copy, not transcribed),
    # "excluded" (the user turned it off).
    def _track_state(self, job: Job, stream: AudioStream) -> str:
        idx = stream.index
        selected = job.selected_track_indices or {s.index for s in (job.tracks or [])}
        if idx not in selected:
            return "skipped_dup" if self._skip_twin_name(job, stream) else "excluded"
        if self._shared_voice_partner_name(job, stream):
            return "conflict"
        if (job.track_subtractions or {}).get(idx):
            return "isolated"
        return "clean"

    def _skip_twin_name(self, job: Job, stream: AudioStream) -> str | None:
        """Name of the included track this (skipped) one duplicates, if any -
        the reason it was skipped. From exact-copy detection first, then a
        shared-voice 'duplicate' pairing."""
        idx = stream.index
        dup = (job.duplicate_of or {}).get(idx)
        if dup is not None:
            return self._track_name_by_index(job, dup)
        selected = job.selected_track_indices or set()
        for w in (job.shared_voice_warnings or []):
            if w.kind != "duplicate":
                continue
            other = w.component_index if idx == w.mix_index else (w.mix_index if idx == w.component_index else None)
            if other is not None and other in selected:
                return self._track_name_by_index(job, other)
        return None

    def _shared_voice_partner_name(self, job: Job, stream: AudioStream) -> str | None:
        """If this track still shares a voice with another INCLUDED track (the
        voice would be written twice), the bleeding voice's name; else None."""
        for w in self._unhandled_shared_voice(job):
            if stream.index == w.mix_index:
                return self._track_name_by_index(job, w.component_index)
            if stream.index == w.component_index:
                return self._track_name_by_index(job, w.mix_index)
        return None

    # (text, color) chip shown at the right of a track row. "clean" shows nothing
    # - the default needs no badge; chips mark only the exceptions.
    def _track_chip(self, state: str) -> tuple[str, str]:
        return {
            "clean": ("", MUTED_FG),
            "isolated": ("◐ Cleaned up", ACCENT_FG),
            "skipped_dup": ("⊘ Skipped", MUTED_FG),
            "excluded": ("○ Off", MUTED_FG),
            "conflict": ("⚠ Same voice", WARN_FG),
        }.get(state, ("", MUTED_FG))

    def _track_reason(self, job: Job, stream: AudioStream, state: str) -> tuple[str, str]:
        """(reason sentence, color) for the line under a row. Empty for a plain
        clean track - nothing needs explaining."""
        name = self._current_track_name(job, stream)
        if state == "isolated":
            refs = (job.track_subtractions or {}).get(stream.index, [])
            removed = " and ".join(self._track_name_by_index(job, r) for r in refs)
            return (f"{removed}'s voice removed, so this is only {name}.", MUTED_FG)
        if state == "skipped_dup":
            twin = self._skip_twin_name(job, stream) or "another track"
            return (f"Same audio as {twin} — already covered, so it's skipped.", MUTED_FG)
        if state == "excluded":
            return ("You turned this track off.", MUTED_FG)
        if state == "conflict":
            partner = self._shared_voice_partner_name(job, stream)
            return (f"{partner}'s voice is also here — {partner} would be written twice.", WARN_FG)
        return ("", MUTED_FG)

    def _update_track_status(self, job: Job, stream: AudioStream) -> None:
        """Refresh a row's chip, reason line, and dimming from its state."""
        row = self.rows.get(job.id)
        if row is None:
            return
        state = self._track_state(job, stream)
        label = row.track_status_labels.get(stream.index)
        if label is not None:
            text, color = self._track_chip(state)
            label.configure(text=text, foreground=color)
        # A skipped/excluded track isn't a speaker, so disable its name field
        # (greys it natively) - you can't name something that won't be written.
        entry = row.track_name_entries.get(stream.index)
        if entry is not None:
            editable = job.status == "ready"
            dimmed = state in ("skipped_dup", "excluded")
            try:
                entry.configure(state="disabled" if (dimmed or not editable) else "normal")
            except tk.TclError:
                pass
        self._render_track_reason(job, stream, state)

    def _render_track_reason(self, job: Job, stream: AudioStream, state: str) -> None:
        row = self.rows.get(job.id)
        if row is None:
            return
        holder = row.track_reason_frames.get(stream.index)
        if holder is None:
            return
        for child in holder.winfo_children():
            child.destroy()
        text, color = self._track_reason(job, stream, state)
        if not text:
            return
        ttk.Label(holder, text=text, foreground=color, font=("", 11),
                  wraplength=430, justify="left").pack(side="left")
        editable = job.status == "ready"
        link = None
        if state in ("skipped_dup", "excluded"):
            link = ("Include anyway", lambda: self._set_track_included(job, stream.index, True))
        elif state == "isolated":
            link = ("Undo", lambda: self._clear_track_subtraction(job, stream))
        if link and editable:
            lbl, cmd = link
            action = tk.Label(holder, text=lbl, fg=ACCENT_FG, cursor="hand2", font=("", 11))
            action.pack(side="left", padx=(6, 0))
            action.bind("<Button-1>", lambda _e, c=cmd: c())

    def _update_tracks_header(self, job: Job) -> None:
        row = self.rows.get(job.id)
        if row is None or row.tracks_header is None:
            return
        row.tracks_header.configure(text=self._people_summary_text(job))

    def _people_summary_text(self, job: Job) -> str:
        """People-first header: how many speakers you'll get, not a track count.
        Directly answers 'it only kept one track' by stating the result up front."""
        tracks = job.tracks or []
        selected = job.selected_track_indices or set()
        kept = sum(1 for s in tracks if s.index in selected)
        skipped = len(tracks) - kept
        if kept == 0:
            return "No speakers selected"
        who = "speaker" if kept == 1 else "speakers"
        text = f"{kept} {who} will be transcribed"
        if skipped:
            t = "track" if skipped == 1 else "tracks"
            text += f" · {skipped} {t} skipped"
        return text

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
        # A preview of this track is now playing the wrong channel - stop it
        # rather than trying to hot-swap the audio mid-play. The playhead stays,
        # so ▶ resumes at the same spot on the newly-chosen channel.
        if self._playing == (job.id, stream_index):
            self._stop_playback()
        self._redraw_track(job, stream_index)

    # -- track preview playback -----------------------------------------------

    def _audio_index(self, job: Job, stream: AudioStream) -> int:
        """ffplay's `-ast a:N` index - this stream's position among the file's
        audio streams, which is exactly its position in job.tracks."""
        for i, s in enumerate(job.tracks or []):
            if s.index == stream.index:
                return i
        return 0

    def _job_duration(self, job: Job) -> float:
        """Recording length in seconds, probed lazily if not known yet."""
        if job.audio_duration_seconds is None:
            try:
                job.audio_duration_seconds = probe_duration_seconds(job.source_path)
            except Exception:
                job.audio_duration_seconds = 0.0
        return job.audio_duration_seconds or 0.0

    def _on_track_play(self, job: Job, stream: AudioStream) -> None:
        """Play/stop button: stop if this track is playing, else (re)start it
        from its current playhead (so it acts like resume)."""
        key = (job.id, stream.index)
        if self._playing == key:
            self._stop_playback()  # playhead is left in place so ▶ resumes
            return
        self._start_playback(job, stream, self._playhead.get(key, 0.0))

    def _start_playback(self, job: Job, stream: AudioStream, fraction: float) -> None:
        if job.status != "ready" or not TrackPlayer.available():
            return
        # Only one track is audible at a time, but each track keeps its own
        # playhead, so you can jump between tracks and resume each where you
        # left off.
        self._stop_playback()
        key = (job.id, stream.index)
        fraction = max(0.0, min(0.999, fraction))
        self._playhead[key] = fraction
        self._playing = key
        duration = self._job_duration(job)
        self._player.play(
            job.source_path,
            audio_index=self._audio_index(job, stream),
            channel_index=(job.selected_channel_by_track or {}).get(stream.index),
            start_seconds=fraction * duration,
            on_finish=lambda: self.root.after(
                0, lambda: self._on_playback_finished(job.id, stream.index)
            ),
        )
        self._set_play_button(job.id, stream.index, ICON_STOP, enabled=True)
        self._redraw_track(job, stream.index)
        # Listen -> label: focus this track's name field so you can type who it
        # is the moment you recognize the voice.
        row = self.rows.get(job.id)
        name_entry = row.track_name_entries.get(stream.index) if row else None
        if name_entry is not None and str(name_entry.cget("state")) != "disabled":
            try:
                name_entry.focus_set()
            except tk.TclError:
                pass

    def _on_playback_finished(self, job_id: int, stream_index: int) -> None:
        if self._playing != (job_id, stream_index):
            return  # already superseded/stopped
        self._playing = None
        self._playhead.pop((job_id, stream_index), None)  # played to the end
        self._set_play_button(job_id, stream_index, ICON_START, enabled=True)
        self._redraw_track_by_id(job_id, stream_index)

    def _stop_playback(self) -> None:
        """Stop audio but leave the playhead where it is (so ▶ resumes)."""
        if self._playing is None:
            return
        job_id, stream_index = self._playing
        self._playing = None
        self._player.stop()
        self._set_play_button(job_id, stream_index, ICON_START, enabled=True)
        self._redraw_track_by_id(job_id, stream_index)

    def _clear_job_playheads(self, job_id: int) -> None:
        for key in [k for k in self._playhead if k[0] == job_id]:
            self._playhead.pop(key, None)
            self._redraw_track_by_id(*key)

    # -- waveform seeking / scrubbing -----------------------------------------

    def _fraction_at(self, canvas: tk.Canvas, x: int) -> float:
        width = canvas.winfo_width()
        if width <= 1:
            return 0.0
        return max(0.0, min(0.999, x / width))

    def _on_waveform_press(self, job: Job, stream: AudioStream, event) -> None:
        if job.status != "ready" or not TrackPlayer.available():
            return
        event.widget.focus_set()
        # Silence any current audio while scrubbing; playback (re)starts on
        # release, so dragging doesn't thrash ffplay with a restart per pixel.
        self._player.stop()
        self._playing = None
        self._scrub_key = (job.id, stream.index)
        self._scrub_fraction = self._fraction_at(event.widget, event.x)
        self._playhead[self._scrub_key] = self._scrub_fraction
        self._set_play_button(job.id, stream.index, ICON_START, enabled=True)
        self._redraw_track(job, stream.index)

    def _on_waveform_drag(self, job: Job, stream: AudioStream, event) -> None:
        if self._scrub_key != (job.id, stream.index):
            return
        self._scrub_fraction = self._fraction_at(event.widget, event.x)
        self._playhead[self._scrub_key] = self._scrub_fraction
        self._redraw_track(job, stream.index)

    def _on_waveform_release(self, job: Job, stream: AudioStream, event) -> None:
        if self._scrub_key != (job.id, stream.index):
            return
        fraction = self._scrub_fraction
        self._scrub_key = None
        self._start_playback(job, stream, fraction)

    def _nudge_playhead(self, job: Job, stream: AudioStream, delta_seconds: float) -> str:
        # Keyboard seek (Left/Right when the waveform is focused).
        if job.status != "ready" or not TrackPlayer.available():
            return "break"
        key = (job.id, stream.index)
        duration = self._job_duration(job) or 1.0
        current = self._playhead.get(key, 0.0) * duration
        self._start_playback(job, stream, (current + delta_seconds) / duration)
        return "break"

    def _set_play_button(self, job_id: int, stream_index: int, icon: str, enabled: bool) -> None:
        row = self.rows.get(job_id)
        if row is None:
            return
        button = row.track_play_buttons.get(stream_index)
        if button is None:
            return
        try:
            if not button.winfo_exists():
                return
            job = self.jobs.get(job_id)
            ready = job is not None and job.status == "ready"
            button.configure(text=icon, state="normal" if (enabled and ready) else "disabled")
        except tk.TclError:
            pass

    def _set_canvas_enabled(self, canvas: tk.Canvas, enabled: bool) -> None:
        try:
            if not canvas.winfo_exists():
                return
            canvas.configure(takefocus=1 if enabled else 0, cursor="hand2" if enabled else "")
        except tk.TclError:
            pass

    def _redraw_track_by_id(self, job_id: int, stream_index: int) -> None:
        job = self.jobs.get(job_id)
        if job is not None:
            self._redraw_track(job, stream_index)

    def _playhead_tick(self) -> None:
        """While a track plays, advance its playhead from the real ffplay
        position and redraw. Runs continuously; near-free when idle."""
        if self._playing is not None and self._scrub_key is None:
            job_id, stream_index = self._playing
            job = self.jobs.get(job_id)
            position = self._player.position()
            if job is not None and position is not None and (job.audio_duration_seconds or 0) > 0:
                self._playhead[(job_id, stream_index)] = max(
                    0.0, min(1.0, position / job.audio_duration_seconds)
                )
                self._redraw_track_by_id(job_id, stream_index)
        self.root.after(100, self._playhead_tick)

    def shutdown(self) -> None:
        """Stop playback - called when the window closes."""
        self._player.stop()
        self._playing = None

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
            key = (job.id, stream_index)
            canvas._envelope = self._representative_envelope(job, stream)
            canvas._playhead = self._playhead.get(key)
            canvas._duration = job.audio_duration_seconds or 0.0
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

        # Note exact-duplicate tracks (near-identical loudness-over-time) for the
        # "duplicate of X" reason text. This is NOT used to unselect tracks: when
        # one loud voice dominates every track their envelopes all match, which
        # would wrongly collapse a real 2-speaker call to 1. The authoritative
        # keep/isolate/drop decision is the waveform-based shared-voice plan
        # (_analyze_shared_voice_async → auto-apply), which correctly drops only
        # the true duplicate and keeps each distinct speaker.
        job.duplicate_of = {}
        for i, stream in enumerate(job.tracks):
            for earlier in job.tracks[:i]:
                score = correlation(representative_envelope[stream.index], representative_envelope[earlier.index])
                if score >= 0.995:
                    job.duplicate_of[stream.index] = earlier.index
                    break

        self._recompute_estimate(job)
        self._render_tracks_section(job)
        # Once tracks are known, check (in the background) whether any carry the
        # same voice - the mic-bleed "doubling" safety net.
        if len(job.tracks) > 1 and not job.shared_voice_analyzed:
            self._analyze_shared_voice_async(job)

    # --- shared-voice safety net ---------------------------------------------
    # Representative window to analyze: tracks share a voice structurally, so a
    # mid-call segment with both speakers active is enough (and cheap vs.
    # decoding whole 90-minute tracks).
    _SHARED_VOICE_SEGMENT_S = 90.0

    def _compute_shared_voice(self, source: Path, tracks: list, duration: float | None) -> list:
        """Decode a representative segment of each track and detect shared voices.
        Pure of UI state so it runs identically from the async panel analysis and
        the synchronous Start-time check. Returns [] on any failure."""
        try:
            seg = self._SHARED_VOICE_SEGMENT_S
            start = max(0.0, (duration * 0.35) if duration else 0.0)
            if duration and start + seg > duration:
                start = max(0.0, duration - seg)
            signals = []
            for stream in tracks:
                pcm = track_separation.decode_track_pcm(
                    source, stream.index, start_s=start, duration_s=seg
                )
                signals.append(track_separation.TrackSignal(
                    index=stream.index, name="", samples=pcm
                ))
            return track_separation.detect_shared_voice(signals)
        except Exception:
            return []

    def _analyze_shared_voice_async(self, job: Job) -> None:
        job.shared_voice_analyzed = True  # guard: run once per job
        tracks = list(job.tracks or [])
        duration = job.audio_duration_seconds
        source = job.source_path

        def analyze() -> None:
            warnings = self._compute_shared_voice(source, tracks, duration)
            self.root.after(0, lambda: self._on_shared_voice_analyzed(job, warnings))

        threading.Thread(target=analyze, daemon=True).start()

    def _on_shared_voice_analyzed(self, job: Job, warnings: list) -> None:
        job.shared_voice_warnings = warnings
        # App auto-picks the speakers by default: if a voice is on more than one
        # track, apply the keep/isolate/skip plan now so the panel opens already
        # set up ("you'll get 2 speakers"). The user can still revert to every
        # track, or adjust per row. We don't re-apply once the user has fixed or
        # explicitly reverted.
        if (
            warnings
            and not job.shared_voice_fixed
            and not job.shared_voice_dismissed
            and self._unhandled_shared_voice(job)
        ):
            self._apply_shared_voice_fix(job, auto=True)
        else:
            self._render_tracks_section(job)

    def _unhandled_shared_voice(self, job: Job):
        """Shared-voice warnings that still apply given the current selection and
        subtractions - i.e. would still double a speaker if transcribed now.

        Not suppressed by `shared_voice_dismissed`: if the user reverted to every
        track, the doubling is real and must be shown (amber). `dismissed` only
        stops detection from silently re-applying the plan."""
        if not job.shared_voice_warnings:
            return []
        selected = job.selected_track_indices
        if selected is None:
            selected = {s.index for s in (job.tracks or [])}
        subs = job.track_subtractions or {}
        unhandled = []
        for w in job.shared_voice_warnings:
            if w.mix_index not in selected or w.component_index not in selected:
                continue  # one side not being transcribed -> no doubling
            if w.kind == "contains" and w.component_index in subs.get(w.mix_index, []):
                continue  # already isolating the mix minus the component
            unhandled.append(w)
        return unhandled

    def _shared_voice_fix_plan(self, job: Job):
        """(subtractions, drop) the one-click fix would apply, given selection."""
        selected = sorted(
            job.selected_track_indices
            if job.selected_track_indices is not None
            else {s.index for s in (job.tracks or [])}
        )
        return track_separation.resolve_shared_voice(selected, job.shared_voice_warnings or [])

    def _plan_speaker_summary(self, job: Job) -> str:
        """"You'll get 2 speakers: Client · Salesperson" - the result, by name."""
        selected = job.selected_track_indices or {s.index for s in (job.tracks or [])}
        names = [self._current_track_name(job, s) for s in (job.tracks or []) if s.index in selected]
        if not names:
            return ""
        who = "speaker" if len(names) == 1 else "speakers"
        return f"You'll get {len(names)} {who}:  " + " · ".join(names)

    def _doubling_title(self, job: Job, warnings) -> str:
        """Amber headline. Names the bled voice only when it's a single, clear
        'contains' case; otherwise stays generic rather than listing a confusing
        pile of track names."""
        bled = sorted({w.component_index for w in warnings if w.kind == "contains"})
        if len(bled) == 1:
            return f'⚠ "{self._track_name_by_index(job, bled[0])}" would be written twice'
        return "⚠ The same voice would be written twice"

    def _refresh_shared_voice(self, job: Job) -> None:
        """Cheap re-evaluation (no detector re-run) after the user changes the
        selection or a subtraction: rebuild the banner and refresh the markers."""
        self._render_shared_voice_banner(job)
        for stream in (job.tracks or []):
            self._update_track_status(job, stream)

    def _render_shared_voice_banner(self, job: Job) -> None:
        """The plan card above the track rows. Three cases:
          - handled (the app set up each person once): calm GREEN confirmation.
          - a voice would still be written twice (user reverted / left it on):
            AMBER warning with a one-click way back to the handled plan.
          - nothing shared: no card at all.
        """
        row = self.rows.get(job.id)
        if row is None or row.shared_voice_holder is None:
            return
        container = row.shared_voice_holder
        for child in container.winfo_children():
            child.destroy()

        editable = job.status == "ready"
        unhandled = self._unhandled_shared_voice(job)

        if not unhandled and job.shared_voice_fixed:
            frame = ttk.Frame(container)
            frame.pack(fill="x", pady=(0, 6))
            ttk.Label(
                frame, text="✓ Set up to transcribe each person once",
                foreground=OK_FG, font=("", 11, "bold"), wraplength=460, justify="left",
            ).pack(anchor="w")
            summary = self._plan_speaker_summary(job)
            if summary:
                ttk.Label(frame, text=summary, foreground="#555",
                          wraplength=460, justify="left").pack(anchor="w", pady=(1, 2))
            if editable:
                link = tk.Label(frame, text="Transcribe every track instead",
                                fg=ACCENT_FG, cursor="hand2", font=("", 11))
                link.pack(anchor="w")
                link.bind("<Button-1>", lambda _e: self._revert_to_all_tracks(job))
            return

        if unhandled:
            frame = ttk.Frame(container)
            frame.pack(fill="x", pady=(0, 6))
            ttk.Label(
                frame,
                text=self._doubling_title(job, unhandled),
                foreground=WARN_FG, font=("", 11, "bold"), wraplength=460, justify="left",
            ).pack(anchor="w")
            ttk.Label(
                frame,
                text="The same voice is on more than one included track.",
                foreground="#555", wraplength=460, justify="left",
            ).pack(anchor="w", pady=(1, 2))
            if editable:
                link = tk.Label(frame, text="Set up to transcribe each person once",
                                fg=ACCENT_FG, cursor="hand2", font=("", 11))
                link.pack(anchor="w")
                link.bind("<Button-1>", lambda _e: self._reapply_shared_voice_fix(job))

    def _revert_to_all_tracks(self, job: Job) -> None:
        """User chose to transcribe every track (undo the auto plan). Marked so
        detection won't silently re-apply it."""
        self._undo_shared_voice_fix(job)
        job.shared_voice_dismissed = True
        self._render_tracks_section(job)

    def _reapply_shared_voice_fix(self, job: Job) -> None:
        """Go back to the handled plan after the user had reverted."""
        job.shared_voice_dismissed = False
        self._apply_shared_voice_fix(job)

    def _apply_shared_voice_fix(self, job: Job, auto: bool = False) -> None:
        """Apply the suggested subtractions and drop redundant tracks, then
        re-render. Snapshotted so it's one-click reversible, and also adjustable
        via the per-track controls it leaves in place. `auto=True` when applied
        automatically on detection (vs. a user click) - behaviour is identical."""
        if job.selected_track_indices is None:
            job.selected_track_indices = {s.index for s in (job.tracks or [])}
        # Snapshot for Undo (before mutating).
        self._shared_voice_undo[job.id] = (
            set(job.selected_track_indices),
            {k: list(v) for k, v in (job.track_subtractions or {}).items()},
        )
        subs, drop = self._shared_voice_fix_plan(job)
        if job.track_subtractions is None:
            job.track_subtractions = {}
        for mix, refs in subs.items():
            existing = set(job.track_subtractions.get(mix, []))
            job.track_subtractions[mix] = sorted(existing | set(refs))
        job.selected_track_indices -= drop
        job.shared_voice_fixed = True
        self._recompute_estimate(job)
        self._render_tracks_section(job)

    def _undo_shared_voice_fix(self, job: Job) -> None:
        snap = self._shared_voice_undo.pop(job.id, None)
        if snap is not None:
            selected, subtractions = snap
            job.selected_track_indices = set(selected)
            job.track_subtractions = {k: list(v) for k, v in subtractions.items()}
        job.shared_voice_fixed = False
        self._recompute_estimate(job)
        self._render_tracks_section(job)


    def _draw_envelope(self, canvas: tk.Canvas) -> None:
        envelope = getattr(canvas, "_envelope", None)
        playhead = getattr(canvas, "_playhead", None)
        duration = getattr(canvas, "_duration", 0.0) or 0.0
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
            canvas.create_line(0, mid, width, mid, fill=WAVEFORM_UNPLAYED)
        else:
            n = len(envelope)
            step = width / (n - 1)
            points: list[float] = []
            for i, level in enumerate(envelope):
                points.append(i * step)
                points.append(mid - level * (mid - 1))
            # Whole waveform muted first...
            canvas.create_line(*points, fill=WAVEFORM_UNPLAYED, width=1.5, smooth=True)
            # ...then redraw the already-played part (left of the playhead) in
            # the accent color so progress through the track reads at a glance.
            if playhead is not None and playhead > 0:
                play_x = playhead * width
                played: list[float] = []
                for i, level in enumerate(envelope):
                    x = i * step
                    if x > play_x:
                        break
                    played.append(x)
                    played.append(mid - level * (mid - 1))
                if len(played) >= 4:
                    canvas.create_line(*played, fill=WAVEFORM_PLAYED, width=1.5, smooth=True)

        # Playhead line + a small cap at the top that reads as a grabbable handle.
        if playhead is not None:
            x = max(1, min(width - 1, playhead * width))
            canvas.create_line(x, 0, x, height, fill=WAVEFORM_PLAYHEAD, width=1)
            canvas.create_oval(x - 3, 0, x + 3, 6, fill=WAVEFORM_PLAYHEAD, outline=WAVEFORM_PLAYHEAD)

        # Position / duration, tucked into the bottom-right corner.
        if duration > 0:
            if playhead is not None:
                label = f"{format_timecode(playhead * duration)} / {format_timecode(duration)}"
            else:
                label = format_timecode(duration)
            canvas.create_text(
                width - 3, height - 1, text=label, anchor="se",
                fill=WAVEFORM_TIME, font=("TkDefaultFont", 8),
            )

    def _set_track_included(self, job: Job, stream_index: int, included: bool) -> None:
        """Include/exclude a track and re-render (used by the inline reason links)."""
        if job.status != "ready":
            return
        if job.selected_track_indices is None:
            job.selected_track_indices = {s.index for s in job.tracks or []}
        if included:
            job.selected_track_indices.add(stream_index)
        else:
            if len(job.selected_track_indices) <= 1:
                return
            job.selected_track_indices.discard(stream_index)
        self._recompute_estimate(job)
        self._render_tracks_section(job)

    def _clear_track_subtraction(self, job: Job, stream: AudioStream) -> None:
        """Undo an auto-applied isolation on one track (stop removing a voice)."""
        if job.status != "ready":
            return
        if job.track_subtractions:
            job.track_subtractions.pop(stream.index, None)
        self._recompute_estimate(job)
        self._render_tracks_section(job)

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
        self._update_tracks_header(job)
        self._recompute_estimate(job)
        self._refresh_shared_voice(job)

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
        job.estimated_polish_seconds = estimate_polish_seconds(
            job.audio_duration_seconds, job.polish
        )
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

    def _on_row_polish_changed(self, job_id: int) -> None:
        job = self.jobs.get(job_id)
        row = self.rows.get(job_id)
        if job is None or row is None:
            return
        job.polish = bool(row.polish_var.get())
        self._persist_polish_selection(job.polish)

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
        # The shared-voice setup needs the tracks analyzed. If the user never
        # opened the track panel, do it now (in the background) and resume Start
        # when it's ready - so auto-separation fires even without expanding.
        if job.shared_voice_warnings is None and (job.tracks is None or len(job.tracks) > 1):
            self._analyze_then_start(job)
            return
        # No modal: the plan is auto-applied and the inline card already explains
        # it (and warns, in amber, if the user reverted to every track). The
        # worker also auto-separates + de-duplicates as a backstop, so Start just
        # runs.
        job.status = "queued"
        self._refresh_row(job)
        self.job_queue.put(job)

    def _analyze_then_start(self, job: Job) -> None:
        """Probe tracks (if needed) + run shared-voice detection, then re-enter
        Start. Keeps the UI responsive with a brief 'checking tracks' status."""
        job.status_detail = "checking tracks…"
        self._refresh_row(job)
        source = job.source_path
        known = list(job.tracks) if job.tracks is not None else None
        duration = job.audio_duration_seconds

        def work() -> None:
            streams = known if known is not None else safe_probe_audio_streams(source)
            warnings = (
                self._compute_shared_voice(source, streams, duration)
                if len(streams) > 1 else []
            )
            self.root.after(0, lambda: finish(streams, warnings))

        def finish(streams, warnings) -> None:
            if job.tracks is None:
                job.tracks = streams
            if job.selected_track_indices is None:
                job.selected_track_indices = {s.index for s in streams}
            job.shared_voice_warnings = warnings
            job.shared_voice_analyzed = True
            job.status_detail = None
            # Auto-apply the plan (same as when the panel is open) so starting
            # without expanding still transcribes each person once.
            if (
                warnings
                and not job.shared_voice_fixed
                and not job.shared_voice_dismissed
                and self._unhandled_shared_voice(job)
            ):
                self._apply_shared_voice_fix(job, auto=True)
            self._start_job(job)  # re-enter: now analyzed

        threading.Thread(target=work, daemon=True).start()

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
        job.transcribe_fraction = None
        job.transcribe_position_fraction = None
        job.polish_started_at = None
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

    def _polish_share(self, job: Job) -> float:
        """Fraction of the overall bar reserved for the polish tail, from the
        transcribe/polish time estimates (clamped). 0 when polish is off."""
        if not job.polish:
            return 0.0
        transcribe = job.estimated_seconds or 0.0
        polish = job.estimated_polish_seconds or 0.0
        if transcribe + polish <= 0:
            return 0.0
        return max(POLISH_SHARE_MIN, min(POLISH_SHARE_MAX, polish / (transcribe + polish)))

    def _progress_view(self, job: Job, elapsed: float) -> dict:
        """Unify the two progress sources so the row and the top panel agree.

        `real` progress (a true fraction of the audio processed, reported by
        the engine) wins when present: the bar tracks it and a "12:30 / 41:05"
        timecode shows the real position, with a live ETA derived from actual
        throughput. Otherwise we fall back to the duration-based time estimate
        - and deliberately show NO timecode there, so a timecode on screen
        always means a true position, never a guess.

        Transcription fills only `[0, 1 - polish_share]` so the polish pass has
        room for its own moving tail, and the ETA folds in the polish estimate
        so "~X left" accounts for the whole job, not just transcription.
        """
        duration = job.audio_duration_seconds
        share = self._polish_share(job)
        polish_eta = job.estimated_polish_seconds or 0.0  # whole pass still ahead
        if job.transcribe_fraction is not None:
            raw = min(0.999, max(0.0, job.transcribe_fraction))
            fraction = raw * (1 - share)
            timecode = None
            if duration and job.transcribe_position_fraction is not None:
                position = max(0.0, min(1.0, job.transcribe_position_fraction)) * duration
                timecode = f"{format_timecode(position)} / {format_timecode(duration)}"
            transcribe_eta = (elapsed / raw - elapsed) if (raw > 0 and elapsed) else None
            eta = transcribe_eta + polish_eta if transcribe_eta is not None else None
            return {"fraction": fraction, "real": True, "timecode": timecode, "eta": eta}
        if job.estimated_seconds:
            fraction = min(0.99, (elapsed or 0.0) / job.estimated_seconds) * (1 - share)
            eta = max(0.0, job.estimated_seconds - (elapsed or 0.0)) + polish_eta
            return {"fraction": fraction, "real": False, "timecode": None, "eta": eta}
        return {"fraction": None, "real": False, "timecode": None, "eta": None}

    def _polish_view(self, job: Job) -> dict:
        """Progress for the polish phase: a time-animated tail of the same bar,
        picking up where transcription left off so it never resets to 0."""
        share = self._polish_share(job) or POLISH_SHARE_MIN
        est = job.estimated_polish_seconds or POLISH_BASE_SECONDS
        elapsed = (
            (datetime.now() - job.polish_started_at).total_seconds()
            if job.polish_started_at
            else 0.0
        )
        progress = min(0.99, elapsed / est) if est > 0 else 0.99
        return {"fraction": (1 - share) + share * progress, "eta": max(0.0, est - elapsed)}

    def _transcribe_detail_text(self, job: Job, view: dict) -> str:
        """Compact per-row label. Timecode is the hero when progress is real;
        the fallback is clearly marked as an estimate."""
        eta = format_minutes_left(view["eta"])
        if view["real"]:
            parts: list[str] = []
            if job.status_detail:  # multi-track: "track 2/4 - Bob"
                parts.append(job.status_detail)
            if view["timecode"]:
                parts.append(view["timecode"])
            if eta:
                parts.append(f"~{eta} left")
            return "  ·  ".join(parts) or f"{round(view['fraction'] * 100)}%"
        return f"~ Estimated {eta} left" if eta else "Estimating…"

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
        # Previewing shares the same source file/CPU as transcription, so a
        # job leaving "ready" (Start clicked) stops any preview of it, clears
        # its playheads, and greys its play buttons - re-enabled when "ready".
        if job.status != "ready" and self._playing is not None and self._playing[0] == job.id:
            self._stop_playback()
        if job.status != "ready":
            self._clear_job_playheads(job.id)
        play_enabled = job.status == "ready" and TrackPlayer.available()
        for stream_index, play_button in row.track_play_buttons.items():
            is_playing = self._playing == (job.id, stream_index)
            play_button.configure(
                text=ICON_STOP if is_playing else ICON_START,
                state="normal" if play_enabled else "disabled",
            )
            canvas = row.track_canvases.get(stream_index)
            if canvas is not None:
                self._set_canvas_enabled(canvas, play_enabled)

        if job.status == "transcribing":
            view = self._progress_view(job, elapsed or 0.0)
            if view["fraction"] is None:
                row.progress_bar.configure(mode="indeterminate")
                row.progress_bar.start(15)
                row.detail_label.configure(text="Transcribing…")
            else:
                row.progress_bar.stop()
                row.progress_bar.configure(mode="determinate")
                row.progress_bar["value"] = view["fraction"] * 100
                row.detail_label.configure(text=self._transcribe_detail_text(job, view))
        elif job.status == "polishing":
            view = self._polish_view(job)
            row.progress_bar.stop()
            row.progress_bar.configure(mode="determinate")
            row.progress_bar["value"] = view["fraction"] * 100
            eta = format_minutes_left(view["eta"])
            row.detail_label.configure(
                text=f"Polishing…  ·  ~{eta} left" if eta else "Polishing…"
            )
        elif job.status == "loading model":
            row.progress_bar.configure(mode="indeterminate")
            row.progress_bar.start(15)
            row.detail_label.configure(text="Loading model…")
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
                total = job.estimated_seconds + (job.estimated_polish_seconds or 0.0)
                row.detail_label.configure(text=f"~{format_seconds(total)} estimated")
            else:
                row.detail_label.configure(text="estimating..." if job.audio_duration_seconds is None else "-")
        elif job.status == "queued":
            row.progress_bar.stop()
            row.progress_bar.configure(mode="determinate")
            row.progress_bar["value"] = 0
            if job.estimated_seconds:
                total = job.estimated_seconds + (job.estimated_polish_seconds or 0.0)
                row.detail_label.configure(text=f"waiting... (~{format_seconds(total)})")
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
                "Loading model into memory…  (a few seconds; the first use of a "
                "model downloads it once and can take a while)"
            )
            return

        if job.status == "transcribing":
            view = self._progress_view(job, elapsed)
            if view["fraction"] is None:
                self.progress_bar.configure(mode="indeterminate")
                self.progress_bar.start(15)
                self.progress_detail_var.set("Transcribing…  •  estimating progress")
                return
            self.progress_bar.stop()
            self.progress_bar.configure(mode="determinate")
            self.progress_bar["value"] = view["fraction"] * 100
            eta = format_minutes_left(view["eta"])
            if view["real"]:
                parts = []
                if view["timecode"]:
                    parts.append(view["timecode"])
                parts.append(f"{round(view['fraction'] * 100)}%")
                if eta:
                    parts.append(f"about {eta} left")
                self.progress_detail_var.set("  •  ".join(parts))
            else:
                detail = "Estimating progress"
                if eta:
                    detail += f"  •  about {eta} left"
                self.progress_detail_var.set(detail)
            return

        if job.status == "polishing":
            view = self._polish_view(job)
            self.progress_bar.stop()
            self.progress_bar.configure(mode="determinate")
            self.progress_bar["value"] = view["fraction"] * 100
            eta = format_minutes_left(view["eta"])
            detail = "Cleaning up the transcript with Claude"
            if eta:
                detail += f"  •  about {eta} left"
            self.progress_detail_var.set(detail)
            return

        self.progress_bar.stop()
        self.progress_bar.configure(mode="determinate")
        self.progress_bar["value"] = 0
        self.progress_detail_var.set(f"{format_seconds(elapsed)} elapsed")

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
