from pathlib import Path

from src import history_db
from src.catalog import DEFAULT_REAL_TIME_FACTOR


def test_record_and_fetch_history(tmp_path: Path) -> None:
    conn = history_db.init_db(tmp_path / "history.db")

    history_db.record_run(
        conn,
        source_path="/videos/call.mp4",
        output_path="/videos/call.md",
        model="mlx-community/whisper-small-mlx",
        language="en",
        audio_duration_seconds=100.0,
        elapsed_seconds=25.0,
        started_at="2026-07-31T10:00:00",
        finished_at="2026-07-31T10:00:25",
        status="success",
    )

    rows = history_db.fetch_history(conn)
    assert len(rows) == 1
    assert rows[0].source_path == "/videos/call.mp4"
    assert rows[0].status == "success"


def test_estimate_seconds_falls_back_to_default_ratio_without_history(tmp_path: Path) -> None:
    conn = history_db.init_db(tmp_path / "history.db")
    model = "mlx-community/whisper-small-mlx"

    estimate = history_db.estimate_seconds(conn, model, "en", 200.0)

    assert estimate == 200.0 * DEFAULT_REAL_TIME_FACTOR[model]


def test_estimate_seconds_improves_with_recorded_history(tmp_path: Path) -> None:
    conn = history_db.init_db(tmp_path / "history.db")
    model = "mlx-community/whisper-small-mlx"

    # Record a run with a real-time factor of 0.5, well above the default table.
    history_db.record_run(
        conn,
        source_path="/videos/a.mp4",
        output_path="/videos/a.md",
        model=model,
        language="en",
        audio_duration_seconds=100.0,
        elapsed_seconds=50.0,
        started_at="2026-07-31T10:00:00",
        finished_at="2026-07-31T10:00:50",
        status="success",
    )

    estimate = history_db.estimate_seconds(conn, model, "en", 100.0)

    assert estimate == 50.0


def test_estimate_seconds_falls_back_across_languages(tmp_path: Path) -> None:
    conn = history_db.init_db(tmp_path / "history.db")
    model = "mlx-community/whisper-small-mlx"

    history_db.record_run(
        conn,
        source_path="/videos/a.mp4",
        output_path="/videos/a.md",
        model=model,
        language="cs",
        audio_duration_seconds=100.0,
        elapsed_seconds=40.0,
        started_at="2026-07-31T10:00:00",
        finished_at="2026-07-31T10:00:40",
        status="success",
    )

    # No history for English yet, but there is history for the same model in Czech.
    estimate = history_db.estimate_seconds(conn, model, "en", 100.0)

    assert estimate == 40.0
