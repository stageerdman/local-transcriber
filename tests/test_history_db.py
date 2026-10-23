import sqlite3
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


def test_turns_json_round_trips_on_a_fresh_db(tmp_path: Path) -> None:
    conn = history_db.init_db(tmp_path / "history.db")
    turns_json = '[{"speaker": "Person 1", "text": "Hi"}]'

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
        turns_json=turns_json,
    )

    rows = history_db.fetch_history(conn)
    assert rows[0].turns_json == turns_json
    assert history_db.turns_json_for_output_path(conn, "/videos/call.md") == turns_json


def test_turns_json_for_output_path_returns_the_most_recent_match(tmp_path: Path) -> None:
    conn = history_db.init_db(tmp_path / "history.db")
    common = dict(
        source_path="/videos/call.mp4", output_path="/videos/call.md",
        model="m", language="en", audio_duration_seconds=1.0, elapsed_seconds=1.0,
        started_at="2026-07-31T10:00:00", finished_at="2026-07-31T10:00:01", status="success",
    )
    history_db.record_run(conn, **common, turns_json='[{"text": "old"}]')
    history_db.record_run(conn, **common, turns_json='[{"text": "new"}]')

    assert history_db.turns_json_for_output_path(conn, "/videos/call.md") == '[{"text": "new"}]'


def test_turns_json_for_output_path_is_none_when_no_row_has_it(tmp_path: Path) -> None:
    conn = history_db.init_db(tmp_path / "history.db")
    history_db.record_run(
        conn,
        source_path="/videos/solo.mp4", output_path="/videos/solo.md",
        model="m", language="en", audio_duration_seconds=1.0, elapsed_seconds=1.0,
        started_at="2026-07-31T10:00:00", finished_at="2026-07-31T10:00:01", status="success",
    )
    assert history_db.turns_json_for_output_path(conn, "/videos/solo.md") is None
    assert history_db.turns_json_for_output_path(conn, "/videos/missing.md") is None


def test_a_pre_migration_db_gains_the_column_and_old_rows_read_back_null(tmp_path: Path) -> None:
    db_path = tmp_path / "history.db"
    # Build a DB exactly as it looked before `turns_json` existed - the old
    # schema, written directly, bypassing `init_db`'s migration.
    old_conn = sqlite3.connect(db_path)
    old_conn.execute(history_db.SCHEMA)
    old_conn.execute(
        """
        INSERT INTO transcriptions (
            source_path, output_path, model, language, audio_duration_seconds,
            elapsed_seconds, started_at, finished_at, status, error
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ("/videos/old.mp4", "/videos/old.md", "m", "en", 1.0, 1.0,
         "2026-01-01T00:00:00", "2026-01-01T00:00:01", "success", None),
    )
    old_conn.commit()
    old_conn.close()

    conn = history_db.init_db(db_path)  # runs the migration on an existing file

    rows = history_db.fetch_history(conn)
    assert len(rows) == 1
    assert rows[0].source_path == "/videos/old.mp4"
    assert rows[0].turns_json is None
    assert history_db.turns_json_for_output_path(conn, "/videos/old.md") is None

    # And a fresh write on the now-migrated DB still round-trips normally.
    history_db.record_run(
        conn,
        source_path="/videos/new.mp4", output_path="/videos/new.md",
        model="m", language="en", audio_duration_seconds=1.0, elapsed_seconds=1.0,
        started_at="2026-01-02T00:00:00", finished_at="2026-01-02T00:00:01", status="success",
        turns_json='[{"text": "hi"}]',
    )
    assert history_db.turns_json_for_output_path(conn, "/videos/new.md") == '[{"text": "hi"}]'


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
