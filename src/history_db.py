from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from src.catalog import DEFAULT_REAL_TIME_FACTOR, FALLBACK_REAL_TIME_FACTOR

SCHEMA = """
CREATE TABLE IF NOT EXISTS transcriptions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_path TEXT NOT NULL,
    output_path TEXT,
    model TEXT NOT NULL,
    language TEXT,
    audio_duration_seconds REAL,
    elapsed_seconds REAL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT NOT NULL,
    error TEXT
);
"""


def default_db_path() -> Path:
    return Path.home() / "Library" / "Application Support" / "LocalTranscriber" / "history.db"


def _migrate(conn: sqlite3.Connection) -> None:
    """Add columns that didn't exist when this DB was first created.

    This codebase's first schema migration (every prior change was additive
    via `CREATE TABLE IF NOT EXISTS` on a fresh DB) - `ALTER TABLE ... ADD
    COLUMN` isn't idempotent, so each column is only added if `table_info`
    doesn't already report it, which also makes this safe to call on every
    `init_db()`, not just once.
    """
    existing = {row[1] for row in conn.execute("PRAGMA table_info(transcriptions)")}
    if "turns_json" not in existing:
        conn.execute("ALTER TABLE transcriptions ADD COLUMN turns_json TEXT")
        conn.commit()


def init_db(db_path: Path | None = None) -> sqlite3.Connection:
    db_path = db_path or default_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute(SCHEMA)
    conn.commit()
    _migrate(conn)
    return conn


@dataclass
class HistoryRow:
    id: int
    source_path: str
    output_path: str | None
    model: str
    language: str | None
    audio_duration_seconds: float | None
    elapsed_seconds: float | None
    started_at: str
    finished_at: str | None
    status: str
    error: str | None
    # JSON text (`dialogue.blocks_to_dicts`), set for a multi-track job - see
    # `app.worker`. `None` for single-track jobs (nothing to diarize) and for
    # any row written before this column existed.
    turns_json: str | None = None


def record_run(
    conn: sqlite3.Connection,
    *,
    source_path: str,
    output_path: str | None,
    model: str,
    language: str | None,
    audio_duration_seconds: float | None,
    elapsed_seconds: float | None,
    started_at: str,
    finished_at: str | None,
    status: str,
    error: str | None = None,
    turns_json: str | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO transcriptions (
            source_path, output_path, model, language, audio_duration_seconds,
            elapsed_seconds, started_at, finished_at, status, error, turns_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            source_path,
            output_path,
            model,
            language,
            audio_duration_seconds,
            elapsed_seconds,
            started_at,
            finished_at,
            status,
            error,
            turns_json,
        ),
    )
    conn.commit()


def fetch_history(conn: sqlite3.Connection, limit: int = 200) -> list[HistoryRow]:
    cursor = conn.execute(
        """
        SELECT id, source_path, output_path, model, language, audio_duration_seconds,
               elapsed_seconds, started_at, finished_at, status, error, turns_json
        FROM transcriptions
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    )
    return [HistoryRow(*row) for row in cursor.fetchall()]


def turns_json_for_output_path(conn: sqlite3.Connection, output_path: str) -> str | None:
    """The most recently recorded `turns_json` for this output path.

    `output_path` isn't unique (a file can be re-transcribed to the same
    destination), so this is the latest match, not the only one. `None` when
    there's no row for this path, or the matching row predates this column /
    was a single-track job - callers fall back to re-parsing the `.md` either
    way (see `app.bridge._parse_transcript`)."""
    row = conn.execute(
        """
        SELECT turns_json FROM transcriptions
        WHERE output_path = ? AND turns_json IS NOT NULL
        ORDER BY id DESC LIMIT 1
        """,
        (output_path,),
    ).fetchone()
    return row[0] if row else None


def estimate_seconds(
    conn: sqlite3.Connection,
    model: str,
    language: str | None,
    duration_seconds: float,
) -> float:
    ratio = _average_ratio(conn, model, language)
    if ratio is None:
        ratio = _average_ratio(conn, model, None, match_any_language=True)
    if ratio is None:
        ratio = DEFAULT_REAL_TIME_FACTOR.get(model, FALLBACK_REAL_TIME_FACTOR)
    return duration_seconds * ratio


def _average_ratio(
    conn: sqlite3.Connection,
    model: str,
    language: str | None,
    match_any_language: bool = False,
) -> float | None:
    if match_any_language:
        query = """
            SELECT elapsed_seconds, audio_duration_seconds FROM transcriptions
            WHERE model = ? AND status = 'success'
              AND audio_duration_seconds > 0 AND elapsed_seconds IS NOT NULL
            ORDER BY id DESC LIMIT 20
        """
        params = (model,)
    else:
        query = """
            SELECT elapsed_seconds, audio_duration_seconds FROM transcriptions
            WHERE model = ? AND language IS ? AND status = 'success'
              AND audio_duration_seconds > 0 AND elapsed_seconds IS NOT NULL
            ORDER BY id DESC LIMIT 20
        """
        params = (model, language)

    rows = conn.execute(query, params).fetchall()
    if not rows:
        return None

    ratios = [elapsed / duration for elapsed, duration in rows]
    return sum(ratios) / len(ratios)
