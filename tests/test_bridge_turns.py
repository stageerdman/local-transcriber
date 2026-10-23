"""Phase 7 (word-level-dialogue-blocks): the bridge should serve real
structured turn data (speaker/start/end/words) when it's available, from
`job.turns` for a just-finished job or `history_db.turns_json_for_output_path`
when reopening a past one - falling back to the markdown regex parse
(`_parse_transcript`) for legacy rows and single-track jobs, same as before
this phase existed."""
from pathlib import Path

import pytest

from app.bridge import Bridge
from app.jobs import Job
from src import history_db


def _bare_bridge(db) -> Bridge:
    b = Bridge.__new__(Bridge)
    b.db = db
    return b


@pytest.fixture
def db(tmp_path: Path):
    return history_db.init_db(tmp_path / "history.db")


def _write_md(path: Path, body: str = "**Person 1** [00:00]\nHello.\n") -> None:
    path.write_text(body, encoding="utf-8")


def test_job_payload_uses_job_turns_when_present(tmp_path: Path, db) -> None:
    b = _bare_bridge(db)
    md = tmp_path / "call.md"
    _write_md(md)
    job = Job(source_path=tmp_path / "call.mp4", model="m", language="cs")
    job.status = "done"
    job.output_path = md
    job.turns = [
        {"speaker": "Person 1", "start": 0.0, "end": 1.0, "interrupting": False,
         "text": "Hi there.", "words": [{"start": 0.0, "end": 1.0, "text": "Hi there.", "lead_space": True}]},
    ]

    payload = b._job_payload(job)

    transcript = payload["transcript"]
    assert transcript["turns"][0]["text"] == "Hi there."
    assert transcript["turns"][0]["start"] == 0.0
    assert transcript["turns"][0]["words"][0]["text"] == "Hi there."


def test_job_payload_falls_back_to_markdown_parse_without_job_turns(tmp_path: Path, db) -> None:
    b = _bare_bridge(db)
    md = tmp_path / "call.md"
    _write_md(md, "**Person 1** [00:00]\nFrom the file.\n")
    job = Job(source_path=tmp_path / "call.mp4", model="m", language="cs")
    job.status = "done"
    job.output_path = md
    job.turns = None  # single-track job: nothing to diarize

    payload = b._job_payload(job)

    assert payload["transcript"]["turns"][0]["text"] == "From the file."
    assert "words" not in payload["transcript"]["turns"][0]


def test_open_call_uses_history_turns_json_when_present(tmp_path: Path, db) -> None:
    b = _bare_bridge(db)
    md = tmp_path / "call.md"
    _write_md(md)
    history_db.record_run(
        db,
        source_path=str(tmp_path / "call.mp4"), output_path=str(md),
        model="m", language="cs", audio_duration_seconds=1.0, elapsed_seconds=1.0,
        started_at="2026-10-22T00:00:00", finished_at="2026-10-22T00:00:01", status="success",
        turns_json='[{"speaker": "Person 1", "start": 2.0, "end": 3.0, "interrupting": false, '
                   '"text": "From history.", "words": []}]',
    )

    data = b.open_call(str(md))

    assert data["turns"][0]["text"] == "From history."
    assert data["turns"][0]["start"] == 2.0
    assert data["name"] == "call"


def test_open_call_falls_back_to_markdown_parse_for_a_legacy_row(tmp_path: Path, db) -> None:
    b = _bare_bridge(db)
    md = tmp_path / "call.md"
    _write_md(md, "**Person 1** [00:00]\nLegacy text.\n")
    history_db.record_run(
        db,
        source_path=str(tmp_path / "call.mp4"), output_path=str(md),
        model="m", language="cs", audio_duration_seconds=1.0, elapsed_seconds=1.0,
        started_at="2026-10-22T00:00:00", finished_at="2026-10-22T00:00:01", status="success",
        # no turns_json - a row from before this column, or a single-track job
    )

    data = b.open_call(str(md))

    assert data["turns"][0]["text"] == "Legacy text."


def test_open_call_reports_a_missing_file(tmp_path: Path, db) -> None:
    b = _bare_bridge(db)
    assert "error" in b.open_call(str(tmp_path / "gone.md"))
