import json
from pathlib import Path

from src.report import TranscriptionReport


def test_report_writes_valid_json_with_success_and_error_entries(tmp_path: Path) -> None:
    report = TranscriptionReport(
        selected_input_folder="/input",
        output_folder="/output",
        started_at="2026-05-18T14:30:00",
        finished_at=None,
        model_used="mlx-community/whisper-small-mlx",
        total_files_found=2,
    )
    report.add_success(Path("/input/a.mp3"), Path("/output/audio/a.mp3"), Path("/output/text/a.txt"))
    report.add_error(Path("/input/b.mp4"), "conversion failed", Path("/output/audio/b.mp3"), None)
    report.finish()

    report_path = tmp_path / "transcription_report.json"
    report.write(report_path)

    data = json.loads(report_path.read_text(encoding="utf-8"))
    assert data["successful_transcriptions"] == 1
    assert data["failed_files"] == 1
    assert data["processed_files"][0]["status"] == "success"
    assert data["processed_files"][1]["status"] == "error"
    assert data["processed_files"][1]["error"] == "conversion failed"
