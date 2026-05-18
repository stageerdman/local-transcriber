from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal


Status = Literal["success", "error"]


@dataclass
class ProcessedFile:
    original_path: str
    audio_output_path: str | None
    transcript_output_path: str | None
    status: Status
    error: str | None = None


@dataclass
class TranscriptionReport:
    selected_input_folder: str
    output_folder: str
    started_at: str
    finished_at: str | None
    model_used: str
    total_files_found: int
    successful_transcriptions: int = 0
    failed_files: int = 0
    skipped_unsupported_files: int = 0
    processed_files: list[ProcessedFile] = field(default_factory=list)

    def add_success(self, original_path: Path, audio_output_path: Path, transcript_output_path: Path) -> None:
        self.successful_transcriptions += 1
        self.processed_files.append(
            ProcessedFile(
                original_path=str(original_path),
                audio_output_path=str(audio_output_path),
                transcript_output_path=str(transcript_output_path),
                status="success",
            )
        )

    def add_error(
        self,
        original_path: Path,
        error: str,
        audio_output_path: Path | None = None,
        transcript_output_path: Path | None = None,
    ) -> None:
        self.failed_files += 1
        self.processed_files.append(
            ProcessedFile(
                original_path=str(original_path),
                audio_output_path=str(audio_output_path) if audio_output_path else None,
                transcript_output_path=str(transcript_output_path) if transcript_output_path else None,
                status="error",
                error=error,
            )
        )

    def finish(self) -> None:
        self.finished_at = datetime.now().isoformat(timespec="seconds")

    def write(self, output_path: Path) -> None:
        output_path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
