from __future__ import annotations

import sys
import traceback
from datetime import datetime
from pathlib import Path

from src.converter import check_ffmpeg_installed, prepare_mp3
from src.folder_picker import pick_folder
from src.media_finder import find_media_files
from src.naming import build_base_name, unique_base_name
from src.report import TranscriptionReport
from src.transcription import create_model, ensure_faster_whisper_available, transcribe_mp3


MODEL_NAME = "medium"
DOWNLOADS_TRANSCRIPTS = Path.home() / "Downloads" / "transcripts"


def create_run_folder() -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_folder = DOWNLOADS_TRANSCRIPTS / timestamp
    index = 2
    while run_folder.exists():
        run_folder = DOWNLOADS_TRANSCRIPTS / f"{timestamp}_{index}"
        index += 1

    (run_folder / "text").mkdir(parents=True, exist_ok=False)
    (run_folder / "audio").mkdir(parents=True, exist_ok=True)
    return run_folder


def main() -> int:
    print("Opening folder picker...")
    input_folder = pick_folder()
    if input_folder is None:
        print("No folder selected. Exiting.")
        return 0

    print(f"Selected input folder: {input_folder}")

    if not check_ffmpeg_installed():
        print("ERROR: ffmpeg is not installed or not on PATH.")
        print("Install it with Homebrew: brew install ffmpeg")
        return 1

    try:
        ensure_faster_whisper_available()
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 1

    run_folder = create_run_folder()
    text_folder = run_folder / "text"
    audio_folder = run_folder / "audio"
    print(f"Output path: {run_folder}")

    media_files = find_media_files(input_folder)
    print(f"Files found: {len(media_files)}")

    started_at = datetime.now().isoformat(timespec="seconds")
    report = TranscriptionReport(
        selected_input_folder=str(input_folder),
        output_folder=str(run_folder),
        started_at=started_at,
        finished_at=None,
        model_used=MODEL_NAME,
        total_files_found=len(media_files),
    )

    if not media_files:
        report.finish()
        report.write(run_folder / "transcription_report.json")
        print("No supported media files found.")
        print(f"Report written: {run_folder / 'transcription_report.json'}")
        return 0

    print(f"Loading faster-whisper model: {MODEL_NAME}")
    try:
        model = create_model(MODEL_NAME)
    except Exception as exc:
        report.finish()
        report.write(run_folder / "transcription_report.json")
        print(f"ERROR: Failed to load model: {exc}")
        print(f"Report written: {run_folder / 'transcription_report.json'}")
        return 1

    used_names: set[str] = set()
    for index, media_file in enumerate(media_files, start=1):
        base_name = unique_base_name(build_base_name(media_file, input_folder), used_names)
        audio_output_path = audio_folder / f"{base_name}.mp3"
        transcript_output_path = text_folder / f"{base_name}.txt"

        print()
        print(f"[{index}/{len(media_files)}] Processing: {media_file}")
        try:
            if media_file.suffix.lower() == ".mp3":
                print(f"Copying MP3 to: {audio_output_path}")
            else:
                print(f"Converting to MP3: {audio_output_path}")
            prepare_mp3(media_file, audio_output_path)

            print(f"Transcribing: {audio_output_path}")
            transcript_text = transcribe_mp3(model, audio_output_path, language="en")
            transcript_output_path.write_text(transcript_text, encoding="utf-8")
            print(f"Saved transcript: {transcript_output_path}")

            report.add_success(media_file, audio_output_path, transcript_output_path)
        except Exception as exc:
            error_message = f"{type(exc).__name__}: {exc}"
            print(f"ERROR: {error_message}")
            traceback.print_exc()
            report.add_error(media_file, error_message, audio_output_path, transcript_output_path)

    report.finish()
    report_path = run_folder / "transcription_report.json"
    report.write(report_path)

    print()
    print("Final summary")
    print(f"Output path: {run_folder}")
    print(f"Files found: {report.total_files_found}")
    print(f"Successful transcriptions: {report.successful_transcriptions}")
    print(f"Failed files: {report.failed_files}")
    print(f"Report written: {report_path}")
    return 0 if report.failed_files == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
