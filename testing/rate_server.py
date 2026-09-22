#!/usr/bin/env python3
"""Tiny local server for rating benchmark transcripts 1-5.

No dependencies beyond the stdlib. Serves static/rate.html plus a couple of
JSON endpoints backed by results/results.json:

    GET  /api/results         -> the whole results.json, incl. transcripts
    POST /api/rate            -> {"id": "...", "rating": 1-5, "notes": "..."}

Run with: python rate_server.py [--port 8765]
"""

from __future__ import annotations

import argparse
import json
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

TESTING_ROOT = Path(__file__).resolve().parent
STATIC_DIR = TESTING_ROOT / "static"
RESULTS_JSON = TESTING_ROOT / "results" / "results.json"
TRANSCRIPTS_DIR = TESTING_ROOT / "results" / "transcripts"


def load_results() -> dict:
    if not RESULTS_JSON.exists():
        return {"video": None, "audio_duration_seconds": None, "runs": {}}
    return json.loads(RESULTS_JSON.read_text())


def save_results(data: dict) -> None:
    RESULTS_JSON.write_text(json.dumps(data, indent=2))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:  # quieter default logging
        print(f"{self.address_string()} - {fmt % args}")

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - http.server naming
        if self.path == "/" or self.path == "":
            self.path = "/rate.html"

        if self.path == "/api/results":
            data = load_results()
            # Inline each transcript's text so the page needs one fetch.
            for run in data.get("runs", {}).values():
                transcript_file = run.get("transcript_file")
                if transcript_file:
                    full_path = TESTING_ROOT / transcript_file
                    run["transcript_text"] = full_path.read_text() if full_path.exists() else ""
            self._send_json(data)
            return

        # Static file serving, restricted to static/.
        safe_name = self.path.lstrip("/")
        file_path = (STATIC_DIR / safe_name).resolve()
        if STATIC_DIR not in file_path.parents and file_path != STATIC_DIR:
            self.send_error(404)
            return
        if not file_path.is_file():
            self.send_error(404)
            return

        content_type = "text/html" if file_path.suffix == ".html" else (
            "application/javascript" if file_path.suffix == ".js" else (
                "text/css" if file_path.suffix == ".css" else "application/octet-stream"
            )
        )
        body = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/rate":
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(length))
            run_id = payload["id"]
            rating = payload.get("rating")
            notes = payload.get("notes", "")
        except (KeyError, json.JSONDecodeError, ValueError):
            self._send_json({"error": "bad request"}, status=400)
            return

        if rating is not None and rating not in (1, 2, 3, 4, 5):
            self._send_json({"error": "rating must be 1-5 or null"}, status=400)
            return

        data = load_results()
        if run_id not in data.get("runs", {}):
            self._send_json({"error": f"unknown run id {run_id!r}"}, status=404)
            return

        data["runs"][run_id]["rating"] = rating
        data["runs"][run_id]["notes"] = notes
        save_results(data)
        self._send_json({"ok": True})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"Serving rating UI at {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
