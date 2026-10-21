from __future__ import annotations

import os
from pathlib import Path

# Load .env (GROQ_API_KEY, etc.) into the process environment before anything
# else in the app runs, so it's set by the time the engine subprocess is
# spawned - multiprocessing's "spawn" context inherits the parent's
# os.environ at Process.start(), so setting it here is enough. Existing
# environment variables win over .env (same convention as most .env loaders).
# No third-party dependency: the file is just KEY=VALUE lines.
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


def _load_dotenv(path: Path) -> None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip()


_load_dotenv(_ENV_PATH)
