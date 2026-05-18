from __future__ import annotations

import re
from pathlib import Path


MAX_BASE_NAME_LENGTH = 120
UNSAFE_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
SPACE_RUN = re.compile(r"\s+")


def sanitize_filename_part(value: str) -> str:
    value = UNSAFE_CHARS.sub(" ", value)
    value = SPACE_RUN.sub(" ", value).strip()
    value = value.rstrip(".")
    return value or "Untitled"


def build_base_name(input_file: Path, root_folder: Path, max_length: int = MAX_BASE_NAME_LENGTH) -> str:
    relative = input_file.resolve().relative_to(root_folder.resolve())
    parts = [sanitize_filename_part(part) for part in relative.parent.parts]
    parts.append(sanitize_filename_part(relative.stem))

    base_name = " - ".join(part for part in parts if part)
    if len(base_name) <= max_length:
        return base_name

    return base_name[:max_length].rstrip(" .-") or "Untitled"


def unique_base_name(base_name: str, used_names: set[str], max_length: int = MAX_BASE_NAME_LENGTH) -> str:
    candidate = base_name[:max_length].rstrip(" .-") or "Untitled"
    if candidate not in used_names:
        used_names.add(candidate)
        return candidate

    index = 2
    while True:
        suffix = f" {index}"
        prefix_length = max_length - len(suffix)
        candidate = f"{base_name[:prefix_length].rstrip(' .-')}{suffix}" or f"Untitled{suffix}"
        if candidate not in used_names:
            used_names.add(candidate)
            return candidate
        index += 1
