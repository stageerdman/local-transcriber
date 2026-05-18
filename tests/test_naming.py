from pathlib import Path

from src.naming import build_base_name, sanitize_filename_part, unique_base_name


def test_build_base_name_from_parent_and_stem(tmp_path: Path) -> None:
    root = tmp_path / "SelectedFolder"
    file_path = root / "Alex" / "call.mp3"
    file_path.parent.mkdir(parents=True)
    file_path.touch()

    assert build_base_name(file_path, root) == "Alex - call"


def test_build_base_name_from_deep_folder_path(tmp_path: Path) -> None:
    root = tmp_path / "SelectedFolder"
    file_path = root / "Sales Calls" / "Alex" / "call.mp4"
    file_path.parent.mkdir(parents=True)
    file_path.touch()

    assert build_base_name(file_path, root) == "Sales Calls - Alex - call"


def test_sanitize_unsafe_characters() -> None:
    assert sanitize_filename_part('a<b>c:d/e\\f|g?h*i') == "a b c d e f g h i"


def test_long_name_is_capped(tmp_path: Path) -> None:
    root = tmp_path / "SelectedFolder"
    long_stem = "a" * 200
    file_path = root / f"{long_stem}.mp3"
    file_path.parent.mkdir(parents=True)
    file_path.touch()

    assert len(build_base_name(file_path, root, max_length=120)) == 120


def test_collision_handling() -> None:
    used: set[str] = set()

    assert unique_base_name("Alex - call", used) == "Alex - call"
    assert unique_base_name("Alex - call", used) == "Alex - call 2"
    assert unique_base_name("Alex - call", used) == "Alex - call 3"
