"""Screenshots taken during a meeting recording: naming, discovery, and the
clipboard watcher. See docs/superpowers/specs/2026-10-02-screenshots-design.md.
"""
from pathlib import Path

from audiologger.screenshot_watch import Screenshot, find_screenshots, screenshot_file_name


def test_the_name_carries_the_offset_into_the_recording():
    assert screenshot_file_name(751.6, set()) == "screenshot_00-12-31.png"
    assert screenshot_file_name(3725.0, set()) == "screenshot_01-02-05.png"


def test_a_second_screenshot_in_the_same_second_gets_a_suffix():
    taken = {"screenshot_00-12-31.png"}
    assert screenshot_file_name(751.2, taken) == "screenshot_00-12-31_2.png"
    taken.add("screenshot_00-12-31_2.png")
    assert screenshot_file_name(751.9, taken) == "screenshot_00-12-31_3.png"


def _touch(folder: Path, *names: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for name in names:
        (folder / name).write_bytes(b"png")


def test_screenshots_are_found_by_name_in_time_order(tmp_path):
    _touch(tmp_path / "screenshots",
           "screenshot_00-12-31_2.png", "screenshot_01-02-05.png", "screenshot_00-12-31.png")

    assert find_screenshots(tmp_path) == [
        Screenshot(at_s=751, path="screenshots/screenshot_00-12-31.png"),
        Screenshot(at_s=751, path="screenshots/screenshot_00-12-31_2.png"),
        Screenshot(at_s=3725, path="screenshots/screenshot_01-02-05.png"),
    ]


def test_other_files_in_the_folder_are_ignored(tmp_path):
    _touch(tmp_path / "screenshots", "notes.txt", "screenshot.bmp", "screenshot_00-00-05.png")

    assert [s.at_s for s in find_screenshots(tmp_path)] == [5]


def test_a_recording_without_screenshots_has_none(tmp_path):
    assert find_screenshots(tmp_path) == []
