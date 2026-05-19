from pathlib import Path

from audiologger.paths import MARKER_FILENAME
from audiologger.recovery import find_orphaned_sessions


def test_no_sessions_returns_empty(tmp_path):
    assert find_orphaned_sessions(tmp_path) == []


def test_finds_session_with_marker(tmp_path):
    sess = tmp_path / "2026-05-18_14-00-00"
    sess.mkdir()
    (sess / MARKER_FILENAME).touch()
    result = find_orphaned_sessions(tmp_path)
    assert result == [sess]


def test_ignores_session_without_marker(tmp_path):
    sess = tmp_path / "2026-05-18_14-00-00"
    sess.mkdir()
    assert find_orphaned_sessions(tmp_path) == []


def test_ignores_files_at_top_level(tmp_path):
    (tmp_path / "notes.txt").touch()
    assert find_orphaned_sessions(tmp_path) == []


def test_returns_sorted(tmp_path):
    for name in ["2026-05-18_15-00-00", "2026-05-18_13-00-00", "2026-05-18_14-00-00"]:
        d = tmp_path / name
        d.mkdir()
        (d / MARKER_FILENAME).touch()
    result = find_orphaned_sessions(tmp_path)
    assert [p.name for p in result] == [
        "2026-05-18_13-00-00",
        "2026-05-18_14-00-00",
        "2026-05-18_15-00-00",
    ]


def test_missing_output_dir_returns_empty(tmp_path):
    assert find_orphaned_sessions(tmp_path / "does-not-exist") == []
