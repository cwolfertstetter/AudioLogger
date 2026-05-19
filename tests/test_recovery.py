from pathlib import Path

from audiologger.paths import MARKER_FILENAME
from audiologger.recovery import find_orphaned_sessions, find_latest_dictation_session


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


# --- find_latest_dictation_session ----------------------------------------

def test_find_latest_dictation_session_returns_none_when_empty(tmp_path):
    assert find_latest_dictation_session(tmp_path) is None


def test_find_latest_dictation_session_ignores_meetings(tmp_path):
    """Dictation session is returned even if meeting session is younger."""
    dictation = tmp_path / "2026-05-18_14-00-00"
    dictation.mkdir()
    (dictation / "mode.txt").write_text("dictation", encoding="utf-8")

    meeting = tmp_path / "2026-05-18_15-00-00"
    meeting.mkdir()
    (meeting / "mode.txt").write_text("meeting", encoding="utf-8")

    result = find_latest_dictation_session(tmp_path)
    assert result == dictation


def test_find_latest_dictation_session_returns_youngest(tmp_path):
    """With multiple dictation sessions, the lexicographically last one is returned."""
    for name in ["2026-05-18_10-00-00", "2026-05-18_12-00-00", "2026-05-18_11-00-00"]:
        d = tmp_path / name
        d.mkdir()
        (d / "mode.txt").write_text("dictation", encoding="utf-8")

    result = find_latest_dictation_session(tmp_path)
    assert result is not None
    assert result.name == "2026-05-18_12-00-00"


def test_find_latest_dictation_session_ignores_dirs_without_mode_file(tmp_path):
    no_mode = tmp_path / "2026-05-18_14-00-00"
    no_mode.mkdir()
    # No mode.txt written

    assert find_latest_dictation_session(tmp_path) is None
