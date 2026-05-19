from datetime import datetime
from pathlib import Path

from audiologger.paths import (
    appdata_dir,
    config_path,
    session_dirname,
    MARKER_FILENAME,
)


def test_appdata_dir_under_roaming(monkeypatch, tmp_path):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    assert appdata_dir() == tmp_path / "AudioLogger"


def test_config_path_under_appdata(monkeypatch, tmp_path):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    assert config_path() == tmp_path / "AudioLogger" / "config.yaml"


def test_session_dirname_format():
    dt = datetime(2026, 5, 18, 14, 32, 15)
    assert session_dirname(dt) == "2026-05-18_14-32-15"


def test_marker_filename_constant():
    assert MARKER_FILENAME == "RECORDING_IN_PROGRESS"
