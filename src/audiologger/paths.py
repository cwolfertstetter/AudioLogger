"""Filesystem paths and session-directory naming."""
import os
from datetime import datetime
from pathlib import Path

MARKER_FILENAME = "RECORDING_IN_PROGRESS"
MODE_FILENAME = "mode.txt"


def appdata_dir() -> Path:
    """Return %APPDATA%/AudioLogger (creates parents only on write)."""
    return Path(os.environ["APPDATA"]) / "AudioLogger"


def config_path() -> Path:
    return appdata_dir() / "config.yaml"


def session_dirname(dt: datetime) -> str:
    """Format: YYYY-MM-DD_HH-MM-SS."""
    return dt.strftime("%Y-%m-%d_%H-%M-%S")
