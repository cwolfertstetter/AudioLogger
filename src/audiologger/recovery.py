"""Detect sessions left behind by a crashed recording (marker-file present)."""
from pathlib import Path

from audiologger.paths import MARKER_FILENAME


def find_orphaned_sessions(output_dir: Path) -> list[Path]:
    """Return sorted list of session directories containing the marker file."""
    if not output_dir.exists():
        return []
    found = [
        d for d in output_dir.iterdir()
        if d.is_dir() and (d / MARKER_FILENAME).exists()
    ]
    return sorted(found, key=lambda p: p.name)


def find_latest_dictation_session(output_dir: Path) -> Path | None:
    """Return the youngest session directory whose mode.txt content is 'dictation'.

    Returns None if no such session exists.
    """
    if not output_dir.exists():
        return None
    candidates = []
    for d in output_dir.iterdir():
        if not d.is_dir():
            continue
        mode_file = d / "mode.txt"
        if not mode_file.exists():
            continue
        try:
            if mode_file.read_text(encoding="utf-8").strip() == "dictation":
                candidates.append(d)
        except OSError:
            continue
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.name)[-1]
