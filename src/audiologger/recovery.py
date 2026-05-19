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
