"""FIFO job queue backed by pending.txt + spawned transcription worker."""
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional


PENDING_FILE = "pending.txt"
STATUS_FILE = "worker_status.json"


@dataclass
class JobStatus:
    running: Optional[str] = None
    queued: list[str] = field(default_factory=list)
    last_failed: Optional[str] = None


def _default_spawner(state_dir: Path) -> subprocess.Popen:
    """Spawn the transcription worker as a detached subprocess."""
    return subprocess.Popen(
        [sys.executable, "-m", "audiologger.transcribe_worker", str(state_dir)],
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


class TranscriptionJobQueue:
    def __init__(
        self,
        state_dir: Path,
        spawner: Callable[[Path], subprocess.Popen] | None = None,
    ):
        self._state_dir = state_dir
        self._state_dir.mkdir(parents=True, exist_ok=True)
        self._spawner = spawner if spawner is not None else _default_spawner
        self._worker: subprocess.Popen | None = None

    @property
    def _pending_path(self) -> Path:
        return self._state_dir / PENDING_FILE

    @property
    def _status_path(self) -> Path:
        return self._state_dir / STATUS_FILE

    def enqueue(self, session_dir: Path) -> None:
        """Append session to pending.txt and ensure worker is running."""
        with self._pending_path.open("a", encoding="utf-8") as f:
            f.write(str(session_dir).replace("\\", "/") + "\n")

        if self._worker is None or self._worker.poll() is not None:
            self._worker = self._spawner(self._state_dir)

    def status(self) -> JobStatus:
        if not self._status_path.exists():
            running = None
            queued: list[str] = []
        else:
            try:
                data = json.loads(self._status_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                data = {}
            running = data.get("running")
            queued = list(data.get("queued", []))

        last_failed: Optional[str] = None
        last_failed_path = self._state_dir / "last_failed.txt"
        if last_failed_path.exists():
            try:
                last_failed = last_failed_path.read_text(encoding="utf-8").strip() or None
            except OSError:
                pass

        return JobStatus(running=running, queued=queued, last_failed=last_failed)
