"""FIFO job queue backed by pending.txt + spawned transcription worker."""
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Optional


PENDING_FILE = "pending.txt"
STATUS_FILE = "worker_status.json"
LOCK_FILE = "pending.lock"
# Touched by a worker that found the queue empty and is shutting down.  Until
# the process is gone, poll() still says it runs, so without this an enqueue in
# that window would wait for the next recording.
EXITING_FILE = "worker_exiting"


@contextmanager
def pending_lock(state_dir: Path) -> Iterator[None]:
    """Serialise pending.txt read-modify-write between tray and worker."""
    state_dir.mkdir(parents=True, exist_ok=True)
    with open(state_dir / LOCK_FILE, "a+b") as f:
        if sys.platform == "win32":
            import msvcrt
            f.seek(0)
            while True:
                try:
                    msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
                    break
                except OSError:  # LK_LOCK gives up after ~10 s; keep waiting
                    continue
            try:
                yield
            finally:
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)


def atomic_write_text(path: Path, text: str) -> None:
    """Write via a temp file + rename, so a reader or a crash never sees half a file.

    On Windows the rename fails while another process has the target open for
    a moment (the tray reading the status), so retry briefly.
    """
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    for attempt in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.05)


@dataclass
class JobStatus:
    running: Optional[str] = None
    queued: list[str] = field(default_factory=list)
    last_failed: Optional[str] = None
    mode: Optional[str] = None
    warming: bool = False
    last_finished: Optional[dict] = None


def _default_spawner(state_dir: Path, prewarm: bool = False) -> subprocess.Popen:
    """Spawn the transcription worker as a detached subprocess."""
    cmd = [sys.executable, "-m", "audiologger.transcribe_worker", str(state_dir)]
    if prewarm:
        cmd.append("--prewarm")
    return subprocess.Popen(
        cmd,
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

    def _worker_alive(self) -> bool:
        """A worker that is running and will still look at pending.txt."""
        if self._worker is None or self._worker.poll() is not None:
            return False
        return not (self._state_dir / EXITING_FILE).exists()

    def _spawn(self, prewarm: bool) -> None:
        (self._state_dir / EXITING_FILE).unlink(missing_ok=True)
        self._worker = self._spawner(self._state_dir, prewarm=prewarm)

    def prewarm(self) -> None:
        """Spawn the worker now (with --prewarm flag) if it's not already running."""
        with pending_lock(self._state_dir):
            if not self._worker_alive():
                self._spawn(prewarm=True)

    def enqueue(self, session_dir: Path) -> None:
        """Append session to pending.txt and ensure worker is running."""
        with pending_lock(self._state_dir):
            with self._pending_path.open("a", encoding="utf-8") as f:
                f.write(str(session_dir).replace("\\", "/") + "\n")
            if not self._worker_alive():
                self._spawn(prewarm=False)

    def has_pending(self) -> bool:
        try:
            text = self._pending_path.read_text(encoding="utf-8")
        except OSError:
            return False
        return any(line.strip() for line in text.splitlines())

    def resume(self) -> None:
        """Start a worker for jobs left in pending.txt by a killed worker.

        A worker terminated mid-job (model change, quit) never reaches the
        cleanup that drops its job, so the job is still queued -- but nothing
        would pick it up until the next recording is enqueued.
        """
        with pending_lock(self._state_dir):
            if self.has_pending() and not self._worker_alive():
                self._spawn(prewarm=False)

    def stop_worker(self, timeout: float = 5) -> None:
        """Terminate the worker if it runs.  Its current job stays queued."""
        worker = self._worker
        if worker is not None and worker.poll() is None:
            worker.terminate()
            worker.wait(timeout=timeout)
        self._worker = None

    def status(self) -> JobStatus:
        if not self._status_path.exists():
            running = None
            queued: list[str] = []
            mode: Optional[str] = None
            warming = False
            last_finished: Optional[dict] = None
        else:
            try:
                data = json.loads(self._status_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                data = {}
            running = data.get("running")
            queued = list(data.get("queued", []))
            mode = data.get("mode", None)
            warming = bool(data.get("warming", False))
            raw_lf = data.get("last_finished", None)
            last_finished = raw_lf if isinstance(raw_lf, dict) else None

        last_failed: Optional[str] = None
        last_failed_path = self._state_dir / "last_failed.txt"
        if last_failed_path.exists():
            try:
                last_failed = last_failed_path.read_text(encoding="utf-8").strip() or None
            except OSError:
                pass

        return JobStatus(
            running=running,
            queued=queued,
            last_failed=last_failed,
            mode=mode,
            warming=warming,
            last_finished=last_finished,
        )
