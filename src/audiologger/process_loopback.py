"""Per-app loopback via Windows Process Loopback API. Filled in Task 13."""
from pathlib import Path


class ProcessLoopbackNotAvailable(Exception):
    """Raised when per-app loopback can't be initialized."""


def record_app_loopback(out_path: Path, app_names: list[str], sample_rate: int, stop_event) -> None:
    """Stub — raises ProcessLoopbackNotAvailable so audio_capture falls back to 'all'."""
    raise ProcessLoopbackNotAvailable("process loopback not yet implemented")
