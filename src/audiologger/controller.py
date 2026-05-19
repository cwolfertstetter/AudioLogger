"""RecordingController — state machine for the record/stop toggle."""
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Callable, Protocol

from audiologger.config import Config
from audiologger.paths import MARKER_FILENAME, session_dirname


class RecordingState(Enum):
    IDLE = auto()
    RECORDING = auto()
    STOPPING = auto()


class CaptureLike(Protocol):
    def start(self) -> None: ...
    def stop(self) -> None: ...


CaptureFactory = Callable[[Path, int, str, list[str]], CaptureLike]
"""(session_dir, sample_rate, audio_source, filtered_app_names) -> CaptureLike"""


class RecordingController:
    SAMPLE_RATE = 48000

    def __init__(
        self,
        *,
        config: Config,
        capture_factory: CaptureFactory,
        mix_fn: Callable[[Path, Path, Path], None],
        enqueue_fn: Callable[[Path], None],
        clock: Callable[[], datetime] = datetime.now,
    ):
        self._config = config
        self._capture_factory = capture_factory
        self._mix_fn = mix_fn
        self._enqueue_fn = enqueue_fn
        self._clock = clock
        self._state = RecordingState.IDLE
        self._current_capture: CaptureLike | None = None
        self._current_session: Path | None = None

    @property
    def state(self) -> RecordingState:
        return self._state

    def toggle(self) -> None:
        if self._state is RecordingState.IDLE:
            self._start()
        elif self._state is RecordingState.RECORDING:
            self._stop()
        # STOPPING: ignore

    def _start(self) -> None:
        out = self._config.output_dir
        out.mkdir(parents=True, exist_ok=True)
        session = out / session_dirname(self._clock())
        session.mkdir()
        (session / MARKER_FILENAME).touch()

        capture = self._capture_factory(
            session,
            self.SAMPLE_RATE,
            self._config.audio_source,
            list(self._config.filtered_app_names),
        )
        capture.start()
        self._current_capture = capture
        self._current_session = session
        self._state = RecordingState.RECORDING

    def _stop(self) -> None:
        self._state = RecordingState.STOPPING
        assert self._current_capture is not None
        assert self._current_session is not None
        self._current_capture.stop()

        session = self._current_session
        mic = session / "mic.wav"
        sysw = session / "system.wav"
        mixed = session / "mixed.wav"
        self._mix_fn(mic, sysw, mixed)

        (session / MARKER_FILENAME).unlink(missing_ok=True)
        self._enqueue_fn(session)

        self._current_capture = None
        self._current_session = None
        self._state = RecordingState.IDLE
