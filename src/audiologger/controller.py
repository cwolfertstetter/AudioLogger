"""RecordingController — state machine for the record/stop toggle."""
import logging
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Callable, Protocol

from audiologger.config import Config
from audiologger.paths import MARKER_FILENAME, MODE_FILENAME, session_dirname
from audiologger.recovery import find_latest_dictation_session

log = logging.getLogger(__name__)


class RecordingState(Enum):
    IDLE = auto()
    RECORDING = auto()
    STOPPING = auto()


class CaptureLike(Protocol):
    warnings: list[str]

    def start(self) -> None: ...
    def stop(self) -> None: ...


CaptureFactory = Callable[[Path, int, str, list[str], bool], CaptureLike]
"""(session_dir, sample_rate, audio_source, filtered_app_names, mic_only) -> CaptureLike"""


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
        self._current_mode: str | None = None

    @property
    def state(self) -> RecordingState:
        return self._state

    def toggle(self, mode: str = "meeting") -> None:
        if self._state is RecordingState.IDLE:
            self._start(mode)
        elif self._state is RecordingState.RECORDING:
            if mode == self._current_mode:
                self._stop()
            else:
                log.warning(
                    "Hotkey for %s ignored — already recording in %s",
                    mode,
                    self._current_mode,
                )
        # STOPPING: ignore

    def _start(self, mode: str = "meeting") -> None:
        out = self._config.output_dir
        out.mkdir(parents=True, exist_ok=True)

        # Resolve effective mode and optional extend target before creating session dir.
        target_session: Path | None = None
        if mode == "dictation_extend":
            target_session = find_latest_dictation_session(out)
            if target_session is None:
                log.info("No previous dictation session found; falling back to dictation mode")
                mode = "dictation"

        session = out / session_dirname(self._clock())
        session.mkdir()
        (session / MARKER_FILENAME).touch()
        (session / MODE_FILENAME).write_text(mode, encoding="utf-8")

        if mode == "dictation_extend" and target_session is not None:
            (session / "target_session.txt").write_text(
                target_session.as_posix(), encoding="utf-8"
            )

        capture = self._capture_factory(
            session,
            self.SAMPLE_RATE,
            self._config.audio_source,
            list(self._config.filtered_app_names),
            mode in ("dictation", "dictation_extend"),
        )
        capture.start()
        self._current_capture = capture
        self._current_session = session
        self._current_mode = mode
        self._state = RecordingState.RECORDING

    def _stop(self) -> None:
        self._state = RecordingState.STOPPING
        if self._current_capture is None or self._current_session is None:
            raise RuntimeError("_stop called without active capture/session")
        capture = self._current_capture
        session = self._current_session
        capture.stop()

        # C1: write capture warnings before dropping reference
        if capture.warnings:
            try:
                (session / "capture_warnings.txt").write_text(
                    "\n".join(capture.warnings) + "\n", encoding="utf-8"
                )
            except OSError:
                log.exception("Failed to write capture_warnings.txt")

        # C2: reset state BEFORE anything that can fail so errors don't lock
        # the state machine in STOPPING permanently.
        self._current_capture = None
        self._current_session = None
        self._current_mode = None
        self._state = RecordingState.IDLE

        mic = session / "mic.wav"
        sysw = session / "system.wav"
        mixed = session / "mixed.wav"
        try:
            self._mix_fn(mic, sysw, mixed)
        except Exception:
            log.exception("mix failed for %s", session)
        try:
            (session / MARKER_FILENAME).unlink(missing_ok=True)
        except OSError:
            log.exception("Failed to remove marker for %s", session)
        try:
            self._enqueue_fn(session)
        except Exception:
            log.exception("Failed to enqueue session %s", session)
