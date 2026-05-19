"""Records mic + system audio to separate WAV files in parallel."""
import logging
import threading
import wave
from pathlib import Path

import numpy as np
import soundcard as sc

from audiologger.process_loopback import (
    ProcessLoopbackNotAvailable,
    record_app_loopback,
)


log = logging.getLogger(__name__)

CHUNK_SECONDS = 1


class AudioCaptureThread:
    """Two-stream recorder: mic + system loopback (or per-app loopback)."""

    def __init__(
        self,
        session_dir: Path,
        sample_rate: int,
        audio_source: str = "all",
        filtered_app_names: list[str] | None = None,
    ):
        self._session_dir = session_dir
        self._sr = sample_rate
        self._audio_source = audio_source
        self._app_names = filtered_app_names or []
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self.warnings: list[str] = []
        # Recorded for transcript header
        self.mic_device_name: str | None = None
        self.system_device_name: str | None = None

    def start(self) -> None:
        mic_path = self._session_dir / "mic.wav"
        sys_path = self._session_dir / "system.wav"
        self._stop.clear()

        t_mic = threading.Thread(
            target=self._run_mic, args=(mic_path,), name="audio-mic", daemon=True
        )
        t_sys = threading.Thread(
            target=self._run_system, args=(sys_path,), name="audio-sys", daemon=True
        )
        t_mic.start()
        t_sys.start()
        self._threads = [t_mic, t_sys]

    def stop(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=10)

    # --- internal ---

    def _open_wav(self, path: Path) -> wave.Wave_write:
        w = wave.open(str(path), "wb")
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(self._sr)
        return w

    def _run_mic(self, out_path: Path) -> None:
        try:
            mic = sc.default_microphone()
            self.mic_device_name = mic.name
        except Exception as e:
            log.warning("No mic available: %s", e)
            self.warnings.append("Mikrofon nicht verfügbar — nur System-Audio aufgenommen.")
            return
        try:
            with self._open_wav(out_path) as wav, mic.recorder(
                samplerate=self._sr, channels=[0]
            ) as rec:
                while not self._stop.is_set():
                    data = rec.record(numframes=self._sr * CHUNK_SECONDS)
                    self._write_chunk(wav, data)
        except Exception:
            log.exception("Mic recording failed")
            self.warnings.append("Mikrofon-Aufnahme abgebrochen.")

    def _run_system(self, out_path: Path) -> None:
        if self._audio_source == "apps":
            try:
                record_app_loopback(out_path, self._app_names, self._sr, self._stop)
                return
            except ProcessLoopbackNotAvailable as e:
                log.warning("Process loopback unavailable, falling back to 'all': %s", e)
                self.warnings.append(
                    "App-Filter nicht verfügbar — gesamtes System-Audio aufgenommen."
                )
                # fall through to default loopback
        try:
            spk = sc.default_speaker()
            self.system_device_name = spk.name
            loopback_mic = sc.get_microphone(id=str(spk.name), include_loopback=True)
        except Exception as e:
            log.warning("No loopback available: %s", e)
            self.warnings.append("System-Audio (Loopback) nicht verfügbar.")
            return
        try:
            with self._open_wav(out_path) as wav, loopback_mic.recorder(
                samplerate=self._sr, channels=[0]
            ) as rec:
                while not self._stop.is_set():
                    data = rec.record(numframes=self._sr * CHUNK_SECONDS)
                    self._write_chunk(wav, data)
        except Exception:
            log.exception("System recording failed")
            self.warnings.append("System-Audio-Aufnahme abgebrochen.")

    def _write_chunk(self, wav: wave.Wave_write, data: np.ndarray) -> None:
        """data is float32 from soundcard, shape (N, 1). Convert to int16."""
        mono = data[:, 0] if data.ndim == 2 else data
        clipped = np.clip(mono, -1.0, 1.0)
        i16 = (clipped * 32767).astype(np.int16)
        wav.writeframes(i16.tobytes())
