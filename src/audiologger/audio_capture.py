"""Records mic + system audio to separate WAV files in parallel."""
import logging
import threading
import wave
from pathlib import Path

import numpy as np
import soundcard as sc

from audiologger.audio_mix import loopback_part_name, mix_loopback_parts
from audiologger.mic_capture import (
    MicrophoneNotAvailable,
    record_microphone,
)
from audiologger.process_loopback import (
    ProcessLoopbackNotAvailable,
    record_app_loopback,
)


log = logging.getLogger(__name__)

CHUNK_SECONDS = 1


def _to_int16(data: np.ndarray) -> np.ndarray:
    """soundcard delivers float32 shaped (N, channels); keep channel 0 as int16."""
    mono = data[:, 0] if data.ndim == 2 else data
    return (np.clip(mono, -1.0, 1.0) * 32767).astype(np.int16)


class LoopbackPartWriter:
    """Writes one output device's loopback into a part file.

    Every output device is looped back during a recording and most of them stay
    silent throughout, so the file is only created once the device first makes
    a sound.  Its start position goes into the file name (see
    audio_mix.loopback_part_name) so the parts can be laid back onto one
    timeline -- by stop(), or by crash recovery if the machine dies mid-call.
    """

    def __init__(self, session_dir: Path, index: int, sample_rate: int):
        self._dir = session_dir
        self._index = index
        self._sr = sample_rate
        self._samples_seen = 0
        self._wav: wave.Wave_write | None = None
        self.path: Path | None = None
        self.failed = False

    def write(self, samples: np.ndarray) -> None:
        if self._wav is None:
            if not samples.any():
                self._samples_seen += len(samples)
                return
            self.path = self._dir / loopback_part_name(self._index, self._samples_seen)
            self._wav = wave.open(str(self.path), "wb")
            self._wav.setnchannels(1)
            self._wav.setsampwidth(2)
            self._wav.setframerate(self._sr)
        self._wav.writeframes(samples.astype(np.int16).tobytes())
        self._samples_seen += len(samples)

    def close(self) -> None:
        if self._wav is not None:
            self._wav.close()
            self._wav = None


class AudioCaptureThread:
    """Records the mic plus system audio.

    System audio is the loopback of *every* output device, mixed: a call app
    picks its own device, and looping back only the Windows default recorded
    twenty minutes of silence on 2026-09-28 while the call played on a headset.
    In "apps" mode the selected processes are captured instead.
    """

    def __init__(
        self,
        session_dir: Path,
        sample_rate: int,
        audio_source: str = "all",
        filtered_app_names: list[str] | None = None,
        mic_only: bool = False,
    ):
        self._session_dir = session_dir
        self._sr = sample_rate
        self._audio_source = audio_source
        self._app_names = filtered_app_names or []
        self._mic_only = mic_only
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._loopbacks: list[tuple[str, LoopbackPartWriter]] = []
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
        t_mic.start()
        self._threads = [t_mic]

        if self._mic_only:
            return
        if self._audio_source == "apps":
            t_sys = threading.Thread(
                target=self._run_system, args=(sys_path,), name="audio-sys", daemon=True
            )
            t_sys.start()
            self._threads.append(t_sys)
        else:
            self._start_device_loopbacks()

    def stop(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=10)
        if self._loopbacks:
            self._finish_device_loopbacks()

    # --- internal ---

    def _open_wav(self, path: Path) -> wave.Wave_write:
        w = wave.open(str(path), "wb")
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(self._sr)
        return w

    def _run_mic(self, out_path: Path) -> None:
        """Capture the default mic via PortAudio.

        soundcard cannot open devices that report a plain WAVE_FORMAT_IEEE_FLOAT
        mix format (common on USB headsets) -- see mic_capture for details.
        """
        try:
            result = record_microphone(out_path, self._sr, self._stop)
        except MicrophoneNotAvailable as e:
            log.warning("No mic available: %s", e)
            self.warnings.append(f"Microphone not available: {e}")
            return
        except Exception:
            log.exception("Mic recording failed")
            self.warnings.append("Microphone recording aborted.")
            return

        self.mic_device_name = result.device_name
        if result.frames == 0:
            log.warning("Microphone %r produced no audio", result.device_name)
            self.warnings.append(
                f"Microphone {result.device_name!r} captured no audio — check that "
                "the right device is selected in Windows and is not muted."
            )

    def _start_device_loopbacks(self) -> None:
        try:
            speakers = list(sc.all_speakers())
        except Exception as e:
            log.warning("Cannot list output devices: %s", e)
            self.warnings.append("System audio (loopback) not available.")
            return
        if not speakers:
            log.warning("No output devices to loop back")
            self.warnings.append("System audio (loopback) not available.")
            return
        for i, speaker in enumerate(speakers):
            writer = LoopbackPartWriter(self._session_dir, i, self._sr)
            self._loopbacks.append((speaker.name, writer))
            t = threading.Thread(
                target=self._run_device_loopback,
                args=(speaker, writer),
                name=f"audio-loopback-{i}",
                daemon=True,
            )
            t.start()
            self._threads.append(t)

    def _run_device_loopback(self, speaker, writer: LoopbackPartWriter) -> None:
        try:
            loopback = sc.get_microphone(id=str(speaker.id), include_loopback=True)
            with loopback.recorder(samplerate=self._sr, channels=[0]) as rec:
                while not self._stop.is_set():
                    writer.write(_to_int16(rec.record(numframes=self._sr * CHUNK_SECONDS)))
        except Exception:
            # One device failing -- an unsupported format, or a headset dropping
            # off the bus -- must not take the others down with it.
            log.exception("Loopback of %r failed", speaker.name)
            writer.failed = True
        finally:
            writer.close()

    def _finish_device_loopbacks(self) -> None:
        contributed = [name for name, w in self._loopbacks if w.path is not None]
        try:
            mixed = mix_loopback_parts(self._session_dir)
        except Exception:
            log.exception("Mixing the loopback parts failed")
            self.warnings.append(
                "System audio could not be mixed; the per-device parts were kept."
            )
            return
        self.system_device_name = ", ".join(contributed) or None
        log.info("System audio captured from: %s", self.system_device_name or "no device")
        if mixed is None:
            if all(w.failed for _, w in self._loopbacks):
                self.warnings.append("System audio (loopback) not available.")
            else:
                self.warnings.append(
                    "No system audio was captured from any output device — "
                    "if this was a call, the other side is missing."
                )

    def _run_system(self, out_path: Path) -> None:
        """Per-app capture ("apps" mode), falling back to the default speaker."""
        if self._audio_source == "apps":
            try:
                record_app_loopback(out_path, self._app_names, self._sr, self._stop)
                return
            except ProcessLoopbackNotAvailable as e:
                log.warning("Process loopback unavailable, falling back to 'all': %s", e)
                self.warnings.append(
                    "App filter unavailable — recording full system audio instead."
                )
                # fall through to default loopback
        try:
            spk = sc.default_speaker()
            self.system_device_name = spk.name
            loopback_mic = sc.get_microphone(id=str(spk.name), include_loopback=True)
        except Exception as e:
            log.warning("No loopback available: %s", e)
            self.warnings.append("System audio (loopback) not available.")
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
            self.warnings.append("System audio recording aborted.")

    def _write_chunk(self, wav: wave.Wave_write, data: np.ndarray) -> None:
        """data is float32 from soundcard, shape (N, 1). Convert to int16."""
        wav.writeframes(_to_int16(data).tobytes())
