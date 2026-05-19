"""Transcription worker subprocess.

Usage:
    python -m audiologger.transcribe_worker <state_dir>

Reads <state_dir>/pending.txt, processes each session directory line-by-line,
writes transcript.md + transcript.json into each session dir. Stays warm 30s
after the last job to handle quickly-following enqueues.
"""
import json
import logging
import sys
import time
import traceback
import wave
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path

from audiologger.config import load_config
from audiologger.paths import config_path
from audiologger.segment import Segment
from audiologger.transcript_merger import merge_segments, render_markdown


log = logging.getLogger("transcribe_worker")
WARM_IDLE_SECONDS = 30
POLL_INTERVAL_SECONDS = 1.0


def _setup_logging(state_dir: Path) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(state_dir / "worker.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def _read_pending(pending_path: Path) -> list[Path]:
    if not pending_path.exists():
        return []
    lines = pending_path.read_text(encoding="utf-8").splitlines()
    return [Path(line.strip()) for line in lines if line.strip()]


def _write_pending(pending_path: Path, sessions: list[Path]) -> None:
    if not sessions:
        pending_path.write_text("", encoding="utf-8")
        return
    pending_path.write_text(
        "\n".join(str(p).replace("\\", "/") for p in sessions) + "\n",
        encoding="utf-8",
    )


def _write_status(status_path: Path, running: str | None, queued: list[str]) -> None:
    status_path.write_text(
        json.dumps({"running": running, "queued": queued}, ensure_ascii=False),
        encoding="utf-8",
    )


def _wav_duration_seconds(path: Path) -> float:
    if not path.exists():
        return 0.0
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / w.getframerate()


class WhisperXPipeline:
    """Lazily loads WhisperX + pyannote diarization once per process."""

    def __init__(self, model_size: str, device: str, compute_type: str,
                 diarization_enabled: bool, hf_token: str | None):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.diarization_enabled = diarization_enabled
        self.hf_token = hf_token
        self._whisper = None
        self._diarize = None

    def _load(self) -> None:
        if self._whisper is not None:
            return
        log.info("Loading WhisperX model %s on %s/%s", self.model_size, self.device, self.compute_type)
        import whisperx
        self._whisper = whisperx.load_model(
            self.model_size, self.device, compute_type=self.compute_type
        )
        if self.diarization_enabled:
            if not self.hf_token:
                log.warning("Diarization enabled but no HuggingFace token; disabling diarization for this run")
                self.diarization_enabled = False
            else:
                log.info("Loading pyannote diarization pipeline")
                self._diarize = whisperx.DiarizationPipeline(
                    use_auth_token=self.hf_token, device=self.device
                )

    def transcribe(self, audio_path: Path, *, diarize: bool) -> list[Segment]:
        if not audio_path.exists():
            return []
        self._load()
        import whisperx
        audio = whisperx.load_audio(str(audio_path))
        result = self._whisper.transcribe(audio, batch_size=16)
        # Align word-level (improves timestamp accuracy; multilingual handled by whisperx)
        try:
            model_a, metadata = whisperx.load_align_model(
                language_code=result["language"], device=self.device
            )
            result = whisperx.align(
                result["segments"], model_a, metadata, audio, self.device,
                return_char_alignments=False,
            )
        except Exception:
            log.exception("Alignment failed; using non-aligned segments")

        if diarize and self.diarization_enabled and self._diarize is not None:
            diarize_segments = self._diarize(audio)
            result = whisperx.assign_word_speakers(diarize_segments, result)

        return self._to_segments(result, diarize)

    def _to_segments(self, result: dict, diarize: bool) -> list[Segment]:
        segments: list[Segment] = []
        for seg in result.get("segments", []):
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", start))
            text = (seg.get("text") or "").strip()
            if not text:
                continue
            if diarize:
                speaker_raw = seg.get("speaker", "SPEAKER_00")
                # pyannote returns "SPEAKER_00", "SPEAKER_01", ... -> "Sprecher 1", "Sprecher 2", ...
                if speaker_raw.startswith("SPEAKER_"):
                    num = int(speaker_raw.split("_")[1]) + 1
                    speaker = f"Sprecher {num}"
                else:
                    speaker = "Andere"
            else:
                speaker = "Andere"
            segments.append(Segment(start=start, end=end, text=text, speaker=speaker))
        return segments


def _force_speaker(segments: list[Segment], speaker: str) -> list[Segment]:
    return [Segment(s.start, s.end, s.text, speaker) for s in segments]


def _source_label(session_dir: Path, mic_present: bool, sys_present: bool) -> str:
    parts = []
    if mic_present:
        parts.append("mic")
    if sys_present:
        parts.append("system (loopback)")
    return " + ".join(parts) if parts else "kein Audio"


def _process_session(session_dir: Path, pipeline: WhisperXPipeline) -> None:
    log.info("Processing session %s", session_dir)
    mic_wav = session_dir / "mic.wav"
    sys_wav = session_dir / "system.wav"
    warnings: list[str] = []

    # C1: prepend capture warnings from the controller
    capture_warnings_file = session_dir / "capture_warnings.txt"
    if capture_warnings_file.exists():
        raw = capture_warnings_file.read_text(encoding="utf-8")
        capture_warnings = [line for line in raw.splitlines() if line.strip()]
        warnings.extend(capture_warnings)

    mic_segments = pipeline.transcribe(mic_wav, diarize=False)
    mic_segments = _force_speaker(mic_segments, "Ich")

    sys_segments = pipeline.transcribe(sys_wav, diarize=True)
    if not pipeline.diarization_enabled:
        warnings.append("Diarization deaktiviert oder nicht verfügbar — Sprecher zusammengefasst als 'Andere'.")

    merged = merge_segments(mic_segments, sys_segments)

    duration = max(
        _wav_duration_seconds(mic_wav),
        _wav_duration_seconds(sys_wav),
    )
    recorded_at_str = session_dir.name.replace("_", " ").replace("-", ":", 2).replace("-", ":")
    # session_dir.name is YYYY-MM-DD_HH-MM-SS -- produce "YYYY-MM-DD HH:MM:SS"
    try:
        dt = datetime.strptime(session_dir.name, "%Y-%m-%d_%H-%M-%S")
        recorded_at_str = dt.strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        pass

    model_label = f"WhisperX {pipeline.model_size}"
    if pipeline.diarization_enabled:
        model_label += " + pyannote/speaker-diarization-3.1"

    md = render_markdown(
        merged,
        recorded_at=recorded_at_str,
        duration_seconds=int(duration),
        source_label=_source_label(session_dir, mic_wav.exists(), sys_wav.exists()),
        model_label=model_label,
        warnings=warnings,
    )
    (session_dir / "transcript.md").write_text(md, encoding="utf-8")

    raw = {
        "mic_segments": [asdict(s) for s in mic_segments],
        "system_segments": [asdict(s) for s in sys_segments],
        "merged": [asdict(s) for s in merged],
        "warnings": warnings,
    }
    (session_dir / "transcript.json").write_text(
        json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log.info("Wrote transcript.md and transcript.json for %s", session_dir.name)


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("Usage: python -m audiologger.transcribe_worker <state_dir>", file=sys.stderr)
        return 2
    state_dir = Path(argv[1])
    _setup_logging(state_dir)

    cfg = load_config(config_path())
    pipeline = WhisperXPipeline(
        model_size=cfg.whisper_model,
        device=cfg.device,
        compute_type=cfg.compute_type,
        diarization_enabled=cfg.diarization_enabled,
        hf_token=cfg.huggingface_token,
    )

    pending_path = state_dir / "pending.txt"
    status_path = state_dir / "worker_status.json"

    last_job_finished = datetime.now()
    while True:
        sessions = _read_pending(pending_path)
        if sessions:
            current = sessions[0]
            last_failed_path = state_dir / "last_failed.txt"
            _write_status(status_path, running=current.name, queued=[s.name for s in sessions[1:]])
            job_failed = False
            try:
                _process_session(current, pipeline)
                # M5: clear last_failed on success
                last_failed_path.unlink(missing_ok=True)
            except Exception:
                job_failed = True
                log.error("Job failed for %s:\n%s", current, traceback.format_exc())
                (current / "job.log").write_text(traceback.format_exc(), encoding="utf-8")
                # M5: record the failed session name
                last_failed_path.write_text(current.name, encoding="utf-8")
            finally:
                # Remove this session from pending whether success or failure
                remaining = _read_pending(pending_path)
                remaining = [p for p in remaining if p != current]
                _write_pending(pending_path, remaining)
            last_job_finished = datetime.now()
            _write_status(status_path, running=None, queued=[s.name for s in remaining])
            continue

        # Idle -- exit after warm window
        if datetime.now() - last_job_finished > timedelta(seconds=WARM_IDLE_SECONDS):
            log.info("Idle %s s, exiting", WARM_IDLE_SECONDS)
            _write_status(status_path, running=None, queued=[])
            return 0

        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
