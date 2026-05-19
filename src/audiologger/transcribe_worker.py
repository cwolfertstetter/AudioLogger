"""Transcription worker subprocess.

Usage:
    python -m audiologger.transcribe_worker <state_dir> [--prewarm]

Reads <state_dir>/pending.txt, processes each session directory line-by-line,
writes transcript.md + transcript.json into each session dir. Stays warm for
worker_warm_seconds (default 600 s) after the last job to handle
quickly-following enqueues.

With --prewarm: eagerly loads whisper + dictation models before entering the
poll loop, so the first job does not pay model-load latency.
"""
import argparse
import json
import logging
import shutil
import sys
import time
import traceback
import wave
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path

from audiologger.audio_mix import append_wav
from audiologger.config import Config, load_config
from audiologger.paths import config_path
from audiologger.segment import Segment
from audiologger.transcript_merger import merge_segments, render_markdown


log = logging.getLogger("transcribe_worker")
DEFAULT_WARM_IDLE_SECONDS = 600
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


def _write_status(
    status_path: Path,
    running: str | None,
    queued: list[str],
    mode: str | None = None,
    warming: bool = False,
    last_finished: dict | None = None,
) -> None:
    status_path.write_text(
        json.dumps(
            {
                "running": running,
                "queued": queued,
                "mode": mode,
                "warming": warming,
                "last_finished": last_finished,
            },
            ensure_ascii=False,
        ),
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
        self._models: dict[str, object] = {}
        self._diarize = None

    def _get_model(self, model_size: str) -> object:
        """Load model on demand and cache by model_size."""
        if model_size not in self._models:
            log.info("Loading WhisperX model %s on %s/%s", model_size, self.device, self.compute_type)
            import whisperx
            self._models[model_size] = whisperx.load_model(
                model_size, self.device, compute_type=self.compute_type
            )
        return self._models[model_size]

    def _ensure_diarize(self) -> None:
        """Load diarization pipeline if not yet loaded."""
        if self._diarize is not None:
            return
        if not self.diarization_enabled:
            return
        if not self.hf_token:
            log.warning("Diarization enabled but no HuggingFace token; disabling diarization for this run")
            self.diarization_enabled = False
            return
        log.info("Loading pyannote diarization pipeline")
        from whisperx.diarize import DiarizationPipeline
        self._diarize = DiarizationPipeline(
            token=self.hf_token, device=self.device
        )

    def transcribe(
        self,
        audio_path: Path,
        *,
        diarize: bool,
        model_size: str | None = None,
        align: bool = True,
    ) -> list[Segment]:
        if not audio_path.exists():
            return []
        effective_model = model_size if model_size is not None else self.model_size
        whisper_model = self._get_model(effective_model)

        import whisperx
        audio = whisperx.load_audio(str(audio_path))
        result = whisper_model.transcribe(audio, batch_size=16)

        if align:
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

        if diarize and self.diarization_enabled:
            self._ensure_diarize()
            if self._diarize is not None:
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
                # pyannote returns "SPEAKER_00", "SPEAKER_01", ... -> "Speaker 1", "Speaker 2", ...
                if speaker_raw.startswith("SPEAKER_"):
                    num = int(speaker_raw.split("_")[1]) + 1
                    speaker = f"Speaker {num}"
                else:
                    speaker = "Others"
            else:
                speaker = "Others"
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
    return " + ".join(parts) if parts else "no audio"


def _read_mode(session_dir: Path) -> str:
    """Read mode.txt from session dir. Returns 'meeting' if missing."""
    mode_file = session_dir / "mode.txt"
    if not mode_file.exists():
        return "meeting"
    return mode_file.read_text(encoding="utf-8").strip() or "meeting"


def _process_session(session_dir: Path, pipeline: WhisperXPipeline, cfg: Config) -> dict:
    """Process one session. Returns a last_finished payload dict."""
    log.info("Processing session %s", session_dir)
    mode = _read_mode(session_dir)

    if mode == "dictation_extend":
        target_name, success = _process_dictation_extend_session(session_dir, pipeline, cfg)
        return {
            "session_id": target_name,
            "mode": "dictation",
            "was_extend": True,
            "success": success,
        }
    elif mode == "dictation":
        chunk_preview = _process_dictation_session(session_dir, pipeline, cfg)
        return {
            "session_id": session_dir.name,
            "mode": "dictation",
            "was_extend": False,
            "success": True,
            "chunk_preview": chunk_preview,
        }
    else:
        _process_meeting_session(session_dir, pipeline)
        return {
            "session_id": session_dir.name,
            "mode": "meeting",
            "was_extend": False,
            "success": True,
        }


def _process_dictation_session(session_dir: Path, pipeline: WhisperXPipeline, cfg: Config) -> str:
    """Fast mic-only transcription; plain text output; clipboard copy.

    Returns chunk_preview (first 140 chars of joined text).
    """
    mic_wav = session_dir / "mic.wav"

    segments = pipeline.transcribe(
        mic_wav,
        diarize=False,
        model_size=cfg.dictation_model,
        align=False,
    )

    joined_text = " ".join(s.text for s in segments)

    # Write plain text transcript
    (session_dir / "transcript.txt").write_text(joined_text, encoding="utf-8")

    # Write JSON
    raw = {
        "text": joined_text,
        "segments": [asdict(s) for s in segments],
        "mode": "dictation",
    }
    (session_dir / "transcript.json").write_text(
        json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Copy to clipboard
    try:
        import pyperclip
        pyperclip.copy(joined_text)
        log.info("Dictation text copied to clipboard (%d chars)", len(joined_text))
    except Exception:
        log.exception("Failed to copy dictation text to clipboard")

    log.info("Wrote transcript.txt and transcript.json for %s", session_dir.name)
    chunk_preview = joined_text[:140] + "..." if len(joined_text) > 140 else joined_text
    return chunk_preview


def _process_dictation_extend_session(
    session_dir: Path, pipeline: WhisperXPipeline, cfg: Config
) -> tuple[str, bool]:
    """Append a new chunk (audio + transcript) to the target dictation session.

    Returns (target_session_name, success).
    On success the temp session_dir is deleted.
    On failure the temp dir is left intact for investigation.
    """
    target_txt = session_dir / "target_session.txt"
    if not target_txt.exists():
        msg = "target_session.txt missing in dictation_extend session"
        log.error(msg)
        (session_dir / "job.log").write_text(msg, encoding="utf-8")
        return (session_dir.name, False)

    target_dir = Path(target_txt.read_text(encoding="utf-8").strip())
    if not target_dir.exists():
        msg = f"Target session directory no longer exists: {target_dir}"
        log.error(msg)
        (session_dir / "job.log").write_text(msg, encoding="utf-8")
        return (session_dir.name, False)

    mic_wav = session_dir / "mic.wav"

    try:
        # 1. Transcribe new chunk
        segments = pipeline.transcribe(
            mic_wav,
            diarize=False,
            model_size=cfg.dictation_model,
            align=False,
        )

        # 2. Offset timestamps by existing audio duration
        existing_duration = _wav_duration_seconds(target_dir / "mic.wav")
        offset_segments = [
            Segment(
                start=s.start + existing_duration,
                end=s.end + existing_duration,
                text=s.text,
                speaker=s.speaker,
            )
            for s in segments
        ]

        new_text = " ".join(s.text for s in segments)

        # 3. Append transcript text
        existing_txt_path = target_dir / "transcript.txt"
        if existing_txt_path.exists():
            existing_txt = existing_txt_path.read_text(encoding="utf-8")
            combined_txt = existing_txt.rstrip() + "\n\n" + new_text + "\n"
        else:
            combined_txt = new_text + "\n"
        existing_txt_path.write_text(combined_txt, encoding="utf-8")

        # 4. Update transcript.json
        existing_json_path = target_dir / "transcript.json"
        if existing_json_path.exists():
            try:
                existing_json = json.loads(existing_json_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                existing_json = {"segments": [], "mode": "dictation"}
        else:
            existing_json = {"segments": [], "mode": "dictation"}

        all_segments = existing_json.get("segments", []) + [asdict(s) for s in offset_segments]
        full_text = combined_txt.strip()
        existing_json["segments"] = all_segments
        existing_json["text"] = full_text
        existing_json["mode"] = "dictation"
        existing_json_path.write_text(
            json.dumps(existing_json, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # 5. Concatenate audio
        target_mic = target_dir / "mic.wav"
        append_wav(target_mic, mic_wav)

        # 6. Copy new chunk text to clipboard
        try:
            import pyperclip
            pyperclip.copy(new_text)
            log.info("Extend chunk text copied to clipboard (%d chars)", len(new_text))
        except Exception:
            log.exception("Failed to copy extend text to clipboard")

        log.info("Extend complete: appended chunk to %s", target_dir.name)

        # 7. Clean up temp session
        shutil.rmtree(session_dir)

        chunk_preview = new_text[:140] + "..." if len(new_text) > 140 else new_text
        return (target_dir.name, True)

    except Exception:
        log.error("dictation_extend failed for %s:\n%s", session_dir, traceback.format_exc())
        (session_dir / "job.log").write_text(traceback.format_exc(), encoding="utf-8")
        return (session_dir.name, False)


def _process_meeting_session(session_dir: Path, pipeline: WhisperXPipeline) -> None:
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
    mic_segments = _force_speaker(mic_segments, "Me")

    sys_segments = pipeline.transcribe(sys_wav, diarize=True)
    if not pipeline.diarization_enabled:
        warnings.append("Diarization disabled or unavailable — all speakers labelled 'Others'.")

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
        # whisperx 3.8+ uses speaker-diarization-community-1; older versions used 3.1.
        model_label += " + pyannote/speaker-diarization-community-1"

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
    parser = argparse.ArgumentParser(prog="audiologger.transcribe_worker")
    parser.add_argument("state_dir", help="Path to the worker state directory")
    parser.add_argument("--prewarm", action="store_true", help="Eagerly load models before entering poll loop")
    args = parser.parse_args(argv[1:])

    state_dir = Path(args.state_dir)
    _setup_logging(state_dir)

    cfg = load_config(config_path())
    warm_idle_seconds = getattr(cfg, "worker_warm_seconds", DEFAULT_WARM_IDLE_SECONDS)

    pipeline = WhisperXPipeline(
        model_size=cfg.whisper_model,
        device=cfg.device,
        compute_type=cfg.compute_type,
        diarization_enabled=cfg.diarization_enabled,
        hf_token=cfg.huggingface_token,
    )

    pending_path = state_dir / "pending.txt"
    status_path = state_dir / "worker_status.json"

    if args.prewarm:
        log.info("Pre-warming: loading whisper model %s", cfg.whisper_model)
        _write_status(status_path, running=None, queued=[], mode=None, warming=True)
        pipeline._get_model(cfg.whisper_model)
        dictation_model = getattr(cfg, "dictation_model", cfg.whisper_model)
        if dictation_model != cfg.whisper_model:
            log.info("Pre-warming: loading dictation model %s", dictation_model)
            pipeline._get_model(dictation_model)
        log.info("Pre-warming complete")
        _write_status(status_path, running=None, queued=[], mode=None, warming=False)

    last_job_finished = datetime.now()
    last_finished: dict | None = None
    while True:
        sessions = _read_pending(pending_path)
        if sessions:
            current = sessions[0]
            last_failed_path = state_dir / "last_failed.txt"
            current_mode = _read_mode(current)
            _write_status(
                status_path,
                running=current.name,
                queued=[s.name for s in sessions[1:]],
                mode=current_mode,
                last_finished=last_finished,
            )
            try:
                last_finished = _process_session(current, pipeline, cfg)
                # M5: clear last_failed on success
                last_failed_path.unlink(missing_ok=True)
            except Exception:
                log.error("Job failed for %s:\n%s", current, traceback.format_exc())
                (current / "job.log").write_text(traceback.format_exc(), encoding="utf-8")
                # M5: record the failed session name
                last_failed_path.write_text(current.name, encoding="utf-8")
                last_finished = {
                    "session_id": current.name,
                    "mode": current_mode,
                    "was_extend": current_mode == "dictation_extend",
                    "success": False,
                }
            finally:
                # Remove this session from pending whether success or failure
                remaining = _read_pending(pending_path)
                remaining = [p for p in remaining if p != current]
                _write_pending(pending_path, remaining)
            last_job_finished = datetime.now()
            _write_status(
                status_path,
                running=None,
                queued=[s.name for s in remaining],
                mode=None,
                last_finished=last_finished,
            )
            continue

        # Idle -- exit after warm window
        if datetime.now() - last_job_finished > timedelta(seconds=warm_idle_seconds):
            log.info("Idle %s s, exiting", warm_idle_seconds)
            _write_status(status_path, running=None, queued=[], mode=None, last_finished=last_finished)
            return 0

        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
