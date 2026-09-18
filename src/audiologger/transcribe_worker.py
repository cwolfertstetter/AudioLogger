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
import re
import shutil
import sys
import time
import traceback
import wave
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

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


def _speech_level(audio: "np.ndarray", sample_rate: int) -> float:
    """RMS of the frames carrying signal, ignoring the silence between them.

    A plain RMS is useless here: the mic track is ~90% silence, so it would
    measure mostly nothing.  Frame into 100 ms windows and keep those within
    20 dB of the loudest.  Returns 0.0 for empty or digitally silent audio.
    """
    a = np.asarray(audio, dtype=np.float32)
    if a.size == 0:
        return 0.0
    win = max(1, int(sample_rate * 0.1))
    if len(a) >= win:
        frames = a[: len(a) // win * win].reshape(-1, win).astype(np.float64)
        frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))
    else:
        frame_rms = np.array([np.sqrt(np.mean(a.astype(np.float64) ** 2))])
    loudest = float(frame_rms.max())
    if loudest <= 0.0:
        return 0.0
    speech = frame_rms[frame_rms >= loudest * 0.1]
    return float(np.sqrt(np.mean(speech ** 2))) if speech.size else 0.0


def normalize_for_transcription(
    audio: "np.ndarray",
    sample_rate: int = 16000,
    *,
    target_rms_dbfs: float = -20.0,
    ceiling_dbfs: float = -1.0,
    max_gain_db: float = 30.0,
) -> "np.ndarray":
    """Boost a quiet track to a usable speech level for Whisper.

    The mic track runs ~23 dB below the system track, and Whisper starts
    inventing text when it is fed near-silence.  The gain is derived from the
    speech in the track, not from the whole track: the mic is ~90% silence, so
    a plain RMS would measure mostly nothing and ask for absurd gain.

    Only ever boosts, never attenuates, never exceeds `ceiling_dbfs`, and never
    applies more than `max_gain_db` so a bare noise floor stays a noise floor.
    Returns the input unchanged when no gain is warranted.  Operates on the
    array handed to the model; the WAV on disk is not touched.
    """
    if audio is None or len(audio) == 0:
        return audio
    a = np.asarray(audio, dtype=np.float32)
    peak = float(np.max(np.abs(a)))
    if peak <= 0.0:
        return audio

    speech_rms = _speech_level(a, sample_rate)
    if speech_rms <= 0.0:
        return audio

    to_db = lambda x: 20.0 * np.log10(max(x, 1e-12))
    gain_db = min(target_rms_dbfs - to_db(speech_rms), ceiling_dbfs - to_db(peak))
    gain_db = float(np.clip(gain_db, 0.0, max_gain_db))
    if gain_db <= 0.0:
        return audio

    log.info("Boosting quiet audio by %.1f dB before transcription", gain_db)
    return np.clip(a * (10.0 ** (gain_db / 20.0)), -1.0, 1.0).astype(np.float32)


def _span(seg) -> tuple[float, float]:
    """Start/end of either a raw whisper dict or a Segment."""
    if isinstance(seg, Mapping):
        return float(seg.get("start", 0.0)), float(seg.get("end", 0.0))
    return float(seg.start), float(seg.end)


def _text_of(seg) -> str:
    return str(seg.get("text", "") if isinstance(seg, Mapping) else seg.text)


def drop_silent_segments(
    segments: list,
    audio: "np.ndarray",
    sample_rate: int = 16000,
    *,
    threshold_db: float = -25.0,
) -> list:
    """Throw away segments that sit on silence.

    Runs on the raw segments whisper returns, before alignment.  Alignment can
    move a segment's window seconds away from the speech it transcribes -- on
    three real recordings that cost 16 of 17 drops, including whole sentences.
    The raw VAD spans do cover their speech, so the gate reads those.

    Whisper does not stay quiet when there is nothing to transcribe -- it drops
    boilerplate from its training data onto silent stretches ("Vielen Dank.",
    "Untertitel 2.97", subtitle credits).  Pinning the language and normalizing
    do not help: they make it understand the right thing, not stay silent.

    So compare each segment against the audio underneath it.  Anything more
    than `threshold_db` below the track's own speech level is not speech.
    Returns the segments untouched when there is nothing to compare against,
    so a silent track never silently loses its whole transcript.

    The default threshold is calibrated on three real recordings: genuine
    speech reached -22.7 dB below the track's speech level, while invented
    segments started at -27.8 dB and ran down to -60 dB.  -25 dB sits in that
    gap with ~3 dB of margin either way.  It cannot catch boilerplate that
    lands on top of real audio ("Untertitel 2.97" sat at -6.7 dB); no level
    test can.
    """
    if not segments:
        return []
    a = np.asarray(audio, dtype=np.float32) if audio is not None else np.zeros(0, dtype=np.float32)
    if a.size == 0:
        return list(segments)
    level = _speech_level(a, sample_rate)
    if level <= 0.0:
        return list(segments)

    floor = level * (10.0 ** (threshold_db / 20.0))
    kept: list = []
    for seg in segments:
        seg_start, seg_end = _span(seg)
        start = max(0, int(seg_start * sample_rate))
        end = min(len(a), int(seg_end * sample_rate))
        if end <= start:
            log.info("Dropping segment at %.1fs (past end of audio): %r",
                     seg_start, _text_of(seg)[:40])
            continue
        window = a[start:end].astype(np.float64)
        if float(np.sqrt(np.mean(window ** 2))) >= floor:
            kept.append(seg)
        else:
            log.info("Dropping segment at %.1fs on silence: %r", seg_start, _text_of(seg)[:40])
    return kept


# Stock phrases from the subtitle files Whisper was trained on.  It falls back
# on them when it has nothing to transcribe but the audio is not quiet enough
# for the level gate to catch -- breathing, room noise, the tail of a word.
# Matched against the whole segment only, so "Vielen Dank für die Erklärung"
# stays.  The German entries are the ones that actually turn up here; the others
# were seen before the language was pinned and cost nothing to keep out.
_BOILERPLATE_EXACT = frozenset({
    "vielen dank",
    "vielen dank fürs zuschauen",
    "vielen dank für die aufmerksamkeit",
    "danke fürs zuschauen",
    "danke für's zuschauen",
    "untertitel im auftrag des zdf",
    "продолжение следует",
    "спасибо за просмотр",
    "takk for at du så med",
    "thanks for watching",
    "thank you for watching",
    "subscribe to my channel",
})

_BOILERPLATE_PATTERNS = (
    # "Untertitelung des ZDF, 2020" / "Untertitel im Auftrag des ..." / "Untertitel von ..."
    re.compile(r"^untertitel(ung)?\s+(des|der|von|im auftrag|by)\b"),
    # "Untertitelung. BR 2018" -- station after a full stop.  What every one of
    # these credits carries is a broadcast year, and talking about subtitles in
    # a meeting does not, so the year is what makes this safe to match.
    re.compile(r"^untertitel(ung)?\b.*\b(19|20)\d{2}\b"),
    # "Untertitel 2.97" -- a bare version number
    re.compile(r"^untertitel\s+[\d.,]+$"),
    re.compile(r"^teksting av\b"),
    re.compile(r"^subtitles?\s+(by|von)\b"),
    re.compile(r"\bamara\.org\b"),
)


def _normalize_for_matching(text: str) -> str:
    """Lowercase, collapse whitespace, strip surrounding punctuation."""
    collapsed = re.sub(r"\s+", " ", text.strip().lower())
    return collapsed.strip(" .!?-–—…\"'")


def drop_boilerplate_segments(segments: list[Segment]) -> list[Segment]:
    """Throw away segments that are nothing but Whisper's subtitle boilerplate.

    The level gate in drop_silent_segments() catches text invented onto
    silence.  This catches text invented onto real audio, where no level test
    can help: on one 71-minute recording the invented segments sat at -23.3 to
    -2.4 dB below the speech level and genuine content at -20.4 to +1.9 dB --
    overlapping ranges.  What separates them is the wording.

    Only whole-segment matches are dropped.  The trade is deliberate: a
    genuine, standalone "Vielen Dank." is lost too, which is worth it against
    the fifteen invented ones in that same recording.
    """
    kept: list[Segment] = []
    for seg in segments:
        normalized = _normalize_for_matching(seg.text)
        if not normalized:  # nothing but punctuation or ellipses
            log.info("Dropping empty segment at %.1fs: %r", seg.start, seg.text[:40])
            continue
        if normalized in _BOILERPLATE_EXACT or any(
            p.search(normalized) for p in _BOILERPLATE_PATTERNS
        ):
            log.info("Dropping boilerplate at %.1fs: %r", seg.start, seg.text[:40])
            continue
        kept.append(seg)
    return kept


@dataclass(frozen=True)
class TranscriptionResult:
    """Segments plus the language they were transcribed in."""
    segments: list[Segment]
    language: str | None


class WhisperXPipeline:
    """Lazily loads WhisperX + pyannote diarization once per process."""

    def __init__(self, model_size: str, device: str, compute_type: str,
                 diarization_enabled: bool, hf_token: str | None,
                 language: str = "de"):
        self.model_size = model_size
        self.language = language
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
        language: str | None = None,
    ) -> TranscriptionResult:
        if not audio_path.exists():
            return TranscriptionResult([], None)
        effective_model = model_size if model_size is not None else self.model_size
        whisper_model = self._get_model(effective_model)

        import whisperx
        audio = whisperx.load_audio(str(audio_path))

        # Guard against empty / near-empty audio (e.g. a stream whose device
        # dropped out mid-recording, leaving only a WAV header). whisperx uses
        # 16 kHz internally; anything under ~0.1 s can't be transcribed and would
        # crash the VAD with "'waveform' must be provided as a (channel, time)".
        MIN_SAMPLES = 1600  # 0.1 s at 16 kHz
        if getattr(audio, "size", 0) < MIN_SAMPLES:
            log.warning(
                "Audio %s has too few samples (%s) — skipping transcription of this stream",
                audio_path.name,
                getattr(audio, "size", 0),
            )
            return TranscriptionResult([], None)

        # Whisper invents text when fed near-silence, and the mic track runs
        # ~23 dB below the system track.  Lift it to a usable speech level
        # first; whisperx.load_audio always returns 16 kHz mono.  The WAV on
        # disk is untouched -- only the array handed to the model changes.
        audio = normalize_for_transcription(audio)

        kwargs: dict = {"batch_size": 16}
        if language:
            kwargs["language"] = language
        result = whisper_model.transcribe(audio, **kwargs)

        # Hold on to this now: whisperx.align below returns a fresh dict with
        # only "segments" and "word_segments", so the detected language would
        # otherwise be gone by the time the result is built -- and the meeting
        # path would silently fall back to the configured language instead of
        # inheriting the system track's.
        detected_language = result.get("language") or language

        # Gate on level here, while the timestamps still come from the VAD and
        # cover the speech they belong to.  Alignment below can move a window
        # seconds away from its words, and a gate reading the moved window
        # deletes real sentences.  Gating first also leaves less to align.
        result["segments"] = drop_silent_segments(result.get("segments", []), audio)

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

        # Second pass at Whisper's invented text: by wording, for the stock
        # subtitle phrases it drops onto real audio, where no level test reaches.
        segments = drop_boilerplate_segments(self._to_segments(result, diarize))
        return TranscriptionResult(segments, detected_language)

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
        language=cfg.language,
    ).segments

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
            language=cfg.language,
        ).segments

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

    # The system track carries continuous speech, so WhisperX detects its
    # language reliably.  The mic track is mostly silence -- the user listens
    # more than they talk -- and detection there is a coin flip: it reported
    # 'en' at 0.19 confidence on a German meeting and then hallucinated canned
    # English phrases onto the silent stretches.  So the mic inherits whatever
    # the system track resolved to, falling back to the configured language.
    sys_result = pipeline.transcribe(sys_wav, diarize=True)
    sys_segments = sys_result.segments
    if not pipeline.diarization_enabled:
        warnings.append("Diarization disabled or unavailable — all speakers labelled 'Others'.")

    # Only a track that actually produced speech has a language worth borrowing.
    # Recording alone leaves system.wav digitally silent, and whisper still names
    # a language for it -- "en" at 0.31 confidence -- which then got forced onto
    # the mic and turned German speech into English prose.
    inherited = sys_result.language if sys_result.segments else None
    mic_result = pipeline.transcribe(
        mic_wav, diarize=False, language=inherited or pipeline.language
    )
    mic_segments = _force_speaker(mic_result.segments, "Me")

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
        language=cfg.language,
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
