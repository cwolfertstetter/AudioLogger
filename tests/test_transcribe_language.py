"""Language selection for transcription.

WhisperX auto-detects the language from the first 30 s of each file.  The mic
track is mostly silence (the user listens more than they talk), so detection
there is a coin flip -- it reported 'en' at 0.19 confidence on a German
recording and then hallucinated canned English/Russian phrases onto the silent
stretches.  The system track carries continuous speech, so its detection is
reliable and is reused for the mic track.
"""
import wave
from pathlib import Path

import numpy as np
import pytest

from audiologger.config import Config
from audiologger.transcribe_worker import (
    Segment,
    WhisperXPipeline,
    _process_meeting_session,
)


class FakeModel:
    def __init__(self, detected="de"):
        self.detected = detected
        self.calls: list[dict] = []

    def transcribe(self, audio, **kw):
        self.calls.append(kw)
        return {
            "language": kw.get("language") or self.detected,
            "segments": [{"start": 0.0, "end": 1.0, "text": "hallo"}],
        }


@pytest.fixture
def pipeline(monkeypatch):
    import whisperx

    monkeypatch.setattr(whisperx, "load_audio", lambda p: np.zeros(16000, dtype=np.float32))
    p = WhisperXPipeline(
        model_size="large-v3", device="cpu", compute_type="int8",
        diarization_enabled=False, hf_token=None, language="de",
    )
    return p


def test_transcribe_forwards_the_requested_language_to_the_model(tmp_path, pipeline):
    model = FakeModel()
    pipeline._models["large-v3"] = model
    audio = tmp_path / "mic.wav"
    audio.touch()

    pipeline.transcribe(audio, diarize=False, align=False, language="de")

    assert model.calls[0].get("language") == "de"


def test_transcribe_auto_detects_when_no_language_is_given(tmp_path, pipeline):
    model = FakeModel(detected="en")
    pipeline._models["large-v3"] = model
    audio = tmp_path / "mic.wav"
    audio.touch()

    result = pipeline.transcribe(audio, diarize=False, align=False)

    assert model.calls[0].get("language") is None
    assert result.language == "en"


def test_transcribe_reports_the_language_that_was_used(tmp_path, pipeline):
    pipeline._models["large-v3"] = FakeModel()
    audio = tmp_path / "mic.wav"
    audio.touch()

    result = pipeline.transcribe(audio, diarize=False, align=False, language="de")

    assert result.language == "de"
    assert [s.text for s in result.segments] == ["hallo"]


# --- meeting session: mic inherits the system track's language ---------------

def write_wav(path: Path, seconds: float = 1.0, sr: int = 48000) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(np.zeros(int(sr * seconds), dtype=np.int16).tobytes())


class FakePipeline:
    """Records (filename, language) per transcribe call, in call order."""

    model_size = "large-v3"
    diarization_enabled = False

    def __init__(self, detected="de", language="de"):
        self._detected = detected
        self.language = language
        self.calls: list[tuple[str, str | None]] = []

    def transcribe(self, audio_path, *, diarize, model_size=None, align=True, language=None):
        from audiologger.transcribe_worker import TranscriptionResult

        self.calls.append((Path(audio_path).name, language))
        return TranscriptionResult(
            segments=[Segment(start=0.0, end=1.0, text="hallo", speaker="Others")],
            language=language or self._detected,
        )


@pytest.fixture
def session(tmp_path):
    d = tmp_path / "2026-09-18_10-40-47"
    d.mkdir()
    write_wav(d / "mic.wav")
    write_wav(d / "system.wav")
    return d


def test_meeting_transcribes_system_first_and_reuses_its_language_for_the_mic(session):
    pipe = FakePipeline(detected="de")

    _process_meeting_session(session, pipe)

    assert pipe.calls[0][0] == "system.wav", "system track must run first"
    assert pipe.calls[0][1] is None, "system track auto-detects"
    assert pipe.calls[1] == ("mic.wav", "de"), "mic inherits the detected language"


def test_meeting_falls_back_to_the_configured_language_when_detection_yields_nothing(session):
    pipe = FakePipeline(detected=None, language="de")

    _process_meeting_session(session, pipe)

    assert pipe.calls[1] == ("mic.wav", "de")


def test_config_language_defaults_to_german():
    assert Config().language == "de"


# --- the detected language must survive alignment -----------------------------
# whisperx.align returns {"segments", "word_segments"} and REPLACES the result
# dict, so the "language" key whisper put there is gone by the time the result
# is built. Without this, the meeting path always fell back to the configured
# language and "inherit from the system track" never actually did anything.

def test_reports_the_detected_language_even_though_alignment_replaces_the_result(
    tmp_path, monkeypatch, pipeline
):
    import whisperx

    pipeline._models["large-v3"] = FakeModel(detected="en")
    monkeypatch.setattr(whisperx, "load_align_model", lambda **kw: (object(), {}))
    monkeypatch.setattr(
        whisperx, "align",
        lambda segments, *a, **kw: {"segments": list(segments), "word_segments": []},
    )
    audio = tmp_path / "system.wav"
    audio.touch()

    result = pipeline.transcribe(audio, diarize=False, align=True)

    assert result.language == "en"


# --- a silent system track has no language to lend ---------------------------
# Recording alone leaves system.wav digitally silent. Whisper still names a
# language for it -- "en" at 0.31 confidence -- and since "en" is truthy it was
# passed on to the mic track, so German speech came back as English prose.
# Nothing was transcribed from that track, so it has nothing to say about
# language either.

class SilentSystemPipeline(FakePipeline):
    """System track yields no segments at all, but still names a language."""

    def transcribe(self, audio_path, *, diarize, model_size=None, align=True, language=None):
        from audiologger.transcribe_worker import TranscriptionResult

        self.calls.append((Path(audio_path).name, language))
        if Path(audio_path).name == "system.wav":
            return TranscriptionResult(segments=[], language="en")
        return TranscriptionResult(
            segments=[Segment(start=0.0, end=1.0, text="hallo", speaker="Others")],
            language=language or self._detected,
        )


def test_a_system_track_without_speech_does_not_dictate_the_mic_language(session):
    pipe = SilentSystemPipeline(detected="de", language="de")

    _process_meeting_session(session, pipe)

    assert pipe.calls[1] == ("mic.wav", "de"), (
        "mic must fall back to the configured language, not the guess made on silence"
    )
