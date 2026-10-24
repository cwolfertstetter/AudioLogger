"""Speaker labels are a bonus, never a reason to lose the transcript.

A rejected HuggingFace token or pyannote running out of GPU memory used to
raise straight out of transcribe(): no transcript.md, and every later meeting
in the same worker failed the same way.
"""
import numpy as np
import pytest

from audiologger.transcribe_worker import WhisperXPipeline


class FakeModel:
    def transcribe(self, audio, **kw):
        return {
            "language": "de",
            "segments": [{"start": 0.0, "end": 1.0, "text": "Wir fangen an."}],
        }


@pytest.fixture
def pipeline(monkeypatch):
    import whisperx

    rng = np.random.default_rng(0)
    speech = (rng.standard_normal(16000) * 0.2).astype(np.float32)
    monkeypatch.setattr(whisperx, "load_audio", lambda p: speech)
    p = WhisperXPipeline(
        model_size="large-v3", device="cpu", compute_type="int8",
        diarization_enabled=True, hf_token="hf_x", language="de",
    )
    p._models["large-v3"] = FakeModel()
    return p


@pytest.fixture
def audio(tmp_path):
    a = tmp_path / "system.wav"
    a.touch()
    return a


def test_a_pipeline_that_cannot_load_still_yields_the_transcript(pipeline, audio, monkeypatch):
    def rejected_token():
        raise RuntimeError("401 Client Error: gated repo")

    monkeypatch.setattr(pipeline, "_ensure_diarize", rejected_token)

    result = pipeline.transcribe(audio, diarize=True, align=False)

    assert [s.text for s in result.segments] == ["Wir fangen an."]
    assert [s.speaker for s in result.segments] == ["Others"]
    assert result.diarized is False


def test_diarization_running_out_of_memory_still_yields_the_transcript(pipeline, audio):
    def out_of_memory(audio):
        raise RuntimeError("CUDA out of memory")

    pipeline._diarize = out_of_memory

    result = pipeline.transcribe(audio, diarize=True, align=False)

    assert [s.speaker for s in result.segments] == ["Others"]
    assert result.diarized is False


def test_without_a_token_speakers_are_others_as_the_header_says(pipeline, audio):
    pipeline.hf_token = None

    result = pipeline.transcribe(audio, diarize=True, align=False)

    assert [s.speaker for s in result.segments] == ["Others"]
    assert result.diarized is False


def test_a_segment_pyannote_gave_no_speaker_is_not_pinned_on_speaker_1(pipeline):
    result = {"segments": [
        {"start": 0.0, "end": 1.0, "text": "a", "speaker": "SPEAKER_01"},
        {"start": 1.0, "end": 2.0, "text": "b"},
    ]}

    segments = pipeline._to_segments(result, diarize=True)

    assert [s.speaker for s in segments] == ["Speaker 2", "Others"]
