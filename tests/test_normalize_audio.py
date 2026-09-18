"""Level normalization before transcription.

The mic track sits ~23 dB below the system track (RMS -50 vs -27 dBFS), which
is where Whisper starts filling silence with canned phrases.  Boost quiet
tracks to a usable speech level before handing them to the model; the audio
files on disk stay untouched.

The gain has to be derived from the speech in the track, not from the whole
track: the mic is ~90% silence, so a plain RMS would be dominated by nothing.
"""
import numpy as np
import pytest

from audiologger.transcribe_worker import normalize_for_transcription

SR = 16000


def tone(seconds: float, dbfs: float, sr: int = SR) -> np.ndarray:
    """Sine at a given RMS level (a sine's peak is 3 dB above its RMS)."""
    amp = 10 ** (dbfs / 20) * np.sqrt(2)
    t = np.arange(int(sr * seconds)) / sr
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def rms_dbfs(a: np.ndarray) -> float:
    return 20 * np.log10(max(float(np.sqrt(np.mean(a.astype(np.float64) ** 2))), 1e-12))


def peak_dbfs(a: np.ndarray) -> float:
    return 20 * np.log10(max(float(np.max(np.abs(a))), 1e-12))


def test_quiet_audio_is_boosted_towards_the_target_level():
    quiet = tone(2.0, -45.0)

    out = normalize_for_transcription(quiet, SR, target_rms_dbfs=-20.0)

    assert rms_dbfs(out) == pytest.approx(-20.0, abs=1.5)


def test_audio_already_at_a_good_level_is_left_alone():
    loud = tone(2.0, -12.0)

    out = normalize_for_transcription(loud, SR, target_rms_dbfs=-20.0)

    assert np.array_equal(out, loud), "must not attenuate, only boost"


def test_output_never_exceeds_the_ceiling():
    quiet = tone(2.0, -45.0)

    out = normalize_for_transcription(quiet, SR, target_rms_dbfs=-3.0, ceiling_dbfs=-1.0)

    assert peak_dbfs(out) <= -1.0 + 1e-6


def test_digital_silence_is_returned_unchanged():
    silence = np.zeros(SR, dtype=np.float32)

    out = normalize_for_transcription(silence, SR)

    assert np.array_equal(out, silence)


def test_empty_audio_is_returned_unchanged():
    empty = np.zeros(0, dtype=np.float32)

    out = normalize_for_transcription(empty, SR)

    assert len(out) == 0


def test_gain_is_capped_so_a_noise_floor_is_not_blown_up():
    almost_nothing = tone(2.0, -90.0)

    out = normalize_for_transcription(almost_nothing, SR, target_rms_dbfs=-20.0, max_gain_db=30.0)

    assert rms_dbfs(out) == pytest.approx(-60.0, abs=1.5), "at most +30 dB"


def test_gain_is_measured_on_the_speech_not_on_the_silence():
    """A mic track is mostly silence; padding must not change the gain."""
    speech = tone(2.0, -40.0)
    padded = np.concatenate([speech, np.zeros(18 * SR, dtype=np.float32)])

    only = normalize_for_transcription(speech, SR, target_rms_dbfs=-20.0)
    with_silence = normalize_for_transcription(padded, SR, target_rms_dbfs=-20.0)

    assert rms_dbfs(with_silence[: len(speech)]) == pytest.approx(rms_dbfs(only), abs=0.5)


# --- wiring ------------------------------------------------------------------

class RecordingModel:
    """Captures the audio array the pipeline hands to Whisper."""

    def __init__(self):
        self.audio = None

    def transcribe(self, audio, **kw):
        self.audio = audio
        return {"language": "de", "segments": []}


def test_transcribe_boosts_quiet_audio_before_handing_it_to_the_model(tmp_path, monkeypatch):
    import whisperx
    from audiologger.transcribe_worker import WhisperXPipeline

    quiet = tone(2.0, -45.0)
    monkeypatch.setattr(whisperx, "load_audio", lambda p: quiet)

    pipe = WhisperXPipeline(
        model_size="large-v3", device="cpu", compute_type="int8",
        diarization_enabled=False, hf_token=None, language="de",
    )
    model = RecordingModel()
    pipe._models["large-v3"] = model
    audio_file = tmp_path / "mic.wav"
    audio_file.touch()

    pipe.transcribe(audio_file, diarize=False, align=False, language="de")

    assert model.audio is not None
    assert rms_dbfs(model.audio) > rms_dbfs(quiet) + 10, "quiet track must be boosted"


def test_transcribe_leaves_a_well_levelled_track_alone(tmp_path, monkeypatch):
    import whisperx
    from audiologger.transcribe_worker import WhisperXPipeline

    loud = tone(2.0, -12.0)
    monkeypatch.setattr(whisperx, "load_audio", lambda p: loud)

    pipe = WhisperXPipeline(
        model_size="large-v3", device="cpu", compute_type="int8",
        diarization_enabled=False, hf_token=None, language="de",
    )
    model = RecordingModel()
    pipe._models["large-v3"] = model
    audio_file = tmp_path / "system.wav"
    audio_file.touch()

    pipe.transcribe(audio_file, diarize=False, align=False, language="de")

    assert np.array_equal(model.audio, loud)
