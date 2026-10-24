"""The worker's job loop: what it leaves in pending.txt and worker_status.json."""
import json
from pathlib import Path

import pytest

from audiologger import transcribe_worker as tw
from audiologger.config import Config, save_config
from audiologger.job_queue import EXITING_FILE


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    save_config(tmp_path / "appdata" / "AudioLogger" / "config.yaml",
                Config(output_dir=tmp_path / "recs", worker_warm_seconds=0))
    monkeypatch.setattr(tw, "_setup_logging", lambda d: None)
    monkeypatch.setattr(tw, "POLL_INTERVAL_SECONDS", 0.0)
    state = tmp_path / "state"
    state.mkdir()
    return state


def queue(state: Path, *sessions: Path) -> None:
    (state / "pending.txt").write_text(
        "".join(s.as_posix() + "\n" for s in sessions), encoding="utf-8")


def status(state: Path) -> dict:
    return json.loads((state / "worker_status.json").read_text(encoding="utf-8"))


def test_a_session_deleted_while_queued_fails_without_killing_the_worker(env, tmp_path, monkeypatch):
    gone = tmp_path / "recs" / "2026-10-01_10-00-00"
    kept = tmp_path / "recs" / "2026-10-01_11-00-00"
    kept.mkdir(parents=True)
    processed = []

    def process(session, pipeline, cfg):
        if not session.exists():
            raise FileNotFoundError(session)
        processed.append(session)
        return {"session_id": session.name, "mode": "meeting", "was_extend": False, "success": True}

    monkeypatch.setattr(tw, "_process_session", process)
    queue(env, gone, kept)

    assert tw.main(["tw", str(env)]) == 0

    assert processed == [kept], "the job after the missing one still runs"
    s = status(env)
    assert s["running"] is None
    assert s["last_finished"]["session_id"] == kept.name


def test_a_re_transcription_queued_while_the_session_runs_is_kept(env, tmp_path, monkeypatch):
    session = tmp_path / "recs" / "2026-10-01_10-00-00"
    session.mkdir(parents=True)
    runs = []

    def process(s, pipeline, cfg):
        runs.append(s)
        if len(runs) == 1:
            # "Re-transcribe last recording" clicked while this job runs
            with (env / "pending.txt").open("a", encoding="utf-8") as f:
                f.write(s.as_posix() + "\n")
        return {"session_id": s.name, "mode": "meeting", "was_extend": False, "success": True}

    monkeypatch.setattr(tw, "_process_session", process)
    queue(env, session)

    tw.main(["tw", str(env)])

    assert runs == [session, session]


def test_every_finished_job_carries_its_own_time(env, tmp_path, monkeypatch):
    """Two runs of the same session must look different to the tray's toast check."""
    session = tmp_path / "recs" / "2026-10-01_10-00-00"
    session.mkdir(parents=True)
    monkeypatch.setattr(tw, "_process_session", lambda s, p, c: {
        "session_id": s.name, "mode": "meeting", "was_extend": False, "success": True})
    queue(env, session)

    tw.main(["tw", str(env)])

    assert status(env)["last_finished"]["finished_at"]


def test_an_idle_worker_marks_itself_as_exiting(env):
    tw.main(["tw", str(env)])

    assert (env / EXITING_FILE).exists()


# --- dictation extend ------------------------------------------------------------

def _wav(path: Path, n: int, value: int = 100) -> None:
    import wave
    import numpy as np
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(np.full(n, value, dtype=np.int16).tobytes())


def _frames(path: Path) -> int:
    import wave
    with wave.open(str(path), "rb") as w:
        return w.getnframes()


class OneLinePipeline:
    def transcribe(self, path, **kw):
        return tw.TranscriptionResult([tw.Segment(0.0, 1.0, "Noch ein Punkt.", "Others")], "de")


@pytest.fixture
def extend_job(tmp_path):
    recs = tmp_path / "recs"
    target = recs / "2026-10-01_10-00-00"
    chunk = recs / "2026-10-01_10-05-00"
    target.mkdir(parents=True)
    chunk.mkdir()
    _wav(target / "mic.wav", 16000)
    (target / "transcript.txt").write_text("Erster Punkt.\n", encoding="utf-8")
    (target / "transcript.json").write_text(json.dumps(
        {"segments": [], "text": "Erster Punkt.", "mode": "dictation"}), encoding="utf-8")
    _wav(chunk / "mic.wav", 8000)
    (chunk / "mode.txt").write_text("dictation_extend", encoding="utf-8")
    (chunk / "target_session.txt").write_text(target.as_posix(), encoding="utf-8")
    return target, chunk


def test_an_extend_that_fails_leaves_the_target_untouched(extend_job, monkeypatch):
    target, chunk = extend_job

    def broken_append(*a, **kw):
        raise ValueError("sample rates differ")

    monkeypatch.setattr(tw, "append_wav", broken_append)

    _, ok = tw._process_dictation_extend_session(chunk, OneLinePipeline(), Config())

    assert not ok
    assert (target / "transcript.txt").read_text(encoding="utf-8") == "Erster Punkt.\n"
    assert _frames(target / "mic.wav") == 16000


def test_an_extend_interrupted_after_commit_is_finished_not_repeated(extend_job, monkeypatch):
    target, chunk = extend_job
    real_replace = tw.os.replace
    calls = []

    def killed_after_first_move(src, dst):
        calls.append(dst)
        if len(calls) == 2:
            raise KeyboardInterrupt  # stands in for TerminateProcess
        real_replace(src, dst)

    monkeypatch.setattr(tw.os, "replace", killed_after_first_move)
    with pytest.raises(KeyboardInterrupt):
        tw._process_dictation_extend_session(chunk, OneLinePipeline(), Config())
    monkeypatch.setattr(tw.os, "replace", real_replace)

    # The job is still queued; the next worker runs it again.
    name, ok = tw._process_dictation_extend_session(chunk, OneLinePipeline(), Config())

    assert ok and name == target.name
    assert (target / "transcript.txt").read_text(encoding="utf-8").count("Noch ein Punkt.") == 1
    assert _frames(target / "mic.wav") == 24000
    assert not chunk.exists()
