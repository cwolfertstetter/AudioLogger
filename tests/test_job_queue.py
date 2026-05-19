from pathlib import Path
from unittest.mock import MagicMock

import pytest

from audiologger.job_queue import TranscriptionJobQueue


@pytest.fixture
def state_dir(tmp_path):
    d = tmp_path / "state"
    d.mkdir()
    return d


def test_enqueue_appends_to_pending(state_dir):
    q = TranscriptionJobQueue(state_dir=state_dir, spawner=MagicMock())
    q.enqueue(Path("C:/recs/session-1"))
    q.enqueue(Path("C:/recs/session-2"))
    pending = (state_dir / "pending.txt").read_text().splitlines()
    assert pending == ["C:/recs/session-1", "C:/recs/session-2"]


def test_enqueue_spawns_worker_when_none_running(state_dir):
    spawner = MagicMock()
    spawner.return_value.poll.return_value = None  # alive
    q = TranscriptionJobQueue(state_dir=state_dir, spawner=spawner)
    q.enqueue(Path("C:/recs/s1"))
    spawner.assert_called_once()


def test_enqueue_does_not_spawn_if_worker_alive(state_dir):
    spawner = MagicMock()
    proc = MagicMock()
    proc.poll.return_value = None  # alive
    spawner.return_value = proc
    q = TranscriptionJobQueue(state_dir=state_dir, spawner=spawner)
    q.enqueue(Path("C:/recs/s1"))
    q.enqueue(Path("C:/recs/s2"))
    assert spawner.call_count == 1


def test_enqueue_respawns_if_worker_exited(state_dir):
    spawner = MagicMock()
    dead_proc = MagicMock()
    dead_proc.poll.return_value = 0  # exited
    alive_proc = MagicMock()
    alive_proc.poll.return_value = None
    spawner.side_effect = [dead_proc, alive_proc]
    q = TranscriptionJobQueue(state_dir=state_dir, spawner=spawner)
    q.enqueue(Path("C:/recs/s1"))
    q.enqueue(Path("C:/recs/s2"))
    assert spawner.call_count == 2


def test_status_reads_heartbeat(state_dir):
    (state_dir / "worker_status.json").write_text(
        '{"running": "session-1", "queued": ["session-2"], "mode": "meeting"}'
    )
    q = TranscriptionJobQueue(state_dir=state_dir, spawner=MagicMock())
    status = q.status()
    assert status.running == "session-1"
    assert status.queued == ["session-2"]
    assert status.mode == "meeting"


def test_status_when_no_heartbeat(state_dir):
    q = TranscriptionJobQueue(state_dir=state_dir, spawner=MagicMock())
    status = q.status()
    assert status.running is None
    assert status.queued == []
    assert status.mode is None


def test_pending_file_created_on_first_enqueue(state_dir):
    q = TranscriptionJobQueue(state_dir=state_dir, spawner=MagicMock())
    assert not (state_dir / "pending.txt").exists()
    q.enqueue(Path("C:/recs/s1"))
    assert (state_dir / "pending.txt").exists()


# --- M5: last_failed.txt reflected in JobStatus.last_failed ---------------

def test_status_last_failed_when_file_present(state_dir):
    (state_dir / "last_failed.txt").write_text("2026-05-18_14-32-15", encoding="utf-8")
    q = TranscriptionJobQueue(state_dir=state_dir, spawner=MagicMock())
    status = q.status()
    assert status.last_failed == "2026-05-18_14-32-15"


def test_status_last_failed_none_when_file_absent(state_dir):
    q = TranscriptionJobQueue(state_dir=state_dir, spawner=MagicMock())
    status = q.status()
    assert status.last_failed is None


def test_status_last_failed_none_when_file_empty(state_dir):
    (state_dir / "last_failed.txt").write_text("", encoding="utf-8")
    q = TranscriptionJobQueue(state_dir=state_dir, spawner=MagicMock())
    status = q.status()
    assert status.last_failed is None


def test_status_mode_default_none_when_missing_key(state_dir):
    """If worker_status.json has no 'mode' key, status.mode is None."""
    (state_dir / "worker_status.json").write_text(
        '{"running": "session-1", "queued": []}', encoding="utf-8"
    )
    q = TranscriptionJobQueue(state_dir=state_dir, spawner=MagicMock())
    status = q.status()
    assert status.mode is None


# --- Pre-warm tests -------------------------------------------------------

def test_prewarm_spawns_worker_when_none_running(state_dir):
    spawner = MagicMock()
    spawner.return_value.poll.return_value = None  # alive after spawn
    q = TranscriptionJobQueue(state_dir=state_dir, spawner=spawner)
    q.prewarm()
    spawner.assert_called_once_with(state_dir, prewarm=True)


def test_prewarm_noop_when_worker_alive(state_dir):
    spawner = MagicMock()
    proc = MagicMock()
    proc.poll.return_value = None  # stays alive
    spawner.return_value = proc
    q = TranscriptionJobQueue(state_dir=state_dir, spawner=spawner)
    q.prewarm()
    q.prewarm()  # second call — worker already alive
    assert spawner.call_count == 1


def test_enqueue_after_prewarm_reuses_worker(state_dir):
    spawner = MagicMock()
    proc = MagicMock()
    proc.poll.return_value = None  # stays alive
    spawner.return_value = proc
    q = TranscriptionJobQueue(state_dir=state_dir, spawner=spawner)
    q.prewarm()
    q.enqueue(Path("C:/recs/s1"))  # worker still alive — no re-spawn
    assert spawner.call_count == 1


def test_status_warming_when_flag_present(state_dir):
    (state_dir / "worker_status.json").write_text(
        '{"running": null, "queued": [], "mode": null, "warming": true}', encoding="utf-8"
    )
    q = TranscriptionJobQueue(state_dir=state_dir, spawner=MagicMock())
    assert q.status().warming is True


def test_status_warming_default_false(state_dir):
    (state_dir / "worker_status.json").write_text(
        '{"running": null, "queued": [], "mode": null}', encoding="utf-8"
    )
    q = TranscriptionJobQueue(state_dir=state_dir, spawner=MagicMock())
    assert q.status().warming is False
