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
        '{"running": "session-1", "queued": ["session-2"]}'
    )
    q = TranscriptionJobQueue(state_dir=state_dir, spawner=MagicMock())
    status = q.status()
    assert status.running == "session-1"
    assert status.queued == ["session-2"]


def test_status_when_no_heartbeat(state_dir):
    q = TranscriptionJobQueue(state_dir=state_dir, spawner=MagicMock())
    status = q.status()
    assert status.running is None
    assert status.queued == []


def test_pending_file_created_on_first_enqueue(state_dir):
    q = TranscriptionJobQueue(state_dir=state_dir, spawner=MagicMock())
    assert not (state_dir / "pending.txt").exists()
    q.enqueue(Path("C:/recs/s1"))
    assert (state_dir / "pending.txt").exists()
