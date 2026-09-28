"""Handoff state never disappears during failed publication or races."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest

from blueprint_pipeline import task_evaluation_policy_canary_handoff_state as module


def test_failed_state_replacement_preserves_previous_checkpoint(tmp_path: Path, monkeypatch):
    path = tmp_path / "state.json"
    module.seal_state(path, {"status": "canary_profile_published"})
    before = path.read_bytes()
    def fail(source, target):
        assert path.read_bytes() == before
        raise OSError("injected replacement failure")
    monkeypatch.setattr(module.os, "replace", fail)
    with pytest.raises(OSError, match="replacement failure"):
        module.seal_state(path, {"status": "canary_launch_submitted"})
    assert path.read_bytes() == before
    assert not list(tmp_path.glob(".handoff-*"))


def test_concurrent_handoffs_share_one_stable_lock(tmp_path: Path):
    entered, released, second_started = Event(), Event(), Event()
    calls = []
    @module.serialized_handoff
    def handoff(*, state_root, index):
        calls.append(index)
        if index == 1:
            entered.set()
            assert released.wait(5)
        return index
    def second():
        second_started.set()
        return handoff(state_root=tmp_path, index=2)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(handoff, state_root=tmp_path, index=1)
        assert entered.wait(5)
        other = pool.submit(second)
        assert second_started.wait(5)
        assert calls == [1] and not other.done()
        released.set()
        assert first.result(timeout=5) == 1
        assert other.result(timeout=5) == 2
    assert calls == [1, 2]
