"""The finite journal prefix alone grants no target or restore authority.

Exercise actual load's bounded row-journal admission with the genuine compact
manifest decoder. Native acquisition/current authority and the full union are
covered separately; a sentinel stops before any payload callback or mutation.
"""
# Covers (for impacted-test selection):
#   src/blueprint_pipeline/control_plane_lane_experiment_restore_reconcile.py

import hashlib
import os
from types import SimpleNamespace

import pytest

from blueprint_pipeline import control_plane_lane_experiment_restore_reconcile as reconcile
from blueprint_pipeline.control_plane_reference_budget import ReferenceCollectionBudget
from tests.test_registered_experiment_manifest import document, encoded


class JournalAdmitted(Exception):
    pass


@pytest.mark.parametrize("extra", [False, True])
def test_full_4096_row_journal_requires_exact_finite_lookahead(tmp_path, monkeypatch, extra):
    binding, value = document()
    raw = encoded(value)
    selected = {"sha256": "sha256:" + hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}
    action = {"action_id": "d" * 32}
    stage_path = tmp_path / (".restore-" + action["action_id"])
    stage_path.mkdir()
    fd = os.open(stage_path, os.O_RDONLY | os.O_DIRECTORY)
    info = os.fstat(fd)
    started = {"sha256": "sha256:" + "e" * 64, "size_bytes": 1500}
    first = {"sha256": "sha256:" + "f" * 64, "size_bytes": 1024}
    budget = ReferenceCollectionBudget(values_limit=100000)
    queries = []
    latest = first
    phases = []

    def read_event(files, operation, observed_action, index, previous):
        nonlocal latest
        assert observed_action is action and previous == latest
        queries.append(index)
        if index == 4098:
            return ({"event_kind": "unknown_extra"}, first) if extra else None
        row = value["members"][index - 2]
        result = {
            "event_kind": "restore_member",
            "body": {
                "restore_started": started,
                "index": index - 2,
                "path": row[0],
                "sha256": row[4],
                "size_bytes": 0,
                "identity": {"dev": 1, "ino": index + 1, "type": "file"},
            },
        }
        latest = {"sha256": "sha256:" + f"{index:064x}", "size_bytes": 847}
        return result, latest

    def payload(*args, **kwargs):
        raise JournalAdmitted

    files = SimpleNamespace(
        budget=budget,
        parents={},
        phase=lambda name: phases.append(name),
        open=lambda *args, **kwargs: fd,
        read=lambda *args, **kwargs: (raw, None),
        payload=payload,
    )
    proof = {
        "stage_identity": {"dev": info.st_dev, "ino": info.st_ino, "type": "directory"},
        "stage_metadata": [getattr(info, name) for name in reconcile.recovery._STAT],
        "stage_manifest": selected,
    }
    monkeypatch.setattr(reconcile.recovery, "_read_event", read_event)
    try:
        error = "experiment_restore_event_limit" if extra else None
        with pytest.raises(ValueError, match=error) if extra else pytest.raises(JournalAdmitted):
            reconcile.load(
                files,
                tmp_path,
                tmp_path,
                None,
                None,
                action,
                binding,
                value["members"],
                started,
                ({"body": proof}, first),
            )
        assert queries == list(range(2, 4099))
        assert len(phases) == 257
        assert budget.failure is None
    finally:
        os.close(fd)
        budget.close()
