"""A phase job's identity is its digests; the paths its bytes live at are provenance."""
from __future__ import annotations

import json

import pytest

from blueprint_pipeline.task_evaluation_sam31_phase_queue import enqueue_sam31_phase


def _ref(path, content: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    import hashlib
    return {"path": str(path), "sha256": "sha256:" + hashlib.sha256(content).hexdigest(), "size_bytes": len(content)}


def _enqueue(tmp_path, *, plan, inputs, parent="prep-1"):
    return enqueue_sam31_phase(queue_root=tmp_path / "queue", parent_preparation_id=parent,
        parent_request_digest="sha256:" + "a" * 64, expected_source_commit="b" * 40,
        plan_ref=plan, phase="sam31_inputs", inputs=inputs)


def test_same_digests_at_other_paths_are_the_same_job(tmp_path):
    """2026-09-13 scene 840938: the look-ahead replay re-drove an adopted prefix from re-rooted
    references and refused its own copied production job as a job_identity_conflict."""
    plan = _ref(tmp_path / "prod" / "plan.json", b'{"plan": 1}')
    inputs = {"camera_contract": _ref(tmp_path / "prod" / "cameras.json", b"cams"),
              "standard_splat": _ref(tmp_path / "prod" / "splat.ply", b"splat")}
    first = _enqueue(tmp_path, plan=plan, inputs=inputs)
    assert first["status"] == "queued"
    stored = json.loads((tmp_path / "queue" / "pending" / (first["child_id"] + ".json")).read_text())
    relocated_plan = _ref(tmp_path / "scratch" / "plan.json", b'{"plan": 1}')
    relocated = {name: _ref(tmp_path / "scratch" / name, (tmp_path / "prod" / ref["path"].rsplit("/", 1)[1]).read_bytes())
                 for name, ref in inputs.items()}
    second = _enqueue(tmp_path, plan=relocated_plan, inputs=relocated)
    assert second["status"] == "already_exists" and second["child_id"] == first["child_id"]
    # The production job keeps its own paths; nothing was rewritten.
    assert json.loads((tmp_path / "queue" / "pending" / (first["child_id"] + ".json")).read_text()) == stored
    # Different bytes behind the same input name are still a different phase, never a silent match.
    changed = {**inputs, "camera_contract": _ref(tmp_path / "prod" / "cameras2.json", b"other cams")}
    third = _enqueue(tmp_path, plan=plan, inputs=changed)
    assert third["status"] == "queued" and third["child_id"] != first["child_id"]
    # Same digests but another preparation or commit claiming the id is a conflict.
    with pytest.raises(ValueError, match="job_identity_conflict"):
        _enqueue(tmp_path, plan=plan, inputs=inputs, parent="prep-2")
