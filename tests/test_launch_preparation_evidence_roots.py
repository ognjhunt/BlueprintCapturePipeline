"""The parent must accept SAM evidence from the phase executors' output tree once the chain is ready."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from blueprint_pipeline import task_evaluation_sam31_preparation_queue as queue

COMMIT = "ac689e03ab6a" + "0" * 28
DIGEST = "sha256:" + "3c" * 32
# The five evidence artifacts the driver returned on 2026-09-05 23:42Z (scene 841757, attempt on ac689e03),
# all under the executors' tree, which the parent's roots did not include.
PRODUCTION_EVIDENCE_PATHS = [
    "/var/lib/blueprint/task-evaluation-inputs/sam31-preparations/3cd17114a55fc6538b4719d378dd09d36c0acf670b19412f4a2cb52c0fe526e2/sam31-0e9267305965ad55a70ab828565429351d10874ab8cea394a4037f51ea43742f/artifacts/produced/adp009d_segment_contribution_cutout_set.v1.json",
    "/var/lib/blueprint/task-evaluation-inputs/sam31-preparations/3cd17114a55fc6538b4719d378dd09d36c0acf670b19412f4a2cb52c0fe526e2/sam31-4019903a00a0b79d193f593d45ae5cac8f9dbc0e4a3f0b3a1b8a0f6a1d0c9e2f/artifacts/allocator/result.json",
    "/var/lib/blueprint/task-evaluation-inputs/sam31-preparations/3cd17114a55fc6538b4719d378dd09d36c0acf670b19412f4a2cb52c0fe526e2/sam31-58d53fbbb2dad208c5bfa22808677a9943a2d7b1e4d6c0f9a8b7c6d5e4f3a2b1/artifacts/produced/fresh_scene_removal_freezes.v1.json",
    "/var/lib/blueprint/task-evaluation-inputs/sam31-preparations/3cd17114a55fc6538b4719d378dd09d36c0acf670b19412f4a2cb52c0fe526e2/sam31-12f8b716b7c008205202484b2ac2c6dce2da07c3b2a1908f7e6d5c4b3a291807/artifacts/produced/calibrated_object_mask_set.v1.json",
    "/var/lib/blueprint/task-evaluation-inputs/sam31-preparations/1758a6805a8a032eff40283ea1bcbcc97e8eca3faac0ca671860e6e5f5f007b6/sam31-f7393671490c3e38533a46789382b4eaea23c506eb17ed0c61d7e7f26a844866/artifacts/allocator/sam31_vast_source_track_canary/semantic_source_track_import_result.v1.json",
]


def test_the_parent_roots_include_the_executors_evidence_tree() -> None:
    roots = queue.preparation_evidence_roots(
        "/var/lib/blueprint/task-evaluation-inputs/prepared-references",
        "/var/lib/blueprint/pipeline-control-plane/task-evaluation-launch-preparations",
    )

    assert queue.SAM31_EXECUTION_ROOT in roots
    for path in PRODUCTION_EVIDENCE_PATHS:
        assert any(Path(path).is_relative_to(root) for root in roots), path
    assert not any(Path("/var/lib/blueprint/spend-authority/consumed/x.json").is_relative_to(root) for root in roots)


def _evidence(root: Path, name: str) -> dict:
    path = root / "artifacts" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"name": name}), encoding="utf-8")
    return {"path": str(path), "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(), "size_bytes": path.stat().st_size}


def test_a_ready_advancement_from_the_executors_tree_passes_where_the_old_roots_refused(tmp_path: Path, monkeypatch) -> None:
    """2026-09-05 23:42Z: after render, tracking, review, masks, freezes, cutout and the
    FlashSplat sweep had all completed, the parent refused the driver's ``ready``
    advancement with sam31_preparation_evidence_path_invalid because its roots were
    its input root, its queue and the control plane; every earlier advancement had
    only referenced child job files under the control plane."""

    queue_root = tmp_path / "launch-preparations"
    input_root = tmp_path / "prepared-references"
    execution_root = tmp_path / "sam31-preparations"
    for path in (queue_root, input_root, execution_root):
        path.mkdir()
    monkeypatch.setattr(queue, "SAM31_EXECUTION_ROOT", execution_root)
    monkeypatch.setattr(queue, "CONTROL_PLANE_ROOT", tmp_path / "control-plane")
    child_root = execution_root / ("3c" * 32) / ("sam31-" + "7" * 64)
    refs = [_evidence(child_root, name) for name in ("cutout", "sweep", "freezes", "masks", "tracks")]
    advancement = {
        "status": "ready", "evidence_refs": refs,
        "sam31_exact_mask_inputs": {name: ref for name, ref in zip(("a", "b", "c", "d", "e"), refs)},
        "sam31_preparation_result": {"status": "exact_mask_inputs_ready"},
        "human_review_required": False, "candidate_policy_queried": False,
    }
    envelope = {
        "request": {"preparation_id": "adp-new-scene-book-to-tray-841757-test", "expected_production_commit": COMMIT, "run_id": "run-1"},
        "request_digest": DIGEST, "stage_one_configuration": {"sam31_review_kind": "ai"},
    }
    old_roots = (input_root, queue_root, tmp_path / "control-plane")

    with pytest.raises(queue.Sam31PreparationQueueError, match="sam31_preparation_evidence_path_invalid"):
        queue.advance_sam31_for_preparation(queue_root=queue_root, envelope_context=dict(envelope), approved_roots=old_roots, advancer=lambda context: dict(advancement))

    result = queue.advance_sam31_for_preparation(
        queue_root=queue_root, envelope_context=dict(envelope),
        approved_roots=queue.preparation_evidence_roots(input_root, queue_root), advancer=lambda context: dict(advancement),
    )

    assert result["status"] == "ready" and len(result["evidence_refs"]) == 5
    assert any((queue_root / "source-progress").rglob("*.json"))
