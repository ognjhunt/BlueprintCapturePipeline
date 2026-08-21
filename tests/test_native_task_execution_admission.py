from __future__ import annotations

from pathlib import Path

import pytest

from blueprint_pipeline.common import write_json
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_task_execution_admission import (
    NativeTaskExecutionAdmissionError,
    prepare_native_task_execution_candidate,
    require_native_task_execution_admission,
    seal_native_task_execution_admission,
)


IMAGE = "nvcr.io/nvidia/isaac-sim:6.0.1@sha256:" + "a" * 64


def _sha(path: Path) -> str:
    import hashlib

    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _asset(path: Path, *, approximation: str, thin: bool = False) -> dict:
    points = (
        "[(-0.00005, -0.025, -0.025), (0.00005, 0.025, 0.025)]"
        if thin
        else "[(-0.02, -0.025, -0.025), (0.02, 0.025, 0.025)]"
    )
    path.write_text(
        f'''#usda 1.0
(defaultPrim="Asset")
def Xform "Asset"
{{
    def Xform "door" (prepend apiSchemas = ["PhysicsRigidBodyAPI"])
    {{
        def Mesh "rim" (
            prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]
        )
        {{
            uniform token physics:approximation = "{approximation}"
            point3f[] points = {points}
        }}
    }}
}}
''',
        encoding="utf-8",
    )
    return {
        "task_id": "task_a",
        "asset_id": "washer",
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha(path),
        "registered_static_qualification_digest": "sha256:" + "b" * 64,
    }


def test_thin_dynamic_convex_hull_fails_before_provider_authority(
    tmp_path: Path,
) -> None:
    asset = _asset(tmp_path / "thin.usda", approximation="convexHull", thin=True)

    with pytest.raises(
        NativeTaskExecutionAdmissionError,
        match="native_task_dynamic_convex_hull_gpu_oblong",
    ):
        prepare_native_task_execution_candidate(
            scene_id="scene",
            runtime_image=IMAGE,
            assets=[asset],
        )


def test_convex_decomposition_candidate_seals_only_after_runtime_and_packet(
    tmp_path: Path,
) -> None:
    asset = _asset(
        tmp_path / "thin.usda",
        approximation="convexDecomposition",
        thin=True,
    )
    candidate = prepare_native_task_execution_candidate(
        scene_id="scene",
        runtime_image=IMAGE,
        assets=[asset],
        destination=tmp_path / "candidate.json",
    )
    intent = candidate["assets"][0]["collision_intent"]
    runtime = {
        "schema_version": "paired_target_native_import_runtime_result.v1",
        "status": "completed",
        "scene_id": "scene",
        "execution_candidate_digest": candidate["candidate_digest"],
        "all_replacements_import_qualified": True,
        "native_gpu_physics_qualified": True,
        "replacements": [
            {
                "task_id": "task_a",
                "asset_id": "washer",
                "native_simulator_import_qualified": True,
                "native_gpu_physics_qualified": True,
                "collision_intent_digest": intent["intent_digest"],
                "blockers": [],
            }
        ],
        "blockers": [],
        "result_digest": "",
    }
    runtime["result_digest"] = canonical_digest(
        runtime, digest_field="result_digest"
    )
    plan = {
        "schema_version": "native_task_arena_scene_plan.v1",
        "scene_id": "scene",
        "task_id": "task_a",
        "plan_digest": "",
    }
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    packet = {
        "schema_version": "native_task_arena_packet_receipt.v1",
        "status": "construction_packet_completed",
        "scene_id": "scene",
        "task_id": "task_a",
        "arena_scene_plan_digest": plan["plan_digest"],
        "source_bindings": [
            {
                "asset_id": "washer",
                "staged_sha256": asset["sha256"],
            }
        ],
        "receipt_digest": "",
    }
    packet["receipt_digest"] = canonical_digest(
        packet, digest_field="receipt_digest"
    )
    root = tmp_path / "packet"
    root.mkdir()
    write_json(root / "native_task_arena_packet_receipt.v1.json", packet)
    write_json(root / "native_task_arena_scene_plan.v1.json", plan)

    admission = seal_native_task_execution_admission(
        candidate=candidate,
        runtime_result=runtime,
        packet_receipt=packet,
        scene_plan=plan,
        task_id="task_a",
        destination=root / "native_task_execution_admission.v1.json",
    )

    assert admission["construction_authorized"] is True
    assert admission["native_gpu_cooking_readback_qualified"] is True
    assert require_native_task_execution_admission(root) == admission


def test_packet_mutation_invalidates_execution_admission(tmp_path: Path) -> None:
    root = tmp_path / "packet"
    root.mkdir()
    write_json(root / "native_task_execution_admission.v1.json", {})
    write_json(root / "native_task_arena_packet_receipt.v1.json", {})
    write_json(root / "native_task_arena_scene_plan.v1.json", {})

    with pytest.raises(NativeTaskExecutionAdmissionError):
        require_native_task_execution_admission(root)
