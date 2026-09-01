from __future__ import annotations

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_task_arena_policy_canary_worker import (
    _construction_lineage_mode,
)


def _scene_plan() -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": "native_task_arena_scene_plan.v1",
        "scene_id": "interiorgs-839873",
        "task_id": "scene-839873-mug-planar-push",
        "plan_digest": "",
    }
    value["plan_digest"] = canonical_digest(value, digest_field="plan_digest")
    return value


def _compiled_result(scene_revision_digest: str) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": "task_evaluation_episode_compilation_result.v1",
        "status": "compiled_for_production_launch",
        "blockers": [],
        "configured_scene_revision_digest": scene_revision_digest,
        "compiled_episode_packet_digest": "sha256:" + "e" * 64,
        "provider_mutation_performed": False,
        "paid_execution_requested": False,
        "result_digest": "",
    }
    value["result_digest"] = canonical_digest(value, digest_field="result_digest")
    return value


def test_controls_pending_canary_accepts_digest_bound_compiled_scene_lineage() -> None:
    revision = "sha256:" + "2" * 64

    mode = _construction_lineage_mode(
        inputs={"scene_revision_digest": revision},
        base_scene_plan=_scene_plan(),
        construction=_compiled_result(revision),
    )

    assert mode == "compiled_configured_scene_diagnostic"


def test_compiled_scene_lineage_cannot_change_the_scene_revision() -> None:
    compiled = _compiled_result("sha256:" + "2" * 64)

    with pytest.raises(
        RuntimeError, match="policy_canary_compiled_scene_lineage_invalid"
    ):
        _construction_lineage_mode(
            inputs={"scene_revision_digest": "sha256:" + "3" * 64},
            base_scene_plan=_scene_plan(),
            construction=compiled,
        )


def test_qualified_native_construction_path_remains_strict() -> None:
    plan = _scene_plan()
    construction: dict[str, object] = {
        "schema_version": "native_task_arena_construction_result.v1",
        "status": "completed",
        "construction_gate_qualified": True,
        "scene_plan_digest": plan["plan_digest"],
        "result_digest": "",
    }
    construction["result_digest"] = canonical_digest(
        construction, digest_field="result_digest"
    )

    assert _construction_lineage_mode(
        inputs={"scene_revision_digest": "sha256:" + "2" * 64},
        base_scene_plan=plan,
        construction=construction,
    ) == "qualified_native_construction_result"
