from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline.common import write_json
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_policy_canary_scene_setup import (
    QUICK_FAMILY_COUNTS,
    materialize_scene839873_policy_canary_setup,
    materialize_setup_preflight_decision,
)


COMMIT = "c" * 40
REVISION = "sha256:" + "9" * 64
ACTIVATION = "sha256:" + "a" * 64
REQUEST = "sha256:" + "b" * 64


def _write(path: Path, value: dict) -> Path:
    write_json(path, value)
    return path


def _inputs(tmp_path: Path) -> dict[str, Path]:
    request = {
        "schema_version": "task_evaluation_launch_request.v1",
        "source_commit": COMMIT,
        "request_digest": REQUEST,
    }
    profile = {
        "schema_version": "task_evaluation_launch_profile.v1",
        "source_commit": COMMIT,
        "profile_id": "scene839873-current",
        "profile_digest": "",
    }
    profile["profile_digest"] = canonical_digest(profile, digest_field="profile_digest")
    scenario = {
        "context_kind": "evaluation_cell",
        "cell_id": "configured_scene_canonical",
        "seed": 839873104,
        "parameter_bindings": [],
        "parameter_applications": [],
    }
    scene = {
        "schema_version": "native_task_arena_scene_plan.v1",
        "scene_id": "interiorgs-839873",
        "task_id": "scene-839873-mug-planar-push",
        "task_kind": "rigid_pick_place",
        "robot": {"robot_id": "franka_panda"},
        "scenario": scenario,
        "task_spec": {
            "prompt": "Move the configured rigid object by planar push.",
            "maximum_action_steps": 240,
            "manipulation_strategy": "planar_push",
        },
        "plan_digest": "",
    }
    scene["plan_digest"] = canonical_digest(scene, digest_field="plan_digest")
    packet = {
        "schema_version": "native_task_arena_packet_receipt.v1",
        "scene_id": scene["scene_id"],
        "task_id": scene["task_id"],
        "arena_scene_plan_digest": scene["plan_digest"],
    }
    runtime = {
        "schema_version": "native_task_runtime_source_packet.v1",
        "packet_sha256": "sha256:" + "d" * 64,
        "packet_size_bytes": 4_287_162_924,
        "redistribution_permitted": True,
    }
    progression = {"configured_scene_revision_digest": REVISION}
    return {
        "request": _write(tmp_path / "request.json", request),
        "profile": _write(tmp_path / "profile.json", profile),
        "scene": _write(tmp_path / "scene.json", scene),
        "packet": _write(tmp_path / "packet.json", packet),
        "runtime": _write(tmp_path / "runtime.json", runtime),
        "progression": _write(tmp_path / "progression.json", progression),
    }


def _kwargs(tmp_path: Path) -> dict:
    inputs = _inputs(tmp_path)
    root = Path(__file__).resolve().parents[1]
    return {
        "source_commit": COMMIT,
        "configured_source_launch_id": "scene839873-configured-source",
        "scene_revision_digest": REVISION,
        "activation_digest": ACTIVATION,
        "capture_session_id": "capture-839873",
        "intake_id": "intake-839873",
        "request_digest": REQUEST,
        "launch_request_path": inputs["request"],
        "launch_profile_path": inputs["profile"],
        "configured_progression_path": inputs["progression"],
        "scene_plan_path": inputs["scene"],
        "packet_receipt_path": inputs["packet"],
        "runtime_source_receipt_path": inputs["runtime"],
        "historical_policy_readiness_path": root
        / "docs/arm_decision_proof_v1/manifests/adp009d_scene_840920_policy_readiness.v1.json",
        "pi05_checkpoint_inventory_path": root
        / "docs/experiments/policy_ranking_thesis_20260726/openpi_polaris_checkpoint_inventory.json",
        "output_dir": tmp_path / "output",
    }


def test_setup_binds_current_scene_pair_quick10_and_unqualified_boundary(
    tmp_path: Path,
) -> None:
    setup = materialize_scene839873_policy_canary_setup(**_kwargs(tmp_path))

    assert setup["status"] == "verified_runnable"
    assert setup["candidate_ids"] == ["pi05_droid", "groot_n17_droid"]
    assert setup["quick_10"]["learned_policy_rollout_count"] == 20
    assert {
        family: sum(cell["family"] == family for cell in setup["quick_10"]["cells"])
        for family in QUICK_FAMILY_COUNTS
    } == QUICK_FAMILY_COUNTS
    assert setup["historical_runtime_smoke"]["current_runtime_proof"] is False
    assert setup["scene_promotion_authorized"] is False
    assert setup["official_ranking_authorized"] is False
    for record in setup["records"].values():
        assert Path(record["path"]).is_file()
        assert record["sha256"].startswith("sha256:")


def test_missing_activation_and_terminal_lineage_emit_typed_blockers(
    tmp_path: Path,
) -> None:
    kwargs = _kwargs(tmp_path)
    kwargs.update(activation_digest="", capture_session_id="", intake_id="")
    decision = materialize_setup_preflight_decision(
        output_path=tmp_path / "decision.json", **kwargs
    )

    assert decision["status"] == "blocked"
    assert decision["blockers"] == [
        "policy_canary_activation_digest_invalid",
        "policy_canary_capture_session_id_missing",
        "policy_canary_intake_id_missing",
    ]
    assert not (tmp_path / "output/task_evaluation_policy_canary_execution_setup.v1.json").exists()


def test_checkpoint_registry_drift_fails_before_specs_are_written(
    tmp_path: Path,
) -> None:
    kwargs = _kwargs(tmp_path)
    readiness_path = Path(kwargs["historical_policy_readiness_path"])
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    readiness["candidates"][0]["checkpoint"]["inventory_digest"] = "sha256:" + "f" * 64
    readiness["readiness_digest"] = canonical_digest(
        readiness, digest_field="readiness_digest"
    )
    changed = _write(tmp_path / "changed-readiness.json", readiness)
    kwargs["historical_policy_readiness_path"] = changed
    decision = materialize_setup_preflight_decision(
        output_path=tmp_path / "decision.json", **kwargs
    )

    assert decision["status"] == "blocked"
    assert decision["blockers"] == [
        "policy_canary_pi05_droid_registry_or_rights_invalid"
    ]
