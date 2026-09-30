"""ADP-009D Day-28: reopen historical accounting without changing evidence."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from blueprint_pipeline import task_evaluation_configured_controls_autostart as auto
from blueprint_pipeline import task_evaluation_openai_usage_validation as usage
from blueprint_pipeline import task_evaluation_retained_controls_evidence as retained
from blueprint_pipeline import task_evaluation_robot_placement_agent as agent
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests.test_task_evaluation_configured_controls_autostart import _intent

_DIGEST = "sha256:" + "a" * 64


def _seal(value, field):
    value[field] = canonical_digest(value, digest_field=field)
    return value


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.chmod(0o640)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    path.chmod(0o440)
    return path


def _lineage(root, model):
    """Synthetic, development-only bytes; no provider, sync or execution calls."""
    root.mkdir()
    owner = {
        "attempt_id": "fixture-construction",
        "attempt_digest": _DIGEST,
        "intent_id": "fixture",
        "intent_digest": _DIGEST,
        "provider": "vast",
        "maximum_spend_usd": 1.0,
        "runtime_digest": _DIGEST,
        "input_digest": _DIGEST,
    }
    receipt = agent._build_placement_receipt(
        run_id="fixture",
        scene_digest=_DIGEST,
        task_digest=_DIGEST,
        scene_context_digest=_DIGEST,
        task_context_digest=_DIGEST,
        overview_images=[],
        prior_native_attempts=[],
        max_rounds=2,
        native_loop_enabled=False,
        task_trajectory_digest=_DIGEST,
        candidate_inventory_digest=_DIGEST,
        candidate_inventory_trajectory_digest=_DIGEST,
        history=[
            record := {
                "proposal": {
                    "candidate_id": "fixture",
                    "support_surface_id": "table",
                    "pose": {
                        "position_world_m": [0.0, 0.0, 0.0],
                        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                    },
                },
                "geometry_gate": {
                    "candidate_id": "fixture",
                    "status": "passed",
                    "geometry_gate_digest": _DIGEST,
                },
                "visual_review": {
                    "status": "passed",
                    "robot_supported_by_declared_surface": True,
                    "robot_not_visibly_clipping_site_geometry": True,
                    "robot_faces_task_workspace": True,
                    "task_workspace_visually_reachable": True,
                    "camera_views_are_sufficient": True,
                    "reason": "Development-only fixture.",
                    "revision_guidance": [],
                },
            }
        ],
        accepted=record,
    )
    receipt["model"] = model
    receipt_path = _json(root / "placement.json", _seal(receipt, "receipt_digest"))
    inventory = _seal({"candidate_inventory_digest": _DIGEST}, "checkpoint_digest")
    inventory_path = _json(root / "inventory.json", inventory)
    checkpoint = _seal(
        {
            "receipt_path": str(receipt_path),
            "inventory_path": str(inventory_path),
            "receipt_sha256": retained._file(receipt_path)["digest"],
            "inventory_sha256": retained._file(inventory_path)["digest"],
        },
        "checkpoint_digest",
    )
    checkpoint_path = _json(root / "checkpoint.json", checkpoint)
    accounting = {
        name: usage._artifact_record(_json(root / (name + ".json"), {"fixture": name}))
        for name in (
            "reservation",
            "completion",
            "exclusive_lock",
            "exclusive_lock_release",
            "inference_reservations",
        )
    }
    packet_record = usage._artifact_record(_json(root / "usage.json", {"development_only": True}))
    sync_record = usage._artifact_record(_json(root / "sync.json", {"status": "skipped"}))
    universe = usage._artifact_record(_json(root / "universe.json", {"development_only": True}))
    base = _json(root / "base.json", {"development_only": True})
    parent = None
    for generation, commit in enumerate(("a" * 40, "b" * 40, "c" * 40)):
        directory = root / str(generation)
        intent_path, intent = _intent(directory)
        owner["source_commit"] = commit
        intent.update(expected_production_commit=commit, configuration_source_commit=commit)
        intent["placement"]["agent_model"] = model
        for phase in intent["phases"].values():
            path = Path(phase["release_window_template_path"])
            template = json.loads(path.read_text())
            template["expected_production_commit"] = commit
            _json(path, _seal(template, "template_digest"))
            _json(
                Path(phase["authorization_path"]),
                {"scene_owner_attempt": {"scene_attempt_binding": owner}},
            )
        if parent is not None:
            parent = _seal({**parent, "execution_commit": commit}, "adoption_digest")
            intent["completed_placement_adoption"] = parent
            intent["placement"]["max_rounds"] = 0
            intent["placement"]["max_inference_cost_usd"] = 0.0
            intent["placement"]["official_cost_authority"]["maximum_cost_usd"] = 0.0
        intent["artifact_inventory"] = {
            name: auto._artifact(path) for name, path in auto._intent_paths(intent).items()
        }
        _json(intent_path, _seal(intent, "intent_digest"))
        plan = {
            "schema_version": "task_evaluation_configured_controls_progression_plan.v2",
            "enabled": True,
            "source_launch_id": "fixture",
            "source_launch_receipt_digest": _DIGEST,
            "source_configuration_commit": commit,
            "expected_production_commit": commit,
            "submitted_by": "development-only",
            "profile_dir": intent["profile_dir"],
            "phases": intent["phases"],
            "artifact_inventory": {
                name: row
                for name, row in intent["artifact_inventory"].items()
                if name.startswith("phases.")
            },
            "future_outputs": {
                phase: {"expected_activation_id": "fixture-" + phase} for phase in intent["phases"]
            },
        }
        plan_path = _json(directory / "plan.json", _seal(plan, "plan_digest"))
        result = {
            "schema_version": auto.RESULT_SCHEMA_VERSION,
            "status": "agent_binding_accepted_plan_materialized",
            "source_launch_id": "fixture",
            "intent_digest": intent["intent_digest"],
            "scene_binding_digest": _DIGEST,
            "task_binding_digest": _DIGEST,
            "cpu_placement_checkpoint_binding_digest": _DIGEST,
            "configured_scene_revision_digest": _DIGEST,
            "trajectory_digest": _DIGEST,
            "candidate_inventory_digest": _DIGEST,
            "selected_candidate_id": "fixture",
            "cpu_inventory_ranker_receipt_digest": _DIGEST,
            "placement_agent_receipt_digest": receipt["receipt_digest"],
            "placement_agent_model": model,
            "placement_agent_reasoning_effort": "high",
            "placement_agent_selected_exact_inventory_member": True,
            "placement_agent_visual_review_completed": True,
            "official_openai_cost_evidence": accounting,
            "openai_inference_usage_packet": packet_record,
            "openai_inference_usage_webapp_sync": {
                "required": False,
                "status": "skipped",
                "artifact": sync_record,
                "packet_digest": _DIGEST,
                "call_count": 2,
            },
            "base_pose_candidate_path": str(base),
            "native_construction_candidate_universe": {
                "path": universe["path"],
                "file_sha256": universe["digest"],
                "inventory_digest": _DIGEST,
                "candidate_count": 1,
            },
            "plan_path": str(plan_path),
            "plan_digest": plan["plan_digest"],
            "cpu_position_ik_qualified": True,
            "native_orientation_collision_contact_camera_and_execution_required": True,
            "provider_mutation_performed": False,
            "paid_execution_requested": True,
        }
        if parent is not None:
            result.update(completed_placement_adoption=parent, placement_calls_reexecuted=False)
        result_path = _json(directory / "result.json", _seal(result, "result_digest"))
        parent = _seal(
            {
                "schema_version": retained.PLACEMENT_SCHEMA,
                "source_intent": retained._file(intent_path),
                "source_result": retained._file(result_path),
                "source_plan": retained._file(plan_path),
                "source_agent_checkpoint": retained._file(checkpoint_path),
                "source_launch_id": "fixture",
                "owner_intent_digest": owner["intent_digest"],
                "execution_commit": commit,
            },
            "adoption_digest",
        )
    cancellation = _seal(
        {
            "schema_version": retained.PLACEMENT_CANCEL_SCHEMA,
            **{
                k: owner[k]
                for k in (
                    "attempt_id",
                    "attempt_digest",
                    "intent_digest",
                    "provider",
                    "maximum_spend_usd",
                )
            },
            "completed_placement_adoption": parent,
            "status": "cancelled_before_native_submission",
            "native_submission_absent": True,
            "model_holds_retained": True,
            "provider_mutation_performed": False,
        },
        "receipt_digest",
    )
    _json(root / retained.DIRECTORY / (owner["attempt_id"] + ".json"), cancellation)
    return owner, cancellation


@pytest.mark.parametrize("model", ["gpt-6-sol", "gpt-5.6-sol"])
def test_historical_intent_result_cancellation_ancestry_preserves_all_bytes(tmp_path, model):
    root = tmp_path / "history"
    owner, cancellation = _lineage(root, model)
    before = {
        path: (path.read_bytes(), path.stat().st_mode) for path in root.rglob("*") if path.is_file()
    }
    assert retained.validated_cancellation(root, owner) == cancellation
    assert {path: (path.read_bytes(), path.stat().st_mode) for path in before} == before
    assert agent.ROBOT_PLACEMENT_AGENT_MODEL == "gpt-6.1-sol"
    assert (
        agent.robot_placement_agents_sdk_config(
            max_inference_cost_usd=2.56,
            allow_live_invocation=False,
        ).model
        == "gpt-6.1-sol"
    )


@pytest.mark.parametrize(
    "defect",
    [
        "unknown_model",
        "modified_digest",
        "altered_cost_authority",
        "altered_cost_bytes",
        "unknown_result_model",
        "different_retained_result_model",
        "altered_cancellation_authority",
    ],
)
def test_retained_model_compatibility_keeps_fail_closed_checks(tmp_path, defect):
    root = tmp_path / "history"
    owner, cancellation = _lineage(root, "gpt-6-sol")
    packet = deepcopy(cancellation["completed_placement_adoption"])
    intent_path = Path(packet["source_intent"]["path"])
    intent = json.loads(intent_path.read_text())
    if defect in {"unknown_model", "modified_digest", "altered_cost_authority"}:
        if defect == "unknown_model":
            intent["placement"]["agent_model"] = "unadmitted-model"
        elif defect == "modified_digest":
            intent["intent_digest"] = "sha256:" + "0" * 64
        else:
            intent["placement"]["official_cost_authority"]["credential_role"] = "unapproved-role"
        if defect != "modified_digest":
            _seal(intent, "intent_digest")
        _json(intent_path, intent)
        packet["source_intent"] = retained._file(intent_path)
        cancellation["completed_placement_adoption"] = _seal(packet, "adoption_digest")
        expected = "configured_controls_autostart_intent_invalid"
    elif defect == "altered_cost_bytes":
        _json(root / "reservation.json", {"fixture": "altered"})
        expected = "configured_controls_autostart_result_invalid"
    elif defect in {"unknown_result_model", "different_retained_result_model"}:
        result_path = Path(packet["source_result"]["path"])
        result = json.loads(result_path.read_text())
        result["placement_agent_model"] = (
            "unadmitted-model" if defect == "unknown_result_model" else "gpt-6.1-sol"
        )
        _json(result_path, _seal(result, "result_digest"))
        packet["source_result"] = retained._file(result_path)
        cancellation["completed_placement_adoption"] = _seal(packet, "adoption_digest")
        expected = "configured_controls_autostart_result_invalid"
    else:
        cancellation["maximum_spend_usd"] = 0.0
        expected = "completed_placement_adoption_cancellation_invalid"
    _seal(cancellation, "receipt_digest")
    with pytest.raises((ValueError, RuntimeError), match=expected):
        retained.validate_placement_cancellation(receipt=cancellation, attempt=owner)
