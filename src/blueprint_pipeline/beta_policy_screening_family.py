"""Source-backed static admission for the beta's native family policy lane.

This CPU-side bridge reads the operator's qualification records without loading
the simulator SDK. Actual source, scene, camera, reset and controls readback
remain mandatory inside the already budgeted provider worker.
"""
from __future__ import annotations

import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .decision_evidence_contracts import canonical_digest
from .task_family_policy_execution import FAMILY_BACKEND, validate_task_family_policy_execution_spec


def static_family_readiness(facts: Mapping[str, Any]) -> dict[str, Any]:
    from .beta_policy_screening_launch import _read, _record
    from .task_family_native_runtime import _qualified_registration
    from .task_family_policy_execution import family_policy_execution_binding

    inputs = facts["inputs"]
    plan = _read(Path(facts["packet_dir"]) / "native_task_arena_scene_plan.v1.json")
    binding = family_policy_execution_binding(plan)
    if (inputs.get("schema_version") != "blueprint.task_family_scene_inputs.v1"
            or inputs.get("scene_revision_digest") != plan.get("plan_digest")
            or inputs.get("task_success_contract_digest") != plan["task_spec"]["task_success_contract"]["contract_digest"]
            or plan.get("plan_digest") != canonical_digest(plan, digest_field="plan_digest")):
        raise ValueError("beta_screening_family_scene_source_binding_invalid")
    configured = os.environ.get("BLUEPRINT_TASK_FAMILY_RUNTIME_CATALOG_PATH")
    catalog_path = Path(configured) if configured else None
    if (catalog_path is None or not catalog_path.is_absolute() or not catalog_path.is_file()
            or catalog_path.is_symlink() or any(parent.is_symlink() for parent in catalog_path.parents)):
        raise ValueError("task_family_qualified_runtime_catalog_not_configured")
    registration, qualification = _qualified_registration(plan["native_runtime_contract"],
        binding["task_family_runtime_selection"], catalog_path, now=time.time())
    readiness = {"schema_version": "beta_family_static_readiness.v1", "source": inputs["source"],
        "source_inputs_digest": inputs["inputs_digest"], "scene_plan_digest": plan["plan_digest"],
        **binding, "task_success_contract_digest": inputs["task_success_contract_digest"],
        "runtime_registration_digest": qualification["runtime_registration_digest"],
        "runtime_qualification_digest": qualification["qualification_digest"],
        "runtime_catalog": _record(catalog_path), "runtime_source": inputs["runtime_source"],
        "pre_policy_site_qualification_required": True}
    readiness["readiness_digest"] = canonical_digest(readiness, digest_field="readiness_digest")
    return {"readiness": readiness, "registration": registration, "scene_plan": plan}


def family_interface_matches(configured: Mapping[str, Any], spec: Mapping[str, Any],
                             static: Mapping[str, Any]) -> bool:
    from .task_family_policy_https_client import INTERFACE_ID, TaskFamilyPolicySpec

    plan, registration = static["scene_plan"], static["registration"]
    validate_task_family_policy_execution_spec(spec, candidate_id=configured["candidate_id"], scene_plan=plan)
    policy = TaskFamilyPolicySpec(**spec["policy_spec"])
    native = plan["native_runtime_contract"]
    if any(getattr(policy, key) != native["robot_configuration"][key] for key in (
            "embodiment_id", "controller", "action_schema_id", "observation_schema_id")):
        return False
    cameras = [camera for camera in native["cameras"] if camera["camera_id"] == policy.camera_id]
    if (len(cameras) != 1 or policy.image_height is None or policy.image_width is None
            or cameras[0]["resolution"] != [policy.image_width, policy.image_height]
            or INTERFACE_ID not in registration["policy_interfaces"]):
        return False
    expected = {"policy_interface_id": INTERFACE_ID, "robot_profile_id": policy.embodiment_id,
        "observation_profile_id": policy.observation_schema_id, "action_schema_id": policy.action_schema_id,
        "controller": policy.controller, "action_bounds": policy.action_bounds,
        "control_hz": spec["control_frequency_hz"], "action_horizon": policy.action_chunk_rows,
        "execution_horizon": policy.open_loop_horizon, "camera_roles": [policy.camera_id],
        "image_height": policy.image_height, "image_width": policy.image_width,
        "image_encoding": "rgb_uint8", "image_preprocessing": "identity", "history_frames": 1,
        "normalization_digest": None}
    runtime = configured.get("runtime") or {}
    source = urlsplit(configured["policy"]["source"])
    return (configured.get("interface") == expected and runtime.get("backend") == FAMILY_BACKEND
            and runtime.get("runtime_id") == registration["runtime_id"]
            and configured["policy"]["kind"] == "hosted"
            and f"{source.scheme}://{source.netloc}" == policy.allowed_origin
            and configured["policy"]["digest"] == policy.model_identity_digest)


def reopen_family_planning_inputs(scene: Mapping[str, Any]) -> dict[str, Any]:
    """Reopen original compilation bytes and the current operator qualification."""
    from .beta_policy_screening_launch import _bound_path
    from .website_task_family_scene_inputs import read_family_scene_inputs

    reference = scene.get("family_source_inputs")
    if not isinstance(reference, Mapping):
        raise ValueError("beta_screening_family_source_reference_required")
    inputs = read_family_scene_inputs(_bound_path(reference))
    facts = {"inputs": inputs, "packet_dir": Path(inputs["base_native_packet"]["path"]).parent}
    static = static_family_readiness(facts)
    if (scene.get("schema_version") != "blueprint.beta_family_screening_scene_inputs.v1"
            or scene.get("inputs_digest") != canonical_digest(scene, digest_field="inputs_digest")
            or scene.get("source_family_inputs_digest") != facts["inputs"]["inputs_digest"]
            or scene.get("evaluation_backend") != FAMILY_BACKEND
            or scene.get("pre_policy_site_qualification_required") is not True
            or scene.get("scenario_scope") != "native_frozen_reset_seeds"
            or scene.get("family_static_readiness_digest") != static["readiness"]["readiness_digest"]
            or scene.get("family_static_readiness") != static["readiness"]):
        raise ValueError("beta_screening_family_static_readiness_changed")
    return {"facts": facts, **static}


def validate_family_candidate_catalog(plan: Mapping[str, Any], candidate: Mapping[str, Any],
                                      spec: Mapping[str, Any], static: Mapping[str, Any]) -> None:
    from .beta_policy_candidate_catalog import load_default_catalog

    catalog = load_default_catalog()
    configured = next((row for row in catalog["candidates"] if row["candidate_id"] == candidate["candidate_id"]), None)
    if (catalog["catalog_digest"] != plan["catalog_digest"] or configured is None
            or configured["configuration_digest"] != candidate["configuration_digest"]
            or not family_interface_matches(configured, spec, static)):
        raise ValueError("beta_screening_family_candidate_catalog_changed")
