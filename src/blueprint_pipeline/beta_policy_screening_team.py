"""Bind a reusable robot/policy registration to each beta task's own scene.

The operator supplies a prepared USD, manifest and served policy identity once.
The task compiler supplies the scene, instruction and episode bounds. Neither
catalog membership nor a downloaded robot description is execution evidence.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any
from urllib.parse import urlsplit

# One effective-catalog implementation, shared with the CPU intake preview.
from .beta_policy_effective_catalog import _registration_rights, catalog_for_profile, team_policy_interface
from .decision_evidence_contracts import canonical_digest
from .policy_interface_binding import TEAM_LOCAL_INTERFACES, TEAM_SCRIPTED_INTERFACES
from .team_local_policy import runtime_for_interface
from .team_embodiment import apply_team_embodiment, validate_embodiment_manifest


def team_interface_matches(configured: Mapping[str, Any], spec: Mapping[str, Any]) -> bool:
    expected = team_policy_interface(spec)
    runtime = configured.get("runtime") or {}
    policy = spec["policy_spec"]
    if spec["policy_interface_id"] in TEAM_SCRIPTED_INTERFACES:
        from .scripted_robot_runtime_dispatch import runtime_for_policy
        local = runtime_for_policy(policy)
        return (configured.get("interface") == expected and configured["policy"]["kind"] == "scripted"
                and configured["policy"]["source"] == local.MODEL_SOURCE
                and configured["policy"]["revision"] == local.MODEL_REVISION
                and configured["policy"]["digest"] == policy["model_identity_digest"]
                and runtime.get("backend") == "robolab.v1"
                and runtime.get("runtime_id") == "native_task_arena_policy_canary.v1")
    if spec["policy_interface_id"] in TEAM_LOCAL_INTERFACES:
        local = runtime_for_interface(spec["policy_interface_id"])
        return (configured.get("interface") == expected
                and configured["policy"]["kind"] == "checkpoint"
                and configured["policy"]["source"] == local.MODEL_SOURCE
                and configured["policy"]["revision"] == local.MODEL_REVISION
                and configured["policy"]["digest"] == policy["model_identity_digest"]
                and runtime.get("backend") == "robolab.v1"
                and runtime.get("runtime_id") == "native_task_arena_policy_canary.v1")
    source = urlsplit(configured["policy"]["source"])
    return (configured.get("interface") == expected
            and configured["policy"]["kind"] == "hosted"
            and f"{source.scheme}://{source.netloc}" == policy["allowed_origin"]
            and configured["policy"]["digest"] == policy["model_identity_digest"]
            and runtime.get("backend") == "robolab.v1"
            and runtime.get("runtime_id") == "native_task_arena_policy_canary.v1")


def build_registered_team_spec(*, candidate_id: str, registration: Mapping[str, Any],
                               scene_plan: Mapping[str, Any], contract: Mapping[str, Any],
                               prompt: str, scene_revision_digest: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Materialize a per-task spec from sealed, reusable operator inputs.

    ``robot_policy`` contains ``manifest`` and ``model`` artifact references,
    ``policy_spec`` (the existing HTTPS client contract), and ``rights_binding``.
    Source imports also carry their prepared closure/provenance. Actual model
    bytes are reopened and the station binding is derived for this task.
    """
    from .beta_policy_screening_launch import _bound_path, _read
    from .team_embodiment import read_model_structure, team_embodiment_binding, verify_model_file

    manifest = validate_embodiment_manifest(_read(_bound_path(registration["manifest"])))
    model_path = _bound_path(registration["model"])
    record = verify_model_file(manifest, model_path)
    for key in ("dependency_closure", "source_import_provenance"):
        if key in registration:
            record[key] = registration[key]
    stance = None
    if manifest["base"]["mode"] == "free":
        raise ValueError("beta_screening_team_free_base_stance_registration_required")
    binding = team_embodiment_binding(manifest, model_record=record, structure=read_model_structure(model_path),
                                      scene_plan=scene_plan, stance=stance)
    native = apply_team_embodiment(scene_plan, binding)
    policy = dict(registration["policy_spec"])
    interface = policy.get("policy_interface_id", "blueprint_policy_https.v1")
    horizon = policy["open_loop_horizon"]
    hz = native["cadence"]["control_frequency_hz"]
    action_hz = policy.get("action_hz", hz)
    if (type(horizon) is not int or horizon < 1 or not isinstance(action_hz, (int, float))
            or isinstance(action_hz, bool) or not math.isfinite(action_hz) or action_hz <= 0
            or not math.isclose(hz / action_hz, round(hz / action_hz))):
        raise ValueError("beta_screening_team_episode_cadence_invalid")
    rights = _registration_rights(registration, candidate_id=candidate_id)
    spec = {"schema_version": "native_task_arena_policy_canary_execution_spec.v1",
            "candidate_id": candidate_id, "policy_interface_id": interface, "evaluation_backend": "robolab.v1",
            "execution_authority": "internal_policy_canary_unqualified", "claim_ceiling": "diagnostic_policy_execution",
            "ranking_permitted": False, "qualification_permitted": False, "scene_promotion_permitted": False,
            "policy_endpoint": {"credential_env": "BLUEPRINT_TEAM_POLICY_BEARER_FILE"}, "policy_spec": policy,
            "candidate_rights_binding": dict(rights), "checkpoint_digest": policy["model_identity_digest"],
            "runtime_identity_digest": canonical_digest({"policy_spec": policy,
                "embodiment_digest": manifest["embodiment_digest"]}),
            "worker_identity_requirement": "none", "require_observed_eef_support": False,
            "task_success_contract": dict(contract), "task_success_contract_digest": contract["contract_digest"],
            "prompt": prompt, "open_loop_horizon": horizon,
            "max_policy_queries": math.ceil(native["cadence"]["maximum_action_steps"] / (hz / action_hz) / horizon),
            "team_embodiment": binding}
    if interface == "dreamlabs_dream1.v1":
        from .dreamlabs_policy_client import validate_provider_model_record
        from .team_candidate_quick10_admission import validate_provider_disclosure

        metadata = validate_provider_model_record(registration["provider_model_metadata"], model_id=policy["model_id"])
        if metadata["model_identity_digest"] != policy["model_identity_digest"]:
            raise ValueError("beta_screening_dream_provider_identity_changed")
        disclosure = registration.get("provider_disclosure") or {}
        validate_provider_disclosure({"provider_disclosure": disclosure}, team_id=registration["team_id"],
            checkpoint_id=registration["checkpoint_id"], candidate_id=candidate_id, model_id=policy["model_id"],
            scene_revision_digest=scene_revision_digest)
        spec.update(provider_disclosure=disclosure, checkpoint_identity_basis="provider_metadata",
                    weights_identity_verified=False)
    if interface in TEAM_LOCAL_INTERFACES:
        spec.update(policy_endpoint={"host": "127.0.0.1", "port": runtime_for_interface(interface).PORT},
                    checkpoint_identity_basis="pinned_file_inventory_verified_before_worker_load",
                    weights_identity_verified=False)
    if interface in TEAM_SCRIPTED_INTERFACES:
        from .scripted_robot_runtime_dispatch import runtime_for_policy
        build_task_plan = runtime_for_policy(policy).build_task_plan
        from .scripted_robot_baseline import EVIDENCE_KIND, ACTION_SOURCE
        spec.update(policy_endpoint={"transport": "in_process"},
                    execution_evidence_kind=EVIDENCE_KIND, action_source=ACTION_SOURCE,
                    checkpoint_identity_basis="owned_scripted_controller_source_inventory_no_weights",
                    weights_identity_verified=False,
                    scripted_task_plan=build_task_plan(scene_plan, policy, manifest))
    team_policy_interface(spec)
    spec["execution_spec_digest"] = canonical_digest(spec, digest_field="execution_spec_digest")
    return spec, dict(registration["model"])


def registration_from_prepared_inputs(*, prepared: Mapping[str, Any], model_path: Any, destination: Any,
                                      candidate_id: str, team_id: str, checkpoint_id: str,
                                      policy_spec: Mapping[str, Any], rights: Mapping[str, Any],
                                      display_name: str, provider_model_metadata: Mapping[str, Any] | None = None,
                                      provider_disclosure: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Connect the public-asset importer output to the reusable registry shape.

    This writes only the nonsensitive manifest; no key, endpoint call, simulator
    allocation, policy license grant or production registry mutation occurs.
    """
    from pathlib import Path
    from .beta_policy_screening_capabilities import _retain
    from .beta_policy_screening_launch import _record
    from .team_embodiment import verify_model_file, validate_source_import_provenance

    if prepared.get("schema_version") != "embodiment_runtime_binding_inputs.v1":
        raise ValueError("beta_screening_prepared_robot_inputs_invalid")
    manifest = validate_embodiment_manifest(prepared["robot_embodiment"])
    model = verify_model_file(manifest, model_path)
    original = prepared["model_record"]
    if any(model[key] != original[key] for key in ("sha256", "size_bytes")):
        raise ValueError("beta_screening_prepared_robot_identity_changed")
    provenance = validate_source_import_provenance(original["source_import_provenance"],
        model_record=model, dependency_closure=original["dependency_closure"])
    registration = {"team_id": team_id, "checkpoint_id": checkpoint_id, "display_name": display_name,
        "model": _record(Path(model_path)), "policy_spec": dict(policy_spec), "rights_binding": dict(rights),
        "dependency_closure": original["dependency_closure"], "source_import_provenance": provenance}
    _registration_rights(registration, candidate_id=candidate_id)
    target = Path(destination).resolve() / (manifest["embodiment_digest"][7:] + ".robot_manifest.json")
    registration["manifest"] = _record(_retain(target, manifest))
    if provider_model_metadata is not None:
        registration["provider_model_metadata"] = dict(provider_model_metadata)
    if provider_disclosure is not None:
        registration["provider_disclosure"] = dict(provider_disclosure)
    return registration


def current_catalog_for_scene(scene: Mapping[str, Any]) -> dict[str, Any]:
    from .team_candidate_quick10_registry import profile_for_capture

    profile = profile_for_capture(scene.get("screening_capture_root", ""))
    return catalog_for_profile(profile or {})


def validate_team_candidate_catalog(plan: Mapping[str, Any], candidate: Mapping[str, Any],
                                    spec: Mapping[str, Any], scene: Mapping[str, Any]) -> None:
    catalog = current_catalog_for_scene(scene)
    configured = next((row for row in catalog["candidates"] if row["candidate_id"] == candidate["candidate_id"]), None)
    if (catalog["catalog_digest"] != plan["catalog_digest"] or configured is None
            or configured["configuration_digest"] != candidate["configuration_digest"]
            or not configured["eligibility"]["enabled"]
            or configured["eligibility"]["commercial_use"] not in {"allowed", "allowed_with_conditions"}
            or not team_interface_matches(configured, spec)):
        raise ValueError("beta_screening_team_candidate_catalog_changed")
    if (spec["policy_interface_id"] == "dreamlabs_dream1.v1"
            and spec["provider_disclosure"]["scene_revision_digest"] != scene["scene_revision_digest"]):
        raise ValueError("beta_screening_dream_disclosure_scene_mismatch")
