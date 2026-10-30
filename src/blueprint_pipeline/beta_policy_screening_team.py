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

from .decision_evidence_contracts import canonical_digest
from .policy_interface_binding import TEAM_ROBOT_INTERFACES
from .team_embodiment import apply_team_embodiment, camera_names, validate_embodiment_manifest


def team_policy_interface(spec: Mapping[str, Any]) -> dict[str, Any]:
    """The exact observations/actions served on the spec's declared robot."""
    interface = spec.get("policy_interface_id")
    if interface not in TEAM_ROBOT_INTERFACES:
        raise ValueError("beta_screening_team_interface_unsupported")
    binding = spec.get("team_embodiment") or {}
    if binding.get("binding_digest") != canonical_digest(binding, digest_field="binding_digest"):
        raise ValueError("beta_screening_team_binding_invalid")
    manifest = validate_embodiment_manifest(binding["manifest"])
    if interface == "openwam_piper_eef20.v1":
        from .openwam_piper_runtime import interface_contract
        return interface_contract(spec)
    if interface == "dreamlabs_dream1.v1":
        from .dreamlabs_policy_client import DreamLabsPolicySpec, validate_robot_contract

        policy = DreamLabsPolicySpec(**spec["policy_spec"])
        config = {key: spec["policy_spec"][key] for key in (
            "model_id", "camera_roles", "action_space", "action_hz", "observation_schema", "robot", "record",
            "sdk_version", "sdk_wheel_sha256") if key in spec["policy_spec"]}
        validate_robot_contract(manifest, config)
        dimensions = (176, 320)
        observation = "dream_modern_eef.v1"
        action = "named_arm_pose_chunk.v1"
        preprocessing, encoding, history = "resize_jpeg_quality85", "jpeg", 9
        roles = list(policy.camera_roles)
    else:
        from .team_policy_https_client import TeamPolicySpec

        policy = TeamPolicySpec(**spec["policy_spec"])
        dimensions_set = {(camera["height"], camera["width"]) for camera in manifest["cameras"]}
        if len(dimensions_set) != 1:
            raise ValueError("beta_screening_team_mixed_camera_dimensions_require_profile")
        dimensions = next(iter(dimensions_set))
        observation = "blueprint_named_camera_joint_base_state.v1"
        action = "blueprint_joint_position_gripper_closure.v1"
        preprocessing, encoding, history = "identity", "rgb_uint8", 1
        roles = camera_names(manifest)
    policy.validate()
    if (policy.embodiment_id != manifest["embodiment_id"]
            or policy.camera_names != camera_names(manifest)
            or policy.action_chunk_rows != manifest["action_chunk_rows"]
            or policy.open_loop_horizon != spec["open_loop_horizon"]
            or policy.model_identity_digest != spec["checkpoint_digest"]):
        raise ValueError("beta_screening_team_policy_robot_mismatch")
    return {"policy_interface_id": interface, "robot_profile_id": manifest["embodiment_id"],
            "observation_profile_id": observation, "action_schema_id": action,
            "control_hz": manifest["control_hz"], "action_horizon": policy.action_chunk_rows,
            "execution_horizon": policy.open_loop_horizon, "camera_roles": roles,
            "image_height": dimensions[0], "image_width": dimensions[1], "image_encoding": encoding,
            "image_preprocessing": preprocessing, "history_frames": history, "normalization_digest": None}


def team_interface_matches(configured: Mapping[str, Any], spec: Mapping[str, Any]) -> bool:
    expected = team_policy_interface(spec)
    runtime = configured.get("runtime") or {}
    policy = spec["policy_spec"]
    if spec["policy_interface_id"] == "openwam_piper_eef20.v1":
        from .openwam_piper_runtime import MODEL_SOURCE, MODEL_REVISION
        return (configured.get("interface") == expected
                and configured["policy"]["kind"] == "checkpoint"
                and configured["policy"]["source"] == MODEL_SOURCE
                and configured["policy"]["revision"] == MODEL_REVISION
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
    if interface == "openwam_piper_eef20.v1":
        from .openwam_piper_runtime import PORT
        spec.update(policy_endpoint={"host": "127.0.0.1", "port": PORT},
                    checkpoint_identity_basis="pinned_file_inventory_verified_before_worker_load",
                    weights_identity_verified=False)
    team_policy_interface(spec)
    spec["execution_spec_digest"] = canonical_digest(spec, digest_field="execution_spec_digest")
    return spec, dict(registration["model"])


def _registration_rights(registration: Mapping[str, Any], *, candidate_id: str) -> dict[str, Any]:
    from .team_candidate_quick10_admission import validate_candidate_rights

    rights = registration["rights_binding"]
    validate_candidate_rights(rights, team_id=registration["team_id"],
        checkpoint_id=registration["checkpoint_id"], candidate_id=candidate_id)
    return dict(rights)


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


def catalog_for_profile(profile: Mapping[str, Any], base_catalog: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Publish exact registered configurations and retain invalid ones with reasons.

    Runtime registration is reusable across tasks. Its revision participates in
    campaign identity, so installing a model, correcting a registration or
    completing private credential delivery causes a fresh bounded plan. The
    coordinator still accounts for every previous campaign against the site.
    """
    from .beta_policy_candidate_catalog import load_default_catalog, seal_catalog
    from .beta_policy_screening_launch import _bound_path, _read
    from .team_candidate_quick10_vast import credential_file_blockers

    base = dict(base_catalog) if base_catalog is not None else load_default_catalog()
    configured = profile.get("screening_candidates") or {}
    registered = {key: row for key, row in configured.items() if isinstance(row, Mapping) and row.get("robot_policy")}
    if not registered:
        return base
    rows = {row["candidate_id"]: row for row in base["candidates"]}
    revisions = {}
    for key, settings in registered.items():
        registration = settings["robot_policy"]
        try:
            rights = _registration_rights(registration, candidate_id=key)
            manifest = validate_embodiment_manifest(_read(_bound_path(registration["manifest"])))
            from .team_embodiment import verify_model_file
            verify_model_file(manifest, _bound_path(registration["model"]))
            policy = registration["policy_spec"]
            # Interface derivation needs no scene and creates no runtime grant.
            binding = {"manifest": manifest}
            binding["binding_digest"] = canonical_digest(binding, digest_field="binding_digest")
            interface = team_policy_interface({"team_embodiment": binding, "policy_spec": policy,
                "policy_interface_id": policy.get("policy_interface_id", "blueprint_policy_https.v1"),
                "checkpoint_digest": policy["model_identity_digest"], "open_loop_horizon": policy["open_loop_horizon"]})
            if interface["policy_interface_id"] == "dreamlabs_dream1.v1":
                from .dreamlabs_policy_client import validate_provider_model_record
                metadata = validate_provider_model_record(registration["provider_model_metadata"], model_id=policy["model_id"])
                if metadata["model_identity_digest"] != policy["model_identity_digest"]:
                    raise ValueError("beta_screening_dream_provider_identity_changed")
            blockers = (["beta_screening_hosted_credential_not_private"]
                        if credential_file_blockers(settings.get("credential_file")) else [])
            local = interface["policy_interface_id"] == "openwam_piper_eef20.v1"
            if local:
                from .openwam_piper_runtime import MINIMUM_GPU_MEMORY_GB, MODEL_SOURCE, MODEL_REVISION
                blockers = []
                memory = registration.get("minimum_gpu_memory_gb", MINIMUM_GPU_MEMORY_GB)
                if type(memory) not in (int, float) or not math.isfinite(memory) or not 0 < memory <= 1024:
                    raise ValueError("openwam_gpu_memory_configuration_invalid")
                policy_identity = {"kind": "checkpoint", "source": MODEL_SOURCE, "revision": MODEL_REVISION,
                    "digest": policy["model_identity_digest"], "digest_kind": "pinned_runtime_file_inventory"}
            else:
                policy_identity = {"kind": "hosted", "source": policy["allowed_origin"],
                    "revision": registration.get("policy_revision") or policy.get("model_id") or policy["policy_id"],
                    "digest": policy["model_identity_digest"], "digest_kind": "provider_metadata"}
            row = {"candidate_id": key, "display_name": registration.get("display_name", key),
                "policy": policy_identity,
                "interface": interface,
                "runtime": {"backend": "robolab.v1", "runtime_id": "native_task_arena_policy_canary.v1",
                    "minimum_gpu_memory_gb": (max(MINIMUM_GPU_MEMORY_GB, memory)
                                              if local else registration.get("minimum_gpu_memory_gb", 46))},
                "supported_task_kinds": ["rigid_pick_place"],
                "eligibility": {"enabled": not blockers, "commercial_use": "allowed_with_conditions",
                    "license_id": "operator-scoped-policy-license", "license_note": "Scoped run license: " + rights["rights_digest"],
                    "requires_credentials": ([] if local else ["DREAM_API_KEY" if interface["policy_interface_id"] == "dreamlabs_dream1.v1"
                                             else "BLUEPRINT_TEAM_POLICY_BEARER_FILE"]), "blockers": blockers}}
            deployment = (rows.get(key) or {}).get("deployment") or registration.get("deployment")
            if deployment is not None:
                if not isinstance(deployment, Mapping):
                    raise ValueError("beta_screening_policy_route_binding_mismatch")
                if deployment["configuration_id"] != manifest["embodiment_id"]:
                    raise ValueError("beta_screening_company_embodiment_binding_mismatch")
                route = deployment.get("policy_route_id")
                if route is not None:
                    if local and route != "customer_controller":
                        raise ValueError("openwam_policy_route_identity_mismatch")
                    # A shared robot or HTTPS transport does not identify a
                    # policy family. Retain explicit operator route provenance
                    # for checkpoint/planner/API variants; legacy generic
                    # controllers and the validated Dream contract remain valid.
                    declared = registration.get("policy_route_id")
                    supplied = registration.get("deployment")
                    if supplied is not None:
                        if not isinstance(supplied, Mapping):
                            raise ValueError("beta_screening_policy_route_binding_mismatch")
                        if (supplied.get("configuration_id"), supplied.get("policy_route_id")) != (
                                deployment["configuration_id"], route):
                            raise ValueError("beta_screening_policy_route_binding_mismatch")
                        declared = declared or supplied.get("policy_route_id")
                    dream = interface["policy_interface_id"] == "dreamlabs_dream1.v1"
                    if dream != (route == "dreamlabs_dream1"):
                        raise ValueError("beta_screening_policy_route_interface_mismatch")
                    if declared is None and route in {"customer_controller", "dreamlabs_dream1"}:
                        declared = route
                    if declared != route:
                        raise ValueError("beta_screening_policy_route_binding_mismatch")
                row["deployment"] = deployment
            # A malformed optional display/memory/revision field is this
            # registration's failure, not a reason to discard the catalog.
            row = seal_catalog({**base, "candidates": [row]})["candidates"][0]
            rows[key] = row
            revisions[key] = {"registration": registration, "settings": {name: value for name, value in settings.items()
                if name not in {"robot_policy", "credential_file"}}, "blockers": blockers,
                "credential_reference_digest": canonical_digest({"path": settings.get("credential_file")})}
        except (ValueError, OSError, KeyError, TypeError) as exc:
            import re
            code = str(exc).split(":", 1)[0]
            reason = code if re.fullmatch(r"[a-z][a-z0-9_]{1,127}", code) else "beta_screening_candidate_configuration_invalid"
            if key in rows:
                row = {**rows[key], "eligibility": {**rows[key]["eligibility"], "enabled": False,
                    "blockers": [reason]}}
            else:
                row = {"candidate_id": key, "display_name": key, "policy": {"kind": "hosted",
                    "source": "operator-registration", "revision": None, "digest": None, "digest_kind": "unverified"},
                    "interface": None, "runtime": None, "supported_task_kinds": [],
                    "eligibility": {"enabled": False, "commercial_use": "under_review", "license_id": "unverified",
                        "license_note": "The operator registration could not be validated.",
                        "requires_credentials": [], "blockers": [reason]}}
            rows[key] = row
            revisions[key] = {"status": reason}
    return seal_catalog({**base, "candidates": list(rows.values()),
                         "revision": base["revision"] + ":registrations:" + canonical_digest(revisions)[7:]})
