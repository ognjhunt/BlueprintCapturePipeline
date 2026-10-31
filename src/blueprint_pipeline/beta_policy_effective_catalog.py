"""The effective beta screening catalog: packaged candidates plus operator registrations.

Each registered robot/policy configuration is published with its exact
identity, and a registration that cannot be validated is retained with its
reason rather than dropped. This is the single implementation; screening
planning, the worker and the website's candidate-readiness preview all call it
(``beta_policy_screening_team`` re-exports it unchanged).

Deriving a catalog row reopens the sealed manifest and model bytes, checks run
rights and the private credential file's metadata, and derives the served
interface without a scene. It creates no plan, grant, reservation or runtime
binding. Its import graph excludes scene compilation, providers, GPU runtimes,
the paid-resource allocator and screening launch, so the CPU intake service can
read it (``tests/test_live_pipeline_import_isolation.py``).
"""
from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

from .beta_policy_candidate_catalog import load_default_catalog, seal_catalog
from .beta_policy_screening_files import _bound_path, _read
from .beta_policy_screening_g1_catalog import catalog_for_profile as g1_catalog_for_profile
from .decision_evidence_contracts import canonical_digest
from .policy_interface_binding import TEAM_ROBOT_INTERFACES, TEAM_LOCAL_INTERFACES, TEAM_SCRIPTED_INTERFACES
from .team_candidate_credential_file import credential_file_blockers
from .team_candidate_rights import validate_candidate_rights
from .team_embodiment import camera_names, validate_embodiment_manifest
from .team_local_policy import runtime_for_interface


def team_policy_interface(spec: Mapping[str, Any]) -> dict[str, Any]:
    """The exact observations/actions served on the spec's declared robot."""
    interface = spec.get("policy_interface_id")
    if interface not in TEAM_ROBOT_INTERFACES:
        raise ValueError("beta_screening_team_interface_unsupported")
    binding = spec.get("team_embodiment") or {}
    if binding.get("binding_digest") != canonical_digest(binding, digest_field="binding_digest"):
        raise ValueError("beta_screening_team_binding_invalid")
    manifest = validate_embodiment_manifest(binding["manifest"])
    if interface in TEAM_SCRIPTED_INTERFACES:
        from .scripted_robot_runtime_dispatch import runtime_for_policy
        return runtime_for_policy(spec["policy_spec"]).interface_contract(spec)
    if interface in TEAM_LOCAL_INTERFACES:
        return runtime_for_interface(interface).interface_contract(spec)
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


def _registration_rights(registration: Mapping[str, Any], *, candidate_id: str) -> dict[str, Any]:
    rights = registration["rights_binding"]
    validate_candidate_rights(rights, team_id=registration["team_id"],
        checkpoint_id=registration["checkpoint_id"], candidate_id=candidate_id)
    return dict(rights)


def catalog_for_profile(profile: Mapping[str, Any], base_catalog: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Publish exact registered configurations and retain invalid ones with reasons.

    Runtime registration is reusable across tasks. Its revision participates in
    campaign identity, so installing a model, correcting a registration or
    completing private credential delivery causes a fresh bounded plan. The
    coordinator still accounts for every previous campaign against the site.
    """
    base = dict(base_catalog) if base_catalog is not None else load_default_catalog()
    base = g1_catalog_for_profile(profile, base)
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
            scripted = interface["policy_interface_id"] in TEAM_SCRIPTED_INTERFACES
            local = interface["policy_interface_id"] in TEAM_LOCAL_INTERFACES or scripted
            if local:
                if scripted:
                    from .scripted_robot_runtime_dispatch import runtime_for_policy
                    module = runtime_for_policy(policy)
                else:
                    module = runtime_for_interface(interface["policy_interface_id"])
                minimum_memory = module.MINIMUM_GPU_MEMORY_GB
                blockers = []
                memory = registration.get("minimum_gpu_memory_gb", minimum_memory)
                if type(memory) not in (int, float) or not math.isfinite(memory) or not 0 < memory <= 1024:
                    raise ValueError("team_local_gpu_memory_configuration_invalid")
                policy_identity = {"kind": "scripted" if scripted else "checkpoint", "source": module.MODEL_SOURCE,
                    "revision": module.MODEL_REVISION, "digest": policy["model_identity_digest"],
                    "digest_kind": "source_code_inventory" if scripted else "pinned_runtime_file_inventory"}
            else:
                policy_identity = {"kind": "hosted", "source": policy["allowed_origin"],
                    "revision": registration.get("policy_revision") or policy.get("model_id") or policy["policy_id"],
                    "digest": policy["model_identity_digest"], "digest_kind": "provider_metadata"}
            row = {"candidate_id": key, "display_name": registration.get("display_name", key),
                "policy": policy_identity,
                "interface": interface,
                "runtime": {"backend": "robolab.v1", "runtime_id": "native_task_arena_policy_canary.v1",
                    "minimum_gpu_memory_gb": (max(minimum_memory, memory)
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
                    expected_route = ("blueprint_scripted_baseline" if scripted else
                                      "pi05_custom_checkpoint" if interface["policy_interface_id"] == "pi05_trlc_joint14.v1" else
                                      "customer_controller")
                    if local and route != expected_route:
                        raise ValueError("team_local_policy_route_identity_mismatch")
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
                    if declared is None and route in {"customer_controller", "dreamlabs_dream1", "blueprint_scripted_baseline"}:
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
                declared_policy = registration.get("policy_spec") if isinstance(registration, Mapping) else None
                scripted = isinstance(declared_policy, Mapping) and declared_policy.get("policy_interface_id") in TEAM_SCRIPTED_INTERFACES
                row = {"candidate_id": key, "display_name": key, "policy": {"kind": "scripted" if scripted else "hosted",
                    "source": "operator-registration", "revision": None, "digest": None, "digest_kind": "unverified"},
                    "interface": None, "runtime": None, "supported_task_kinds": [],
                    "eligibility": {"enabled": False, "commercial_use": "under_review", "license_id": "unverified",
                        "license_note": "The operator registration could not be validated.",
                        "requires_credentials": [], "blockers": [reason]}}
            rows[key] = row
            revisions[key] = {"status": reason}
    return seal_catalog({**base, "candidates": list(rows.values()),
                         "revision": base["revision"] + ":registrations:" + canonical_digest(revisions)[7:]})


__all__ = ["catalog_for_profile", "team_policy_interface"]
