"""Native whole-body G1 capabilities and private beta result projection."""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .native_g1_beta_authority import (
    CANDIDATE_DIGESTS, INVENTORY_SHA256, NAVIGATION_START_OFFSETS, NAVIGATION_TASK_KIND, SOURCE_REVISION, SONIC,
    START_OFFSET_FAMILY, ROBOT_USD_SHA256,
    validate_beta_authority, validate_cell_binding as validate_cell_binding, validate_frozen_cell,
)
from .native_g1_beta_authority_composer import compose_beta_authority
# Candidate identities and the registration catalog revision are shared with
# the effective catalog, which the CPU intake lane reads.
from .beta_policy_screening_g1_catalog import CANDIDATES, catalog_for_profile as catalog_for_profile

BACKEND = "native_g1.v1"
SPEC_SCHEMA = "native_g1_beta_execution_spec.v1"
RESULT_SCHEMA = "native_g1_beta_session_result.v1"
INTERFACE = "humanoidarena_semantic_v3"
MINIMUM_GPU_MEMORY_GB = 48  # Existing native G1 lane floor; not measured peak memory.
REPOSITORY = Path(__file__).resolve().parents[2]


def inventory() -> dict[str, Any]:
    path = REPOSITORY / "configs/g1_humanoidarena_checkpoint_inventory.v1.json"
    data = path.read_bytes()
    if "sha256:" + hashlib.sha256(data).hexdigest() != INVENTORY_SHA256:
        raise ValueError("g1_beta_checkpoint_inventory_changed")
    return json.loads(data)


def checkpoint(candidate_id: str) -> dict[str, Any]:
    from .native_g1_run_preflight import _candidate
    row = _candidate(inventory(), candidate_id)
    if row["inventory_digest"] != CANDIDATE_DIGESTS.get(candidate_id):
        raise ValueError("g1_beta_candidate_identity_changed")
    return row


def interface_contract(candidate_id: str) -> dict[str, Any]:
    row = checkpoint(candidate_id)
    return {"policy_interface_id": INTERFACE, "robot_profile_id": "unitree_g1_dex3_sonic_v1",
        "observation_profile_id": "humanoidarena_head_rgb_state64_v1", "action_schema_id": INTERFACE,
        "control_hz": 50, "action_horizon": 20, "execution_horizon": 20, "camera_roles": ["head"],
        "image_width": 640, "image_height": 480, "image_encoding": "rgb_uint8",
        "image_preprocessing": "publisher_checkpoint_preprocessor_once", "history_frames": 1,
        "normalization_digest": canonical_digest({"files": [f for f in row["files"]
            if f["path"].startswith("policy_preprocessor") or f["path"].startswith("policy_postprocessor")]})}


def runtime_contract() -> dict[str, Any]:
    return {"backend": BACKEND, "runtime_id": "native_g1_beta_session.v1",
            "minimum_gpu_memory_gb": MINIMUM_GPU_MEMORY_GB}


def catalog_candidates() -> list[dict[str, Any]]:
    rows = []
    for candidate_id in CANDIDATES:
        native = checkpoint(candidate_id)
        rows.append({"candidate_id": candidate_id,
            "display_name": "HumanoidArena " + ("π0.5" if "_pi05_" in candidate_id else "DP")
                + " G1 / Dex3 / SONIC " + ("navigation" if candidate_id.endswith("_vision_navi") else "manipulation"),
            "policy": {"kind": "checkpoint", "source": "https://modelscope.cn/models/Twang2026/HumanoidArena_models",
                "revision": "sha256-pinned-file-inventory:" + native["inventory_digest"][7:],
                "digest": native["inventory_digest"], "digest_kind": "pinned_runtime_file_inventory"},
            "interface": interface_contract(candidate_id), "runtime": runtime_contract(),
            # The navigation checkpoints also run a navigation scene, the G1 at its compiled start.
            "supported_task_kinds": (["rigid_pick_place", NAVIGATION_TASK_KIND]
                                     if candidate_id.endswith("_vision_navi") else ["rigid_pick_place"]),
            "eligibility": {"enabled": True, "commercial_use": "allowed_with_conditions",
                "license_id": "publisher-apache-2-inherited-terms-and-sonic-oml",
                "license_note": "Existing exact model/source/SONIC terms reviewed for internal development. "
                    "A current task/capture authority and exact G1 scene remain required; no public media or model redistribution grant.",
                "requires_credentials": [], "blockers": []},
            "deployment": {"configuration_id": "unitree_g1_dex3_sonic_v1", "companies": ["Unitree"],
                "relationship": "public_controller_integration",
                "evidence_url": "https://github.com/William-wAng618/HumanoidArena/tree/" + SOURCE_REVISION},
            "configuration_digest": ""})
    return rows


#: Cells per G1 session (owner decision 2026-11-06): a cell takes up to 50 min measured (55 min cap), so
#: about four fit the 4 h session; the plan's first cells, with their shared ids and seeds.
CELLS_PER_SESSION = 4
#: The execution descriptor field naming a navigation row's dropped start offsets, each with its reason.
DROPPED_START_OFFSETS_FIELD = "g1_start_offsets_dropped"


def frozen_cells(cells: list[dict[str, Any]], dropped_start_offsets: Any = ()) -> list[dict[str, Any]]:
    """The session's cells: the plan's first cells as frozen reset seeds of the authored scene.

    A navigation scene's cells (``robot_start_offset``, one per start offset) are kept as planned, less any
    offset in ``dropped_start_offsets``: its spawn check overlaps, so its cell is not funded
    (``navigation_start_offsets``). ``dropped_start_offsets`` means nothing for any other cell.
    """
    if cells and all(cell.get("family") == START_OFFSET_FAMILY for cell in cells):
        dropped = set(dropped_start_offsets)
        rows = [validate_frozen_cell(deepcopy(cell)) for cell in cells[:CELLS_PER_SESSION]]
        return [row for row in rows if NAVIGATION_START_OFFSETS[row["resolved_scenario"]["ordinal"]][0] not in dropped]
    result = []
    for ordinal, cell in enumerate(cells[:CELLS_PER_SESSION]):
        scenario = {"family": "canonical_anchor", "ordinal": ordinal, "parameters": {}}
        digest = canonical_digest(scenario)
        result.append(validate_frozen_cell({**cell, "family": "canonical_anchor", "resolved_scenario": scenario,
                                           "cell_spec_digest": digest, "resolved_scenario_digest": digest}))
    return result


def _record(record: Mapping[str, Any]) -> dict[str, Any]:
    # Standard library only: the runtime on the rented machine calls this, and its Python has no jsonschema,
    # which beta_policy_screening_plan needs at load.
    from .beta_policy_screening_artifact_reference import artifact_reference
    return artifact_reference(record)


def validate_execution_spec(value: Mapping[str, Any], *, candidate_id: str | None = None) -> dict[str, Any]:
    try:
        spec = json.loads(json.dumps(value, allow_nan=False))
        candidate = spec["candidate_id"]
        if candidate_id is not None and candidate != candidate_id:
            raise ValueError("candidate mismatch")
        native = checkpoint(candidate)
        expected = {"schema_version", "candidate_id", "policy_interface_id", "evaluation_backend",
            "checkpoint_digest", "source_revision", "scene_plan_digest", "source", "site_id", "native_g1",
            "task_success_contract", "task_success_contract_digest", "rights_authority", "runtime_identity",
            "runtime_identity_digest", "execution_spec_digest", "claim_ceiling", "scenario_scope"}
        records = spec["native_g1"]
        if (set(spec) != expected or spec["schema_version"] != SPEC_SCHEMA
                or spec["policy_interface_id"] != INTERFACE or spec["evaluation_backend"] != BACKEND
                or spec["source_revision"] != SOURCE_REVISION or spec["checkpoint_digest"] != native["inventory_digest"]
                or spec["claim_ceiling"] != "development_only" or spec["scenario_scope"] != "native_frozen_reset_seeds"
                or set(records) - {"scene_packet", "publisher_source", "runtime_source", "model_terms_review", "navigation_authority"}
                or not {"scene_packet", "publisher_source", "runtime_source", "model_terms_review"} <= set(records)
                or spec["task_success_contract_digest"] != spec["task_success_contract"]["contract_digest"]
                or spec["execution_spec_digest"] != canonical_digest(spec, digest_field="execution_spec_digest")):
            raise ValueError("invalid")
        for record in records.values():
            _record(record)
        validate_beta_authority(spec["rights_authority"], candidate_id=candidate,
            scene_plan_digest=spec["scene_plan_digest"], current_source=spec["source"],
            task_success_contract_digest=spec["task_success_contract_digest"])
        if (records["model_terms_review"]["sha256"] != spec["rights_authority"]["model_review"]["artifact_sha256"]
                or spec["rights_authority"]["registration_digest"] != canonical_digest({"candidate_id": candidate, "native_g1": records})
                or spec["rights_authority"]["site_id"] != spec["site_id"]):
            raise ValueError("registration scope mismatch")
        if is_navigation_contract(spec["task_success_contract"]):
            # The task spec is the packet's; candidate_execution_spec checked the contract against it.
            _navigation_contract_shape(spec["task_success_contract"], candidate_id=candidate,
                                       task_id=spec["source"]["task_id"])
        else:
            from .adp_task_scoring import validate_rigid_task_success_contract
            validate_rigid_task_success_contract(spec["task_success_contract"],
                                                 expected_task_id=spec["source"]["task_id"])
        runtime = runtime_identity(spec)
        if spec["runtime_identity"] != runtime or spec["runtime_identity_digest"] != canonical_digest(runtime):
            raise ValueError("runtime mismatch")
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        raise ValueError("g1_beta_execution_spec_invalid") from exc
    return spec


def is_navigation_contract(contract: Mapping[str, Any]) -> bool:
    """Whether a success contract is a navigation scene's (``navigation_task_success_contract``): its own kind."""
    from .native_g1_navigation_goal import NAVIGATION_CONTRACT_SCHEMA_VERSION
    return (contract.get("schema_version") == NAVIGATION_CONTRACT_SCHEMA_VERSION
            or contract.get("task_kind") == NAVIGATION_TASK_KIND)


def _navigation_contract_shape(contract: Mapping[str, Any], *, candidate_id: str, task_id: str) -> None:
    """A navigation contract as an execution spec can check it without the task spec: sealed, scoped, its kind's.

    ``validate_navigation_task_success_contract`` re-derives it from the packet's task spec when the spec is
    composed (``candidate_execution_spec``); only a navigation checkpoint runs one.
    """
    from .native_g1_navigation_goal import NAVIGATION_CONTRACT_SCHEMA_VERSION, NAVIGATION_CRITERION
    if (not candidate_id.endswith("_vision_navi")
            or contract.get("schema_version") != NAVIGATION_CONTRACT_SCHEMA_VERSION
            or contract.get("task_kind") != NAVIGATION_TASK_KIND
            or (contract.get("scope") or {}).get("task_id") != task_id
            or contract.get("criterion") != NAVIGATION_CRITERION
            or contract.get("claim_ceiling") != "development_only"
            or contract.get("physical_outcome_claimed") is not False
            or contract.get("contract_digest") != canonical_digest(contract, digest_field="contract_digest")):
        raise ValueError("g1_beta_navigation_contract_invalid")


def runtime_identity(spec: Mapping[str, Any]) -> dict[str, Any]:
    from .native_task_isaaclab_launch import NATIVE_TASK_ARENA_IMAGE
    return {"container_image": NATIVE_TASK_ARENA_IMAGE, "source_revision": SOURCE_REVISION,
        "checkpoint_digest": spec["checkpoint_digest"], "interface": interface_contract(spec["candidate_id"]),
        "sonic": dict(SONIC), "robot_usd_sha256": ROBOT_USD_SHA256,
        "scene_plan_digest": spec["scene_plan_digest"], "native_g1": spec["native_g1"],
        "rights_authority_digest": spec["rights_authority"]["authority_digest"]}


def candidate_execution_spec(*, candidate_id: str, registration: Mapping[str, Any],
                             scene_plan: Mapping[str, Any], base_packet: Mapping[str, Any],
                             source: Mapping[str, Any], contract: Mapping[str, Any], site_id: str) -> dict[str, Any]:
    from .beta_policy_screening_launch import _bound_path, _read, _record as file_record
    from .native_task_arena_bundle import verify_native_task_arena_packet
    from .native_g1_embodiment import validate_g1_robot_plan
    from .native_task_arena_runtime import _resolve_portable_robot
    from .native_g1_scene_packet_request import ROBOT_TASK_FIELDS
    from .native_g1_navigation_goal import validate_g1_navigation_goal_authority
    from .native_g1_publisher_source_stage import verify_g1_publisher_source
    from .native_task_runtime_source_packet import verify_native_task_runtime_source_packet

    native = checkpoint(candidate_id)
    records = {key: _record(value) for key, value in registration.items()}
    if set(records) - {"scene_packet", "publisher_source", "runtime_source", "navigation_authority"}:
        raise ValueError("g1_beta_registration_invalid")
    for key in ("scene_packet", "publisher_source", "runtime_source"):
        if key not in records:
            raise ValueError("g1_beta_" + key + "_missing")
    packet_root, receipt, _ = verify_native_task_arena_packet(_bound_path(records["scene_packet"]).parent)
    g1 = _read(packet_root / "native_task_arena_scene_plan.v1.json")
    base_root, base_receipt, _ = verify_native_task_arena_packet(_bound_path(base_packet).parent)
    base_request = _read(base_root / "native_task_arena_packet_request.v1.json")
    g1_request = _read(packet_root / "native_task_arena_packet_request.v1.json")
    derivation = g1_request.get("g1_scene_derivation") or {}
    if (derivation.get("source_packet_receipt_digest") != base_receipt["receipt_digest"]
            or derivation.get("source_scene_plan_digest") != base_receipt["arena_scene_plan_digest"]
            or any(g1_request.get(key) != base_request.get(key) for key in
                   ("scene_id", "task_id", "task_joint_bindings", "task_state_binding", "appearance_variant"))
            or [{k: v for k, v in row.items() if k != "source"} for row in g1_request["assets"]]
                != [{k: v for k, v in row.items() if k != "source"} for row in base_request["assets"]]):
        raise ValueError("g1_beta_source_scene_derivation_mismatch")
    def asset_identity(row):
        return sorted((f["semantic_role"], f["staged_sha256"], f["staged_size_bytes"])
                      for f in row["source_bindings"])
    if asset_identity(receipt) != asset_identity(base_receipt):
        raise ValueError("g1_beta_source_scene_assets_mismatch")
    task = g1["task_spec"]
    if scene_plan["task_spec"].get("task_success_contract") not in (None, contract):
        raise ValueError("g1_beta_current_task_binding_mismatch")
    source_task = {**scene_plan["task_spec"], "task_success_contract": dict(contract),
                   "task_success_contract_digest": contract["contract_digest"]}
    if candidate_id.endswith("_vision_navi") and source_task.get("g1_navigation_goal") is None:
        raise ValueError("g1_beta_current_navigation_goal_authority_missing")
    # A navigation scene (the G1 at its compiled start) runs only a navigation checkpoint, and only when the G1
    # plan and the base plan are both that scene's.
    navigation = g1.get("task_kind") == NAVIGATION_TASK_KIND
    if navigation and not candidate_id.endswith("_vision_navi"):
        raise ValueError("g1_beta_manipulation_policy_task_mismatch")
    if navigation and not (scene_plan.get("task_kind") == NAVIGATION_TASK_KIND
                           and (base_request.get("task_spec") or {}).get("task_kind") == NAVIGATION_TASK_KIND):
        raise ValueError("g1_beta_current_task_binding_mismatch")
    if navigation:
        from .native_g1_navigation_goal import validate_navigation_task_success_contract
        try:
            validate_navigation_task_success_contract(contract, task_spec=task)
        except ValueError as exc:
            raise ValueError("g1_beta_current_task_binding_mismatch") from exc
    if (g1.get("task_id") != source["task_id"] or g1.get("scene_id") != site_id
            or g1.get("task_kind") not in ("rigid_pick_place", NAVIGATION_TASK_KIND)
            or task.get("task_success_contract") != contract
            or task.get("task_success_contract_digest") != contract["contract_digest"]
            or set(task) != set(source_task)
            or any(task[k] != source_task[k] for k in task if k not in ROBOT_TASK_FIELDS)
            or task.get("control_frequency_hz") != 50):
        raise ValueError("g1_beta_current_task_binding_mismatch")
    robot = validate_g1_robot_plan(_resolve_portable_robot(g1, bundle_root=packet_root), verify_asset=False)
    if robot["usd_sha256"] != ROBOT_USD_SHA256:
        raise ValueError("g1_beta_robot_asset_mismatch")
    cameras = [row for row in g1.get("cameras", []) if row.get("role") == "head"]
    if (len(cameras) != 1 or not cameras[0].get("policy_input")
            or cameras[0].get("intrinsics", {}).get("width") != 640
            or cameras[0].get("intrinsics", {}).get("height") != 480):
        raise ValueError("g1_beta_head_camera_binding_missing")
    if candidate_id.endswith("_vision_navi"):
        if "navigation_authority" not in records:
            raise ValueError("g1_beta_current_navigation_goal_authority_missing")
        validate_g1_navigation_goal_authority(_read(_bound_path(records["navigation_authority"])), plan=g1)
    elif task.get("g1_navigation_goal") is not None:
        raise ValueError("g1_beta_manipulation_policy_task_mismatch")
    # Money: no G1 run starts with its body measured inside the room, as no arm run starts from rest
    # overlap (scene_physics_preflight.require_scene_physics_admission). Unmeasured is only a flag
    # (setup_quality_flags).
    # Money too: no G1 run starts with no collision floor under either foot (``FLOOR_ABSENT_REASON``); it
    # falls through the scene and measures nothing. The packet's collision facts are kept once, as counts.
    from .native_g1_spawn_clearance import (FLOOR_ABSENT_REASON, OVERLAP_REASON, ensure_background_diagnostic,
                                            packet_floor_absent, spawn_clearance_status)
    if navigation:
        # A start offset that overlaps, or has no floor, drops only its own cell (``navigation_start_offsets``);
        # with every one dropped there is no cell left to fund.
        dropped = navigation_start_offsets(packet_root, g1["plan_digest"])[0]
        if FLOOR_ABSENT_REASON in dropped.values():
            ensure_background_diagnostic(packet_root, g1["plan_digest"])
        if len(dropped) == len(NAVIGATION_START_OFFSETS):
            raise ValueError(FLOOR_ABSENT_REASON if set(dropped.values()) == {FLOOR_ABSENT_REASON}
                             else OVERLAP_REASON)
    elif spawn_clearance_status(packet_root, g1["plan_digest"]) == "overlap":
        raise ValueError(OVERLAP_REASON)
    elif packet_floor_absent(packet_root, g1["plan_digest"]):
        ensure_background_diagnostic(packet_root, g1["plan_digest"])
        raise ValueError(FLOOR_ABSENT_REASON)
    verify_g1_publisher_source(_bound_path(records["publisher_source"]).parent)
    runtime = verify_native_task_runtime_source_packet(_bound_path(records["runtime_source"]))
    if runtime.get("runtime_profile") != "unitree_g1" or runtime.get("redistribution_permitted") is not True:
        raise ValueError("g1_beta_native_runtime_source_invalid")
    review_path = REPOSITORY / "docs/arm_decision_proof_v1/manifests" / f"g1_841757_{candidate_id}_development_rights_review.v1.json"
    records["model_terms_review"] = file_record(review_path)
    authority = compose_beta_authority(candidate_id=candidate_id, scene_plan_digest=g1["plan_digest"],
        source_packet_receipt_digest=base_receipt["receipt_digest"], task_success_contract_digest=contract["contract_digest"],
        checkpoint_digest=native["inventory_digest"], current_source=source, site_id=site_id,
        registration_digest=canonical_digest({"candidate_id": candidate_id, "native_g1": records}))
    spec = {"schema_version": SPEC_SCHEMA, "candidate_id": candidate_id, "policy_interface_id": INTERFACE,
        "evaluation_backend": BACKEND, "checkpoint_digest": native["inventory_digest"], "source_revision": SOURCE_REVISION,
        "scene_plan_digest": g1["plan_digest"], "source": dict(source), "site_id": site_id,
        "native_g1": records, "task_success_contract": dict(contract), "task_success_contract_digest": contract["contract_digest"],
        "rights_authority": authority, "claim_ceiling": "development_only", "scenario_scope": "native_frozen_reset_seeds"}
    spec["runtime_identity"] = runtime_identity(spec)
    spec["runtime_identity_digest"] = canonical_digest(spec["runtime_identity"])
    spec["execution_spec_digest"] = canonical_digest(spec, digest_field="execution_spec_digest")
    return validate_execution_spec(spec, candidate_id=candidate_id)


def navigation_start_offsets(packet_dir: Path, plan_digest: str) -> tuple[dict[str, str], list[str]]:
    """A navigation packet's dropped start offsets, each with its reason, and the flags its funded cells carry.

    Each offset reads its own spawn check (``native_g1_spawn_clearance.start_offset_reports``). One that
    overlaps the room is dropped from the session: its cell is not funded (``g1_spawn_overlap``), the row
    keeps the others and also carries ``g1_spawn_overlap`` as a flag. Of the rest, one not measured flags
    ``g1_spawn_clearance_unmeasured`` and one with a foot that has no floor under it ``g1_spawn_foot_unsupported``
    (``unsupported_feet``); neither refuses. With no per-offset report every offset is unmeasured, except
    that the packet's own report, of the start itself, still counts.
    """
    from .native_g1_spawn_clearance import (
        FLOOR_ABSENT_REASON, OVERLAP_REASON, UNMEASURED_REASON, UNSUPPORTED_FOOT_REASON, floor_absent,
        packet_floor_absent, spawn_clearance_status, spawn_setup_flags, start_offset_reports, unsupported_feet)
    rows = {row["offset_id"]: row for row in start_offset_reports(packet_dir, plan_digest) or []}
    dropped, flags = {}, set()
    for offset_id, _, _ in NAVIGATION_START_OFFSETS:
        row = rows.get(offset_id)
        status = row["status"] if row is not None and row.get("status") in ("clear", "overlap") else "unmeasured"
        packet_status = spawn_clearance_status(packet_dir, plan_digest) if offset_id == "start" else None
        if packet_status == "overlap" or (row is None and packet_status == "clear"):
            status = packet_status
        if status == "overlap":
            dropped[offset_id] = OVERLAP_REASON
        elif status == "unmeasured":
            flags.add(UNMEASURED_REASON)
        elif floor_absent(row) if row is not None else packet_floor_absent(packet_dir, plan_digest):
            dropped[offset_id] = FLOOR_ABSENT_REASON
        elif (unsupported_feet(row) if row is not None
              else UNSUPPORTED_FOOT_REASON in spawn_setup_flags(packet_dir, plan_digest)):
            flags.add(UNSUPPORTED_FOOT_REASON)
    flags.update(dropped.values())
    return dropped, sorted(flags)


def _navigation_spec(spec: Mapping[str, Any]) -> bool:
    return is_navigation_contract(spec["task_success_contract"])


def setup_quality_flags(spec: Mapping[str, Any]) -> list[str]:
    """Typed flags a funded G1 row carries into its plan descriptor and result: a spawn not measured, or
    a foot with no floor under it (``native_g1_spawn_clearance.spawn_setup_flags``). Never a refusal.
    A navigation row's are its start offsets' (``navigation_start_offsets``), plus
    ``g1_goal_marker_not_in_initial_view`` when its goal disc starts outside the head camera's frame. A scene floor that is a
    development-only estimate adds ``g1_scene_floor_estimated`` (``packet_floor_estimate``): the spawn
    readings are then against a flat stand-in floor."""
    from .beta_policy_screening_launch import _bound_path
    from .native_g1_spawn_clearance import (FLOOR_ESTIMATED_REASON, goal_view_flags, packet_floor_estimate,
                                            spawn_setup_flags)
    packet = _bound_path(spec["native_g1"]["scene_packet"]).parent
    if _navigation_spec(spec):
        flags = set(navigation_start_offsets(packet, spec["scene_plan_digest"])[1])
        # The goal marker out of the head camera's frame at reset (``goal_view_flags``): a flag, never a gate.
        flags.update(goal_view_flags(packet, spec["scene_plan_digest"]))
    else:
        flags = set(spawn_setup_flags(packet, spec["scene_plan_digest"]))
    if packet_floor_estimate(packet, spec["scene_plan_digest"]) is not None:
        flags.add(FLOOR_ESTIMATED_REASON)
    return sorted(flags)


def dropped_start_offsets(spec: Mapping[str, Any]) -> dict[str, str]:
    """The start offsets a navigation row's session drops (each ``g1_spawn_overlap``); none for any other row."""
    from .beta_policy_screening_launch import _bound_path
    if not _navigation_spec(spec):
        return {}
    return navigation_start_offsets(_bound_path(spec["native_g1"]["scene_packet"]).parent, spec["scene_plan_digest"])[0]


def _sealed(value: Any, schema: str, digest_field: str) -> dict[str, Any]:
    if (not isinstance(value, dict) or value.get("schema_version") != schema
            or value.get(digest_field) != canonical_digest(value, digest_field=digest_field)):
        raise ValueError("g1_beta_retained_evidence_invalid")
    return value


def _witnessed_queries(row: Mapping[str, Any], *, candidate_id: str) -> int:
    from .native_g1_humanoidarena_policy_client import validate_semantic_v3_infer_response
    checkpoints = row.get("query_checkpoints", [])
    if not isinstance(checkpoints, list):
        raise ValueError("g1_beta_query_evidence_invalid")
    for index, checkpoint in enumerate(checkpoints):
        _sealed(checkpoint, "native_g1_policy_query_checkpoint.v1", "checkpoint_digest")
        query = checkpoint.get("query") or {}
        loaded = _sealed(checkpoint.get("checkpoint_load_observation"), "native_g1_checkpoint_load_observation.v1", "receipt_digest")
        actions = validate_semantic_v3_infer_response({"action_chunk": checkpoint.get("action_chunk")})
        if (checkpoint.get("candidate_id") != candidate_id
                or checkpoint.get("scene_plan_digest") != row.get("scene_plan_digest")
                or checkpoint.get("status") != "policy_response_retained"
                or query.get("query_index") != index or query.get("returned_action_count") != len(actions)
                or loaded.get("candidate_id") != candidate_id
                or loaded.get("candidate_inventory_digest") != CANDIDATE_DIGESTS[candidate_id]
                or loaded.get("source_revision") != SOURCE_REVISION
                or loaded.get("policy_type") != ("pi05" if "_pi05_" in candidate_id else "diffusion")
                or loaded.get("loader_completed") is not True or loaded.get("full_state_load_required") is not True
                or loaded.get("inference_observed") is not False
                or not re.fullmatch(r"sha256:[0-9a-f]{64}", str(loaded.get("log_sha256")))):
            raise ValueError("g1_beta_query_evidence_invalid")
    return len(checkpoints)


def _episode_outcome(row: Mapping[str, Any], *, candidate_id: str, contract_digest: str) -> tuple[str, str, bool]:
    witnessed = _witnessed_queries(row, candidate_id=candidate_id)
    if row.get("status") in {"blocked", "not_attempted"}:
        if row["status"] == "not_attempted" and witnessed:
            raise ValueError("g1_beta_unattempted_query_contradiction")
        reason = row.get("reason")
        if not isinstance(reason, str) or not re.fullmatch(r"g1_[a-z0-9_]{1,120}", reason):
            reason = "g1_native_episode_" + row["status"]
        return "void", reason, witnessed > 0
    if row.get("status") != "completed":
        raise ValueError("g1_beta_episode_status_invalid")
    worker = _sealed(row.get("worker_result"), "native_g1_development_worker_result.v1", "result_digest")
    supervised = _sealed(worker.get("supervised_episode"), "native_g1_supervised_built_scene_episode.v1", "result_digest")
    episode = _sealed(row.get("episode"), "native_g1_built_scene_policy_episode.v1", "result_digest")
    trace = _sealed(row.get("trace"), "native_g1_shared_scene_episode_trace.v1", "trace_digest")
    lease = _sealed(supervised.get("server_lease_receipt"), "native_g1_policy_server_lease.v1", "receipt_digest")
    for artifact in (worker, supervised, episode, trace, lease):
        if artifact.get("candidate_id") != candidate_id or artifact.get("scene_plan_digest") != row["scene_plan_digest"]:
            raise ValueError("g1_beta_episode_identity_mismatch")
    if (worker.get("status") != "completed_development_only"
            or not _worker_closed(worker, row)
            or supervised.get("status") != "completed_development_only"
            or supervised.get("server_teardown", {}).get("status") != "child_exited"
            or supervised.get("episode_result_digest") != episode["result_digest"]
            or episode.get("status") != "development_only_scored_episode"
            or episode.get("trace_digest") != trace["trace_digest"]
            or trace.get("seed") != row["seed"]
            or lease.get("status") != "server_ready_process_bound"
            or lease.get("source_revision") != SOURCE_REVISION
            or lease.get("candidate_inventory_digest") != CANDIDATE_DIGESTS[candidate_id]
            or lease.get("checkpoint_bytes_verified") is not True
            or lease.get("loaded_checkpoint_identity_observed") is not True
            or lease.get("listener_owner_verified") is not True):
        raise ValueError("g1_beta_episode_execution_mismatch")
    queries, steps = trace.get("queries"), trace.get("steps")
    if (not isinstance(queries, list) or not queries or trace.get("policy_query_count") != len(queries) or len(queries) != witnessed
            or not isinstance(steps, list) or not steps or trace.get("scene_step_count") != len(steps)
            or any(q.get("query_index") != index or type(q.get("returned_action_count")) is not int
                   or not 1 <= q["returned_action_count"] <= 64 for index, q in enumerate(queries))):
        raise ValueError("g1_beta_returned_actions_missing")
    for query, checkpoint in zip(queries, row["query_checkpoints"], strict=True):
        if (query.get("checkpoint_digest") != checkpoint["checkpoint_digest"]
                or checkpoint["checkpoint_load_observation"] != lease.get("checkpoint_load_observation")
                or any(query.get(key) != value for key, value in checkpoint["query"].items())):
            raise ValueError("g1_beta_query_trace_mismatch")
    navigation = candidate_id.endswith("_vision_navi")
    score = _sealed(episode.get("score"), "native_g1_navigation_goal_score.v1" if navigation else
                    "adp_rigid_task_scoring.v2", "score_digest" if navigation else "report_digest")
    if (score.get("status") != "scored" or (not navigation and score.get("task_success_contract_digest") != contract_digest)
            or episode.get("evaluation_task_kind") != ("g1_navigation_goal" if navigation else "rigid_pick_place")
            or any(artifact.get("ranking_eligible") is not False or artifact.get("physical_outcome_claimed") is not False
                   for artifact in (worker, supervised, episode))):
        raise ValueError("g1_beta_deterministic_score_invalid")
    if navigation and fell_through_scene(score):
        return "void", FELL_THROUGH_REASON, True
    success = score.get("outcome") == "success" if navigation else score.get("task_succeeded")
    if type(success) is not bool:
        raise ValueError("g1_beta_deterministic_score_invalid")
    return ("success" if success else "failure"), "g1_native_deterministic_task_score", True


#: A navigation episode whose robot left the scene through its floor measured the scene, not the
#: policy. The library job's first scored G1 run (2026-11-08) recorded 16,139 m and 14,653 m of
#: root height drift over 3,001 samples: free fall, with no collider under the robot.
FELL_THROUGH_REASON = "g1_robot_fell_through_scene"
#: G1's root stands about 0.75 m high, so a fall onto a floor moves it well under a metre and a
#: navigation scene asks for no climb; drift past this means nothing was under the robot.
FELL_THROUGH_DRIFT_M = 3.0


def fell_through_scene(score: Mapping[str, Any]) -> bool:
    """Whether a navigation score's root height drifted past any fall onto a floor."""
    drift = score.get("maximum_root_height_drift_observed_m")
    return isinstance(drift, (int, float)) and not isinstance(drift, bool) and (
        drift != drift or drift > FELL_THROUGH_DRIFT_M)


def _worker_closed(worker: Mapping[str, Any], row: Mapping[str, Any]) -> bool:
    if worker.get("teardown") == {"environment": "closed", "simulator": "closed"}:
        return True
    # Isaac can exit its interpreter inside close(). Reconstruct the existing
    # parent's recovery exactly from the retained preclose and observed exit.
    preclose = _sealed(row.get("worker_preclose"), "native_g1_development_worker_preclose.v1", "preclose_digest")
    exit_evidence = worker.get("process_exit_evidence") or {}
    if (exit_evidence != {"returncode": 0, "preclose_digest": preclose["preclose_digest"]}
            or type(exit_evidence.get("returncode")) is not int
            or preclose.get("status") != "awaiting_simulator_close" or preclose.get("blocker") is not None
            or preclose.get("teardown") != {"environment": "closed", "simulator": "close_requested"}
            or (preclose.get("supervised_episode") or {}).get("status") != "completed_development_only"):
        return False
    expected = {**{k: v for k, v in preclose.items() if k != "preclose_digest"},
        "schema_version": "native_g1_development_worker_result.v1", "status": "completed_development_only",
        "teardown": {"environment": "closed", "simulator": "process_exited_after_close_request"},
        "blocker": None, "process_exit_evidence": exit_evidence}
    expected["result_digest"] = canonical_digest(expected, digest_field="result_digest")
    return worker == expected


def summarize_result(value: Mapping[str, Any], *, plan: Mapping[str, Any],
                     candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Project exact G1 evidence into the existing private learned-policy envelope."""
    result = _sealed(dict(value), RESULT_SCHEMA, "result_digest")
    execution = candidate["execution_spec"]
    expected = {"candidate_id": candidate["candidate_id"], "plan_digest": plan["plan_digest"],
        "configuration_digest": candidate["configuration_digest"], "admission_digest": candidate["admission_digest"],
        "runtime_identity_digest": execution["runtime_digest"], "source": plan["source"],
        "task_success_contract_digest": execution["task_success_contract_digest"],
        "scenario_plan_digest": plan["scenario_plan_digest"], "claim_ceiling": "development_only"}
    if (any(result.get(k) != v for k, v in expected.items())
            or any(result.get(k) is not False for k in ("ranking_eligible", "qualification_permitted",
                "public_media_publication_authorized", "model_redistribution_authorized", "physical_outcome_claimed"))
            or not isinstance(result.get("episodes"), list)
            or len(result["episodes"]) != candidate["episode_count"]):
        raise ValueError("g1_beta_result_binding_mismatch")
    # The provider cannot attest its own absence. Only controller reconciliation
    # supplies this after the existing canonical scoped-closeout observations.
    closeout = result.get("session_closeout") or {}
    if (any(closeout.get(key) is not True for key in ("provider_zero_confirmed", "staged_objects_deleted", "watchdog_cancelled"))
            or closeout.get("scope") != "exact_instance_and_authority_resource_name"
            or closeout.get("global_provider_zero_claimed") is not False
            or closeout.get("receipt_digest") != canonical_digest(closeout, digest_field="receipt_digest")):
        raise ValueError("g1_beta_result_closeout_unverified")
    outcomes, learned = [], 0
    for row, cell in zip(result["episodes"], candidate["scenario_cells"], strict=True):
        if any(row.get(key) != cell[key] for key in ("cell_id", "seed", "cell_spec_digest", "resolved_scenario", "resolved_scenario_digest")):
            raise ValueError("g1_beta_result_cell_mismatch")
        outcome, reason, queried = _episode_outcome(row, candidate_id=candidate["candidate_id"],
                                                   contract_digest=execution["task_success_contract_digest"])
        learned += int(queried)
        outcomes.append({"cell_id": cell["cell_id"], "outcome": outcome, "reason": reason})
    succeeded = sum(row["outcome"] == "success" for row in outcomes)
    failed = sum(row["outcome"] == "failure" for row in outcomes)
    scored = succeeded + failed
    return {"episodes_scored": scored, "episodes_succeeded": succeeded, "episodes_failed": failed,
        "episodes_void": len(outcomes) - scored, "success_rate": succeeded / scored if scored else None,
        "session_result_digest": result["result_digest"], "episode_outcomes": outcomes,
        "execution_evidence_kind": "learned_policy", "action_source": "learned_policy",
        "scripted_rollout_count": 0, "learned_policy_rollout_count": learned,
        "scripted_controller_invoked": False, "candidate_policy_queried": learned > 0,
        # The plan row's setup flags travel with its estimate (ledger), as observation flags do.
        **({"setup_quality_flags": list(execution["setup_quality_flags"])}
           if execution.get("setup_quality_flags") else {})}
