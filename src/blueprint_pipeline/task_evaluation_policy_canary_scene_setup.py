"""No-spend Scene 839873 readiness and execution-setup materializer."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

from .adp009d_droid_observation import (
    CANDIDATE_REQUIRED_VIEWS,
    DROID_EXTERIOR_VIEW_1,
    DROID_WRIST_VIEW,
)
from .adp009d_policy_candidate_admission import EXPECTED_CANDIDATES
from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .native_task_arena_policy_bundle import _candidate_runtime_binding
from .native_task_isaaclab_launch import NATIVE_TASK_ARENA_IMAGE


SETUP_SCHEMA_VERSION = "task_evaluation_policy_canary_execution_setup.v1"
SPEC_SCHEMA_VERSION = "native_task_arena_policy_canary_execution_spec.v1"
DECISION_SCHEMA_VERSION = "task_evaluation_policy_canary_setup_preflight.v1"
RUN_KIND = "internal_policy_canary"
CLAIM_CEILING = "diagnostic_policy_execution"
SCENE_ID = "839873"
CANDIDATE_IDS = ("pi05_droid", "groot_n17_droid")
EMBODIMENT_ID = "franka_panda_robotiq_2f85_v1"
QUICK_FAMILY_COUNTS = {
    "canonical_anchor": 2,
    "placement_approach": 2,
    "illumination": 1,
    "camera_sensor": 1,
    "bounded_physics": 1,
    "admitted_object_material_cousin": 1,
    "pairwise_stress": 1,
    "held_out_composition": 1,
}
_SHA = re.compile(r"^[0-9a-f]{40}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


class PolicyCanarySetupError(ValueError):
    def __init__(self, blockers: list[str] | tuple[str, ...]):
        self.blockers = tuple(sorted(set(str(item) for item in blockers if str(item))))
        super().__init__(";".join(self.blockers))


def _read(path: str | Path, *, code: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PolicyCanarySetupError([code]) from exc
    if not isinstance(value, dict):
        raise PolicyCanarySetupError([code])
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _record(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if source.is_symlink() or not source.is_file():
        raise PolicyCanarySetupError(["policy_canary_setup_record_missing"])
    return {"path": str(source), "size_bytes": source.stat().st_size, "sha256": _sha256(source)}


def _quick_cells(scene_revision_digest: str) -> list[dict[str, Any]]:
    families = [family for family, count in QUICK_FAMILY_COUNTS.items() for _ in range(count)]
    parameters = [
        {},
        {},
        {"object_start_y_delta_m": -0.02},
        {"object_yaw_delta_degrees": 7.5},
        {"task_light_intensity_scale": 0.85},
        {"external_camera_x_delta_m": 0.02},
        {"dynamic_friction": 0.45},
        {"material_cousin": "admitted_same_geometry_matte"},
        {"object_start_y_delta_m": 0.015, "task_light_intensity_scale": 1.1},
        {
            "object_yaw_delta_degrees": -5.0,
            "external_camera_x_delta_m": -0.015,
            "dynamic_friction": 0.55,
        },
    ]
    cells = []
    for index, (family, resolved) in enumerate(zip(families, parameters, strict=True)):
        seed = int(hashlib.sha256(f"{scene_revision_digest}:{index}".encode()).hexdigest()[:8], 16)
        scenario = {"family": family, "ordinal": index, "parameters": resolved}
        cells.append(
            {
                "cell_id": f"scene839873.quick10.{index:02d}.{family}",
                "seed": seed,
                "family": family,
                "partition": "held_out" if family == "held_out_composition" else "diagnostic",
                "resolved_scenario": scenario,
                "cell_spec_digest": canonical_digest(scenario),
            }
        )
    return cells


def _candidate_spec(
    *, candidate: Mapping[str, Any], source_commit: str, scene_plan: Mapping[str, Any]
) -> dict[str, Any]:
    candidate_id = str(candidate["candidate_id"])
    policy, endpoint, policy_identity = _candidate_runtime_binding(candidate_id)
    policy_spec = asdict(policy)
    checkpoint = dict(candidate["checkpoint"])
    runtime_identity = {
        "source_commit": source_commit,
        "container_image": NATIVE_TASK_ARENA_IMAGE,
        "candidate_source": candidate["source"],
        "policy_identity": policy_identity,
        "checkpoint_inventory_digest": checkpoint["inventory_digest"],
        "observation_adapter": candidate["policy_input_schema"],
        "action_adapter": candidate["action_adapter"],
    }
    rights = {
        "scene_id": scene_plan["scene_id"],
        "task_id": scene_plan["task_id"],
        "scene_plan_digest": scene_plan["plan_digest"],
        "source_license": candidate["source"]["license"],
        "checkpoint_provider_use_status": checkpoint["provider_use_status"],
        "checkpoint_redistribution_status": checkpoint["redistribution_status"],
        "rights_ready": candidate["rights_ready"],
        "secret_material_recorded": False,
        "rights_receipt_digest": "",
    }
    rights["rights_receipt_digest"] = canonical_digest(rights, digest_field="rights_receipt_digest")
    horizon = int(policy.open_loop_horizon)
    max_queries = math.ceil(int(scene_plan["task_spec"]["maximum_action_steps"]) / horizon)
    spec: dict[str, Any] = {
        "schema_version": SPEC_SCHEMA_VERSION,
        "candidate_id": candidate_id,
        "execution_authority": "internal_policy_canary_unqualified",
        "claim_ceiling": CLAIM_CEILING,
        "scene_id": scene_plan["scene_id"],
        "task_id": scene_plan["task_id"],
        "scene_plan_digest": scene_plan["plan_digest"],
        "prompt": scene_plan["task_spec"]["prompt"],
        "policy_endpoint": endpoint,
        "policy_spec": policy_spec,
        "candidate_rights_binding": rights,
        "checkpoint_digest": checkpoint["inventory_digest"],
        "runtime_identity": runtime_identity,
        "runtime_identity_digest": canonical_digest(runtime_identity),
        "max_policy_queries": max_queries,
        "open_loop_horizon": horizon,
        "ranking_permitted": False,
        "qualification_permitted": False,
        "scene_promotion_permitted": False,
        "scoring_authority": "deterministic_simulator_state",
        "execution_spec_digest": "",
    }
    spec["execution_spec_digest"] = canonical_digest(spec, digest_field="execution_spec_digest")
    return spec


def materialize_scene839873_policy_canary_setup(
    *,
    source_commit: str,
    configured_source_launch_id: str,
    scene_revision_digest: str,
    activation_digest: str,
    capture_session_id: str,
    intake_id: str,
    request_digest: str,
    launch_request_path: str | Path,
    launch_profile_path: str | Path,
    configured_progression_path: str | Path,
    scene_plan_path: str | Path,
    packet_receipt_path: str | Path,
    runtime_source_receipt_path: str | Path,
    historical_policy_readiness_path: str | Path,
    pi05_checkpoint_inventory_path: str | Path,
    output_dir: str | Path,
    maximum_hourly_rate_usd: float = 0.8,
    hard_cap_usd: float = 4.0,
    hard_ttl_seconds: int = 14_400,
) -> dict[str, Any]:
    blockers: list[str] = []
    if not _SHA.fullmatch(source_commit):
        blockers.append("policy_canary_source_commit_invalid")
    for name, value in (
        ("activation", activation_digest),
        ("scene_revision", scene_revision_digest),
        ("request", request_digest),
    ):
        if not _DIGEST.fullmatch(str(value or "")):
            blockers.append(f"policy_canary_{name}_digest_invalid")
    for name, value in (
        ("configured_source_launch_id", configured_source_launch_id),
        ("capture_session_id", capture_session_id),
        ("intake_id", intake_id),
    ):
        if not str(value or "").strip():
            blockers.append(f"policy_canary_{name}_missing")
    launch_request = _read(launch_request_path, code="policy_canary_launch_request_invalid")
    profile = _read(launch_profile_path, code="policy_canary_launch_profile_invalid")
    progression = _read(configured_progression_path, code="policy_canary_progression_invalid")
    scene_plan = _read(scene_plan_path, code="policy_canary_scene_plan_invalid")
    packet = _read(packet_receipt_path, code="policy_canary_packet_receipt_invalid")
    runtime = _read(runtime_source_receipt_path, code="policy_canary_runtime_source_invalid")
    readiness = _read(
        historical_policy_readiness_path, code="policy_canary_historical_readiness_invalid"
    )
    if (
        launch_request.get("source_commit") != source_commit
        or profile.get("source_commit") != source_commit
    ):
        blockers.append("policy_canary_current_commit_binding_mismatch")
    if launch_request.get("request_digest") != request_digest:
        blockers.append("policy_canary_request_digest_mismatch")
    if profile.get("profile_digest") != canonical_digest(profile, digest_field="profile_digest"):
        blockers.append("policy_canary_profile_digest_invalid")
    if progression.get("configured_scene_revision_digest") != scene_revision_digest:
        blockers.append("policy_canary_scene_revision_mismatch")
    if scene_plan.get("schema_version") != "native_task_arena_scene_plan.v1" or scene_plan.get(
        "plan_digest"
    ) != canonical_digest(scene_plan, digest_field="plan_digest"):
        blockers.append("policy_canary_scene_plan_invalid")
    if (
        scene_plan.get("scene_id") != "interiorgs-839873"
        or scene_plan.get("task_kind") != "rigid_pick_place"
        or scene_plan.get("robot", {}).get("robot_id") != "franka_panda"
        or scene_plan.get("task_spec", {}).get("manipulation_strategy") != "planar_push"
    ):
        blockers.append("policy_canary_scene_task_embodiment_incompatible")
    if (
        packet.get("scene_id") != scene_plan.get("scene_id")
        or packet.get("task_id") != scene_plan.get("task_id")
        or packet.get("arena_scene_plan_digest") != scene_plan.get("plan_digest")
    ):
        blockers.append("policy_canary_packet_binding_invalid")
    if (
        runtime.get("schema_version") != "native_task_runtime_source_packet.v1"
        or not _DIGEST.fullmatch(str(runtime.get("packet_sha256") or ""))
        or int(runtime.get("packet_size_bytes") or 0) <= 0
        or runtime.get("redistribution_permitted") is not True
    ):
        blockers.append("policy_canary_runtime_source_invalid")
    if readiness.get("readiness_digest") != canonical_digest(
        readiness, digest_field="readiness_digest"
    ):
        blockers.append("policy_canary_historical_readiness_digest_invalid")
    candidates = {
        row.get("candidate_id"): row
        for row in readiness.get("candidates") or []
        if isinstance(row, Mapping)
    }
    if tuple(candidate for candidate in CANDIDATE_IDS if candidate in candidates) != CANDIDATE_IDS:
        blockers.append("policy_canary_candidate_pair_missing")
    for candidate_id in CANDIDATE_IDS:
        row = candidates.get(candidate_id) or {}
        expected = EXPECTED_CANDIDATES[candidate_id]
        if (
            row.get("source", {}).get("revision") != expected["source_revision"]
            or row.get("source", {}).get("tree") != expected["source_tree"]
            or row.get("checkpoint", {}).get("inventory_digest")
            != expected["checkpoint_inventory_digest"]
            or row.get("rights_ready") is not True
            or row.get("observation_adapter_ready") is not True
            or row.get("action_adapter_ready") is not True
            or row.get("checkpoint", {}).get("checkpoint_ready") is not True
            or row.get("checkpoint", {}).get("missing_secrets_or_gated_access") != []
        ):
            blockers.append(f"policy_canary_{candidate_id}_registry_or_rights_invalid")
        if tuple(CANDIDATE_REQUIRED_VIEWS[candidate_id]) != (
            DROID_EXTERIOR_VIEW_1,
            DROID_WRIST_VIEW,
        ):
            blockers.append(f"policy_canary_{candidate_id}_camera_schema_invalid")
        if row.get("policy_output_schema", {}).get("action_space") != "joint_position" or row.get(
            "policy_output_schema", {}
        ).get("joint_order") != [f"panda_joint{i}" for i in range(1, 8)]:
            blockers.append(f"policy_canary_{candidate_id}_action_schema_invalid")
    cells = _quick_cells(scene_revision_digest)
    if Counter(row["family"] for row in cells) != Counter(QUICK_FAMILY_COUNTS):
        blockers.append("policy_canary_quick10_coverage_invalid")
    if blockers:
        raise PolicyCanarySetupError(blockers)
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    spec_paths = {}
    for candidate_id in CANDIDATE_IDS:
        spec = _candidate_spec(
            candidate=candidates[candidate_id],
            source_commit=source_commit,
            scene_plan=scene_plan,
        )
        path = destination / f"{candidate_id}.policy_canary_execution_spec.v1.json"
        write_json(path, spec)
        spec_paths[candidate_id] = path
    setup: dict[str, Any] = {
        "schema_version": SETUP_SCHEMA_VERSION,
        "status": "verified_runnable",
        "run_kind": RUN_KIND,
        "claim_ceiling": CLAIM_CEILING,
        "scene_id": SCENE_ID,
        "configured_source_launch_id": configured_source_launch_id,
        "scene_revision_digest": scene_revision_digest,
        "activation_digest": activation_digest,
        "source_commit": source_commit,
        "provider": "vast",
        "embodiment_id": EMBODIMENT_ID,
        "candidate_ids": list(CANDIDATE_IDS),
        "records": {
            "pi05_execution_spec": _record(spec_paths["pi05_droid"]),
            "groot_execution_spec": _record(spec_paths["groot_n17_droid"]),
            "pi05_checkpoint_inventory": _record(pi05_checkpoint_inventory_path),
        },
        "capture_session_id": capture_session_id,
        "intake_id": intake_id,
        "request_digest": request_digest,
        "runtime_inputs": {
            "native_packet": _record(packet_receipt_path),
            "scene_plan": _record(scene_plan_path),
            "runtime_source": _record(runtime_source_receipt_path),
        },
        "quick_10": {
            "policy_count": 2,
            "episodes_per_policy": 10,
            "learned_policy_rollout_count": 20,
            "cells": cells,
            "matrix_digest": canonical_digest({"cells": cells}),
        },
        "estimate": {
            "basis": "one_warm_vast_session_two_revision_pinned_checkpoints_twenty_rollouts",
            "runtime_seconds_upper_bound": hard_ttl_seconds,
            "maximum_hourly_rate_usd": maximum_hourly_rate_usd,
            "hard_cap_usd": hard_cap_usd,
            "hard_ttl_seconds": hard_ttl_seconds,
            "retry_cap": 0,
            "maximum_provider_allocations": 1,
        },
        "historical_runtime_smoke": {
            "input_evidence_only": True,
            "current_runtime_proof": False,
            "readiness_digest": readiness["readiness_digest"],
            "source_scene_id": readiness.get("scene_id"),
        },
        "scene_promotion_authorized": False,
        "official_ranking_authorized": False,
        "setup_digest": "",
    }
    setup["setup_digest"] = canonical_digest(setup, digest_field="setup_digest")
    write_json(destination / "task_evaluation_policy_canary_execution_setup.v1.json", setup)
    return setup


def materialize_setup_preflight_decision(
    *, output_path: str | Path, **kwargs: Any
) -> dict[str, Any]:
    try:
        setup = materialize_scene839873_policy_canary_setup(**kwargs)
        decision = {
            "schema_version": DECISION_SCHEMA_VERSION,
            "status": "verified_runnable",
            "setup_digest": setup["setup_digest"],
            "blockers": [],
            "decision_digest": "",
        }
    except PolicyCanarySetupError as exc:
        decision = {
            "schema_version": DECISION_SCHEMA_VERSION,
            "status": "blocked",
            "setup_digest": None,
            "blockers": list(exc.blockers),
            "decision_digest": "",
        }
    decision["decision_digest"] = canonical_digest(decision, digest_field="decision_digest")
    write_json(Path(output_path), decision)
    return decision


__all__ = [
    "CANDIDATE_IDS",
    "PolicyCanarySetupError",
    "QUICK_FAMILY_COUNTS",
    "materialize_scene839873_policy_canary_setup",
    "materialize_setup_preflight_decision",
]
