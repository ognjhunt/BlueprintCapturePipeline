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
import tempfile
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
PRESUBMISSION_SETUP_SCHEMA_VERSION = "task_evaluation_policy_canary_setup.v1"
PROFILE_INPUT_SCHEMA_VERSION = "task_evaluation_policy_canary_profile_materialization_input.v1"
EXECUTION_TEMPLATE_SCHEMA_VERSION = (
    "task_evaluation_policy_canary_execution_setup_template.v1"
)
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
    configured_request_digest: str | None = None,
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
    source_request_digest = configured_request_digest or request_digest
    if launch_request.get("request_digest") != source_request_digest:
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
        "configured_request_digest": source_request_digest,
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


def materialize_policy_canary_presubmission_setup(
    *,
    profile_id: str,
    source_commit: str,
    configured_source_launch_id: str,
    scene_revision_digest: str,
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
    """Emit the Website descriptor before a user-created activation exists."""

    if not str(profile_id or "").strip():
        raise PolicyCanarySetupError(["policy_canary_profile_id_missing"])
    # Reuse the complete byte/static preflight without publishing its
    # activation-bound output. The placeholder lineage lives only in a
    # temporary directory and is never returned or persisted as evidence.
    with tempfile.TemporaryDirectory(prefix="policy-canary-presubmission-") as raw:
        verified = materialize_scene839873_policy_canary_setup(
            source_commit=source_commit,
            configured_source_launch_id=configured_source_launch_id,
            scene_revision_digest=scene_revision_digest,
            activation_digest=canonical_digest(
                {
                    "kind": "presubmission_static_preflight_only",
                    "configured_source_launch_id": configured_source_launch_id,
                }
            ),
            capture_session_id="presubmission_not_assigned",
            intake_id="presubmission_not_assigned",
            request_digest=request_digest,
            launch_request_path=launch_request_path,
            launch_profile_path=launch_profile_path,
            configured_progression_path=configured_progression_path,
            scene_plan_path=scene_plan_path,
            packet_receipt_path=packet_receipt_path,
            runtime_source_receipt_path=runtime_source_receipt_path,
            historical_policy_readiness_path=historical_policy_readiness_path,
            pi05_checkpoint_inventory_path=pi05_checkpoint_inventory_path,
            output_dir=Path(raw) / "verification",
            maximum_hourly_rate_usd=maximum_hourly_rate_usd,
            hard_cap_usd=hard_cap_usd,
            hard_ttl_seconds=hard_ttl_seconds,
        )
    launch_profile = _read(launch_profile_path, code="policy_canary_launch_profile_invalid")
    readiness = _read(
        historical_policy_readiness_path,
        code="policy_canary_historical_readiness_invalid",
    )
    policies = []
    readiness_by_id = {
        row["candidate_id"]: row
        for row in readiness["candidates"]
        if isinstance(row, Mapping) and row.get("candidate_id") in CANDIDATE_IDS
    }
    for candidate_id in CANDIDATE_IDS:
        row = readiness_by_id[candidate_id]
        policies.append(
            {
                "candidate_id": candidate_id,
                "display_name": row["model_name"],
                "readiness_status": "verified_runnable",
                "source": row["source"],
                "checkpoint": {
                    key: row["checkpoint"].get(key)
                    for key in (
                        "repository",
                        "revision",
                        "inventory_digest",
                        "total_bytes",
                        "provider_use_status",
                        "redistribution_status",
                    )
                },
                "runtime_dependencies": row["runtime_dependencies"],
                "observation_schema": row["policy_input_schema"],
                "action_schema": row["policy_output_schema"],
                "action_adapter": row["action_adapter"],
                "task_compatibility": ["rigid_relocation", "planar_push"],
                "unavailable_reason": None,
            }
        )
    quick = verified["quick_10"]
    setup: dict[str, Any] = {
        "schema_version": PRESUBMISSION_SETUP_SCHEMA_VERSION,
        "status": "selectable",
        "run_kind": RUN_KIND,
        "claim_ceiling": CLAIM_CEILING,
        "unqualified_warning": "Controls pending — results are unqualified.",
        "configured_source_launch_id": configured_source_launch_id,
        "configured_profile_lineage": {
            "profile_id": launch_profile["profile_id"],
            "profile_digest": launch_profile["profile_digest"],
            "request_digest": request_digest,
        },
        "scene": {
            "scene_id": SCENE_ID,
            "scene_revision_digest": scene_revision_digest,
            "controls_status": "configured_controls_pending",
            "task_id": "scene-839873-mug-planar-push",
            "task_kind": "rigid_relocation",
        },
        "source_commit": source_commit,
        "registry": {
            "candidate_ids": list(CANDIDATE_IDS),
            "registry_digest": readiness["readiness_digest"],
            "historical_runtime_smoke_is_input_only": True,
            "current_runtime_execution_proven": False,
        },
        "robot_presets": [
            {
                "robot_preset_id": EMBODIMENT_ID,
                "display_name": "Franka Panda + Robotiq 2F-85",
                "runtime_robot_id": "franka_panda",
                "runtime_image": NATIVE_TASK_ARENA_IMAGE,
                "observation_cameras": ["external", "wrist"],
                "action_schema": "absolute_7_joint_positions_plus_gripper",
                "compatible_candidate_ids": list(CANDIDATE_IDS),
                "readiness_status": "verified_runnable",
            }
        ],
        "policies": policies,
        "presets": [
            {
                "preset_id": "quick_10",
                "label": "Quick — 10 episodes per policy",
                "availability": "available",
                "recommended": True,
                **{
                    key: quick[key]
                    for key in (
                        "policy_count",
                        "episodes_per_policy",
                        "learned_policy_rollout_count",
                        "cells",
                        "matrix_digest",
                    )
                },
            },
            {
                "preset_id": "standard_100",
                "label": "Standard — 100 episodes per policy",
                "availability": "disabled",
                "disabled_reason": "standard_runtime_contract_not_qualified",
            },
            {
                "preset_id": "deep_500",
                "label": "Deep — 500 episodes per policy",
                "availability": "disabled",
                "disabled_reason": "deep_runtime_contract_not_qualified",
            },
        ],
        "estimate": verified["estimate"],
        "diagnostics": {
            "controls_mode": "nonblocking_diagnostic_pending",
            "diagnostic_control_rollouts_listed_separately": True,
            "failed_controls_preserved": True,
            "uninterpretable_outcomes_not_ranked": True,
            "scene_promotion_forbidden": True,
            "official_ranking_forbidden": True,
        },
        "setup_digest": "",
    }
    setup["setup_digest"] = canonical_digest(setup, digest_field="setup_digest")
    wrapper: dict[str, Any] = {
        "schema_version": PROFILE_INPUT_SCHEMA_VERSION,
        "profile_id": profile_id,
        "configured_source_launch_id": configured_source_launch_id,
        "source_commit": source_commit,
        "internal_policy_canary_setup": setup,
        "materialization_digest": "",
    }
    wrapper["materialization_digest"] = canonical_digest(
        wrapper, digest_field="materialization_digest"
    )
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    setup_path = destination / "task_evaluation_policy_canary_setup.v1.json"
    wrapper_path = destination / (
        "task_evaluation_policy_canary_profile_materialization_input.v1.json"
    )
    write_json(setup_path, setup)
    write_json(wrapper_path, wrapper)
    execution_template: dict[str, Any] = {
        "schema_version": EXECUTION_TEMPLATE_SCHEMA_VERSION,
        "source_commit": source_commit,
        "configured_source_launch_id": configured_source_launch_id,
        "scene_revision_digest": scene_revision_digest,
        "configured_request_digest": request_digest,
        "launch_request_path": str(Path(launch_request_path).expanduser().resolve()),
        "launch_profile_path": str(Path(launch_profile_path).expanduser().resolve()),
        "configured_progression_path": str(
            Path(configured_progression_path).expanduser().resolve()
        ),
        "scene_plan_path": str(Path(scene_plan_path).expanduser().resolve()),
        "packet_receipt_path": str(Path(packet_receipt_path).expanduser().resolve()),
        "runtime_source_receipt_path": str(
            Path(runtime_source_receipt_path).expanduser().resolve()
        ),
        "historical_policy_readiness_path": str(
            Path(historical_policy_readiness_path).expanduser().resolve()
        ),
        "pi05_checkpoint_inventory_path": str(
            Path(pi05_checkpoint_inventory_path).expanduser().resolve()
        ),
        "maximum_hourly_rate_usd": maximum_hourly_rate_usd,
        "hard_cap_usd": hard_cap_usd,
        "hard_ttl_seconds": hard_ttl_seconds,
        "profile_materialization_input": _record(wrapper_path),
        "provider_mutation_performed": False,
        "paid_execution_requested": False,
        "template_digest": "",
    }
    execution_template["template_digest"] = canonical_digest(
        execution_template, digest_field="template_digest"
    )
    execution_template_path = destination / (
        "task_evaluation_policy_canary_execution_setup_template.v1.json"
    )
    write_json(execution_template_path, execution_template)
    return {
        "setup": setup,
        "setup_path": str(setup_path),
        "profile_materialization_input": wrapper,
        "profile_materialization_input_path": str(wrapper_path),
        "execution_setup_template": execution_template,
        "execution_setup_template_path": str(execution_template_path),
    }


def materialize_scene839873_policy_canary_setup_from_template(
    *,
    template_path: str | Path,
    activation_envelope: Mapping[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Fill only Website-owned activation lineage into a staged static template."""

    template = _read(template_path, code="policy_canary_execution_template_invalid")
    if (
        template.get("schema_version") != EXECUTION_TEMPLATE_SCHEMA_VERSION
        or template.get("provider_mutation_performed") is not False
        or template.get("paid_execution_requested") is not False
        or template.get("template_digest")
        != canonical_digest(template, digest_field="template_digest")
    ):
        raise PolicyCanarySetupError(["policy_canary_execution_template_invalid"])
    wrapper_record = template.get("profile_materialization_input")
    if not isinstance(wrapper_record, Mapping):
        raise PolicyCanarySetupError(["policy_canary_profile_materialization_input_invalid"])
    wrapper_path = Path(str(wrapper_record.get("path") or "")).expanduser().resolve()
    if (
        wrapper_path.is_symlink()
        or not wrapper_path.is_file()
        or wrapper_path.stat().st_size != wrapper_record.get("size_bytes")
        or _sha256(wrapper_path) != wrapper_record.get("sha256")
    ):
        raise PolicyCanarySetupError(["policy_canary_profile_materialization_input_invalid"])
    wrapper = _read(
        wrapper_path, code="policy_canary_profile_materialization_input_invalid"
    )
    if (
        wrapper.get("schema_version") != PROFILE_INPUT_SCHEMA_VERSION
        or wrapper.get("configured_source_launch_id")
        != template.get("configured_source_launch_id")
        or wrapper.get("source_commit") != template.get("source_commit")
        or wrapper.get("materialization_digest")
        != canonical_digest(wrapper, digest_field="materialization_digest")
    ):
        raise PolicyCanarySetupError(["policy_canary_profile_materialization_input_invalid"])
    if (
        activation_envelope.get("schema_version")
        != "task_evaluation_policy_canary_dispatch_envelope.v1"
        or activation_envelope.get("run_kind") != RUN_KIND
        or activation_envelope.get("claim_ceiling") != CLAIM_CEILING
        or activation_envelope.get("source_commit") != template.get("source_commit")
        or activation_envelope.get("envelope_digest")
        != canonical_digest(activation_envelope, digest_field="envelope_digest")
    ):
        raise PolicyCanarySetupError(["policy_canary_activation_envelope_invalid"])
    activation_record = activation_envelope.get("activation_result")
    if not isinstance(activation_record, Mapping):
        raise PolicyCanarySetupError(["policy_canary_activation_result_invalid"])
    activation_path = Path(str(activation_record.get("path") or "")).expanduser().resolve()
    if (
        activation_path.is_symlink()
        or not activation_path.is_file()
        or activation_path.stat().st_size != activation_record.get("size_bytes")
        or _sha256(activation_path) != activation_record.get("sha256")
    ):
        raise PolicyCanarySetupError(["policy_canary_activation_result_invalid"])
    activation = _read(
        activation_path, code="policy_canary_activation_result_invalid"
    )
    return materialize_scene839873_policy_canary_setup(
        source_commit=str(template["source_commit"]),
        configured_source_launch_id=str(template["configured_source_launch_id"]),
        scene_revision_digest=str(template["scene_revision_digest"]),
        activation_digest=str(activation["policy_campaign_activation_digest"]),
        capture_session_id=str(activation_envelope["capture_session_id"]),
        intake_id=str(activation_envelope["intake_id"]),
        request_digest=str(activation_envelope["request_digest"]),
        configured_request_digest=str(template["configured_request_digest"]),
        launch_request_path=template["launch_request_path"],
        launch_profile_path=template["launch_profile_path"],
        configured_progression_path=template["configured_progression_path"],
        scene_plan_path=template["scene_plan_path"],
        packet_receipt_path=template["packet_receipt_path"],
        runtime_source_receipt_path=template["runtime_source_receipt_path"],
        historical_policy_readiness_path=template[
            "historical_policy_readiness_path"
        ],
        pi05_checkpoint_inventory_path=template[
            "pi05_checkpoint_inventory_path"
        ],
        output_dir=output_dir,
        maximum_hourly_rate_usd=float(template["maximum_hourly_rate_usd"]),
        hard_cap_usd=float(template["hard_cap_usd"]),
        hard_ttl_seconds=int(template["hard_ttl_seconds"]),
    )


__all__ = [
    "CANDIDATE_IDS",
    "PolicyCanarySetupError",
    "QUICK_FAMILY_COUNTS",
    "materialize_scene839873_policy_canary_setup",
    "materialize_policy_canary_presubmission_setup",
    "materialize_scene839873_policy_canary_setup_from_template",
    "materialize_setup_preflight_decision",
]
