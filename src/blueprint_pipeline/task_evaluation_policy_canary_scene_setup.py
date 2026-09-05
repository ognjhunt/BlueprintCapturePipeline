"""No-spend Scene 839873 readiness and execution-setup materializer."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess  # nosec B404 - fixed runuser/sha256sum argv for service readback
import tempfile
from typing import Any, Callable

from .adp_task_scoring import (
    TaskNeutralScoringError,
    seal_rigid_task_success_contract,
    validate_rigid_task_success_contract,
)
from .adp009d_droid_observation import (
    CANDIDATE_REQUIRED_VIEWS,
    DROID_EXTERIOR_VIEW_1,
    DROID_WRIST_VIEW,
)
from .adp009d_policy_candidate_admission import EXPECTED_CANDIDATES
from .common import utc_now_iso, write_json
from .decision_evidence_contracts import (
    canonical_digest,
    cross_runtime_canonical_digest,
)
from .droid_policy_canary_embodiment import (
    DROID_EMBODIMENT_ID,
    DROID_POLICY_CANARY_PRESET_ID,
    concrete_droid_task_instruction,
)
from .native_task_arena_policy_bundle import _candidate_runtime_binding
from .native_task_isaaclab_launch import NATIVE_TASK_ARENA_IMAGE
from .host_resident_launch_inputs import PRODUCTION_LAUNCH_INPUT_ROOTS
from .task_evaluation_canary_hotfix_overlay import (
    CanaryHotfixOverlayError,
    resolve_service_account,
    service_account_access_blockers,
)
from .task_evaluation_policy_canary_setup import (
    policy_canary_setup_digest,
    validate_policy_canary_setup,
)
from .task_evaluation_policy_run_contract import (
    policy_run_setup_digest,
    validate_policy_run_setup,
)


SETUP_SCHEMA_VERSION = "task_evaluation_policy_canary_execution_setup.v1"
SPEC_SCHEMA_VERSION = "native_task_arena_policy_canary_execution_spec.v1"
DECISION_SCHEMA_VERSION = "task_evaluation_policy_canary_setup_preflight.v1"
PRESUBMISSION_SETUP_SCHEMA_VERSION = "task_evaluation_policy_canary_setup.v1"
PROFILE_INPUT_SCHEMA_VERSION = "task_evaluation_policy_canary_profile_materialization_input.v1"
EXECUTION_TEMPLATE_SCHEMA_VERSION = "task_evaluation_policy_canary_execution_setup_template.v1"
RUN_KIND = "internal_policy_canary"
CLAIM_CEILING = "diagnostic_policy_execution"
SCENE_ID = "839873"
CANDIDATE_IDS = ("pi05_droid", "groot_n17_droid")
EMBODIMENT_ID = DROID_POLICY_CANARY_PRESET_ID
FORBIDDEN_SCENE_DIGEST_PREFIXES = ("d6c3cd3e",)
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
_MATERIALIZATION_DIRECTORY_MODE = 0o750
_MATERIALIZATION_FILE_MODE = 0o440
_RUNUSER_PATH = "/usr/sbin/runuser"
_SHA256SUM_PATH = "/usr/bin/sha256sum"


class PolicyCanarySetupError(ValueError):
    def __init__(self, blockers: list[str] | tuple[str, ...]):
        self.blockers = tuple(sorted(set(str(item) for item in blockers if str(item))))
        super().__init__(";".join(self.blockers))


def _mapping_or_empty(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


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


def _under_production_root(path: Path) -> bool:
    resolved = path.resolve()
    return any(
        resolved == Path(value).resolve() or Path(value).resolve() in resolved.parents
        for value in PRODUCTION_LAUNCH_INPUT_ROOTS
    )


def _digest_as_service_account(path: Path, *, account: Mapping[str, Any]) -> str:
    """Reopen exact bytes as the consumer rather than trusting chmod/chown."""

    if os.geteuid() == int(account["uid"]):
        return _sha256(path)
    if os.geteuid() != 0:
        return ""
    try:
        completed = subprocess.run(  # nosec B603 - fixed executable and argv
            [
                _RUNUSER_PATH,
                "-u",
                str(account["user"]),
                "--",
                _SHA256SUM_PATH,
                str(path),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if completed.returncode != 0 or not completed.stdout.strip():
        return ""
    return "sha256:" + completed.stdout.split()[0]


def _seal_presubmission_service_access(
    *,
    destination: Path,
    artifacts: Mapping[str, Path],
    account_resolver: Callable[[], Mapping[str, Any]] = resolve_service_account,
    chown: Callable[[Path, int, int], None] = os.chown,
    stat_reader: Callable[[Path], Any] = os.stat,
    digest_reader: Callable[..., str] = _digest_as_service_account,
    access_checker: Callable[..., list[str]] = service_account_access_blockers,
) -> dict[str, Any]:
    """Seal root-authored launch dependencies for the ``blueprint`` reader.

    Only the newly materialized directory and its three named JSON children are
    changed. Existing ancestors are never rechmodded recursively; if one hides
    the bytes from the service account, materialization fails before profile
    publication.
    """

    try:
        account = dict(account_resolver())
    except CanaryHotfixOverlayError as exc:
        if _under_production_root(destination):
            raise PolicyCanarySetupError(
                ["policy_canary_materialization_service_account_unknown"]
            ) from exc
        return {
            "status": "not_applicable_no_service_account",
            "service_user": "blueprint",
            "verified_roles": [],
        }

    expected = {
        "setup",
        "profile_materialization_input",
        "execution_setup_template",
    }
    if set(artifacts) != expected:
        raise PolicyCanarySetupError(
            ["policy_canary_materialization_dependency_inventory_invalid"]
        )
    targets = {role: Path(path).resolve() for role, path in artifacts.items()}
    if destination.is_symlink() or not destination.is_dir():
        raise PolicyCanarySetupError(
            ["policy_canary_materialization_destination_invalid"]
        )
    for path in targets.values():
        if path.parent != destination or path.is_symlink() or not path.is_file():
            raise PolicyCanarySetupError(
                ["policy_canary_materialization_dependency_invalid"]
            )

    try:
        chown(destination, -1, int(account["gid"]))
        destination.chmod(_MATERIALIZATION_DIRECTORY_MODE)
        for path in targets.values():
            chown(path, -1, int(account["gid"]))
            path.chmod(_MATERIALIZATION_FILE_MODE)
    except OSError as exc:
        raise PolicyCanarySetupError(
            ["policy_canary_materialization_service_access_install_failed"]
        ) from exc

    destination_metadata = stat_reader(destination)
    if (
        destination_metadata.st_gid != int(account["gid"])
        or stat.S_IMODE(destination_metadata.st_mode) != _MATERIALIZATION_DIRECTORY_MODE
        or access_checker(destination, account=account, stat_reader=stat_reader)
    ):
        raise PolicyCanarySetupError(
            ["policy_canary_materialization_service_access_denied:destination"]
        )

    verified: list[dict[str, Any]] = []
    for role, path in targets.items():
        metadata = stat_reader(path)
        expected_digest = _sha256(path)
        if (
            metadata.st_gid != int(account["gid"])
            or stat.S_IMODE(metadata.st_mode) != _MATERIALIZATION_FILE_MODE
            or access_checker(path, account=account, stat_reader=stat_reader)
            or digest_reader(path, account=account) != expected_digest
        ):
            raise PolicyCanarySetupError(
                [f"policy_canary_materialization_service_access_denied:{role}"]
            )
        verified.append(
            {
                "role": role,
                "name": path.name,
                "mode": f"{_MATERIALIZATION_FILE_MODE:04o}",
                "sha256": expected_digest,
            }
        )
    return {
        "status": "readable_by_service_account",
        "service_user": account["user"],
        "service_group": account["group"],
        "directory_mode": f"{_MATERIALIZATION_DIRECTORY_MODE:04o}",
        "verified_roles": verified,
    }


def _forbidden_scene_digest_blockers(
    records: Mapping[str, str | Path],
) -> list[str]:
    """Refuse known-corrupt scene lineage before emitting launch material."""

    blockers: list[str] = []
    for role, raw_path in records.items():
        source = Path(raw_path).expanduser().resolve()
        try:
            payload = source.read_bytes().lower()
        except OSError:
            # The typed record reader owns missing/unreadable-file reporting.
            continue
        for prefix in FORBIDDEN_SCENE_DIGEST_PREFIXES:
            if prefix.encode("ascii") in payload:
                blockers.append(
                    f"policy_canary_forbidden_scene_digest_present:{role}:{prefix}"
                )
    return blockers


def _immutable_ref(value: Mapping[str, Any], *, code: str) -> dict[str, Any]:
    ref = dict(value)
    if (
        set(ref) != {"uri", "digest", "size_bytes"}
        or not str(ref.get("uri") or "").startswith(("gs://", "s3://", "https://"))
        or not _DIGEST.fullmatch(str(ref.get("digest") or ""))
        or isinstance(ref.get("size_bytes"), bool)
        or not isinstance(ref.get("size_bytes"), int)
        or ref["size_bytes"] <= 0
    ):
        raise PolicyCanarySetupError([code])
    return ref


def _quick_cells(
    scene_revision_digest: str, *, scene_id: str = SCENE_ID
) -> list[dict[str, Any]]:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", scene_id):
        raise PolicyCanarySetupError(["policy_canary_scene_id_invalid"])
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
        seed = int(
            hashlib.sha256(f"{scene_revision_digest}:{index}".encode()).hexdigest()[:8],
            16,
        ) % (2**31)
        scenario = {"family": family, "ordinal": index, "parameters": resolved}
        cells.append(
            {
                "cell_id": f"scene{scene_id}.quick10.{index:02d}.{family}",
                "seed": seed,
                "family": family,
                "partition": "held_out" if family == "held_out_composition" else "diagnostic",
                "resolved_scenario": scenario,
                "cell_spec_digest": cross_runtime_canonical_digest(scenario),
            }
        )
    return cells


def _require_strict_owner_success_contract(
    *, task_spec: Mapping[str, Any], contract: Mapping[str, Any]
) -> None:
    """Reject compatibility or weaker scoring for an explicitly strict task.

    This joins an already confirmed contract; it never confirms an agent's
    proposed thresholds and never changes deterministic scoring.
    """
    configured = _mapping_or_empty(task_spec.get("configured_success_criteria"))
    if configured.get("owner_success_contract_required") is not True:
        return
    provenance = _mapping_or_empty(contract.get("provenance"))
    if (provenance.get("author_source") == "compatibility_default"
            or provenance.get("confirmation_status") != "confirmed"):
        raise TaskNeutralScoringError(["policy_canary_owner_success_contract_unconfirmed"])
    criteria = _mapping_or_empty(contract.get("criteria"))
    temporal = _mapping_or_empty(criteria.get("temporal_invariants"))
    motion = _mapping_or_empty(criteria.get("motion"))
    destination = _mapping_or_empty(criteria.get("destination_containment"))
    support = _mapping_or_empty(criteria.get("support"))
    orientation = _mapping_or_empty(criteria.get("orientation"))
    settling = _mapping_or_empty(criteria.get("settling"))
    gripper = _mapping_or_empty(criteria.get("gripper_state"))
    minimum_lift = configured.get("minimum_lift_m")
    force_limit = temporal.get("maximum_task_contact_force_n")
    if (
        _mapping_or_empty(temporal.get("no_drop")).get("mode") != "required"
        or temporal.get("maximum_retries") != 0
        or temporal.get("maximum_regrasps") != 0
        or temporal.get("workspace_excursions") != "forbidden"
        or temporal.get("containment_excursions") != "forbidden"
        or not isinstance(force_limit, (int, float))
        or isinstance(force_limit, bool) or not math.isfinite(force_limit) or force_limit <= 0
        or not temporal.get("forbidden_contact_classes")
        or destination.get("mode") != "required"
        or support.get("height_mode") != "required"
        or support.get("contact_mode") != "required"
        or gripper.get("mode") != "released"
        or _mapping_or_empty(criteria.get("terminal_task_contact")).get("mode") != "cleared"
        or orientation.get("mode") != "required"
        or settling.get("mode") != "required"
        or not isinstance(minimum_lift, (int, float)) or isinstance(minimum_lift, bool)
        or not math.isfinite(minimum_lift) or minimum_lift <= 0
        or motion.get("minimum_lift_m") != minimum_lift
        or task_spec.get("minimum_lift_m") != minimum_lift
    ):
        raise TaskNeutralScoringError(["policy_canary_owner_success_contract_criteria_mismatch"])
    # Duplicated limits must describe the same native task, not a more permissive
    # contract selected after observing the episode.
    joins = (
        (destination.get("position_bounds_world_m"), task_spec.get("destination_position_bounds_world_m")),
        (support.get("height_interval_m"), task_spec.get("support_height_interval_m")),
        (orientation.get("reference_xyzw"), task_spec.get("destination_orientation_xyzw")),
        (orientation.get("tolerance_rad"), task_spec.get("destination_orientation_tolerance_rad")),
        (settling.get("window_samples"), task_spec.get("settle_window_samples")),
        (settling.get("position_tolerance_m"), task_spec.get("settle_position_tolerance_m")),
        (settling.get("orientation_tolerance_rad"), task_spec.get("settle_orientation_tolerance_rad")),
        (gripper.get("threshold_m"), task_spec.get("release_gripper_width_min_m")),
    )
    if any(left != right or right is None for left, right in joins):
        raise TaskNeutralScoringError(["policy_canary_owner_success_contract_native_limits_mismatch"])
    if configured.get("whole_subject_containment_required") is True:
        # The production scorer uses all eight corners only on this destination-
        # relative route. Requiring a center-only world box is not equivalent.
        if task_spec.get("destination_relation") != "inside":
            raise TaskNeutralScoringError(["policy_canary_owner_success_contract_full_containment_missing"])
        for key in ("subject_collision_bounds_scoring_frame_m",
                    "destination_interior_bounds_body_frame_m",
                    "destination_position_bounds_destination_frame_m"):
            bounds = _mapping_or_empty(task_spec.get(key))
            lower, upper = bounds.get("minimum"), bounds.get("maximum")
            if (not isinstance(lower, list) or not isinstance(upper, list)
                    or len(lower) != 3 or len(upper) != 3
                    or not all(isinstance(value, (int, float)) and not isinstance(value, bool)
                               and math.isfinite(value) for value in lower + upper)
                    or not all(lo < hi for lo, hi in zip(lower, upper, strict=True))):
                raise TaskNeutralScoringError(
                    ["policy_canary_owner_success_contract_full_containment_missing"]
                )
    if configured.get("object_must_rest_on_destination_support") is True:
        affordance = _mapping_or_empty(task_spec.get("interaction_affordance"))
        paths = affordance.get("intended_support_prim_paths")
        if (not task_spec.get("destination_support_asset_id")
                or not isinstance(paths, list) or len(paths) != 1
                or not isinstance(paths[0], str) or not paths[0].startswith("/")):
            raise TaskNeutralScoringError(["policy_canary_owner_success_contract_exact_support_missing"])
    if configured.get("retreat_clearance_required") is True:
        # The current rigid contract/scorer has no distance-qualified retreat
        # criterion. An open gripper and cleared contact cannot prove clearance.
        raise TaskNeutralScoringError(["policy_canary_owner_success_contract_retreat_scoring_unsupported"])


def _candidate_spec(
    *,
    candidate: Mapping[str, Any],
    source_commit: str,
    scene_plan: Mapping[str, Any],
    task_success_contract: Mapping[str, Any],
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
        "prompt": concrete_droid_task_instruction(
            {**scene_plan["task_spec"], "task_id": scene_plan["task_id"]}
        ),
        "policy_endpoint": endpoint,
        "policy_spec": policy_spec,
        "candidate_rights_binding": rights,
        "checkpoint_digest": checkpoint["inventory_digest"],
        "runtime_identity": runtime_identity,
        "runtime_identity_digest": canonical_digest(runtime_identity),
        "worker_identity_requirement": (
            "groot_droid_runtime_measurement"
            if policy_identity.get("status") == "runtime_measurement_required"
            else "none"
        ),
        "require_observed_eef_support": candidate_id == "groot_n17_droid",
        "max_policy_queries": max_queries,
        "open_loop_horizon": horizon,
        "ranking_permitted": False,
        "qualification_permitted": False,
        "scene_promotion_permitted": False,
        "scoring_authority": "deterministic_simulator_state",
        "task_success_contract": deepcopy(dict(task_success_contract)),
        "task_success_contract_digest": task_success_contract["contract_digest"],
        "execution_spec_digest": "",
    }
    spec["execution_spec_digest"] = canonical_digest(spec, digest_field="execution_spec_digest")
    return spec


def materialize_scene839873_policy_canary_setup(
    *,
    source_commit: str,
    configured_source_commit: str | None = None,
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
    task_success_contract: Mapping[str, Any] | None = None,
    require_confirmed_task_success_contract: bool = True,
    activation_release_window_template: Mapping[str, Any] | None = None,
    activation_lineage: Mapping[str, Any] | None = None,
    activation_authorization: Mapping[str, Any] | None = None,
    scene_id: str = SCENE_ID,
) -> dict[str, Any]:
    del (
        activation_release_window_template,
        activation_lineage,
        activation_authorization,
    )
    blockers: list[str] = []
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", scene_id):
        blockers.append("policy_canary_scene_id_invalid")
    if not _SHA.fullmatch(source_commit):
        blockers.append("policy_canary_source_commit_invalid")
    configured_commit = configured_source_commit or source_commit
    if not _SHA.fullmatch(configured_commit):
        blockers.append("policy_canary_configured_source_commit_invalid")
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
    blockers.extend(
        _forbidden_scene_digest_blockers(
            {
                "launch_request": launch_request_path,
                "launch_profile": launch_profile_path,
                "configured_progression": configured_progression_path,
                "scene_plan": scene_plan_path,
                "packet_receipt": packet_receipt_path,
            }
        )
    )
    if (
        launch_request.get("source_commit") != configured_commit
        or profile.get("source_commit") != configured_commit
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
        scene_plan.get("scene_id") != f"interiorgs-{scene_id}"
        or scene_plan.get("task_kind") != "rigid_pick_place"
        or scene_plan.get("robot", {}).get("robot_id") != "franka_panda"
        or scene_plan.get("task_spec", {}).get("manipulation_strategy")
        not in {"planar_push", "pick_and_place"}
    ):
        blockers.append("policy_canary_scene_task_embodiment_incompatible")
    task_spec = _mapping_or_empty(scene_plan.get("task_spec"))
    if scene_id != SCENE_ID and (
        not str(task_spec.get("instruction_subject_label") or "").strip()
        or not str(task_spec.get("visible_target_label") or "").strip()
    ):
        blockers.append("policy_canary_scene_instruction_grounding_missing")
    explicit_execution_contract = (
        task_success_contract is not None
        and require_confirmed_task_success_contract
    )
    raw_success_contract = task_success_contract
    if raw_success_contract is None:
        raw_success_contract = launch_request.get("task_success_contract")
        explicit_execution_contract = raw_success_contract is not None
    if raw_success_contract is None:
        raw_success_contract = scene_plan.get("task_spec", {}).get(
            "task_success_contract"
        )
    owner_contract_required = _mapping_or_empty(
        task_spec.get("configured_success_criteria")
    ).get("owner_success_contract_required") is True
    try:
        if raw_success_contract is None and owner_contract_required:
            raise TaskNeutralScoringError(["policy_canary_owner_success_contract_required"])
        if raw_success_contract is None:
            raw_success_contract = seal_rigid_task_success_contract(
                task_spec=scene_plan.get("task_spec", {}),
                site_id=str(scene_plan.get("scene_id") or ""),
                task_id=str(scene_plan.get("task_id") or ""),
                author_source="compatibility_default",
                author_id="blueprint:manipulation_strategy_defaults.v1",
                confirmation_status="confirmed",
            )
        task_success_contract = validate_rigid_task_success_contract(
            raw_success_contract,
            require_confirmed=explicit_execution_contract or owner_contract_required,
            expected_site_id=str(scene_plan.get("scene_id") or ""),
            expected_task_id=str(scene_plan.get("task_id") or ""),
        )
        _require_strict_owner_success_contract(
            task_spec=task_spec, contract=task_success_contract
        )
        if launch_request.get("task_success_contract") is not None and (
            launch_request.get("task_success_contract_digest")
            != task_success_contract["contract_digest"]
        ):
            blockers.append("policy_canary_task_success_contract_digest_mismatch")
    except TaskNeutralScoringError as exc:
        task_success_contract = {}
        blockers.extend(exc.errors)
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
    cells = _quick_cells(scene_revision_digest, scene_id=scene_id)
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
            task_success_contract=task_success_contract,
        )
        path = destination / f"{candidate_id}.policy_canary_execution_spec.v1.json"
        write_json(path, spec)
        spec_paths[candidate_id] = path
    setup: dict[str, Any] = {
        "schema_version": SETUP_SCHEMA_VERSION,
        "status": "verified_runnable",
        "run_kind": RUN_KIND,
        "claim_ceiling": CLAIM_CEILING,
        "scene_id": scene_id,
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
        "task_success_contract": task_success_contract,
        "task_success_contract_digest": task_success_contract["contract_digest"],
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
    configured_offering_configuration_run_id: str,
    configured_source_commit: str | None = None,
    offering_digest: str,
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
    policy_controller_configuration: Mapping[str, Any],
    native_controller_configuration: Mapping[str, Any],
    runtime_source_bundle: Mapping[str, Any],
    runtime_source_implementation_commit: str | None = None,
    model_rights: Mapping[str, Any],
    activation_release_window_template: Mapping[str, Any],
    activation_lineage: Mapping[str, Any],
    activation_authorization: Mapping[str, Any],
    output_dir: str | Path,
    task_success_contract: Mapping[str, Any] | None = None,
    policy_observation_setup: Mapping[str, Any] | None = None,
    maximum_hourly_rate_usd: float = 0.8,
    hard_cap_usd: float = 4.0,
    hard_ttl_seconds: int = 9_000,
    scene_id: str | None = None,
) -> dict[str, Any]:
    """Emit the Website descriptor before a user-created activation exists."""

    if not str(profile_id or "").strip():
        raise PolicyCanarySetupError(["policy_canary_profile_id_missing"])
    if not str(configured_offering_configuration_run_id or "").strip():
        raise PolicyCanarySetupError(
            ["policy_canary_configured_offering_configuration_run_id_missing"]
        )
    if not _DIGEST.fullmatch(str(offering_digest or "")):
        raise PolicyCanarySetupError(["policy_canary_offering_digest_invalid"])
    scene_plan_identity = _read(
        scene_plan_path, code="policy_canary_scene_plan_invalid"
    ).get("scene_id")
    derived_scene_id = str(scene_plan_identity or "").removeprefix("interiorgs-")
    selected_scene_id = scene_id or derived_scene_id
    if not selected_scene_id:
        raise PolicyCanarySetupError(["policy_canary_scene_id_missing"])
    # Reuse the complete byte/static preflight without publishing its
    # activation-bound output. The placeholder lineage lives only in a
    # temporary directory and is never returned or persisted as evidence.
    with tempfile.TemporaryDirectory(prefix="policy-canary-presubmission-") as raw:
        verified = materialize_scene839873_policy_canary_setup(
            source_commit=configured_source_commit or source_commit,
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
            task_success_contract=task_success_contract,
            require_confirmed_task_success_contract=False,
            scene_id=selected_scene_id,
        )
    readiness = _read(
        historical_policy_readiness_path,
        code="policy_canary_historical_readiness_invalid",
    )
    readiness_by_id = {
        row["candidate_id"]: row
        for row in readiness["candidates"]
        if isinstance(row, Mapping) and row.get("candidate_id") in CANDIDATE_IDS
    }
    observation_schema_id = "droid_two_camera_robot_state_v1"
    action_schema_id = "droid_absolute_joint_position_v1"
    simulator_runtime_id = "isaac_native_arena_v1"
    embodiment_id = DROID_EMBODIMENT_ID
    task_family_id = "rigid_relocation"
    policies = []
    for candidate_id in CANDIDATE_IDS:
        row = readiness_by_id[candidate_id]
        checkpoint = row["checkpoint"]
        readiness_ref = {
            "uri": (
                "blueprint://policy-canary/readiness/"
                f"{candidate_id}/{readiness['readiness_digest']}"
            ),
            "digest": readiness["readiness_digest"],
        }
        policies.append(
            {
                "candidate_id": candidate_id,
                "display_name": row["model_name"],
                "checkpoint": {
                    "uri": checkpoint["repository"],
                    "digest": checkpoint["inventory_digest"],
                    "size_bytes": checkpoint["total_bytes"],
                },
                "adapter_id": (
                    "openpi_droid_to_official_arena_droid_abs_joint_v2"
                    if candidate_id == "pi05_droid"
                    else "groot_n17_droid_to_official_arena_droid_abs_joint_v2"
                ),
                "license_id": (
                    "apache-2.0-gemma-terms"
                    if candidate_id == "pi05_droid"
                    else "nvidia-open-model-license"
                ),
                "compatibility": {
                    "robot_preset_ids": [EMBODIMENT_ID],
                    "embodiment_ids": [embodiment_id],
                    "observation_schema_ids": [observation_schema_id],
                    "action_schema_ids": [action_schema_id],
                    "simulator_runtime_ids": [simulator_runtime_id],
                    "task_family_ids": [task_family_id],
                },
                "readiness": {
                    "status": "verified_runnable",
                    "receipt": readiness_ref,
                    "reason": None,
                },
            }
        )
    quick = verified["quick_10"]
    cells = [
        {
            "cell_id": row["cell_id"],
            "family": row["family"],
            "seed": row["seed"] % (2**31),
            "partition": (
                "canonical"
                if row["family"] == "canonical_anchor"
                else "held_out"
                if row["family"] == "held_out_composition"
                else "stress"
            ),
            "label": f"{row['family'].replace('_', ' ')} {index + 1}",
            "cell_digest": row["cell_spec_digest"],
        }
        for index, row in enumerate(quick["cells"])
    ]
    empty_counts = {family: 0 for family in QUICK_FAMILY_COUNTS}
    as_of = utc_now_iso()

    def estimate(*, minimum: float, maximum: float, preset_id: str) -> dict[str, Any]:
        basis = {
            "preset_id": preset_id,
            "source_commit": source_commit,
            "runtime_image": NATIVE_TASK_ARENA_IMAGE,
            "maximum_hourly_rate_usd": maximum_hourly_rate_usd,
            "hard_cap_usd": hard_cap_usd,
            "hard_ttl_seconds": hard_ttl_seconds,
        }
        return {
            "duration_minutes": {"minimum": minimum, "maximum": maximum},
            "maximum_authorized_cost_usd": hard_cap_usd,
            "hard_ttl_seconds": hard_ttl_seconds,
            "basis_digest": canonical_digest(basis),
            "as_of": as_of,
        }

    empty_matrix_digest = canonical_digest({"ordered_cells": []})
    setup: dict[str, Any] = {
        "schema_version": PRESUBMISSION_SETUP_SCHEMA_VERSION,
        "source_launch_id": configured_source_launch_id,
        "offering_digest": offering_digest,
        "scene_revision_digest": scene_revision_digest,
        "run_kind": RUN_KIND,
        "claim_ceiling": CLAIM_CEILING,
        "registry_digest": readiness["readiness_digest"],
        "robot_presets": [
            {
                "robot_preset_id": EMBODIMENT_ID,
                "display_name": "DROID-compatible Franka Panda + Robotiq 2F-85",
                "embodiment_id": embodiment_id,
                "task_family_id": task_family_id,
                "simulator_runtime_id": simulator_runtime_id,
                "runtime_image": {
                    "uri": NATIVE_TASK_ARENA_IMAGE,
                    "digest": "sha256:" + NATIVE_TASK_ARENA_IMAGE.rsplit("@sha256:", 1)[1],
                },
                "observation_schema": {
                    "schema_id": observation_schema_id,
                    "cameras": ["external", "wrist"],
                    "modalities": [
                        "rgb_uint8",
                        "joint_position",
                        "gripper_position",
                        "language_instruction",
                    ],
                },
                "action_schema": {
                    "schema_id": action_schema_id,
                    "space": "absolute_7_joint_positions_plus_gripper",
                    "control_hz": 15,
                },
                "readiness": {
                    "status": "verified_runnable",
                    "receipt": {
                        "uri": (
                            "blueprint://policy-canary/readiness/"
                            f"droid-franka-robotiq/{readiness['readiness_digest']}"
                        ),
                        "digest": readiness["readiness_digest"],
                    },
                    "reason": None,
                },
                "policy_candidates": policies,
            }
        ],
        "episode_presets": [
            {
                "preset_id": "quick_10",
                "label": "Quick",
                "episodes_per_policy": 10,
                "availability": "enabled",
                "recommended": True,
                "matrix": {
                    "matrix_digest": canonical_digest({"ordered_cells": cells}),
                    "resolver_id": f"scene{selected_scene_id}_quick10_deterministic",
                    "resolver_version": "v1",
                    "deterministic": True,
                    "cells": cells,
                    "expected_family_counts": dict(QUICK_FAMILY_COUNTS),
                    "coverage_gaps": [],
                },
                "estimate": estimate(minimum=20, maximum=60, preset_id="quick_10"),
            },
            {
                "preset_id": "standard_100",
                "label": "Standard",
                "episodes_per_policy": 100,
                "availability": "coming_later",
                "recommended": False,
                "matrix": {
                    "matrix_digest": empty_matrix_digest,
                    "resolver_id": "standard_100_not_enabled",
                    "resolver_version": "v1",
                    "deterministic": True,
                    "cells": [],
                    "expected_family_counts": empty_counts,
                    "coverage_gaps": [],
                },
                "estimate": estimate(minimum=0, maximum=0, preset_id="standard_100"),
            },
            {
                "preset_id": "deep_500",
                "label": "Deep",
                "episodes_per_policy": 500,
                "availability": "coming_later",
                "recommended": False,
                "matrix": {
                    "matrix_digest": empty_matrix_digest,
                    "resolver_id": "deep_500_not_enabled",
                    "resolver_version": "v1",
                    "deterministic": True,
                    "cells": [],
                    "expected_family_counts": empty_counts,
                    "coverage_gaps": [],
                },
                "estimate": estimate(minimum=0, maximum=0, preset_id="deep_500"),
            },
        ],
        "diagnostics": {
            "zero_action": "nonblocking",
            "deterministic_scripted_positive": "nonblocking",
        },
        "task_success_contract": deepcopy(verified["task_success_contract"]),
        "task_success_contract_digest": verified[
            "task_success_contract_digest"
        ],
        "setup_digest": "",
    }
    setup["setup_digest"] = policy_canary_setup_digest(setup)
    setup = validate_policy_canary_setup(setup)
    controller_ref = _immutable_ref(
        policy_controller_configuration,
        code="policy_canary_controller_configuration_invalid",
    )
    native_controller_ref = _immutable_ref(
        native_controller_configuration,
        code="policy_canary_native_controller_configuration_invalid",
    )
    runtime_source_ref = _immutable_ref(
        runtime_source_bundle,
        code="policy_canary_runtime_source_bundle_invalid",
    )
    runtime_source_commit = runtime_source_implementation_commit or source_commit
    if not _SHA.fullmatch(runtime_source_commit):
        raise PolicyCanarySetupError(
            ["policy_canary_runtime_source_implementation_commit_invalid"]
        )
    rights_ref = _immutable_ref(
        model_rights,
        code="policy_canary_model_rights_invalid",
    )
    observation_setup = None
    if policy_observation_setup is not None:
        raw_observation_setup = dict(policy_observation_setup)
        expected_keys = {
            "schema_version",
            "appearance_asset",
            "appearance_authoring_receipt",
            "wrist_camera_mount_registry",
            "fresh_native_mount_sweep_required",
            "policy_master_resolution_wh",
            "overview_review_resolution_wh",
        }
        if (
            set(raw_observation_setup) != expected_keys
            or raw_observation_setup.get("schema_version")
            != "task_evaluation_policy_observation_setup.v1"
            or raw_observation_setup.get("fresh_native_mount_sweep_required") is not True
            or raw_observation_setup.get("policy_master_resolution_wh") != [640, 360]
            or raw_observation_setup.get("overview_review_resolution_wh") != [1280, 720]
        ):
            raise PolicyCanarySetupError(["policy_canary_observation_setup_invalid"])
        observation_setup = {
            **raw_observation_setup,
            "appearance_asset": _immutable_ref(
                raw_observation_setup["appearance_asset"],
                code="policy_canary_observation_appearance_asset_invalid",
            ),
            "appearance_authoring_receipt": _immutable_ref(
                raw_observation_setup["appearance_authoring_receipt"],
                code="policy_canary_observation_appearance_receipt_invalid",
            ),
            "wrist_camera_mount_registry": _immutable_ref(
                raw_observation_setup["wrist_camera_mount_registry"],
                code="policy_canary_wrist_camera_mount_registry_invalid",
            ),
        }
    progression = _read(
        configured_progression_path,
        code="policy_canary_progression_invalid",
    )
    configured_preparation = progression.get("episode_preparation_request")
    if not isinstance(configured_preparation, Mapping):
        raise PolicyCanarySetupError(["policy_canary_configured_preparation_request_missing"])
    required_template_fields = (
        "scene",
        "construction",
        "robot",
        "task",
        "sensors",
        "runtime",
        "execution_adapter",
    )
    if any(field not in configured_preparation for field in required_template_fields):
        raise PolicyCanarySetupError(["policy_canary_configured_preparation_request_incomplete"])
    preparation_template: dict[str, Any] = {
        "schema_version": "task_evaluation_policy_run_preparation_template.v1",
        **{field: deepcopy(configured_preparation[field]) for field in required_template_fields},
        "execution_adapter": {
            **deepcopy(configured_preparation["execution_adapter"]),
            "runtime_source_bundle": runtime_source_ref,
            "runtime_source_implementation_commit": runtime_source_commit,
            **(
                {"policy_observation_setup": observation_setup}
                if observation_setup is not None
                else {}
            ),
        },
        "controller": {
            "identity": {
                "id": "paired-droid-policy-canary",
                "version": "v2",
            },
            "kind": "policy_container",
            "configuration": native_controller_ref,
            "model_or_asset_rights": rights_ref,
        },
        "publication": {"service_account_readback_required": True},
        "spend": {
            "hard_cap_usd": hard_cap_usd,
            "hard_ttl_seconds": hard_ttl_seconds,
            "maximum_hourly_rate_usd": maximum_hourly_rate_usd,
            "provider_allowlist": ["vast"],
            "retry_cap": 0,
            "selected_provider": "vast",
        },
        "template_digest": "",
    }
    preparation_template["template_digest"] = cross_runtime_canonical_digest(
        preparation_template, digest_field="template_digest"
    )
    legacy_cells = [
        {
            "cell_id": row["cell_id"],
            "family": row["family"],
            "partition": (
                "held_out" if row["family"] == "held_out_composition" else "qualification"
            ),
            "scored": True,
            "seed": row["seed"],
            "cell_spec_digest": row["cell_spec_digest"],
            "resolved_scenario": row["resolved_scenario"],
        }
        for row in quick["cells"]
    ]
    legacy_estimate = {
        "status": "estimated",
        "duration_minutes": {"minimum": 20, "maximum": 60},
        "cost_usd": {"minimum": 0, "maximum": hard_cap_usd},
        "basis_digest": setup["episode_presets"][0]["estimate"]["basis_digest"],
        "as_of": setup["episode_presets"][0]["estimate"]["as_of"],
    }
    legacy_preset_specs = (
        (
            "quick_10",
            "Quick",
            10,
            "enabled",
            True,
            dict(QUICK_FAMILY_COUNTS),
            legacy_cells,
            None,
            0,
            legacy_estimate,
        ),
        (
            "standard_100",
            "Standard",
            100,
            "coming_later",
            False,
            {
                "canonical_anchor": 2,
                "placement_approach": 20,
                "illumination": 14,
                "camera_sensor": 14,
                "bounded_physics": 14,
                "admitted_object_material_cousin": 10,
                "pairwise_stress": 13,
                "held_out_composition": 13,
            },
            None,
            "quick_10",
            10,
            {"status": "unavailable"},
        ),
        (
            "deep_500",
            "Deep",
            500,
            "coming_later",
            False,
            {
                "canonical_anchor": 2,
                "placement_approach": 100,
                "illumination": 70,
                "camera_sensor": 70,
                "bounded_physics": 70,
                "admitted_object_material_cousin": 50,
                "pairwise_stress": 69,
                "held_out_composition": 69,
            },
            None,
            "standard_100",
            100,
            {"status": "unavailable"},
        ),
    )
    legacy_presets = []
    for (
        preset_id,
        label,
        count,
        availability,
        default,
        family_counts,
        preset_cells,
        parent_id,
        parent_count,
        preset_estimate,
    ) in legacy_preset_specs:
        scenario_set_digest = (
            cross_runtime_canonical_digest({"ordered_cells": preset_cells})
            if preset_cells is not None
            else cross_runtime_canonical_digest(
                {
                    "preset_id": preset_id,
                    "status": "coming_later",
                    "scenario_count_per_policy": count,
                }
            )
        )
        preset = {
            "preset_id": preset_id,
            "label": label,
            "scenario_count_per_policy": count,
            "availability": availability,
            "default": default,
            "family_counts": family_counts,
            "scenario_set_digest": scenario_set_digest,
            "parent_preset_id": parent_id,
            "parent_prefix_count": parent_count,
            "nesting_proof_digest": cross_runtime_canonical_digest(
                {
                    "preset_id": preset_id,
                    "scenario_set_digest": scenario_set_digest,
                    "parent_preset_id": parent_id,
                    "parent_prefix_count": parent_count,
                    "selection_rule": "published_ordered_prefix",
                }
            ),
            "estimate": preset_estimate,
        }
        if preset_cells is not None:
            preset["cells"] = preset_cells
        legacy_presets.append(preset)
    legacy_setup: dict[str, Any] = {
        "schema_version": "task_evaluation_policy_run_setup.v1",
        "source_launch_id": configured_source_launch_id,
        "offering_digest": offering_digest,
        "embodiment_id": EMBODIMENT_ID,
        "candidate_ids": list(CANDIDATE_IDS),
        "matrix_profile_id": "franka_rigid_relocation_nested_v1",
        "preregistration": rights_ref,
        "scenario_compiler": {
            "compiler_id": "franka_rigid_relocation_nested_prefix",
            "compiler_version": "v1",
            "selection_rule": "published_ordered_prefix",
            "outcome_independent": True,
            "agent_may_select_cells": False,
        },
        "presets": legacy_presets,
        "preparation_template": preparation_template,
        "setup_digest": "",
    }
    legacy_setup["setup_digest"] = policy_run_setup_digest(legacy_setup)
    legacy_setup = validate_policy_run_setup(legacy_setup)
    configured_preparation_digest = str(progression.get("episode_preparation_request_digest") or "")
    if not _DIGEST.fullmatch(configured_preparation_digest):
        configured_preparation_digest = canonical_digest(configured_preparation)
    activation_automation = {
        "mode": "automatic_after_no_spend_compilation",
        "release_window_template": json.loads(
            json.dumps(dict(activation_release_window_template), allow_nan=False)
        ),
        "lineage": json.loads(json.dumps(dict(activation_lineage), allow_nan=False)),
        "authorization_template": json.loads(
            json.dumps(dict(activation_authorization), allow_nan=False)
        ),
        "requested_mutations": {
            "profile_publication": False,
            "catalog_synchronization": False,
            "standing_authorization": False,
            "policy_campaign_queue": True,
        },
    }
    execution_plan: dict[str, Any] = {
        "schema_version": "task_evaluation_policy_canary_execution_plan.v1",
        "source_commit": source_commit,
        "configured_source_launch_id": configured_source_launch_id,
        "scene_id": selected_scene_id,
        "configured_offering_configuration_run_id": (configured_offering_configuration_run_id),
        "scene_revision_digest": scene_revision_digest,
        "public_setup_digest": setup["setup_digest"],
        "task_success_contract": deepcopy(setup["task_success_contract"]),
        "task_success_contract_digest": setup["task_success_contract_digest"],
        "configured_preparation_request_digest": configured_preparation_digest,
        "policy_controller_configuration": controller_ref,
        "model_rights": rights_ref,
        "resolved_scenarios": quick["cells"],
        "legacy_policy_run_setup": legacy_setup,
        "preparation_template": preparation_template,
        "resource_authority": {
            "maximum_hourly_rate_usd": maximum_hourly_rate_usd,
            "hard_cap_usd": hard_cap_usd,
            "hard_ttl_seconds": hard_ttl_seconds,
            "maximum_provider_allocations": 1,
            "retry_cap": 0,
        },
        "activation_automation": activation_automation,
        "lineage_aliases": {
            "capture_session_id": configured_source_launch_id,
            "capture_session_id_semantics": (
                "configured_scene_offering_source_launch_id_no_capture_upload_session"
            ),
            "intake_id": configured_offering_configuration_run_id,
            "intake_id_semantics": "configured_scene_offering_configuration_run_id",
        },
        "provider_mutation_performed": False,
        "paid_execution_requested": False,
        "plan_digest": "",
    }
    execution_plan["plan_digest"] = canonical_digest(execution_plan, digest_field="plan_digest")
    configured_base_profile = _read(
        launch_profile_path, code="policy_canary_launch_profile_invalid"
    )
    if configured_base_profile.get("profile_digest") != canonical_digest(
        configured_base_profile, digest_field="profile_digest"
    ):
        raise PolicyCanarySetupError(["policy_canary_profile_digest_invalid"])
    wrapper: dict[str, Any] = {
        "schema_version": PROFILE_INPUT_SCHEMA_VERSION,
        "profile_id": profile_id,
        "configured_base_profile_id": configured_base_profile["profile_id"],
        "configured_base_profile_digest": configured_base_profile["profile_digest"],
        "configured_source_launch_id": configured_source_launch_id,
        "scene_id": selected_scene_id,
        "source_commit": source_commit,
        "internal_policy_canary_setup": setup,
        "internal_policy_canary_execution_plan": execution_plan,
        "task_success_contract": deepcopy(setup["task_success_contract"]),
        "task_success_contract_digest": setup["task_success_contract_digest"],
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
        "configured_source_commit": configured_source_commit or source_commit,
        "configured_source_launch_id": configured_source_launch_id,
        "scene_id": selected_scene_id,
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
    service_account_access = _seal_presubmission_service_access(
        destination=destination,
        artifacts={
            "setup": setup_path,
            "profile_materialization_input": wrapper_path,
            "execution_setup_template": execution_template_path,
        },
    )
    return {
        "setup": setup,
        "setup_path": str(setup_path),
        "profile_materialization_input": wrapper,
        "profile_materialization_input_path": str(wrapper_path),
        "execution_setup_template": execution_template,
        "execution_setup_template_path": str(execution_template_path),
        "service_account_access": service_account_access,
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
    wrapper = _read(wrapper_path, code="policy_canary_profile_materialization_input_invalid")
    if (
        wrapper.get("schema_version") != PROFILE_INPUT_SCHEMA_VERSION
        or wrapper.get("configured_source_launch_id") != template.get("configured_source_launch_id")
        or wrapper.get("source_commit") != template.get("source_commit")
        or wrapper.get("task_success_contract")
        != _mapping_or_empty(wrapper.get("internal_policy_canary_setup")).get(
            "task_success_contract"
        )
        or wrapper.get("task_success_contract_digest")
        != _mapping_or_empty(wrapper.get("task_success_contract")).get(
            "contract_digest"
        )
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
    activation = _read(activation_path, code="policy_canary_activation_result_invalid")
    activation_task_success_contract = _mapping_or_empty(
        activation_envelope.get("task_success_contract")
        or activation.get("task_success_contract")
    )
    activation_task_success_contract_digest = activation_envelope.get(
        "task_success_contract_digest"
    ) or activation.get("task_success_contract_digest")
    try:
        validated_activation_contract = validate_rigid_task_success_contract(
            activation_task_success_contract
        )
    except TaskNeutralScoringError as exc:
        raise PolicyCanarySetupError(list(exc.errors)) from exc
    if (
        activation_task_success_contract_digest
        != validated_activation_contract["contract_digest"]
    ):
        raise PolicyCanarySetupError(
            ["policy_canary_task_success_contract_digest_mismatch"]
        )
    return materialize_scene839873_policy_canary_setup(
        source_commit=str(template["source_commit"]),
        configured_source_commit=str(
            template.get("configured_source_commit") or template["source_commit"]
        ),
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
        historical_policy_readiness_path=template["historical_policy_readiness_path"],
        pi05_checkpoint_inventory_path=template["pi05_checkpoint_inventory_path"],
        output_dir=output_dir,
        maximum_hourly_rate_usd=float(template["maximum_hourly_rate_usd"]),
        hard_cap_usd=float(template["hard_cap_usd"]),
        hard_ttl_seconds=int(template["hard_ttl_seconds"]),
        task_success_contract=validated_activation_contract,
        require_confirmed_task_success_contract=True,
        scene_id=str(template.get("scene_id") or SCENE_ID),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    for operation in ("preflight", "presubmission"):
        command = subparsers.add_parser(operation)
        command.add_argument(
            "--parameters",
            required=True,
            help="JSON object containing the materializer's exact keyword arguments.",
        )
    args = parser.parse_args(argv)
    parameters = _read(args.parameters, code="policy_canary_cli_parameters_invalid")
    if args.operation == "preflight":
        result = materialize_setup_preflight_decision(**parameters)
    else:
        result = materialize_policy_canary_presubmission_setup(**parameters)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CANDIDATE_IDS",
    "PolicyCanarySetupError",
    "QUICK_FAMILY_COUNTS",
    "materialize_scene839873_policy_canary_setup",
    "materialize_policy_canary_presubmission_setup",
    "materialize_scene839873_policy_canary_setup_from_template",
    "materialize_setup_preflight_decision",
    "main",
]
