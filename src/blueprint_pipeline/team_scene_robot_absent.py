"""A website job's scene half without Blueprint's idle arm, when the arm's part cannot be made.

Owner decision 2026-11-15: the simulation matters for V1 only as a scene package a deployment company downloads
and runs its own evaluations on; Blueprint runs no robot of its own. The scene half (``team_scene_packet``) places
one idle DROID Franka at the first clear stance, and its packet (``native_task_arena_packet``) is built around
that arm: its station, its cameras, its contact sensors. A station that cannot be found, a camera the arm's
policy could not use, or an arm outside the room's look refused the whole half, and with it the package, for a
reason that only concerns the arm.

For the website compile (``package_only``) such a reason is no refusal. ``robot_only_reason`` names the
errors that concern only the arm (by their codes, never by a guess); when every error of a refusal is one, the
half is written without the arm: ``materialize_scene_package`` stages the same scene files the packet stages
(the room's collision authored the same way), and writes a plan that holds what the package is made of (the
placed rows, the task with its success rule, the cadence, the object's extent and the room's representation)
with ``robot`` and ``cameras`` empty and ``robot_absent`` naming why. Nothing is bought, nothing is guessed
in the arm's place, and the record says the arm is absent (``team_scene_inputs.v1.json``,
``robot_station_absent``). Any other error refuses as before.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .common import write_json
from .decision_evidence_contracts import canonical_digest

DIRECTORY = "scene-package"
PLAN_FILENAME = "scene_package_plan.v1.json"
RECEIPT_FILENAME = "scene_package_receipt.v1.json"
PLAN_SCHEMA = "blueprint.scene_package_plan.v1"
RECEIPT_SCHEMA = "blueprint.scene_package_receipt.v1"
FIELD = "robot_station_absent"
#: The exact codes that can only mean "no place for, or no parts of, Blueprint's reference arm". Each was read at
#: its raise site; nothing about the room, the object or the task is here, and no catch-all is.
ROBOT_ONLY_CODES = frozenset({
    # team_embodiment_stance.ground_stance: every floor spot was tried and none is clear for the arm's pedestal.
    "team_embodiment_task_motion_no_clear_stance",  # each spot is in the moving part's path
    "team_embodiment_free_base_no_clear_stance",  # no spot at all is clear of obstacles
    # The pedestal's clearance setting (``PEDESTAL_SEARCH``). Its height (``stance_height_invalid``) is not here:
    # a support top below the floor found can be a wrong room as well.
    "team_embodiment_stance_clearance_invalid",
    # team_embodiment: a registered body's station is out of its reach, or the plan names no station.
    "team_embodiment_station_out_of_reach", "team_embodiment_station_unavailable",
    # The operator's DROID profile (cameras and reset of the arm), read by team_scene_packet._droid_profile.
    "team_scene_droid_profile_binding_invalid", "team_scene_droid_profile_invalid",
    # The derived collider written only to move furniture out of the arm's way (_robot_access_applied).
    "team_scene_robot_access_readback_disagrees",
    # native_task_runtime_contract: the arm's base pose and reset joints.
    "native_task_runtime_robot_base_pose_invalid", "native_task_runtime_robot_reset_joints_missing",
    # native_task_arena_scene_plan: the arm's contact sensors, and its base outside the room's look.
    "native_task_robot_contact_topology_unavailable", "native_task_arena_forbidden_robot_contact_topology_missing",
    "native_task_appearance_frame_excludes_scene_position:robot_base",
    # native_task_arena_packet._clone_request: the splat's quality gate exists for the arm's policy cameras. The
    # look file still ships, and the package says it did not pass (``APPEARANCE_QUALITY``).
    "native_task_arena_particlefield_quality_missing_or_invalid"})
#: Codes that carry a detail after ``:`` (a camera role, a joint name, an error type) and are the arm's whatever it is.
ROBOT_ONLY_CODES_WITH_DETAIL = frozenset({
    "team_scene_robot_access_unwritten",
    "native_task_runtime_robot_reset_joint_invalid", "native_task_runtime_robot_reset_joint_missing",
    "native_task_runtime_robot_reset_joint_unexpected",
    # The arm's cameras (the profile's poses on the arm's station) in the runtime contract and the plan.
    "native_task_runtime_camera_invalid", "native_task_runtime_camera_role_invalid",
    "native_task_runtime_camera_missing", "native_task_runtime_camera_policy_role_invalid",
    "native_task_runtime_camera_scoring_forbidden", "native_task_runtime_camera_pose_frame_invalid",
    "native_task_runtime_camera_parent_invalid", "native_task_runtime_camera_pose_invalid",
    "native_task_runtime_camera_convention_invalid", "native_task_runtime_camera_intrinsics_invalid",
    "native_task_arena_camera_framing_infeasible"})
#: The arm's only when the search's own evidence (``arm_refused_with_floor``, set at the raise site) shows a spot with floor
#: under it refused for the stand's or the arm's reasons; spots with no floor under them alone are the room's.
ROOM_MESH_NO_CLEAR_STANCE = "team_embodiment_room_mesh_no_clear_stance"
#: Written in a recorded absence with its error type only: a message can hold a host path.
TYPE_ONLY_DETAIL = frozenset({"team_scene_robot_access_unwritten"})
#: Recorded in the absence (and the status) when the look file failed its quality gate; the README says so.
APPEARANCE_QUALITY = "native_task_arena_particlefield_quality_missing_or_invalid"


def _codes(error: BaseException) -> list[str]:
    errors = getattr(error, "errors", None)
    if isinstance(errors, (list, tuple)) and errors:
        return [str(item) for item in errors]
    return [code for code in str(error).replace(",", ";").split(";") if code]


def _arm_refused_spots(error: BaseException) -> bool:
    """A spot with floor under it was refused for the stand's or the arm's reasons (set at the raise site)."""
    return any(getattr(item, "arm_refused_with_floor", None) is True for item in (error, error.__cause__))


def _robot_only(code: str, error: BaseException) -> bool:
    if code == ROOM_MESH_NO_CLEAR_STANCE:
        return _arm_refused_spots(error)
    return code in ROBOT_ONLY_CODES or (
        ":" in code and code.split(":", 1)[0] in ROBOT_ONLY_CODES_WITH_DETAIL and code.split(":", 1)[1] != "")


def _recorded(code: str) -> str:
    base, _, detail = code.partition(":")
    return f"{base}:{detail.split(':', 1)[0]}" if base in TYPE_ONLY_DETAIL else code


def robot_only_reason(error: BaseException) -> str | None:
    """The refusal's own codes, joined, when every one concerns only the arm; else None.

    A code whose message can name a host path (``TYPE_ONLY_DETAIL``) is kept with its error type only.
    """

    codes = _codes(error)
    if not codes or not all(_robot_only(code, error) for code in codes):
        return None
    return ",".join(sorted({_recorded(code) for code in codes}))[:400]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def stage_records(adapter: Mapping[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """The revision's stage-5 and stage-4 records, whole and sealed as published, from the adapted task's documents.

    The package reads them with their own readers (``website_company_simulation.physics_check``).
    """

    documents = ((adapter.get("source_documents") or {}).get("documents") or {})
    return tuple(dict(documents[key]) if isinstance(documents.get(key), Mapping) else None  # type: ignore[return-value]
                 for key in ("native_import_qualification", "static_qualification"))


def _rows(assets: Sequence[Mapping[str, Any]], *, task_spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Each row's spawn type and visibility, as the packet's runtime contract resolves them. A role's first row is
    the one the composition reads; a row of a role repeated after it (a further object of the job,
    ``website_further_objects``) keeps its own place, name and pose, and never replaces it."""

    from .articulated_runtime_composition import plan_articulated_runtime_composition

    by_role: dict[str, Mapping[str, Any]] = {}
    for row in assets:
        by_role.setdefault(str(row["semantic_role"]), row)
    task_object = by_role["task_object"]
    composition = plan_articulated_runtime_composition(
        task_spec=task_spec, twin_usd_filename=str(task_object["filename"]),
        scene_collision_filename=str(by_role["scene_collision"]["filename"]),
        appearance_filename=(str(by_role["scene_appearance"]["filename"]) if "scene_appearance" in by_role else None),
        twin_position_world_m=(task_object.get("pose_world") or {}).get("position_world_m"),
        twin_object_type=str(task_object.get("object_type") or "") or None)
    planned = {row["semantic_role"]: row for row in composition["objects"]}
    rows = []
    for row in assets:
        role = str(row["semantic_role"])
        rows.append({**{key: row[key] for key in ("semantic_role", "filename", "pose_world", "source")},
                     **{key: row[key] for key in ("name", "asset_id", "source_asset_id", "reset_state",
                                                  "articulation_adaptation") if key in row},
                     "object_type": (planned.get(role) or {}).get("object_type") or row.get("object_type") or "RIGID",
                     "visible": bool((planned.get(role) or {}).get("visible", row.get("visible", True)))})
    return rows


def materialize_scene_package(*, assets: Sequence[Mapping[str, Any]], task_spec: Mapping[str, Any],
                              scene_id: str, task_id: str, physics_frequency_hz: Any, evidence_root: Path,
                              output_dir: Path, reason: str, representation: str | None,
                              stage_5_record: Mapping[str, Any] | None,
                              stage_4_record: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Stage the scene's files under ``output_dir/assets`` and write the arm-free plan and its receipt."""

    from .native_task_arena_packet import NativeTaskArenaPacketError, _asset_source, _stage_verified_asset
    from .native_task_arena_runtime import author_gpu_compatible_scene_collision
    from .native_task_arena_scene_plan import _cadence
    from .native_task_camera_framing_expectation import measure_task_object_extent_m
    from .usd_binary_read_cache import editable

    evidence, output = Path(evidence_root).resolve(), Path(output_dir).resolve()
    if output.exists():
        raise NativeTaskArenaPacketError(["native_task_arena_packet_output_exists"])
    (output / "assets").mkdir(parents=True)
    objects, bindings, subject_named = [], [], False
    for row in _rows(assets, task_spec=task_spec):
        role, filename = row["semantic_role"], str(row["filename"])
        source, digest, _size = _asset_source(row, evidence_root=evidence)
        destination = output / "assets" / filename
        if destination.exists() or Path(filename).name != filename:
            raise NativeTaskArenaPacketError([f"native_task_arena_packet_asset_filename_invalid:{role}"])
        adaptation = None
        if role == "scene_collision":
            # Authored as the packet authors it, so a company gets the same collision either way.
            adaptation = author_gpu_compatible_scene_collision(source, destination, opened=editable(source))
        if adaptation is None:
            _stage_verified_asset(source, destination, link_within=evidence)
            if _sha256(destination) != digest:
                raise NativeTaskArenaPacketError([f"native_task_arena_packet_asset_copy_mismatch:{role}"])
        staged = _sha256(destination)
        bindings.append({"semantic_role": role, "source": dict(row["source"]),
                         "staged_relative_path": f"assets/{filename}", "staged_sha256": staged,
                         "staged_size_bytes": destination.stat().st_size,
                         **({"static_scene_collision_adaptation": adaptation} if adaptation else {})})
        # Each row's own runtime name when it has one (a further object's), else its role; the first task row is the
        # task's subject.
        objects.append({"name": str(row.get("name") or role), "semantic_role": role,
                        "asset_id": row.get("asset_id", role),
                        "task_subject": role == "task_object" and not subject_named, "object_type": row["object_type"],
                        "usd_path": f"assets/{filename}", "sha256": staged,
                        "size_bytes": destination.stat().st_size, "visible": row["visible"],
                        "pose_world": row["pose_world"],
                        "reset_state": row.get("reset_state") or {"root_pose_world": row["pose_world"],
                                                                  "joint_positions": {}}})
        subject_named = subject_named or role == "task_object"
    subject = next(row for row in objects if row["semantic_role"] == "task_object")
    try:
        observability: dict[str, Any] | None = {
            "task_object_extent_m": measure_task_object_extent_m(output / subject["usd_path"])}
    except Exception:  # noqa: BLE001 - an unmeasured extent is left out of the rule, never guessed
        observability = None
    plan: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA, "scene_id": scene_id, "task_id": task_id,
        "task_kind": str(task_spec.get("task_kind") or ""), "task_spec": json.loads(json.dumps(task_spec)),
        "asset_directory": "assets", "objects": objects,
        "appearance_frame_alignment": {"representation": representation} if representation else None,
        "task_object_observability": observability,
        "cadence": _cadence({"task_spec": task_spec}, physics_frequency_hz=physics_frequency_hz),
        "robot": None, "cameras": [], "robot_absent": {"reason": reason},
        "stage_5": dict(stage_5_record) if stage_5_record is not None else None,
        "stage_4": dict(stage_4_record) if stage_4_record is not None else None,
        "claim_boundary": {"plan_is_not_native_application_proof": True, "plan_is_not_policy_episode_evidence": True,
                           "simulator_execution_is_not_physical_truth": True},
        "plan_digest": ""}
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    write_json(output / PLAN_FILENAME, plan)
    receipt = {"schema_version": RECEIPT_SCHEMA, "status": "scene_package_completed", "scene_id": scene_id,
               "task_id": task_id, "plan_digest": plan["plan_digest"], "source_bindings": bindings,
               "artifacts": [{"role": "scene_package_plan", "relative_path": PLAN_FILENAME,
                              "size_bytes": (output / PLAN_FILENAME).stat().st_size,
                              "sha256": _sha256(output / PLAN_FILENAME)}],
               "robot_included": False, "native_application_claimed": False, "policy_episode_claimed": False,
               "receipt_digest": ""}
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    write_json(output / RECEIPT_FILENAME, receipt)
    return {"plan": plan, "receipt": output / RECEIPT_FILENAME}


def _sealed_record(record_path: Any) -> Mapping[str, Any]:
    """The half's record when it reads and seals itself, else an empty one."""

    try:
        record = json.loads(Path(str(record_path or "")).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return {}
    if not isinstance(record, Mapping) or record.get("inputs_digest") != canonical_digest(
            record, digest_field="inputs_digest"):
        return {}
    return record


#: Owner direction 2026-11-15: Blueprint builds the scene and the package and runs no test of its own, so a half
#: the website compile writes records that no paid robot-free probe is queued for it (``team_scene_packet``). Only a
#: half that records it is spared the probe and its holds; a half compiled before keeps its probe state.
PROBE_FIELD = "robot_free_probe"
PROBE_NOT_QUEUED = "not_queued"


def half_queues_no_probe(record_path: Any) -> bool:
    """Whether the half at ``record_path`` records, sealed, that no robot-free probe is queued for it."""

    return _sealed_record(record_path).get(PROBE_FIELD) == PROBE_NOT_QUEUED


__all__ = ["DIRECTORY", "FIELD", "PLAN_FILENAME", "PLAN_SCHEMA", "RECEIPT_FILENAME", "ROBOT_ONLY_CODES", "ROBOT_ONLY_CODES_WITH_DETAIL", "APPEARANCE_QUALITY",
           "PROBE_FIELD", "PROBE_NOT_QUEUED", "half_queues_no_probe", "materialize_scene_package", "robot_only_reason", "stage_records"]
