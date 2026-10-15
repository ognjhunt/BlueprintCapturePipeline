"""A prepared scene's own half of a team evaluation, compiled on the CPU.

Beta journey step: L5 team policy evaluation. A team brings its robot and its
policy; the scene brings its geometry, its task and the success contract the
simulator judges by. Nothing else is needed before a team can run, so a
published configured scene revision (robot-neutral by construction) is
compiled here straight into the scene half a team run reads:

- the scene's packet: its collision, its native appearance and the task object
  from the revision's configured bundle, and the task the rigid adapter derives
  from the revision's own documents (``task_evaluation_rigid_relocation_native_adapter``);
- a robot station: the packet format carries one robot, so Blueprint's DROID
  Franka stands in it (cameras and reset from the operator's DROID profile)
  on a mount level with the task's support, at the closest clear floor spot
  facing the task (``team_embodiment_stance``). A team's own fixed arm takes the
  same station and must reach the task from it; a free base finds its own
  stance; a team on the DROID interface runs this Franka;
- the sealed success contract: the scene owner's when the task requires one,
  else the strategy's compatibility default (as every canary setup seals it);
- ``team_scene_inputs.v1.json``, which the team registry reads
  (``team_candidate_quick10_registry``) in place of a frozen internal pair.

No robot placement inference, no GPU and no prior run. The packet is never
construction evidence: its receipt says no native application happened, the
first shard of every team run checks the scene before any team episode (the
robot-free probe), and void episodes past the cap refuse the run. Passive
destination bodies and articulated tasks are refused by name.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest

SCHEMA = "blueprint.team_scene_inputs.v1"
CLAIM_CEILING = "diagnostic_policy_execution"
RECORD_FILENAME = "team_scene_inputs.v1.json"
PACKET_DIRECTORY = "native-task-packet"
# A pedestal for an arm's base: its half extent on the floor, metres.
STATION_FOOTPRINT_HALF_EXTENT_M = (0.15, 0.15)
EXECUTION_AUTHORITY = {"maximum_provider_allocations": 1, "retry_cap": 0,
                       "single_warm_provider_session_required": True,
                       "caller_surviving_watchdog_required": True,
                       "billing_teardown_provider_zero_required": True}
# Controls are never run before a team evaluation; a contract that requires them per cell is refused
# by the registry (``team_candidate_strict_controls_runtime_unavailable``).
PENDING_CONTROLS = {"mode": "nonblocking_diagnostic_pending", "typed_gap": "controls_pending_at_submission",
                    "policy_execution_blocked": False}
RIGID_STRATEGIES = ("pick_and_place", "planar_push")


class TeamScenePacketError(ValueError):
    """The scene cannot carry a team evaluation; the message names why."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _record(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha256(path)}


def _read(path: Path, code: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TeamScenePacketError(code) from exc
    if not isinstance(value, dict):
        raise TeamScenePacketError(code)
    return value


def _droid_profile(path: Path, digest: str) -> dict[str, Any]:
    if Path(path).is_symlink() or not Path(path).is_file() or _sha256(Path(path)) != digest:
        raise TeamScenePacketError("team_scene_droid_profile_binding_invalid")
    profile = _read(path, "team_scene_droid_profile_invalid")
    roles = {str(row.get("role")): row for row in profile.get("cameras") or [] if isinstance(row, Mapping)}
    if (profile.get("schema_version") != "native_task_arena_packet_request.v1"
            or profile.get("request_digest") != canonical_digest(profile, digest_field="request_digest")
            or set(roles) != {"external", "wrist", "overview"} or roles["wrist"].get("pose_frame") != "robot_body"
            or not isinstance(roles["external"].get("intrinsics"), Mapping)
            or not isinstance(profile.get("robot_joint_reset_positions_rad"), Mapping)):
        raise TeamScenePacketError("team_scene_droid_profile_invalid")
    return profile


def _runtime_source(path: Path) -> dict[str, Any]:
    receipt = _read(path, "team_scene_runtime_source_invalid")
    if (receipt.get("schema_version") != "native_task_runtime_source_packet.v1"
            or not str(receipt.get("packet_sha256") or "").startswith("sha256:")
            or int(receipt.get("packet_size_bytes") or 0) <= 0 or receipt.get("redistribution_permitted") is not True):
        raise TeamScenePacketError("team_scene_runtime_source_invalid")
    return _record(path)


def _cameras(profile: Mapping[str, Any], *, base: list[float], start: list[float],
             target: list[float]) -> list[dict[str, Any]]:
    from .task_evaluation_diagnostic_native_arena_compiler import _world_camera

    roles = {str(row["role"]): dict(row) for row in profile["cameras"]}
    intrinsics = roles["external"]["intrinsics"]
    direction = [target[0] - start[0], target[1] - start[1]]
    norm = math.hypot(*direction) or 1.0
    lateral = [-direction[1] / norm, direction[0] / norm, 0.0]
    midpoint = [(start[i] + target[i]) / 2.0 for i in range(3)]
    overview = _world_camera("overview", position=[midpoint[0] + 0.9 * lateral[0], midpoint[1] + 0.9 * lateral[1],
                                                   max(base[2], start[2]) + 1.45], target=midpoint,
                             intrinsics=intrinsics)
    overview["policy_input"] = False
    return [_world_camera("external", position=[base[0], base[1], base[2] + 1.35], target=start,
                          intrinsics=intrinsics), json.loads(json.dumps(roles["wrist"])), overview]


def materialize_team_scene(*, revision: Mapping[str, Any], documents: Mapping[str, Path],
                           configured_scene_bundle_path: str | Path, droid_profile_path: str | Path,
                           droid_profile_digest: str, runtime_source_receipt_path: str | Path,
                           output_root: str | Path,
                           native_appearance_materializer: Callable[..., Mapping[str, Any]] | None = None
                           ) -> dict[str, Any]:
    """Compile the revision's scene half under ``output_root`` and seal its team scene inputs."""

    from . import task_evaluation_configured_controls_deferred_inputs as deferred
    from . import task_evaluation_native_arena_episode_compiler as compiler
    from .adp_articulated_task_success_contract import (
        seal_task_success_contract,
        validate_task_success_contract,
    )
    from .adp_task_scoring import TaskNeutralScoringError
    from .native_task_arena_packet import materialize_native_task_arena_packet
    from .task_evaluation_rigid_owner_contract import _derive_configured_owner_success_contract
    from .task_evaluation_rigid_relocation_native_adapter import (
        DEFINITION_CONTRACT_PATH,
        SOURCE_OBJECT_CONTRACT_PATH,
        adapt_rigid_relocation_task_template,
    )
    from .team_embodiment import TeamEmbodimentError
    from .team_embodiment_stance import ground_stance

    root = Path(output_root).resolve()
    if root.exists():
        raise TeamScenePacketError("team_scene_output_exists")
    if revision["task_template"].get("destination") is not None:
        raise TeamScenePacketError("team_scene_destination_body_unsupported")
    if _read(documents[DEFINITION_CONTRACT_PATH], "team_scene_task_invalid").get("strategy") not in RIGID_STRATEGIES:
        raise TeamScenePacketError("team_scene_task_kind_unsupported")
    bundle = Path(configured_scene_bundle_path)
    declared = revision.get("configured_scene_bundle") or {}
    if (bundle.is_symlink() or not bundle.is_file() or bundle.stat().st_size != declared.get("size_bytes")
            or _sha256(bundle) != declared.get("digest")):
        raise TeamScenePacketError("team_scene_configured_bundle_binding_invalid")
    profile = _droid_profile(Path(droid_profile_path), droid_profile_digest)
    runtime_source = _runtime_source(Path(runtime_source_receipt_path))
    root.mkdir(parents=True, mode=0o750)

    references = deferred._materialized_references(revision=revision, documents=documents)
    adapter = adapt_rigid_relocation_task_template(configured_revision=revision, materialized_references=references)
    definition = adapter["native_task_definition"]
    task_spec = compiler._runtime_subject_task_spec(definition["task_spec"])
    task_spec["success_criteria"] = adapter["native_success_criteria"].get("criteria")
    scene_id, task_id = str(revision["scene_identity"]["id"]), str(revision["task_template"]["identity"]["id"])
    try:
        owner = _derive_configured_owner_success_contract(task_spec, site_id=scene_id, task_id=task_id,
                                                         team_namespace=revision.get("team_namespace"))
    except TaskNeutralScoringError as exc:
        raise TeamScenePacketError("team_scene_owner_contract_invalid:" + ",".join(exc.errors)) from exc
    if owner is not None:
        task_spec.update(task_success_contract=owner, task_success_contract_digest=owner["contract_digest"])

    assets = compiler._extract_configured_assets(bundle, output_root=root)
    appearance = dict((native_appearance_materializer or compiler._materialize_native_particlefield_appearance)(
        source_path=assets["appearance"], output_root=root / "native-appearance"))
    pose = definition["task_object_pose_world"]
    rows = [compiler.evidence_asset_row(Path(appearance["path"]), root=root, semantic_role="scene_appearance"),
            compiler.evidence_asset_row(assets["collision"], root=root, semantic_role="scene_collision"),
            {**compiler.evidence_asset_row(assets["replacement"], root=root, semantic_role="task_object",
                                           pose_world=pose),
             "asset_id": task_spec["subject_asset_id"], "source_asset_id": task_spec["source_subject_identity"],
             "object_type": "RIGID", "reset_state": {"root_pose_world": pose, "joint_positions": {}}}]

    # The station: a mount level with the task's support, on the closest clear floor facing the task.
    support = adapter["source_documents"]["documents"]["support_plane"]
    subject = _read(documents[SOURCE_OBJECT_CONTRACT_PATH], "team_scene_subject_bounds_invalid")
    start = [float(v) for v in task_spec["start_pose_world"][:3]]
    try:
        station = ground_stance(
            collision_path=assets["collision"], collision_sha256=_sha256(assets["collision"]), offset=[0.0, 0.0, 0.0],
            subject_center=start, subject_minimum=[float(v) for v in subject["aabb_min_xyz_m"]],
            subject_maximum=[float(v) for v in subject["aabb_max_xyz_m"]],
            footprint_half_extent_m=list(STATION_FOOTPRINT_HALF_EXTENT_M), root_world_z_m=float(support["top_z_m"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise TeamScenePacketError("team_scene_station_inputs_invalid") from exc
    except TeamEmbodimentError as exc:
        raise TeamScenePacketError(",".join(exc.errors)) from exc
    base = {"position_world_m": station["position_world_m"], "orientation_xyzw": station["orientation_xyzw"]}
    request: dict[str, Any] = {
        "schema_version": "native_task_arena_packet_request.v1", "scene_id": scene_id, "task_id": task_id,
        "task_spec": task_spec, "task_joint_bindings": definition.get("task_joint_bindings") or [],
        "task_state_binding": definition.get("task_state_binding"), "assets": rows,
        "robot_base_pose_world": base,
        "robot_joint_reset_positions_rad": dict(profile["robot_joint_reset_positions_rad"]),
        "cameras": _cameras(profile, base=station["position_world_m"], start=start,
                            target=[float(v) for v in task_spec["target_position_world_m"]]),
        "scenario": adapter["native_episode_execution"].get("scenario"),
        "physics_frequency_hz": adapter["native_episode_execution"].get("physics_frequency_hz"),
        "configured_task_template_adapter": adapter,
        "appearance_variant": compiler.packet_appearance_variant(appearance), "request_digest": ""}
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    packet = root / PACKET_DIRECTORY
    materialize_native_task_arena_packet(request=request, evidence_root=root, output_dir=packet,
                                         link_sources_within=root)
    plan = _read(packet / "native_task_arena_scene_plan.v1.json", "team_scene_packet_invalid")
    receipt = packet / "native_task_arena_packet_receipt.v1.json"
    try:
        contract = owner or seal_task_success_contract(
            task_kind="rigid_pick_place", task_spec=plan["task_spec"], site_id=str(plan["scene_id"]),
            task_id=str(plan["task_id"]), author_source="compatibility_default",
            author_id="blueprint:manipulation_strategy_defaults.v1", confirmation_status="confirmed")
        contract = validate_task_success_contract(contract, task_kind="rigid_pick_place")
    except TaskNeutralScoringError as exc:
        raise TeamScenePacketError("team_scene_success_contract_invalid:" + ",".join(exc.errors)) from exc
    packet_record = _record(receipt)
    value = {
        "schema_version": SCHEMA, "claim_ceiling": CLAIM_CEILING,
        "scene_revision_digest": revision["revision_digest"],
        "configuration_digest": canonical_digest({"scene_revision_digest": revision["revision_digest"],
                                                  "packet": packet_record["sha256"],
                                                  "station": station["stance_digest"]}),
        "plan_digest": plan["plan_digest"], "task_success_contract": contract,
        "task_success_contract_digest": contract["contract_digest"],
        "base_native_packet": packet_record, "runtime_source": runtime_source, "construction_result": packet_record,
        "execution_authority": dict(EXECUTION_AUTHORITY), "control_diagnostic": dict(PENDING_CONTROLS),
        "robot_station": station, "droid_profile_digest": droid_profile_digest, "inputs_digest": ""}
    value["inputs_digest"] = canonical_digest(value, digest_field="inputs_digest")
    path = root / RECORD_FILENAME
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    path.chmod(0o444)
    return {"path": str(path), "inputs_digest": value["inputs_digest"], "scene_id": scene_id, "task_id": task_id,
            "robot_station": station}


def read_team_scene_inputs(path: str | Path) -> dict[str, Any]:
    """The scene half a team run stages from (``team_candidate_quick10_staging.SCENE_INPUT_FIELDS``), or raise."""

    from .team_candidate_quick10_staging import SCENE_INPUT_FIELDS

    value = _read(Path(path), "team_scene_inputs_invalid")
    if (value.get("schema_version") != SCHEMA or value.get("claim_ceiling") != CLAIM_CEILING
            or value.get("inputs_digest") != canonical_digest(value, digest_field="inputs_digest")
            or value.get("construction_result") != value.get("base_native_packet")):
        raise TeamScenePacketError("team_scene_inputs_invalid")
    for field in ("base_native_packet", "runtime_source"):
        record = value.get(field) or {}
        if not Path(str(record.get("path") or "")).is_file() or _record(Path(record["path"])) != record:
            raise TeamScenePacketError(f"team_scene_inputs_{field}_changed")
    return {**{key: value[key] for key in SCENE_INPUT_FIELDS if key in value},
            "control_diagnostic": dict(value["control_diagnostic"])}


def prepare_team_scene(*, run_root: str | Path, capture_root: str | Path, scenario_id: str,
                       output_root: str | Path, fetcher: Callable[[Mapping[str, Any]], bytes] | None = None,
                       **packet_inputs: Any) -> dict[str, Any]:
    """Compile a completed source launch's published scene and record it where the team registry reads it.

    The launch's run root is also where its robot-free probe receipt is retained
    (``robot_free_readiness_retention``), so it is the scene's probe root.
    """

    from .robot_free_readiness_retention import _revision
    from .task_evaluation_configured_controls_progression_worker import _validate_source
    from .task_evaluation_scene_rigid_robot_free_readiness import fetch_revision_documents
    from .team_candidate_quick10_registry import write_derived_profile

    result, _receipt, _zero = _validate_source(Path(run_root))
    revision = _revision(result["configured_scene_revision_reference"],
                         revision_digest=result["configured_scene_revision_digest"], fetcher=fetcher)
    root = Path(output_root)
    documents = fetch_revision_documents(revision=revision, output_root=root, fetcher=fetcher)
    compiled = materialize_team_scene(revision=revision, documents=documents, output_root=root / "scene",
                                      **packet_inputs)
    descriptor = _read(Path(capture_root) / "capture_descriptor.json", "team_scene_capture_descriptor_invalid")
    written = write_derived_profile({
        "capture_root": str(Path(capture_root).resolve()), "scene_id": str(descriptor.get("scene_id") or ""),
        "task_id": compiled["task_id"], "scenario_id": scenario_id,
        "robot_free_probe_root": str(Path(run_root).resolve()),
        "scene_revision_digest": revision["revision_digest"], "team_scene_inputs_path": compiled["path"]})
    return {**compiled, "registry": written}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ("run-root", "capture-root", "scenario-id", "output-root", "configured-scene-bundle",
                 "droid-profile", "droid-profile-digest", "runtime-source-receipt"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args(argv)
    result = prepare_team_scene(
        run_root=args.run_root, capture_root=args.capture_root, scenario_id=args.scenario_id,
        output_root=args.output_root, configured_scene_bundle_path=args.configured_scene_bundle,
        droid_profile_path=args.droid_profile, droid_profile_digest=args.droid_profile_digest,
        runtime_source_receipt_path=args.runtime_source_receipt)
    if result["registry"].get("status") == "written":
        # The job lists the evaluation once its offer is republished with the scene half.
        from .agent_execution_offer import publish_agent_execution_offer

        result["offer"] = publish_agent_execution_offer(Path(args.capture_root))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if (result.get("offer") or {}).get("status") == "published" else 1


__all__ = [
    "CLAIM_CEILING",
    "RECORD_FILENAME",
    "SCHEMA",
    "STATION_FOOTPRINT_HALF_EXTENT_M",
    "TeamScenePacketError",
    "materialize_team_scene",
    "prepare_team_scene",
    "read_team_scene_inputs",
]


if __name__ == "__main__":
    raise SystemExit(main())
