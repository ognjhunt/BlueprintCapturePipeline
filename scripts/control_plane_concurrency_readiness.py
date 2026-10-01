"""Real CPU placement and readiness for joined developmental load scenes.

ADP-009D/day 28. The robot collision body is an explicit input fixture. Geometry,
task trajectory, position IK, reset/orientation screening and document validators
execute on CPU; this establishes no native or physical robot qualification.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from urllib.parse import urlsplit

from scripts.control_plane_concurrency_fixture import FilesystemObjectStore
from scripts.control_plane_concurrency_provider import file_record


def fetch_bound_reference(reference: dict, *, object_root: Path, path: Path) -> Path:
    parsed = urlsplit(reference["uri"])
    if parsed.scheme != "s3" or parsed.query or parsed.fragment:
        raise ValueError("harness_readiness_reference_uri_invalid")
    size = reference["size_bytes"]
    if isinstance(size, bool) or not isinstance(size, int) or not 0 < size <= 64 * 1024**2:
        raise ValueError("harness_readiness_reference_size_invalid")
    store = FilesystemObjectStore(object_root)
    kwargs = {"Bucket": parsed.netloc, "Key": parsed.path.lstrip("/")}
    if store.head_object(**kwargs)["ContentLength"] != size:
        raise ValueError("harness_readiness_reference_size_mismatch")
    with store.get_object(**kwargs)["Body"] as source, path.open("xb") as target:
        shutil.copyfileobj(source, target, 1024 * 1024)
    if file_record(path) != {"digest": reference["digest"], "size_bytes": size}:
        raise ValueError("harness_readiness_reference_digest_mismatch")
    return path


def materialize_fixture_placement(
    *, configured: dict, object_root: Path, output_root: Path
) -> dict:
    from pxr import Usd, UsdGeom
    from tests.test_task_evaluation_robot_placement_geometry import _box
    from blueprint_pipeline.task_evaluation_configured_controls_deferred_inputs import (
        REVISION_DOCUMENTS,
        derive_native_trajectory_plan,
    )
    from blueprint_pipeline.task_evaluation_robot_placement_geometry import (
        build_robot_placement_geometry_index,
        summarize_robot_placement_geometry,
        enumerate_robot_placement_geometry_candidates,
        validate_robot_placement_geometry_candidate,
    )
    from blueprint_pipeline.task_evaluation_robot_placement_inventory import (
        build_candidate_inventory_checkpoint,
    )
    from blueprint_pipeline.task_evaluation_robot_placement_agent_cli import run_robot_placement_cli
    from blueprint_pipeline.task_evaluation_robot_placement_readiness_candidate import (
        materialize_robot_placement_readiness_candidate,
    )
    from blueprint_pipeline.task_evaluation_robot_placement_trajectory import (
        placement_trajectory_from_native_plan,
    )

    output_root.mkdir(mode=0o700)
    revision = configured["revision"]
    documents = {}
    for index, (contract, (section, field)) in enumerate(REVISION_DOCUMENTS.items()):
        documents[contract] = fetch_bound_reference(
            revision[section][field],
            object_root=object_root,
            path=output_root / f"document-{index:02d}.json",
        )
    plan = derive_native_trajectory_plan(revision=revision, documents=documents)
    trajectory = placement_trajectory_from_native_plan(plan)
    (output_root / "native-plan.json").write_text(json.dumps(plan))
    (output_root / "placement-trajectory.json").write_text(json.dumps(trajectory))
    collision = fetch_bound_reference(
        revision["geometry"]["configured_collision"],
        object_root=object_root,
        path=output_root / "scene-collision.usda",
    )
    robot = output_root / "fixture-robot-body.usda"
    stage = Usd.Stage.CreateNew(str(robot))
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/Robot").GetPrim())
    _box(stage, "/Robot/FixtureBody", (-0.15, -0.15, 0), (0.15, 0.15, 0.75))
    stage.GetRootLayer().Save()
    stage = None
    index = build_robot_placement_geometry_index(
        scene_collision_usd_path=collision, robot_asset_usd_path=robot
    )
    target = trajectory["phases"][0]["position_world_m"]
    summary = summarize_robot_placement_geometry(index, target_position_world_m=target)
    # This closed fixture bounds its inventory to eight real geometry-ranked
    # proposals, then evaluates the entire authored trajectory for each. It
    # does not claim to benchmark an unbounded placement search.
    geometric = enumerate_robot_placement_geometry_candidates(
        index=index,
        target_position_world_m=target,
        maximum_candidates=8,
        geometry_worker_count=1,
        trajectory_worker_count=1,
    )
    candidates = []
    for proposal in geometric:
        gate = validate_robot_placement_geometry_candidate(
            index=index,
            proposal=proposal,
            target_position_world_m=target,
            trajectory_waypoints_world_m=[row["position_world_m"] for row in trajectory["phases"]],
            trajectory_phase_ids=[row["phase_id"] for row in trajectory["phases"]],
            trajectory_orientations_world_xyzw=[
                row["orientation_world_xyzw"] for row in trajectory["phases"]
            ],
        )
        if gate["status"] == "passed":
            candidates.append(
                {
                    **proposal,
                    "geometry_gate_digest": gate["geometry_gate_digest"],
                    "trajectory_position_ik_gate": gate["trajectory_position_ik_gate"],
                    "trajectory_position_ik_gate_digest": gate["trajectory_position_ik_gate"][
                        "trajectory_position_ik_gate_digest"
                    ],
                    "trajectory_minimum_manipulability": gate["trajectory_position_ik_gate"][
                        "minimum_manipulability"
                    ],
                }
            )
    candidates.sort(
        key=lambda row: (-row["trajectory_minimum_manipulability"], row["candidate_id"])
    )
    checkpoint = build_candidate_inventory_checkpoint(
        robot_id="franka_panda",
        target_position_world_m=target,
        maximum_candidates=8,
        trajectory_digest=trajectory["trajectory_digest"],
        geometry_summary_digest=summary["geometry_summary_digest"],
        candidates=candidates,
    )
    scene_binding = {
        "scene_identity": revision["scene_identity"],
        "configured_scene_revision_digest": revision["revision_digest"],
        "robot_mount_interface_digest": revision["registration"]["robot_mount_interface"]["digest"],
        "workspace_clearance_digest": revision["registration"]["workspace_clearance"]["digest"],
    }
    task_binding = {
        "task_identity": revision["task_template"]["identity"],
        "robot_id": "franka_panda",
        "task_definition_digest": revision["task_template"]["definition"]["digest"],
    }
    receipt = run_robot_placement_cli(
        run_id=revision["configuration_run_id"],
        scene_collision_usd=collision,
        robot_asset_usd=robot,
        target_position_world_m=target,
        scene_binding=scene_binding,
        task_binding=task_binding,
        overview_image_paths=[],
        output_dir=output_root / "placement",
        max_rounds=1,
        candidate_inventory_cap=8,
        max_input_tokens=1000,
        max_inference_cost_usd=0,
        allow_live_invocation=False,
        tracing_disabled=True,
        task_trajectory=trajectory,
        candidate_inventory_checkpoint=checkpoint,
        deterministic_selection=True,
        render_geometry_previews=False,
    )
    candidate = materialize_robot_placement_readiness_candidate(
        configured_revision=revision,
        scene_binding=scene_binding,
        task_binding=task_binding,
        placement_receipt=receipt,
        candidate_inventory=checkpoint,
        output_path=output_root / "base-pose-candidate.json",
    )
    return {
        "plan": plan,
        "trajectory": trajectory,
        "inventory": checkpoint,
        "receipt": receipt,
        "candidate": candidate,
        "fixture_robot_geometry": file_record(robot),
        "claim_ceiling": "development_only",
        "actual_provider_calls": 0,
    }


def stage_fixture_episode_preparation(
    *,
    configured: dict,
    placement: dict,
    object_root: Path,
    runtime_binding: dict,
    output_root: Path,
    queue_root: Path,
    source_commit: str,
) -> dict:
    import pwd
    import os
    from tests.test_task_evaluation_franka_robotiq_readiness_inputs import _camera
    from blueprint_pipeline.task_evaluation_configured_controls_progression import (
        stage_configured_controls_episode_preparation,
    )
    from scripts.control_plane_concurrency_provider import fixture_publisher

    documents = output_root.parent / (output_root.name + "-documents")
    documents.mkdir(mode=0o700)
    revision = configured["revision"]
    mount = fetch_bound_reference(
        revision["registration"]["robot_mount_interface"],
        object_root=object_root,
        path=documents / "mount.json",
    )
    calibration = fetch_bound_reference(
        revision["registration"]["camera_calibration"],
        object_root=object_root,
        path=documents / "calibration.json",
    )
    cameras = [_camera("external"), _camera("wrist"), _camera("overview")]
    positions = [row["position_world_m"] for row in placement["trajectory"]["phases"]]
    center = [
        (min(row[axis] for row in positions) + max(row[axis] for row in positions)) / 2
        for axis in (0, 1)
    ]
    height = max(row[2] for row in positions) + 1.0
    for camera in cameras:
        if camera["pose_frame"] == "world":
            camera["frame_from_camera_matrix"] = [
                1,
                0,
                0,
                center[0],
                0,
                -1,
                0,
                center[1],
                0,
                0,
                -1,
                height,
                0,
                0,
                0,
                1,
            ]
    return stage_configured_controls_episode_preparation(
        terminal_result=configured["terminal"],
        publication_result=configured["publication"],
        configured_revision=revision,
        expected_production_commit=source_commit,
        robot_mount_interface_path=mount,
        scene_camera_calibration_path=calibration,
        base_pose_candidate=placement["candidate"],
        cameras=cameras,
        runtime_binding=runtime_binding,
        output_root=output_root,
        publisher=fixture_publisher(object_root),
        queue_root=queue_root,
        submitted_by=pwd.getpwuid(os.geteuid()).pw_name,
    )
