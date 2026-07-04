from __future__ import annotations

import json
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from blueprint_pipeline import cpu_simulator_preflight as csp
from blueprint_pipeline.cpu_simulator_preflight import build_cpu_simulator_preflight
from blueprint_pipeline.episode_spec import (
    FakeEpisodeSpecAgentAdapter,
    build_episode_specs,
    build_task_anchor_proposals,
)
from blueprint_pipeline.robot_eval_dataset import build_real_site_robot_eval_dataset
from blueprint_pipeline.scene_asset_preflight import build_scene_asset_preflight
from blueprint_pipeline.simulation_automation import build_simulation_automation


pytestmark = pytest.mark.slow


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_capture_root(tmp_path: Path) -> Path:
    capture_root = tmp_path / "local-blueprint" / "scenes" / "scene-1" / "captures" / "capture-1"
    _write_json(
        capture_root / "capture_descriptor.json",
        {
            "scene_id": "scene-1",
            "capture_id": "capture-1",
            "metadata": {"site_identity": {"site_id": "site-1"}, "site_type": "stockroom"},
        },
    )
    _write_json(
        capture_root / "raw" / "manifest.json",
        {
            "scene_id": "scene-1",
            "capture_id": "capture-1",
            "site_identity": {"site_id": "site-1"},
            "site_type": "stockroom",
        },
    )
    return capture_root


def _write_task_anchor(capture_root: Path) -> None:
    _write_json(
        capture_root / "pipeline" / "evaluation_prep" / "task_anchor_manifest.json",
        {
            "schema_version": "task_anchor_manifest.v1",
            "updated_at": "2026-06-06T00:00:00+00:00",
            "tasks": [
                {
                    "task_id": "place_return_in_bin",
                    "task_text": "Place the return item in the labeled bin",
                    "task_category": "pick_place",
                    "start_zone": [0.0, 0.0, 0.0],
                    "goal_zone": [1.0, 0.5, 0.0],
                    "target_object_ids": ["bin_1"],
                }
            ],
        },
    )
    _write_json(
        capture_root / "pipeline" / "evaluation_prep" / "object_geometry_manifest.json",
        {
            "schema_version": "object_geometry_manifest.v1",
            "objects": [
                {
                    "object_id": "bin_1",
                    "label": "returns bin",
                    "placement_bbox": {
                        "center": [1.0, 0.5, 0.25],
                        "extents": [0.4, 0.4, 0.5],
                    },
                }
            ],
        },
    )


def _write_ascii_ply(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 4",
                "property float x",
                "property float y",
                "property float z",
                "end_header",
                "0 0 0",
                "2 0 0",
                "2 2 0.2",
                "0 2 0.1",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_minimal_glb_with_accessor_bounds(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "asset": {"version": "2.0"},
        "scenes": [{"nodes": [0]}],
        "nodes": [{"name": "worldlabs_collider", "mesh": 0}],
        "meshes": [
            {
                "name": "scene_collider",
                "primitives": [{"attributes": {"POSITION": 0}}],
            }
        ],
        "accessors": [
            {
                "type": "VEC3",
                "componentType": 5126,
                "count": 8,
                "min": [-1.0, -2.0, 0.0],
                "max": [3.0, 4.0, 1.5],
            }
        ],
    }
    raw_json = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    raw_json += b" " * ((4 - (len(raw_json) % 4)) % 4)
    total_length = 12 + 8 + len(raw_json)
    path.write_bytes(
        b"glTF"
        + struct.pack("<II", 2, total_length)
        + struct.pack("<II", len(raw_json), 0x4E4F534A)
        + raw_json
    )


def test_task_anchor_proposals_prefer_capture_manifest_task_intent(tmp_path: Path) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_json(
        capture_root / "raw" / "manifest.json",
        {
            "scene_id": "scene-1",
            "capture_id": "capture-1",
            "workflowName": "First GPU humanoid navigation smoke",
            "taskSteps": [
                "load captured scene",
                "spawn humanoid at valid start pose",
                "navigate to selected waypoint",
            ],
            "zone": "sample-zone",
        },
    )
    automation_dir = capture_root / "pipeline" / "simulation_automation"
    _write_json(
        automation_dir / "scene_asset_inspection.json",
        {
            "schema_version": "scene_asset_inspection.v1",
            "assets": [
                {
                    "semantic_hints": [
                        {"label": "world", "source": "glb_node_or_mesh_name"},
                        {"label": "geometry_0", "source": "glb_node_or_mesh_name"},
                    ]
                }
            ],
        },
    )

    manifest = build_task_anchor_proposals(
        capture_root=capture_root,
        pipeline_dir=capture_root / "pipeline",
        automation_dir=automation_dir,
        generated_at="2026-06-18T00:00:00+00:00",
    )

    assert manifest["proposals"][0]["task_id"] == (
        "capture_intent_First_GPU_humanoid_navigation_smoke"
    )
    assert manifest["proposals"][0]["task_text"] == (
        "Navigate humanoid from validated start zone to selected waypoint"
    )
    assert manifest["proposals"][0]["target_object_ids"] == ["selected_waypoint"]
    assert manifest["proposals"][0]["source"] == "raw/manifest.json"
    proposal_ids = [proposal["task_id"] for proposal in manifest["proposals"]]
    assert "scene_anchor_world" in proposal_ids[1:]
    assert "scene_anchor_geometry_0" in proposal_ids[1:]


def test_scene_asset_preflight_inspects_ply_bounds_without_collision_claim(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    ply_path = tmp_path / "fixtures" / "scene.ply"
    _write_ascii_ply(ply_path)

    result = build_scene_asset_preflight(capture_root=capture_root, scene_assets=[ply_path])

    automation_dir = capture_root / "pipeline" / "simulation_automation"
    inspection = _read_json(automation_dir / "scene_asset_inspection.json")
    inventory = _read_json(automation_dir / "scene_asset_inventory.json")
    dependency_audit = _read_json(automation_dir / "scene_asset_dependency_audit.json")
    frame = _read_json(automation_dir / "scene_frame_estimate.json")
    scorecard = _read_json(automation_dir / "cpu_preflight_scorecard.json")
    collider_proxy = _read_json(automation_dir / "collider_proxy_plan.json")
    cpu_proxy = _read_json(automation_dir / "cpu_scene_proxy_manifest.json")
    scene_preflight = _read_json(automation_dir / "scene_asset_preflight.json")

    assert result["status"] == "ready_for_episode_setup"
    assert inventory["assets"][0]["asset_type"] == "ply"
    assert dependency_audit["dependency_count"] == 0
    assert inspection["assets"][0]["vertex_count"] == 4
    assert inspection["assets"][0]["bounds"]["min"] == [0.0, 0.0, 0.0]
    assert frame["frame"]["centroid"] == [1.0, 1.0, 0.07500000000000001]
    assert collider_proxy["real_collider_proven"] is False
    assert collider_proxy["proxy_estimated"] is True
    assert collider_proxy["missing_collider"] is True
    assert "proxy_estimated" in collider_proxy["labels"]
    assert cpu_proxy["status"] == "ready_for_cpu_spawn_checks"
    assert scene_preflight["collider_summary"]["review_required"] is True
    assert scorecard["cpu_proxy_collision_estimated"] is True
    assert scorecard["portable_collider_glb_missing"] is True
    assert scorecard["real_collider_proven"] is False
    assert scorecard["proxy_estimated"] is True
    assert scorecard["simulator_execution_not_run"] is True
    assert scorecard["proof_booleans"]["simulator_execution_proven"] is False


def test_scene_asset_preflight_inspects_materialized_glb_collider_bounds(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    glb_path = capture_root / "pipeline" / "worldlabs_assets" / "worldlabs_collider.glb"
    _write_minimal_glb_with_accessor_bounds(glb_path)

    result = build_scene_asset_preflight(capture_root=capture_root)

    automation_dir = capture_root / "pipeline" / "simulation_automation"
    inventory = _read_json(automation_dir / "scene_asset_inventory.json")
    frame = _read_json(automation_dir / "scene_frame_estimate.json")
    scorecard = _read_json(automation_dir / "cpu_preflight_scorecard.json")
    collider_proxy = _read_json(automation_dir / "collider_proxy_plan.json")

    assert result["status"] == "ready_for_episode_setup"
    assert inventory["assets"][0]["asset_type"] == "glb"
    assert inventory["assets"][0]["bounds_present"] is True
    assert frame["status"] == "complete"
    assert frame["frame"]["bounds"]["min"] == [-1.0, -2.0, 0.0]
    assert frame["frame"]["bounds"]["max"] == [3.0, 4.0, 1.5]
    assert frame["frame"]["estimate_method"] == "gltf_position_accessor_min_max"
    assert scorecard["portable_collider_glb_present"] is True
    assert scorecard["portable_collider_glb_missing"] is False
    assert scorecard["real_collider_proven"] is True
    assert scorecard["cpu_proxy_collision_estimated"] is True
    assert scorecard["proof_booleans"]["physics_contact_validated"] is False
    assert collider_proxy["status"] == "real_collider_metadata_present"


def test_simulation_automation_rerun_does_not_promote_generated_fixtures_to_source_assets(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    ply_path = tmp_path / "fixtures" / "scene.ply"
    _write_ascii_ply(ply_path)

    build_simulation_automation(capture_root=capture_root, scene_assets=[ply_path])
    build_simulation_automation(capture_root=capture_root, scene_assets=[ply_path])

    automation_dir = capture_root / "pipeline" / "simulation_automation"
    inventory = _read_json(automation_dir / "scene_asset_inventory.json")
    asset_types = [asset["asset_type"] for asset in inventory["assets"]]

    assert asset_types == ["ply"]
    assert inventory["assets"][0]["path"] == str(ply_path.resolve())


def test_usd_dependency_audit_records_missing_and_remote_references(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    usd_path = tmp_path / "fixtures" / "scene-with-refs.usda"
    usd_path.parent.mkdir(parents=True, exist_ok=True)
    usd_path.write_text(
        "\n".join(
            [
                "#usda 1.0",
                "(",
                "    metersPerUnit = 1",
                '    upAxis = "Z"',
                "    subLayers = [@missing_layer.usda@]",
                ")",
                'def Xform "Kitchen"',
                "{",
                '    prepend references = @props/missing_counter.usda@',
                '    custom asset diffuseTexture = @https://assets.example/remote.png@',
                "}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    build_scene_asset_preflight(capture_root=capture_root, scene_assets=[usd_path])
    automation_dir = capture_root / "pipeline" / "simulation_automation"
    audit = _read_json(automation_dir / "scene_asset_dependency_audit.json")
    inventory = _read_json(automation_dir / "scene_asset_inventory.json")
    scorecard = _read_json(automation_dir / "cpu_preflight_scorecard.json")

    assert inventory["assets"][0]["asset_type"] == "usd"
    assert audit["missing_local_file_count"] == 2
    assert audit["remote_ref_count"] == 1
    assert audit["unresolved_ref_count"] == 3
    assert {item["relationship"] for item in audit["dependencies"]} >= {
        "sublayer",
        "reference",
        "texture_or_material_asset",
    }
    assert "missing_scene_asset_dependencies" in scorecard["blockers"]


def test_usd_owner_system_material_library_is_warning_not_hard_dependency(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    usd_path = tmp_path / "fixtures" / "scene-with-owner-material.usda"
    usd_path.parent.mkdir(parents=True, exist_ok=True)
    usd_path.write_text(
        "\n".join(
            [
                "#usda 1.0",
                "(",
                "    metersPerUnit = 1",
                '    upAxis = "Z"',
                ")",
                'def Xform "LightwheelAsset"',
                "{",
                '    custom asset materialLibrary = @OmniPBR.mdl@',
                "}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    build_simulation_automation(capture_root=capture_root, scene_assets=[usd_path])
    automation_dir = capture_root / "pipeline" / "simulation_automation"
    audit = _read_json(automation_dir / "scene_asset_dependency_audit.json")
    scorecard = _read_json(automation_dir / "cpu_preflight_scorecard.json")
    cpu_manifest = _read_json(automation_dir / "cpu_preflight_manifest.json")
    gpu_handoff = _read_json(automation_dir / "gpu_handoff_packet.json")

    assert audit["missing_local_file_count"] == 1
    assert audit["hard_missing_local_file_count"] == 0
    assert audit["owner_system_material_warning_count"] == 1
    assert audit["status"] == "complete"
    assert "missing_scene_asset_dependencies" not in scorecard["blockers"]
    assert "missing_scene_asset_dependencies" not in cpu_manifest["hard_preflight_blockers"]
    assert "missing_scene_asset_dependencies" not in gpu_handoff["blockers"]


def test_binary_usd_requires_openusd_without_crashing(tmp_path: Path) -> None:
    capture_root = _build_capture_root(tmp_path)
    usd_path = tmp_path / "fixtures" / "binary-scene.usd"
    usd_path.parent.mkdir(parents=True, exist_ok=True)
    usd_path.write_bytes(b"PXR-USDC\x00\x08\x00\x00\x00binary-usd-placeholder")

    result = build_scene_asset_preflight(capture_root=capture_root, scene_assets=[usd_path])

    automation_dir = capture_root / "pipeline" / "simulation_automation"
    inspection = _read_json(automation_dir / "scene_asset_inspection.json")
    audit = _read_json(automation_dir / "scene_asset_dependency_audit.json")
    scorecard = _read_json(automation_dir / "cpu_preflight_scorecard.json")

    assert result["status"] == "ready_for_episode_setup"
    assert inspection["assets"][0]["status"] == "openusd_required_for_binary_usd"
    assert audit["dependency_count"] == 0
    assert scorecard["isaac_usd_import_candidate"] is True
    assert scorecard["isaac_usd_collision_unverified"] is True


def test_urdf_collision_metadata_sets_real_collider_without_readiness_claim(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    urdf_path = tmp_path / "fixtures" / "robot_scene.urdf"
    mesh_path = tmp_path / "fixtures" / "collision.obj"
    mesh_path.parent.mkdir(parents=True, exist_ok=True)
    mesh_path.write_text(
        "o collider_box\nv 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n",
        encoding="utf-8",
    )
    urdf_path.write_text(
        "\n".join(
            [
                '<robot name="collision_scene">',
                '  <link name="floor">',
                "    <collision>",
                "      <geometry>",
                '        <mesh filename="collision.obj" />',
                "      </geometry>",
                "    </collision>",
                "  </link>",
                "</robot>",
            ]
        ),
        encoding="utf-8",
    )

    build_scene_asset_preflight(capture_root=capture_root, scene_assets=[urdf_path])
    automation_dir = capture_root / "pipeline" / "simulation_automation"
    inventory = _read_json(automation_dir / "scene_asset_inventory.json")
    collider_proxy = _read_json(automation_dir / "collider_proxy_plan.json")
    cpu_proxy = _read_json(automation_dir / "cpu_scene_proxy_manifest.json")
    scorecard = _read_json(automation_dir / "cpu_preflight_scorecard.json")

    assert {asset["asset_type"] for asset in inventory["assets"]} == {"obj", "urdf"}
    assert any(
        asset["discovery_source"] == "local_dependency_reference"
        for asset in inventory["assets"]
        if asset["asset_type"] == "obj"
    )
    assert collider_proxy["real_collider_proven"] is True
    assert collider_proxy["proxy_estimated"] is True
    assert collider_proxy["missing_collider"] is False
    assert cpu_proxy["status"] == "ready_for_cpu_spawn_checks"
    assert scorecard["real_collider_proven"] is True
    assert scorecard["status"] == "ready_for_episode_setup"
    assert scorecard["proof_booleans"]["physics_contact_validated"] is False


def test_episode_spec_and_cpu_preflight_emit_review_required_optional_dependency_blockers(
    tmp_path: Path,
    monkeypatch,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_task_anchor(capture_root)
    ply_path = tmp_path / "fixtures" / "scene.ply"
    _write_ascii_ply(ply_path)
    build_scene_asset_preflight(capture_root=capture_root, scene_assets=[ply_path])

    episode_result = build_episode_specs(
        capture_root=capture_root,
        agent_adapter=FakeEpisodeSpecAgentAdapter(),
    )
    automation_dir = capture_root / "pipeline" / "simulation_automation"
    episode_spec = _read_json(automation_dir / "episode_spec.v1.json")
    proposals = _read_json(automation_dir / "agent_episode_spec_proposals.json")

    assert episode_result["episode_count"] == 3
    assert episode_spec["default_robot_profiles_used"] is True
    assert episode_spec["episodes"][0]["review_required"] is True
    assert "simulator_execution_not_run" in episode_spec["episodes"][0]["missing_proof_labels"]
    assert proposals["agent_authority"] == "review_input_proposal_operator"
    assert proposals["proof_booleans_mutable_by_agent"] is False
    assert proposals["proposal_count"] == 1

    monkeypatch.setenv("BLUEPRINT_ALLOW_CPU_SIMULATOR_PREFLIGHT", "true")
    monkeypatch.setattr(
        "blueprint_pipeline.cpu_simulator_preflight.importlib.util.find_spec",
        lambda _name: None,
    )
    cpu_result = build_cpu_simulator_preflight(
        capture_root=capture_root,
        allow_cpu_simulator_preflight=True,
        backends=["mujoco", "pybullet"],
    )
    setup = _read_json(automation_dir / "episode_setup_manifest.json")
    manifest = _read_json(automation_dir / "cpu_simulator_preflight_manifest.json")
    spawn_validation = _read_json(automation_dir / "spawn_pose_validation_manifest.json")
    cpu_manifest = _read_json(automation_dir / "cpu_preflight_manifest.json")
    readiness = _read_json(automation_dir / "pre_gpu_readiness_summary.json")

    assert cpu_result["status"] == "ready_blocked_optional_dependencies_or_gates"
    assert setup["status"] == "ready_for_optional_cpu_smoke"
    assert spawn_validation["episode_count"] == 3
    assert spawn_validation["validations"][0]["candidate_count"] >= 3
    assert cpu_manifest["ready_for_owner_gpu_preflight"] is True
    assert readiness["remaining_unproven_step"] == "actual_owner_system_gpu_simulator_execution"
    assert readiness["ready_for_robot_evaluation"] is False
    assert Path(automation_dir / setup["generated_fixtures"]["mujoco_mjcf"]).is_file()
    assert Path(automation_dir / setup["generated_fixtures"]["pybullet_urdf"]).is_file()
    assert manifest["backend_results"]["mujoco"]["reason"] == "missing_optional_dependency"
    assert manifest["backend_results"]["pybullet"]["install_and_run"]["install"] == (
        "python -m pip install pybullet"
    )
    assert manifest["simulator_execution_proven"] is False
    assert manifest["rank_fidelity_result_proven"] is False

    build_simulation_automation(
        capture_root=capture_root,
        scene_assets=[ply_path],
        allow_cpu_simulator_preflight=True,
        cpu_preflight_backends=["mujoco", "pybullet"],
    )
    gpu_handoff = _read_json(automation_dir / "gpu_handoff_packet.json")
    assert gpu_handoff["hard_preflight_blockers"] == []
    assert "spawn_outside_scene_bounds" not in {
        detail["blocker_id"] for detail in gpu_handoff["pre_gpu_blocker_details"]
    }


def test_usd_collision_labels_are_split_for_dataset_and_proof_boundaries(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_task_anchor(capture_root)
    usd_path = tmp_path / "fixtures" / "scene.usda"
    usd_path.parent.mkdir(parents=True, exist_ok=True)
    usd_path.write_text(
        "\n".join(
            [
                "#usda 1.0",
                "(",
                "    metersPerUnit = 1",
                '    upAxis = "Z"',
                ")",
                'def Xform "Kitchen"',
                "{",
                '    def Mesh "counter_visual"',
                "    {",
                "    }",
                "}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    build_scene_asset_preflight(capture_root=capture_root, scene_assets=[usd_path])

    build_real_site_robot_eval_dataset(capture_root=capture_root)
    robot_eval_dir = capture_root / "pipeline" / "robot_eval_dataset"
    site_card = _read_json(robot_eval_dir / "site_card.json")
    proof_boundaries = _read_json(robot_eval_dir / "proof_boundaries.json")
    backlog = _read_json(robot_eval_dir / "annotation_backlog.json")
    collider = site_card["geometry"]["collider"]

    assert collider["isaac_usd_import_candidate"] is True
    assert collider["isaac_usd_collision_unverified"] is True
    assert collider["portable_collider_glb_missing"] is True
    assert collider["simulator_execution_not_run"] is True
    assert "isaac_usd_collision_unverified" in collider["backend_blockers"]
    assert "portable_collider_glb_missing" in proof_boundaries["collider_backend_blockers"]
    assert any(
        item["backlog_id"] == "isaac_usd_collision_unverified"
        for item in backlog["items"]
    )


def test_simulation_automation_conversion_plan_splits_usd_and_portable_collider_blockers(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    usd_path = tmp_path / "fixtures" / "scene.usda"
    usd_path.parent.mkdir(parents=True, exist_ok=True)
    usd_path.write_text(
        "#usda 1.0\n(\n    metersPerUnit = 1\n    upAxis = \"Z\"\n)\ndef Xform \"Scene\" {}\n",
        encoding="utf-8",
    )

    build_simulation_automation(capture_root=capture_root, scene_assets=[usd_path])
    automation_dir = capture_root / "pipeline" / "simulation_automation"
    conversion = _read_json(automation_dir / "asset_conversion_plan.json")
    isaac = conversion["frameworks"]["isaac_sim"]
    mujoco = conversion["frameworks"]["mujoco"]
    pybullet = conversion["frameworks"]["pybullet"]

    assert isaac["status"] == "isaac_usd_import_candidate"
    assert isaac["blockers"] == ["isaac_usd_collision_unverified"]
    assert mujoco["blockers"] == ["portable_collider_glb_missing"]
    assert pybullet["blockers"] == ["portable_collider_glb_missing"]


def test_cpu_preflight_helper_edges(tmp_path: Path) -> None:
    assert csp._string_list(None) == []
    assert csp._string_list("one") == ["one"]
    assert csp._string_list(123) == ["123"]
    assert csp._read_optional_mapping(tmp_path / "missing.json") == {}
    assert csp._float_list(["bad"], fallback=(1.0, 2.0, 3.0)) == [1.0, 2.0, 3.0]
    assert csp._finite_xyz(["bad", 0, 0]) is None
    assert csp._finite_xyz([float("nan"), 0, 0]) is None

    automation_dir = tmp_path / "automation"
    _write_json(
        automation_dir / "scene_frame_estimate.json",
        {"frame": {"bounds": {"min": [0, 0, 2], "max": [1, 1, 3]}, "floor_z_estimate": "bad"}},
    )
    assert csp._frame_bounds(automation_dir)["floor_z"] == 2.0

    frame = {"bounds": {"min": [0, 0, 0], "max": [1, 1, 1]}, "floor_z": 0.0}
    duplicate_candidates = csp._candidate_spawn_poses(
        {"robot_spawn_pose": {"xyz": [0.5, 0.5, 0.05], "rpy": []}},
        frame,
    )
    assert [item["candidate_id"] for item in duplicate_candidates].count("frame_center_floor") == 0
    invalid_candidates = csp._candidate_spawn_poses(
        {"robot_spawn_pose": {"xyz": [float("nan"), 0, 0], "rpy": []}},
        {},
    )
    assert invalid_candidates[0]["xyz"][0] != invalid_candidates[0]["xyz"][0]


def test_cpu_preflight_spawn_validation_edges() -> None:
    invalid = csp._validate_spawn_candidate(
        {"candidate_id": "bad", "xyz": ["bad", 0, 0]},
        frame={},
        proxy_manifest={},
    )
    assert "spawn_pose_not_finite" in invalid["blockers"]
    assert "scene_bounds_missing_or_invalid" in invalid["blockers"]

    frame = {"bounds": {"min": [0, 0, 0], "max": [0.01, 2000, 0.01]}, "floor_z": 1.0}
    result = csp._validate_spawn_candidate(
        {"candidate_id": "edge", "xyz": [10, 10, 4], "rpy": ["bad"]},
        frame=frame,
        proxy_manifest={
            "proxy_obstacles": [
                {"obstacle_id": "box", "min_xyz": [9, 9, 3], "max_xyz": [11, 11, 5]}
            ]
        },
    )
    assert "scene_scale_suspiciously_small" in result["warnings"]
    assert "scene_scale_suspiciously_large" in result["warnings"]
    assert "spawn_outside_scene_bounds" in result["blockers"]
    assert "spawn_height_far_above_floor_estimate" in result["warnings"]
    assert "spawn_inside_known_or_proxy_geometry" in result["blockers"]

    below = csp._validate_spawn_candidate(
        {"candidate_id": "below", "xyz": [0, 0, 0]},
        frame={"bounds": {"min": [-1, -1, -1], "max": [1, 1, 2]}, "floor_z": 1.0},
        proxy_manifest={},
    )
    assert "spawn_below_floor_estimate" in below["blockers"]

    inverted = csp._validate_spawn_candidate(
        {"candidate_id": "inverted", "xyz": [0, 0, 0]},
        frame={"bounds": {"min": [1, 1, 1], "max": [0, 0, 0]}, "floor_z": 0.0},
        proxy_manifest={},
    )
    assert "scene_bounds_empty_or_inverted" in inverted["blockers"]


def test_cpu_preflight_optional_backend_smokes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pybullet_calls: list[str] = []
    fake_pybullet = SimpleNamespace(
        DIRECT=0,
        ER_TINY_RENDERER=1,
        connect=lambda _mode: 7,
        resetSimulation=lambda **_kwargs: pybullet_calls.append("reset"),
        setGravity=lambda *_args, **_kwargs: pybullet_calls.append("gravity"),
        loadURDF=lambda *_args, **_kwargs: 42,
        stepSimulation=lambda **_kwargs: pybullet_calls.append("step"),
        getCameraImage=lambda *_args, **_kwargs: (32, 32, None),
        disconnect=lambda **_kwargs: pybullet_calls.append("disconnect"),
    )
    monkeypatch.setitem(sys.modules, "pybullet", fake_pybullet)
    pybullet = csp._run_pybullet_smoke(
        urdf_path=tmp_path / "scene.urdf",
        steps=0,
        allow_render=True,
        generated_at="2026-06-20T00:00:00+00:00",
    )
    assert pybullet["status"] == "completed_local_cpu_smoke"
    assert pybullet["render_result_shape"] == [32, 32]
    assert "disconnect" in pybullet_calls

    class FakeModel:
        @staticmethod
        def from_xml_path(_path: str) -> str:
            return "model"

    class FakeData:
        def __init__(self, model: str) -> None:
            self.model = model

    steps: list[str] = []
    fake_mujoco = SimpleNamespace(
        MjModel=FakeModel,
        MjData=FakeData,
        mj_step=lambda _model, _data: steps.append("step"),
    )
    monkeypatch.setitem(sys.modules, "mujoco", fake_mujoco)
    mujoco = csp._run_mujoco_smoke(
        mjcf_path=tmp_path / "scene.xml",
        steps=2,
        generated_at="2026-06-20T00:00:00+00:00",
    )
    assert mujoco["status"] == "completed_local_cpu_smoke"
    assert steps == ["step", "step"]

    unsupported = csp._backend_result(
        backend="unknown",
        automation_dir=tmp_path,
        allow_cpu_simulator_preflight=True,
        env_allowed=True,
        steps=1,
        allow_render=False,
        generated_at="2026-06-20T00:00:00+00:00",
    )
    assert unsupported["reason"] == "unsupported_cpu_backend"
    monkeypatch.setattr(csp.importlib.util, "find_spec", lambda _name: object())
    delegated = csp._backend_result(
        backend="pybullet",
        automation_dir=tmp_path,
        allow_cpu_simulator_preflight=True,
        env_allowed=True,
        steps=1,
        allow_render=False,
        generated_at="2026-06-20T00:00:00+00:00",
    )
    assert delegated["status"] == "completed_local_cpu_smoke"
    delegated_mujoco = csp._backend_result(
        backend="mujoco",
        automation_dir=tmp_path,
        allow_cpu_simulator_preflight=True,
        env_allowed=True,
        steps=1,
        allow_render=False,
        generated_at="2026-06-20T00:00:00+00:00",
    )
    assert delegated_mujoco["status"] == "completed_local_cpu_smoke"


def test_cpu_preflight_builds_episode_specs_when_missing_and_defaults_backends(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    automation_dir = capture_root / "pipeline" / "simulation_automation"

    def fake_build_episode_specs(*, capture_root: Path) -> dict[str, object]:
        _write_json(
            Path(capture_root) / "pipeline" / "simulation_automation" / "episode_spec.v1.json",
            {
                "episodes": [
                    {
                        "episode_id": "episode-1",
                        "task_id": "task-1",
                        "scenario_id": "scenario-1",
                        "robot_spawn_pose": {"xyz": [0.0, 0.0, 0.25]},
                        "missing_proof_labels": ["simulator_execution_not_run"],
                    }
                ]
            },
        )
        return {"episode_count": 1}

    monkeypatch.setattr(csp, "build_episode_specs", fake_build_episode_specs)
    monkeypatch.setattr(csp.importlib.util, "find_spec", lambda _name: None)

    result = build_cpu_simulator_preflight(
        capture_root=capture_root,
        allow_cpu_simulator_preflight=False,
        backends=["unsupported"],
    )
    manifest = _read_json(automation_dir / "cpu_simulator_preflight_manifest.json")

    assert result["status"] == "ready_blocked_optional_dependencies_or_gates"
    assert manifest["selected_backends"] == ["mujoco", "pybullet"]
    assert (automation_dir / "episode_spec.v1.json").is_file()


def test_cpu_preflight_main_success_and_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    capture_root = _build_capture_root(tmp_path)
    monkeypatch.setattr(
        csp,
        "build_cpu_simulator_preflight",
        lambda **_: {
            "cpu_simulator_preflight_manifest_path": str(tmp_path / "manifest.json"),
            "status": "ready_blocked_optional_dependencies_or_gates",
        },
    )

    assert csp.main(
        [
            "--capture-root",
            str(capture_root),
            "--allow-cpu-simulator-preflight",
            "--backend",
            "mujoco",
            "--smoke-steps",
            "1",
            "--allow-render",
        ]
    ) == 0
    assert "status=ready_blocked_optional_dependencies_or_gates" in capsys.readouterr().out

    def raise_value_error(**_kwargs: object) -> dict[str, object]:
        raise ValueError("bad capture")

    monkeypatch.setattr(csp, "build_cpu_simulator_preflight", raise_value_error)
    assert csp.main(["--capture-root", str(capture_root)]) == 1
    assert "[cpu-simulator-preflight] FAILED: bad capture" in capsys.readouterr().out
