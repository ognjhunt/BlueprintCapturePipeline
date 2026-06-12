from __future__ import annotations

import json
import struct
from pathlib import Path

from blueprint_pipeline.cpu_simulator_preflight import build_cpu_simulator_preflight
from blueprint_pipeline.episode_spec import FakeEpisodeSpecAgentAdapter, build_episode_specs
from blueprint_pipeline.robot_eval_dataset import build_real_site_robot_eval_dataset
from blueprint_pipeline.scene_asset_preflight import build_scene_asset_preflight
from blueprint_pipeline.simulation_automation import build_simulation_automation


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
    assert manifest["robot_readiness_proven"] is False

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
