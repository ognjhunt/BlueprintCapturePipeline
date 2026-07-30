from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from blueprint_pipeline.nvidia_warehouse_native_camera_canary import (
    _apply_runtime_asset_relocations,
    _camera_quaternion_wxyz,
    _load_materialization_manifest,
    _project_world_points,
    _simulation_app_launch_config,
    import_simulation_app,
    run_native_camera_canary,
)
from blueprint_pipeline.nvidia_warehouse_workcell import CANARY_SPEC_SCHEMA_VERSION
from blueprint_pipeline.policy_ranking_thesis import canonical_sha256


def _spec(path: Path) -> None:
    value = {
        "schema_version": CANARY_SPEC_SCHEMA_VERSION,
        "cameras": {
            "external": {"resolution": [640, 480]},
            "wrist": {"resolution": [640, 480]},
        },
    }
    value["spec_sha256"] = canonical_sha256(value)
    path.write_text(json.dumps(value), encoding="utf-8")


def _backend(*, output_dir: Path, **_kwargs):
    output_dir.mkdir(parents=True)
    views = {}
    for view_id in ("external", "wrist"):
        paths = {}
        for phase, color in (("initial", (30, 60, 90)), ("commanded", (90, 60, 30))):
            path = output_dir / f"{view_id}_{phase}.png"
            image = Image.new("RGB", (640, 480), color=color)
            for x in range(0, 640, 20):
                image.putpixel((x, x % 480), (255, 255, 255))
            image.save(path)
            paths[f"{phase}_frame_path"] = str(path)
        views[view_id] = {
            **paths,
            "required_entities_projected_in_frame": {
                "franka": True,
                "spraycan": True,
                "tray": True,
            },
        }
    return {
        "isaac_sim_major_version": 6,
        "scene_loaded": True,
        "missing_dataset_local_dependencies": [],
        "franka_dof_count": 9,
        "spraycan_collision_mesh_count": 3,
        "spraycan_runtime_rigid_body": True,
        "views": views,
        "wrist_camera_world_displacement_m": 0.02,
        "wrist_camera_local_transform_delta": 0.0,
        "external_wrist_timestamp_pairs_exact": True,
    }


def test_usd_camera_convention_projects_negative_z_forward_and_builds_identity_pose() -> None:
    quaternion = _camera_quaternion_wxyz((0.0, 0.0, -1.0), (0.0, 1.0, 0.0))
    assert quaternion == pytest.approx([1.0, 0.0, 0.0, 0.0])
    projected = _project_world_points(
        camera_to_world=np.eye(4),
        points={"center": [0.0, 0.0, -1.0], "behind": [0.0, 0.0, 1.0]},
        width=640,
        height=480,
        vfov_deg=60.0,
    )
    assert projected == {"center": True, "behind": False}


def test_simulation_app_import_falls_back_when_isaacsim_shim_is_not_callable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    isaacsim = types.ModuleType("isaacsim")
    isaacsim.SimulationApp = None
    omni = types.ModuleType("omni")
    omni.__path__ = []
    omni_isaac = types.ModuleType("omni.isaac")
    omni_isaac.__path__ = []
    omni_kit = types.ModuleType("omni.isaac.kit")

    class LegacySimulationApp:
        pass

    omni_kit.SimulationApp = LegacySimulationApp
    monkeypatch.setitem(sys.modules, "isaacsim", isaacsim)
    monkeypatch.setitem(sys.modules, "omni", omni)
    monkeypatch.setitem(sys.modules, "omni.isaac", omni_isaac)
    monkeypatch.setitem(sys.modules, "omni.isaac.kit", omni_kit)

    assert import_simulation_app() is LegacySimulationApp


def test_simulation_app_launch_config_disables_process_terminating_fast_shutdown() -> None:
    first = _simulation_app_launch_config()
    second = _simulation_app_launch_config()

    assert first["fast_shutdown"] is False
    assert first["headless"] is True
    assert first is not second


def test_runtime_asset_relocations_require_exact_local_binding(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    owner = assets / "Props" / "clock" / "clock.usd"
    replacement = assets / "Props" / "clock" / "Textures" / "albedo.png"
    owner.parent.mkdir(parents=True)
    replacement.parent.mkdir(parents=True)
    owner.write_bytes(b"usd")
    replacement.write_bytes(b"png")
    observed = []

    result = _apply_runtime_asset_relocations(
        assets_root=assets,
        manifest={
            "runtime_asset_relocations": [
                {
                    "owner_relative_path": "Props/clock/clock.usd",
                    "source_asset_uri": "omniverse://art/clock/Textures/albedo.png",
                    "replacement_relative_path": "Props/clock/Textures/albedo.png",
                    "replacement_authored_path": "./Textures/albedo.png",
                }
            ]
        },
        layer_relocator=lambda path, source, replacement_path: (
            observed.append((path, source, replacement_path)) or 1
        ),
    )

    assert result["relocation_count"] == 1
    assert result["authored_replacement_count"] == 1
    assert observed == [
        (
            owner,
            "omniverse://art/clock/Textures/albedo.png",
            "./Textures/albedo.png",
        )
    ]


def test_materialization_manifest_resolves_from_extracted_bundle_layout(
    tmp_path: Path,
) -> None:
    extracted = tmp_path / "input"
    assets = extracted / "assets"
    assets.mkdir(parents=True)
    manifest_path = extracted / "materialization_manifest.json"
    manifest_path.write_text(
        json.dumps({"runtime_asset_relocations": []}), encoding="utf-8"
    )

    resolved_path, manifest = _load_materialization_manifest(assets)

    assert resolved_path == manifest_path
    assert manifest == {"runtime_asset_relocations": []}


def test_materialization_manifest_prefers_direct_materialization_root(
    tmp_path: Path,
) -> None:
    assets = tmp_path / "assets"
    assets.mkdir()
    direct = assets / "materialization_manifest.json"
    direct.write_text(
        json.dumps({"runtime_asset_relocations": [{"source": "direct"}]}),
        encoding="utf-8",
    )
    (tmp_path / "materialization_manifest.json").write_text(
        json.dumps({"runtime_asset_relocations": [{"source": "parent"}]}),
        encoding="utf-8",
    )

    resolved_path, manifest = _load_materialization_manifest(assets)

    assert resolved_path == direct
    assert manifest["runtime_asset_relocations"] == [{"source": "direct"}]


def test_native_camera_canary_requires_scene_robot_rigid_object_and_two_synced_views(
    tmp_path: Path,
) -> None:
    spec_path = tmp_path / "spec.json"
    _spec(spec_path)
    result = run_native_camera_canary(
        spec_path=spec_path,
        assets_root=tmp_path / "assets",
        output_dir=tmp_path / "result",
        backend=_backend,
    )

    assert result["status"] == "passed"
    assert result["blockers"] == []
    assert result["assessment"]["views"]["external"]["frames"]["initial"][
        "relative_path"
    ] == "runtime/external_initial.png"
    assert result["paid_policy_or_wam_model_invoked"] is False
    assert result["claim_boundary"]["policy_wam_loop_proven"] is False


def test_native_camera_canary_fails_static_or_slipping_wrist_mount(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.json"
    _spec(spec_path)

    def broken_backend(**kwargs):
        value = _backend(**kwargs)
        value["wrist_camera_world_displacement_m"] = 0.0
        value["wrist_camera_local_transform_delta"] = 0.01
        return value

    result = run_native_camera_canary(
        spec_path=spec_path,
        assets_root=tmp_path / "assets",
        output_dir=tmp_path / "result",
        backend=broken_backend,
    )

    assert result["status"] == "failed"
    assert "native_wrist_camera_did_not_move_with_hand" in result["blockers"]
    assert "native_wrist_camera_mount_not_rigid" in result["blockers"]


def test_native_camera_canary_fails_closed_on_missing_wrist_measurements(
    tmp_path: Path,
) -> None:
    spec_path = tmp_path / "spec.json"
    _spec(spec_path)

    def incomplete_backend(**kwargs):
        value = _backend(**kwargs)
        value.pop("wrist_camera_world_displacement_m")
        value["wrist_camera_local_transform_delta"] = float("nan")
        return value

    result = run_native_camera_canary(
        spec_path=spec_path,
        assets_root=tmp_path / "assets",
        output_dir=tmp_path / "result",
        backend=incomplete_backend,
    )

    assert result["status"] == "failed"
    assert result["assessment"]["wrist_camera_world_displacement_m"] is None
    assert result["assessment"]["wrist_camera_local_transform_delta"] is None
    assert result["backend_result"]["wrist_camera_local_transform_delta"] is None
    assert "native_wrist_camera_world_displacement_missing_or_invalid" in result["blockers"]
    assert "native_wrist_camera_local_transform_missing_or_invalid" in result["blockers"]
