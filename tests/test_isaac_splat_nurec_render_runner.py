"""Hermetic contract tests for the GPU-only reconstruction qualification runner."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
import urllib.request

import jsonschema
import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest


REPO = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO / "scripts" / "run_isaac_splat_nurec_render.py"
SPEC = importlib.util.spec_from_file_location("isaac_splat_nurec_render_runner", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)

DIGEST = "sha256:" + "a" * 64
IMAGE = "registry.example/blueprint/isaac@sha256:" + "b" * 64
SCHEMA_PATH = REPO / "docs/schemas/isaac_splat_nurec_render_result.v3.schema.json"


def _stage(**updates):
    value = {
        "meters_per_unit": 1.0,
        "up_axis": "Z",
        "transforms_valid": True,
        "dependency_inspection_available": True,
        "missing_asset_count": 0,
        "particlefield_prim_count": 1,
        "active_collision_prim_count": 1,
        "obvious_scale_mismatch_detected": False,
        "expected_prim_paths": {
            "appearance": "/World/BlueprintReconstruction/Appearance",
            "collision": "/World/BlueprintReconstruction/Collision",
        },
    }
    value.update(updates)
    return value


def _physics(**updates):
    value = {
        "ground_contact_surface_present": True,
        "steps_executed": 240,
        "live_rigid_body_pose_observed": True,
        "test_body_fell_through_floor": False,
        "contact_event_count": 2,
        "probe_configuration": {
            "test_body": {
                "shape": "cube",
                "size_m": 0.1,
                "mass_kg": 1.0,
                "spawn_height_above_ground_m": 0.5,
            },
            "gravity_m_s2": -9.81,
            "physics_dt_seconds": 1.0 / 60.0,
        },
    }
    value.update(updates)
    return value


def _cameras(**updates):
    value = {
        "id": "fixed-1",
        "artifact_reference": "frames/fixed-1.png",
        "digest": DIGEST,
        "width": 16,
        "height": 12,
        "pixel_mean": 100.0,
        "pixel_std": 12.0,
        "nonblank": True,
    }
    value.update(updates)
    return [value]


def _policy_trace_options() -> dict:
    start = [0.0, -0.55, 0.0, -2.6, 0.0, 2.05, 0.75]
    return {
        "robot_prim_path": "/World/Franka",
        "articulated_policy_trace_request": {
            "schema_version": "franka_articulated_policy_trace_request.v1",
            "robot_id": "franka_panda",
            "robot_prim_path": "/World/Franka",
            "controller_id": "deterministic_franka_joint_position_pair.v1",
            "joint_names": [f"panda_joint{index}" for index in range(1, 8)],
            "start_joint_positions_rad": start,
            "physics_dt_seconds": 1.0 / 60.0,
            "reset_settle_steps": 30,
            "sample_interval_steps": 10,
            "distinctness_threshold_rad": 0.1,
            "identical_start_tolerance_rad": 0.02,
            "candidates": [
                {
                    "policy_id": "franka-fixed-hold-v1",
                    "duration_steps": 120,
                    "final_joint_positions_rad": start,
                },
                {
                    "policy_id": "franka-inspection-sweep-v1",
                    "duration_steps": 120,
                    "final_joint_positions_rad": [
                        0.35,
                        -0.55,
                        0.0,
                        -2.6,
                        0.0,
                        2.05,
                        0.75,
                    ],
                },
            ],
            "egocentric_camera": {
                "parent_link_name": "panda_hand",
                "local_position_m": [0.05, 0.0, 0.04],
                "local_target_m": [0.3, 0.0, 0.04],
                "local_up": [0.0, 0.0, 1.0],
                "fov_degrees": 70.0,
                "width": 320,
                "height": 240,
            },
            "physical_success_claimed": False,
        },
    }


def test_policy_trace_request_and_pair_distinctness_are_fail_closed() -> None:
    request, blockers = runner._validate_policy_trace_request(_policy_trace_options())
    assert blockers == []
    assert request is not None
    traces = [
        {
            "status": "completed",
            "observed_start_joint_positions_rad": request["start_joint_positions_rad"],
            "observed_end_joint_positions_rad": request["start_joint_positions_rad"],
        },
        {
            "status": "completed",
            "observed_start_joint_positions_rad": [
                value + 0.001 for value in request["start_joint_positions_rad"]
            ],
            "observed_end_joint_positions_rad": request["candidates"][1][
                "final_joint_positions_rad"
            ],
        },
    ]
    assessment = runner._trace_pair_distinctness(traces, request)
    assert assessment["status"] == "completed"
    assert assessment["identical_frozen_start_observed"] is True
    assert assessment["distinct"] is True

    invalid = _policy_trace_options()
    invalid["articulated_policy_trace_request"]["candidates"][1]["final_joint_positions_rad"] = (
        invalid["articulated_policy_trace_request"]["start_joint_positions_rad"]
    )
    request, blockers = runner._validate_policy_trace_request(invalid)
    assert request is None
    assert "franka_policy_trace_sweep_not_distinct" in blockers


def test_joint_position_target_compatibility_uses_direct_legacy_api() -> None:
    calls = []

    class _Articulation:
        def set_joint_position_targets(self, positions, *, joint_indices):
            calls.append((positions, joint_indices))

    mode = runner._set_articulation_joint_position_targets(
        _Articulation(), [0.1, 0.2], [1, 2], action_factory=pytest.fail
    )

    assert mode == "direct_set_joint_position_targets"
    assert calls == [([0.1, 0.2], [1, 2])]


def test_joint_position_target_compatibility_uses_isaac6_action_api() -> None:
    actions = []

    class _Action:
        def __init__(self, **values):
            self.values = values

    class _Articulation:
        def apply_action(self, action):
            actions.append(action.values)

    mode = runner._set_articulation_joint_position_targets(
        _Articulation(), [0.3, 0.4], [3, 4], action_factory=_Action
    )

    assert mode == "articulation_apply_action"
    assert actions == [{"joint_positions": [0.3, 0.4], "joint_indices": [3, 4]}]


def test_joint_position_target_compatibility_uses_controller_fallback() -> None:
    actions = []

    class _Action:
        def __init__(self, **values):
            self.values = values

    class _Controller:
        def apply_action(self, action):
            actions.append(action.values)

    class _Articulation:
        def get_articulation_controller(self):
            return _Controller()

    mode = runner._set_articulation_joint_position_targets(
        _Articulation(), [0.5], [5], action_factory=_Action
    )

    assert mode == "articulation_controller_apply_action"
    assert actions == [{"joint_positions": [0.5], "joint_indices": [5]}]


def test_qualification_blockers_accept_only_complete_v3_evidence() -> None:
    assert (
        runner._qualification_blockers(
            package_digest=DIGEST,
            stage=_stage(),
            physics_probe=_physics(),
            cameras=_cameras(),
        )
        == []
    )


def test_camera_ids_map_to_valid_collision_checked_usd_prim_names() -> None:
    assert runner._camera_usd_prim_names(
        ["scene-overview-south", "ground-probe-local", "1.entry"]
    ) == ["scene_overview_south", "ground_probe_local", "_1_entry"]

    with pytest.raises(ValueError, match="isaac_camera_usd_prim_names_collide"):
        runner._camera_usd_prim_names(["camera-a", "camera_a"])


def test_runtime_bundle_upload_rejects_non_https_signed_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    result_path = out_dir / "result.json"
    result_path.write_text("{}", encoding="utf-8")
    monkeypatch.setenv(
        "BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL",
        "file:///tmp/untrusted-output.zip",
    )
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: pytest.fail("non-HTTPS URL must not be opened"),
    )

    runner._bundle_and_upload(out_dir, result_path)


def test_runtime_bundle_upload_accepts_validated_https_signed_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    result_path = out_dir / "result.json"
    result_path.write_text("{}", encoding="utf-8")
    monkeypatch.setenv(
        "BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL",
        "https://objects.example/output.zip?signature=bound",
    )
    opened: list[tuple[str, int]] = []

    class _Response:
        def read(self) -> bytes:
            return b""

    def _urlopen(request, *, timeout):
        opened.append((request.full_url, timeout))
        return _Response()

    monkeypatch.setattr(urllib.request, "urlopen", _urlopen)

    runner._bundle_and_upload(out_dir, result_path)

    assert opened == [("https://objects.example/output.zip?signature=bound", 300)]


def test_robot_only_pass_authors_hidden_lights_when_provider_stage_has_none() -> None:
    from pxr import Sdf, Usd, UsdGeom, UsdLux

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")

    prim, report = runner._ensure_robot_only_lights(
        stage,
        "/World/RobotEvidenceLights",
        Sdf=Sdf,
        UsdGeom=UsdGeom,
        UsdLux=UsdLux,
    )

    assert prim.IsValid()
    assert report["authored_for_robot_only_pass"] is True
    assert report["claim_boundary"] == "render_lighting_support_only_not_scene_or_task_evidence"
    assert UsdGeom.Imageable(prim).GetVisibilityAttr().Get() == "invisible"
    assert stage.GetPrimAtPath("/World/RobotEvidenceLights/Dome").IsA(UsdLux.DomeLight)
    assert stage.GetPrimAtPath("/World/RobotEvidenceLights/Distant").IsA(UsdLux.DistantLight)

    same_prim, reused = runner._ensure_robot_only_lights(
        stage,
        "/World/RobotEvidenceLights",
        Sdf=Sdf,
        UsdGeom=UsdGeom,
        UsdLux=UsdLux,
    )
    assert same_prim == prim
    assert reused["authored_for_robot_only_pass"] is False


def test_robot_evidence_material_is_explicitly_support_only() -> None:
    from pxr import Sdf, Usd, UsdGeom, UsdShade

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    robot = UsdGeom.Xform.Define(stage, "/World/Franka").GetPrim()

    report = runner._author_robot_evidence_material(
        stage,
        robot,
        Sdf=Sdf,
        UsdShade=UsdShade,
    )

    binding = UsdShade.MaterialBindingAPI(robot).GetDirectBinding()
    assert str(binding.GetMaterialPath()) == "/World/BlueprintRobotEvidenceMaterial"
    assert report["binding_strength"] == "strongerThanDescendants"
    assert report["claim_boundary"] == (
        "render_material_support_only_not_robot_asset_or_task_evidence"
    )


def test_visual_robot_is_excluded_from_environment_collision_probe() -> None:
    from pxr import Sdf, Usd, UsdGeom

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    robot = UsdGeom.Xform.Define(stage, "/World/Franka").GetPrim()
    report = {"composited": True, "prim_path": "/World/Franka"}

    runner._exclude_robot_from_environment_physics_probe(stage, report, Sdf=Sdf)

    assert robot.IsActive() is False
    assert report["excluded_from_environment_physics_probe"] is True
    assert report["physics_probe_claim_boundary"] == (
        "visual_robot_excluded_so_probe_measures_provider_environment_collision_only"
    )


def test_provider_package_mode_has_versioned_schema_and_dynamic_exact_prim_paths() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert (
        'PROVIDER_QUALIFICATION_RESULT_SCHEMA = "provider_nurec_isaac_runtime_result.v1"' in source
    )
    assert 'NUREC_FIELD_TYPE = "OmniNuRecFieldAsset"' in source
    assert "--provider-package-mode" in source
    assert "--expected-appearance-prim" in source
    assert "--expected-collision-prim" in source
    stage = _stage(expected_prim_paths={"appearance": None, "collision": None})
    assert "isaac_expected_prims_not_loaded" in runner._qualification_blockers(
        package_digest=DIGEST,
        stage=stage,
        physics_probe=_physics(),
        cameras=_cameras(),
    )


def test_v3_schema_accepts_only_explicit_completed_qualification_evidence() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    evidence = {
        "schema_version": "isaac_splat_nurec_render_result.v3",
        "status": "completed",
        "isaac_verification_request_digest": DIGEST,
        "package_digest": DIGEST,
        "fixed_camera_spec_digest": DIGEST,
        "runtime_container_image_digest": IMAGE,
        "runtime_implementation_digest": DIGEST,
        "runtime_identity": {
            "runtime": "isaac_sim",
            "renderer": "RayTracedLighting",
            "python_version": "3.11.0",
            "headless": True,
        },
        "raw_secret_values_recorded": False,
        "cost_usd": 0.25,
        "duration_seconds": 90.0,
        "stage": _stage(),
        "physics_probe": _physics(),
        "cameras": _cameras(),
        "proof_boundary": {
            "isaac_load_render_physics_presence_compatibility": True,
            "simulator_task_success_proven": False,
            "physics_navigation_control_proven": False,
            "physical_success_proven": False,
            "physical_robot_readiness_proven": False,
            "deployment_readiness_proven": False,
        },
    }
    evidence["isaac_runtime_result_digest"] = canonical_digest(
        evidence, digest_field="isaac_runtime_result_digest"
    )
    jsonschema.validate(evidence, schema)


def test_v3_schema_rejects_physical_claim_promotion() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    evidence = {
        "schema_version": "isaac_splat_nurec_render_result.v3",
        "status": "blocked",
        "isaac_verification_request_digest": DIGEST,
        "package_digest": DIGEST,
        "fixed_camera_spec_digest": DIGEST,
        "runtime_container_image_digest": IMAGE,
        "runtime_implementation_digest": DIGEST,
        "runtime_identity": {
            "runtime": "isaac_sim",
            "renderer": "RayTracedLighting",
            "python_version": "3.11.0",
            "headless": True,
        },
        "raw_secret_values_recorded": False,
        "proof_boundary": {
            "isaac_load_render_physics_presence_compatibility": False,
            "simulator_task_success_proven": False,
            "physics_navigation_control_proven": False,
            "physical_success_proven": True,
            "physical_robot_readiness_proven": False,
            "deployment_readiness_proven": False,
        },
    }
    evidence["isaac_runtime_result_digest"] = canonical_digest(
        evidence, digest_field="isaac_runtime_result_digest"
    )
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(evidence, schema)


def test_qualification_blockers_reject_blank_missing_collision_and_no_contact() -> None:
    blockers = runner._qualification_blockers(
        package_digest=DIGEST,
        stage=_stage(active_collision_prim_count=0),
        physics_probe=_physics(contact_event_count=0),
        cameras=_cameras(pixel_std=float("nan"), nonblank=False, digest=None),
    )
    assert "isaac_collision_geometry_inactive" in blockers
    assert "isaac_test_body_contact_not_observed" in blockers
    assert "isaac_fixed_render_invalid:0" in blockers


def test_qualification_blockers_reject_obvious_scale_mismatch() -> None:
    blockers = runner._qualification_blockers(
        package_digest=DIGEST,
        stage=_stage(obvious_scale_mismatch_detected=True),
        physics_probe=_physics(),
        cameras=_cameras(),
    )
    assert "isaac_obvious_scale_mismatch" in blockers


def test_qualification_blockers_reject_fall_through_and_unobserved_live_pose() -> None:
    blockers = runner._qualification_blockers(
        package_digest=DIGEST,
        stage=_stage(),
        physics_probe=_physics(
            live_rigid_body_pose_observed=False,
            test_body_fell_through_floor=True,
        ),
        cameras=_cameras(),
    )
    assert "isaac_test_body_pose_unavailable" in blockers
    assert "isaac_test_body_fell_through_floor" in blockers


def test_ground_selection_is_existing_active_floor_not_arbitrary_room_mesh() -> None:
    collisions = [
        {
            "prim_path": "/World/CollisionRoom",
            "active": True,
            "static": True,
            "world_bounds": {"min": [-2.0, -2.0, 0.0], "max": [2.0, 2.0, 3.0]},
        },
        {
            "prim_path": "/World/CollisionFloor",
            "active": True,
            "static": True,
            "world_bounds": {"min": [-2.0, -2.0, -0.1], "max": [2.0, 2.0, 0.0]},
        },
    ]
    selected, error = runner._select_ground_surface(collisions)
    assert error is None
    assert selected["prim_path"] == "/World/CollisionFloor"
    assert selected["probe_height_m"] == 0.0

    selected, error = runner._select_ground_surface(
        collisions,
        requested_path="/World/CollisionRoom",
    )
    assert selected is None
    assert error == "combined_ground_collider_requires_declared_height"

    dynamic_floor = dict(collisions[1], static=False)
    selected, error = runner._select_ground_surface([dynamic_floor])
    assert selected is None
    assert error == "ground_contact_surface_not_identified"


def test_probe_classifier_keeps_missing_pose_and_fall_state_unknown() -> None:
    result = runner._classify_physics_probe(
        ground_surface={"probe_height_m": 0.0},
        requested_steps=240,
        executed_steps=240,
        initial_position=None,
        final_position=None,
        contact_event_count=1,
        errors=["pose unavailable"],
    )
    assert result["live_rigid_body_pose_observed"] is False
    assert result["test_body_fell_through_floor"] is None


def test_qualification_mode_rejects_package_digest_mismatch_before_isaac_import(
    tmp_path: Path,
) -> None:
    package = tmp_path / "scene.usdc"
    package.write_bytes(b"not-the-declared-package")
    cameras = tmp_path / "cameras.json"
    cameras.write_text("[]", encoding="utf-8")
    out_dir = tmp_path / "out"
    process = subprocess.run(
        [
            sys.executable,
            str(RUNNER_PATH),
            "--usdc",
            str(package),
            "--cameras",
            str(cameras),
            "--out-dir",
            str(out_dir),
            "--qualification-mode",
            "--package-digest",
            DIGEST,
            "--verification-request-digest",
            DIGEST,
            "--camera-spec-digest",
            DIGEST,
            "--runtime-container-image-digest",
            IMAGE,
            "--runtime-implementation-digest",
            DIGEST,
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert process.returncode == 2
    result = json.loads((out_dir / "isaac_runtime_result.json").read_text(encoding="utf-8"))
    jsonschema.validate(result, json.loads(SCHEMA_PATH.read_text(encoding="utf-8")))
    assert result["schema_version"] == "isaac_splat_nurec_render_result.v3"
    assert result["package_digest"] == DIGEST
    assert result["blockers"] == ["isaac_exact_package_digest_mismatch"]


def test_legacy_cli_does_not_silently_emit_v2() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert 'LEGACY_RESULT_SCHEMA = "isaac_splat_nurec_render_result.v1"' in source
    assert 'QUALIFICATION_RESULT_SCHEMA = "isaac_splat_nurec_render_result.v3"' in source
    assert "qualification_mode = bool(args.qualification_mode)" in source
