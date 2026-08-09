from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
from typing import Any
import zipfile

import pytest
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from blueprint_pipeline import paid_resource_allocator as allocator
from blueprint_pipeline import adp009d_franka_vast as franka_vast
from blueprint_pipeline.articulated_native_diagnostic_bundle import (
    ArticulatedNativeDiagnosticError,
    REQUEST_SCHEMA,
    build_articulated_native_diagnostic_bundle,
    build_articulated_native_diagnostic_request,
)
from blueprint_pipeline.articulated_native_diagnostic_runtime import (
    _require_runtime_symbols,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.vast_provider_adapter import (
    _blueprint_bundle_preflight,
    _container_missing_max_seconds,
    _is_isaac_provider_bundle,
    _probe_env,
    _probe_shell_script,
    _provider_expected_video_count,
    _resolve_launch_mode,
    _resolve_probe_image,
    _select_offer,
)
from blueprint_pipeline.vast_retained_instance import bind_all_in_cost


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _implementation_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _asset(path: Path) -> Path:
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/Asset").GetPrim()
    stage.SetDefaultPrim(root)
    UsdPhysics.ArticulationRootAPI.Apply(root)
    for name in ("cabinet", "upper_door", "lower_door"):
        prim = UsdGeom.Xform.Define(stage, f"/Asset/{name}").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
        mesh = UsdGeom.Cube.Define(stage, f"/Asset/{name}/collision").GetPrim()
        UsdPhysics.CollisionAPI.Apply(mesh)
    joints = UsdGeom.Scope.Define(stage, "/Asset/joints")
    assert joints
    for name, body in (
        ("upper_door_hinge", "upper_door"),
        ("lower_door_hinge", "lower_door"),
    ):
        joint = UsdPhysics.RevoluteJoint.Define(stage, f"/Asset/joints/{name}")
        joint.CreateAxisAttr("Z")
        joint.CreateLowerLimitAttr(0.0)
        joint.CreateUpperLimitAttr(90.0)
        joint.CreateBody0Rel().SetTargets(["/Asset/cabinet"])
        joint.CreateBody1Rel().SetTargets([f"/Asset/{body}"])
    material = UsdShade.Material.Define(stage, "/Asset/render_materials/exterior")
    shader = UsdShade.Shader.Define(
        stage, "/Asset/render_materials/exterior_shader"
    )
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.8, 0.74, 0.70)
    )
    material.CreateSurfaceOutput().ConnectToSource(
        shader.ConnectableAPI(), "surface"
    )
    UsdShade.MaterialBindingAPI.Apply(root).Bind(material)
    stage.GetRootLayer().Save()
    return path


def _request(asset: Path, *, scene_id: str = "840796") -> dict:
    value = {
        "schema_version": REQUEST_SCHEMA,
        "program_id": "arm-decision-proof-v1",
        "scene_id": scene_id,
        "learned_policy_outcomes_observed": False,
        "asset": {"sha256": _sha256(asset)},
        "articulation": {
            "root_prim_path": "/Asset",
            "fixed_base_body_prim_path": "/Asset/cabinet",
            "driven_joint_prim_path": "/Asset/joints/upper_door_hinge",
            "locked_joint_prim_paths": ["/Asset/joints/lower_door_hinge"],
            "expected_joint_count": 2,
            "commanded_angles_degrees": [0.0, 30.0, 55.0],
        },
        "runtime": {
            "settle_steps_per_command": 120,
            "joint_readback_tolerance_degrees": 1.0,
            "locked_joint_tolerance_degrees": 0.5,
            "fixed_base_translation_tolerance_m": 1e-5,
            "fixed_base_rotation_tolerance_degrees": 0.01,
            "maximum_abs_joint_velocity_rad_s_after_settle": 0.05,
            "drive_stiffness": 2000.0,
            "drive_damping": 150.0,
            "drive_max_force": 5000.0,
        },
        "render_appearance": {
            "static_appearance_receipt_digest": "sha256:" + "c" * 64,
            "required_material_paths": ["/Asset/render_materials/exterior"],
            "resolution": [320, 180],
            "vertical_fov_degrees": 55.0,
            "minimum_pixel_stddev": 2.0,
            "cameras": [
                {
                    "camera_id": "external",
                    "role": "material_readback",
                    "position_asset_m": [1.7, 2.3, 1.45],
                    "look_at_asset_m": [0.0, 0.0, 0.8],
                },
                {
                    "camera_id": "overview",
                    "role": "review_only",
                    "position_asset_m": [3.05, 3.3, 1.9],
                    "look_at_asset_m": [0.0, 0.0, 0.8],
                },
            ],
        },
    }
    value["request_digest"] = canonical_digest(
        value, digest_field="request_digest"
    )
    return value


@pytest.mark.parametrize("scene_id", ["840313", "840796"])
def test_request_is_scene_neutral_and_allows_multiple_joints(
    tmp_path: Path, scene_id: str
) -> None:
    asset = _asset(tmp_path / f"{scene_id}.usda")
    request = build_articulated_native_diagnostic_request(
        _request(asset, scene_id=scene_id)
    )

    assert request["scene_id"] == scene_id
    assert request["articulation"]["expected_joint_count"] == 2
    assert request["articulation"]["locked_joint_prim_paths"] == [
        "/Asset/joints/lower_door_hinge"
    ]
    assert request["request_digest"].startswith("sha256:")


def test_request_rejects_policy_leakage_and_driven_joint_lock(tmp_path: Path) -> None:
    asset = _asset(tmp_path / "asset.usda")
    request = _request(asset)
    request["learned_policy_outcomes_observed"] = True
    request["articulation"]["locked_joint_prim_paths"] = [
        "/Asset/joints/upper_door_hinge"
    ]
    request.pop("request_digest")

    with pytest.raises(ArticulatedNativeDiagnosticError) as excinfo:
        build_articulated_native_diagnostic_request(request)

    assert "articulated_native_request_policy_outcome_leakage" in excinfo.value.codes
    assert "articulated_native_request_locked_joints_invalid" in excinfo.value.codes


def test_bundle_is_deterministic_and_binds_exact_asset(tmp_path: Path) -> None:
    asset = _asset(tmp_path / "asset.usda")
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(_request(asset)), encoding="utf-8")
    harness = tmp_path / "harness.json"
    harness.write_text('{"schema_version":"fixture"}\n', encoding="utf-8")
    before = asset.read_bytes()
    rows = []
    for name in ("first", "second"):
        rows.append(
            build_articulated_native_diagnostic_bundle(
                job_dir=tmp_path / name,
                asset_path=asset,
                request_path=request_path,
                harness_manifest_path=harness,
                implementation_commit=_implementation_commit(),
                generated_at="fixed",
            )
        )

    assert rows[0]["bundle_sha256"] == rows[1]["bundle_sha256"]
    assert rows[0]["diagnostic_kind"] == "blank_stage_articulated_asset"
    assert rows[0]["asset_binding"]["sha256"] == _sha256(asset)
    assert rows[0]["static_inventory"]["joint_prim_paths"] == [
        "/Asset/joints/lower_door_hinge",
        "/Asset/joints/upper_door_hinge",
    ]
    assert rows[0]["static_inventory"]["render_material_paths"] == [
        "/Asset/render_materials/exterior"
    ]
    assert rows[0]["candidate_policy_queried"] is False
    assert rows[0]["controls_requested"] is False
    assert asset.read_bytes() == before
    with zipfile.ZipFile(rows[0]["bundle_path"]) as archive:
        names = set(archive.namelist())
        entrypoint = archive.read(
            "provider_runtime/run_adp_arena_provider_runtime.sh"
        ).decode()
        runtime_source = archive.read(
            "provider_runtime/articulated_native_diagnostic_runtime.py"
        ).decode()
    assert "provider_runtime/assets/articulated_task_asset.usda" in names
    assert "provider_runtime/articulated_native_diagnostic_runtime.py" in names
    assert "adp009d_native_microcheck.json" in entrypoint
    assert "policy_provisioning" not in entrypoint
    assert "IsaacLab-Arena" not in entrypoint
    assert '"enable_cameras": True' in runtime_source
    assert "native_material_render_readback" in runtime_source
    assert "coverage_silhouette_audit_used" in runtime_source

    preflight = _blueprint_bundle_preflight(
        job_dir=tmp_path / "preflight",
        generated_at="fixed",
        enable_blueprint_bundle=True,
        enable_isaac_smoke=True,
        provider_bundle_kind="adp009d_articulated_native",
        bundle_path=Path(rows[0]["bundle_path"]),
        provider_bundle_url="https://example.com/bundle.zip?sig=redacted",
        provider_output_put_url="https://example.com/output.zip?sig=redacted",
    )
    assert preflight["blockers"] == []
    assert preflight["status"] == "passed"
    assert preflight["zip_required_entries_present"] is True


def test_articulated_bundle_uses_complete_native_isaac_transport_closure(
    tmp_path: Path,
) -> None:
    kind = "adp009d_articulated_native"

    assert _is_isaac_provider_bundle(kind) is True
    assert _provider_expected_video_count(kind) == 0
    assert _container_missing_max_seconds(kind) == 720
    assert (
        _resolve_launch_mode(
            requested="auto",
            enable_isaac_smoke=False,
            enable_blueprint_bundle=True,
            provider_bundle_kind=kind,
        )
        == "ssh_direct"
    )
    assert (
        _resolve_probe_image(
            public_image="public",
            isaac_image="isaac",
            enable_isaac_smoke=False,
            enable_blueprint_bundle=True,
            provider_bundle_kind=kind,
        )
        == "isaac"
    )
    env = _probe_env(
        job_dir=tmp_path,
        enable_isaac_smoke=False,
        provider_bundle_kind=kind,
        forward_hf_token=False,
    )
    assert env["ACCEPT_EULA"] == "Y"
    assert env["PRIVACY_CONSENT"] == "Y"

    script = _probe_shell_script(
        "https://example.com",
        enable_isaac_smoke=True,
        enable_blueprint_bundle=True,
        provider_bundle_kind=kind,
    )
    assert "BLUEPRINT_VAST_CUDA_RUNTIME_DEFERRED_TO_ISAAC_SIMULATION_APP" in script
    assert "cuda_runtime_rc=0" in script
    assert "run_adp_arena_provider_runtime.sh" in script
    assert "adp_arena_provider_runtime_output.zip" in script
    assert "run_wam_provider_runtime.sh" not in script


def test_paid_offer_and_post_create_gates_include_storage_in_hourly_cap(
    tmp_path: Path,
) -> None:
    offer = {
        "ask_contract_id": 7,
        "gpu_name": "RTX 6000 Ada",
        "dph_total": 0.70,
        "storage_cost": 0.54,
        "gpu_ram": 49_152,
        "num_gpus": 1,
        "rentable": True,
        "machine_id": 11,
        "driver_version": "580.119.02",
        "compute_cap": 890,
    }

    assert (
        _select_offer(
            [offer],
            max_hourly_rate=0.80,
            min_gpu_ram_mb=46_000,
            disk_gb=200,
        )
        is None
    )
    admitted = _select_offer(
        [offer],
        max_hourly_rate=0.90,
        min_gpu_ram_mb=46_000,
        disk_gb=200,
    )
    assert admitted is not None
    assert admitted["compute_hourly_rate_usd"] == pytest.approx(0.70)
    assert admitted["storage_hourly_rate_usd"] == pytest.approx(0.15)
    assert admitted["hourly_rate_usd"] == pytest.approx(0.85)

    binding = bind_all_in_cost(
        tmp_path,
        selected_offer=admitted,
        instance_payload={"dph_total": 0.86, "storage_total_cost": 0.16},
        instance_id=9,
        disk_gb=200,
        max_live_minutes=60,
        hard_cap_usd=1.0,
        max_hourly_rate_usd=0.80,
    )
    assert binding["all_in_hourly_rate_usd"] == pytest.approx(0.86)
    assert binding["all_in_hourly_rate_under_max_hourly"] is False


def test_runtime_capability_probe_reports_complete_missing_module_set() -> None:
    available = {
        name: SimpleNamespace(SingleArticulation=object, Camera=object)
        for name in (
            "carb",
            "numpy",
            "omni.timeline",
            "omni.usd",
            "isaacsim.core.prims",
            "isaacsim.sensors.camera",
            "PIL.Image",
            "pxr.Gf",
            "pxr.UsdGeom",
            "pxr.UsdLux",
            "pxr.UsdPhysics",
            "pxr.UsdShade",
        )
    }

    def importer(name: str) -> Any:
        if name in {"isaacsim.core.prims", "PIL.Image", "pxr.UsdLux"}:
            raise ModuleNotFoundError(name)
        return available[name]

    with pytest.raises(RuntimeError) as excinfo:
        _require_runtime_symbols(importer)

    message = str(excinfo.value)
    assert "isaacsim.core.prims.SingleArticulation:ModuleNotFoundError" in message
    assert "PIL.Image:ModuleNotFoundError" in message
    assert "pxr.UsdLux:ModuleNotFoundError" in message


def test_provider_runtime_uses_isaac6_articulation_api_not_removed_dynamic_control() -> None:
    source = Path(
        allocator.__file__
    ).with_name("articulated_native_diagnostic_runtime.py").read_text(encoding="utf-8")

    assert "isaacsim.core.prims" in source
    assert "SingleArticulation" in source
    assert "omni.isaac.dynamic_control" not in source
    assert "joint_command_started" in source
    assert "joint_command_completed" in source
    assert "runtime_capabilities_resolved" in source


def test_bundle_rejects_changed_asset(tmp_path: Path) -> None:
    asset = _asset(tmp_path / "asset.usda")
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(_request(asset)), encoding="utf-8")
    asset.write_text(asset.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    harness = tmp_path / "harness.json"
    harness.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ArticulatedNativeDiagnosticError) as excinfo:
        build_articulated_native_diagnostic_bundle(
            job_dir=tmp_path / "bundle",
            asset_path=asset,
            request_path=request_path,
            harness_manifest_path=harness,
            implementation_commit=_implementation_commit(),
            generated_at="fixed",
        )

    assert "articulated_native_asset_digest_mismatch" in excinfo.value.codes


def test_bundle_rejects_missing_render_material_before_paid_runtime(
    tmp_path: Path,
) -> None:
    asset = _asset(tmp_path / "asset.usda")
    request = _request(asset)
    request["render_appearance"]["required_material_paths"] = [
        "/Asset/render_materials/missing"
    ]
    request["request_digest"] = canonical_digest(
        request, digest_field="request_digest"
    )
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    harness = tmp_path / "harness.json"
    harness.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ArticulatedNativeDiagnosticError) as excinfo:
        build_articulated_native_diagnostic_bundle(
            job_dir=tmp_path / "bundle",
            asset_path=asset,
            request_path=request_path,
            harness_manifest_path=harness,
            implementation_commit=_implementation_commit(),
            generated_at="fixed",
        )

    assert (
        "articulated_native_render_material_missing:"
        "/Asset/render_materials/missing"
    ) in excinfo.value.codes


def test_bundle_rejects_caller_asserted_commit_that_is_not_current(
    tmp_path: Path,
) -> None:
    asset = _asset(tmp_path / "asset.usda")
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(_request(asset)), encoding="utf-8")
    harness = tmp_path / "harness.json"
    harness.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ArticulatedNativeDiagnosticError) as excinfo:
        build_articulated_native_diagnostic_bundle(
            job_dir=tmp_path / "bundle",
            asset_path=asset,
            request_path=request_path,
            harness_manifest_path=harness,
            implementation_commit="0" * 40,
            generated_at="fixed",
        )

    assert "articulated_native_implementation_commit_mismatch" in excinfo.value.codes


def test_checked_in_second_scene_request_binds_material_readback() -> None:
    request_path = (
        Path(__file__).resolve().parents[1]
        / "docs/arm_decision_proof_v1/manifests"
        / "second_scene_840796_articulated_native_diagnostic_request.v2.json"
    )
    request = build_articulated_native_diagnostic_request(
        json.loads(request_path.read_text(encoding="utf-8"))
    )

    assert request["asset"]["sha256"] == (
        "sha256:9f487fbee7006c6c276ce37c9e1e7e1653a465ac2fbbe15cffa63653111cc720"
    )
    assert request["render_appearance"]["required_material_paths"] == [
        "/Asset/render_materials/observed_exterior",
        "/Asset/render_materials/generated_unobserved",
    ]
    assert {row["role"] for row in request["render_appearance"]["cameras"]} == {
        "material_readback",
        "review_only",
    }


def _allocator_args(tmp_path: Path) -> list[str]:
    return [
        "gpu-canary",
        "--probe-kind",
        "adp009d-franka-native-microcheck",
        "--provider",
        "vast",
        "--provider-launch-request",
        str(tmp_path / "unused-request.json"),
        "--release-evidence",
        str(tmp_path / "unused-release.json"),
        "--model-cache-evidence",
        str(tmp_path / "unused-model.json"),
        "--preflight-bundle",
        str(tmp_path / "unused-preflight.json"),
        "--admission-out",
        str(tmp_path / "admission.json"),
        "--bound-request-out",
        str(tmp_path / "unused-bound.json"),
        "--adapter-output",
        str(tmp_path / "adapter.json"),
        "--pod-name",
        "adp009d-articulated-native",
        "--adp009d-harness-manifest",
        str(tmp_path / "harness.json"),
        "--adp009d-articulated-diagnostic-asset",
        str(tmp_path / "asset.usda"),
        "--adp009d-articulated-diagnostic-request",
        str(tmp_path / "request.json"),
        "--adp009d-diagnostic-only",
        "--adp-job-dir",
        str(tmp_path / "job"),
        "--adp-max-hourly-rate-usd",
        "1.0",
        "--adp-max-spend-usd",
        "2.0",
        "--adp-hard-ttl-seconds",
        "7200",
    ]


def test_allocator_routes_articulated_diagnostic_without_canned_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: dict[str, Any] = {}
    monkeypatch.setattr(
        allocator,
        "_control_plane_checkout_blockers",
        lambda: ([], {"orchestrator_source_commit": "a" * 40, "checkout_clean": True}),
    )

    def fake_build(**kwargs):
        observed["build"] = kwargs
        return {
            "status": "ready",
            "bundle_path": str(tmp_path / "bundle.zip"),
            "bundle_sha256": "sha256:" + "b" * 64,
            "input_digest": "sha256:" + "c" * 64,
            "request_digest": "sha256:" + "d" * 64,
            "diagnostic_kind": "blank_stage_articulated_asset",
        }

    monkeypatch.setattr(
        allocator, "build_articulated_native_diagnostic_bundle", fake_build
    )
    def fake_run(**kwargs):
        observed["run"] = kwargs
        return {"status": "dry_run_ready"}

    monkeypatch.setattr(allocator, "run_adp009d_native_microcheck_vast", fake_run)

    assert allocator.main(_allocator_args(tmp_path)) == 0
    assert "approved_can_path" not in observed["build"]
    assert observed["build"]["asset_path"] == str(tmp_path / "asset.usda")
    admission = json.loads((tmp_path / "admission.json").read_text())
    binding = admission["allocation_binding"]
    assert binding["articulated_native_diagnostic_requested"] is True
    assert binding["diagnostic_kind"] == "blank_stage_articulated_asset"
    assert binding["articulated_native_request_digest"] == "sha256:" + "d" * 64


def test_allocator_forbids_policy_in_blank_stage_articulated_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        allocator,
        "_control_plane_checkout_blockers",
        lambda: ([], {"orchestrator_source_commit": "a" * 40, "checkout_clean": True}),
    )
    args = _allocator_args(tmp_path) + [
        "--adp009d-policy-candidate",
        "pi05_droid",
    ]

    assert allocator.main(args) == 2
    result = json.loads((tmp_path / "adapter.json").read_text())
    assert "adp009d_execution_modes_conflict" in result["blockers"]
    assert (
        "adp009d_articulated_native_policy_or_controls_forbidden"
        in result["blockers"]
    )


def test_transport_selects_articulated_bundle_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, Any] = {}

    def fake_run(**kwargs):
        observed.update(kwargs)
        return {"status": "dry_run_ready"}

    monkeypatch.setattr(franka_vast, "run_arena_native_control_vast", fake_run)
    result = franka_vast.run_adp009d_native_microcheck_vast(
        job_dir="job",
        prepared_bundle={
            "status": "ready",
            "policy_candidate_id": None,
            "diagnostic_kind": "blank_stage_articulated_asset",
        },
        paid_resource_admission_grant=None,
        execute=False,
    )

    assert result["status"] == "dry_run_ready"
    assert observed["provider_bundle_kind"] == "adp009d_articulated_native"
    assert observed["minimum_driver_version"] == "580.65.06"
    assert observed["candidate_policy_query_expected"] is False
