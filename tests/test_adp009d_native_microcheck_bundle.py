from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from blueprint_pipeline import paid_resource_allocator as allocator
from blueprint_pipeline import adp009d_isaac_runtime as isaac_runtime
from blueprint_pipeline.adp009d_native_microcheck_bundle import (
    APPROVED_CAN_ADAPTER_FILENAME,
    DEFAULT_IMAGE,
    PROBE_KIND,
    SUPPORT_COLLIDER_PRIM,
    TASK_COLLISION_DERIVATIVE_FILENAME,
    TASK_COLLISION_MANIFEST_FILENAME,
    TASK_COLLISION_MAX_EDGE_M,
    TARGET_COLLIDER_PRIM,
    _clip_triangle_to_aabb,
    _refine_triangle_to_edge_limit,
    _inspect_sage_collision_source,
    build_native_microcheck_bundle,
    build_native_microcheck_bundle_isolated,
)
from blueprint_pipeline import adp009d_franka_vast as franka_vast
from blueprint_pipeline.paid_resource_admission import PaidResourceAdmissionGrant
from blueprint_pipeline.vast_provider_adapter import (
    _blueprint_bundle_preflight,
    _inspect_provider_runtime_output_zip,
    _provider_expected_video_count,
)


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, str]]:
    approved = tmp_path / "approved.usda"
    approved.write_text(
        '''#usda 1.0
(defaultPrim = "canned_beverage")
def Xform "canned_beverage"
{
    def Scope "colliders"
    {
        def Mesh "body_collider" (
            prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]
        )
        {
            uniform token physics:approximation = "sdf"
        }
    }
}
''',
        encoding="utf-8",
    )
    sage = tmp_path / "sage_collision.usd"
    target_name = TARGET_COLLIDER_PRIM.rsplit("/", 1)[-1]
    support_name = SUPPORT_COLLIDER_PRIM.rsplit("/", 1)[-1]
    sage.write_text(
        (
            '#usda 1.0\n(\n    defaultPrim = "Root"\n    metersPerUnit = 1\n'
            '    upAxis = "Z"\n)\n\n'
            'def Xform "Root"\n{\n'
            f'''    def Mesh "{target_name}" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]
    )
    {{
        uniform token physics:approximation = "convexDecomposition"
        point3f[] points = [(3, -3, 0.3), (4, -3, 0.3), (3, -2, 0.3)]
        int[] faceVertexCounts = [3]
        int[] faceVertexIndices = [0, 1, 2]
    }}
    def Mesh "{support_name}" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]
    )
    {{
        uniform token physics:approximation = "convexDecomposition"
        point3f[] points = [(3, -3, 0.3), (4, -3, 0.3), (3, -2, 0.3)]
        int[] faceVertexCounts = [3]
        int[] faceVertexIndices = [0, 1, 2]
    }}
'''
            '}\n'
        ),
        encoding="utf-8",
    )
    harness = tmp_path / "harness.json"
    harness.write_text('{"schema_version":"test"}\n', encoding="utf-8")
    bindings = {
        "approved_can.usda": _digest(approved),
        "sage_collision.usd": _digest(sage),
    }
    return approved, sage, harness, bindings


def test_bundle_is_deterministic_and_keeps_sealed_sources_unchanged(tmp_path: Path) -> None:
    approved, sage, harness, bindings = _inputs(tmp_path)
    approved_before = approved.read_bytes()
    sage_before = sage.read_bytes()
    kwargs = {
        "approved_can_path": approved,
        "sage_collision_path": sage,
        "harness_manifest_path": harness,
        "implementation_commit": "a" * 40,
        "generated_at": "fixed",
        "expected_asset_bindings": bindings,
    }
    first = build_native_microcheck_bundle(job_dir=tmp_path / "first", **kwargs)
    second = build_native_microcheck_bundle(job_dir=tmp_path / "second", **kwargs)

    assert first["probe_kind"] == PROBE_KIND
    assert first["container_image"] == DEFAULT_IMAGE
    assert first["bundle_sha256"] == second["bundle_sha256"]
    assert first["candidate_policy_queried"] is False
    assert approved.read_bytes() == approved_before
    assert sage.read_bytes() == sage_before
    Usd = pytest.importorskip("pxr.Usd")
    source_stage = Usd.Stage.Open(str(sage))
    overlay_stage = Usd.Stage.Open(
        str(Path(first["bundle_path"]).parent / "provider_runtime/assets/sage_collision_overlay.usda")
    )
    assert source_stage.GetPrimAtPath(TARGET_COLLIDER_PRIM).IsActive()
    assert not overlay_stage.GetPrimAtPath(TARGET_COLLIDER_PRIM).IsActive()
    assert overlay_stage.GetPrimAtPath(SUPPORT_COLLIDER_PRIM).IsActive()
    assert (
        overlay_stage.GetPrimAtPath(SUPPORT_COLLIDER_PRIM)
        .GetAttribute("physics:approximation")
        .Get()
        == "none"
    )
    adapter_path = (
        Path(first["bundle_path"]).parent
        / "provider_runtime/assets"
        / APPROVED_CAN_ADAPTER_FILENAME
    )
    adapter_stage = Usd.Stage.Open(str(adapter_path))
    assert _digest(adapter_path) == isaac_runtime.APPROVED_CAN_ADAPTER_SHA256
    can_collider = adapter_stage.GetPrimAtPath(
        "/canned_beverage/colliders/body_collider"
    )
    api_schemas = can_collider.GetMetadata("apiSchemas")
    assert "PhysxSDFMeshCollisionAPI" in list(api_schemas.GetAddedOrExplicitItems())
    assert can_collider.GetAttribute("physics:approximation").Get() == "sdf"
    assert can_collider.GetAttribute(
        "physxSDFMeshCollision:sdfResolution"
    ).Get() == 256
    with zipfile.ZipFile(first["bundle_path"]) as archive:
        names = set(archive.namelist())
        overlay = archive.read("provider_runtime/assets/sage_collision_overlay.usda").decode()
        entrypoint = archive.read("provider_runtime/run_adp_arena_provider_runtime.sh").decode()
    assert TARGET_COLLIDER_PRIM.rsplit("/", 1)[-1] in overlay
    assert "active = false" in overlay
    assert 'uniform token physics:approximation = "none"' in overlay
    assert "adp009d_native_microcheck.json" in entrypoint
    assert "provider_runtime/assets/approved_can.usda" in names
    assert f"provider_runtime/assets/{APPROVED_CAN_ADAPTER_FILENAME}" in names
    assert "provider_runtime/assets/sage_collision.usd" in names
    assert f"provider_runtime/assets/{TASK_COLLISION_DERIVATIVE_FILENAME}" in names
    assert f"provider_runtime/assets/{TASK_COLLISION_MANIFEST_FILENAME}" in names
    derivative_root = Path(first["bundle_path"]).parent / "provider_runtime/assets"
    derivative = Usd.Stage.Open(str(derivative_root / TASK_COLLISION_DERIVATIVE_FILENAME))
    derivative_manifest = json.loads(
        (derivative_root / TASK_COLLISION_MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    assert not derivative.GetPrimAtPath(TARGET_COLLIDER_PRIM).IsActive()
    assert derivative.GetPrimAtPath(SUPPORT_COLLIDER_PRIM).IsActive()
    assert derivative_manifest["sealed_source_sha256"] == bindings["sage_collision.usd"]
    assert derivative_manifest["sealed_source_mutated"] is False
    assert derivative_manifest["observed_maximum_edge_m"] <= TASK_COLLISION_MAX_EDGE_M
    assert derivative_manifest["relative_surface_area_error"] <= 1.0e-6


def test_isolated_bundle_builder_returns_fresh_digest_bound_receipt(tmp_path: Path) -> None:
    approved, sage, harness, bindings = _inputs(tmp_path)
    receipt = build_native_microcheck_bundle_isolated(
        job_dir=tmp_path / "isolated",
        approved_can_path=approved,
        sage_collision_path=sage,
        harness_manifest_path=harness,
        implementation_commit="d" * 40,
        generated_at="fixed",
        expected_asset_bindings=bindings,
    )

    assert receipt["status"] == "ready"
    assert receipt["implementation_commit"] == "d" * 40
    assert receipt["bundle_sha256"] == _digest(Path(receipt["bundle_path"]))


def test_task_collision_retriangulation_preserves_area_and_limits_edges() -> None:
    triangle = ((0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (0.0, 3.0, 0.0))

    leaves = _refine_triangle_to_edge_limit(triangle, max_edge_m=0.5)

    def area(value):
        a, b, c = value
        return abs(
            (b[0] - a[0]) * (c[1] - a[1])
            - (b[1] - a[1]) * (c[0] - a[0])
        ) * 0.5

    assert sum(area(value) for value in leaves) == pytest.approx(6.0)
    assert max(
        sum(
            (value[edge][axis] - value[(edge + 1) % 3][axis]) ** 2
            for axis in range(3)
        )
        ** 0.5
        for value in leaves
        for edge in range(3)
    ) <= 0.5 + 1.0e-9


def test_task_collision_aabb_clip_preserves_only_exact_coplanar_surface() -> None:
    triangle = ((-2.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 2.0, 0.0))

    clipped = _clip_triangle_to_aabb(
        triangle,
        roi_min=(-1.0, -1.0, -1.0),
        roi_max=(1.0, 1.0, 1.0),
    )

    assert clipped
    assert all(-1.0 <= coordinate <= 1.0 for face in clipped for point in face for coordinate in point)
    assert sum(
        abs(
            (face[1][0] - face[0][0]) * (face[2][1] - face[0][1])
            - (face[1][1] - face[0][1]) * (face[2][0] - face[0][0])
        )
        * 0.5
        for face in clipped
    ) == pytest.approx(2.0)


def test_bundle_passes_existing_arena_transport_preflight(tmp_path: Path) -> None:
    approved, sage, harness, bindings = _inputs(tmp_path)
    receipt = build_native_microcheck_bundle(
        job_dir=tmp_path / "bundle",
        approved_can_path=approved,
        sage_collision_path=sage,
        harness_manifest_path=harness,
        implementation_commit="b" * 40,
        generated_at="fixed",
        expected_asset_bindings=bindings,
    )

    preflight = _blueprint_bundle_preflight(
        job_dir=tmp_path / "preflight",
        generated_at="fixed",
        enable_blueprint_bundle=True,
        enable_isaac_smoke=True,
        provider_bundle_kind="adp009d_isaac",
        bundle_path=Path(receipt["bundle_path"]),
        provider_bundle_url="https://example.com/bundle.zip?sig=redacted",
        provider_output_put_url="https://example.com/output.zip?sig=redacted",
    )

    assert preflight["status"] == "passed"
    assert preflight["blockers"] == []


def test_bundle_rejects_unbound_asset_bytes(tmp_path: Path) -> None:
    approved, sage, harness, bindings = _inputs(tmp_path)
    bindings["sage_collision.usd"] = "sha256:" + "0" * 64

    with pytest.raises(ValueError, match="adp009d_bound_asset_digest_mismatch:sage_collision.usd"):
        build_native_microcheck_bundle(
            job_dir=tmp_path / "bundle",
            approved_can_path=approved,
            sage_collision_path=sage,
            harness_manifest_path=harness,
            implementation_commit="c" * 40,
            expected_asset_bindings=bindings,
        )


def test_sage_source_audit_rejects_any_rigid_environment_mesh(tmp_path: Path) -> None:
    _approved, sage, _harness, _bindings = _inputs(tmp_path)
    text = sage.read_text(encoding="utf-8").replace(
        'prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]',
        'prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI", "PhysicsRigidBodyAPI"]',
        1,
    )
    sage.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="adp009d_sage_static_collision_has_rigid_body"):
        _inspect_sage_collision_source(sage, enforce_sealed_profile=False)


def test_runtime_requires_composed_static_triangle_meshes(tmp_path: Path) -> None:
    _approved, sage, _harness, _bindings = _inputs(tmp_path)
    profile = _inspect_sage_collision_source(sage, enforce_sealed_profile=False)
    from blueprint_pipeline.adp009d_native_microcheck_bundle import _overlay_text

    overlay = tmp_path / "overlay.usda"
    overlay.write_text(_overlay_text(profile), encoding="utf-8")
    Usd = pytest.importorskip("pxr.Usd")
    stage = Usd.Stage.Open(str(overlay))

    observed = isaac_runtime._inspect_sage_static_triangle_colliders(
        stage,
        "/Root",
        expected_profile={
            "active_mesh_count": 1,
            "active_point_count": 3,
            "active_face_count": 1,
            "rigid_body_count": 0,
            "triangle_mesh_count": 1,
        },
    )

    assert observed["target_collider_active"] is False
    assert observed["support_collider_active"] is True
    assert observed["approximation"] == "none"


def test_runtime_binds_official_droid_franka_and_sealed_anchor() -> None:
    assert isaac_runtime.ARENA_REVISION == "8b4a3a47fc53de23e8205089d71109a2e2348acd"
    assert isaac_runtime.ISAAC_LAB_REVISION == "e57379c634b42db5a0fe9f754341be6e2a7c7c43"
    assert isaac_runtime.EXPECTED_ASSETS == {
        "approved_can.usda": "sha256:61c2a03bef425803d82cc5ef24ced5b2ccb4160923c53bb10c6ad0e3f52532ec",
        "sage_collision.usd": "sha256:b265706c24f6a8ace3ee6743fd138583c4e21d83f61b99a06fd435e6ac2d6b41",
    }
    assert isaac_runtime.ROBOT_BASE_POSITION_M == (3.4681748, -2.8100837, 0.2766791)
    assert isaac_runtime.CAN_START_POSITION_M == (
        3.4681748,
        -3.3100837,
        0.5264650138348479,
    )


def test_runtime_converts_warp_arrays_before_indexing() -> None:
    wp = pytest.importorskip("warp")
    torch = pytest.importorskip("torch")
    value = wp.array([[1.0, 2.0]], dtype=wp.float32, device="cpu")

    converted = isaac_runtime._to_torch(value)

    assert isinstance(converted, torch.Tensor)
    assert converted.tolist() == [[1.0, 2.0]]


def test_runtime_fails_closed_on_missing_sdf_schema(tmp_path: Path) -> None:
    Usd = pytest.importorskip("pxr.Usd")
    stage = Usd.Stage.CreateInMemory()
    prim = stage.DefinePrim("/Can/collider", "Mesh")
    prim.ApplyAPI("PhysicsCollisionAPI")
    prim.ApplyAPI("PhysicsMeshCollisionAPI")
    prim.CreateAttribute("physics:approximation", pytest.importorskip("pxr.Sdf").ValueTypeNames.Token).Set(
        "sdf"
    )

    with pytest.raises(RuntimeError, match="physx_sdf_schema_missing"):
        isaac_runtime._inspect_physx_sdf_collider(stage, "/Can/collider")


def test_runtime_rejects_any_physx_collision_fallback() -> None:
    message = (
        "PhysicsUSD: Parse collision - triangle mesh collision cannot be a part "
        "of a dynamic body, falling back to convexHull approximation: /World/Can"
    )

    with pytest.raises(RuntimeError, match="physx_collision_fallback_detected"):
        isaac_runtime._fail_on_physx_collision_fallback([message])


def test_runtime_rejects_physx_triangle_stability_warning() -> None:
    message = (
        "PhysX warning: TriangleMesh: triangles are too big, reduce their size "
        "to increase simulation stability!"
    )

    with pytest.raises(
        RuntimeError, match="physx_collision_stability_warning_detected"
    ):
        isaac_runtime._fail_on_physx_collision_stability([message])


def test_runtime_uses_documented_legacy_cooker_after_measured_ujitso_stall() -> None:
    source = Path(isaac_runtime.__file__).read_text(encoding="utf-8")

    assert (
        isaac_runtime.PHYSX_COLLISION_COOKING_PROFILE
        == "legacy_cooker_after_ujitso_stall.v1"
    )
    assert "SETTING_UJITSO_COLLISION_COOKING" in source
    assert "settings.set_bool(key, False)" in source
    assert '"ujitso_resolved_enabled": bool(resolved_enabled)' in source
    assert '"collider_geometry_or_parameters_changed": False' in source
    assert '"measured_ujitso_environment_construction_stall_v14"' in source
    assert '_phase("physx_collision_cooking_configuration")' in source
    assert (
        '_phase("physx_collision_cooking_configuration", "completed")' in source
    )


def test_runtime_does_not_import_unneeded_arena_asset_registry() -> None:
    source = Path(isaac_runtime.__file__).read_text(encoding="utf-8")

    assert "isaaclab_arena.assets.object_library" not in source
    assert "class SpawnerObject(Object):" in source
    assert "object_type=ObjectType.SPAWNER" in source


def test_runtime_preflights_exact_arena_environment_import_closure() -> None:
    source = Path(isaac_runtime.__file__).read_text(encoding="utf-8")

    assert 'def _preflight_environment_imports() -> dict[str, str]:' in source
    assert '_phase("runtime_import_preflight")' in source
    assert "from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder" in source
    assert '"hydra-core"' in source
    assert '"h5py"' in source
    assert "from isaaclab_ov.renderers import OVRTXRendererCfg" in source
    assert "from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper" in source
    assert "from rsl_rl.runners import DistillationRunner, OnPolicyRunner" in source
    assert "import zmq" in source
    assert '"traceback": traceback.format_exc()' in source


def test_runtime_emits_granular_environment_construction_phases() -> None:
    source = Path(isaac_runtime.__file__).read_text(encoding="utf-8")

    for phase in (
        "embodiment_configuration",
        "sealed_scene_configuration",
        "arena_environment_definition",
        "arena_builder_registration",
        "manager_based_environment_construction",
    ):
        assert f'_phase("{phase}")' in source
        assert f'_phase("{phase}", "completed")' in source


def test_runtime_binds_and_verifies_canonical_reset_pose() -> None:
    source = Path(isaac_runtime.__file__).read_text(encoding="utf-8")

    assert len(isaac_runtime.RESET_JOINT_NAMES) == len(isaac_runtime.RESET_JOINTS)
    assert isaac_runtime.RESET_JOINT_NAMES[:7] == tuple(
        f"panda_joint{index}" for index in range(1, 8)
    )
    assert "robot.init_state = robot.init_state.replace" in source
    assert "joint_pos=canonical_joint_positions" in source
    assert ".joint_pos.update(" not in source
    assert "_configure_deterministic_reset_events(embodiment)" in source
    assert 'reset_writer.params["mean"] = 0.0' in source
    assert 'reset_writer.params["std"] = 0.0' in source
    assert "randomize_franka_joint_state = None" not in source
    assert 'blocker="canonical_reset_arm_pose_mismatch"' in source
    assert 'blocker="canonical_hold_arm_pose_drift"' in source
    assert '"post_warmup_arm_maximum_error_rad"' in source
    assert 'result["diagnostics"] = _json_safe(diagnostics)' in source
    assert "_assert_canonical_object_stability(" in source
    assert '"canonical_hold_object_stability": object_stability' in source
    assert "approved_can_support_loss_after_zero_action" not in source


def test_canonical_reset_uses_official_arena_droid_safe_pose() -> None:
    assert isaac_runtime.RESET_JOINTS[:7] == pytest.approx(
        (0.0, -0.2 * 3.14159265359, 0.0, -0.8 * 3.14159265359, 0.0, 0.6 * 3.14159265359, 0.0)
    )


def test_canonical_reset_uses_measured_contact_stable_open_gripper() -> None:
    assert isaac_runtime.RESET_JOINTS[7:] == pytest.approx(
        (
            0.104255385697,
            0.104152053595,
            -0.128436118364,
            0.125143155456,
            -0.071244180202,
            -0.080966427922,
        )
    )


def test_canonical_reset_replaces_overlapping_arena_regex_defaults() -> None:
    class FakeInitialState:
        def __init__(self, joint_pos: dict[str, float]) -> None:
            self.joint_pos = joint_pos

        def replace(self, *, joint_pos: dict[str, float]):
            return FakeInitialState(joint_pos)

    class FakeRobot:
        init_state = FakeInitialState(
            {
                "panda_joint1": 0.0,
                "right_outer.*": 0.0,
                "left_inner.*": 0.0,
                "right_inner.*": 0.0,
            }
        )

    class FakeSceneConfig:
        robot = FakeRobot()

    class FakeEmbodiment:
        scene_config = FakeSceneConfig()

    embodiment = FakeEmbodiment()
    isaac_runtime._bind_canonical_joint_positions(embodiment)

    resolved = embodiment.scene_config.robot.init_state.joint_pos
    assert resolved == dict(
        zip(
            isaac_runtime.RESET_JOINT_NAMES,
            isaac_runtime.RESET_JOINTS,
            strict=True,
        )
    )
    assert not any("*" in name for name in resolved)


def test_canonical_reset_keeps_zero_noise_state_writer_event() -> None:
    class FakeTerm:
        def __init__(self, params: dict[str, object]) -> None:
            self.params = params

    class FakeEvents:
        init_franka_arm_pose = FakeTerm({"default_pose": [99.0]})
        randomize_franka_joint_state = FakeTerm({"mean": 1.0, "std": 2.0})

    class FakeEmbodiment:
        event_config = FakeEvents()

    embodiment = FakeEmbodiment()
    isaac_runtime._configure_deterministic_reset_events(embodiment)

    assert embodiment.event_config.init_franka_arm_pose.params["default_pose"] == list(
        isaac_runtime.RESET_JOINTS
    )
    assert embodiment.event_config.randomize_franka_joint_state.params == {
        "mean": 0.0,
        "std": 0.0,
    }


def test_canonical_pose_failure_retains_per_joint_diagnostics() -> None:
    diagnostics = isaac_runtime._canonical_pose_diagnostics(
        actual_arm=[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        expected_arm=[0.0] * 7,
        absolute_error=[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        maximum_error=1.0,
        tolerance_rad=1.0e-3,
    )
    error = isaac_runtime.CanonicalPoseError(
        "canonical_reset_arm_pose_mismatch", diagnostics
    )

    assert diagnostics["joint_names"] == list(isaac_runtime.RESET_JOINT_NAMES[:7])
    assert diagnostics["requested_joint_positions_rad"] == [0.0] * 7
    assert diagnostics["observed_joint_positions_rad"] == [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert diagnostics["absolute_error_rad"] == [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert diagnostics["maximum_error_rad"] == 1.0
    assert str(error) == "canonical_reset_arm_pose_mismatch:maximum_error_rad=1.000000000"


def test_canonical_object_stability_rejects_contact_displacement() -> None:
    initial = [3.4681747, -3.3100836, 0.5264650, 0.0, 0.0, 0.0, 1.0]
    final = [
        3.2928247,
        -3.4017854,
        0.5575565,
        -0.6239471,
        -0.3326830,
        0.0779095,
        -0.7028100,
    ]
    with pytest.raises(isaac_runtime.CanonicalObjectStabilityError) as exc_info:
        isaac_runtime._assert_canonical_object_stability(initial, final)

    diagnostics = exc_info.value.diagnostics
    assert diagnostics["xy_displacement_m"] == pytest.approx(0.1978808, abs=1e-6)
    assert diagnostics["final_tilt_degrees"] == pytest.approx(89.9986, abs=1e-3)
    assert diagnostics["thresholds"] == {
        "xy_displacement_m": 0.005,
        "absolute_z_displacement_m": 0.005,
        "tilt_degrees": 2.0,
    }


def test_canonical_object_stability_accepts_settled_pose() -> None:
    initial = [3.4681747, -3.3100836, 0.5264650, 0.0, 0.0, 0.0, 1.0]
    final = [3.4681745, -3.3100832, 0.5264686, -1.16e-5, -8.21e-5, 0.0, 1.0]
    diagnostics = isaac_runtime._assert_canonical_object_stability(initial, final)

    assert diagnostics["xy_displacement_m"] < 1.0e-6
    assert diagnostics["absolute_z_displacement_m"] < 5.0e-6
    assert diagnostics["final_tilt_degrees"] < 0.01


def test_runtime_retains_camera_semantic_mapping_and_quality_diagnostics() -> None:
    source = Path(isaac_runtime.__file__).read_text(encoding="utf-8")

    assert '(camera.data.info or {}).get("semantic_segmentation")' in source
    assert '"id_to_labels": semantic_info' in source
    assert '"pixel_counts_by_id": semantic_pixel_counts' in source
    assert '"finite_metric_depth_fraction"' in source
    assert '"foreground_semantic_pixel_fraction"' in source


def test_worker_rewrites_only_public_isaac_lab_submodule_transport() -> None:
    source = Path(isaac_runtime.__file__).with_name("adp009d_native_microcheck_worker.py")
    text = source.read_text(encoding="utf-8")

    assert "url.https://github.com/.insteadOf=git@github.com:" in text
    assert '"submodules/IsaacLab"' in text


def test_worker_uses_smallest_pinned_official_arena_physx_install_closure(tmp_path: Path) -> None:
    from blueprint_pipeline import adp009d_native_microcheck_worker as worker

    source = tmp_path / "arena"
    commands = worker._install_commands(source)
    flattened = "\n".join(" ".join(command) for command in commands)

    assert worker.INSTALL_PROFILE_ID == "isaaclab_arena_physx_task_runtime.v3"
    assert worker.ISAAC_LAB_INSTALL_TARGETS == (
        "assets",
        "ov",
        "physx",
        "rl[rsl-rl]",
        "tasks",
        "teleop",
    )
    assert "isaaclab.sh -i assets,ov,physx,rl[rsl-rl],tasks,teleop" in flattened
    assert worker.H5PY_VERSION == "3.16.0"
    assert worker.H5PY_LINUX_CP312_WHEEL_URL in flattened
    assert f"#sha256={worker.H5PY_LINUX_CP312_WHEEL_SHA256}" in flattened
    assert all(url in flattened for url in worker.HYDRA_RUNTIME_URLS)
    assert "hydra_core-1.3.5-py3-none-any.whl" in flattened
    assert "omegaconf-2.3.1-py3-none-any.whl" in flattened
    assert "antlr4-python3-runtime-4.9.3.tar.gz" in flattened
    assert all(url in flattened for url in worker.ARENA_REMOTE_TRANSPORT_URLS)
    assert "msgpack-1.1.0-cp312-cp312-manylinux" in flattened
    assert "pyzmq-27.0.1-cp312-abi3-manylinux" in flattened
    assert "import antlr4, h5py, hydra" in flattened
    assert "from isaaclab_ov.renderers import OVRTXRendererCfg" in flattened
    assert "from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper" in flattened
    assert "from rsl_rl.runners import DistillationRunner, OnPolicyRunner" in flattened
    assert f"pip install --editable {source}" in flattened
    assert "isaaclab_arena" in flattened
    assert "isaaclab_newton" in flattened
    assert "[dev]" not in flattened
    assert "isaaclab.sh -i\n" not in flattened
    assert "source/isaaclab*/" not in flattened
    assert "apt-get" not in flattened
    worker._validate_install_commands(commands)


def test_worker_rejects_expanded_install_profile() -> None:
    from blueprint_pipeline import adp009d_native_microcheck_worker as worker

    with pytest.raises(RuntimeError, match="adp009d_runtime_install_profile_expanded"):
        worker._validate_install_commands(
            [["/isaac-sim/python.sh", "-m", "pip", "install", "isaaclab_mimic"]]
            + worker._install_commands(Path("arena"))
        )


def test_worker_rejects_missing_h5py_wheel_pin() -> None:
    from blueprint_pipeline import adp009d_native_microcheck_worker as worker

    commands = [
        command
        for command in worker._install_commands(Path("arena"))
        if worker.H5PY_LINUX_CP312_WHEEL_URL not in command
    ]
    with pytest.raises(RuntimeError, match="adp009d_runtime_h5py_pin_missing"):
        worker._validate_install_commands(commands)


def test_worker_rejects_missing_hydra_runtime_pin() -> None:
    from blueprint_pipeline import adp009d_native_microcheck_worker as worker

    commands = [
        command
        for command in worker._install_commands(Path("arena"))
        if not any(url in command for url in worker.HYDRA_RUNTIME_URLS)
    ]
    with pytest.raises(RuntimeError, match="adp009d_runtime_hydra_pin_missing"):
        worker._validate_install_commands(commands)


def test_worker_rejects_missing_arena_transport_pin() -> None:
    from blueprint_pipeline import adp009d_native_microcheck_worker as worker

    commands = [
        command
        for command in worker._install_commands(Path("arena"))
        if not any(url in command for url in worker.ARENA_REMOTE_TRANSPORT_URLS)
    ]
    with pytest.raises(
        RuntimeError, match="adp009d_runtime_arena_transport_pin_missing"
    ):
        worker._validate_install_commands(commands)


def test_native_output_is_result_bearing_without_legacy_mp4_requirement(tmp_path: Path) -> None:
    output_zip = tmp_path / "provider-output.zip"
    with zipfile.ZipFile(output_zip, "w") as archive:
        archive.writestr(
            "adp009d_native_microcheck.json",
            json.dumps(
                {
                    "schema_version": "adp009d_native_microcheck.v1",
                    "status": "completed",
                }
            ),
        )

    expected_video_count = _provider_expected_video_count("adp009d_isaac")
    inspection = _inspect_provider_runtime_output_zip(
        output_zip,
        video_extract_dir=tmp_path / "videos",
        expected_video_count=expected_video_count,
    )

    assert expected_video_count == 0
    assert inspection["runtime_result_present"] is True
    assert inspection["runtime_result_status"] == "completed"
    assert inspection["mp4_validation"]["blockers"] == []


def test_native_transport_prefers_one_48gb_class_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict = {}

    def fake_run(**kwargs):
        observed.update(kwargs)
        return {"status": "dry_run_ready"}

    monkeypatch.setattr(franka_vast, "run_arena_native_control_vast", fake_run)

    result = franka_vast.run_adp009d_native_microcheck_vast(
        job_dir="job",
        prepared_bundle={"status": "ready"},
        paid_resource_admission_grant=None,
        execute=False,
    )

    assert result["status"] == "dry_run_ready"
    assert observed["min_gpu_ram_mb"] == 46_000
    assert observed["preferred_gpu_keywords"] == ("L40S", "RTX 6000 Ada", "RTX A6000")
    assert observed["provider_bundle_kind"] == "adp009d_isaac"


def _allocator_args(tmp_path: Path, *, execute: bool) -> list[str]:
    values = [
        "gpu-canary",
        "--probe-kind",
        PROBE_KIND,
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
        "adp009d-native",
        "--adp009d-approved-can",
        str(tmp_path / "can.usda"),
        "--adp009d-sage-collision",
        str(tmp_path / "sage.usd"),
        "--adp009d-harness-manifest",
        str(tmp_path / "harness.json"),
        "--adp-job-dir",
        str(tmp_path / "job"),
        "--adp-max-hourly-rate-usd",
        "1.0",
        "--adp-max-spend-usd",
        "4.0",
        "--adp-hard-ttl-seconds",
        "14400",
    ]
    if execute:
        values.append("--execute")
    return values


@pytest.mark.parametrize("execute", [False, True])
def test_allocator_routes_microcheck_only_through_canonical_grant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, execute: bool
) -> None:
    observed: dict = {}
    monkeypatch.setattr(
        allocator,
        "_control_plane_checkout_blockers",
        lambda: ([], {"orchestrator_source_commit": "a" * 40, "checkout_clean": True}),
    )
    monkeypatch.setattr(
        allocator,
        "build_native_microcheck_bundle",
        lambda **kwargs: {
            "status": "ready",
            "bundle_sha256": "sha256:" + "b" * 64,
            "input_digest": "sha256:" + "c" * 64,
        },
    )

    def fake_run(**kwargs):
        observed.update(kwargs)
        return {"status": "completed" if kwargs["execute"] else "dry_run_ready"}

    monkeypatch.setattr(allocator, "run_adp009d_native_microcheck_vast", fake_run)

    assert allocator.main(_allocator_args(tmp_path, execute=execute)) == 0
    assert observed["execute"] is execute
    assert isinstance(observed["paid_resource_admission_grant"], PaidResourceAdmissionGrant) is execute
    admission = json.loads((tmp_path / "admission.json").read_text())
    assert admission["probe_kind"] == PROBE_KIND
    assert admission["retry_cap"] == 0
    assert admission["candidate_policy_queried"] is False


def test_semantic_override_layer_is_digest_bound_and_used_at_every_spawn_site() -> None:
    """Semantics must come from one digest-bound runtime override, not literals."""

    from blueprint_pipeline.decision_evidence_contracts import canonical_digest

    layer = isaac_runtime.SEMANTIC_OVERRIDE_LAYER
    assert layer["sealed_source_usd_mutated"] is False
    # The in-container digest helper must agree with the repository contract so
    # a downstream composer can recompute it.
    assert isaac_runtime._canonical_digest(layer) == canonical_digest(layer)
    assert isaac_runtime._semantic_tags("robot") == [("class", "robot")]
    assert isaac_runtime._semantic_tags("approved_can") == [("class", "approved_can")]

    source = Path(isaac_runtime.__file__).read_text(encoding="utf-8")
    # No spawn site may reintroduce a hard-coded tag that bypasses the override.
    assert '"semantic_tags": _semantic_tags("approved_can")' in source
    assert "spawn.semantic_tags = _semantic_tags(\"robot\")" in source
    assert '[("class", "robot")]' not in source
    assert '[("class", "approved_can")]' not in source


def test_bundle_ships_the_approach_helper_next_to_the_runtime() -> None:
    """The runtime imports the helper as a flat sibling inside the bundle."""

    from blueprint_pipeline import adp009d_native_microcheck_bundle as bundle_module

    source = Path(bundle_module.__file__).read_text(encoding="utf-8")
    assert 'runtime / "adp009d_approach_capture.py"' in source
    assert 'runtime / "adp009d_isaac_runtime.py"' in source


def test_entrypoint_records_how_the_worker_died(tmp_path: Path) -> None:
    """A native abort and a Python error need different repairs.

    A co-resident policy server exhausting VRAM kills Isaac as SIGABRT or
    SIGKILL, which no Python except clause in the runtime can catch -- the
    runtime's own main only catches Exception.  The shell's exit status is the
    only place that distinction survives.
    """

    from blueprint_pipeline.adp009d_native_microcheck_bundle import ENTRYPOINT

    # The fallback must receive the exit status, not just the output directory.
    assert '"$OUT_DIR" "$runner_rc"' in ENTRYPOINT
    assert '"worker_exit_code"' in ENTRYPOINT
    assert '"worker_terminating_signal"' in ENTRYPOINT
    assert "adp009d_worker_terminated_by_signal" in ENTRYPOINT
    # Still emits the generic blocker so existing consumers keep working.
    assert "adp009d_worker_failed_without_runtime_result" in ENTRYPOINT


def test_entrypoint_signal_decoding_is_correct(tmp_path: Path) -> None:
    """Exercise the embedded decoder on the exit codes that actually occur."""

    import signal as signal_module

    def decode(code: int) -> str | None:
        number = code - 128 if code > 128 else None
        if number is None:
            return None
        try:
            return signal_module.Signals(number).name
        except ValueError:
            return None

    assert decode(134) == "SIGABRT"   # CUDA abort / assertion failure
    assert decode(139) == "SIGSEGV"   # native segfault
    assert decode(137) == "SIGKILL"   # OOM killer
    # An ordinary Python failure is not a signal.
    assert decode(1) is None
    assert decode(2) is None
    # Out-of-range values must not raise.
    assert decode(255) is None


def test_worker_environment_facts_are_captured_before_anything_can_fail(
    tmp_path: Path,
) -> None:
    """Three questions the off-worker design could not answer, answered on-worker."""

    from blueprint_pipeline.adp009d_native_microcheck_bundle import ENTRYPOINT
    from blueprint_pipeline.adp009d_worker_environment_facts import collect_facts

    # Captured first: a later failure must not erase them.
    facts_index = ENTRYPOINT.index("adp009d_worker_environment_facts.py")
    runner_index = ENTRYPOINT.index("adp_arena_provider_runner.py")
    assert facts_index < runner_index
    # And never allowed to fail the run itself.
    assert "|| true" in ENTRYPOINT[facts_index : facts_index + 200]

    facts = collect_facts()
    assert facts["schema_version"] == "adp009d_worker_environment_facts.v1"
    # The three decisions this exists to inform.
    assert "isaac_python_executable" in facts
    assert "torch_version" in facts or "torch_error" in facts
    assert "system_python3_executable" in facts or "system_python3_error" in facts
    # Environment values can carry credentials; only key names are recorded.
    assert all(isinstance(key, str) for key in facts["isaac_environment_keys"])


def test_provisioning_ships_only_when_a_candidate_is_bound(tmp_path: Path) -> None:
    """An unbound run must not carry a policy script it will not use."""

    from blueprint_pipeline.adp009d_native_microcheck_bundle import ENTRYPOINT

    # Guarded on the script existing, so an unbound bundle simply skips it.
    assert 'if [ -f "$RUNTIME_DIR/adp009d_policy_provisioning.sh" ]; then' in ENTRYPOINT
    # Non-fatal, and the exit code is retained rather than inferred from silence.
    assert "provisioning_exit_code" in ENTRYPOINT
    assert "adp009d_policy_provisioning.log" in ENTRYPOINT


def test_provisioning_never_depends_on_a_preserved_execute_bit() -> None:
    """extractall drops Unix permissions, so an -x test skips the script in silence.

    A live run shipped the script at mode 755 inside the archive, extracted it
    non-executable, and produced no provisioning log, no status file and no
    error -- the block simply did not run.
    """

    from blueprint_pipeline.adp009d_native_microcheck_bundle import ENTRYPOINT

    assert '[ -f "$RUNTIME_DIR/adp009d_policy_provisioning.sh" ]' in ENTRYPOINT
    assert '[ -x "$RUNTIME_DIR/adp009d_policy_provisioning.sh" ]' not in ENTRYPOINT
    assert 'bash "$RUNTIME_DIR/adp009d_policy_provisioning.sh"' in ENTRYPOINT
    # A bound candidate whose script is missing must be visible, not silent.
    assert '"provisioning_ran": false' in ENTRYPOINT
    assert '"provisioning_ran": true' in ENTRYPOINT


def test_aura_is_rendered_by_isaac_not_composited_afterward() -> None:
    """A 15 Hz closed loop cannot call a second renderer between steps.

    Post-compositing works for a static frame -- that is how the review montage
    was made -- but an episode queries the policy every 1/15 s, and the goal
    prompt rules a learned result invalid unless both cameras see the Aura
    background together with the moving arm and can in one time-synchronised
    frame.  So the appearance is a scene asset that Isaac's own RTX renders,
    the same omni.rtx the standalone OVRTX lane wraps.
    """

    from pathlib import Path as _Path

    from blueprint_pipeline import adp009d_isaac_runtime as runtime

    source = _Path(runtime.__file__).read_text(encoding="utf-8")

    assert 'AURA_PARTICLEFIELD_FILENAME = "aura_ghost_removed_surflets.usd"' in source
    assert "aura_appearance" in source
    # Added to the rendered scene, not to a separate compositing step.
    assert "assets=[sage, approved_can, light]" in source
    assert "[aura_appearance] if aura_appearance is not None else []" in source


def test_the_appearance_is_visual_only_and_never_a_collider() -> None:
    """SAGE stays the sole collision authority; appearance cannot change contact."""

    from pathlib import Path as _Path

    from blueprint_pipeline import adp009d_isaac_runtime as runtime

    source = _Path(runtime.__file__).read_text(encoding="utf-8")
    # Only the appearance's own construction: the approved can nearby is a
    # rigid body with real physics, and sweeping it in would prove nothing.
    block = source[source.index("aura_appearance = Object(") :]
    block = block[: block.index(")", block.index('"visible": True')) + 1]

    assert "ObjectType.BASE" in block
    assert '"visible": True' in block
    # Never rigid, never a collider, no physics API of any kind.
    assert "ObjectType.RIGID" not in block
    assert "collision" not in block.lower()
    assert "physics" not in block.lower()


def test_a_bundle_without_the_appearance_still_builds() -> None:
    """The micro-check must remain runnable without a policy or an appearance."""

    from pathlib import Path as _Path

    from blueprint_pipeline import adp009d_isaac_runtime as runtime

    source = _Path(runtime.__file__).read_text(encoding="utf-8")
    # Presence is checked, not assumed, so an appearance-free bundle runs.
    assert "if aura_particlefield_path.is_file():" in source
    assert "aura_appearance = None" in source



def test_the_receipt_never_claims_a_render_it_did_not_observe() -> None:
    """An earlier field said "rendered" while only checking a file existed.

    It reported True on a run whose frames were byte-for-byte comparable to a
    run with no appearance at all -- max difference 12 of 255, sampling noise.
    A receipt asserting a render it never saw is worse than one that is silent.
    """

    from pathlib import Path as _Path

    from blueprint_pipeline import adp009d_isaac_runtime as runtime

    source = _Path(runtime.__file__).read_text(encoding="utf-8")
    assert '"aura_particlefield_shipped"' in source
    assert '"aura_appearance_rendered_in_isaac"' not in source
    # The render claim exists but is explicitly unproven rather than assumed.
    assert '"aura_appearance_render_verified": None' in source


def test_gaussian_accumulation_is_requested_on_the_camera_render_products() -> None:
    """Surfels in the scene graph are not surfels in the image.

    The standalone OVRTX worker authors these settings on its own render
    product; Isaac Lab's cameras create theirs without them, which is why the
    ParticleField was present and invisible.
    """

    from pathlib import Path as _Path

    from blueprint_pipeline import adp009d_isaac_runtime as runtime

    source = _Path(runtime.__file__).read_text(encoding="utf-8")
    for setting in (
        "rtx/rtpt/gaussian/accumulatedDepth/enabled",
        "rtx/rtpt/gaussian/accumulatedAlbedo/enabled",
        "rtx/rtpt/gaussian/maxGaussiansToAccumulate",
    ):
        assert setting in source
    # Applied to the policy cameras, before the scene is built.
    assert source.index("accumulatedDepth") < source.index("sealed_scene_configuration")


def test_shipped_modules_import_in_the_flat_provider_layout() -> None:
    """They land flat in provider_runtime; a relative import fails there.

    A live run reached the episode and died on "attempted relative import with
    no known parent package" -- the same dual-layout problem already solved for
    the approach helper and not carried to the newer modules.
    """

    from pathlib import Path as _Path

    import blueprint_pipeline

    root = _Path(blueprint_pipeline.__file__).parent
    for name in (
        "adp009d_policy_episode",
        "adp009d_isaac_episode_adapter",
    ):
        source = (root / f"{name}.py").read_text(encoding="utf-8")
        assert "try:  # flat provider-bundle layout" in source
        assert "except ModuleNotFoundError:" in source


def test_every_module_the_episode_imports_is_shipped() -> None:
    """A dual-layout import cannot save a module that is simply absent.

    The episode imports its digest helper; omitting it from the bundle made the
    flat arm fail with ModuleNotFoundError and the fallback raise a different
    exception than the except clause names, so neither arm could succeed.
    """

    import ast
    from pathlib import Path as _Path

    import blueprint_pipeline

    root = _Path(blueprint_pipeline.__file__).parent
    bundle_source = (root / "adp009d_native_microcheck_bundle.py").read_text(
        encoding="utf-8"
    )

    for module in ("adp009d_policy_episode", "adp009d_isaac_episode_adapter"):
        tree = ast.parse((root / f"{module}.py").read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
                assert f'"{node.module}.py"' in bundle_source, (
                    f"{module} imports {node.module}, which the bundle does not ship"
                )


def test_the_particlefield_declares_a_default_prim(tmp_path) -> None:
    """Without one, a USD reference resolves to nothing.

    Arena brings this asset in with Object(usd_path=...), a reference.  A live
    run added it and produced frames byte-comparable to a run with no
    appearance -- max difference 12 of 255 -- because nothing composed, which
    is also why authoring gaussian accumulation settings changed nothing.  Both
    assets that do compose here carry one: sage_task_collision has /Root and
    the approved can has /canned_beverage.
    """

    import inspect

    from blueprint_pipeline import particlefield_usd

    source = inspect.getsource(particlefield_usd.write_gaussian_surflet_particlefield_usd)
    assert "stage.SetDefaultPrim(world.GetPrim())" in source
    # Set before the field is defined, so the file is never written without one.
    assert source.index("SetDefaultPrim") < source.index("ParticleField.Define")
    # And recorded, so an asset lacking one is visible in its own receipt.
    assert '"default_prim"' in inspect.getsource(particlefield_usd)


def test_the_runtime_probes_the_live_stage_for_the_appearance_prim() -> None:
    """One run must yield both the fix and the confirmation of why it failed."""

    from pathlib import Path as _Path

    from blueprint_pipeline import adp009d_isaac_runtime as runtime

    source = _Path(runtime.__file__).read_text(encoding="utf-8")
    assert '"aura_stage_probe"' in source
    assert "matching_prim_paths" in source
    assert "GetAppliedSchemas()" in source
    # A diagnostic must never fail the run it is diagnosing.
    probe = source[source.index("aura_stage_probe: dict") :]
    probe = probe[: probe.index("live_collider = ")]
    assert "except Exception" in probe


def _download_script(kind: str) -> str:
    from blueprint_pipeline.vast_provider_adapter import _probe_shell_script

    return _probe_shell_script(
        heartbeat_url="https://example.invalid/heartbeat",
        enable_blueprint_bundle=True,
        provider_bundle_kind=kind,
    )


def test_every_bundle_kind_downloads_over_http11_with_retries() -> None:
    """A stream reset is a transport property, not a property of the zip.

    This was an allowlist of the four kinds observed to fail.  A kind absent
    from it died with `curl: (92) HTTP/2 stream 1 was not closed cleanly`
    after transferring zero bytes -- exactly what the flag prevents.
    """

    from blueprint_pipeline.vast_provider_adapter import VAST_PROVIDER_BUNDLE_KINDS

    for kind in VAST_PROVIDER_BUNDLE_KINDS:
        script = _download_script(kind)
        assert "--http1.1" in script, kind
        # --retry alone covers transient HTTP statuses and timeouts, not a
        # transport error like 92, so it would not have retried this failure.
        assert "--retry-all-errors" in script, kind


def test_a_failing_download_tool_falls_through_to_the_next_one() -> None:
    """The chain handled a tool being absent, not a tool failing.

    ``curl ...; return $?`` ended the download on the first curl error with
    wget and python still available and untried.
    """

    script = _download_script("isaac")
    assert 'curl' in script and '-o "$blueprint_download_dst" && return 0' in script
    assert 'wget -O "$blueprint_download_dst" "$blueprint_download_src" && return 0' in script
    assert "; return $?; fi; " not in script.split("blueprint_download_src")[1][:400]
    # And the fallthrough is visible in the log rather than silent.
    assert "BLUEPRINT_VAST_DOWNLOAD_TRANSPORT_FAILED:curl" in script
