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
        """#usda 1.0
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
""",
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
            "}\n"
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
        str(
            Path(first["bundle_path"]).parent
            / "provider_runtime/assets/sage_collision_overlay.usda"
        )
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
    can_collider = adapter_stage.GetPrimAtPath("/canned_beverage/colliders/body_collider")
    api_schemas = can_collider.GetMetadata("apiSchemas")
    assert "PhysxSDFMeshCollisionAPI" in list(api_schemas.GetAddedOrExplicitItems())
    assert can_collider.GetAttribute("physics:approximation").Get() == "sdf"
    assert can_collider.GetAttribute("physxSDFMeshCollision:sdfResolution").Get() == 256
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
        return abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) * 0.5

    assert sum(area(value) for value in leaves) == pytest.approx(6.0)
    assert (
        max(
            sum((value[edge][axis] - value[(edge + 1) % 3][axis]) ** 2 for axis in range(3)) ** 0.5
            for value in leaves
            for edge in range(3)
        )
        <= 0.5 + 1.0e-9
    )


def test_task_collision_aabb_clip_preserves_only_exact_coplanar_surface() -> None:
    triangle = ((-2.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 2.0, 0.0))

    clipped = _clip_triangle_to_aabb(
        triangle,
        roi_min=(-1.0, -1.0, -1.0),
        roi_max=(1.0, 1.0, 1.0),
    )

    assert clipped
    assert all(
        -1.0 <= coordinate <= 1.0 for face in clipped for point in face for coordinate in point
    )
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
    prim.CreateAttribute(
        "physics:approximation", pytest.importorskip("pxr.Sdf").ValueTypeNames.Token
    ).Set("sdf")

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

    with pytest.raises(RuntimeError, match="physx_collision_stability_warning_detected"):
        isaac_runtime._fail_on_physx_collision_stability([message])


def test_runtime_uses_documented_legacy_cooker_after_measured_ujitso_stall() -> None:
    source = Path(isaac_runtime.__file__).read_text(encoding="utf-8")

    assert isaac_runtime.PHYSX_COLLISION_COOKING_PROFILE == "legacy_cooker_after_ujitso_stall.v1"
    assert "SETTING_UJITSO_COLLISION_COOKING" in source
    assert "settings.set_bool(key, False)" in source
    assert '"ujitso_resolved_enabled": bool(resolved_enabled)' in source
    assert '"collider_geometry_or_parameters_changed": False' in source
    assert '"measured_ujitso_environment_construction_stall_v14"' in source
    assert '_phase("physx_collision_cooking_configuration")' in source
    assert '_phase("physx_collision_cooking_configuration", "completed")' in source


def test_runtime_does_not_import_unneeded_arena_asset_registry() -> None:
    source = Path(isaac_runtime.__file__).read_text(encoding="utf-8")

    assert "isaaclab_arena.assets.object_library" not in source
    assert "class SpawnerObject(Object):" in source
    assert "object_type=ObjectType.SPAWNER" in source


def test_runtime_preflights_exact_arena_environment_import_closure() -> None:
    source = Path(isaac_runtime.__file__).read_text(encoding="utf-8")

    assert "def _preflight_environment_imports() -> dict[str, str]:" in source
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
    error = isaac_runtime.CanonicalPoseError("canonical_reset_arm_pose_mismatch", diagnostics)

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
    with pytest.raises(RuntimeError, match="adp009d_runtime_arena_transport_pin_missing"):
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
    assert observed["forward_hf_token"] is False
    assert observed["allowed_active_instance_ids"] == ()
    assert observed["candidate_policy_query_expected"] is False

    franka_vast.run_adp009d_native_microcheck_vast(
        job_dir="job",
        prepared_bundle={"status": "ready"},
        paid_resource_admission_grant=None,
        execute=False,
        authorize_gated_backbone=True,
        allowed_active_instance_ids=(47190772,),
    )
    assert observed["forward_hf_token"] is True
    assert observed["allowed_active_instance_ids"] == (47190772,)

    franka_vast.run_adp009d_native_microcheck_vast(
        job_dir="job",
        prepared_bundle={
            "status": "ready",
            "policy_candidate_id": "groot_n17_droid",
        },
        paid_resource_admission_grant=None,
        execute=False,
    )
    assert observed["candidate_policy_query_expected"] is True


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
    assert (
        isinstance(observed["paid_resource_admission_grant"], PaidResourceAdmissionGrant) is execute
    )
    admission = json.loads((tmp_path / "admission.json").read_text())
    assert admission["probe_kind"] == PROBE_KIND
    assert admission["retry_cap"] == 0
    assert admission["candidate_policy_queried"] is False


def test_allocator_requires_and_binds_explicit_gated_backbone_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
    monkeypatch.setattr(allocator, "normalize_model_access_env", lambda: None)
    monkeypatch.setattr(
        allocator,
        "model_access_secret_status",
        lambda: {"huggingface": {"auth_ready": True}},
    )
    access = {
        "status": "authorized",
        "receipt_digest": "sha256:" + "d" * 64,
        "blockers": [],
        "raw_secret_recorded": False,
    }
    monkeypatch.setattr(
        allocator,
        "probe_gated_backbone_access",
        lambda: access,
    )

    def fake_run(**kwargs):
        observed.update(kwargs)
        return {"status": "dry_run_ready"}

    monkeypatch.setattr(allocator, "run_adp009d_native_microcheck_vast", fake_run)
    args = _allocator_args(tmp_path, execute=False) + [
        "--adp009d-policy-candidate",
        "groot_n17_droid",
        "--adp009d-authorize-gated-backbone",
    ]

    assert allocator.main(args) == 0
    assert observed["authorize_gated_backbone"] is True
    admission = json.loads((tmp_path / "admission.json").read_text())
    assert admission["gated_backbone_access"] == access
    assert admission["allocation_binding"]["gated_backbone_authorized"] is True
    assert (
        admission["allocation_binding"]["gated_backbone_access_receipt_digest"]
        == access["receipt_digest"]
    )


def test_allocator_binds_concurrent_instance_authority_through_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
        return {"status": "dry_run_ready"}

    monkeypatch.setattr(allocator, "run_adp009d_native_microcheck_vast", fake_run)
    args = _allocator_args(tmp_path, execute=False) + [
        "--adp-allowed-active-vast-instance-id",
        "47190772",
    ]

    assert allocator.main(args) == 0
    assert observed["allowed_active_instance_ids"] == [47190772]
    admission = json.loads((tmp_path / "admission.json").read_text())
    assert admission["explicit_concurrent_gpu_authority_bound"] is True
    assert admission["allocation_binding"]["allowed_active_vast_instance_ids"] == [47190772]


def test_allocator_abstains_before_paid_mutation_without_gated_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        allocator,
        "_control_plane_checkout_blockers",
        lambda: ([], {"orchestrator_source_commit": "a" * 40, "checkout_clean": True}),
    )
    called = False

    def fake_run(**_kwargs):
        nonlocal called
        called = True
        return {"status": "dry_run_ready"}

    monkeypatch.setattr(allocator, "run_adp009d_native_microcheck_vast", fake_run)
    args = _allocator_args(tmp_path, execute=False) + [
        "--adp009d-policy-candidate",
        "groot_n17_droid",
    ]

    assert allocator.main(args) == 2
    assert called is False
    result = json.loads((tmp_path / "adapter.json").read_text())
    assert result["blockers"] == ["adp009d_gated_backbone_authority_missing"]


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
    assert 'spawn.semantic_tags = _semantic_tags("robot")' in source
    assert '[("class", "robot")]' not in source
    assert '[("class", "approved_can")]' not in source


def test_bundle_ships_the_approach_helper_next_to_the_runtime() -> None:
    """The runtime imports the helper as a flat sibling inside the bundle."""

    from blueprint_pipeline import adp009d_native_microcheck_bundle as bundle_module

    source = Path(bundle_module.__file__).read_text(encoding="utf-8")
    assert 'runtime / "adp009d_approach_capture.py"' in source
    assert 'runtime / "adp009d_isaac_runtime.py"' in source
    assert '"adp009d_gated_backbone.py"' in source


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


def test_entrypoint_skips_arena_after_only_policy_fails_provisioning() -> None:
    """A known single-policy blocker must not spend time installing Arena."""

    from blueprint_pipeline.adp009d_native_microcheck_bundle import ENTRYPOINT

    fail_fast = ENTRYPOINT.index("adp009d_single_candidate_policy_provisioning_failed")
    arena_runner = ENTRYPOINT.index('"$RUNTIME_DIR/adp_arena_provider_runner.py"')
    assert fail_fast < arena_runner
    assert 'candidate_count" = "1"' in ENTRYPOINT
    assert '"$provisioning_worst_rc" -ne 0' in ENTRYPOINT
    assert '"arena_setup_skipped": True' in ENTRYPOINT

    import subprocess

    count_start = ENTRYPOINT.index("candidate_count=")
    count_end = ENTRYPOINT.index("\nif [", count_start)
    count_command = ENTRYPOINT[count_start:count_end]
    for value, expected in (("groot_n17_droid", "1"), ("a,b", "2"), ("", "0")):
        completed = subprocess.run(
            [
                "bash",
                "-c",
                f'provisioning_candidates="$1"\n{count_command}\nprintf "%s" "$candidate_count"',
                "candidate-count-test",
                value,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        assert completed.stdout == expected


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

    assert decode(134) == "SIGABRT"  # CUDA abort / assertion failure
    assert decode(139) == "SIGSEGV"  # native segfault
    assert decode(137) == "SIGKILL"  # OOM killer
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
    # Per candidate now: ranking two policies needs both provisioned on one host.
    assert 'script="$RUNTIME_DIR/adp009d_policy_provisioning.$candidate.sh"' in ENTRYPOINT
    # Non-fatal, and the exit code is retained rather than inferred from silence.
    assert "provisioning_exit_code" in ENTRYPOINT
    # Per candidate, so one policy's provisioning log cannot overwrite
    # the other's -- which would erase the evidence for one arm of the
    # comparison.
    assert "adp009d_policy_provisioning.$candidate.log" in ENTRYPOINT


def test_provisioning_never_depends_on_a_preserved_execute_bit() -> None:
    """extractall drops Unix permissions, so an -x test skips the script in silence.

    A live run shipped the script at mode 755 inside the archive, extracted it
    non-executable, and produced no provisioning log, no status file and no
    error -- the block simply did not run.
    """

    from blueprint_pipeline.adp009d_native_microcheck_bundle import ENTRYPOINT

    # Tested with -f and invoked through bash, never relying on the execute
    # bit: zipfile.extractall does not preserve Unix permissions.
    assert '[ -f "$script" ] || continue' in ENTRYPOINT
    assert '[ -x "$script" ]' not in ENTRYPOINT
    assert 'bash "$script"' in ENTRYPOINT
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
    # Resolved rather than name-matched, so a NuRec .usdz and a
    # ParticleField .usd are both admissible on the same code path.
    assert "aura_particlefield_path, aura_appearance_format = _resolve_aura_appearance" in source
    assert "if aura_particlefield_path is not None:" in source
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
    assert '"aura_appearance_shipped"' in source
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
    bundle_source = (root / "adp009d_native_microcheck_bundle.py").read_text(encoding="utf-8")

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
    assert "curl" in script and '-o "$blueprint_download_dst" && return 0' in script
    assert 'wget -O "$blueprint_download_dst" "$blueprint_download_src" && return 0' in script
    assert "; return $?; fi; " not in script.split("blueprint_download_src")[1][:400]
    # And the fallthrough is visible in the log rather than silent.
    assert "BLUEPRINT_VAST_DOWNLOAD_TRANSPORT_FAILED:curl" in script


def _liveness_rows(status: str) -> dict:
    return {"instances": [{"id": 4711, "actual_status": status, "cur_state": "running"}]}


def test_a_dead_container_is_detected_by_asking_the_provider(monkeypatch) -> None:
    """A frozen log cannot distinguish a dead container from a slow one.

    A live run polled a container that had exited mid-Isaac-startup for
    twenty-six minutes while the API reported ``actual_status: exited``
    throughout, because the poller only ever asked for logs.
    """

    from blueprint_pipeline import vast_provider_adapter as adapter

    monkeypatch.setattr(adapter, "_api_json", lambda **kw: (200, _liveness_rows("exited")))
    result = adapter._instance_liveness(instance_id=4711, api_key="k")
    assert result["exited"] is True
    assert result["status"] == "exited"


def test_a_probe_error_is_not_evidence_of_death(monkeypatch) -> None:
    """Only a positive reading counts, so a blip cannot kill a healthy run."""

    from blueprint_pipeline import vast_provider_adapter as adapter

    def boom(**_kw):
        raise TimeoutError("network")

    monkeypatch.setattr(adapter, "_api_json", boom)
    result = adapter._instance_liveness(instance_id=4711, api_key="k")
    assert result["exited"] is not True
    assert result["observed"] is False
    assert "TimeoutError" in result["probe_error"]


def test_a_running_container_is_not_reported_dead(monkeypatch) -> None:
    from blueprint_pipeline import vast_provider_adapter as adapter

    monkeypatch.setattr(adapter, "_api_json", lambda **kw: (200, _liveness_rows("running")))
    assert adapter._instance_liveness(instance_id=4711, api_key="k")["exited"] is False


def test_an_instance_destroyed_out_from_under_the_run_counts_as_dead(monkeypatch) -> None:
    from blueprint_pipeline import vast_provider_adapter as adapter

    monkeypatch.setattr(adapter, "_api_json", lambda **kw: (200, {"instances": []}))
    result = adapter._instance_liveness(instance_id=4711, api_key="k")
    assert result["exited"] is True
    assert result["status"] == "absent"


def test_the_exit_is_named_rather_than_blamed_on_absent_log_progress() -> None:
    """Reporting the symptom sent one run chasing a render bug.

    The break must be checked ahead of the generic watchdogs, and two
    consecutive readings are required so one API glitch cannot kill a run.
    """

    from pathlib import Path as _Path

    from blueprint_pipeline import vast_provider_adapter as adapter

    source = _Path(adapter.__file__).read_text(encoding="utf-8")
    assert 'break_reason = "instance_exited"' in source
    assert "instance_exited_count >= 2" in source
    # Ahead of the no-progress timeout in the break chain.
    assert source.index('break_reason = "instance_exited"') < source.index(
        'break_reason = "no_log_progress_timeout"'
    )
    # And it blacklists the machine, like the other startup-plane failures.
    assert '"vast_heartbeat_instance_exited",' in source
    blockers = source[source.index("startup_control_plane_blocked = any(") :]
    assert "vast_heartbeat_instance_exited" in blockers[:600]


def test_the_first_render_step_is_visible_in_the_phase_log() -> None:
    """It pays the whole cost of composing the scene's appearance.

    Without a marker of its own, a live run stuck there reported
    ``reset_1:completed`` -- several phases earlier -- leaving "stuck in the
    first render" indistinguishable from "stuck comparing two tensors".
    """

    from pathlib import Path as _Path

    from blueprint_pipeline import adp009d_isaac_runtime as runtime

    source = _Path(runtime.__file__).read_text(encoding="utf-8")
    assert '_phase("zero_action_step")' in source
    assert '_phase("zero_action_step", "completed")' in source
    # Emitted before the step, or it cannot report a hang inside it.
    assert source.index('_phase("zero_action_step")') < source.index("lambda: env.step(action)")
    # And the warmup loop announces itself before its first tenth-frame marker.
    assert '_phase("camera_warmup")' in source


def test_the_first_render_runs_under_a_hard_budget() -> None:
    """A wedged renderer must name itself rather than burn the whole TTL.

    A live run sat in the first step for over twenty minutes emitting an
    omni.usd "failed to wait for idle" every seventy seconds.
    """

    from pathlib import Path as _Path

    from blueprint_pipeline import adp009d_isaac_runtime as runtime

    source = _Path(runtime.__file__).read_text(encoding="utf-8")
    assert "_run_under_render_budget(" in source
    assert "first_render_budget_exceeded" in source
    # Isaac blocks in native code, so only a hard exit from the watchdog thread
    # can get the diagnosis out; a normal exit runs shutdown handlers that are
    # themselves blocked on the same idle wait.
    assert "os._exit(93)" in source
    # The diagnosis must carry the stage probe, or it names a symptom only.
    assert '"aura_stage_probe": aura_stage_probe,' in source


def test_the_render_budget_lets_a_fast_render_through() -> None:
    from blueprint_pipeline import adp009d_isaac_runtime as runtime

    result = runtime._run_under_render_budget(
        lambda: "rendered", phase_name="zero_action_step", diagnostics={}
    )
    assert result == "rendered"


def test_the_render_budget_is_overridable_for_a_slow_scene(monkeypatch) -> None:
    from blueprint_pipeline import adp009d_isaac_runtime as runtime

    monkeypatch.setenv(runtime.FIRST_RENDER_BUDGET_SECONDS_ENV, "900")
    assert runtime._run_under_render_budget(lambda: 1, phase_name="p", diagnostics={}) == 1


def test_camera_warmup_is_configurable_but_never_below_settling(monkeypatch) -> None:
    """Forty frames costs forty-three minutes once appearance is composed.

    Each frame waits the full omni.usd idle timeout, so the warmup alone ran
    past the paid TTL and the run ended having saved no frame at all.
    """

    from blueprint_pipeline import adp009d_isaac_runtime as runtime

    monkeypatch.delenv(runtime.CAMERA_WARMUP_FRAMES_ENV, raising=False)
    assert runtime._camera_warmup_frames() == runtime.DEFAULT_CAMERA_WARMUP_FRAMES

    # Above the floor, honoured as asked.
    monkeypatch.setenv(runtime.CAMERA_WARMUP_FRAMES_ENV, "60")
    assert runtime._camera_warmup_frames() == 60

    # A frame saved from an unsettled camera is worse than a slow run: it
    # looks like data.
    # Below it, clamped: four frames rendered mean 0.2 / max 1.
    monkeypatch.setenv(runtime.CAMERA_WARMUP_FRAMES_ENV, "4")
    assert runtime._camera_warmup_frames() == runtime.MIN_CAMERA_WARMUP_FRAMES
    assert runtime.MIN_CAMERA_WARMUP_FRAMES == 40
    monkeypatch.setenv(runtime.CAMERA_WARMUP_FRAMES_ENV, "not-a-number")
    assert runtime._camera_warmup_frames() == runtime.DEFAULT_CAMERA_WARMUP_FRAMES


def test_the_saved_frame_index_follows_the_actual_warmup() -> None:
    """It was a constant 40 that merely happened to match the loop."""

    from pathlib import Path as _Path

    from blueprint_pipeline import adp009d_isaac_runtime as runtime

    source = _Path(runtime.__file__).read_text(encoding="utf-8")
    assert "frame_index=warmup_frames," in source
    assert "frame_index=40," not in source


def test_frames_only_mode_exits_so_the_frames_actually_get_uploaded() -> None:
    """Frames are zipped only after the runtime exits.

    At roughly a minute per rendered frame the phases after the camera saves
    -- a four-hundred-step approach, a four-hundred-and-eighty step episode --
    run for hours, and the TTL kills the instance before anything uploads.  A
    run that never exits delivers nothing, however many frames it rendered.
    """

    from pathlib import Path as _Path

    from blueprint_pipeline import adp009d_isaac_runtime as runtime

    source = _Path(runtime.__file__).read_text(encoding="utf-8")
    assert "STOP_AFTER_FRAMES_ENV" in source
    # Placed after the camera saves and before the gripper probe, or it stops
    # before there is anything worth uploading.
    assert source.index('STOP_AFTER_FRAMES_ENV, ""') > source.index("camera_rows.append(")
    assert source.index('STOP_AFTER_FRAMES_ENV, ""') < source.index("--- gripper convention probe")
    # And it must never read as a passing micro-check.
    assert '"supports_microcheck_success_claim": False,' in source


def test_the_tuning_vars_reach_the_worker() -> None:
    """An env var the bundle never exports is a setting that does nothing."""

    from blueprint_pipeline import adp009d_native_microcheck_bundle as bundle

    assert 'BLUEPRINT_ADP009D_CAMERA_WARMUP_FRAMES="@@CAMERA_WARMUP_FRAMES@@"' in bundle.ENTRYPOINT
    assert 'BLUEPRINT_ADP009D_STOP_AFTER_FRAMES="@@STOP_AFTER_FRAMES@@"' in bundle.ENTRYPOINT
    assert 'BLUEPRINT_ADP009D_CAMERA_RESOLUTION="@@CAMERA_RESOLUTION@@"' in bundle.ENTRYPOINT
    import inspect

    source = inspect.getsource(bundle)
    # Substituted, not merely declared: an unreplaced @@ placeholder would
    # export the literal string and read as truthy.
    assert '"@@CAMERA_WARMUP_FRAMES@@",' in source
    assert '"@@STOP_AFTER_FRAMES@@",' in source
    assert '"@@CAMERA_RESOLUTION@@", camera_resolution' in source


def test_policy_bundle_binds_policy_resolution_without_host_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The remote worker does not inherit an export from the launch shell."""

    monkeypatch.delenv("BLUEPRINT_ADP009D_CAMERA_RESOLUTION", raising=False)
    approved, sage, harness, bindings = _inputs(tmp_path)
    receipt = build_native_microcheck_bundle(
        job_dir=tmp_path / "policy-resolution",
        approved_can_path=approved,
        sage_collision_path=sage,
        harness_manifest_path=harness,
        implementation_commit="e" * 40,
        policy_candidate_id="pi05_droid",
        generated_at="fixed",
        expected_asset_bindings=bindings,
    )

    with zipfile.ZipFile(receipt["bundle_path"]) as archive:
        entrypoint = archive.read("provider_runtime/run_adp_arena_provider_runtime.sh").decode()
        manifest = json.loads(archive.read("provider_runtime/adp_arena_provider_manifest.json"))
    assert 'export BLUEPRINT_ADP009D_CAMERA_RESOLUTION="policy"' in entrypoint
    assert manifest["camera_resolution_binding"] == "policy"


def test_frames_only_bundle_skips_policy_provisioning() -> None:
    """A frame diagnostic cannot consume or score policy output."""

    from blueprint_pipeline import adp009d_native_microcheck_bundle as bundle

    entrypoint = bundle.ENTRYPOINT
    skip = entrypoint.index("policy_provisioning_skipped_frames_only")
    provision_loop = entrypoint.index("for candidate in $(printf '%s' \"$provisioning_candidates\"")
    runner = entrypoint.index('/isaac-sim/python.sh "$RUNTIME_DIR/adp_arena_provider_runner.py"')
    assert skip < provision_loop < runner
    assert 'provisioning_candidates=""' in entrypoint[:provision_loop]
    assert '"skip_reason": "frames_only_diagnostic"' in entrypoint


def test_a_degenerate_frame_is_refused_rather_than_saved() -> None:
    """A frame this dark is not a dark scene.

    Named for the symptom, not a cause: forty warmup frames produced max 1 /
    mean 0.167 where four produced max 1 / mean 0.2, which disproved the
    starvation the guard was first named for.  A run saved one of these and
    reported success; saving one produces evidence that looks like data.
    """

    from pathlib import Path as _Path

    from blueprint_pipeline import adp009d_isaac_runtime as runtime

    source = _Path(runtime.__file__).read_text(encoding="utf-8")
    assert "camera_frame_degenerate" in source
    assert "FRAME_DEGENERATE_MAX_VALUE" in source
    # Checked before the frame is written, not after.
    save = source[source.index("def _save_camera(") :]
    assert save.index("camera_frame_degenerate") < save.index("Image.fromarray")
    # Far below any real render: v43's blank-but-converged frame was mean 227,
    # so this cannot reject a legitimately dim scene.
    assert runtime.FRAME_DEGENERATE_MAX_VALUE <= 2


def test_the_stage_probe_inspects_the_field_not_its_parent() -> None:
    """Reading matches[0] returned the Xform holding the field.

    That reported an empty applied-schema list for a prim that carried all
    nine, which reads as a broken asset when the asset was fine.
    """

    from pathlib import Path as _Path

    from blueprint_pipeline import adp009d_isaac_runtime as runtime

    source = _Path(runtime.__file__).read_text(encoding="utf-8")
    assert 'm.endswith("GaussianSurflets")' in source
    # And it records which prim it actually read, so the answer is checkable.
    assert '"inspected_prim_path"' in source


def _surfel_arrays(count: int = 8):
    import numpy as np

    from blueprint_pipeline.particlefield_usd import (
        GaussianSurfelData,
        build_gaussian_surflet_arrays,
    )

    rng = np.random.default_rng(20260807)
    # log-scales around exp(-7) ~ 0.9mm, the real field's median planar extent.
    return build_gaussian_surflet_arrays(
        GaussianSurfelData(
            count=count,
            xyz=rng.normal(size=(count, 3)).astype("float32"),
            opacity=np.full(count, 3.0, dtype="float32"),
            f_dc=rng.normal(size=(count, 3)).astype("float32"),
            scales=np.full((count, 2), -7.0, dtype="float32"),
            quats=np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype="float32"), (count, 1)),
            sh_rest=rng.normal(size=(count, 3 * 15)).astype("float32"),
            mask_logits=np.zeros((count, 3), dtype="float32"),
            properties=(),
        )
    )


def test_the_structural_z_extent_is_flat_not_one_metre() -> None:
    """It was authored as 1.0 because a structural component is "unused".

    That is multiplicative-identity thinking applied to an extent in metres,
    where the neutral value is zero.  Against a median learned extent of
    0.8mm it made every surfel a one-metre needle, 1237x thicker than wide,
    and 414k of them at mean opacity 0.90 put 47m^3 of opaque geometry inside
    a 117m^3 room with the camera in it.  Every frame came back max 1 of 255.
    """

    import numpy as np

    arrays = _surfel_arrays()
    scales = arrays["scales"]
    planar = scales[:, :2]
    z = scales[:, 2]
    assert (z < planar.min(axis=1)).all(), "z must be flatter than the surfel is wide"
    assert not np.isclose(z, 1.0).any()
    # Proportional, not a constant: a fixed epsilon would be thicker than wide
    # for the smallest surfels in the real field.
    assert np.allclose(z / planar.min(axis=1), z[0] / planar.min(axis=1)[0])


def test_the_receipt_reports_radiance_outside_display_range() -> None:
    """The coefficients are sealed learned data, so this reports and never clamps.

    Thirty percent of the real field's DC terms decode above 1.0, peaking at
    4.6x overbright; a reader has to be able to see that.
    """

    arrays = _surfel_arrays()
    assert "sh_dc_out_of_display_range_fraction" in arrays
    assert 0.0 <= arrays["sh_dc_out_of_display_range_fraction"] <= 1.0
    assert "sh_dc_radiance_max" in arrays
    # Never rescaled: the DC coefficients must survive untouched.
    assert "structural_z_scale_fraction" in arrays


def test_the_written_receipt_carries_the_scale_and_radiance_facts(tmp_path) -> None:
    """The array builder and the receipt drifted apart.

    Renaming the structural-Z key left the receipt reading the old name, and
    the array-level test could not see it because it never wrote a file.
    """

    import numpy as np

    from blueprint_pipeline.particlefield_usd import (
        GaussianSurfelData,
        write_gaussian_surflet_particlefield_usd,
    )

    count = 8
    rng = np.random.default_rng(20260807)
    data = GaussianSurfelData(
        count=count,
        xyz=rng.normal(size=(count, 3)).astype("float32"),
        opacity=np.full(count, 3.0, dtype="float32"),
        f_dc=rng.normal(size=(count, 3)).astype("float32"),
        scales=np.full((count, 2), -7.0, dtype="float32"),
        quats=np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype="float32"), (count, 1)),
        sh_rest=rng.normal(size=(count, 3 * 15)).astype("float32"),
        mask_logits=np.zeros((count, 3), dtype="float32"),
        properties=(),
    )
    receipt = write_gaussian_surflet_particlefield_usd(
        data, tmp_path / "surflets.usd", receipt_path=tmp_path / "receipt.json"
    )
    if receipt["status"] == "blocked":
        assert receipt["blockers"] == ["usd_core_gaussian_surflet_schema_unavailable"]
        return
    assert receipt["status"] == "completed", receipt
    assert receipt["structural_z_scale_fraction"] > 0
    assert receipt["structural_z_scale_median_m"] < 0.001
    assert "sh_dc_out_of_display_range_fraction" in receipt
    assert receipt["default_prim"] == "/World"


def test_the_gaussian_accumulation_cap_is_sweepable(monkeypatch) -> None:
    """Forty-eight cannot build a surface from 0.81mm surfels in a 9.9m room.

    The cap was never actually exercised: the run that "tested" these settings
    used an asset with no default prim, so the reference resolved to nothing
    and the conclusion that they changed nothing was drawn against an empty
    scene.  The first frames to genuinely render the field came back as
    isolated speckles at 16% pixel coverage.
    """

    from blueprint_pipeline import adp009d_isaac_runtime as runtime

    monkeypatch.delenv(runtime.MAX_GAUSSIANS_TO_ACCUMULATE_ENV, raising=False)
    assert runtime._max_gaussians_to_accumulate() > 48
    monkeypatch.setenv(runtime.MAX_GAUSSIANS_TO_ACCUMULATE_ENV, "256")
    assert runtime._max_gaussians_to_accumulate() == 256
    monkeypatch.setenv(runtime.MAX_GAUSSIANS_TO_ACCUMULATE_ENV, "nonsense")
    assert runtime._max_gaussians_to_accumulate() == runtime.DEFAULT_MAX_GAUSSIANS_TO_ACCUMULATE


def test_the_cap_is_read_at_use_not_frozen_as_a_literal() -> None:
    """A literal here is not sweepable, which is the whole point."""

    from pathlib import Path as _Path

    from blueprint_pipeline import adp009d_isaac_runtime as runtime

    source = _Path(runtime.__file__).read_text(encoding="utf-8")
    assert '"rtx/rtpt/gaussian/maxGaussiansToAccumulate", _max_gaussians_to_accumulate()' in source
    assert '"rtx/rtpt/gaussian/maxGaussiansToAccumulate", 48' not in source


def test_frames_only_returns_a_result_rather_than_none() -> None:
    """_run is declared to return a dict and main reads result["status"].

    A bare return handed back None and the caller died on it *after* the
    frames were already saved, turning a successful diagnostic into an opaque
    "'NoneType' object has no attribute 'get'".
    """

    from pathlib import Path as _Path

    from blueprint_pipeline import adp009d_isaac_runtime as runtime

    source = _Path(runtime.__file__).read_text(encoding="utf-8")
    block = source[source.index('STOP_AFTER_FRAMES_ENV, ""') :]
    block = block[: block.index("--- gripper convention probe")]
    assert "return {" in block
    assert "\n            return\n" not in block, "bare return returns None"
    # Blocked, never completed: it skipped every phase after the frames, so it
    # must not exit zero or read as a passing micro-check downstream.
    assert '"status": "blocked",' in block
    assert '"blockers": ["stopped_after_frames_diagnostic_mode"],' in block
    assert '"supports_microcheck_success_claim": False,' in block
    # And it reports the swept value, or a sweep cannot be attributed.
    assert '"max_gaussians_to_accumulate"' in block


def test_the_appearance_format_is_resolved_not_assumed(tmp_path) -> None:
    """NuRec first: Isaac renders that format natively; ParticleField never has.

    A fixed .usd filename would rename a NuRec .usdz into something Isaac
    opens as a flat layer, and the appearance format is the whole question
    this lane is deciding.
    """

    from blueprint_pipeline import adp009d_isaac_runtime as runtime

    assets = tmp_path / "assets"
    assets.mkdir()
    assert runtime._resolve_aura_appearance(tmp_path) == (None, None)

    surflets = assets / "aura_ghost_removed_appearance.usd"
    surflets.write_text("#usda 1.0\n")
    path, kind = runtime._resolve_aura_appearance(tmp_path)
    assert path == surflets and kind == "particlefield_gaussian_surflet"

    nurec = assets / "aura_ghost_removed_appearance.usdz"
    nurec.write_bytes(b"PK\x03\x04")
    path, kind = runtime._resolve_aura_appearance(tmp_path)
    assert path == nurec and kind == "nurec_volume", "NuRec must win the tie"


def test_the_receipt_names_the_appearance_format() -> None:
    """A render result that does not say which authoring was in the scene
    cannot settle between two authorings of the same field."""

    from pathlib import Path as _Path

    from blueprint_pipeline import adp009d_isaac_runtime as runtime

    source = _Path(runtime.__file__).read_text(encoding="utf-8")
    assert '"aura_appearance_format":' in source
    # The old name asserted a format rather than reporting one.
    assert '"aura_particlefield_shipped"' not in source
    assert '"aura_appearance_shipped":' in source


def test_the_bundle_keeps_the_appearance_extension() -> None:
    import inspect

    from blueprint_pipeline import adp009d_native_microcheck_bundle as bundle

    source = inspect.getsource(bundle)
    assert "aura_ghost_removed_appearance{aura_source.suffix}" in source
    assert 'assets / "aura_ghost_removed_surflets.usd")' not in source
    assert "adp009d_aura_appearance_extension_unsupported" in source


def test_the_stage_probe_finds_a_nurec_volume_too() -> None:
    """It matched "Gauss" and "Aura" case-sensitively.

    A NuRec volume composes at /World/gauss/gauss -- lowercase -- so the probe
    reported zero matching prims for a scene that had visibly rendered the
    whole room.  A probe that says "absent" about something present is worse
    than no probe.
    """

    from pathlib import Path as _Path

    from blueprint_pipeline import adp009d_isaac_runtime as runtime

    source = _Path(runtime.__file__).read_text(encoding="utf-8")
    probe = source[source.index("aura_stage_probe: dict") :]
    probe = probe[: probe.index("live_collider = ")]
    assert ".lower()" in probe
    assert '"nurec"' in probe
    assert "omni:nurec:isNuRecVolume" in probe


def test_the_frames_only_receipt_names_the_appearance_format() -> None:
    """Omitting it read as None for a run whose appearance had rendered."""

    from pathlib import Path as _Path

    from blueprint_pipeline import adp009d_isaac_runtime as runtime

    source = _Path(runtime.__file__).read_text(encoding="utf-8")
    block = source[source.index('STOP_AFTER_FRAMES_ENV, ""') :]
    block = block[: block.index("--- gripper convention probe")]
    assert '"aura_appearance_format"' in block
    assert '"aura_appearance_shipped"' in block


def test_the_policy_resolution_reproduces_the_full_size_content_exactly() -> None:
    """Both candidates consume far less than 1280x720 was rendering.

    pi05 pads into 224x224 keeping 224x126 of content; groot keeps 320x180.
    Rendering at 320x180 reproduces both exactly while drawing a sixteenth of
    the pixels -- the rest was rendered and thrown away in the resize, and at
    roughly a minute per frame on a slow host that waste is what makes an
    episode take hours.
    """

    from blueprint_pipeline.adp009d_droid_observation import (
        describe_observation_conversion,
    )
    from blueprint_pipeline.adp009d_isaac_runtime import POLICY_CAMERA_RESOLUTION

    width, height = POLICY_CAMERA_RESOLUTION
    for candidate in ("pi05_droid", "groot_n17_droid"):
        full = describe_observation_conversion(candidate, source_hw=(720, 1280))
        small = describe_observation_conversion(candidate, source_hw=(height, width))
        assert full["content_resolution_hw"] == small["content_resolution_hw"], candidate
        assert small["scene_content_cropped"] is False
    assert (720 * 1280) / (height * width) == 16.0


def test_the_render_resolution_never_drops_below_what_a_policy_consumes(monkeypatch) -> None:
    """A lower resolution cannot be padded back up.

    The policy would see a genuinely lower-detail scene than the contract
    says, which is a silent change to the thing being evaluated.
    """

    from blueprint_pipeline import adp009d_isaac_runtime as runtime

    monkeypatch.setenv(runtime.CAMERA_RESOLUTION_ENV, "64x36")
    assert runtime._camera_resolution() == runtime.POLICY_CAMERA_RESOLUTION
    monkeypatch.setenv(runtime.CAMERA_RESOLUTION_ENV, "policy")
    assert runtime._camera_resolution() == runtime.POLICY_CAMERA_RESOLUTION
    monkeypatch.setenv(runtime.CAMERA_RESOLUTION_ENV, "640x360")
    assert runtime._camera_resolution() == (640, 360)
    monkeypatch.delenv(runtime.CAMERA_RESOLUTION_ENV)
    assert runtime._camera_resolution() == runtime.DIAGNOSTIC_CAMERA_RESOLUTION


def test_the_receipt_describes_the_conversion_that_happened() -> None:
    """A fixed 1280x720 would describe a conversion that did not occur."""

    from blueprint_pipeline.adp009d_droid_observation import (
        describe_observation_conversion,
    )

    described = describe_observation_conversion("pi05_droid", source_hw=(180, 320))
    assert described["source_resolution_hw"] == [180, 320]


def _batch(candidate: str, ranks: list[int]) -> dict:
    ladder = ["never_moved", "moved", "grasped", "lifted", "translated", "placed"]
    return {
        "candidate_id": candidate,
        "episodes_scored": len(ranks),
        "episodes_failed": 0,
        "outcome_counts": {},
        "episodes": [
            {
                "status": "scored",
                "outcome": ladder[r],
                "outcome_rank": r,
                "policy_outcome_interpretable": True,
            }
            for r in ranks
        ],
    }


def test_the_summary_ranks_candidates_and_names_the_leader() -> None:
    """An ordering is reported rather than withheld.

    A reader asking which policy did better on these episodes deserves the
    answer the data gives; what it must not do is impersonate an adjudicated
    result.
    """

    from blueprint_pipeline.adp009d_episode_batch import summarize_candidate_batches

    summary = summarize_candidate_batches(
        [_batch("groot_n17_droid", [1, 1, 2]), _batch("pi05_droid", [3, 4, 5])]
    )
    assert summary["ranking"] == ["pi05_droid", "groot_n17_droid"]
    assert summary["leader"] == "pi05_droid"
    assert summary["candidates"][0]["rank"] == 1
    assert summary["candidates"][0]["mean_outcome_rank"] == 4.0
    assert summary["tied"] is False


def test_the_ranking_orders_by_progress_not_binary_success() -> None:
    """Binary success throws away the difference between never moving the can
    and lifting it but failing to place it."""

    from blueprint_pipeline.adp009d_episode_batch import summarize_candidate_batches

    # Neither ever places; one gets much further.
    summary = summarize_candidate_batches([_batch("a", [0, 0, 0]), _batch("b", [3, 3, 4])])
    assert summary["leader"] == "b"
    assert summary["ranking_basis"] == "mean_outcome_rank_on_the_task_scoring_ladder"


def test_a_tie_is_reported_as_a_tie() -> None:
    from blueprint_pipeline.adp009d_episode_batch import summarize_candidate_batches

    summary = summarize_candidate_batches([_batch("a", [2, 2]), _batch("b", [2, 2])])
    assert summary["tied"] is True
    assert summary["leader"] is None


def test_the_sample_size_caveat_travels_with_the_ranking() -> None:
    """Attached to the ordering, not offered instead of it."""

    from blueprint_pipeline.adp009d_episode_batch import summarize_candidate_batches

    summary = summarize_candidate_batches([_batch("a", [1]), _batch("b", [2])])
    assert summary["ranking"] == ["b", "a"]
    assert summary["supports_policy_ranking"] is False
    assert "paired sample size" in summary["why_not_adjudicated"]


def test_unverified_action_delivery_blocks_the_top_level_runtime_result() -> None:
    from blueprint_pipeline.adp009d_isaac_runtime import _policy_episode_blockers

    blockers = _policy_episode_blockers(
        candidate_ids=["pi05_droid"],
        policy_episode={
            "batches": [
                {
                    "episodes_scored": 3,
                    "episodes_policy_outcome_uninterpretable": 3,
                }
            ]
        },
        policy_episode_error=None,
    )

    assert blockers == ["policy_episode_action_delivery_unverified:3"]


def test_interpretable_episode_evidence_does_not_create_a_runtime_blocker() -> None:
    from blueprint_pipeline.adp009d_isaac_runtime import _policy_episode_blockers

    assert (
        _policy_episode_blockers(
            candidate_ids=["pi05_droid"],
            policy_episode={
                "batches": [
                    {
                        "episodes_scored": 3,
                        "episodes_policy_outcome_uninterpretable": 0,
                        "episodes_media_incomplete": 0,
                    }
                ]
            },
            policy_episode_error=None,
        )
        == []
    )


def test_missing_episode_media_blocks_the_top_level_runtime_result() -> None:
    from blueprint_pipeline.adp009d_isaac_runtime import _policy_episode_blockers

    assert _policy_episode_blockers(
        candidate_ids=["pi05_droid"],
        policy_episode={
            "batches": [
                {
                    "episodes_scored": 3,
                    "episodes_policy_outcome_uninterpretable": 0,
                    "episodes_media_incomplete": 3,
                }
            ]
        },
        policy_episode_error=None,
    ) == ["policy_episode_media_incomplete:3"]


def test_the_runtime_runs_a_batch_per_bound_candidate() -> None:
    """Ranking two policies needs both in the same scene on the same host.

    Two runs on two machines would pay the boot twice and compare across
    hardware whose render speed already differs by 3x.
    """

    from pathlib import Path as _Path

    from blueprint_pipeline import adp009d_isaac_runtime as runtime

    source = _Path(runtime.__file__).read_text(encoding="utf-8")
    assert "for bound_candidate in candidate_ids:" in source


def test_policy_query_is_gated_on_a_replayed_wrist_observable_start() -> None:
    """A discovery pose that reset immediately erases is not an episode start."""

    from pathlib import Path as _Path

    from blueprint_pipeline import adp009d_isaac_runtime as runtime

    source = _Path(runtime.__file__).read_text(encoding="utf-8")
    assert 'episode_start_selection.get("status") == "ready"' in source
    assert "def _restore_wrist_observable_episode_start()" in source
    assert "reset_callback=_restore_wrist_observable_episode_start" in source
    assert "live_can_offset" in source
    assert "EPISODE_START_OBJECT_OFFSET_TOLERANCE_M" in source
    preflight = source.index("_restore_wrist_observable_episode_start()")
    first_client = source.index("policy=_client_for(server_receipt)")
    assert preflight < first_client
    assert "wrist_episode_start_selection" in source
    assert "wrist_episode_start_restore_receipts" in source
    assert "run_episode_batch(" in source
    assert "summarize_candidate_batches(" in source
    # v82's exact SimReady can occupied only 0.55% of the stable task view.
    # Both bound policy cameras must now pass semantic salience at selection
    # and again after reset replay before inference can begin.
    assert 'env.unwrapped.scene["external_camera"]' in source
    assert '"external_observability": external_observability' in source
    assert "restored_external_observability" in source
    # v83 proved that post-spawn pose mutation changed the sensor buffer but
    # not the USD/render prim.  Author the closer eye through Arena's camera
    # config before spawn, preserving the official DROID orientation.
    assert "external_camera_cfg.offset.pos = tuple(" in source
    assert '"Arena CameraCfg.offset before prim spawn"' in source
    assert "external_camera.set_world_poses_from_view(" not in source
    assert '"external_task_camera_plan": external_task_camera_plan' in source
    # A per-candidate receipt, or two candidates overwrite each other's.
    assert 'f"adp009d_policy_server_receipt.{bound_candidate}.json"' in source
    # One candidate failing to serve must not deny the other its episodes.
    assert '"policy_server_receipt_missing"' in source


def test_each_candidate_gets_its_own_policy_venv() -> None:
    """A shared venv failed the second candidate outright.

    uv refuses to create over an existing environment, so a live two-policy
    run had groot_n17_droid die at "A virtual environment already exists"
    after pi05_droid had made it.  And had creation succeeded it would still
    be wrong: openpi pins JAX and its own torch, GR00T a different torch, so
    whichever installed second would silently break the first.
    """

    import re

    from blueprint_pipeline.adp009d_policy_provisioning import (
        build_provisioning_script,
        policy_venv_root,
    )

    a = build_provisioning_script("pi05_droid")
    b = build_provisioning_script("groot_n17_droid")
    paths_a = set(re.findall(r"/opt/adp009d-policy-venv/[a-z0-9_]+", a))
    paths_b = set(re.findall(r"/opt/adp009d-policy-venv/[a-z0-9_]+", b))
    assert paths_a and paths_b
    assert not (paths_a & paths_b), (paths_a, paths_b)
    assert policy_venv_root("pi05_droid") != policy_venv_root("groot_n17_droid")
    # And no unresolved template placeholder reaches the worker as literal text.
    for script in (a, b):
        assert "{candidate_id}" not in script
        assert "{venv_root}" not in script


def test_the_bundle_ships_every_module_the_runtime_imports() -> None:
    """An import the runtime makes must be a file the bundle carries.

    adp009d_episode_batch was wired in and never shipped, so a live run
    reached the episode and died on ModuleNotFoundError after provisioning
    had already succeeded and the scene was built.
    """

    import inspect
    import re
    from pathlib import Path as _Path

    from blueprint_pipeline import adp009d_isaac_runtime as runtime
    from blueprint_pipeline import adp009d_native_microcheck_bundle as bundle

    shipped = set(re.findall(r'"(adp009d_[a-z0-9_]+\.py)"', inspect.getsource(bundle)))
    shipped |= {"episode_visual_evidence.py"}
    source = _Path(runtime.__file__).read_text(encoding="utf-8")
    # Flat-layout imports are the ones resolved from the bundle directory.
    imported = set(re.findall(r"^\s*from (adp009d_[a-z0-9_]+) import", source, re.M))
    missing = {f"{name}.py" for name in imported} - shipped
    assert not missing, f"runtime imports modules the bundle never ships: {sorted(missing)}"


def test_a_run_asked_for_episodes_that_scored_none_is_not_completed() -> None:
    """It reported completed with an empty blocker list.

    A live run carried a ModuleNotFoundError in policy_episode_error and still
    said completed, because the micro-check's own checks had passed and
    nothing contradicted it.  That is a success claim outrunning its evidence.
    Episodes were a bonus when that was written; they are the deliverable now.
    """

    from blueprint_pipeline.adp009d_isaac_runtime import _policy_episode_blockers

    assert _policy_episode_blockers(
        candidate_ids=["pi05_droid"],
        policy_episode={"batches": []},
        policy_episode_error="ModuleNotFoundError: adp009d_episode_batch",
    ) == [
        "policy_episode_error:ModuleNotFoundError: adp009d_episode_batch",
        "policy_episodes_requested_but_none_scored",
    ]
    # A diagnostic run with no candidate bound still has no episode obligation.
    assert (
        _policy_episode_blockers(
            candidate_ids=[],
            policy_episode=None,
            policy_episode_error=None,
        )
        == []
    )


def test_the_blocker_list_is_not_overwritten_by_a_later_literal() -> None:
    """A hardcoded empty list appeared after the computed one in the same dict.

    Python keeps the last, so every blocker the run computed was discarded on
    the way out -- which is why a failed episode reported completed with
    nothing to read.
    """

    import re
    from pathlib import Path as _Path

    from blueprint_pipeline import adp009d_isaac_runtime as runtime

    source = _Path(runtime.__file__).read_text(encoding="utf-8")
    # The success-path return specifically.  Earlier returns and the outer
    # exception handler legitimately carry their own blocker lists.
    start = source.index('"status": "completed" if not episode_blockers')
    body = source[start : source.index("    finally:", start)]
    assert len(re.findall(r'^\s+"blockers":', body, re.M)) == 1
    assert '"blockers": []' not in body


def test_the_claim_fields_follow_the_episodes() -> None:
    """They were hardcoded False beside a run that had queried two policies."""

    from pathlib import Path as _Path

    from blueprint_pipeline import adp009d_isaac_runtime as runtime

    source = _Path(runtime.__file__).read_text(encoding="utf-8")
    start = source.index('"status": "completed" if not episode_blockers')
    body = source[start : source.index("    finally:", start)]
    assert '"candidate_policy_queried": bool(' in body
    assert '"candidate_outcomes_accessed": bool(' in body
    assert '"candidate_policy_queried": False' not in body


def test_provisioning_progress_reaches_the_container_log() -> None:
    """The watchdog reads container stdout, not the provisioning log file.

    The entrypoint redirects the script's output to a file, so markers emitted
    inside the script went where nothing was looking and two runs were killed
    at thirty minutes while provisioning correctly.
    """

    from blueprint_pipeline.adp009d_native_microcheck_bundle import ENTRYPOINT

    # Emitted by the entrypoint itself, outside the redirection.
    assert 'echo "BLUEPRINT_WAM_RUNTIME_PHASE:adp009d:provision_${candidate}:started"' in ENTRYPOINT
    assert "provision_${candidate}:completed:rc=$rc" in ENTRYPOINT
    # And a tick while the fetch runs, because a checkpoint download has
    # nothing to say for minutes at a time.
    assert "while kill -0" in ENTRYPOINT
    assert "_working:" in ENTRYPOINT
    # The exit code must still be the script's, not the loop's.
    assert 'wait "$provisioning_pid"' in ENTRYPOINT
    assert ENTRYPOINT.index('wait "$provisioning_pid"') < ENTRYPOINT.index("rc=$?\n  echo")


def test_a_hanging_candidate_cannot_consume_the_whole_run() -> None:
    """Provisioning blocks the runtime, so an unbounded wait costs everything.

    A candidate whose server never answers would consume the entire TTL and
    the run would end with no episodes at all -- not even from the candidate
    that provisioned fine, which defeats the point of tolerating one bad arm.
    Observed: a server start ran forty-six minutes against its own
    fifteen-minute readiness timeout.
    """

    from blueprint_pipeline.adp009d_native_microcheck_bundle import ENTRYPOINT

    assert "BLUEPRINT_ADP009D_PROVISION_TIMEOUT_SECONDS" in ENTRYPOINT
    assert "provision_${candidate}:abandoned" in ENTRYPOINT
    # Terminated, then killed: a process ignoring TERM must not survive.
    assert 'setsid bash "$script"' in ENTRYPOINT
    assert 'kill -TERM -- "-$provisioning_pid"' in ENTRYPOINT
    assert 'kill -KILL -- "-$provisioning_pid"' in ENTRYPOINT
    assert ENTRYPOINT.index('kill -TERM -- "-$provisioning_pid"') < ENTRYPOINT.index(
        'kill -KILL -- "-$provisioning_pid"'
    )
    # The loop still runs the remaining candidates after abandoning one.
    abandoned = ENTRYPOINT.index("provision_${candidate}:abandoned")
    assert "break" in ENTRYPOINT[abandoned : abandoned + 400]
    assert "done" in ENTRYPOINT[abandoned:]


def test_no_shipped_module_uses_an_unguarded_top_level_relative_import() -> None:
    """The bundle ships modules flat, with no package for a relative import.

    Three separate runs reached the episode and died here -- twice on
    "attempted relative import with no known parent package", once on a
    module that was not shipped at all -- and no run has ever issued a policy
    query as a result.  Every one of those was a packaging defect, not a
    defect in the episode.

    Function-level relative imports are deliberately not flagged: they are
    lazy, and a shipped module can carry one on a path the worker never
    reaches.  A top-level one always executes.
    """

    import ast
    import inspect
    import re
    from pathlib import Path as _Path

    from blueprint_pipeline import adp009d_native_microcheck_bundle as bundle

    shipped = set(re.findall(r'"(adp009d_[a-z0-9_]+\.py)"', inspect.getsource(bundle)))
    shipped |= {
        "decision_evidence_contracts.py",
        "droid_policy_bridge.py",
        "episode_visual_evidence.py",
        "groot_n17_droid_policy_runtime.py",
    }
    root = _Path(bundle.__file__).parent
    offenders = []
    for name in sorted(shipped):
        path = root / name
        if not path.is_file():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:  # top level only
            if isinstance(node, ast.Try):
                continue  # a guarded dual-layout import
            if isinstance(node, ast.ImportFrom) and node.level:
                offenders.append(f"{name}:{node.lineno}")
    assert not offenders, (
        "shipped modules with unguarded top-level relative imports, which "
        f"cannot resolve in the flat bundle: {offenders}"
    )


def test_the_episode_clients_share_the_same_candidate_adapters_as_readiness() -> None:
    """The probe unwrapped it; the episode did not.

    openpi returns {"actions": ...} and the episode passed whatever it got
    straight to the action planner, which tried to build a float out of the
    dict.  So the readiness round trip looked healthy -- it unwrapped -- while
    the episode could not use the very same reply.
    """

    from pathlib import Path as _Path

    from blueprint_pipeline import adp009d_isaac_runtime as runtime

    source = _Path(runtime.__file__).read_text(encoding="utf-8")
    block = source[source.index("def _client_for(") :]
    block = block[: block.index("out_dir = Path(")]
    # OpenPI is unwrapped locally.  GR00T uses the same identity-bound adapter
    # as readiness, which translates its nested response and checks modalities.
    assert block.count("isinstance(response, dict)") == 1
    assert 'response["actions"]' in block
    assert "GrootN17DroidPolicyClient" in block
    assert "worker_identity_receipt" in block


def test_the_readiness_probe_and_the_episode_agree_on_the_response_shape() -> None:
    """They disagreed once, and the disagreement passed readiness."""

    import inspect
    from pathlib import Path as _Path

    from blueprint_pipeline import adp009d_isaac_runtime as runtime
    from blueprint_pipeline import adp009d_policy_server_worker as worker

    probe = inspect.getsource(worker.attempt_round_trip)
    episode = _Path(runtime.__file__).read_text(encoding="utf-8")
    assert "isinstance(response, dict)" in probe
    assert "isinstance(response, dict)" in episode
