from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile

import pytest
from pxr import Sdf, Usd

from blueprint_pipeline.public_scene_simready_native import (
    EXPECTED_CAMERA_IDS,
    OVRTX_QUALITY_STEPS,
    OVRTX_VERSION,
    OVSTAGE_VERSION,
    ISAAC_SIM_VERSION,
    materialize_native_probe,
    opencv_camera_to_usd_row_matrix,
)
from blueprint_pipeline.public_scene_simready_isaac_bundle import (
    build_simready_isaac_bundle,
)


ROOT = Path(__file__).resolve().parents[1]


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _evidence(tmp_path: Path) -> tuple[Path, dict]:
    evidence = tmp_path / "evidence"
    cameras = evidence / "inpainting_inputs/840313_ins160_v1/cameras.v1.json"
    cameras.parent.mkdir(parents=True)
    camera_rows = []
    for index, camera_id in enumerate(EXPECTED_CAMERA_IDS):
        camera_rows.append(
            {
                "camera_id": camera_id,
                "T_world_camera_opencv": [
                    [1.0, 0.0, 0.0, 1.0 + index],
                    [0.0, 1.0, 0.0, 2.0],
                    [0.0, 0.0, 1.0, 3.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                "intrinsics": {
                    "model": "PINHOLE",
                    "fx": 1646.981314951341,
                    "fy": 1646.981314951341,
                    "cx": 1024.0,
                    "cy": 768.0,
                    "width": 2048,
                    "height": 1536,
                },
            }
        )
    cameras.write_text(json.dumps(camera_rows), encoding="utf-8")
    scene = evidence / "simready/replacement_840313_match_v2"
    assets = scene / "assets"
    assets.mkdir(parents=True)
    composition = scene / "collision_and_replacement.usda"
    replacement = assets / "adp009a_840313_canned_beverage_match_v2.usda"
    collision = assets / "840313_collision.usd"
    composition.write_text(
        '''#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Z"
)
def Xform "World"
{
    def Xform "Environment"
    {
        def Cube "_LTFTHJVAZ3VMPTUJU888888" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        )
        {
            bool physics:collisionEnabled = true
            double size = 1
        }
    }
    def Xform "BlueprintReplacement" (
        prepend references = @assets/adp009a_840313_canned_beverage_match_v2.usda@</canned_beverage>
    )
    {
    }
}
''',
        encoding="utf-8",
    )
    replacement.write_text(
        '''#usda 1.0
(
    defaultPrim = "canned_beverage"
    metersPerUnit = 1
    upAxis = "Z"
)
def Xform "canned_beverage" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
)
{
    bool physics:rigidBodyEnabled = true
    float physics:mass = 0.33
    def Cylinder "body_collider" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        bool physics:collisionEnabled = true
        double height = 0.16
        double radius = 0.03
        float3[] extent = [(-0.03, -0.03, 0), (0.03, 0.03, 0.16)]
    }
    def Material "contact" (
        prepend apiSchemas = ["PhysicsMaterialAPI"]
    )
    {
        float physics:dynamicFriction = 0.4
        float physics:restitution = 0.05
        float physics:staticFriction = 0.5
    }
}
''',
        encoding="utf-8",
    )
    collision.write_text("#usda 1.0\n", encoding="utf-8")
    receipt = {
        "composition": {
            "composed_replacement_prim_path": "/World/BlueprintReplacement",
            "composed_support_collision_prim_path": (
                "/World/Environment/_LTFTHJVAZ3VMPTUJU888888"
            ),
            "relative_path": composition.relative_to(evidence).as_posix(),
            "sha256": _digest(composition),
            "replacement_asset_copy": {
                "relative_path": replacement.relative_to(evidence).as_posix(),
                "sha256": _digest(replacement),
            },
            "sage_collision_copy": {
                "relative_path": collision.relative_to(evidence).as_posix(),
                "sha256": _digest(collision),
            },
        },
        "placement": {
            "support_aligned_base_placement_m": [3.4681748, -3.3100837, 0.5264650138]
        },
    }
    return evidence, receipt


def test_opencv_camera_conversion_preserves_translation_and_flips_camera_axes() -> None:
    converted = opencv_camera_to_usd_row_matrix(
        [
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 2.0],
            [0.0, 0.0, 1.0, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    assert converted == [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [1.0, 2.0, 3.0, 1.0],
    ]


def test_native_probe_derives_exact_camera_and_drop_inputs(tmp_path: Path) -> None:
    evidence, receipt = _evidence(tmp_path)

    manifest = materialize_native_probe(
        evidence_root=evidence,
        destination=tmp_path / "probe",
        replacement_receipt=receipt,
    )

    assert manifest["status"] == "ready"
    assert manifest["ovrtx"]["camera_count"] == 8
    assert manifest["ovrtx"]["render_mode"] == "PathTracing"
    assert manifest["ovrtx"]["quality_steps"] == OVRTX_QUALITY_STEPS
    assert manifest["ovrtx"]["version"] == OVRTX_VERSION
    assert manifest["ovrtx"]["ovstage_version"] == OVSTAGE_VERSION
    assert manifest["ovrtx"]["modalities"] == ["rgb", "depth"]
    assert manifest["ovrtx"]["optional_modalities_not_required"] == ["normal"]
    assert manifest["ovphysx"]["drop_height_m"] == 0.05
    assert manifest["isaac"]["version"] == ISAAC_SIM_VERSION
    assert manifest["isaac"]["probe_names"] == ["drop", "gripper", "slide", "tip"]
    assert manifest["isaac"]["status"] == "frozen_not_executed"
    config = json.loads(
        (tmp_path / "probe/ovrtx_configs/approach_wide.json").read_text()
    )
    assert config["width"] == 2048
    assert config["height"] == 1536
    assert config["camera_transform_matrix_usd"][3][:3] == [1.0, 2.0, 3.0]
    assert config["quality_steps"] == OVRTX_QUALITY_STEPS
    physics_config = json.loads(
        (tmp_path / "probe/ovphysx_config.json").read_text()
    )
    assert physics_config["device"] == "gpu"
    inventory = physics_config["usd_scene_inventory"]
    assert inventory["rigid_bodies"] == ["/World/BlueprintReplacement"]
    assert inventory["support_collider_path"] in inventory["colliders"]
    assert inventory["masses"][0]["mass"] == pytest.approx(0.33)
    assert inventory["materials"][0]["static_friction"] == pytest.approx(0.5)
    drop = Usd.Stage.Open(str(tmp_path / "probe/drop_stage.usda"))
    assert drop is not None
    replacement = drop.GetPrimAtPath("/World/BlueprintReplacement")
    applied_api_schemas = replacement.GetMetadata("apiSchemas")
    assert "PhysxContactReportAPI" in list(applied_api_schemas.explicitItems)
    translate = replacement.GetAttribute("xformOp:translate").Get()
    assert tuple(translate) == pytest.approx((3.4681748, -3.3100837, 0.5764650138))
    assert (
        drop.GetPrimAtPath("/World/BlueprintReplacement/colliders/body_collider")
        .GetAttribute("physics:approximation")
        .Get()
        == "convexHull"
    )
    assert (
        drop.GetPrimAtPath(inventory["support_collider_path"])
        .GetAttribute("physics:approximation")
        .Get()
        == "none"
    )
    isaac_spec = json.loads((tmp_path / "probe/isaac_probe_spec.json").read_text())
    assert isaac_spec["status"] == "frozen_before_execution"
    assert isaac_spec["replacement_dimensions_m"] == pytest.approx(
        [0.06, 0.06, 0.16]
    )
    assert isaac_spec["replacement_mass_kg"] == pytest.approx(0.33)
    for stage_name in ("drop", "slide", "tip", "gripper"):
        stage_record = isaac_spec["stages"][stage_name]
        stage_path = tmp_path / "probe" / stage_record["relative_path"]
        assert stage_path.is_file()
        assert _digest(stage_path) == stage_record["sha256"]
        assert Usd.Stage.Open(str(stage_path)) is not None
    slide = Usd.Stage.Open(str(tmp_path / "probe/isaac_slide_stage.usda"))
    assert slide.GetPrimAtPath("/World/BlueprintReplacement").GetAttribute(
        "physics:velocity"
    ).Get() == pytest.approx((0.2, 0.0, 0.0))
    tip = Usd.Stage.Open(str(tmp_path / "probe/isaac_tip_stage.usda"))
    assert tip.GetPrimAtPath("/World/BlueprintReplacement").GetAttribute(
        "xformOp:rotateXYZ"
    ).Get() == pytest.approx((0.0, 6.0, 0.0))
    gripper = Usd.Stage.Open(str(tmp_path / "probe/isaac_gripper_stage.usda"))
    assert gripper.GetPrimAtPath("/World/BlueprintProbeGripper/left_finger").IsValid()
    assert gripper.GetPrimAtPath("/World/BlueprintProbeGripper/right_finger").IsValid()

    runner_source = (ROOT / "scripts/adp_content_agents_provider_runner.py").read_text()
    native_render_command = runner_source[
        runner_source.index('str(render_worker),') : runner_source.index(
            'log_path=output_root / "native_ovrtx"', runner_source.index('str(render_worker),')
        )
    ]
    assert '"rgb"' in native_render_command
    assert '"depth"' in native_render_command
    assert '"normal"' not in native_render_command


def test_native_probe_rejects_changed_collision_bytes(tmp_path: Path) -> None:
    evidence, receipt = _evidence(tmp_path)
    collision = evidence / receipt["composition"]["sage_collision_copy"]["relative_path"]
    collision.write_bytes(collision.read_bytes() + b"changed")

    with pytest.raises(ValueError, match="source_identity_mismatch"):
        materialize_native_probe(
            evidence_root=evidence,
            destination=tmp_path / "probe",
            replacement_receipt=receipt,
        )


def test_native_isaac_bundle_is_self_contained_and_deterministic(
    tmp_path: Path,
) -> None:
    evidence, replacement_receipt = _evidence(tmp_path)
    probe_root = tmp_path / "probe"
    materialize_native_probe(
        evidence_root=evidence,
        destination=probe_root,
        replacement_receipt=replacement_receipt,
    )

    receipt = build_simready_isaac_bundle(
        probe_root=probe_root,
        job_dir=tmp_path / "job",
        worker_source=ROOT / "scripts/run_adp009b_simready_isaac_worker.py",
        source_commit_sha="a" * 40,
        generated_at="2026-08-06T00:00:00Z",
    )

    assert receipt["status"] == "ready"
    bundle = Path(receipt["bundle_path"])
    assert receipt["bundle_sha256"] == _digest(bundle)
    with zipfile.ZipFile(bundle) as archive:
        assert archive.testzip() is None
        entries = set(archive.namelist())
    assert "isaac_provider_runtime_bundle.zip" not in entries
    assert {
        "provider_runtime/run_isaac_realistic_runtime.sh",
        "provider_runtime/isaac_realistic_runtime_runner.py",
        "provider_runtime/isaac_provider_eval_manifest.json",
        "provider_runtime/native/isaac_probe_spec.json",
        "provider_runtime/native/isaac_gripper_stage.usda",
    }.issubset(entries)


def test_native_isaac_worker_rejects_changed_stage_before_runtime(
    tmp_path: Path,
) -> None:
    evidence, replacement_receipt = _evidence(tmp_path)
    probe_root = tmp_path / "probe"
    materialize_native_probe(
        evidence_root=evidence,
        destination=probe_root,
        replacement_receipt=replacement_receipt,
    )
    stage = probe_root / "isaac_slide_stage.usda"
    stage.write_bytes(stage.read_bytes() + b"# changed\n")
    module_spec = importlib.util.spec_from_file_location(
        "blueprint_simready_isaac_worker",
        ROOT / "scripts/run_adp009b_simready_isaac_worker.py",
    )
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    output = tmp_path / "result.json"

    result = module.run(probe_root / "isaac_probe_spec.json", output)

    assert result["status"] == "blocked"
    assert result["native_isaac_executed"] is False
    assert result["blockers"] == ["simready_isaac_execution_failed:ValueError"]
    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert persisted["result_digest"] == module._canonical_digest(
        persisted, field="result_digest"
    )


def test_ovrtx_worker_authors_matrix_camera_and_path_tracing(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location(
        "blueprint_ovrtx_worker", ROOT / "scripts/run_ovrtx_preflight_worker.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    scene = tmp_path / "scene.usda"
    scene.write_text("#usda 1.0\n", encoding="utf-8")
    config = {
        "width": 2048,
        "height": 1536,
        "camera_transform_matrix_usd": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [1.0, 2.0, 3.0, 1.0],
        ],
        "render_mode": "PathTracing",
    }

    layer = module._camera_layer(scene, config)

    assert "matrix4d xformOp:transform" in layer
    assert "(1.0, 2.0, 3.0, 1.0)" in layer
    assert 'token omni:rtx:rendermode = "PathTracing"' in layer
    assert 'def RenderVar "Normal"' in layer
    assert "OmniRtxSettingsPtAdvancedAPI_1" in layer
    parsed = Sdf.Layer.CreateAnonymous("blueprint-ovrtx-test.usda")
    assert parsed.ImportFromString(layer) is True


def test_ovphysx_worker_uses_digest_bound_external_usd_inventory(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location(
        "blueprint_ovphysx_worker", ROOT / "scripts/run_ovphysx_preflight_worker.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    scene = tmp_path / "scene.usda"
    scene.write_text("#usda 1.0\n", encoding="utf-8")
    inventory = {
        "source_sha256": _digest(scene),
        "rigid_bodies": ["/World/Object"],
        "colliders": ["/World/Object/collider"],
        "joints": [],
        "masses": [{"path": "/World/Object", "mass": 0.33}],
        "materials": [{"path": "/World/Object/material", "static_friction": 0.5}],
    }

    observed = module._scene_inventory({"usd_scene_inventory": inventory}, scene)

    assert observed["source_sha256"] == _digest(scene)
    scene.write_text("#usda 1.0\n# changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source digest changed"):
        module._scene_inventory({"usd_scene_inventory": inventory}, scene)
