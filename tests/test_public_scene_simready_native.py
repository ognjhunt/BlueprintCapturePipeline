from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
from pxr import Sdf, Usd

from blueprint_pipeline.public_scene_simready_native import (
    EXPECTED_CAMERA_IDS,
    OVRTX_QUALITY_STEPS,
    OVRTX_VERSION,
    OVSTAGE_VERSION,
    materialize_native_probe,
    opencv_camera_to_usd_row_matrix,
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
    assert manifest["ovphysx"]["drop_height_m"] == 0.05
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
    translate = drop.GetPrimAtPath("/World/BlueprintReplacement").GetAttribute(
        "xformOp:translate"
    ).Get()
    assert tuple(translate) == pytest.approx((3.4681748, -3.3100837, 0.5764650138))


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
