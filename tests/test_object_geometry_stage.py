from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from blueprint_pipeline.object_geometry_stage import run_object_geometry_stage

pytestmark = pytest.mark.skipif(importlib.util.find_spec("trimesh") is None, reason="trimesh not installed")
if importlib.util.find_spec("trimesh") is not None:  # pragma: no branch
    import trimesh


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_l_shape_mesh(path: Path) -> None:
    leg_a = trimesh.creation.box(extents=(2.0, 0.6, 0.8))
    leg_a.apply_translation((0.0, 0.0, 0.4))
    leg_b = trimesh.creation.box(extents=(0.6, 1.6, 0.8))
    leg_b.apply_translation((-0.7, 0.5, 0.4))
    mesh = trimesh.util.concatenate([leg_a, leg_b])
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(path)


def _build_capture(tmp_path: Path) -> Path:
    capture_root = tmp_path / "bucket" / "scenes" / "scene_geom" / "captures" / "cap_geom"
    raw_root = capture_root / "raw"
    pipeline_root = capture_root / "pipeline"
    mesh_path = raw_root / "arkit" / "objects" / "couch_1.ply"
    _build_l_shape_mesh(mesh_path)
    _write_json(capture_root / "capture_descriptor.json", {"scene_id": "scene_geom", "capture_id": "cap_geom"})
    _write_json(
        raw_root / "arkit" / "objects" / "index.json",
        {
            "objects": [
                {
                    "id": "1",
                    "label": "sectional couch",
                    "pointCloudFile": "couch_1.ply",
                    "boundingBox": {
                        "center": [0.0, 0.0, 0.4],
                        "extents": [2.0, 1.6, 0.8],
                        "axes": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                        "orientationQuaternion": [1, 0, 0, 0],
                    },
                }
            ]
        },
    )
    _write_json(
        pipeline_root / "task_scope_record.json",
        {
            "task_statement": "Navigate around sectional couch_1",
            "target_object_ids": ["1"],
            "articulation_required_ids": [],
        },
    )
    _write_json(
        pipeline_root / "task_targets.json",
        {
            "target_object_ids": ["1"],
            "articulation_required_ids": [],
        },
    )
    return capture_root


def test_object_geometry_stage_extracts_mesh_hulls_and_ai_hints(tmp_path: Path) -> None:
    capture_root = _build_capture(tmp_path)

    result = run_object_geometry_stage(
        capture_root=capture_root,
        provider_name="palatial",
        ai_hint_runner=lambda payload: {"shape_label": "sectional_cluster", "view_strategy": "synthetic"},
    )

    manifest_path = Path(result["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    obj = manifest["objects"][0]

    assert manifest_path.is_file()
    assert Path(obj["mesh_glb_path"]).is_file()
    assert obj["mesh_source"] == "source_mesh"
    assert len(obj["collision_hulls"]) >= 2
    assert len(obj["support_surfaces"]) >= 1
    assert obj["source_mode"] == "synthetic_virtual"
    assert obj["visual_replacement_masks"]
    assert obj["ai_hints"]["source"] == "ai_runner"
    assert obj["ai_hints"]["shape_label"] == "sectional_cluster"
    assert obj["task_critical"] is True
    assert obj["grounding_level"] == "reconstructed"
    assert obj["canonical_truth"] is True
