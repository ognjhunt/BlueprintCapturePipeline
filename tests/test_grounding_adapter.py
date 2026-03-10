"""Tests for the spatial grounding adapter boundary."""

from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline.capture_bridge import CaptureDescriptor
from blueprint_pipeline.grounding_adapter import infer_spatial_grounding


def _descriptor() -> CaptureDescriptor:
    return CaptureDescriptor.from_dict(
        {
            "schema_version": "v1",
            "scene_id": "scene_grounding",
            "capture_id": "cap_grounding",
            "capture_source": "iphone",
            "capture_tier": "tier1_iphone",
            "raw_prefix_uri": "gs://bucket/scenes/scene_grounding/iphone/cap_grounding/raw",
            "frames_index_uri": "gs://bucket/scenes/scene_grounding/captures/cap_grounding/frames/index.jsonl",
            "nurec_mode": "mono_pose_assisted",
        }
    )


def test_legacy_grounding_backend_normalizes_object_index() -> None:
    payload = infer_spatial_grounding(
        descriptor=_descriptor(),
        storage_root=Path("/tmp"),
        object_index_uri="gs://bucket/objects/index.json",
        object_index_entries=[
            {
                "id": "obj_1",
                "label": "mug",
                "boundingBox": {"center": [1.0, 2.0, 3.0], "extents": [0.1, 0.1, 0.2]},
                "mean_confidence": 0.8,
            }
        ],
        backend="legacy",
    )

    assert payload["backend"] == "legacy"
    assert payload["grounded_objects"][0]["object_id"] == "obj_1"
    assert payload["grounded_objects"][0]["boundingBox"]["center"] == [1.0, 2.0, 3.0]


def test_holi_adapter_falls_back_to_placeholder_when_command_missing() -> None:
    payload = infer_spatial_grounding(
        descriptor=_descriptor(),
        storage_root=Path("/tmp"),
        object_index_uri="gs://bucket/objects/index.json",
        object_index_entries=[
            {
                "id": "obj_1",
                "label": "mug",
                "boundingBox": {"center": [1.0, 2.0, 3.0], "extents": [0.1, 0.1, 0.2]},
                "mean_confidence": 0.8,
            }
        ],
        backend="holi_adapter",
    )

    assert payload["backend"] == "holi_adapter"
    assert payload["backend_status"] == "placeholder_fallback"
    assert payload["placeholder"] is True
    assert payload["grounded_objects"][0]["object_id"] == "obj_1"


def test_holi_adapter_uses_external_command_output(tmp_path: Path, monkeypatch) -> None:
    object_index_path = tmp_path / "bucket" / "objects" / "index.json"
    object_index_path.parent.mkdir(parents=True, exist_ok=True)
    object_index_path.write_text(json.dumps({"objects": []}), encoding="utf-8")
    script_path = tmp_path / "emit_grounding.py"
    script_path.write_text(
        "\n".join(
            [
                "import json, sys",
                "out = sys.argv[1]",
                "payload = {",
                "  'grounded_objects': [",
                "    {",
                "      'object_id': 'obj_ext',",
                "      'label': 'drawer',",
                "      'confidence': 0.93,",
                "      'boundingBox': {'center': [0.0, 1.0, 0.5], 'extents': [0.4, 0.4, 0.3]}",
                "    }",
                "  ],",
                "  'articulation_hints': [{'instance_id': 'obj_ext', 'label': 'drawer'}],",
                "}",
                "with open(out, 'w', encoding='utf-8') as f:",
                "    json.dump(payload, f)",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HOLI_SPATIAL_COMMAND", f"python {script_path} {{OUTPUT_JSON}}")

    payload = infer_spatial_grounding(
        descriptor=_descriptor(),
        storage_root=tmp_path,
        object_index_uri="gs://bucket/objects/index.json",
        object_index_entries=[],
        backend="holi_adapter",
    )

    assert payload["backend"] == "holi_adapter"
    assert payload["grounded_objects"][0]["object_id"] == "obj_ext"
    assert payload["articulation_hints"][0]["instance_id"] == "obj_ext"
