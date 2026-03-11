from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path

from blueprint_pipeline.simready_stage import run_simready_stage


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_png(path: Path, *, width: int = 96, height: int = 72, color: tuple[int, int, int] = (200, 110, 80)) -> None:
    def _chunk(chunk_type: bytes, data: bytes) -> bytes:
        payload = chunk_type + data
        return (
            struct.pack(">I", len(data))
            + payload
            + struct.pack(">I", zlib.crc32(payload) & 0xFFFFFFFF)
        )

    rows = []
    for _ in range(height):
        row = bytearray()
        for _ in range(width):
            row.extend(bytes(color))
        rows.append(b"\x00" + bytes(row))
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    payload = (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", zlib.compress(b"".join(rows)))
        + _chunk(b"IEND", b"")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _build_capture(
    tmp_path: Path,
    *,
    scene_id: str,
    capture_id: str,
    task_text: str,
    include_crop: bool = False,
) -> Path:
    capture_root = tmp_path / "bucket" / "scenes" / scene_id / "captures" / capture_id
    raw_root = capture_root / "raw"
    pipeline_root = capture_root / "pipeline"
    _write_json(
        capture_root / "capture_descriptor.json",
        {
            "scene_id": scene_id,
            "capture_id": capture_id,
            "metadata": {"task_statement": task_text},
        },
    )

    crop_path = raw_root / "object_crops" / "mug_1.png"
    if include_crop:
        _write_png(crop_path)

    _write_json(
        raw_root / "object_index.json",
        {
            "objects": [
                {
                    "id": "1",
                    "label": "mug",
                    "reference_crop": str(crop_path) if include_crop else "",
                    "all_crops": [str(crop_path)] if include_crop else [],
                    "boundingBox": {
                        "center": [0.0, 0.0, 0.85],
                        "extents": [0.12, 0.12, 0.18],
                        "axes": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                        "orientationQuaternion": [1, 0, 0, 0],
                    },
                },
                {
                    "id": "2",
                    "label": "table",
                    "boundingBox": {
                        "center": [0.0, 0.0, 0.38],
                        "extents": [1.2, 0.8, 0.76],
                        "axes": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                        "orientationQuaternion": [1, 0, 0, 0],
                    },
                },
                {
                    "id": "3",
                    "label": "door",
                    "boundingBox": {
                        "center": [1.4, 0.0, 1.1],
                        "extents": [0.1, 0.9, 2.2],
                        "axes": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                        "orientationQuaternion": [1, 0, 0, 0],
                    },
                },
            ]
        },
    )
    (raw_root / "3dgs_compressed.ply").parent.mkdir(parents=True, exist_ok=True)
    (raw_root / "3dgs_compressed.ply").write_bytes(b"ply")
    _write_json(
        pipeline_root / "task_scope_record.json",
        {
            "task_statement": task_text,
            "target_object_ids": ["1"],
            "articulation_required_ids": ["3"],
        },
    )
    _write_json(
        pipeline_root / "task_targets.json",
        {
            "target_object_ids": ["1"],
            "articulation_required_ids": ["3"],
        },
    )
    _write_json(
        pipeline_root / "geometry_evidence.json",
        {
            "measured_route_width_m": 1.1,
        },
    )
    return capture_root


def test_simready_stage_builds_synthetic_workcell(tmp_path: Path) -> None:
    capture_root = _build_capture(
        tmp_path,
        scene_id="scene_syn",
        capture_id="cap_syn",
        task_text="Pick up mug_1 and place it in the target zone",
    )

    result = run_simready_stage(capture_root=capture_root, provider_name="manual")

    manifest_path = Path(result["manifest_path"])
    validation_path = Path(result["validation_path"])
    scene_path = Path(result["scene_path"])
    views = json.loads((capture_root / "pipeline/simready/simready_object_views.json").read_text(encoding="utf-8"))
    assets = json.loads((capture_root / "pipeline/simready/simready_assets.json").read_text(encoding="utf-8"))
    packets = json.loads((capture_root / "pipeline/simready/object_packets.json").read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    geometry_manifest = json.loads((capture_root / "pipeline/object_geometry/object_geometry_manifest.json").read_text(encoding="utf-8"))

    assert manifest_path.is_file()
    assert validation_path.is_file()
    assert scene_path.is_file()
    assert Path(geometry_manifest["objects"][0]["mesh_glb_path"]).is_file()
    assert views["objects"][0]["source_mode"] == "synthetic_virtual"
    assert len(views["objects"][0]["selected_views"]) == 4
    assert any(item["support_object_id"] == "2" for item in packets["objects"] if item["object_id"] == "1")
    asset_kinds = {item["object_id"]: item["asset_kind"] for item in assets["assets"]}
    assert asset_kinds["1"] == "proxy_fallback"
    assert asset_kinds["3"] == "functional_proxy"
    assert validation["overall_status"] == "passed"


def test_simready_stage_prefers_real_reference_crops(tmp_path: Path) -> None:
    capture_root = _build_capture(
        tmp_path,
        scene_id="scene_real",
        capture_id="cap_real",
        task_text="Pick up mug_1 and place it in the target zone",
        include_crop=True,
    )

    run_simready_stage(capture_root=capture_root, provider_name="palatial")

    views = json.loads((capture_root / "pipeline/simready/simready_object_views.json").read_text(encoding="utf-8"))
    requests = json.loads((capture_root / "pipeline/simready/simready_asset_requests.json").read_text(encoding="utf-8"))
    assets = json.loads((capture_root / "pipeline/simready/simready_assets.json").read_text(encoding="utf-8"))

    mug_views = next(item for item in views["objects"] if item["object_id"] == "1")
    mug_request = next(item for item in requests["requests"] if item["object_id"] == "1")
    mug_asset = next(item for item in assets["assets"] if item["object_id"] == "1")

    assert mug_views["source_mode"] == "real_capture"
    assert mug_views["selected_views"][0]["image_path"].endswith("mug_1.png")
    assert mug_request["provider_name"] == "palatial"
    assert mug_request["selected_view_images"][0].endswith("mug_1.png")
    assert Path(str(mug_asset["reference_image_path"])).is_file()
