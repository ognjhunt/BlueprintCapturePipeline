"""Tests for SAM3D-first materialization behavior."""

import json
from pathlib import Path
from typing import Any, Dict, List

from blueprint_pipeline.sam3d_assets import materialize_candidate_assets


class _StubRunner:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.calls: List[Dict[str, Any]] = []

    def materialize_text_assets(
        self,
        *,
        scene_id: str,
        assets_prefix: str,
        objects: List[Dict[str, Any]],
        room_type: str,
        generation_enabled: bool,
        retrieval_enabled: bool,
        retrieval_mode: str,
        generation_provider_chain: str,
    ) -> Dict[str, Any]:
        self.calls.append(
            {
                "scene_id": scene_id,
                "assets_prefix": assets_prefix,
                "objects": objects,
                "generation_enabled": generation_enabled,
                "retrieval_enabled": retrieval_enabled,
                "retrieval_mode": retrieval_mode,
                "generation_provider_chain": generation_provider_chain,
            }
        )

        provenance = []
        for obj in objects:
            oid = obj["id"]
            obj_dir = self.root / assets_prefix / oid
            obj_dir.mkdir(parents=True, exist_ok=True)
            (obj_dir / "model.usd").write_text("#usda 1.0", encoding="utf-8")
            (obj_dir / "metadata.json").write_text("{}", encoding="utf-8")
            generated = obj_dir / "generated_asset"
            generated.mkdir(parents=True, exist_ok=True)
            (generated / "sam3d.glb").write_bytes(b"glb")
            provenance.append(
                {
                    "object_id": oid,
                    "path": f"{assets_prefix}/{oid}/model.usd",
                    "materialization": "generated_sam3d",
                }
            )

        return {"provenance_assets": provenance, "retrieval_report": {"method_counts": {"generated": 1}}}


def test_reference_image_copied_for_image_conditioned_generation(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("STAGE_D_MATERIALIZATION_MODE", "image_conditioned")
    monkeypatch.setenv("STAGE_D_IMAGE_CLEANUP_PROVIDER", "skip")
    monkeypatch.delenv("STAGE_D_IMAGE_TO_3D_COMMAND", raising=False)
    runner = _StubRunner(tmp_path)

    crop_file = tmp_path / "crop.png"
    crop_file.write_bytes(b"fake image data")

    candidates = [
        {
            "object_id": "drawer_1",
            "asset_dir": "obj_drawer_1",
            "label": "drawer",
            "sim_role": "articulated_furniture",
            "dimensions_est": {"width": 0.8, "height": 0.4, "depth": 0.5},
            "physics_hints": {"dynamic": False},
            "articulation": {"required": True, "requirement_source": "keyword"},
            "obb": {
                "center": [1.0, 0.5, 2.0],
                "extents": [0.8, 0.4, 0.5],
                "axes": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "orientationQuaternion": [1, 0, 0, 0],
            },
            "reference_crop": str(crop_file),
            "all_crops": [str(crop_file)],
        }
    ]

    materialize_candidate_assets(
        runner=runner,  # type: ignore[arg-type]
        storage_root=tmp_path,
        scene_id="scene_1",
        assets_prefix="scenes/scene_1/assets",
        room_type="kitchen",
        swap_candidates=candidates,
    )

    # Adapter path should not be used in image_conditioned mode.
    assert runner.calls == []

    # Verify crop was copied to asset directory
    ref_png = tmp_path / "scenes/scene_1/assets/obj_drawer_1/reference.png"
    assert ref_png.is_file()
    assert ref_png.read_bytes() == b"fake image data"

    metadata = json.loads(
        (
            tmp_path
            / "scenes/scene_1/assets/obj_drawer_1/metadata.json"
        ).read_text(encoding="utf-8")
    )
    assert metadata["object_id"] == "drawer_1"
    assert metadata["reference_image"]
    assert metadata["source_kind"] in {"image_to_3d", "image_conditioned_proxy_box"}


def test_image_conditioned_generation_emits_contract_outputs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("STAGE_D_MATERIALIZATION_MODE", "image_conditioned")
    monkeypatch.setenv("STAGE_D_IMAGE_CLEANUP_PROVIDER", "skip")
    monkeypatch.delenv("STAGE_D_IMAGE_TO_3D_COMMAND", raising=False)
    runner = _StubRunner(tmp_path)
    candidates = [
        {
            "object_id": "drawer_1",
            "asset_dir": "obj_drawer_1",
            "label": "drawer",
            "sim_role": "articulated_furniture",
            "dimensions_est": {"width": 0.8, "height": 0.4, "depth": 0.5},
            "physics_hints": {"dynamic": False},
            "articulation": {"required": True, "requirement_source": "keyword"},
            "obb": {
                "center": [1.0, 0.5, 2.0],
                "extents": [0.8, 0.4, 0.5],
                "axes": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "orientationQuaternion": [1, 0, 0, 0],
            },
        }
    ]

    payload = materialize_candidate_assets(
        runner=runner,  # type: ignore[arg-type]
        storage_root=tmp_path,
        scene_id="scene_1",
        assets_prefix="scenes/scene_1/assets",
        room_type="kitchen",
        swap_candidates=candidates,
    )

    record = payload["records"][0]
    assert record["status"] == "success"
    assert (tmp_path / record["model_path"]).is_file()
    assert (tmp_path / record["mesh_glb_path"]).is_file()
    assert (tmp_path / record["metadata_path"]).is_file()


def test_adapter_mode_keeps_legacy_adapter_path(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("STAGE_D_MATERIALIZATION_MODE", "adapter")
    runner = _StubRunner(tmp_path)
    candidates = [
        {
            "object_id": "drawer_1",
            "asset_dir": "obj_drawer_1",
            "label": "drawer",
            "sim_role": "articulated_furniture",
            "dimensions_est": {"width": 0.8, "height": 0.4, "depth": 0.5},
            "physics_hints": {"dynamic": False},
            "articulation": {"required": True, "requirement_source": "keyword"},
            "obb": {
                "center": [1.0, 0.5, 2.0],
                "extents": [0.8, 0.4, 0.5],
                "axes": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "orientationQuaternion": [1, 0, 0, 0],
            },
        }
    ]

    payload = materialize_candidate_assets(
        runner=runner,  # type: ignore[arg-type]
        storage_root=tmp_path,
        scene_id="scene_1",
        assets_prefix="scenes/scene_1/assets",
        room_type="kitchen",
        swap_candidates=candidates,
    )

    assert len(runner.calls) == 1
    call = runner.calls[0]
    assert call["generation_enabled"] is True
    assert call["retrieval_enabled"] is False
    assert call["generation_provider_chain"] == "sam3d,hunyuan3d"
    assert payload["records"][0]["status"] == "success"
