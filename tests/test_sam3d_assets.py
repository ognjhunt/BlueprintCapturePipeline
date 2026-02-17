"""Tests for SAM3D-first materialization behavior."""

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


def test_reference_image_passed_to_adapter(tmp_path: Path) -> None:
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

    # Verify reference_image was passed to adapter objects
    call = runner.calls[0]
    adapter_obj = call["objects"][0]
    assert adapter_obj["reference_image"] == str(crop_file)
    assert adapter_obj["reference_images"] == [str(crop_file)]

    # Verify crop was copied to asset directory
    ref_png = tmp_path / "scenes/scene_1/assets/obj_drawer_1/reference.png"
    assert ref_png.is_file()
    assert ref_png.read_bytes() == b"fake image data"


def test_sam3d_first_generation_and_mesh_glb_emission(tmp_path: Path) -> None:
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

    call = runner.calls[0]
    assert call["generation_enabled"] is True
    assert call["retrieval_enabled"] is False
    assert call["generation_provider_chain"] == "sam3d,hunyuan3d"

    record = payload["records"][0]
    assert record["status"] == "success"
    assert (tmp_path / record["model_path"]).is_file()
    assert (tmp_path / record["mesh_glb_path"]).is_file()
