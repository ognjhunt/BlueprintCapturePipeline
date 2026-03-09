"""Tests for SAM3D-first materialization behavior."""

import json
from pathlib import Path
from typing import Any, Dict, List

from blueprint_pipeline.sam3d_assets import (
    materialize_candidate_assets,
    materialize_scene_shell_assets,
)


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


def test_articulated_candidates_route_to_retrieval_first(tmp_path: Path, monkeypatch) -> None:
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

    # Articulated-required objects route to retrieval-first branch.
    assert len(runner.calls) == 1
    call = runner.calls[0]
    assert call["generation_enabled"] is False
    assert call["retrieval_enabled"] is True
    assert call["retrieval_mode"] == "ann_primary"

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
    assert metadata["source_kind"] in {"articulated_retrieval", "articulated_retrieval_proxy_box"}
    assert metadata["router_branch"] == "articulated_required"


def test_image_conditioned_generation_emits_contract_outputs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("STAGE_D_MATERIALIZATION_MODE", "image_conditioned")
    monkeypatch.setenv("STAGE_D_IMAGE_CLEANUP_PROVIDER", "skip")
    monkeypatch.delenv("STAGE_D_IMAGE_TO_3D_COMMAND", raising=False)
    from blueprint_pipeline import sam3d_assets

    def _fake_proxy_mesh(_candidate, mesh_glb_path: Path) -> None:  # noqa: ANN001
        mesh_glb_path.parent.mkdir(parents=True, exist_ok=True)
        mesh_glb_path.write_bytes(b"glb")

    monkeypatch.setattr(sam3d_assets, "_write_proxy_mesh_glb", _fake_proxy_mesh)
    runner = _StubRunner(tmp_path)
    candidates = [
        {
            "object_id": "drawer_1",
            "asset_dir": "obj_drawer_1",
            "label": "drawer",
            "sim_role": "manipulable_object",
            "dimensions_est": {"width": 0.8, "height": 0.4, "depth": 0.5},
            "physics_hints": {"dynamic": True},
            "articulation": {"required": False, "requirement_source": "keyword"},
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
    assert runner.calls == []


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


def test_non_articulated_router_uses_topk_ranked_references(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("STAGE_D_MATERIALIZATION_MODE", "image_conditioned")
    monkeypatch.setenv("STAGE_D_IMAGE_CLEANUP_PROVIDER", "skip")
    monkeypatch.setenv("STAGE_D_IMAGE_TO_3D_COMMAND", "fake_cmd {REFERENCE_IMAGE} {OUTPUT_GLB}")
    monkeypatch.setenv("STAGE_D_IMAGE_TO_3D_TOPK", "2")

    from blueprint_pipeline import sam3d_assets

    captured: dict[str, object] = {}

    def _fake_run_image_to_3d_command(
        *,
        command_template: str,
        reference_image: Path,
        reference_images: list[Path],
        output_glb: Path,
        output_dir: Path,
        scene_id: str,
        object_id: str,
        asset_dir_name: str,
        room_type: str,
        timeout_seconds: int,
    ):
        captured["reference_image"] = str(reference_image)
        captured["reference_images"] = [str(item) for item in reference_images]
        output_glb.parent.mkdir(parents=True, exist_ok=True)
        output_glb.write_bytes(b"glb")
        return True, "ok", {"command": ["fake_cmd"]}

    monkeypatch.setattr(sam3d_assets, "_run_image_to_3d_command", _fake_run_image_to_3d_command)

    # Create 3 distinct crops; top-2 should be selected.
    crop1 = tmp_path / "crop1.png"
    crop2 = tmp_path / "crop2.png"
    crop3 = tmp_path / "crop3.png"
    crop1.write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x10\x00\x00\x00\x10\x08\x06\x00\x00\x00\x1f\xf3\xffa"
        b"\x00\x00\x00\x0cIDATx\x9cc```\xf8\xff\xff?\x00\x05\xfe\x02\xfe\xdc\xccY\xa5\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    crop2.write_bytes(crop1.read_bytes())
    crop3.write_bytes(crop1.read_bytes())

    runner = _StubRunner(tmp_path)
    candidates = [
        {
            "object_id": "box_1",
            "asset_dir": "obj_box_1",
            "label": "box",
            "sim_role": "manipulable_object",
            "dimensions_est": {"width": 0.4, "height": 0.4, "depth": 0.4},
            "physics_hints": {"dynamic": True},
            "articulation": {"required": False, "requirement_source": "keyword"},
            "obb": {
                "center": [0.0, 0.0, 0.0],
                "extents": [0.4, 0.4, 0.4],
                "axes": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "orientationQuaternion": [1, 0, 0, 0],
            },
            "reference_crop": str(crop1),
            "all_crops": [str(crop1), str(crop2), str(crop3)],
        }
    ]

    payload = sam3d_assets.materialize_candidate_assets(
        runner=runner,  # type: ignore[arg-type]
        storage_root=tmp_path,
        scene_id="scene_1",
        assets_prefix="scenes/scene_1/assets",
        room_type="warehouse",
        swap_candidates=candidates,
    )

    assert payload["records"][0]["status"] == "success"
    refs = captured.get("reference_images")
    assert isinstance(refs, list)
    assert len(refs) == 2


def test_non_articulated_router_uses_ttt_lrm_provider_command(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("STAGE_D_MATERIALIZATION_MODE", "image_conditioned")
    monkeypatch.setenv("STAGE_D_IMAGE_CLEANUP_PROVIDER", "skip")
    monkeypatch.delenv("STAGE_D_IMAGE_TO_3D_COMMAND", raising=False)
    monkeypatch.setenv(
        "STAGE_D_TTTLRM_IMAGE_TO_3D_COMMAND",
        "ttt_cli --provider {PROVIDER} --input {REFERENCE_IMAGE} --output {OUTPUT_GLB}",
    )

    from blueprint_pipeline import sam3d_assets

    captured: dict[str, object] = {}

    def _fake_run_image_to_3d_command(
        *,
        command_template: str,
        reference_image: Path,
        reference_images: list[Path],
        output_glb: Path,
        output_dir: Path,
        scene_id: str,
        object_id: str,
        asset_dir_name: str,
        room_type: str,
        timeout_seconds: int,
    ):
        captured["command_template"] = command_template
        output_glb.parent.mkdir(parents=True, exist_ok=True)
        output_glb.write_bytes(b"glb")
        return True, "ok", {"command": ["ttt_cli"], "return_code": 0}

    monkeypatch.setattr(sam3d_assets, "_run_image_to_3d_command", _fake_run_image_to_3d_command)

    crop = tmp_path / "crop.png"
    crop.write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x10\x00\x00\x00\x10\x08\x06\x00\x00\x00\x1f\xf3\xffa"
        b"\x00\x00\x00\x0cIDATx\x9cc```\xf8\xff\xff?\x00\x05\xfe\x02\xfe\xdc\xccY\xa5\x00\x00\x00\x00IEND\xaeB`\x82"
    )

    runner = _StubRunner(tmp_path)
    payload = sam3d_assets.materialize_candidate_assets(
        runner=runner,  # type: ignore[arg-type]
        storage_root=tmp_path,
        scene_id="scene_1",
        assets_prefix="scenes/scene_1/assets",
        room_type="warehouse",
        swap_candidates=[
            {
                "object_id": "box_2",
                "asset_dir": "obj_box_2",
                "label": "box",
                "sim_role": "manipulable_object",
                "dimensions_est": {"width": 0.4, "height": 0.4, "depth": 0.4},
                "physics_hints": {"dynamic": True},
                "articulation": {"required": False, "requirement_source": "keyword"},
                "obb": {
                    "center": [0.0, 0.0, 0.0],
                    "extents": [0.4, 0.4, 0.4],
                    "axes": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    "orientationQuaternion": [1, 0, 0, 0],
                },
                "reference_crop": str(crop),
                "all_crops": [str(crop)],
            }
        ],
        generation_provider_chain="ttt_lrm,proxy_box",
    )

    assert payload["records"][0]["status"] == "success"
    assert "ttt_lrm" in str(captured.get("command_template", ""))
    metadata = json.loads(
        (
            tmp_path
            / "scenes/scene_1/assets/obj_box_2/metadata.json"
        ).read_text(encoding="utf-8")
    )
    assert metadata["image_to_3d_selected_provider"] == "ttt_lrm"
    assert metadata["source_kind"] == "image_to_3d_ttt_lrm"


def test_materialize_scene_shell_prefers_usdz_by_default(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from blueprint_pipeline import sam3d_assets

    nurec_dir = tmp_path / "scenes/scene_1/captures/cap_1/pipeline/nurec"
    nurec_dir.mkdir(parents=True, exist_ok=True)
    (nurec_dir / "export_last.usdz").write_bytes(b"usdz")
    (nurec_dir / "visual_mesh.glb").write_bytes(b"glb")
    (nurec_dir / "nvblox_mesh.ply").write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 3",
                "property float x",
                "property float y",
                "property float z",
                "element face 1",
                "property list uchar int vertex_indices",
                "end_header",
                "0 0 0",
                "1 0 0",
                "0 1 0",
                "3 0 1 2",
            ]
        ),
        encoding="utf-8",
    )

    def _fake_ply_to_glb(ply_path: Path, glb_path: Path) -> None:
        glb_path.parent.mkdir(parents=True, exist_ok=True)
        glb_path.write_bytes(b"glb")

    monkeypatch.setattr(sam3d_assets, "_ply_to_glb", _fake_ply_to_glb)
    monkeypatch.setattr(sam3d_assets, "_prune_scene_shell_mesh", lambda *_a, **_k: {"enabled": False})
    monkeypatch.setattr(
        sam3d_assets,
        "_simplify_scene_shell_mesh",
        lambda *_a, **_k: {"enabled": False},
    )

    payload = materialize_scene_shell_assets(
        storage_root=tmp_path,
        assets_prefix="scenes/scene_1/assets",
        nurec_outputs={
            "artifacts": {
                "visual_usdz": "gs://bucket/scenes/scene_1/captures/cap_1/pipeline/nurec/export_last.usdz",
                "visual_mesh_glb": "gs://bucket/scenes/scene_1/captures/cap_1/pipeline/nurec/visual_mesh.glb",
                "collision_mesh_ply": "gs://bucket/scenes/scene_1/captures/cap_1/pipeline/nurec/nvblox_mesh.ply",
            }
        },
        swap_candidates=[],
    )

    visual_dir = tmp_path / "scenes/scene_1/assets/obj_nurec_visual"
    assert payload["visual_asset"] == "scenes/scene_1/assets/obj_nurec_visual/model.usd"
    assert (visual_dir / "model.glb").is_file()
    assert (visual_dir / "model.usdz").is_file()
    model_usd = (visual_dir / "model.usd").read_text(encoding="utf-8")
    assert "model.usdz" in model_usd
    assert "model.glb" not in model_usd
    metadata = json.loads((visual_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["source"] == "nurec_export_volume"


def test_materialize_scene_shell_uses_mesh_when_requested(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from blueprint_pipeline import sam3d_assets

    monkeypatch.setenv("NUREC_VISUAL_PRIMARY", "mesh")
    nurec_dir = tmp_path / "scenes/scene_1/captures/cap_1/pipeline/nurec"
    nurec_dir.mkdir(parents=True, exist_ok=True)
    (nurec_dir / "export_last.usdz").write_bytes(b"usdz")
    (nurec_dir / "visual_mesh.glb").write_bytes(b"glb")
    (nurec_dir / "nvblox_mesh.ply").write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 3",
                "property float x",
                "property float y",
                "property float z",
                "element face 1",
                "property list uchar int vertex_indices",
                "end_header",
                "0 0 0",
                "1 0 0",
                "0 1 0",
                "3 0 1 2",
            ]
        ),
        encoding="utf-8",
    )

    def _fake_ply_to_glb(ply_path: Path, glb_path: Path) -> None:
        glb_path.parent.mkdir(parents=True, exist_ok=True)
        glb_path.write_bytes(b"glb")

    monkeypatch.setattr(sam3d_assets, "_ply_to_glb", _fake_ply_to_glb)
    monkeypatch.setattr(sam3d_assets, "_prune_scene_shell_mesh", lambda *_a, **_k: {"enabled": False})
    monkeypatch.setattr(
        sam3d_assets,
        "_simplify_scene_shell_mesh",
        lambda *_a, **_k: {"enabled": False},
    )

    payload = materialize_scene_shell_assets(
        storage_root=tmp_path,
        assets_prefix="scenes/scene_1/assets",
        nurec_outputs={
            "artifacts": {
                "visual_usdz": "gs://bucket/scenes/scene_1/captures/cap_1/pipeline/nurec/export_last.usdz",
                "visual_mesh_glb": "gs://bucket/scenes/scene_1/captures/cap_1/pipeline/nurec/visual_mesh.glb",
                "collision_mesh_ply": "gs://bucket/scenes/scene_1/captures/cap_1/pipeline/nurec/nvblox_mesh.ply",
            }
        },
        swap_candidates=[],
    )

    visual_dir = tmp_path / "scenes/scene_1/assets/obj_nurec_visual"
    assert payload["visual_asset"] == "scenes/scene_1/assets/obj_nurec_visual/model.usd"
    assert (visual_dir / "model.glb").is_file()
    assert (visual_dir / "model.usdz").is_file()
    model_usd = (visual_dir / "model.usd").read_text(encoding="utf-8")
    assert "model.glb" in model_usd
    metadata = json.loads((visual_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["source"] == "nurec_visual_mesh"


def test_materialize_scene_shell_forces_cleaned_mesh_primary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from blueprint_pipeline import sam3d_assets

    nurec_dir = tmp_path / "scenes/scene_1/captures/cap_1/pipeline/nurec"
    nurec_dir.mkdir(parents=True, exist_ok=True)
    (nurec_dir / "export_last.usdz").write_bytes(b"usdz")
    (nurec_dir / "visual_mesh.glb").write_bytes(b"orig_glb")
    (nurec_dir / "inpainted_visual_mesh.glb").write_bytes(b"cleaned_glb")
    (nurec_dir / "nvblox_mesh.ply").write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 3",
                "property float x",
                "property float y",
                "property float z",
                "element face 1",
                "property list uchar int vertex_indices",
                "end_header",
                "0 0 0",
                "1 0 0",
                "0 1 0",
                "3 0 1 2",
            ]
        ),
        encoding="utf-8",
    )

    def _fake_ply_to_glb(ply_path: Path, glb_path: Path) -> None:
        glb_path.parent.mkdir(parents=True, exist_ok=True)
        glb_path.write_bytes(b"glb")

    monkeypatch.setattr(sam3d_assets, "_ply_to_glb", _fake_ply_to_glb)
    monkeypatch.setattr(sam3d_assets, "_prune_scene_shell_mesh", lambda *_a, **_k: {"enabled": False})
    monkeypatch.setattr(
        sam3d_assets,
        "_simplify_scene_shell_mesh",
        lambda *_a, **_k: {"enabled": False},
    )

    materialize_scene_shell_assets(
        storage_root=tmp_path,
        assets_prefix="scenes/scene_1/assets",
        nurec_outputs={
            "artifacts": {
                "visual_usdz": "gs://bucket/scenes/scene_1/captures/cap_1/pipeline/nurec/export_last.usdz",
                "visual_mesh_glb": "gs://bucket/scenes/scene_1/captures/cap_1/pipeline/nurec/visual_mesh.glb",
                "inpainted_visual_mesh_glb": "gs://bucket/scenes/scene_1/captures/cap_1/pipeline/nurec/inpainted_visual_mesh.glb",
                "collision_mesh_ply": "gs://bucket/scenes/scene_1/captures/cap_1/pipeline/nurec/nvblox_mesh.ply",
            }
        },
        swap_candidates=[],
    )

    visual_dir = tmp_path / "scenes/scene_1/assets/obj_nurec_visual"
    model_usd = (visual_dir / "model.usd").read_text(encoding="utf-8")
    metadata = json.loads((visual_dir / "metadata.json").read_text(encoding="utf-8"))
    assert "model.glb" in model_usd
    assert metadata["source"] == "inpaint360gs_cleaned_mesh"
