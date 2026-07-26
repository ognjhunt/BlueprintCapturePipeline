from pathlib import Path

import pytest
from PIL import Image

from blueprint_pipeline.openpi_policy_ranking_gpu_bootstrap import (
    _download_signed_input,
    _upload_output,
    build_multi_scene_private_input_bundle,
    build_private_input_bundle,
    extract_private_input_bundle,
)


def test_private_bundle_roundtrip_and_hash_binding(tmp_path: Path) -> None:
    background = tmp_path / "background.png"
    Image.new("RGB", (224, 224), color=(1, 2, 3)).save(background)
    bundle = tmp_path / "input.zip"
    receipt = build_private_input_bundle(
        background_path=background,
        output_zip=bundle,
        source_scene_id="scene",
        source_revision="a" * 40,
        source_asset_sha256="b" * 64,
    )
    extracted = extract_private_input_bundle(
        bundle_path=bundle,
        expected_bundle_sha256=receipt["bundle_sha256"],
        output_dir=tmp_path / "extracted",
    )
    assert Path(extracted["background_path"]).read_bytes() == background.read_bytes()
    assert extracted["manifest"]["raw_3dgs_included"] is False

    with pytest.raises(ValueError, match="bundle_sha256_mismatch"):
        extract_private_input_bundle(
            bundle_path=bundle,
            expected_bundle_sha256="0" * 64,
            output_dir=tmp_path / "wrong",
        )


def test_signed_transport_rejects_non_https_before_network(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="gpu_input_url_not_safe_https"):
        _download_signed_input("http://storage.example/input", tmp_path / "input.zip")
    archive = tmp_path / "output.zip"
    archive.write_bytes(b"zip")
    with pytest.raises(ValueError, match="gpu_output_url_not_safe_https"):
        _upload_output("file:///tmp/output.zip", archive)


def test_multi_scene_bundle_keeps_warehouse_and_captured_claims_separate(
    tmp_path: Path,
) -> None:
    captured = tmp_path / "captured.png"
    warehouse = tmp_path / "warehouse.png"
    Image.new("RGB", (224, 224), color=(1, 2, 3)).save(captured)
    Image.new("RGB", (224, 224), color=(4, 5, 6)).save(warehouse)
    bundle = tmp_path / "multi.zip"
    receipt = build_multi_scene_private_input_bundle(
        scenes=[
            {
                "background_path": captured,
                "source_scene_id": "captured",
                "source_scene_kind": "captured_3dgs",
                "source_revision": "a" * 40,
                "source_asset_sha256": "b" * 64,
            },
            {
                "background_path": warehouse,
                "source_scene_id": "warehouse",
                "source_scene_kind": "controlled_nvidia_usd",
                "source_revision": "c" * 40,
                "source_asset_sha256": "d" * 64,
            },
        ],
        output_zip=bundle,
    )
    extracted = extract_private_input_bundle(
        bundle_path=bundle,
        expected_bundle_sha256=receipt["bundle_sha256"],
        output_dir=tmp_path / "extracted-multi",
    )
    assert receipt["manifest"]["scene_count"] == 2
    assert [row["scene_kind"] for row in extracted["scene_backgrounds"]] == [
        "captured_3dgs",
        "controlled_nvidia_usd",
    ]
    assert Path(extracted["scene_backgrounds"][1]["background_path"]).read_bytes() == (
        warehouse.read_bytes()
    )
