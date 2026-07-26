from pathlib import Path

import pytest
from PIL import Image

from blueprint_pipeline.openpi_policy_ranking_gpu_bootstrap import (
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
