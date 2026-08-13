from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.replacement_asset_frame_registration import (
    ReplacementAssetFrameRegistrationError,
    seal_replacement_asset_frame_registration,
    validate_replacement_asset_frame_registration,
)


def _references(tmp_path: Path) -> list[Path]:
    rows = [tmp_path / "front.png", tmp_path / "oblique.png"]
    for index, path in enumerate(rows):
        path.write_bytes(b"reference" + bytes([index]))
    return rows


def test_registration_seals_180_degree_heading_correction(tmp_path: Path) -> None:
    result = seal_replacement_asset_frame_registration(
        scene_id="scene",
        task_id="task",
        asset_id="asset",
        asset_local_forward_axis=[0, -1, 0],
        asset_local_up_axis=[0, 0, 1],
        observed_world_forward_axis=[0, 1, 0],
        observed_world_up_axis=[0, 0, 1],
        reference_image_paths=_references(tmp_path),
        reviewed_by="human",
        output_path=tmp_path / "registration.json",
    )
    assert result["T_observed_world_axes_from_asset_local_axes"] == [
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    assert validate_replacement_asset_frame_registration(result) == result


def test_registration_rejects_implicit_identity_and_tampered_reference(tmp_path: Path) -> None:
    result = seal_replacement_asset_frame_registration(
        scene_id="scene",
        task_id="task",
        asset_id="asset",
        asset_local_forward_axis=[0, -1, 0],
        asset_local_up_axis=[0, 0, 1],
        observed_world_forward_axis=[0, -1, 0],
        observed_world_up_axis=[0, 0, 1],
        reference_image_paths=_references(tmp_path),
        reviewed_by="human",
        output_path=tmp_path / "registration.json",
    )
    altered = json.loads(json.dumps(result))
    altered["identity_assumed_without_review"] = True
    altered["registration_digest"] = canonical_digest(altered, digest_field="registration_digest")
    with pytest.raises(ReplacementAssetFrameRegistrationError):
        validate_replacement_asset_frame_registration(altered)
    Path(result["reference_images"][0]["path"]).write_bytes(b"tampered")
    with pytest.raises(ReplacementAssetFrameRegistrationError):
        validate_replacement_asset_frame_registration(result)
