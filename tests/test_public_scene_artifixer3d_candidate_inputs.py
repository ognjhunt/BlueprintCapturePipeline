from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.public_scene_artifixer3d_candidate_inputs import (
    ArtiFixer3DCandidateInputError,
    SCHEMA_VERSION,
    materialize_artifixer3d_candidate_inputs,
)
from blueprint_pipeline.public_scene_aura_exact_residual_preflight import (
    materialize_aura_exact_residual_preflight,
)
from tests.test_public_scene_aura_exact_residual_preflight import _packet


def _preflight(tmp_path: Path, *, count: int = 2) -> Path:
    packet = _packet(tmp_path, count=count)
    output = tmp_path / "preflight.json"
    materialize_aura_exact_residual_preflight(
        input_packet_path=packet, output_path=output
    )
    return output


def test_prepares_one_to_five_exact_support_candidate_inputs(tmp_path: Path) -> None:
    preflight = _preflight(tmp_path, count=5)

    receipt = materialize_artifixer3d_candidate_inputs(
        calibrated_residual_preflight_path=preflight,
        output_root=tmp_path / "artifixer",
    )

    assert receipt["schema_version"] == SCHEMA_VERSION
    assert receipt["status"] == "candidate_inputs_prepared_no_model_no_execution"
    assert receipt["replacement_object_count"] == 5
    assert receipt["maximum_replacement_objects"] == 5
    assert len(receipt["tasks"]) == 5
    assert all(row["camera_count"] == 1 for row in receipt["tasks"])
    assert receipt["execution"]["provider_mutations_performed"] == 0
    assert receipt["adapter"]["opacity_role"] == (
        "binary_exact_repair_support_surrogate_not_native_3dgrut_opacity"
    )
    assert receipt["claim_boundary"]["policy_input_use_permitted"] is False
    assert receipt["receipt_digest"] == canonical_digest(
        receipt, digest_field="receipt_digest"
    )


def test_masks_references_and_builds_exact_inverse_opacity(tmp_path: Path) -> None:
    preflight = _preflight(tmp_path)
    receipt = materialize_artifixer3d_candidate_inputs(
        calibrated_residual_preflight_path=preflight,
        output_root=tmp_path / "artifixer",
    )

    task = receipt["tasks"][0]
    task_root = Path(task["scene_directory"])
    frame = task["frames"][0]
    before = np.asarray(
        Image.open(frame["input_retained_frame"]["path"]).convert("RGB"),
        dtype=np.uint8,
    )
    mask = np.asarray(
        Image.open(task_root / frame["exact_repair_mask"]["relative_path"]).convert(
            "L"
        ),
        dtype=np.uint8,
    ) > 0
    reference = np.asarray(
        Image.open(task_root / frame["masked_reference_rgb"]["relative_path"]).convert(
            "RGB"
        ),
        dtype=np.uint8,
    )
    opacity = np.asarray(
        Image.open(
            task_root / frame["binary_opacity_surrogate"]["relative_path"]
        ).convert("L"),
        dtype=np.uint8,
    )
    assert np.array_equal(reference[~mask], before[~mask])
    assert np.count_nonzero(reference[mask]) == 0
    assert np.all(opacity[~mask] == 255)
    assert np.all(opacity[mask] == 0)
    assert frame["outside_support_changed_pixels"] == 0

    transforms = json.loads(Path(task["transforms"]["path"]).read_text())
    source = json.loads(preflight.read_text())
    source_camera = next(
        row
        for row in source["camera_inputs"]
        if row["task_id"] == task["task_id"]
        and row["camera_id"] == frame["camera_id"]
    )
    expected = np.asarray(
        source_camera["calibration"]["spec"]["pose"][
            "T_world_camera_opencv"
        ],
        dtype=np.float64,
    ) @ np.diag([1.0, -1.0, -1.0, 1.0])
    assert np.allclose(transforms["frames"][0]["transform_matrix"], expected)


def test_rejects_tampered_preflight_or_nonempty_output(tmp_path: Path) -> None:
    preflight = _preflight(tmp_path)
    value = json.loads(preflight.read_text())
    value["replacement_object_count"] = 5
    preflight.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(
        ArtiFixer3DCandidateInputError, match="calibrated_preflight_invalid"
    ):
        materialize_artifixer3d_candidate_inputs(
            calibrated_residual_preflight_path=preflight,
            output_root=tmp_path / "artifixer",
        )

    valid = _preflight(tmp_path / "valid")
    output = tmp_path / "occupied"
    output.mkdir()
    (output / "user-owned.txt").write_text("preserve", encoding="utf-8")
    with pytest.raises(ArtiFixer3DCandidateInputError, match="output_not_empty"):
        materialize_artifixer3d_candidate_inputs(
            calibrated_residual_preflight_path=valid, output_root=output
        )
    assert (output / "user-owned.txt").read_text(encoding="utf-8") == "preserve"


def test_rejects_symlinked_preflight(tmp_path: Path) -> None:
    preflight = _preflight(tmp_path)
    link = tmp_path / "preflight-link.json"
    link.symlink_to(preflight)

    with pytest.raises(ArtiFixer3DCandidateInputError, match="preflight_missing"):
        materialize_artifixer3d_candidate_inputs(
            calibrated_residual_preflight_path=link,
            output_root=tmp_path / "artifixer",
        )
