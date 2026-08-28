from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_scene_configuration_diagnostic_output import (
    seal_validated_diagnostic_checkpoint_reference,
)


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def test_seals_validated_prefix_two_for_cold_diagnostic_retry(tmp_path: Path) -> None:
    root = tmp_path / "after-stage-2"
    root.mkdir()
    manifest = (
        root / "task_evaluation_scene_configuration_diagnostic_checkpoint.v1.json"
    )
    manifest.write_text("{}\n", encoding="utf-8")
    artifact = root / "stage-2.usda"
    artifact.write_bytes(b"collision configuration")
    checkpoint = {
        "checkpoint_digest": "sha256:" + "2" * 64,
        "completed_stage_prefix_count": 2,
    }
    destination = tmp_path / "checkpoint-reference.v1.json"

    reference = seal_validated_diagnostic_checkpoint_reference(
        checkpoint_root=root,
        destination=destination,
        source_provider_result_digest="sha256:" + "3" * 64,
        checkpoint_validator=lambda **_kwargs: checkpoint,
    )

    assert reference["status"] == (
        "validated_diagnostic_checkpoint_ready_for_next_retry"
    )
    assert reference["completed_stage_prefix_count"] == 2
    assert reference["manifest_sha256"] == _sha256(manifest)
    assert reference["file_count"] == 2
    assert reference["total_bytes"] == manifest.stat().st_size + artifact.stat().st_size
    assert reference["reference_digest"] == canonical_digest(
        reference, digest_field="reference_digest"
    )
    assert json.loads(destination.read_text(encoding="utf-8")) == reference
    assert destination.stat().st_mode & 0o777 == 0o440


def test_refuses_prefix_zero_without_writing_reference(tmp_path: Path) -> None:
    root = tmp_path / "prefix-zero"
    root.mkdir()
    (root / "task_evaluation_scene_configuration_diagnostic_checkpoint.v1.json").write_text(
        "{}\n", encoding="utf-8"
    )
    destination = tmp_path / "checkpoint-reference.v1.json"

    with pytest.raises(
        ValueError,
        match="scene_configuration_diagnostic_checkpoint_reference_invalid",
    ):
        seal_validated_diagnostic_checkpoint_reference(
            checkpoint_root=root,
            destination=destination,
            source_provider_result_digest="sha256:" + "3" * 64,
            checkpoint_validator=lambda **_kwargs: {
                "checkpoint_digest": "sha256:" + "0" * 64,
                "completed_stage_prefix_count": 0,
            },
        )

    assert not destination.exists()
