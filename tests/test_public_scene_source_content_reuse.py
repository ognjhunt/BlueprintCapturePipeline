"""Administrative deploys reuse validated source content without altering provenance."""
import json

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_scene_configuration_submission_inputs import (
    source_inputs, Staging,
)
from tests.test_task_evaluation_scene_configuration_submission import production_fixture, SHA


def test_source_installation_and_geometry_commits_remain_retained_provenance(tmp_path):
    fixture = production_fixture(tmp_path)
    task = json.loads(fixture["task_request"].read_text())
    task["expected_production_commit"] = "b" * 40
    before = {key: fixture[key].read_bytes() for key in ("installation_receipt", "source_preparation")}
    result = source_inputs(installation_path=fixture["installation_receipt"],
        preparation_path=fixture["source_preparation"], publisher_path=fixture["publisher_intake"],
        task=task, commit="b" * 40)
    assert result["preparation"]["source_commit"] == SHA
    assert all(fixture[key].read_bytes() == value for key, value in before.items())
    raw = result["raw"]["appearance_3dgs"]["path"]
    raw.write_bytes(b"changed publisher bytes")
    with pytest.raises(ValueError, match="input_bytes_mismatch"):
        source_inputs(installation_path=fixture["installation_receipt"],
            preparation_path=fixture["source_preparation"], publisher_path=fixture["publisher_intake"],
            task=task, commit="b" * 40)


def test_invalid_provenance_or_current_task_mismatch_fails_closed(tmp_path):
    fixture = production_fixture(tmp_path)
    task = json.loads(fixture["task_request"].read_text())
    task["expected_production_commit"] = "b" * 40
    kwargs = dict(installation_path=fixture["installation_receipt"],
        preparation_path=fixture["source_preparation"], publisher_path=fixture["publisher_intake"], task=task)
    with pytest.raises(ValueError, match="source_preparation_commit_mismatch"):
        source_inputs(**kwargs, commit=SHA)
    preparation = json.loads(fixture["source_preparation"].read_text())
    preparation["source_commit"] = "not-a-commit"
    preparation["receipt_digest"] = canonical_digest(preparation, digest_field="receipt_digest")
    fixture["source_preparation"].write_text(json.dumps(preparation))
    with pytest.raises(ValueError, match="source_preparation_commit_mismatch"):
        source_inputs(**kwargs, commit="b" * 40)


def test_immutable_publisher_stage_uses_same_inode_and_keeps_publication_forbidden(tmp_path):
    source = tmp_path / "publisher.ply"
    source.write_bytes(b"immutable publisher bytes")
    source.chmod(0o440)
    stage = Staging(tmp_path / "stage", "namespace")
    ref = stage.copy(source, "source.ply", publisher_uri="https://publisher.example/exact.ply")
    assert source.stat().st_ino == (stage.root / "source.ply").stat().st_ino
    assert stage.files[ref["uri"]]["publication_allowed"] is False
