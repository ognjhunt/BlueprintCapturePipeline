from __future__ import annotations

from pathlib import Path

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.fresh_scene_artifixer_candidate_preparation import (
    REQUEST_SCHEMA_VERSION,
    materialize_fresh_scene_artifixer_candidate_preparation,
)
from tests.test_public_scene_segment_mask_repair_preflight import _fixture


def test_prepares_two_task_artifixer_inputs_without_model_execution(tmp_path: Path) -> None:
    cutout, authority, _masks = _fixture(tmp_path, task_count=2)
    request = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "segment_cutout_set_path": str(cutout),
        "execution_authority_path": str(authority),
        "selected_task_ids": ["task_1", "task_2"],
        "object_absent_reference_receipt_paths": [],
        "request_digest": "",
    }
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    result = materialize_fresh_scene_artifixer_candidate_preparation(
        request=request, output_root=tmp_path / "prepared"
    )

    assert result["task_count"] == 2
    assert result["next_required_stage"] == "semantic_teacher_receipts"
    assert result["semantic_teacher_execution_started"] is False
    assert result["artifixer3d_execution_started"] is False
    assert result["provider_mutations_performed"] == 0
    assert result["canonical_source_altered"] is False
