from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline.policy_episode_graded_report import build_policy_canary_graded_report_sidecar
from scripts.submit_policy_canary_graded_report_via_webapp import (
    GradedReportSubmissionError,
    endpoint_for,
    read_exact_sidecar,
    validate_webapp_receipt,
)
from tests.test_policy_episode_graded_report import _site_record


def _sidecar(tmp_path: Path) -> dict:
    record, source, evidence = _site_record(tmp_path, correction=True)
    return build_policy_canary_graded_report_sidecar(
        source_site_record=record,
        source_result_path=source,
        evidence_root=evidence,
        record_id="capture-run-c257ae6e11a18e883637739477e5ded8",
        generated_at_iso="2026-09-23T21:00:00Z",
    )


def _receipt(sidecar: dict, *, replay: bool, **overrides: object) -> bytes:
    value = {
        "schema_version": "capture_task_evaluation_graded_report_receipt.v1",
        "already_exists": replay,
        "run_id": sidecar["source_binding"]["source_run_id"],
        "result_record_id": sidecar["source_binding"]["record_id"],
        "sidecar_digest": sidecar["sidecar_digest"],
        "episode_count": len(sidecar["episodes"]),
        "original_publication_preserved": True,
        "deterministic_scores_unchanged": True,
        "ranking_or_promotion_effect": "none",
    }
    value.update(overrides)
    return json.dumps(value).encode()


def test_reads_exact_sidecar_and_builds_run_scoped_endpoint(tmp_path: Path) -> None:
    sidecar = _sidecar(tmp_path)
    path = tmp_path / "sidecar.json"
    body = (json.dumps(sidecar, indent=2, sort_keys=True) + "\n").encode()
    path.write_bytes(body)

    parsed, observed = read_exact_sidecar(path)

    assert parsed == sidecar
    assert observed == body
    assert endpoint_for(origin="https://tryblueprint.io", run_id="scene-1/quick10").endswith(
        "/api/internal/pipeline/capture-task-evaluation-runs/scene-1%2Fquick10/graded-reports"
    )
    with pytest.raises(GradedReportSubmissionError, match="origin_invalid"):
        endpoint_for(origin="http://tryblueprint.io", run_id="scene-1")


def test_an_edited_sidecar_is_refused_before_it_is_signed(tmp_path: Path) -> None:
    sidecar = _sidecar(tmp_path)
    sidecar["candidates"][0]["mean_graded_score"] = 0.1
    path = tmp_path / "sidecar.json"
    path.write_text(json.dumps(sidecar))
    with pytest.raises(GradedReportSubmissionError, match="request_invalid"):
        read_exact_sidecar(path)


def test_validates_created_and_explicit_replay_receipts(tmp_path: Path) -> None:
    sidecar = _sidecar(tmp_path)
    assert validate_webapp_receipt(
        status_code=201, response_body=_receipt(sidecar, replay=False),
        sidecar=sidecar, allow_replay=False,
    )["already_exists"] is False
    assert validate_webapp_receipt(
        status_code=200, response_body=_receipt(sidecar, replay=True),
        sidecar=sidecar, allow_replay=True,
    )["already_exists"] is True
    with pytest.raises(GradedReportSubmissionError, match="replay_not_authorized"):
        validate_webapp_receipt(
            status_code=200, response_body=_receipt(sidecar, replay=True),
            sidecar=sidecar, allow_replay=False,
        )
    with pytest.raises(GradedReportSubmissionError, match="response_invalid"):
        validate_webapp_receipt(
            status_code=201,
            response_body=_receipt(sidecar, replay=False, ranking_or_promotion_effect="ranked"),
            sidecar=sidecar, allow_replay=False,
        )
