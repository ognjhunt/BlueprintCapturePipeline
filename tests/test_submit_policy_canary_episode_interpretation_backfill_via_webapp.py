from __future__ import annotations

import json

import pytest

from scripts.submit_policy_canary_episode_interpretation_backfill_via_webapp import (
    EpisodeInterpretationBackfillSubmissionError,
    endpoint_for,
    read_exact_sidecar,
    validate_webapp_receipt,
)
from tests.test_policy_canary_episode_interpretation_backfill import (
    _projection,
    _site_record,
)
from blueprint_pipeline.policy_canary_episode_interpretation_backfill import (
    build_policy_canary_episode_interpretation_sidecar,
)


def _sidecar() -> dict:
    return build_policy_canary_episode_interpretation_sidecar(
        source_site_record=_site_record(),
        backfill_projection=_projection(interpreted=True),
        record_id="capture-run-c257ae6e11a18e883637739477e5ded8",
        verified_at_iso="2026-09-03T23:00:00Z",
    )


def test_reads_exact_sidecar_and_builds_run_scoped_endpoint(tmp_path) -> None:
    sidecar = _sidecar()
    path = tmp_path / "sidecar.json"
    body = (json.dumps(sidecar, indent=2, sort_keys=True) + "\n").encode()
    path.write_bytes(body)

    parsed, observed = read_exact_sidecar(path)

    assert parsed == sidecar
    assert observed == body
    assert endpoint_for(
        origin="https://tryblueprint.io",
        run_id=sidecar["source_binding"]["source_run_id"],
    ).endswith(
        "/api/internal/pipeline/capture-task-evaluation-runs/"
        "scene839873-quick10/episode-interpretation-backfills"
    )


def test_validates_created_and_explicit_replay_receipts() -> None:
    sidecar = _sidecar()

    def body(*, replay: bool) -> bytes:
        return json.dumps(
            {
                "schema_version": (
                    "capture_task_evaluation_episode_interpretation_backfill_receipt.v1"
                ),
                "status": "completed",
                "already_exists": replay,
                "run_id": sidecar["source_binding"]["source_run_id"],
                "result_record_id": sidecar["source_binding"]["record_id"],
                "sidecar_digest": sidecar["sidecar_digest"],
                "original_publication_preserved": True,
                "deterministic_scores_unchanged": True,
                "ranking_or_promotion_effect": "none",
            }
        ).encode()

    assert validate_webapp_receipt(
        status_code=201,
        response_body=body(replay=False),
        sidecar=sidecar,
        allow_replay=False,
    )["already_exists"] is False
    assert validate_webapp_receipt(
        status_code=200,
        response_body=body(replay=True),
        sidecar=sidecar,
        allow_replay=True,
    )["already_exists"] is True
    with pytest.raises(
        EpisodeInterpretationBackfillSubmissionError,
        match="replay_not_authorized",
    ):
        validate_webapp_receipt(
            status_code=200,
            response_body=body(replay=True),
            sidecar=sidecar,
            allow_replay=False,
        )
