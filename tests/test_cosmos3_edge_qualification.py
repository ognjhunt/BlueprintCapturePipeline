from __future__ import annotations

import json
from pathlib import Path

import jsonschema

from blueprint_pipeline.cosmos3_edge_qualification import build_cosmos3_edge_qualification


def _write(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _manifest() -> dict[str, object]:
    attempts = []
    stability = []
    for cell in ("a", "b"):
        for mode in ("forward_dynamics", "inverse_dynamics", "reasoning"):
            for repeat in (1, 2):
                attempts.append(
                    {
                        "attempt_id": f"{cell}:{mode}:repeat_{repeat}",
                        "cell_id": cell,
                        "mode": mode,
                        "status": "completed",
                    }
                )
            stability.append(
                {
                    "cell_id": cell,
                    "mode": mode,
                    "repeat_count": 2,
                    "exact_output_digest_stable": True,
                }
            )
    return {
        "schema_version": "cosmos3_edge_experiment_attempt_manifest.v1",
        "status": "completed_advisory",
        "blockers": [],
        "attempts": attempts,
        "output_stability": stability,
        "claim_boundary": {"cosmos3_nano_sc3_qualification_inherited": False},
    }


def test_edge_qualification_requires_blueprint_evaluator_anchors_and_failure_calibration(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "attempts.json"
    receipt = tmp_path / "receipt.json"
    scorecard = tmp_path / "scorecard.json"
    output = tmp_path / "qualification.json"
    _write(manifest, _manifest())
    _write(receipt, {"status": "validated", "model_family": "cosmos3edge"})
    receipt_sha = __import__("hashlib").sha256(receipt.read_bytes()).hexdigest()
    rows = []
    for index, attempt in enumerate(_manifest()["attempts"]):  # type: ignore[index]
        rows.append(
            {
                "attempt_id": attempt["attempt_id"],
                "accepted_anchor_id": "anchor-a" if attempt["cell_id"] == "a" else "anchor-b",
                "anchor_review_status": "accepted",
                "grounding_score": 1.0,
                "abstention_correct": True,
                "expected_rank": index + 1,
                "observed_score": 100 - index,
                "failure_expected": index == 0,
                "failure_detected": index == 0,
            }
        )
    _write(
        scorecard,
        {
            "schema_version": "cosmos3_edge_blueprint_scorecard.v1",
            "frozen_before_scoring": True,
            "configured_evaluator_id": "blueprint-wam-evaluator-v1",
            "evaluator_runtime_receipt_sha256": receipt_sha,
            "attempt_scores": rows,
        },
    )
    result = build_cosmos3_edge_qualification(
        attempt_manifest_path=manifest,
        evaluator_runtime_receipt_path=receipt,
        scorecard_path=scorecard,
        output_path=output,
        expected_evaluator_id="blueprint-wam-evaluator-v1",
    )
    assert result["status"] == "qualified_advisory"
    assert result["metrics"]["spearman_rank_correlation"] == 1.0
    assert result["claim_boundary"]["default_model_change_allowed"] is False
    schema = json.loads(
        (
            Path(__file__).parents[1] / "docs/schemas/cosmos3_edge_qualification.schema.json"
        ).read_text()
    )
    jsonschema.Draft202012Validator(schema).validate(result)


def test_edge_qualification_fails_closed_on_wrong_evaluator(tmp_path: Path) -> None:
    manifest = tmp_path / "attempts.json"
    receipt = tmp_path / "receipt.json"
    scorecard = tmp_path / "scorecard.json"
    _write(manifest, _manifest())
    _write(receipt, {"status": "validated", "model_family": "cosmos3edge"})
    _write(
        scorecard,
        {
            "schema_version": "cosmos3_edge_blueprint_scorecard.v1",
            "frozen_before_scoring": True,
            "configured_evaluator_id": "wrong",
            "evaluator_runtime_receipt_sha256": "x",
            "attempt_scores": [],
        },
    )
    result = build_cosmos3_edge_qualification(
        attempt_manifest_path=manifest,
        evaluator_runtime_receipt_path=receipt,
        scorecard_path=scorecard,
        output_path=tmp_path / "out.json",
        expected_evaluator_id="expected",
    )
    assert result["status"] == "not_qualified"
    assert "edge_configured_evaluator_identity_mismatch" in result["blockers"]
