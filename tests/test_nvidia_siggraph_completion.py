from __future__ import annotations

import json
from pathlib import Path

import jsonschema

from blueprint_pipeline.nvidia_siggraph_completion import build_completion_matrix


def test_completion_matrix_covers_every_memo_lane_without_claiming_external_runs(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    verification = tmp_path / "verification.json"
    verification.write_text(
        json.dumps(
            {
                "schema_version": "nvidia_siggraph_2026_verification_receipt.v1",
                "status": "passed",
                "exit_code": 0,
                "command": "pytest targeted and full suite",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "matrix.json"
    result = build_completion_matrix(
        repository_root=root,
        output_path=output,
        verification_receipt_path=verification,
    )
    assert result["implementation_complete"] is True
    assert result["status"] == "implementation_complete_external_qualification_pending"
    assert len(result["requirements"]) >= 18
    assert all(row["implementation_status"] == "implemented" for row in result["requirements"])
    assert result["external_qualification_state"]["cosmos3_edge_rank_fidelity"] == "unproven"
    assert result["claim_boundary"]["production_promotion_allowed"] is False
    schema = json.loads(
        (root / "docs/schemas/nvidia_siggraph_completion_matrix.schema.json").read_text()
    )
    jsonschema.Draft202012Validator(schema).validate(result)


def test_completion_matrix_fails_closed_without_verification_receipt(tmp_path: Path) -> None:
    result = build_completion_matrix(
        repository_root=Path(__file__).resolve().parents[1],
        output_path=tmp_path / "matrix.json",
    )
    assert result["implementation_complete"] is False
    assert "verification_receipt_missing" in result["blockers"]
