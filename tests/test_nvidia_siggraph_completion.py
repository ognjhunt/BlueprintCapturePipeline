from __future__ import annotations

from pathlib import Path


from blueprint_pipeline.nvidia_siggraph_completion import build_completion_matrix


def test_completion_matrix_fails_closed_without_verification_receipt(tmp_path: Path) -> None:
    result = build_completion_matrix(
        repository_root=Path(__file__).resolve().parents[1],
        output_path=tmp_path / "matrix.json",
    )
    assert result["implementation_complete"] is False
    assert "verification_receipt_missing" in result["blockers"]
