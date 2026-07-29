from __future__ import annotations

from pathlib import Path

import pytest

from blueprint_pipeline.policy_ranking_task_instruction_extraction import (
    extract_task_instruction,
)


def test_extracts_only_allowlisted_instruction_before_outcomes(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata.yaml"
    metadata.write_text(
        "session_id: session-a\n"
        "language_instruction: put the bowl in the plate\n"
        "success: true\n"
        "outcome: completed\n",
        encoding="utf-8",
    )

    receipt = extract_task_instruction(metadata, session_id="session-a")

    assert receipt["task_instruction"] == "put the bowl in the plate"
    assert receipt["instruction_line_number"] == 2
    assert receipt["bytes_streamed_before_instruction_found"] < metadata.stat().st_size
    assert receipt["access_contract"]["yaml_document_deserialized"] is False
    assert receipt["access_contract"]["outcome_fields_parsed"] is False
    assert "success" not in receipt and "outcome" not in receipt


def test_supports_quoted_scalar_and_rejects_outcome_before_instruction(tmp_path: Path) -> None:
    quoted = tmp_path / "quoted.yaml"
    quoted.write_text('language_instruction: "pick up the bottle"\n', encoding="utf-8")
    assert (
        extract_task_instruction(quoted, session_id="session-a")["task_instruction"]
        == "pick up the bottle"
    )

    blocked = tmp_path / "blocked.yaml"
    blocked.write_text(
        "success: false\nlanguage_instruction: pick up the bottle\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="outcome_like_field_precedes_instruction"):
        extract_task_instruction(blocked, session_id="session-a")
