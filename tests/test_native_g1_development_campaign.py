"""A team movement choice must stay bound to the book task packet."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import cross_runtime_canonical_digest
from blueprint_pipeline.native_g1_development_campaign import (
    _chosen_movement_handoff,
    _movement_handoff,
)
from tests.test_native_g1_development_selection import _packet_choice_inputs


def _book_handoff(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    args, choice = _packet_choice_inputs(tmp_path, monkeypatch)
    handoff = {
        "schema_version": "task_evaluation_packet_policy_handoff.v1",
        "claim_ceiling": "planning_only",
        "setup": json.loads(args["setup_path"].read_text(encoding="utf-8")),
        "choice": choice,
    }
    handoff["handoff_digest"] = cross_runtime_canonical_digest(
        handoff, digest_field="handoff_digest"
    )
    return handoff


def test_explicit_movement_handoff_binds_to_same_team_packet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    book = _book_handoff(tmp_path, monkeypatch)
    movement = _movement_handoff(book)
    path = tmp_path / "movement_handoff.json"
    path.write_text(json.dumps(movement), encoding="utf-8")

    chosen = _chosen_movement_handoff(book, path)

    assert chosen == movement
    assert chosen["setup"]["setup_digest"] == book["setup"]["setup_digest"]
    assert chosen["choice"]["objective_id"] == "g1_navigation_goal"


def test_book_choice_cannot_be_reused_as_movement_choice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    book = _book_handoff(tmp_path, monkeypatch)
    path = tmp_path / "movement_handoff.json"
    path.write_text(json.dumps(book), encoding="utf-8")

    with pytest.raises(ValueError, match="g1_campaign_movement_choice_or_setup_mismatch"):
        _chosen_movement_handoff(book, path)
