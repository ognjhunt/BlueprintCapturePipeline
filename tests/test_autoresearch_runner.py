from __future__ import annotations

import json
from pathlib import Path

import pytest

from autoresearch.common import REPO_ROOT, load_target_manifest, run_pytest, validate_target_manifest
from autoresearch.runner import (
    DiffSummary,
    compute_candidate_diff,
    rank_candidate_records,
    validate_diff_summary,
)
from autoresearch.score import compute_total_score, should_accept_candidate


def test_validate_target_manifest_rejects_paths_outside_skill_dir(tmp_path: Path) -> None:
    manifest = load_target_manifest(REPO_ROOT / "autoresearch" / "targets" / "intake_normalizer.json")
    manifest["mutable_paths"] = ["skillpacks/industrial_readiness/skills/intake_normalizer/SKILL.md"]
    manifest["optional_mutable_paths"] = ["docs/README.md"]
    with pytest.raises(ValueError, match="outside the target skill directory"):
        validate_target_manifest(manifest)


def test_diff_validator_rejects_forbidden_paths_and_oversized_diff(tmp_path: Path) -> None:
    base_dir = tmp_path / "base"
    candidate_dir = tmp_path / "candidate"
    allowed = "skillpacks/industrial_readiness/skills/intake_normalizer/SKILL.md"
    forbidden = "docs/forbidden.md"
    (base_dir / Path(allowed)).parent.mkdir(parents=True, exist_ok=True)
    (candidate_dir / Path(allowed)).parent.mkdir(parents=True, exist_ok=True)
    (base_dir / Path(forbidden)).parent.mkdir(parents=True, exist_ok=True)
    (candidate_dir / Path(forbidden)).parent.mkdir(parents=True, exist_ok=True)
    (base_dir / allowed).write_text("line\n", encoding="utf-8")
    (candidate_dir / allowed).write_text("line\n" + ("change\n" * 130), encoding="utf-8")
    (base_dir / forbidden).write_text("old\n", encoding="utf-8")
    (candidate_dir / forbidden).write_text("new\n", encoding="utf-8")

    diff = compute_candidate_diff(base_dir, candidate_dir)
    ok, reasons, forbidden_detected = validate_diff_summary(
        diff,
        allowed_paths={allowed},
        max_changed_files=3,
        max_changed_lines=120,
    )
    assert not ok
    assert forbidden_detected is True
    assert any("forbidden paths" in reason for reason in reasons)
    assert any("diff size" in reason for reason in reasons)


def test_score_calculator_and_acceptance_rules_are_exact() -> None:
    total = compute_total_score(
        test_pass_rate=1.0,
        structured_output_rate=0.9,
        rubric_score=0.8,
        penalties=10.0,
    )
    assert total == pytest.approx(124.0)

    accepted, reason = should_accept_candidate(
        best_score=124.0,
        best_diff_size_lines=12,
        candidate_score=124.0,
        candidate_diff_size_lines=10,
    )
    assert accepted is True
    assert reason == "tie_smaller_diff"

    accepted, reason = should_accept_candidate(
        best_score=124.0,
        best_diff_size_lines=12,
        candidate_score=124.0,
        candidate_diff_size_lines=14,
    )
    assert accepted is False
    assert reason == "tie_larger_or_equal_diff"


def test_rank_candidate_records_keeps_top_three_ordered() -> None:
    ranked = rank_candidate_records(
        [
            {"accepted_iteration": 3, "total_score": 101.0, "diff_size_lines": 12},
            {"accepted_iteration": 1, "total_score": 105.0, "diff_size_lines": 20},
            {"accepted_iteration": 2, "total_score": 105.0, "diff_size_lines": 10},
            {"accepted_iteration": 4, "total_score": 99.0, "diff_size_lines": 5},
        ]
    )
    assert [item["accepted_iteration"] for item in ranked[:3]] == [2, 1, 3]


def test_preflight_repo_tests_pass_for_known_subset() -> None:
    summary = run_pytest(["tests/test_capture_orchestrator.py"], cwd=REPO_ROOT)
    assert summary.exit_code == 0
    assert summary.failed == 0
    assert summary.passed >= 3
