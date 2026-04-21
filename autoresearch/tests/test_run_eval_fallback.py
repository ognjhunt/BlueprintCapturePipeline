from __future__ import annotations

from pathlib import Path

from autoresearch.common import REPO_ROOT, load_target_manifest
from autoresearch.run_eval import evaluate_candidate


def _candidate_dir_for(target: str, tmp_path: Path) -> tuple[Path, Path]:
    manifest_path = REPO_ROOT / "autoresearch" / "targets" / f"{target}.json"
    manifest = load_target_manifest(manifest_path)
    candidate_dir = tmp_path / target
    for relative_path in manifest["mutable_paths"]:
        path = candidate_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {target}\n", encoding="utf-8")
    return manifest_path, candidate_dir


def test_evaluate_candidate_falls_back_for_intake_normalizer(monkeypatch, tmp_path: Path) -> None:
    manifest_path, candidate_dir = _candidate_dir_for("intake_normalizer", tmp_path)

    def raise_quota_error(**_kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("codex rate limit")

    monkeypatch.setattr("autoresearch.run_eval._invoke_generation_agent", raise_quota_error)

    output_dir = tmp_path / "eval"
    payload = evaluate_candidate(
        target_manifest_path=manifest_path,
        candidate_dir=candidate_dir,
        output_dir=output_dir,
    )

    assert payload["fallback_used"] is True
    assert payload["case_results"][0]["fallback_used"] is True
    assert payload["pytest"]["failed"] == 0


def test_evaluate_candidate_falls_back_for_readiness_report_writer(monkeypatch, tmp_path: Path) -> None:
    manifest_path, candidate_dir = _candidate_dir_for("readiness_report_writer", tmp_path)

    def raise_quota_error(**_kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("codex rate limit")

    monkeypatch.setattr("autoresearch.run_eval._invoke_generation_agent", raise_quota_error)

    output_dir = tmp_path / "eval"
    payload = evaluate_candidate(
        target_manifest_path=manifest_path,
        candidate_dir=candidate_dir,
        output_dir=output_dir,
    )

    memo = (output_dir / "cases" / "not_ready_yet" / "readiness_report.md").read_text(encoding="utf-8")
    assert payload["fallback_used"] is True
    assert "## Human Signoff Boundary" in memo
    assert "## Required Human Actions" in memo


def test_evaluate_candidate_falls_back_for_recapture_planner(monkeypatch, tmp_path: Path) -> None:
    manifest_path, candidate_dir = _candidate_dir_for("recapture_planner", tmp_path)

    def raise_quota_error(**_kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("codex rate limit")

    monkeypatch.setattr("autoresearch.run_eval._invoke_generation_agent", raise_quota_error)

    output_dir = tmp_path / "eval"
    payload = evaluate_candidate(
        target_manifest_path=manifest_path,
        candidate_dir=candidate_dir,
        output_dir=output_dir,
    )

    assert payload["fallback_used"] is True
    assert payload["case_results"][0]["structured_checks"]["schema_valid"] is True
    assert payload["case_results"][0]["structured_checks"]["required_fields_present"] is True
