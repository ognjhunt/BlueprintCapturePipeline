from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

from blueprint_pipeline.policy_ranking_successor_cosmos import canonical_sha256


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/experiments/policy_ranking_cosmos3_followup_20260728"


def _json(name: str) -> dict[str, object]:
    return json.loads((EVIDENCE / name).read_text(encoding="utf-8"))


def test_terminal_verdict_keeps_all_components_and_exact_overall_enum() -> None:
    verdict = _json("final_verdict.json")
    assert verdict["overall_verdict"] == "inconclusive"
    assert set(verdict["components"]) == {
        "cosmos_wam_qualification",
        "frozen_benchmark_calibration",
        "captured_site_transfer",
        "economics_and_speed",
    }
    assert {row["verdict"] for row in verdict["components"].values()} == {"inconclusive"}
    assert verdict["ranking_and_abstention"]["evaluator_request_count"] == 0
    assert verdict["ranking_and_abstention"]["benchmark_labels_unsealed"] is False
    assert verdict["historical_oscar_baseline"]["serially_chained_with_cosmos"] is False


def test_causal_report_is_self_bound_and_records_failed_abstaining_screen() -> None:
    report = _json("allocation_3_live_evidence/causal_screen.json")
    expected = report.pop("report_sha256")
    assert canonical_sha256(report) == expected
    assert report["status"] == "completed"
    assert report["screen_passed"] is False
    assert report["confirmatory_power_sufficient"] is False
    assert report["powered_causal_gate_passed"] is False
    assert report["evaluator_eligible"] is False
    assert report["benchmark_labels_seen"] is False
    assert report["gates"] == {
        "nonidentical_scene_response": True,
        "directional_temporal_placebo_rejection": False,
        "seed_robustness": False,
    }
    assert sum(row["temporal_placebo_rejection_pass"] for row in report["rows"]) == 1


def test_raw_runtime_archive_is_bound_and_complete() -> None:
    path = EVIDENCE / "allocation_3_live_evidence/runtime_output.zip"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "c9d1b7ac9c978e48f83643b59bc4fec90c1770ef4d1864da06ffbc19f59c2a6e"
    )
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        assert "wam_runtime_result.json" in names
        assert "immutable_request_journal.jsonl" in names
        assert len([name for name in names if name.startswith("requests/")]) == 10
        assert len([name for name in names if name.startswith("responses/")]) == 10
        assert len([name for name in names if name.startswith("videos/")]) == 10
        runtime = json.loads(archive.read("wam_runtime_result.json"))
    assert runtime["checkpoint"] == "nvidia/Cosmos3-Nano"
    assert runtime["complete_action_condition_seed_matrix_generated"] is True
    assert runtime["provider_scientific_matrix_responses_valid"] == 10
    assert runtime["output_duplicate_hash_count"] == 0


def test_provider_zero_is_explicitly_separate_from_scientific_evidence() -> None:
    proof = _json("provider_zero_evidence.json")
    assert proof["provider_zero_proven"] is True
    assert proof["vast"]["global_live_instance_count"] == 0
    assert proof["digitalocean_spaces"]["task_prefix_object_count"] == 0
    assert proof["digitalocean_spaces"]["exact_bundle_signed_get_http_status_after_delete"] == 404
    assert proof["digitalocean_spaces"]["exact_output_signed_get_http_status_after_delete"] == 404
    assert "not scientific ranking evidence" in proof["claim_boundary"]


def test_terminal_evidence_manifest_binds_every_listed_file() -> None:
    manifest = _json("terminal_evidence_manifest.json")
    for row in manifest["artifacts"]:
        path = EVIDENCE / row["path"]
        assert path.stat().st_size == row["size_bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
    for row in manifest["implementation_and_test_bindings"]:
        path = ROOT / row["path"]
        assert path.stat().st_size == row["size_bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
