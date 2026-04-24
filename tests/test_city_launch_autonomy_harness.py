from argparse import Namespace
import json
from pathlib import Path

from blueprint_pipeline.city_launch_autonomy_harness import run_harness


def _args(tmp_path: Path, *, resume: bool = False) -> Namespace:
    return Namespace(
        city_slug="durham-nc",
        budget_cents=250000,
        capture_path=["iphone", "meta_glasses"],
        run_id="test-run",
        resume=resume,
        execute_local=False,
        include_ios_tests=False,
        include_pipeline_tests=False,
        pipeline_repo=str(tmp_path / "BlueprintCapturePipeline"),
        capture_repo=str(tmp_path / "BlueprintCapture"),
        webapp_repo=str(tmp_path / "Blueprint-WebApp"),
        output_root=str(tmp_path / "ops" / "city-launch-runs"),
    )


def test_autonomous_city_launch_harness_writes_packets_proof_and_blockers(tmp_path: Path) -> None:
    summary = run_harness(_args(tmp_path))
    run_root = tmp_path / "ops" / "city-launch-runs" / "durham-nc" / "test-run"

    assert summary["status"] == "blocked_repo_or_contract_failure"
    assert summary["blocker_count"] > 0
    assert (run_root / "manifest.json").is_file()
    assert (run_root / "proof.launch-proof.json").is_file()
    assert (run_root / "blockers.jsonl").is_file()
    assert (run_root / "work-packets" / "ios_compile_and_real_device.json").is_file()
    assert (run_root / "work-packets" / "meta_glasses_physical_pilot.json").is_file()

    proof = json.loads((run_root / "proof.launch-proof.json").read_text(encoding="utf-8"))
    assert proof["city_slug"] == "durham-nc"
    assert proof["budget_cents"] == 250000
    assert proof["contract_only"] is False
    assert proof["capture_paths"] == ["iphone", "meta_glasses"]
    assert proof["launch_proof_status"] == "blocked_repo_or_contract_failure"

    blockers = (run_root / "blockers.jsonl").read_text(encoding="utf-8").splitlines()
    assert any("release.config_validated_by_archive_script" in line for line in blockers)
    assert any("meta_glasses.physical_device_smoke_passed" in line for line in blockers)


def test_autonomous_city_launch_harness_applies_lane_result_evidence_on_resume(tmp_path: Path) -> None:
    run_harness(_args(tmp_path))
    run_root = tmp_path / "ops" / "city-launch-runs" / "durham-nc" / "test-run"
    lane_results = run_root / "lane-results"
    lane_results.mkdir(parents=True)
    (lane_results / "release.json").write_text(
        json.dumps(
            {
                "lane_id": "ios_compile_and_real_device",
                "status": "passed",
                "evidence": {
                    "release.config_validated_by_archive_script": True,
                    "city.mock_fallback_disabled": True,
                    "city.internal_test_space_disabled": True,
                    "pipeline.capture_descriptor_exists": True,
                    "pipeline.qa_report_exists": True,
                    "pipeline.pipeline_handoff_exists": True,
                    "meta_glasses.video_first_positioning_confirmed": True,
                    "meta_glasses.native_geometry_not_marketed": True,
                    "open_capture.review_gated": True,
                    "open_capture.payout_cents": 0,
                    "open_capture.paid_anywhere_claim_disabled": True,
                    "payouts.marketing_claims_require_stripe_ready": True,
                    "ops.launch_owner": "launch-ops",
                },
            }
        ),
        encoding="utf-8",
    )

    summary = run_harness(_args(tmp_path, resume=True))

    assert summary["status"] == "blocked_external_dependency"
    blockers = (run_root / "blockers.jsonl").read_text(encoding="utf-8").splitlines()
    assert not any("release.config_validated_by_archive_script" in line for line in blockers)
