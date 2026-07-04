from argparse import Namespace
from io import BytesIO
import json
from pathlib import Path
import stat

import blueprint_pipeline.city_launch_autonomy_harness as harness
from blueprint_pipeline.city_launch_autonomy_harness import run_harness

import pytest

pytestmark = pytest.mark.slow


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
        capture_root=None,
        proof_file=[],
        include_webapp_city_status=False,
    )


def _write_archive_script(capture_repo: Path, body: str) -> None:
    scripts_dir = capture_repo / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    script = scripts_dir / "archive_external_alpha.sh"
    script.write_text(body, encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)


class _FakeHttpResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self) -> "_FakeHttpResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_autonomous_city_launch_harness_writes_packets_proof_and_blockers(tmp_path: Path) -> None:
    summary = run_harness(_args(tmp_path))
    run_root = tmp_path / "ops" / "city-launch-runs" / "durham-nc" / "test-run"

    assert summary["status"] == "blocked_repo_or_contract_failure"
    assert summary["blocker_count"] > 0
    assert (run_root / "manifest.json").is_file()
    assert (run_root / "proof.launch-proof.json").is_file()
    assert (run_root / "blockers.jsonl").is_file()
    assert (run_root / "launch-gap-report.json").is_file()
    assert (run_root / "work-packets" / "ios_compile_and_real_device.json").is_file()
    assert (run_root / "work-packets" / "meta_glasses_physical_pilot.json").is_file()
    assert (run_root / "lane-results" / "privacy_safe_provider.not-executed.json").is_file()

    proof = json.loads((run_root / "proof.launch-proof.json").read_text(encoding="utf-8"))
    assert proof["city_slug"] == "durham-nc"
    assert proof["budget_cents"] == 250000
    assert proof["contract_only"] is False
    assert proof["capture_paths"] == ["iphone", "meta_glasses"]
    assert proof["launch_proof_status"] == "blocked_repo_or_contract_failure"

    blockers = (run_root / "blockers.jsonl").read_text(encoding="utf-8").splitlines()
    assert any("release.config_validated_by_archive_script" in line for line in blockers)
    assert any("meta_glasses.physical_device_smoke_passed" in line for line in blockers)

    gap_report = json.loads((run_root / "launch-gap-report.json").read_text(encoding="utf-8"))
    assert gap_report["schema_version"] == "city-launch-gap-report.v1"
    assert gap_report["first_blocker"]["proof_field"] == "release.config_validated_by_archive_script"
    assert any(
        gap["proof_field"] == "city.backend_supported"
        and gap["dependency_class"] == "live_city_backend"
        for gap in gap_report["external_dependencies"]
    )
    assert any("capture_descriptor.json with site_id" in item for item in gap_report["expected_capture_root_shape"])


def test_default_launch_proof_separates_repo_safe_payout_guardrails_from_live_provider_evidence() -> None:
    proof = harness.default_proof("durham-nc", 250000, ["iphone"])

    assert proof["open_capture"]["review_gated"] is True
    assert proof["open_capture"]["paid_anywhere_claim_disabled"] is True
    assert proof["payouts"]["provider_name"] == "stripe"
    assert proof["payouts"]["contract_readiness_not_live_readiness"] is True
    assert proof["payouts"]["live_payout_execution_human_gate"] is True
    assert proof["payouts"]["live_provider_ready"] is False
    assert proof["ops"]["human_finance_review_gate"] is True
    assert proof["ops"]["payout_exception_monitor_repo_contract"] is True

    blockers = harness.build_blockers(proof)
    live_provider_blocker = next(
        blocker for blocker in blockers if blocker["proof_field"] == "payouts.live_provider_ready"
    )
    assert live_provider_blocker["human_required"] is True


def test_autonomous_city_launch_harness_applies_lane_result_evidence_on_resume(tmp_path: Path) -> None:
    run_harness(_args(tmp_path))
    run_root = tmp_path / "ops" / "city-launch-runs" / "durham-nc" / "test-run"
    lane_results = run_root / "lane-results"
    lane_results.mkdir(parents=True, exist_ok=True)
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
                    "pipeline.pipeline_processed_capture": True,
                    "privacy_provider.final_walkthrough_uri": "gs://bucket/privacy/final_walkthrough.mov",
                    "privacy_provider.worldlabs_input_uri": "gs://bucket/pipeline/worldlabs_input/worldlabs_input.mp4",
                    "privacy_provider.raw_bypass_disabled": True,
                    "retrieval.dense_index_exists": True,
                    "retrieval.site_reference_manifest_exists": True,
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


def test_autonomous_city_launch_harness_persists_local_lane_results(tmp_path: Path, monkeypatch) -> None:
    def _run_command(command):  # type: ignore[no-untyped-def]
        return {
            "id": command.id,
            "status": "passed",
            "exit_code": 0,
            "command": command.command,
            "cwd": command.cwd,
            "stdout_tail": "",
            "stderr_tail": "",
            "proof_on_pass": list(command.proof_on_pass),
        }

    monkeypatch.setattr("blueprint_pipeline.city_launch_autonomy_harness.run_command", _run_command)

    args = _args(tmp_path)
    args.execute_local = True
    args.include_pipeline_tests = True

    summary = run_harness(args)
    run_root = tmp_path / "ops" / "city-launch-runs" / "durham-nc" / "test-run"

    lane_result_path = run_root / "lane-results" / "site_identity_dense_export.local-execution.json"
    proof = json.loads((run_root / "proof.launch-proof.json").read_text(encoding="utf-8"))
    lane_result = json.loads(lane_result_path.read_text(encoding="utf-8"))

    assert summary["blocker_count"] > 0
    assert lane_result["status"] == "passed"
    assert lane_result["evidence"]["harness.local_execution.executed_command_count"] == 1
    assert proof["harness"]["local_execution"]["executed_command_count"] >= 1
    assert proof["harness"]["site_identity_dense_export_tests_passed"] is True


def test_autonomous_city_launch_harness_ignores_not_executed_and_contract_only_evidence(tmp_path: Path) -> None:
    run_harness(_args(tmp_path))
    run_root = tmp_path / "ops" / "city-launch-runs" / "durham-nc" / "test-run"
    lane_results = run_root / "lane-results"
    (lane_results / "release.not-executed.json").write_text(
        json.dumps(
            {
                "lane_id": "ios_compile_and_real_device",
                "status": "succeeded",
                "evidence": {"release.config_validated_by_archive_script": True},
            }
        ),
        encoding="utf-8",
    )
    (lane_results / "contract-only.json").write_text(
        json.dumps(
            {
                "lane_id": "pipeline",
                "status": "contract_only",
                "contract_only": True,
                "evidence": {"pipeline.pipeline_processed_capture": True},
            }
        ),
        encoding="utf-8",
    )

    run_harness(_args(tmp_path, resume=True))

    proof = json.loads((run_root / "proof.launch-proof.json").read_text(encoding="utf-8"))
    assert proof["release"]["config_validated_by_archive_script"] is False
    assert proof["pipeline"]["pipeline_processed_capture"] is False


def test_capture_root_evidence_rejects_placeholder_webapp_sync(tmp_path: Path) -> None:
    capture_root = tmp_path / "capture"
    pipeline_root = capture_root / "pipeline"
    eval_root = pipeline_root / "evaluation_prep"
    pipeline_root.mkdir(parents=True)
    eval_root.mkdir(parents=True)
    (capture_root / "capture_descriptor.json").write_text("{}", encoding="utf-8")
    (eval_root / "site_world_registration.json").write_text(
        json.dumps({"runtime_base_url": "https://runtime.test"}),
        encoding="utf-8",
    )
    (pipeline_root / "webapp_sync_result.json").write_text(
        json.dumps(
            {
                "status": "succeeded",
                "placeholder_fallback_allowed": True,
                "webapp_response_ids": {"listing_id": "placeholder-listing"},
                "buyer_access_check": {
                    "buyer_access_checked": True,
                    "buyer_accessible": True,
                },
            }
        ),
        encoding="utf-8",
    )
    args = _args(tmp_path, resume=False)
    args.capture_root = str(capture_root)

    run_harness(args)

    run_root = tmp_path / "ops" / "city-launch-runs" / "durham-nc" / "test-run"
    proof = json.loads((run_root / "proof.launch-proof.json").read_text(encoding="utf-8"))
    assert proof["hosted_session"]["runtime_url"] == "https://runtime.test"
    assert proof["hosted_session"]["webapp_listing_id"] == ""
    assert proof["hosted_session"]["buyer_access_checked"] is False


def test_capture_root_evidence_uses_canonical_retrieval_stage_paths(tmp_path: Path) -> None:
    capture_root = tmp_path / "local-blueprint" / "scenes" / "scene-1" / "captures" / "capture-1"
    site_memory = tmp_path / "local-blueprint" / "sites" / "site-1" / "reference_memory"
    (capture_root / "world_model_export").mkdir(parents=True)
    site_memory.mkdir(parents=True)
    (capture_root / "capture_descriptor.json").write_text(
        json.dumps({"site_id": "site-1"}),
        encoding="utf-8",
    )
    (capture_root / "world_model_export" / "dense_index.jsonl").write_text(
        '{"reference_id":"ref-1","included_in_index":true}\n',
        encoding="utf-8",
    )
    (site_memory / "site_reference_manifest.json").write_text(
        json.dumps({"site_id": "site-1", "reference_count": 1}),
        encoding="utf-8",
    )
    args = _args(tmp_path, resume=False)
    args.capture_root = str(capture_root)

    run_harness(args)

    run_root = tmp_path / "ops" / "city-launch-runs" / "durham-nc" / "test-run"
    proof = json.loads((run_root / "proof.launch-proof.json").read_text(encoding="utf-8"))
    lane_result = json.loads(
        (run_root / "lane-results" / "site_identity_dense_export.capture-root-evidence.json").read_text(
            encoding="utf-8"
        )
    )
    assert proof["retrieval"]["dense_index_exists"] is True
    assert proof["retrieval"]["site_reference_manifest_exists"] is True
    assert lane_result["blockers"] == ["privacy_safe_walkthrough_missing_for_retrieval_index"]


def test_capture_root_evidence_reports_retrieval_input_shape_gaps(tmp_path: Path) -> None:
    capture_root = tmp_path / "capture"
    capture_root.mkdir(parents=True)
    (capture_root / "capture_descriptor.json").write_text("{}", encoding="utf-8")
    args = _args(tmp_path, resume=False)
    args.capture_root = str(capture_root)

    run_harness(args)

    run_root = tmp_path / "ops" / "city-launch-runs" / "durham-nc" / "test-run"
    lane_result = json.loads(
        (run_root / "lane-results" / "site_identity_dense_export.capture-root-evidence.json").read_text(
            encoding="utf-8"
        )
    )
    assert "retrieval.dense_index_exists" in lane_result["blockers"]
    assert "retrieval.site_reference_manifest_exists" in lane_result["blockers"]
    assert "capture_descriptor_missing_site_id_or_metadata_site_identity_site_id" in lane_result["blockers"]
    assert "privacy_safe_walkthrough_missing_for_retrieval_index" in lane_result["blockers"]


def test_capture_root_input_shape_gap_reports_external_dependency_status(tmp_path: Path) -> None:
    capture_root = tmp_path / "capture"
    capture_root.mkdir(parents=True)
    (capture_root / "capture_descriptor.json").write_text("{}", encoding="utf-8")
    run_root = tmp_path / "ops" / "city-launch-runs" / "durham-nc" / "test-run"
    lane_results = run_root / "lane-results"
    lane_results.mkdir(parents=True, exist_ok=True)
    (lane_results / "zz-live-evidence.json").write_text(
        json.dumps(
            {
                "lane_id": "proof-fixture",
                "status": "passed",
                "evidence": {
                    "release.config_validated_by_archive_script": True,
                    "city.backend_supported": True,
                    "city.live_approved_job_count": 1,
                    "city.live_capture_target_count": 1,
                    "city.mock_fallback_disabled": True,
                    "city.internal_test_space_disabled": True,
                    "capture.real_device_capture_uploaded": True,
                    "capture.capture_submissions_document_exists": True,
                    "capture.raw_upload_complete_exists": True,
                    "pipeline.capture_descriptor_exists": True,
                    "pipeline.qa_report_exists": True,
                    "pipeline.pipeline_handoff_exists": True,
                    "pipeline.pubsub_handoff_succeeded": True,
                    "pipeline.pipeline_processed_capture": True,
                    "privacy_provider.final_walkthrough_uri": "gs://bucket/privacy/final_walkthrough.mov",
                    "privacy_provider.worldlabs_input_uri": "gs://bucket/privacy/worldlabs_input.mp4",
                    "privacy_provider.raw_bypass_disabled": True,
                    "hosted_session.runtime_url": "https://runtime.test/session",
                    "hosted_session.webapp_listing_id": "listing-1",
                    "hosted_session.buyer_access_checked": True,
                    "meta_glasses.physical_device_smoke_passed": True,
                    "meta_glasses.video_first_positioning_confirmed": True,
                    "meta_glasses.native_geometry_not_marketed": True,
                    "open_capture.review_gated": True,
                    "open_capture.paid_anywhere_claim_disabled": True,
                    "payouts.backend_configured": True,
                    "payouts.stripe_state_checked": True,
                    "payouts.provider_name": "stripe",
                    "payouts.provider_state_checked": True,
                    "payouts.live_provider_ready": True,
                    "payouts.contract_readiness_not_live_readiness": True,
                    "payouts.live_payout_execution_human_gate": True,
                    "payouts.identity_kyc_state_documented": True,
                    "payouts.background_check_state_documented": True,
                    "payouts.marketing_claims_require_stripe_ready": True,
                    "ops.launch_owner": "launch-ops",
                    "ops.failed_upload_monitor": True,
                    "ops.submission_registration_monitor": True,
                    "ops.push_device_sync_monitor": True,
                    "ops.bridge_pipeline_monitor": True,
                    "ops.payout_exception_monitor_repo_contract": True,
                    "ops.human_finance_review_gate": True,
                    "ops.payout_exception_monitor": True,
                    "ops.session_events_queryable": True,
                    "ops.cloud_logging_handoff_alert": True,
                },
            }
        ),
        encoding="utf-8",
    )
    args = _args(tmp_path, resume=False)
    args.capture_root = str(capture_root)

    summary = run_harness(args)

    gap_report = json.loads((run_root / "launch-gap-report.json").read_text(encoding="utf-8"))
    assert summary["status"] == "blocked_external_dependency"
    assert summary["first_blocker"]["proof_field"] == "retrieval.dense_index_exists"
    assert gap_report["repo_or_contract_gaps"][0]["dependency_class"] == "capture_root_or_retrieval_export"
    assert gap_report["repo_or_contract_gaps"][0]["lane_result_paths"] == [
        str(run_root / "lane-results" / "site_identity_dense_export.capture-root-evidence.json")
    ]


def test_internal_glasses_status_requires_no_iphone_path_and_downstream_proof(tmp_path: Path) -> None:
    run_root = tmp_path / "ops" / "city-launch-runs" / "durham-nc" / "test-run"
    lane_results = run_root / "lane-results"
    lane_results.mkdir(parents=True, exist_ok=True)
    (lane_results / "glasses-downstream.json").write_text(
        json.dumps(
            {
                "lane_id": "meta_glasses",
                "status": "passed",
                "evidence": {
                    "pipeline.capture_descriptor_exists": True,
                    "pipeline.qa_report_exists": True,
                    "pipeline.pipeline_handoff_exists": True,
                    "pipeline.pubsub_handoff_succeeded": True,
                    "pipeline.pipeline_processed_capture": True,
                    "privacy_provider.final_walkthrough_uri": "gs://bucket/privacy/glasses_final_walkthrough.mov",
                    "privacy_provider.worldlabs_input_uri": "gs://bucket/privacy/glasses_worldlabs_input.mp4",
                    "privacy_provider.raw_bypass_disabled": True,
                    "retrieval.dense_index_exists": True,
                    "retrieval.site_reference_manifest_exists": True,
                    "hosted_session.runtime_url": "https://runtime.test/glasses",
                    "hosted_session.webapp_listing_id": "internal-listing-1",
                    "hosted_session.buyer_access_checked": True,
                    "meta_glasses.physical_device_smoke_passed": True,
                    "meta_glasses.video_first_positioning_confirmed": True,
                    "meta_glasses.native_geometry_not_marketed": True,
                },
            }
        ),
        encoding="utf-8",
    )
    args = _args(tmp_path, resume=False)
    args.capture_path = ["meta_glasses"]

    summary = run_harness(args)

    proof = json.loads((run_root / "proof.launch-proof.json").read_text(encoding="utf-8"))
    assert summary["status"] == "ready_for_internal_glasses_pilot"
    assert proof["capture_paths"] == ["meta_glasses"]
    assert proof["launch_proof_status"] == "ready_for_internal_glasses_pilot"


def test_internal_glasses_status_does_not_mask_iphone_launch_blockers(tmp_path: Path) -> None:
    run_root = tmp_path / "ops" / "city-launch-runs" / "durham-nc" / "test-run"
    lane_results = run_root / "lane-results"
    lane_results.mkdir(parents=True, exist_ok=True)
    (lane_results / "meta-ready.json").write_text(
        json.dumps(
            {
                "lane_id": "meta_glasses",
                "status": "passed",
                "evidence": {
                    "pipeline.capture_descriptor_exists": True,
                    "pipeline.qa_report_exists": True,
                    "pipeline.pipeline_handoff_exists": True,
                    "pipeline.pubsub_handoff_succeeded": True,
                    "pipeline.pipeline_processed_capture": True,
                    "privacy_provider.final_walkthrough_uri": "gs://bucket/privacy/final_walkthrough.mov",
                    "privacy_provider.worldlabs_input_uri": "gs://bucket/privacy/worldlabs_input.mp4",
                    "privacy_provider.raw_bypass_disabled": True,
                    "retrieval.dense_index_exists": True,
                    "retrieval.site_reference_manifest_exists": True,
                    "hosted_session.runtime_url": "https://runtime.test/session",
                    "hosted_session.webapp_listing_id": "listing-1",
                    "hosted_session.buyer_access_checked": True,
                    "meta_glasses.physical_device_smoke_passed": True,
                    "meta_glasses.video_first_positioning_confirmed": True,
                    "meta_glasses.native_geometry_not_marketed": True,
                },
            }
        ),
        encoding="utf-8",
    )
    args = _args(tmp_path, resume=False)
    args.capture_path = ["iphone", "meta_glasses"]

    summary = run_harness(args)

    proof = json.loads((run_root / "proof.launch-proof.json").read_text(encoding="utf-8"))
    assert summary["status"] == "blocked_repo_or_contract_failure"
    assert proof["launch_proof_status"] == "blocked_repo_or_contract_failure"
    assert summary["first_blocker"]["proof_field"] == "release.config_validated_by_archive_script"


def test_status_treats_malformed_capture_paths_as_not_market_ready() -> None:
    proof = harness.default_proof("durham-nc", 250000, ["iphone"])
    proof["capture_paths"] = "iphone"
    blockers = harness.build_blockers(proof)

    assert harness.determine_status(proof, blockers) == "blocked_repo_or_contract_failure"


def test_webapp_city_status_route_merges_supported_city(tmp_path: Path, monkeypatch) -> None:
    def fake_urlopen(url: str, timeout: int):
        assert url == "https://webapp.test/api/public/launch/status?city=Durham&state_code=NC"
        assert timeout == harness.WEBAPP_STATUS_TIMEOUT_SECONDS
        return _FakeHttpResponse(
            {
                "ok": True,
                "currentCity": {
                    "citySlug": "durham-nc",
                    "isSupported": True,
                    "status": "live",
                },
                "supportedCities": [
                    {"citySlug": "durham-nc"},
                ],
            }
        )

    monkeypatch.setenv("PIPELINE_SYNC_WEBAPP_URL", "https://webapp.test/api/internal/pipeline/attachments")
    monkeypatch.setattr(harness.urllib.request, "urlopen", fake_urlopen)
    args = _args(tmp_path)
    args.include_webapp_city_status = True

    run_harness(args)

    run_root = tmp_path / "ops" / "city-launch-runs" / "durham-nc" / "test-run"
    proof = json.loads((run_root / "proof.launch-proof.json").read_text(encoding="utf-8"))
    lane_result = json.loads(
        (run_root / "lane-results" / "city_backend.webapp-status-route.json").read_text(
            encoding="utf-8"
        )
    )
    assert proof["city"]["backend_supported"] is True
    assert lane_result["status"] == "succeeded"
    assert lane_result["blockers"] == []


def test_webapp_city_status_route_rejects_planned_city(tmp_path: Path, monkeypatch) -> None:
    def fake_urlopen(_url: str, timeout: int):
        assert timeout == harness.WEBAPP_STATUS_TIMEOUT_SECONDS
        return _FakeHttpResponse(
            {
                "ok": True,
                "currentCity": {
                    "citySlug": "durham-nc",
                    "isSupported": False,
                    "status": "planned",
                },
                "supportedCities": [],
            }
        )

    monkeypatch.setenv("PIPELINE_SYNC_WEBAPP_URL", "https://webapp.test")
    monkeypatch.setattr(harness.urllib.request, "urlopen", fake_urlopen)
    args = _args(tmp_path)
    args.include_webapp_city_status = True

    run_harness(args)

    run_root = tmp_path / "ops" / "city-launch-runs" / "durham-nc" / "test-run"
    proof = json.loads((run_root / "proof.launch-proof.json").read_text(encoding="utf-8"))
    lane_result = json.loads(
        (run_root / "lane-results" / "city_backend.webapp-status-route.json").read_text(
            encoding="utf-8"
        )
    )
    assert proof["city"]["backend_supported"] is False
    assert lane_result["status"] == "blocked"
    assert "webapp_status_route_city_not_live_supported" in lane_result["blockers"]


def test_webapp_city_status_route_records_http_error_body(tmp_path: Path, monkeypatch) -> None:
    def fake_urlopen(url: str, timeout: int):
        assert timeout == harness.WEBAPP_STATUS_TIMEOUT_SECONDS
        raise harness.urllib.error.HTTPError(
            url,
            500,
            "Internal Server Error",
            {},
            BytesIO(json.dumps({"error": "Firebase Admin SDK missing credentials"}).encode("utf-8")),
        )

    monkeypatch.setenv("PIPELINE_SYNC_WEBAPP_URL", "https://webapp.test")
    monkeypatch.setattr(harness.urllib.request, "urlopen", fake_urlopen)
    args = _args(tmp_path)
    args.include_webapp_city_status = True

    run_harness(args)

    run_root = tmp_path / "ops" / "city-launch-runs" / "durham-nc" / "test-run"
    lane_result = json.loads(
        (run_root / "lane-results" / "city_backend.webapp-status-route.json").read_text(
            encoding="utf-8"
        )
    )
    assert "webapp_status_route_http_500" in lane_result["blockers"]
    assert "webapp_status_route_error:Firebase Admin SDK missing credentials" in lane_result["blockers"]


def test_webapp_city_status_route_rejects_unavailable_source_status(tmp_path: Path, monkeypatch) -> None:
    def fake_urlopen(_url: str, timeout: int):
        assert timeout == harness.WEBAPP_STATUS_TIMEOUT_SECONDS
        return _FakeHttpResponse(
            {
                "ok": True,
                "currentCity": {
                    "citySlug": "durham-nc",
                    "isSupported": True,
                    "status": "live",
                },
                "supportedCities": [
                    {"citySlug": "durham-nc"},
                ],
                "sourceStatus": {
                    "cityLaunchActivations": "unavailable",
                    "cityLaunchProspects": "unavailable",
                    "cityLaunchCandidateSignals": "available",
                    "warnings": ["cityLaunchActivations:8 RESOURCE_EXHAUSTED: Quota exceeded."],
                },
            }
        )

    monkeypatch.setenv("PIPELINE_SYNC_WEBAPP_URL", "https://webapp.test")
    monkeypatch.setattr(harness.urllib.request, "urlopen", fake_urlopen)
    args = _args(tmp_path)
    args.include_webapp_city_status = True

    run_harness(args)

    run_root = tmp_path / "ops" / "city-launch-runs" / "durham-nc" / "test-run"
    proof = json.loads((run_root / "proof.launch-proof.json").read_text(encoding="utf-8"))
    lane_result = json.loads(
        (run_root / "lane-results" / "city_backend.webapp-status-route.json").read_text(
            encoding="utf-8"
        )
    )
    assert proof["city"]["backend_supported"] is False
    assert lane_result["status"] == "blocked"
    assert "webapp_status_route_source_unavailable" in lane_result["blockers"]
    assert "RESOURCE_EXHAUSTED" in "\n".join(lane_result["blockers"])


def test_autonomous_city_launch_harness_merges_real_cross_repo_proof_file(tmp_path: Path) -> None:
    proof_file = tmp_path / "capture.launch-proof.json"
    proof_file.write_text(
        json.dumps(
            {
                "schema_version": "city-launch-proof.v1",
                "contract_only": False,
                "city_slug": "durham-nc",
                "city": {
                    "backend_supported": True,
                    "live_approved_job_count": 2,
                    "live_capture_target_count": 3,
                },
                "capture": {
                    "capture_submissions_document_exists": True,
                    "raw_upload_complete_exists": True,
                    "real_device_capture_uploaded": True,
                },
            }
        ),
        encoding="utf-8",
    )
    args = _args(tmp_path)
    args.proof_file = [str(proof_file)]

    run_harness(args)

    run_root = tmp_path / "ops" / "city-launch-runs" / "durham-nc" / "test-run"
    proof = json.loads((run_root / "proof.launch-proof.json").read_text(encoding="utf-8"))
    assert proof["city"]["backend_supported"] is True
    assert proof["city"]["live_approved_job_count"] == 2
    assert proof["capture"]["real_device_capture_uploaded"] is True


def test_autonomous_city_launch_harness_rejects_contract_only_cross_repo_proof_file(tmp_path: Path) -> None:
    proof_file = tmp_path / "example.launch-proof.json"
    proof_file.write_text(
        json.dumps(
            {
                "schema_version": "city-launch-proof.v1",
                "contract_only": True,
                "city_slug": "durham-nc",
                "city": {"backend_supported": True},
                "capture": {"real_device_capture_uploaded": True},
            }
        ),
        encoding="utf-8",
    )
    args = _args(tmp_path)
    args.proof_file = [str(proof_file)]

    run_harness(args)

    run_root = tmp_path / "ops" / "city-launch-runs" / "durham-nc" / "test-run"
    proof = json.loads((run_root / "proof.launch-proof.json").read_text(encoding="utf-8"))
    lane_result = json.loads(
        (run_root / "lane-results" / "cross_repo_proof.example.launch-proof.json").read_text(
            encoding="utf-8"
        )
    )
    assert proof["city"]["backend_supported"] is False
    assert proof["capture"]["real_device_capture_uploaded"] is False
    assert lane_result["status"] == "blocked"
    assert "contract_only_proof_rejected" in lane_result["blockers"]


def test_autonomous_city_launch_harness_merges_release_config_validation_success(tmp_path: Path) -> None:
    capture_repo = tmp_path / "BlueprintCapture"
    pipeline_repo = tmp_path / "BlueprintCapturePipeline"
    webapp_repo = tmp_path / "Blueprint-WebApp"
    pipeline_repo.mkdir()
    webapp_repo.mkdir()
    _write_archive_script(
        capture_repo,
        "#!/usr/bin/env bash\nset -euo pipefail\necho 'Release config validated: local'\n",
    )
    args = _args(tmp_path)
    args.execute_local = True

    run_harness(args)

    run_root = tmp_path / "ops" / "city-launch-runs" / "durham-nc" / "test-run"
    proof = json.loads((run_root / "proof.launch-proof.json").read_text(encoding="utf-8"))
    assert proof["release"]["config_validated_by_archive_script"] is True


def test_autonomous_city_launch_harness_reports_release_config_validation_failure(tmp_path: Path) -> None:
    capture_repo = tmp_path / "BlueprintCapture"
    pipeline_repo = tmp_path / "BlueprintCapturePipeline"
    webapp_repo = tmp_path / "Blueprint-WebApp"
    pipeline_repo.mkdir()
    webapp_repo.mkdir()
    _write_archive_script(
        capture_repo,
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "echo 'BLUEPRINT_BACKEND_BASE_URL must be set in /tmp/release.xcconfig.' >&2\n"
        "exit 1\n",
    )
    args = _args(tmp_path)
    args.execute_local = True

    run_harness(args)

    run_root = tmp_path / "ops" / "city-launch-runs" / "durham-nc" / "test-run"
    blockers = (run_root / "blockers.jsonl").read_text(encoding="utf-8")
    assert "BLUEPRINT_BACKEND_BASE_URL must be set in /tmp/release.xcconfig." in blockers


def test_default_launch_proof_fills_missing_numeric_required_fields(monkeypatch) -> None:
    monkeypatch.setattr(
        harness,
        "REQUIRED_LAUNCH_PROOF_FIELDS",
        harness.REQUIRED_LAUNCH_PROOF_FIELDS
        + (("metrics.synthetic_count", "number>=1", "proof", "missing synthetic count", False),),
    )

    proof = harness.default_proof("durham-nc", 250000, ["iphone"])

    assert proof["metrics"]["synthetic_count"] == 0


def test_status_and_gap_helpers_cover_default_classification_paths(tmp_path: Path) -> None:
    proof = harness.default_proof("durham-nc", 250000, ["iphone"])
    assert harness.determine_status(proof, []) == "ready_to_market_iphone_city_beta"
    assert harness._work_packet_for_proof_field("unknown.field") == "proof"
    assert harness._dependency_class({"proof_field": "unknown.field", "lane_id": "marketing"}) == (
        "launch_policy_gap"
    )

    payout_provider = harness._expected_evidence_for_blocker(
        {"proof_field": "payouts.provider_name"}
    )
    payout_gate = harness._expected_evidence_for_blocker(
        {"proof_field": "payouts.live_payout_execution_human_gate"}
    )
    payout_generic = harness._expected_evidence_for_blocker(
        {"proof_field": "payouts.some_other_state"}
    )
    ops_gate = harness._expected_evidence_for_blocker(
        {"proof_field": "ops.payout_exception_monitor_repo_contract"}
    )
    assert "separates mocked contract readiness" in payout_provider
    assert "KYC/background posture" in payout_gate
    assert "Stripe/payment backend state" in payout_generic
    assert "finance review" in ops_gate

    lane_results = tmp_path / "lane-results"
    lane_results.mkdir()
    (lane_results / "city_backend.webapp-status-route.json").write_text(
        json.dumps({"status": "blocked"}),
        encoding="utf-8",
    )
    report = harness.build_launch_gap_report(
        city_slug="durham-nc",
        status="blocked_external_dependency",
        blockers=[
            {
                "id": "city.backend_supported",
                "proof_field": "city.backend_supported",
                "lane_id": "city_backend",
                "message": "city missing",
            }
        ],
        packets=[],
        capture_root="",
        run_root=tmp_path / "run",
        work_packet_root=tmp_path / "work-packets",
        lane_results_root=lane_results,
    )
    assert report["external_dependencies"][0]["lane_result_paths"] == [
        str(lane_results / "city_backend.webapp-status-route.json")
    ]


def test_city_launch_private_path_helpers_cover_fallbacks(tmp_path: Path, monkeypatch) -> None:
    invalid_json = tmp_path / "invalid.json"
    invalid_json.write_text("{bad", encoding="utf-8")
    assert harness._optional_json(invalid_json) == {}

    monkeypatch.setattr(
        harness,
        "resolve_local_capture_context",
        lambda _capture_root: (_ for _ in ()).throw(RuntimeError("invalid root")),
    )
    assert harness._capture_root_site_reference_candidates(
        capture_root=tmp_path / "capture",
        descriptor={"site_id": "site-1"},
    ) == []

    mov_root = tmp_path / "mov-root"
    (mov_root / "privacy").mkdir(parents=True)
    (mov_root / "privacy" / "final_walkthrough.mov").write_bytes(b"mov")
    assert harness._has_privacy_safe_walkthrough(
        capture_root=mov_root,
        descriptor={},
        privacy_manifest={},
    ) is True

    mp4_root = tmp_path / "mp4-root"
    (mp4_root / "privacy").mkdir(parents=True)
    (mp4_root / "privacy" / "final_walkthrough.mp4").write_bytes(b"mp4")
    assert harness._has_privacy_safe_walkthrough(
        capture_root=mp4_root,
        descriptor={},
        privacy_manifest={},
    ) is True

    assert harness._city_query_from_slug("durham") == ("Durham", None)
    assert harness._webapp_origin_from_configured_url("localhost:3000/internal") == (
        "localhost:3000/internal"
    )


def test_webapp_city_status_route_records_missing_url(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("PIPELINE_SYNC_WEBAPP_URL", raising=False)

    results = harness.collect_webapp_city_status_evidence(
        city_slug="durham-nc",
        lane_results_root=tmp_path / "lane-results",
    )

    assert results[0]["status"] == "blocked"
    assert results[0]["blockers"] == ["PIPELINE_SYNC_WEBAPP_URL_missing"]


def test_webapp_city_status_route_ignores_unparseable_http_error_body(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fake_urlopen(url: str, timeout: int):
        raise harness.urllib.error.HTTPError(
            url,
            502,
            "Bad Gateway",
            {},
            BytesIO(b"not-json"),
        )

    monkeypatch.setenv("PIPELINE_SYNC_WEBAPP_URL", "https://webapp.test")
    monkeypatch.setattr(harness.urllib.request, "urlopen", fake_urlopen)

    results = harness.collect_webapp_city_status_evidence(
        city_slug="durham-nc",
        lane_results_root=tmp_path / "lane-results",
    )

    assert results[0]["blockers"][0] == "webapp_status_route_http_502"
    assert not any("webapp_status_route_error:" in item for item in results[0]["blockers"])


def test_webapp_city_status_route_records_unreachable_exception(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fake_urlopen(_url: str, timeout: int):
        raise TimeoutError("timed out")

    monkeypatch.setenv("PIPELINE_SYNC_WEBAPP_URL", "https://webapp.test")
    monkeypatch.setattr(harness.urllib.request, "urlopen", fake_urlopen)

    results = harness.collect_webapp_city_status_evidence(
        city_slug="durham-nc",
        lane_results_root=tmp_path / "lane-results",
    )

    assert "webapp_status_route_unreachable:TimeoutError" in results[0]["blockers"]


def test_webapp_city_status_route_rejects_non_mapping_payload(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fake_urlopen(_url: str, timeout: int):
        return _FakeHttpResponse(["not", "a", "mapping"])

    monkeypatch.setenv("PIPELINE_SYNC_WEBAPP_URL", "https://webapp.test")
    monkeypatch.setattr(harness.urllib.request, "urlopen", fake_urlopen)

    results = harness.collect_webapp_city_status_evidence(
        city_slug="durham-nc",
        lane_results_root=tmp_path / "lane-results",
    )

    assert "webapp_status_route_invalid_json" in results[0]["blockers"]


def test_cross_repo_proof_evidence_reports_missing_mismatch_and_empty_files(
    tmp_path: Path,
) -> None:
    lane_results = tmp_path / "lane-results"
    mismatch = tmp_path / "mismatch.launch-proof.json"
    mismatch.write_text(
        json.dumps({"contract_only": False, "city_slug": "raleigh-nc", "city": {"backend_supported": True}}),
        encoding="utf-8",
    )
    empty = tmp_path / "empty.launch-proof.json"
    empty.write_text(
        json.dumps({"contract_only": False, "city_slug": "durham-nc", "unrelated": True}),
        encoding="utf-8",
    )

    results = harness.collect_cross_repo_proof_evidence(
        city_slug="durham-nc",
        proof_files=[tmp_path / "missing.launch-proof.json", mismatch, empty],
        lane_results_root=lane_results,
    )

    assert results[0]["blockers"] == [
        f"proof_file_missing:{tmp_path / 'missing.launch-proof.json'}"
    ]
    assert results[1]["blockers"] == [
        "city_slug_mismatch:raleigh-nc"
    ]
    assert results[2]["blockers"] == [
        "proof_file_contains_no_required_evidence"
    ]


def test_capture_root_evidence_uses_final_walkthrough_file_when_manifest_uri_missing(
    tmp_path: Path,
) -> None:
    capture_root = tmp_path / "capture"
    (capture_root / "privacy").mkdir(parents=True)
    final_walkthrough = capture_root / "privacy" / "final_walkthrough.mov"
    final_walkthrough.write_bytes(b"privacy-safe")

    results = harness.collect_capture_root_evidence(
        capture_root=capture_root,
        lane_results_root=tmp_path / "lane-results",
    )

    privacy_result = next(result for result in results if result["lane_id"] == "privacy_safe_provider")
    assert privacy_result["evidence"]["privacy_provider.final_walkthrough_uri"] == str(
        final_walkthrough
    )


def test_execute_local_with_cross_repo_proof_applies_generated_lane_results(
    tmp_path: Path,
    monkeypatch,
) -> None:
    proof_file = tmp_path / "capture.launch-proof.json"
    proof_file.write_text(
        json.dumps(
            {
                "contract_only": False,
                "city_slug": "durham-nc",
                "city": {"backend_supported": True},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        harness,
        "execute_local_packets",
        lambda *, packets, proof, run_root, args: [],
    )
    args = _args(tmp_path)
    args.execute_local = True
    args.proof_file = [str(proof_file)]

    run_harness(args)

    run_root = tmp_path / "ops" / "city-launch-runs" / "durham-nc" / "test-run"
    proof = json.loads((run_root / "proof.launch-proof.json").read_text(encoding="utf-8"))
    assert proof["city"]["backend_supported"] is True


def test_city_launch_main_prints_summary_and_maps_status_codes(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(
        harness,
        "run_harness",
        lambda _args: {"status": "ready_to_market_iphone_city_beta", "blocker_count": 0},
    )
    ready_code = harness.main(
        [
            "--city-slug",
            "durham-nc",
            "--budget-cents",
            "250000",
            "--capture-path",
            "iphone",
            "--output-root",
            str(tmp_path),
        ]
    )
    ready_stdout = capsys.readouterr().out

    assert ready_code == 0
    assert "ready_to_market_iphone_city_beta" in ready_stdout

    monkeypatch.setattr(
        harness,
        "run_harness",
        lambda _args: {"status": "blocked_repo_or_contract_failure", "blocker_count": 1},
    )
    blocked_code = harness.main(
        [
            "--city-slug",
            "durham-nc",
            "--budget-cents",
            "250000",
            "--capture-path",
            "iphone",
            "--output-root",
            str(tmp_path),
        ]
    )

    assert blocked_code == 2
