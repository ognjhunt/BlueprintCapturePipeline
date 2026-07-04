"""Tests for the provider reliability manifest — the ops-facing paid-run ledger.

Covers the service-grade failure scenarios the paid lane must distinguish:
capacity unavailable (fail before spend), startup-marker timeout, post-marker
no-progress stall, runner failure, successful teardown proof, optional media
skipped, and stale artifact rejection.
"""

from __future__ import annotations

from blueprint_pipeline.provider_reliability_manifest import (
    BLOCKER_CAPACITY_UNAVAILABLE,
    BLOCKER_IMAGE_CONTRACT_INVALID,
    BLOCKER_POST_MARKER_NO_PROGRESS,
    BLOCKER_RUNTIME_CONTRACT_INVALID,
    BLOCKER_SPEND_GATE_CLOSED,
    BLOCKER_STALE_ARTIFACT,
    BLOCKER_STARTUP_MARKER_TIMEOUT,
    BLOCKER_TEARDOWN_UNPROVEN,
    RUN_PHASES,
    build_artifact_collection,
    build_pre_spend_preflight,
    build_provider_reliability_manifest,
    build_teardown_proof,
    evaluate_post_marker_stall,
)
from blueprint_pipeline.success_claim_contracts import (
    build_artifact_freshness_evidence,
)


def _valid_runtime_contract() -> dict:
    return {
        "startup_marker": "WORKER_STARTED",
        "progress_marker": "RENDER_PROGRESS",
        "startup_timeout_seconds": 600,
        "no_progress_timeout_seconds": 900,
    }


def _valid_image_contract() -> dict:
    return {
        "image_ref": "nijelhunt/blueprint-capture-pipeline:sam3-ready-v7",
        "pinned": True,
        "digest": "sha256:abc123",
    }


def _passing_preflight() -> dict:
    return build_pre_spend_preflight(
        provider="runpod",
        credential_present=True,
        capacity_evidence={"available": True, "offer_count": 3},
        image_contract=_valid_image_contract(),
        runtime_contract=_valid_runtime_contract(),
        spend_gate_open=True,
    )


def _pass(phase: str = "x") -> dict:
    return {"status": "PASS", "blockers": [], "phase": phase}


def _fail(*blockers: str) -> dict:
    return {"status": "FAIL", "blockers": list(blockers)}


def _proven_teardown() -> dict:
    return build_teardown_proof(
        provider="runpod",
        allocation_id="pod-123",
        terminate_requested=True,
        provider_terminal_status="TERMINATED",
        verified_at="2026-07-04T12:00:00Z",
    )


# ---------------------------------------------------------------------------
# Pre-spend preflight: capacity unavailable and contract failures block spend.
# ---------------------------------------------------------------------------


class TestPreSpendPreflight:
    def test_valid_contracts_allow_spend(self) -> None:
        result = _passing_preflight()
        assert result["status"] == "PASS"
        assert result["spend_allowed"] is True
        assert result["blockers"] == []

    def test_capacity_unavailable_fails_before_spend(self) -> None:
        result = build_pre_spend_preflight(
            provider="runpod",
            credential_present=True,
            capacity_evidence={"available": False, "detail": "no_4090_offers"},
            image_contract=_valid_image_contract(),
            runtime_contract=_valid_runtime_contract(),
            spend_gate_open=True,
        )
        assert result["status"] == "FAIL"
        assert result["spend_allowed"] is False
        assert f"{BLOCKER_CAPACITY_UNAVAILABLE}:no_4090_offers" in result["blockers"]

    def test_missing_capacity_evidence_fails_closed(self) -> None:
        result = build_pre_spend_preflight(
            provider="vast",
            credential_present=True,
            capacity_evidence=None,
            image_contract=_valid_image_contract(),
            runtime_contract=_valid_runtime_contract(),
            spend_gate_open=True,
        )
        assert result["spend_allowed"] is False
        assert any(
            b.startswith(BLOCKER_CAPACITY_UNAVAILABLE) for b in result["blockers"]
        )

    def test_non_boolean_capacity_is_not_capacity(self) -> None:
        result = build_pre_spend_preflight(
            provider="vast",
            credential_present=True,
            capacity_evidence={"available": "yes"},
            image_contract=_valid_image_contract(),
            runtime_contract=_valid_runtime_contract(),
            spend_gate_open=True,
        )
        assert result["spend_allowed"] is False

    def test_unpinned_image_blocks_spend(self) -> None:
        image = _valid_image_contract()
        image["pinned"] = False
        result = build_pre_spend_preflight(
            provider="runpod",
            credential_present=True,
            capacity_evidence={"available": True},
            image_contract=image,
            runtime_contract=_valid_runtime_contract(),
            spend_gate_open=True,
        )
        assert result["spend_allowed"] is False
        assert any(
            b.startswith(BLOCKER_IMAGE_CONTRACT_INVALID) for b in result["blockers"]
        )

    def test_runtime_contract_without_markers_blocks_spend(self) -> None:
        result = build_pre_spend_preflight(
            provider="runpod",
            credential_present=True,
            capacity_evidence={"available": True},
            image_contract=_valid_image_contract(),
            runtime_contract={"startup_timeout_seconds": 600},
            spend_gate_open=True,
        )
        assert result["spend_allowed"] is False
        blockers = result["blockers"]
        assert f"{BLOCKER_RUNTIME_CONTRACT_INVALID}:startup_marker_undefined" in blockers
        assert f"{BLOCKER_RUNTIME_CONTRACT_INVALID}:progress_marker_undefined" in blockers
        assert (
            f"{BLOCKER_RUNTIME_CONTRACT_INVALID}:no_progress_timeout_seconds_not_positive"
            in blockers
        )

    def test_spend_gate_closed_blocks_even_with_valid_contracts(self) -> None:
        result = build_pre_spend_preflight(
            provider="runpod",
            credential_present=True,
            capacity_evidence={"available": True},
            image_contract=_valid_image_contract(),
            runtime_contract=_valid_runtime_contract(),
            spend_gate_open=None,
        )
        assert result["spend_allowed"] is False
        assert any(b.startswith(BLOCKER_SPEND_GATE_CLOSED) for b in result["blockers"])

    def test_missing_credential_blocks_spend(self) -> None:
        result = build_pre_spend_preflight(
            provider="lambda",
            credential_present=False,
            capacity_evidence={"available": True},
            image_contract=_valid_image_contract(),
            runtime_contract=_valid_runtime_contract(),
            spend_gate_open=True,
        )
        assert result["spend_allowed"] is False


# ---------------------------------------------------------------------------
# Stall policy: startup-marker timeout and post-marker no-progress.
# ---------------------------------------------------------------------------


class TestStallPolicy:
    def test_startup_marker_timeout_terminates(self) -> None:
        result = evaluate_post_marker_stall(
            startup_marker_seen=False,
            startup_elapsed_seconds=601,
            startup_timeout_seconds=600,
            last_progress_age_seconds=None,
            no_progress_timeout_seconds=900,
        )
        assert result["should_terminate"] is True
        assert result["stall_mode"] == "container_startup"
        assert any(
            b.startswith(BLOCKER_STARTUP_MARKER_TIMEOUT) for b in result["blockers"]
        )

    def test_startup_within_timeout_waits(self) -> None:
        result = evaluate_post_marker_stall(
            startup_marker_seen=False,
            startup_elapsed_seconds=120,
            startup_timeout_seconds=600,
            last_progress_age_seconds=None,
            no_progress_timeout_seconds=900,
        )
        assert result["should_terminate"] is False
        assert result["stall_mode"] is None

    def test_post_marker_no_progress_terminates(self) -> None:
        result = evaluate_post_marker_stall(
            startup_marker_seen=True,
            startup_elapsed_seconds=2000,
            startup_timeout_seconds=600,
            last_progress_age_seconds=901,
            no_progress_timeout_seconds=900,
        )
        assert result["should_terminate"] is True
        assert result["stall_mode"] == "runtime_execution"
        assert any(
            b.startswith(BLOCKER_POST_MARKER_NO_PROGRESS) for b in result["blockers"]
        )

    def test_no_progress_marker_ever_uses_startup_clock(self) -> None:
        # Startup marker seen long ago, but the worker never wrote a progress
        # marker: silence must count as a stall, not indefinite patience.
        result = evaluate_post_marker_stall(
            startup_marker_seen=True,
            startup_elapsed_seconds=950,
            startup_timeout_seconds=600,
            last_progress_age_seconds=None,
            no_progress_timeout_seconds=900,
        )
        assert result["should_terminate"] is True
        assert result["stall_mode"] == "runtime_execution"
        assert any("no_progress_marker_ever_written" in b for b in result["blockers"])

    def test_recent_progress_keeps_running(self) -> None:
        result = evaluate_post_marker_stall(
            startup_marker_seen=True,
            startup_elapsed_seconds=5000,
            startup_timeout_seconds=600,
            last_progress_age_seconds=30,
            no_progress_timeout_seconds=900,
        )
        assert result["should_terminate"] is False


# ---------------------------------------------------------------------------
# Teardown proof.
# ---------------------------------------------------------------------------


class TestTeardownProof:
    def test_successful_teardown_is_proven(self) -> None:
        result = _proven_teardown()
        assert result["status"] == "PASS"
        assert result["billing_stopped"] is True

    def test_terminate_request_alone_is_not_proof(self) -> None:
        result = build_teardown_proof(
            provider="runpod",
            allocation_id="pod-123",
            terminate_requested=True,
            provider_terminal_status=None,
        )
        assert result["status"] == "FAIL"
        assert result["billing_stopped"] is False
        assert (
            f"{BLOCKER_TEARDOWN_UNPROVEN}:terminal_status_not_verified"
            in result["blockers"]
        )

    def test_stopped_is_not_terminal_for_billing(self) -> None:
        # RunPod STOPPED/EXITED pods still bill disk; only terminal states count.
        result = build_teardown_proof(
            provider="runpod",
            allocation_id="pod-123",
            terminate_requested=True,
            provider_terminal_status="STOPPED",
            verified_at="2026-07-04T12:00:00Z",
        )
        assert result["status"] == "FAIL"
        assert any("non_terminal_status:stopped" in b for b in result["blockers"])

    def test_keep_alive_records_open_billing(self) -> None:
        result = build_teardown_proof(
            provider="runpod",
            allocation_id="pod-123",
            terminate_requested=False,
            provider_terminal_status=None,
            keep_alive_requested=True,
            keep_alive_reason="persistent warm WAM lane",
        )
        assert result["status"] == "FAIL"
        assert result["keep_alive_requested"] is True
        assert any(
            "allocation_intentionally_kept_alive" in b for b in result["blockers"]
        )


# ---------------------------------------------------------------------------
# Artifact collection: optional media skipped, stale artifact rejection.
# ---------------------------------------------------------------------------


class TestArtifactCollection:
    def _fresh(self) -> dict:
        return build_artifact_freshness_evidence(
            artifact_run_id="run-1", current_run_id="run-1"
        )

    def _stale(self) -> dict:
        return build_artifact_freshness_evidence(
            artifact_run_id="run-0", current_run_id="run-1"
        )

    def test_required_fresh_artifacts_pass(self) -> None:
        result = build_artifact_collection(
            required_artifacts=[
                {"name": "frames.zip", "present": True, "freshness": self._fresh()},
            ],
        )
        assert result["status"] == "PASS"
        assert result["collected"][0]["name"] == "frames.zip"

    def test_stale_artifact_rejected(self) -> None:
        result = build_artifact_collection(
            required_artifacts=[
                {"name": "frames.zip", "present": True, "freshness": self._stale()},
            ],
        )
        assert result["status"] == "FAIL"
        assert any(
            b.startswith(f"{BLOCKER_STALE_ARTIFACT}:frames.zip")
            for b in result["blockers"]
        )
        assert result["collected"] == []

    def test_present_without_freshness_evidence_rejected(self) -> None:
        result = build_artifact_collection(
            required_artifacts=[{"name": "frames.zip", "present": True}],
        )
        assert result["status"] == "FAIL"
        assert any(
            "freshness_evidence_missing" in b for b in result["blockers"]
        )

    def test_optional_media_skipped_with_reason_passes(self) -> None:
        result = build_artifact_collection(
            required_artifacts=[
                {"name": "frames.zip", "present": True, "freshness": self._fresh()},
            ],
            optional_artifacts=[
                {
                    "name": "review_video.mp4",
                    "skipped": True,
                    "skip_reason": "media rendering disabled for this run",
                },
            ],
        )
        assert result["status"] == "PASS"
        assert result["skipped"] == [
            {
                "name": "review_video.mp4",
                "skip_reason": "media rendering disabled for this run",
                "required": False,
            }
        ]

    def test_optional_skip_without_reason_fails(self) -> None:
        result = build_artifact_collection(
            required_artifacts=[],
            optional_artifacts=[{"name": "review_video.mp4", "skipped": True}],
        )
        assert result["status"] == "FAIL"
        assert any(
            "optional_artifact_skipped_without_reason" in b for b in result["blockers"]
        )

    def test_required_artifact_cannot_be_skipped(self) -> None:
        result = build_artifact_collection(
            required_artifacts=[
                {"name": "frames.zip", "skipped": True, "skip_reason": "oops"},
            ],
        )
        assert result["status"] == "FAIL"
        assert "required_artifact_skipped:frames.zip" in result["blockers"]

    def test_missing_required_artifact_fails(self) -> None:
        result = build_artifact_collection(
            required_artifacts=[{"name": "frames.zip", "present": False}],
        )
        assert result["status"] == "FAIL"
        assert "required_artifact_missing:frames.zip" in result["blockers"]


# ---------------------------------------------------------------------------
# Composed manifest: phase attribution and teardown mandate.
# ---------------------------------------------------------------------------


class TestReliabilityManifest:
    def test_capacity_failure_never_reaches_launch(self) -> None:
        preflight = build_pre_spend_preflight(
            provider="runpod",
            credential_present=True,
            capacity_evidence={"available": False, "detail": "no_offers"},
            image_contract=_valid_image_contract(),
            runtime_contract=_valid_runtime_contract(),
            spend_gate_open=True,
        )
        manifest = build_provider_reliability_manifest(
            run_id="run-1",
            provider="runpod",
            pre_spend_preflight=preflight,
            # No launch, no teardown needed proof-wise, but manifest still
            # reports the teardown contract missing.
        )
        assert manifest["failed_phase"] == "pre_spend_preflight"
        assert manifest["furthest_phase_reached"] == "pre_spend_preflight"
        assert manifest["run_completed"] is False
        assert any(
            b.startswith(BLOCKER_CAPACITY_UNAVAILABLE)
            for b in manifest["failure_blockers"]
        )

    def test_marker_timeout_attributed_to_container_startup(self) -> None:
        stall = evaluate_post_marker_stall(
            startup_marker_seen=False,
            startup_elapsed_seconds=700,
            startup_timeout_seconds=600,
            last_progress_age_seconds=None,
            no_progress_timeout_seconds=900,
        )
        manifest = build_provider_reliability_manifest(
            run_id="run-2",
            provider="vast",
            pre_spend_preflight=_passing_preflight(),
            provider_launch=_pass("provider_launch"),
            container_startup=stall,
            teardown=_proven_teardown(),
        )
        assert manifest["failed_phase"] == "container_startup"
        assert any(
            b.startswith(BLOCKER_STARTUP_MARKER_TIMEOUT)
            for b in manifest["failure_blockers"]
        )
        # Launch succeeded — the manifest distinguishes launch from startup.
        assert manifest["phases"]["provider_launch"]["passed"] is True

    def test_post_marker_stall_attributed_to_runtime_execution(self) -> None:
        stall = evaluate_post_marker_stall(
            startup_marker_seen=True,
            startup_elapsed_seconds=3000,
            startup_timeout_seconds=600,
            last_progress_age_seconds=1200,
            no_progress_timeout_seconds=900,
        )
        manifest = build_provider_reliability_manifest(
            run_id="run-3",
            provider="runpod",
            pre_spend_preflight=_passing_preflight(),
            provider_launch=_pass("provider_launch"),
            container_startup=_pass("container_startup"),
            runtime_execution=stall,
            teardown=_proven_teardown(),
        )
        assert manifest["failed_phase"] == "runtime_execution"
        assert any(
            b.startswith(BLOCKER_POST_MARKER_NO_PROGRESS)
            for b in manifest["failure_blockers"]
        )

    def test_runner_failure_recorded_with_exact_phase(self) -> None:
        manifest = build_provider_reliability_manifest(
            run_id="run-4",
            provider="runpod",
            pre_spend_preflight=_passing_preflight(),
            provider_launch=_pass("provider_launch"),
            container_startup=_pass("container_startup"),
            runtime_execution=_fail("runner_failed:isaac_exit_code_1"),
            teardown=_proven_teardown(),
        )
        assert manifest["failed_phase"] == "runtime_execution"
        assert manifest["failure_blockers"] == ["runner_failed:isaac_exit_code_1"]
        assert manifest["teardown_proven"] is True
        assert manifest["open_billing_risk"] is False

    def test_missing_teardown_is_open_billing_risk(self) -> None:
        manifest = build_provider_reliability_manifest(
            run_id="run-5",
            provider="runpod",
            pre_spend_preflight=_passing_preflight(),
            provider_launch=_pass("provider_launch"),
            container_startup=_pass("container_startup"),
            runtime_execution=_pass("runtime_execution"),
            artifact_collection=_pass("artifact_collection"),
            artifact_quality=_pass("artifact_quality"),
            task_evaluation=_pass("task_evaluation"),
        )
        assert manifest["open_billing_risk"] is True
        assert manifest["run_completed"] is False
        assert any(
            b.startswith(BLOCKER_TEARDOWN_UNPROVEN) for b in manifest["blockers"]
        )

    def test_fully_successful_run_completes(self) -> None:
        manifest = build_provider_reliability_manifest(
            run_id="run-6",
            provider="runpod",
            session_dir="output/session-6/pipeline",
            pre_spend_preflight=_passing_preflight(),
            provider_launch=_pass("provider_launch"),
            container_startup=_pass("container_startup"),
            runtime_execution=_pass("runtime_execution"),
            artifact_collection=_pass("artifact_collection"),
            artifact_quality=_pass("artifact_quality"),
            task_evaluation=_pass("task_evaluation"),
            teardown=_proven_teardown(),
        )
        assert manifest["run_completed"] is True
        assert manifest["failed_phase"] is None
        assert manifest["open_billing_risk"] is False
        assert manifest["blockers"] == []

    def test_phase_order_is_stable_contract(self) -> None:
        assert RUN_PHASES == (
            "pre_spend_preflight",
            "provider_launch",
            "container_startup",
            "runtime_execution",
            "artifact_collection",
            "artifact_quality",
            "task_evaluation",
            "teardown",
        )

    def test_first_failure_wins_over_later_failures(self) -> None:
        manifest = build_provider_reliability_manifest(
            run_id="run-7",
            provider="vast",
            pre_spend_preflight=_passing_preflight(),
            provider_launch=_fail("capacity_unavailable:race_lost"),
            runtime_execution=_fail("runner_failed:should_not_be_primary"),
            teardown=_proven_teardown(),
        )
        assert manifest["failed_phase"] == "provider_launch"
        assert manifest["failure_blockers"] == ["capacity_unavailable:race_lost"]

    def test_missing_run_id_is_blocked(self) -> None:
        manifest = build_provider_reliability_manifest(
            run_id="",
            provider="runpod",
            teardown=_proven_teardown(),
        )
        assert manifest["run_completed"] is False
        assert "run_id_missing" in manifest["blockers"]


class TestNotApplicablePhases:
    def test_render_lane_completes_without_task_evaluation(self) -> None:
        manifest = build_provider_reliability_manifest(
            run_id="run-8",
            provider="runpod",
            pre_spend_preflight=_passing_preflight(),
            provider_launch=_pass("provider_launch"),
            container_startup=_pass("container_startup"),
            runtime_execution=_pass("runtime_execution"),
            artifact_collection=_pass("artifact_collection"),
            teardown=_proven_teardown(),
            not_applicable_phases=("artifact_quality", "task_evaluation"),
        )
        assert manifest["run_completed"] is True
        assert manifest["failed_phase"] is None
        assert manifest["not_applicable_phases"] == [
            "artifact_quality",
            "task_evaluation",
        ]
        # Not-applicable phases never read as passed — no fake task-success PASS.
        assert manifest["phases"]["task_evaluation"]["passed"] is False
        assert manifest["phases"]["task_evaluation"]["not_applicable"] is True

    def test_teardown_can_never_be_declared_not_applicable(self) -> None:
        manifest = build_provider_reliability_manifest(
            run_id="run-9",
            provider="runpod",
            pre_spend_preflight=_passing_preflight(),
            provider_launch=_pass("provider_launch"),
            container_startup=_pass("container_startup"),
            runtime_execution=_pass("runtime_execution"),
            artifact_collection=_pass("artifact_collection"),
            not_applicable_phases=(
                "teardown",
                "pre_spend_preflight",
                "artifact_quality",
                "task_evaluation",
            ),
        )
        assert manifest["run_completed"] is False
        assert manifest["open_billing_risk"] is True
        assert "teardown" not in manifest["not_applicable_phases"]
        assert "pre_spend_preflight" not in manifest["not_applicable_phases"]
