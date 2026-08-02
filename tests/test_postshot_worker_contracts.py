from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from blueprint_pipeline.postshot_worker_contracts import (
    PHASE_LIMITS_SECONDS,
    WatchDecision,
    assert_secret_free,
    build_attempt_ledger,
    build_deletion_receipt,
    build_external_watchdog_record,
    build_live_cost_estimate,
    build_postshot_train_args,
    build_provider_zero_proof,
    build_worker_pulse,
    evaluate_missing_pulse,
    evaluate_pulses,
    parse_nvidia_smi_csv,
    sanitize_path,
    sanitize_text,
    validate_pulse,
    TINY_CANARY_SPEC,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


def _pulse(
    *,
    sequence: int,
    observed_at: str,
    previous: dict | None = None,
    process_alive: bool = True,
    exit_code: int | None = None,
    log_bytes: int = 100,
    gpu_util: float | None = 0,
    output_bytes: int = 0,
    phase: str = "P1",
    grace_until: str | None = "2026-01-01T00:00:00Z",
    secrets: tuple[str, ...] = (),
) -> dict:
    return build_worker_pulse(
        run_id="postshot-20260801T215521Z",
        attempt=5,
        arm="P1",
        phase=phase,
        sequence=sequence,
        observed_at=observed_at,
        instance={"id": "i-target", "type": "g6.xlarge", "state": "running"},
        postshot_process={"pid": 42, "start_time_utc": observed_at, "alive": process_alive, "exit_code": exit_code},
        postshot_log={"tail": "progress", "byte_count": log_bytes, "digest": "sha256:" + "0" * 64},
        gpu={"name": "NVIDIA L4", "driver_version": "566.03", "utilization_percent": gpu_util, "memory_used_mib": 0, "memory_total_mib": 23034, "temperature_c": 44, "power_w": 70},
        outputs=[{"path": "P1.psht", "bytes": output_bytes, "digest": "sha256:" + "1" * 64}],
        disk_free_bytes=1000000,
        last_credible_progress_at=observed_at,
        live_cost_estimate_usd=1.0,
        incremental_cap_usd=15.0,
        ttl_deadline="2026-01-01T05:00:00Z",
        next_automatic_kill_deadline="2026-01-01T05:00:00Z",
        result_upload_state="not_started",
        credential_object_deletion_state="deleted",
        startup_grace_until=grace_until,
        previous_pulse=previous,
        secrets=secrets,
    )


def test_postshot_global_flags_and_profile_spaces_are_preserved() -> None:
    args = build_postshot_train_args(
        login_email="operator@example.invalid",
        login_password="correct horse battery staple",
        dataset=r"C:\work\dataset",
        profile="Splat MCMC",
        output_project=r"C:\work\out\P2.psht",
        output_splat=r"C:\work\out\P2.ply",
    )
    assert args[:5] == ["--login", "operator@example.invalid", "--password", "correct horse battery staple", "train"]
    assert args[4] == "train"
    assert args[args.index("--profile") + 1] == "Splat MCMC"
    assert "--no-recenter-points" in args


def test_tiny_canary_spec_is_bounded_and_uses_supported_flags() -> None:
    assert TINY_CANARY_SPEC["train_steps_limit_ksteps"] == 1
    assert TINY_CANARY_SPEC["max_image_size"] == 256
    assert TINY_CANARY_SPEC["max_num_splats_ksplats"] == 100
    args = build_postshot_train_args(
        login_email="operator@example.invalid",
        login_password="secret-value",
        dataset=r"C:\work\dataset",
        profile="Splat3",
        output_project=r"C:\work\out\C0_canary_splat3.psht",
        output_splat=r"C:\work\out\C0_canary_splat3.ply",
        max_image_size=TINY_CANARY_SPEC["max_image_size"],
        train_steps_limit_ksteps=TINY_CANARY_SPEC["train_steps_limit_ksteps"],
        max_num_splats_ksplats=TINY_CANARY_SPEC["max_num_splats_ksplats"],
    )
    assert "--max-steps" not in args
    assert args[args.index("--train-steps-limit") + 1] == "1"
    assert args[args.index("--max-num-splats") + 1] == "100"


def test_redaction_removes_secrets_urls_and_hostile_filename_text() -> None:
    secret = "p@ssword-value"
    raw = f"password={secret} https://bucket.example.invalid/a?X-Amz-Signature=secret"
    redacted = sanitize_text(raw, (secret,))
    assert secret not in redacted
    assert "https://" not in redacted
    assert "[REDACTED_URL]" in redacted
    safe_name = sanitize_path("../../$(touch PWNED) scene?.ply")
    assert ".." not in safe_name
    assert "$" not in safe_name
    assert safe_name.endswith("scene_.ply")


def test_gpu_csv_parser_is_tolerant_and_numeric() -> None:
    parsed = parse_nvidia_smi_csv("NVIDIA L4, 566.03, 37 %, 123 MiB, 23034 MiB, 44, 70.5 W")
    assert parsed["name"] == "NVIDIA L4"
    assert parsed["utilization_percent"] == 37.0
    assert parsed["memory_used_mib"] == 123.0
    assert parsed["parse_error"] is None
    malformed = parse_nvidia_smi_csv("bad")
    assert malformed["parse_error"] == "field_count_too_small"


def test_pulse_digest_schema_sequence_and_secret_free_tail() -> None:
    secret = "pw-123"
    first = _pulse(sequence=1, observed_at="2026-01-01T01:00:00Z", log_bytes=10, output_bytes=2, secrets=(secret,))
    assert first["schema_version"] == "worker_pulse.v2"
    assert first["sequence"] == 1
    assert validate_pulse(first) == []
    assert secret not in json.dumps(first)
    second = _pulse(sequence=1, observed_at="2026-01-01T01:02:00Z", previous=first)
    assert "sequence_not_monotonic" in validate_pulse(second, first)
    tampered = dict(first)
    tampered["phase"] = "P2"
    assert "pulse_digest" in validate_pulse(tampered)
    assert_secret_free(first, (secret,))


def test_gpu_active_training_counts_as_progress_when_log_is_quiet() -> None:
    pulse = _pulse(sequence=1, observed_at="2026-01-01T01:00:00Z", log_bytes=0, gpu_util=42, output_bytes=0)
    assert pulse["progress"]["log_progress"] is False
    assert pulse["progress"]["output_progress"] is False
    assert pulse["progress"]["gpu_active"] is True
    assert pulse["progress"]["credible_training_progress"] is True


def test_startup_grace_does_not_count_as_no_progress() -> None:
    pulse = _pulse(sequence=1, observed_at="2026-01-01T01:00:00Z", log_bytes=0, gpu_util=0, output_bytes=0, grace_until="2026-01-01T02:00:00Z")
    assert pulse["progress"]["startup_grace_active"] is True
    decision = evaluate_pulses([pulse], now_epoch=1767229200, phase_started_epoch=1767229200, launched_epoch=1767229200, live_cost_estimate_usd=1, incremental_cap_usd=15)
    assert decision == WatchDecision("continue", "startup_or_grace", False)


def test_dead_process_aborts_immediately() -> None:
    pulse = _pulse(sequence=1, observed_at="2026-01-01T01:00:00Z", process_alive=False, exit_code=109)
    decision = evaluate_pulses([pulse], now_epoch=1767229200, phase_started_epoch=1767229200, launched_epoch=1767229200, live_cost_estimate_usd=1, incremental_cap_usd=15)
    assert decision.action == "abort"
    assert decision.reason == "postshot_process_dead"


def test_three_consecutive_quiet_pulses_abort_after_grace() -> None:
    pulses: list[dict] = []
    previous = None
    for sequence in range(1, 4):
        current = _pulse(sequence=sequence, observed_at=f"2026-01-01T01:0{sequence}:00Z", previous=previous, log_bytes=0, gpu_util=0, output_bytes=0)
        pulses.append(current)
        previous = current
    decision = evaluate_pulses(pulses, now_epoch=1767229470, phase_started_epoch=1767229200, launched_epoch=1767229200, live_cost_estimate_usd=1, incremental_cap_usd=15)
    assert decision.action == "abort"
    assert decision.reason == "no_credible_progress_3_pulses"


def test_stale_pulse_phase_timeout_and_spend_cap_are_distinct() -> None:
    stale = _pulse(sequence=1, observed_at="2026-01-01T00:00:00Z")
    assert evaluate_pulses([stale], now_epoch=1767229200, phase_started_epoch=1767229200, launched_epoch=1767229200, live_cost_estimate_usd=1, incremental_cap_usd=15).reason == "pulse_stale_gt_300s"
    phase = _pulse(sequence=1, observed_at="2026-01-01T03:30:00Z")
    phase["phase_started_at_utc"] = "2026-01-01T01:00:00Z"
    phase["pulse_digest"] = canonical_digest(phase, digest_field="pulse_digest")
    assert evaluate_pulses([phase], now_epoch=1767238201, phase_started_epoch=1767229200, launched_epoch=1767229200, live_cost_estimate_usd=1, incremental_cap_usd=15).reason == "phase_timeout:P1"
    assert evaluate_pulses([phase], now_epoch=1767229200, phase_started_epoch=1767229200, launched_epoch=1767229200, live_cost_estimate_usd=15, incremental_cap_usd=15).reason == "incremental_spend_cap_reached"


def test_missing_pulse_terminates_after_five_minutes() -> None:
    assert evaluate_missing_pulse(last_pulse_epoch=None, now_epoch=1000).action == "terminate"
    assert evaluate_missing_pulse(last_pulse_epoch=800, now_epoch=1000).action == "continue"
    assert evaluate_missing_pulse(last_pulse_epoch=699, now_epoch=1000).reason == "pulse_stale_gt_300s"


def test_watcher_restart_keeps_exact_run_identity() -> None:
    record = build_external_watchdog_record(run_id="postshot-20260801T215521Z", instance_id="i-target", pid=100, started_at_utc="2026-01-01T00:00:00Z", ttl_deadline_utc="2026-01-01T05:00:00Z", log_path="/tmp/watchdog.log", command_digest="sha256:" + "0" * 64)
    restarted = dict(record)
    restarted.update({"pid": 101, "reattached_from_pid": record["pid"], "started_at_utc": "2026-01-01T00:02:00Z"})
    assert restarted["run_id"] == record["run_id"]
    assert restarted["instance_id"] == record["instance_id"]
    assert restarted["reattached_from_pid"] == 100


def test_results_before_termination_still_requires_provider_zero() -> None:
    proof = build_provider_zero_proof(run_id="postshot-20260801T215521Z", region="us-east-1", instances=[{"id": "i-target", "state": "running", "results_uploaded": True}], volumes=[{"id": "vol-target", "state": "in-use"}], snapshots=[], images=[], elastic_ips=[])
    assert proof["provider_zero"] is False
    assert "run_owned_instance_not_terminated" in proof["blockers"]
    assert "run_owned_volume_present" in proof["blockers"]


def test_provider_zero_is_scoped_and_keeps_security_group_separate() -> None:
    blocked = build_provider_zero_proof(run_id="postshot-20260801T215521Z", region="us-east-1", instances=[{"id": "i-target", "state": "stopped"}], volumes=[], snapshots=[], images=[], elastic_ips=[], security_groups=[{"id": "sg-keep", "name": "blueprint-postshot-bakeoff"}], checked_at_utc="2026-01-01T00:00:00Z")
    assert blocked["provider_zero"] is False
    assert blocked["blockers"] == ["run_owned_instance_not_terminated"]
    passed = build_provider_zero_proof(run_id="postshot-20260801T215521Z", region="us-east-1", instances=[{"id": "i-target", "state": "terminated"}], volumes=[], snapshots=[], images=[], elastic_ips=[], security_groups=[{"id": "sg-keep"}], checked_at_utc="2026-01-01T00:00:00Z")
    assert passed["provider_zero"] is True
    assert passed["security_group_is_not_provider_zero_evidence"] is True


def test_deletion_receipt_requires_absence_and_costs_are_separate() -> None:
    receipt = build_deletion_receipt(run_id="postshot-20260801T215521Z", checked_at_utc="2026-01-01T00:00:00Z", objects=[{"key": "blueprint-postshot-bakeoff/postshot-x/dataset.zip", "object_kind": "dataset", "delete_requested": True, "absent_verified": True}, {"key": "blueprint-postshot-bakeoff/postshot-x/license.env", "object_kind": "license", "delete_requested": True, "absent_verified": True}])
    assert receipt["all_absent_verified"] is True
    estimate = build_live_cost_estimate(as_of_utc="2026-01-01T00:00:00Z", instance_usd=1, ebs_usd=0.1, transfer_usd=0.2, object_storage_usd=0.3, license_increment_usd=0.4)
    assert estimate["reconciled"] is False
    assert estimate["total_usd"] == pytest.approx(2.0)


def test_phase_limits_and_historical_ledger_are_explicit() -> None:
    assert PHASE_LIMITS_SECONDS["windows_boot"] == 600
    assert PHASE_LIMITS_SECONDS["tiny_training_canary"] == 1200
    assert PHASE_LIMITS_SECONDS["whole_instance"] == 18000
    ledger = build_attempt_ledger(attempts=[{"attempt": 1, "classification": "cli_invocation_failure"}], historical_bakeoff_budget_usd=250, historical_postshot_spend_estimate_usd=3.8, generated_at_utc="2026-01-01T00:00:00Z")
    assert ledger["append_only"] is True
    assert ledger["historical_spend_reconciliation_status"] == "required_not_observed"
    assert ledger["ledger_digest"].startswith("sha256:")


@pytest.mark.parametrize("scenario", ["healthy", "gpu-active-quiet", "silent-hung", "dead", "nonzero", "secret-echo", "p1-success-p2-failed", "watcher-restart", "heartbeat-loss", "results-uploaded-instance-fails"])
def test_fake_postshot_executable_scenarios(tmp_path: Path, scenario: str) -> None:
    executable = Path(__file__).parent / "fixtures" / "fake_postshot.py"
    result = subprocess.run([sys.executable, str(executable), "--scenario", scenario, "--output-dir", str(tmp_path), "--profile", "Splat MCMC", "--secret-email", "operator@example.invalid", "--secret-password", "secret-value", "--hang-seconds", "0.01"], check=False, capture_output=True, text=True)
    if scenario == "silent-hung":
        assert result.returncode == 0
    elif scenario == "dead":
        assert result.returncode == 73
    elif scenario == "nonzero":
        assert result.returncode == 109
    elif scenario == "p1-success-p2-failed":
        assert result.returncode == 7
    else:
        assert result.returncode == 0
    assert (tmp_path / "train-log.txt").is_file()
    if scenario == "secret-echo":
        raw = (tmp_path / "train-log.txt").read_text(encoding="utf-8")
        assert "secret-value" in raw
        sanitized = sanitize_text(raw, ("operator@example.invalid", "secret-value"))
        assert "secret-value" not in sanitized
        assert "operator@example.invalid" not in sanitized
