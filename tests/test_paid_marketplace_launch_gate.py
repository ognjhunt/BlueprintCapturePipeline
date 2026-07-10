from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_gate_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "run_paid_marketplace_launch_gate.py"
    spec = importlib.util.spec_from_file_location("run_paid_marketplace_launch_gate", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _result(
    gate,
    result_id: str,
    *,
    status: str = "passed",
    blocking: bool = True,
    tags: tuple[str, ...] = ("iphone", "glasses", "android"),
    evidence_class: str | None = None,
    skip_reason: str | None = None,
    evidence_note: str | None = None,
):
    return gate.CommandResult(
        id=result_id,
        label=result_id.replace("_", " ").title(),
        repo="BlueprintCapturePipeline",
        command=["true"],
        cwd=str(Path.cwd()),
        status=status,
        blocking=blocking,
        source_tags=tags,
        evidence_class=evidence_class,
        skip_reason=skip_reason,
        evidence_note=evidence_note,
    )


def _base_results(gate):
    return [
        _result(gate, "webapp_request_sync_contracts"),
        _result(gate, "webapp_creator_payout_contracts"),
        _result(gate, "webapp_marketplace_fulfillment_contracts"),
        _result(gate, "webapp_buyer_artifact_access_contracts"),
        _result(gate, "capture_bridge_contracts"),
        _result(gate, "pipeline_launch_gate"),
        _result(gate, "android_bundle_contracts", tags=("android",)),
        _result(gate, "ios_launch_contracts", tags=("iphone", "glasses")),
    ]


def _source_map(source_status: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {entry["source"]: entry for entry in source_status}


def test_summarize_sources_all_pass_contract_ready_statuses() -> None:
    gate = _load_gate_module()

    sources = _source_map(gate.summarize_sources(_base_results(gate)))

    assert sources["iPhone"]["status"] == "external_beta_contract_ready"
    assert sources["glasses"]["status"] == "internal_only_contract_ready"
    assert sources["Android"]["status"] == "internal_only_contract_ready"


def test_summarize_sources_ios_manual_adds_device_suffix() -> None:
    gate = _load_gate_module()
    results = _base_results(gate)
    results[-1] = _result(
        gate,
        "ios_launch_contracts",
        status="manual_required",
        tags=("iphone", "glasses"),
    )

    sources = _source_map(gate.summarize_sources(results))

    assert sources["iPhone"]["status"] == (
        "external_beta_contract_ready_manual_device_confirmation_required"
    )
    assert sources["glasses"]["status"] == (
        "internal_only_contract_ready_manual_device_confirmation_required"
    )


def test_summarize_sources_blocking_webapp_failure_blocks_all_sources() -> None:
    gate = _load_gate_module()
    results = _base_results(gate)
    results[0] = _result(gate, "webapp_request_sync_contracts", status="failed")

    sources = _source_map(gate.summarize_sources(results))

    assert sources["iPhone"]["status"] == "blocked"
    assert sources["glasses"]["status"] == "blocked"
    assert sources["Android"]["status"] == "blocked"


def test_summarize_sources_blocking_buyer_artifact_access_failure_blocks_all_sources() -> None:
    gate = _load_gate_module()
    results = _base_results(gate)
    results[3] = _result(gate, "webapp_buyer_artifact_access_contracts", status="failed")

    sources = _source_map(gate.summarize_sources(results))

    assert sources["iPhone"]["status"] == "blocked"
    assert sources["glasses"]["status"] == "blocked"
    assert sources["Android"]["status"] == "blocked"


def test_summarize_sources_android_operator_toolchain_manual_status() -> None:
    gate = _load_gate_module()
    results = _base_results(gate)
    results[6] = _result(
        gate,
        "android_bundle_contracts",
        status="manual_required",
        tags=("android",),
        evidence_class="operator_toolchain_required",
        skip_reason="ANDROID_HOME or ANDROID_SDK_ROOT is not configured in this shell.",
        evidence_note="Android SDK evidence is operator/toolchain required.",
    )

    sources = _source_map(gate.summarize_sources(results))

    assert sources["Android"]["status"] == (
        "android_contract_evidence_missing_operator_toolchain_required"
    )
    assert "Android is not contract-ready from this report" in sources["Android"]["automated_claim"]
    assert "live-device proof" in sources["Android"]["automated_claim"]


def test_summarize_sources_android_manual_without_toolchain_class() -> None:
    gate = _load_gate_module()
    results = _base_results(gate)
    results[6] = _result(
        gate,
        "android_bundle_contracts",
        status="manual_required",
        tags=("android",),
    )

    sources = _source_map(gate.summarize_sources(results))

    assert sources["Android"]["status"] == (
        "android_contract_evidence_missing_manual_bundle_confirmation_required"
    )
    assert "Android is not contract-ready from this report" in sources["Android"]["automated_claim"]


def test_build_claims_blocks_claims_when_blocking_result_failed() -> None:
    gate = _load_gate_module()

    claims = gate.build_claims([
        _result(gate, "pipeline_launch_gate", status="failed", blocking=True),
    ])

    assert claims["justified"] == []
    assert (
        "Do not claim the paid marketplace beta gate passes while any blocking automated "
        "contract suite is failing."
    ) in claims["not_justified"]


def test_build_claims_names_noncanonical_python_failure() -> None:
    gate = _load_gate_module()

    claims = gate.build_claims(
        [
            _result(
                gate,
                "pipeline_python_interpreter_matrix",
                status="failed",
                blocking=True,
            )
        ]
    )

    assert claims["justified"] == []
    assert (
        "Do not use non-canonical Python interpreter output as launch proof; "
        "rerun the paid gate on Python 3.12."
    ) in claims["not_justified"]


def test_python_interpreter_launch_evidence_result_uses_canonical_matrix() -> None:
    gate = _load_gate_module()
    repo = Path(__file__).resolve().parents[1]

    result = gate.python_interpreter_launch_evidence_result(repo)

    assert result.status == "passed"
    assert result.evidence_class == "canonical_python_interpreter"


def test_python_interpreter_launch_evidence_result_blocks_python_313(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = _load_gate_module()
    repo = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(gate.sys, "version_info", SimpleNamespace(major=3, minor=13))

    result = gate.python_interpreter_launch_evidence_result(repo)

    assert result.status == "failed"
    assert result.blocking is True
    assert result.evidence_class == "canonical_python_interpreter_required"
    assert result.skip_reason == "current_python_3.13_not_canonical_3.12"


def test_build_claims_all_pass_has_contract_level_justified_claims() -> None:
    gate = _load_gate_module()

    claims = gate.build_claims(_base_results(gate))

    assert claims["justified"]
    assert claims["not_justified"]


def test_evidence_boundary_lists_operator_toolchain_evidence() -> None:
    gate = _load_gate_module()
    result = _result(
        gate,
        "android_bundle_contracts",
        status="manual_required",
        evidence_class="operator_toolchain_required",
        skip_reason="ANDROID_HOME or ANDROID_SDK_ROOT is not configured in this shell.",
        evidence_note="Android SDK/Gradle evidence is unrun in this shell.",
    )

    boundary = gate.evidence_boundary([result])

    assert boundary["operator_toolchain_evidence"] == [
        {
            "id": "android_bundle_contracts",
            "label": "Android Bundle Contracts",
            "repo": "BlueprintCapturePipeline",
            "reason": "ANDROID_HOME or ANDROID_SDK_ROOT is not configured in this shell.",
            "note": "Android SDK/Gradle evidence is unrun in this shell.",
        }
    ]


def test_closeout_summary_filters_passed_and_manual_required_entries() -> None:
    gate = _load_gate_module()
    report = {
        "automated_checks": [
            {"label": "passed check", "status": "passed"},
            {"label": "failed check", "status": "failed"},
        ],
        "manual_checks": [
            {"id": "manual_live", "status": "manual_live_evidence_required"},
            {"id": "complete", "status": "complete"},
        ],
    }

    summary = gate.closeout_summary(report)

    assert summary["automated_contracts_prove"] == ["passed check"]
    assert summary["remaining_manual_evidence_ids"] == ["manual_live"]


def test_manual_checks_include_full_operator_legal_payments_and_delivery_ledger() -> None:
    gate = _load_gate_module()

    ids = {item["id"] for item in gate.manual_checks()}

    assert {
        "legal_consent_posture_signoff",
        "operator_dpa_data_processing_terms",
        "cross_border_data_residency_posture",
        "paperclip_ops_relay_secret_rotation",
        "iphone_real_device_claim_flow",
        "glasses_real_device_claim_flow",
        "android_real_device_claim_flow",
        "buyer_payment_settlement",
        "capturer_payout_settlement",
        "stripe_connected_account_live_readiness",
        "payout_exception_monitor_live",
        "identity_kyc_provider_decision",
        "background_check_provider_decision",
        "human_finance_review_owner",
        "buyer_artifact_access",
    } == ids


def test_closeout_summary_does_not_claim_passed_contracts_on_automation_failure() -> None:
    gate = _load_gate_module()
    report = {
        "overall_status": "automation_failed",
        "automated_checks": [
            {"id": "webapp_request_sync_contracts", "label": "WebApp sync", "status": "failed"},
        ],
        "manual_checks": [],
    }

    summary = gate.closeout_summary(report)

    assert "did not pass" in summary["operator_readout"]
    assert "Automated repository contracts passed" not in summary["operator_readout"]
    assert summary["automated_contracts_failed"] == [
        {
            "id": "webapp_request_sync_contracts",
            "label": "WebApp sync",
            "status": "failed",
            "skip_reason": None,
        }
    ]


def test_closeout_summary_treats_operator_toolchain_rows_as_missing_evidence() -> None:
    gate = _load_gate_module()
    report = {
        "overall_status": "automated_contracts_passed_manual_ops_required",
        "automated_checks": [
            {
                "id": "android_bundle_contracts",
                "label": "Android bundle contract",
                "status": "manual_required",
                "blocking": True,
                "evidence_class": "operator_toolchain_required",
            },
            {"id": "pipeline_launch_gate", "label": "Pipeline launch gate", "status": "passed"},
        ],
        "manual_checks": [],
    }

    summary = gate.closeout_summary(report)

    assert "operator-toolchain checks did not run" in summary["operator_readout"]
    assert "missing evidence, not passed contracts" in summary["operator_readout"]
    assert "Automated repository contracts passed." not in summary["operator_readout"]


def test_default_specs_include_buyer_artifact_access_contracts(tmp_path: Path) -> None:
    gate = _load_gate_module()

    specs = gate.default_specs(
        pipeline_repo=tmp_path / "BlueprintCapturePipeline",
        capture_repo=tmp_path / "BlueprintCapture",
        webapp_repo=tmp_path / "Blueprint-WebApp",
        run_ios_tests=False,
    )

    buyer_access = next(
        spec for spec in specs if spec.id == "webapp_buyer_artifact_access_contracts"
    )

    assert buyer_access.command == [
        "npx",
        "vitest",
        "run",
        "server/tests/marketplace-entitlements.test.ts",
        "server/tests/pipeline-routes.test.ts",
        "server/tests/firebase-storage-config.test.ts",
    ]


def test_should_skip_android_sdk_missing_and_evidence_class(monkeypatch, tmp_path: Path) -> None:
    gate = _load_gate_module()
    monkeypatch.delenv("ANDROID_HOME", raising=False)
    monkeypatch.delenv("ANDROID_SDK_ROOT", raising=False)
    monkeypatch.setattr(gate, "android_sdk_root_from_env_or_common_paths", lambda: None)
    spec = gate.CommandSpec(
        id="android_bundle_contracts",
        label="Android bundle contract",
        repo="BlueprintCapture",
        cwd=tmp_path,
        command=["./gradlew", "test"],
        source_tags=("android",),
    )

    reason = gate.should_skip(spec)
    evidence_class, evidence_note = gate.skip_evidence_class(spec, reason)

    assert reason == "ANDROID_HOME or ANDROID_SDK_ROOT is not configured in this shell."
    assert evidence_class == "operator_toolchain_required"
    assert "Android SDK/Gradle unit evidence is unrun" in evidence_note


def test_android_sdk_common_path_detection_feeds_paid_gate_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    gate = _load_gate_module()
    sdk_root = tmp_path / "Library" / "Android" / "sdk"
    platform_tools = sdk_root / "platform-tools"
    platform_tools.mkdir(parents=True)
    (platform_tools / "adb").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    (tmp_path / "gradlew").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    monkeypatch.setattr(gate.Path, "home", staticmethod(lambda: tmp_path))
    monkeypatch.delenv("ANDROID_HOME", raising=False)
    monkeypatch.delenv("ANDROID_SDK_ROOT", raising=False)
    captured: dict[str, object] = {}

    def fake_run(*_args, **kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        return subprocess.CompletedProcess(args=["./gradlew"], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(gate.subprocess, "run", fake_run)
    spec = gate.CommandSpec(
        id="android_bundle_contracts",
        label="Android bundle contract",
        repo="BlueprintCapture",
        cwd=tmp_path,
        command=["./gradlew", "testDebugUnitTest"],
        source_tags=("android",),
    )

    assert gate.should_skip(spec) is None
    result = gate.run_command(spec)

    assert result.status == "passed"
    env = captured["env"]
    assert isinstance(env, dict)
    assert env["ANDROID_HOME"] == str(sdk_root)
    assert env["ANDROID_SDK_ROOT"] == str(sdk_root)


def test_should_skip_ios_xcodebuild_missing_and_evidence_class(monkeypatch, tmp_path: Path) -> None:
    gate = _load_gate_module()
    monkeypatch.setattr(gate.shutil, "which", lambda _name: None)
    spec = gate.CommandSpec(
        id="ios_launch_contracts",
        label="iPhone and glasses Capture unit contracts",
        repo="BlueprintCapture",
        cwd=tmp_path,
        command=["xcodebuild", "test"],
        source_tags=("iphone", "glasses"),
    )

    reason = gate.should_skip(spec)
    evidence_class, evidence_note = gate.skip_evidence_class(spec, reason)

    assert reason == "xcodebuild is not available in this shell."
    assert evidence_class == "operator_toolchain_required"
    assert "iOS simulator unit evidence is unrun" in evidence_note


def test_resolve_ios_simulator_destination_prefers_available_udid(monkeypatch) -> None:
    gate = _load_gate_module()
    payload = {
        "devices": {
            "com.apple.CoreSimulator.SimRuntime.iOS-26-0": [
                {"name": "iPhone 17 Pro", "udid": "IPHONE-17-PRO", "isAvailable": True},
                {"name": "iPhone 17", "udid": "IPHONE-17", "isAvailable": True},
            ],
        },
    }
    monkeypatch.setattr(
        gate.subprocess,
        "check_output",
        lambda *_args, **_kwargs: json.dumps(payload),
    )

    destination = gate.resolve_ios_simulator_destination(
        preferred_name="iPhone 17 Pro",
        preferred_os="26.0",
        preferred_udid=None,
    )

    assert destination == "platform=iOS Simulator,id=IPHONE-17-PRO"


def test_resolve_ios_simulator_destination_falls_back_to_available_iphone(monkeypatch) -> None:
    gate = _load_gate_module()
    payload = {
        "devices": {
            "com.apple.CoreSimulator.SimRuntime.iOS-25-0": [
                {"name": "iPhone 16e", "udid": "IPHONE-16E", "isAvailable": True},
            ],
            "com.apple.CoreSimulator.SimRuntime.iOS-26-0": [
                {"name": "iPad Pro 13-inch (M4)", "udid": "IPAD-PRO", "isAvailable": True},
                {"name": "iPhone 17", "udid": "IPHONE-17", "isAvailable": True},
            ],
        },
    }
    monkeypatch.setattr(
        gate.subprocess,
        "check_output",
        lambda *_args, **_kwargs: json.dumps(payload),
    )

    destination = gate.resolve_ios_simulator_destination(
        preferred_name="iPhone 15",
        preferred_os=None,
        preferred_udid=None,
    )

    assert destination == "platform=iOS Simulator,id=IPHONE-17"


def test_paid_gate_returns_nonzero_for_release_evidence_blockers(tmp_path: Path) -> None:
    gate = _load_gate_module()
    results = [
        gate.CommandResult(
            id="contracts",
            label="contracts",
            repo="BlueprintCapturePipeline",
            command=["pytest"],
            cwd=str(tmp_path),
            status="passed",
            blocking=True,
            source_tags=(),
        )
    ]

    status, exit_code = gate.launch_status_and_exit_code(
        results,
        {
            "status": "blocked",
            "exit_code": 1,
            "blockers": ["missing_evidence:restore_drill"],
        },
    )

    assert status == "release_evidence_blocked"
    assert exit_code == 1


def test_paid_gate_manual_closeout_does_not_override_release_evidence() -> None:
    gate = _load_gate_module()
    report = {
        "overall_status": "release_evidence_blocked",
        "automated_checks": [],
        "manual_checks": [
            {"id": "operator_signoff", "status": "manual_signoff_complete"}
        ],
        "release_evidence_graph": {
            "blockers": ["provider_canary_failed:failed"]
        },
    }

    closeout = gate.closeout_summary(report)

    assert closeout["release_evidence_blockers"] == ["provider_canary_failed:failed"]
    assert "release-bound evidence graph is blocked" in closeout["operator_readout"]
