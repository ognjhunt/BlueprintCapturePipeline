from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


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


def test_summarize_sources_android_operator_toolchain_manual_status() -> None:
    gate = _load_gate_module()
    results = _base_results(gate)
    results[5] = _result(
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
        "internal_only_contract_ready_operator_toolchain_evidence_required"
    )
    assert "operator/toolchain requirement" in sources["Android"]["automated_claim"]


def test_summarize_sources_android_manual_without_toolchain_class() -> None:
    gate = _load_gate_module()
    results = _base_results(gate)
    results[5] = _result(
        gate,
        "android_bundle_contracts",
        status="manual_required",
        tags=("android",),
    )

    sources = _source_map(gate.summarize_sources(results))

    assert sources["Android"]["status"] == (
        "internal_only_contract_ready_manual_bundle_confirmation_required"
    )


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


def test_should_skip_android_sdk_missing_and_evidence_class(monkeypatch, tmp_path: Path) -> None:
    gate = _load_gate_module()
    monkeypatch.delenv("ANDROID_HOME", raising=False)
    monkeypatch.delenv("ANDROID_SDK_ROOT", raising=False)
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
