from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

import pytest

from blueprint_pipeline import paid_resource_allocator as allocator
from blueprint_pipeline import policy_ranking_successor_gpu_admission as admission
from blueprint_pipeline.paid_resource_admission import PaidResourceAdmissionGrant


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "docs/experiments/policy_ranking_successor_experiment_20260727"


def _load(name: str) -> dict[str, Any]:
    return json.loads((EXPERIMENT / name).read_text(encoding="utf-8"))


def _inspect_bundle(path: Path | None = None) -> dict[str, Any]:
    return admission.inspect_successor_bundle(
        path or EXPERIMENT / "cosmos3_successor_provider_bundle.zip",
        receipt=_load("cosmos3_successor_provider_bundle_receipt.json"),
        smoke_inventory=_load("smoke_request_inventory.json"),
    )


def test_frozen_successor_bundle_passes_integrity_inspection() -> None:
    result = _inspect_bundle()

    assert result["status"] == "passed"
    assert result["blockers"] == []
    assert result["bundle_sha256"] == (
        "26d28cba2cbecaa9ab1373e9192b2f84e952f3313bea190ea08661d4cd14dcad"
    )


def test_successor_gpu_admission_requires_explicit_authorization() -> None:
    result = admission.build_successor_gpu_admission(
        authorization={},
        environment=_load("environment_and_source_manifest.json"),
        smoke_inventory=_load("smoke_request_inventory.json"),
        provider_preflight=_load("vast_compute_preflight.json"),
        bundle_inspection=_inspect_bundle(),
        expected_source_commit="a" * 40,
        execute=False,
    )

    assert result["status"] == "blocked"
    assert "successor_compute_not_explicitly_authorized" in result["blockers"]
    assert result["provider_mutations_performed"] == 0


def test_successor_gpu_admission_accepts_only_frozen_rtx_envelope() -> None:
    preflight = _load("vast_compute_preflight.json")
    result = admission.build_successor_gpu_admission(
        authorization=_load("compute_authorization.json"),
        environment=_load("environment_and_source_manifest.json"),
        smoke_inventory=_load("smoke_request_inventory.json"),
        provider_preflight=preflight,
        bundle_inspection=_inspect_bundle(),
        expected_source_commit="b" * 40,
        execute=True,
        observed_now_epoch=float(preflight["observed_at_epoch"]) + 1,
    )

    assert result["status"] == "admitted"
    assert result["blockers"] == []
    assert result["limits"]["hard_cap_usd"] == 3.25
    assert result["limits"]["allowed_gpu_keywords"] == ["RTX PRO 6000"]
    assert result["shared_paid_lane_admission"]["status"] == "admitted"


def test_successor_gpu_lane_passes_opaque_grant_and_hardware_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    def fake_runner(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"status": "completed", "blockers": []}

    monkeypatch.setattr(admission, "run_vast_wam_authorized_runner", fake_runner)
    monkeypatch.setattr(
        admission,
        "AUTHORIZATION_CONSUMPTION_ROOT",
        tmp_path / "authority-consumption",
    )
    preflight = _load("vast_compute_preflight.json")
    result = admission.run_successor_gpu_lane(
        authorization_path=EXPERIMENT / "compute_authorization.json",
        environment_path=EXPERIMENT / "environment_and_source_manifest.json",
        smoke_inventory_path=EXPERIMENT / "smoke_request_inventory.json",
        provider_preflight_path=EXPERIMENT / "vast_compute_preflight.json",
        provider_bundle_path=EXPERIMENT / "cosmos3_successor_provider_bundle.zip",
        provider_bundle_receipt_path=(
            EXPERIMENT / "cosmos3_successor_provider_bundle_receipt.json"
        ),
        admission_out=tmp_path / "admission.json",
        bound_request_out=tmp_path / "bound.json",
        adapter_output=tmp_path / "adapter.json",
        job_dir=tmp_path / "job",
        public_base_url="https://example.test",
        token_file=tmp_path / "token",
        secret_env_file=tmp_path / "urls.env",
        output_path=tmp_path / "output.zip",
        session_budget_ledger=tmp_path / "budget.json",
        expected_source_commit="c" * 40,
        execute=True,
        observed_now_epoch=float(preflight["observed_at_epoch"]) + 1,
    )

    assert result["status"] == "completed"
    assert isinstance(captured["paid_resource_admission_grant"], PaidResourceAdmissionGrant)
    assert captured["hard_cap_usd"] == 3.25
    assert captured["max_live_minutes"] == 180
    assert captured["disk_gb"] == 250
    assert captured["min_gpu_ram_mb"] == 95_000
    assert captured["max_compute_cap"] == 0
    assert captured["gpu_selection_policy"]["allowed_gpu_keywords"] == (
        "RTX PRO 6000",
    )

    second = admission.run_successor_gpu_lane(
        authorization_path=EXPERIMENT / "compute_authorization.json",
        environment_path=EXPERIMENT / "environment_and_source_manifest.json",
        smoke_inventory_path=EXPERIMENT / "smoke_request_inventory.json",
        provider_preflight_path=EXPERIMENT / "vast_compute_preflight.json",
        provider_bundle_path=EXPERIMENT / "cosmos3_successor_provider_bundle.zip",
        provider_bundle_receipt_path=(
            EXPERIMENT / "cosmos3_successor_provider_bundle_receipt.json"
        ),
        admission_out=tmp_path / "admission-second.json",
        bound_request_out=tmp_path / "bound-second.json",
        adapter_output=tmp_path / "adapter-second.json",
        job_dir=tmp_path / "job-second",
        public_base_url="https://example.test",
        token_file=tmp_path / "token",
        secret_env_file=tmp_path / "urls.env",
        output_path=tmp_path / "output-second.zip",
        session_budget_ledger=tmp_path / "budget-second.json",
        expected_source_commit="c" * 40,
        execute=True,
        observed_now_epoch=float(preflight["observed_at_epoch"]) + 1,
    )
    assert second["status"] == "blocked"
    assert second["blockers"] == ["successor_compute_authorization_already_consumed"]


def test_successor_bundle_is_bound_to_receipt_and_embedded_inputs(
    tmp_path: Path,
) -> None:
    altered = tmp_path / "altered.zip"
    shutil.copyfile(EXPERIMENT / "cosmos3_successor_provider_bundle.zip", altered)
    with zipfile.ZipFile(altered, "a") as archive:
        archive.writestr("provider_runtime/unregistered_marker.txt", "changed")

    result = _inspect_bundle(altered)

    assert result["status"] == "blocked"
    assert "successor_cosmos_provider_bundle_receipt_hash_mismatch" in result["blockers"]


def test_successor_lane_writes_blocked_artifacts_for_unreadable_input(
    tmp_path: Path,
) -> None:
    admission_out = tmp_path / "admission.json"
    bound_out = tmp_path / "bound.json"
    adapter_out = tmp_path / "adapter.json"
    result = admission.run_successor_gpu_lane(
        authorization_path=EXPERIMENT / "compute_authorization.json",
        environment_path=tmp_path / "missing-environment.json",
        smoke_inventory_path=EXPERIMENT / "smoke_request_inventory.json",
        provider_preflight_path=EXPERIMENT / "vast_compute_preflight.json",
        provider_bundle_path=EXPERIMENT / "cosmos3_successor_provider_bundle.zip",
        provider_bundle_receipt_path=(
            EXPERIMENT / "cosmos3_successor_provider_bundle_receipt.json"
        ),
        admission_out=admission_out,
        bound_request_out=bound_out,
        adapter_output=adapter_out,
        job_dir=tmp_path / "job",
        public_base_url=None,
        token_file=None,
        secret_env_file=None,
        output_path=None,
        session_budget_ledger=None,
        expected_source_commit="e" * 40,
        execute=False,
    )

    assert result["status"] == "blocked"
    assert "successor_environment_unreadable" in result["blockers"]
    assert admission_out.is_file()
    assert bound_out.is_file()
    assert adapter_out.is_file()


def test_paid_resource_allocator_dispatches_successor_lane_only_through_probe_kind(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        allocator,
        "_source_checkout_blockers",
        lambda _commit: ([], "d" * 40),
    )

    def fake_lane(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"status": "dry_run_ready", "provider_mutations_performed": 0}

    monkeypatch.setattr(allocator, "run_successor_gpu_lane", fake_lane)
    code = allocator.main(
        [
            "gpu-canary",
            "--probe-kind",
            allocator.POLICY_RANKING_SUCCESSOR_COSMOS_PROBE_KIND,
            "--provider-launch-request",
            "authorization.json",
            "--release-evidence",
            "environment.json",
            "--model-cache-evidence",
            "inventory.json",
            "--preflight-bundle",
            "preflight.json",
            "--episode-bundle",
            "bundle.zip",
            "--successor-bundle-receipt",
            "receipt.json",
            "--admission-out",
            str(tmp_path / "admission.json"),
            "--bound-request-out",
            str(tmp_path / "bound.json"),
            "--adapter-output",
            str(tmp_path / "adapter.json"),
            "--pod-name",
            str(tmp_path / "job"),
            "--expected-source-commit",
            "d" * 40,
        ]
    )

    assert code == 0
    assert json.loads(capsys.readouterr().out) == {"success": True}
    assert captured["execute"] is False
    assert captured["provider_bundle_path"] == "bundle.zip"
    assert captured["provider_bundle_receipt_path"] == "receipt.json"
    assert captured["expected_source_commit"] == "d" * 40
