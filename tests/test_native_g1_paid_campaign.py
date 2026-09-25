"""Paid G1 admission must bind exact bytes and reject incomplete outcomes."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline import native_g1_paid_campaign as lane
from blueprint_pipeline.native_g1_provider_bundle import (
    MANIFEST,
    PROVIDER_BUNDLE_KIND,
    SCHEMA,
    _runtime_code_files,
    load_verified_g1_provider_bundle,
)
from blueprint_pipeline.provider_runtime_bundle_contract import (
    provider_runtime_contract_blockers,
)
from blueprint_pipeline import vast_provider_adapter as vast
from blueprint_pipeline.wam_provider_output import inspect_provider_runtime_output_zip


def _args(tmp_path: Path, **changes: object) -> SimpleNamespace:
    value = {
        "provider": "vast",
        "adp_job_dir": str(tmp_path / "job"),
        "admission_out": str(tmp_path / "admission.json"),
        "adapter_output": str(tmp_path / "result.json"),
        "adp_max_hourly_rate_usd": 1.0,
        "adp_max_spend_usd": 4.0,
        "adp_hard_ttl_seconds": 7200,
        "adp_allowed_active_vast_instance_id": [],
        "adp_machine_avoidlist": None,
        "execute": True,
        "g1_campaign_bundle_receipt": None,
        "g1_campaign_manipulation_packet": None,
        "g1_campaign_movement_packet": None,
        "g1_campaign_book_handoff": None,
        "g1_campaign_navigation_authority": None,
        "g1_campaign_publisher_source": None,
        "g1_campaign_runtime_source_receipt": None,
        "g1_campaign_rights_review": [],
    }
    value.update(changes)
    return SimpleNamespace(**value)


def test_exact_main_and_sealed_receipt_required_before_paid_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        lane, "run_arena_native_control_vast",
        lambda **_kwargs: pytest.fail("provider called without admission"),
    )
    result = lane.dispatch_g1_paid_campaign(
        _args(tmp_path),
        control_identity={
            "orchestrator_source_commit": "a" * 40,
            "origin_main_commit": "b" * 40,
            "remote_main_commit": "b" * 40,
        },
        control_blockers=[],
    )
    assert result["status"] == "blocked"
    assert "g1_paid_campaign_controller_not_exact_main" in result["blockers"]
    assert "g1_paid_campaign_execute_requires_dry_run_bundle_receipt" in result["blockers"]
    assert json.loads((tmp_path / "admission.json").read_text())["status"] == "blocked"


def test_bundle_receipt_rejects_mutated_bytes(tmp_path: Path) -> None:
    commit = "a" * 40
    manifest = {
        "schema_version": SCHEMA,
        "status": "ready",
        "provider_bundle_kind": PROVIDER_BUNDLE_KIND,
        "implementation_commit": commit,
    }
    manifest["manifest_digest"] = canonical_digest(manifest, digest_field="manifest_digest")
    archive_path = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(MANIFEST, json.dumps(manifest))
    receipt = {
        **manifest,
        "bundle_path": str(archive_path),
        "bundle_size_bytes": archive_path.stat().st_size,
        "bundle_sha256": "sha256:" + hashlib.sha256(archive_path.read_bytes()).hexdigest(),
    }
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt))
    assert load_verified_g1_provider_bundle(
        path, expected_implementation_commit=commit
    )["bundle_sha256"] == receipt["bundle_sha256"]
    with zipfile.ZipFile(archive_path, "a") as archive:
        archive.writestr("extra.txt", "changed")
    with pytest.raises(ValueError, match="g1_provider_bundle_receipt_binding_invalid"):
        load_verified_g1_provider_bundle(path, expected_implementation_commit=commit)


def test_bundle_code_closure_includes_shared_subpackages() -> None:
    package = Path(lane.__file__).resolve().parent
    relative = {path.relative_to(package).as_posix() for path in _runtime_code_files(package)}
    assert "native_g1_provider_runtime.py" in relative
    assert "core/common.py" in relative
    assert "__pycache__" not in relative


def test_g1_provider_kind_uses_isaac_and_retains_episode_media() -> None:
    assert vast._is_isaac_provider_bundle(PROVIDER_BUNDLE_KIND)
    assert vast._provider_expected_video_count(PROVIDER_BUNDLE_KIND) == 0
    assert vast._resolve_launch_mode(
        requested="auto",
        enable_isaac_smoke=True,
        enable_blueprint_bundle=True,
        provider_bundle_kind=PROVIDER_BUNDLE_KIND,
    ) == "ssh_direct"
    entrypoint = " ".join((
        "native_task_runtime_source_provision", "native_g1_provider_runtime",
        "native_g1_provider_campaign_result.v1.json",
        "g1_provider_runner_exited_without_terminal_result",
    ))
    runner = " ".join((
        "verify_g1_provider_inputs", "execute_g1_policy_runtime_build",
        "run_g1_development_pair", "_query_count", "development_only",
    ))
    assert provider_runtime_contract_blockers(
        provider_bundle_kind=PROVIDER_BUNDLE_KIND,
        entrypoint_text=entrypoint,
        runner_text=runner,
    ) == []
    script = vast._probe_shell_script(
        "https://example.invalid/heartbeat",
        enable_isaac_smoke=True,
        enable_blueprint_bundle=True,
        provider_bundle_kind=PROVIDER_BUNDLE_KIND,
    )
    assert "preserve_all_output = True" in script
    assert "if preserve_all_output or size <= size_limit" in script


def test_returned_g1_terminal_result_is_detected_by_provider_adapter(tmp_path: Path) -> None:
    path = tmp_path / "output.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "native_g1_provider_campaign_result.v1.json",
            json.dumps({"status": "completed", "blockers": []}),
        )
    inspected = inspect_provider_runtime_output_zip(path, expected_video_count=0)
    assert inspected["runtime_result_present"] is True
    assert inspected["runtime_result_status"] == "completed"


def test_completed_transport_without_episode_evidence_is_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commit = "a" * 40
    bundle = {
        "status": "ready",
        "bundle_sha256": "sha256:" + "b" * 64,
        "campaign_plan_digest": "sha256:" + "c" * 64,
    }
    monkeypatch.setattr(lane, "load_verified_g1_provider_bundle", lambda *_a, **_k: bundle)
    monkeypatch.setattr(
        lane, "run_arena_native_control_vast",
        lambda **_kwargs: {
            "status": "completed",
            "attempt_root": str(tmp_path / "missing-evidence"),
            "continuing_spend_from_this_run": False,
            "blockers": [],
        },
    )
    result = lane.dispatch_g1_paid_campaign(
        _args(tmp_path, g1_campaign_bundle_receipt=str(tmp_path / "receipt.json")),
        control_identity={
            "orchestrator_source_commit": commit,
            "origin_main_commit": commit,
            "remote_main_commit": commit,
        },
        control_blockers=[],
    )
    assert result["status"] == "blocked"
    assert result["continuing_spend_from_this_run"] is False
    assert result["blockers"] == ["g1_paid_campaign_output_verification_failed:ValueError"]
