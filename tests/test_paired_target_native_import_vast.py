from __future__ import annotations

import json
from pathlib import Path
import zipfile

import pytest

import blueprint_pipeline.paid_resource_allocator as allocator
import blueprint_pipeline.paired_target_native_import_vast as vast
from blueprint_pipeline.common import write_json
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.paired_target_native_import_bundle import (
    build_paired_target_native_import_bundle,
    validate_paired_target_native_import_bundle,
)
from blueprint_pipeline.provider_runtime_bundle_contract import (
    provider_runtime_contract_blockers,
)
from blueprint_pipeline.vast_provider_adapter import _blueprint_bundle_preflight
from blueprint_pipeline.wam_provider_output import inspect_provider_runtime_output_zip


COMMIT = "a" * 40


def _source_request(tmp_path: Path) -> Path:
    asset = tmp_path / "asset.usda"
    asset.write_text("#usda 1.0\n", encoding="utf-8")
    import hashlib

    sha = "sha256:" + hashlib.sha256(asset.read_bytes()).hexdigest()
    source = {
        "schema_version": "paired_target_native_render_request.v1",
        "status": "native_render_requests_materialized_pending_isaac_execution",
        "scene_id": "scene",
        "replacement_object_count": 1,
        "native_isaac_executed": False,
        "provider_allocation_performed": False,
        "generated_output_is_capture_or_physical_evidence": False,
        "tasks": [
            {
                "task_id": "task_a",
                "co_present_replacements": [
                    {
                        "task_id": "task_a",
                        "asset_id": "washer",
                        "visual_usd": {
                            "path": str(asset),
                            "size_bytes": asset.stat().st_size,
                            "sha256": sha,
                        },
                        "asset_frame_registration": {
                            "registration_digest": "sha256:" + "b" * 64
                        },
                        "task_subject": True,
                        "passive_co_present": False,
                    }
                ],
            }
        ],
        "receipt_digest": "",
    }
    source["receipt_digest"] = canonical_digest(source, digest_field="receipt_digest")
    path = tmp_path / "request.json"
    write_json(path, source)
    return path


def _bundle(tmp_path: Path) -> tuple[Path, dict]:
    root = tmp_path / "bundle"
    build_paired_target_native_import_bundle(
        native_render_request_path=_source_request(tmp_path),
        runner_path=Path(__file__).parents[1]
        / "scripts"
        / "run_paired_target_native_import_probe.py",
        output_root=root,
        implementation_commit=COMMIT,
    )
    path = root / "paired_target_native_import_bundle_receipt.v1.json"
    return path, validate_paired_target_native_import_bundle(path)


def _authority(bundle: dict) -> dict:
    value = {
        "schema_version": vast.PAID_ATTEMPT_AUTHORITY_SCHEMA_VERSION,
        "authority_kind": "explicit_user_direction_in_current_goal",
        "authority_reference": "goal authority",
        "authorized_by": "user",
        "authorized_on": "2026-08-13",
        "purpose": "one_shot_paired_target_native_import_probe",
        "provider": "vast",
        "paid_compute_authorized": True,
        "maximum_paid_attempts": 1,
        "maximum_provider_allocations": 1,
        "maximum_automatic_retries": 0,
        "automatic_paid_retry_authorized": False,
        "zero_retry": True,
        "bundle_receipt": {"path": bundle["receipt_path"]},
        "bundle_receipt_digest": bundle["receipt_digest"],
        "bundle_sha256": bundle["bundle_sha256"],
        "probe_spec_sha256": bundle["probe_spec_sha256"],
        "source_request_digest": bundle["source_request_digest"],
        "replacement_count": bundle["replacement_count"],
        "blueprint_commit": COMMIT,
        "container_image": bundle["container_image"],
        "hard_attempt_spend_cap_usd": 1.0,
        "maximum_hourly_rate_usd": 1.0,
        "maximum_single_resource_ttl_seconds": 3600,
        "aggregate_goal_spend_before_attempt_usd": 10.0,
        "aggregate_goal_spend_cap_usd": 12.0,
        "prior_terminal_artifixer": {},
        "active_instance_allowlist": {
            "external_provider_owned": [],
            "same_goal_concurrent": [],
        },
        "native_simulator_import_probe_only": True,
        "candidate_policy_queried": False,
        "raw_nonredistributable_bytes_uploaded": False,
        "canonical_interiorgs_uploaded_or_mutated": False,
        "physical_success_established": False,
        "authorization_digest": "",
    }
    receipt_path = Path(bundle["receipt_path"])
    value["bundle_receipt"].update(
        {
            "size_bytes": receipt_path.stat().st_size,
            "sha256": vast._sha256(receipt_path),
        }
    )
    value["authorization_digest"] = canonical_digest(
        value, digest_field="authorization_digest"
    )
    return value


def test_bundle_reopen_and_provider_contract_cover_dynamic_assets(tmp_path: Path) -> None:
    _, bundle = _bundle(tmp_path)
    with zipfile.ZipFile(bundle["bundle_path"]) as archive:
        entrypoint = archive.read(
            "provider_runtime/run_paired_target_native_import_probe.sh"
        ).decode()
        runner = archive.read(
            "provider_runtime/run_paired_target_native_import_probe.py"
        ).decode()
    assert provider_runtime_contract_blockers(
        provider_bundle_kind="paired_target_native_import",
        entrypoint_text=entrypoint,
        runner_text=runner,
    ) == []
    preflight = _blueprint_bundle_preflight(
        job_dir=tmp_path / "preflight",
        generated_at="2026-08-13T00:00:00Z",
        enable_blueprint_bundle=True,
        enable_isaac_smoke=True,
        provider_bundle_kind="paired_target_native_import",
        bundle_path=Path(bundle["bundle_path"]),
        provider_bundle_url="https://example.invalid/bundle",
        provider_output_put_url="https://example.invalid/output",
    )
    assert preflight["status"] == "passed"
    assert preflight["blockers"] == []


def test_output_inspector_recognizes_paired_result(tmp_path: Path) -> None:
    result = {
        "schema_version": "paired_target_native_import_runtime_result.v1",
        "status": "completed",
        "blockers": [],
    }
    output = tmp_path / "output.zip"
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr(
            "paired_target_native_import_runtime_result.v1.json", json.dumps(result)
        )
    inspected = inspect_provider_runtime_output_zip(output, expected_video_count=0)
    assert inspected["runtime_result_present"] is True
    assert inspected["runtime_result_status"] == "completed"


def test_dry_run_validates_authority_and_never_mutates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, bundle = _bundle(tmp_path)
    authority = _authority(bundle)
    monkeypatch.setattr(
        vast,
        "validate_paired_target_native_import_paid_attempt_authority",
        lambda value, **kwargs: value,
    )
    monkeypatch.setattr(
        vast,
        "stage_wam_provider_bundle_object_store",
        lambda **kwargs: pytest.fail("dry run staged bytes"),
    )
    result = vast.run_paired_target_native_import_vast(
        job_dir=tmp_path / "job",
        prepared_bundle=bundle,
        paid_resource_admission_grant=None,
        paid_attempt_authority=authority,
        execute=False,
    )
    assert result["status"] == "dry_run_ready"
    assert result["provider_mutations_performed"] == 0


def test_authority_consumption_is_single_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BLUEPRINT_SPEND_AUTHORITY_ROOT", str(tmp_path / "authority-root"))
    authority = {"authorization_digest": "sha256:" + "c" * 64, "bundle_sha256": "x"}
    first = vast.consume_paired_target_native_import_authority_once(
        authority, blueprint_commit=COMMIT
    )
    second = vast.consume_paired_target_native_import_authority_once(
        authority, blueprint_commit=COMMIT
    )
    assert first["status"] == "consumed"
    assert second["status"] == "blocked"


def test_live_run_requires_qualified_runtime_watchdog_cleanup_and_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, bundle = _bundle(tmp_path)
    authority = _authority(bundle)
    monkeypatch.setenv("BLUEPRINT_SPEND_AUTHORITY_ROOT", str(tmp_path / "authority-root"))
    monkeypatch.setattr(
        vast,
        "validate_paired_target_native_import_paid_attempt_authority",
        lambda value, **kwargs: value,
    )

    def fake_stage(**kwargs):
        root = Path(kwargs["job_dir"])
        root.mkdir(parents=True)
        for name in (
            "provider_bundle_url.txt",
            "provider_output_put_url.txt",
            "provider_output_get_url.txt",
        ):
            (root / name).write_text("https://example.invalid/bound")
        return {"status": "completed"}

    class Handle:
        started_instance_id_path = tmp_path / "started.txt"

    def fake_adapter(**kwargs):
        output = Path(kwargs["provider_runtime_output_zip"])
        output.parent.mkdir(parents=True)
        runtime = {
            "schema_version": "paired_target_native_import_runtime_result.v1",
            "status": "completed",
            "request_digest": bundle["request_digest"],
            "replacement_count": 1,
            "native_isaac_executed": True,
            "all_replacements_import_qualified": True,
            "candidate_policy_queried": False,
            "physical_equivalence_claimed": False,
            "blockers": [],
            "result_digest": "",
        }
        runtime["result_digest"] = canonical_digest(runtime, digest_field="result_digest")
        with zipfile.ZipFile(output, "w") as archive:
            archive.writestr(vast.RESULT_FILENAME, json.dumps(runtime))
        provider = Path(kwargs["job_dir"])
        write_json(
            provider / "vast_teardown_manifest.json",
            {
                "continuing_spend_from_this_run": False,
                "vast_instance_ids": [77],
            },
        )
        write_json(
            provider / "vast_provider_adapter_result.json",
            {"continuing_spend_from_this_run": False},
        )
        return {
            "status": "completed",
            "provider_create_attempted": True,
            "estimated_cost_usd": 0.08,
        }

    monkeypatch.setattr(vast, "stage_wam_provider_bundle_object_store", fake_stage)
    monkeypatch.setattr(vast, "arm_independent_vast_watchdog", lambda **kwargs: ({"status": "armed"}, Handle()))
    monkeypatch.setattr(vast, "run_vast_provider_adapter", fake_adapter)
    monkeypatch.setattr(
        vast,
        "cleanup_staged_wam_provider_objects",
        lambda path: {"all_objects_absent": True, "signed_url_files_removed": True},
    )
    monkeypatch.setattr(
        vast,
        "close_independent_vast_watchdog",
        lambda **kwargs: {"status": "provider_terminal", "provider_absence_confirmed": True},
    )
    monkeypatch.setattr(vast, "seal_lane_terminal_artifacts", lambda result, **kwargs: result)
    result = vast.run_paired_target_native_import_vast(
        job_dir=tmp_path / "job",
        prepared_bundle=bundle,
        paid_resource_admission_grant=object(),  # type: ignore[arg-type]
        paid_attempt_authority=authority,
        execute=True,
    )
    assert result["status"] == "completed"
    assert result["estimated_cost_usd"] == 0.08
    assert result["continuing_spend_from_this_run"] is False
    assert result["all_staged_objects_absent"] is True
    assert result["candidate_policy_queried"] is False


def test_allocator_binds_new_mode_without_execute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt_path, bundle = _bundle(tmp_path)
    authority = _authority(bundle)
    authority_path = tmp_path / "authority.json"
    write_json(authority_path, authority)
    monkeypatch.setattr(
        allocator,
        "_control_plane_checkout_blockers",
        lambda: ([], {"orchestrator_source_commit": COMMIT, "checkout_clean": True}),
    )
    monkeypatch.setattr(
        allocator,
        "validate_paired_target_native_import_paid_attempt_authority",
        lambda value, **kwargs: value,
    )
    observed = {}

    def fake_run(**kwargs):
        observed.update(kwargs)
        return {"status": "dry_run_ready"}

    monkeypatch.setattr(allocator, "run_paired_target_native_import_vast", fake_run)
    args = [
        "gpu-canary",
        "--probe-kind",
        vast.PROBE_KIND,
        "--provider",
        "vast",
        "--provider-launch-request",
        str(tmp_path / "unused.json"),
        "--release-evidence",
        str(tmp_path / "unused2.json"),
        "--model-cache-evidence",
        str(tmp_path / "unused3.json"),
        "--preflight-bundle",
        str(tmp_path / "unused4.json"),
        "--admission-out",
        str(tmp_path / "admission.json"),
        "--bound-request-out",
        str(tmp_path / "unused5.json"),
        "--adapter-output",
        str(tmp_path / "adapter.json"),
        "--pod-name",
        "paired-native",
        "--expected-source-commit",
        COMMIT,
        "--paired-target-native-import-bundle-receipt",
        str(receipt_path),
        "--paired-target-native-import-attempt-authority",
        str(authority_path),
        "--adp-job-dir",
        str(tmp_path / "job"),
        "--adp-max-hourly-rate-usd",
        "1.0",
        "--adp-max-spend-usd",
        "1.0",
        "--adp-hard-ttl-seconds",
        "3600",
    ]
    assert allocator.main(args) == 0
    assert observed["execute"] is False
    assert observed["paid_attempt_authority"] == authority
    admission = json.loads((tmp_path / "admission.json").read_text())
    assert admission["status"] == "admitted"
    assert admission["allocation_binding"]["replacement_count"] == 1
