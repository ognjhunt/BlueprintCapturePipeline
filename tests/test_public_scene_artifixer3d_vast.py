from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest, canonical_json
from blueprint_pipeline.provider_runtime_bundle_contract import (
    PROVIDER_RUNTIME_BUNDLE_KINDS,
    provider_runtime_contract_blockers,
)
from blueprint_pipeline.public_scene_artifixer3d_bundle import (
    build_artifixer3d_bundle,
    materialize_artifixer3d_use_attestation,
)
from blueprint_pipeline.public_scene_artifixer3d_vast import (
    INSTANCE_LABEL_PREFIX,
    PROBE_KIND,
    _materialize_raw_result,
    consume_artifixer3d_paid_attempt_authority_once,
    inspect_artifixer3d_container_image,
    materialize_artifixer3d_paid_attempt_authority,
    materialize_artifixer3d_postblocked_provider_zero,
    run_artifixer3d_vast,
    validate_artifixer3d_bundle,
    validate_artifixer3d_paid_attempt_authority,
)
from blueprint_pipeline.vast_provider_adapter import (
    _blueprint_bundle_preflight,
    _probe_shell_script,
)
from tests.test_public_scene_artifixer3d_bundle import (
    _candidate,
    _repository,
    _source,
)


def _write(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def _record(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _upgrade_parent_authority(candidate_path: Path) -> None:
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    authority_path = Path(candidate["execution_authority"]["path"])
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    authority["paid_compute"] = {
        "provider": "vast",
        "hard_total_spend_cap_usd": 12.0,
        "maximum_concurrent_instances": 1,
        "zero_retry": True,
        "provider_zero_required_for_lane": True,
        "external_instance_allowlist": [47373597],
    }
    authority["authority_digest"] = canonical_digest(
        authority, digest_field="authority_digest"
    )
    _write(authority_path, authority)
    candidate["execution_authority"] = {
        **_record(authority_path),
        "authority_digest": authority["authority_digest"],
    }
    candidate["receipt_digest"] = canonical_digest(
        candidate, digest_field="receipt_digest"
    )
    _write(candidate_path, candidate)


def _bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, count: int = 2
) -> tuple[Path, dict[str, object]]:
    source, commit, tree = _source(tmp_path)
    import blueprint_pipeline.public_scene_artifixer3d_bundle as subject

    monkeypatch.setattr(subject, "ARTIFIXER_COMMIT", commit)
    monkeypatch.setattr(subject, "ARTIFIXER_TREE", tree)
    candidate = _candidate(tmp_path, count=count)
    _upgrade_parent_authority(candidate)
    attestation = tmp_path / "attestation.json"
    materialize_artifixer3d_use_attestation(
        candidate_inputs_receipt_path=candidate,
        output_path=attestation,
        authorized_by="fixture_user",
    )
    receipt = build_artifixer3d_bundle(
        candidate_inputs_receipt_path=candidate,
        use_attestation_path=attestation,
        artifixer_source_directory=source,
        output_root=tmp_path / "bundle",
        repository_root=_repository(tmp_path),
        allowed_active_instance_ids=[],
        artifixer3d_steps=10,
    )
    return tmp_path / "bundle/public_scene_artifixer3d_bundle_receipt.json", receipt


def _prior_chain(tmp_path: Path) -> tuple[Path, Path]:
    dependency_records: dict[str, object] = {}
    for name in (
        "previous_terminal_execution_result",
        "previous_runtime_result",
        "previous_teardown",
        "previous_watchdog",
        "previous_object_store_cleanup",
        "prior_provider_runtime_campaign",
    ):
        path = tmp_path / "prior_dependencies" / f"{name}.json"
        _write(path, {"fixture": name})
        dependency_records[name] = _record(path)
    prior: dict[str, object] = {
        "schema_version": "public_scene_aura_exact_residual_paid_attempt_authority.v1",
        "automatic_paid_retry_authorized": False,
        "maximum_automatic_retries": 0,
        "maximum_paid_attempts": 1,
        "prior_goal_spend_usd": 1.1,
        "aggregate_goal_spend_cap_usd": 12.0,
        "additional_terminal_spend_receipts": [],
        "prior_manual_corrected_attempt_authority": None,
        **dependency_records,
        "authorization_digest": "",
    }
    prior["authorization_digest"] = canonical_digest(
        prior, digest_field="authorization_digest"
    )
    prior_path = tmp_path / "prior_authority.json"
    _write(prior_path, prior)

    root = tmp_path / "prior_terminal"
    teardown_path = root / "vast_provider_run/vast_teardown_manifest.json"
    watchdog_path = root / "independent_vast_watchdog/groot_oscar_runpod_canary_watchdog.json"
    cleanup_path = root / "object_store_staging/wam_provider_object_store_cleanup.json"
    _write(
        teardown_path,
        {
            "schema_version": "vast_teardown_manifest.v1",
            "status": "completed",
            "continuing_spend_from_this_run": False,
        },
    )
    _write(
        watchdog_path,
        {
            "schema_version": "groot_oscar_runpod_canary_watchdog.v1",
            "status": "provider_terminal",
            "provider_absence_confirmed": True,
            "final_global_inventory": {
                "live_resource_count": 0,
                "api_confirmed": True,
            },
        },
    )
    _write(
        cleanup_path,
        {
            "schema_version": "wam_provider_object_store_cleanup.v1",
            "all_objects_absent": True,
            "signed_url_files_removed": True,
        },
    )
    terminal: dict[str, object] = {
        "schema_version": "public_scene_aura_exact_residual_vast_run.v1",
        "status": "completed",
        "retry_cap": 0,
        "continuing_spend_from_this_run": False,
        "all_staged_objects_absent": True,
        "authorization_consumption": {
            "status": "consumed",
            "authorization_digest": prior["authorization_digest"],
        },
        "estimated_cost_usd": 0.6,
        "teardown_manifest_path": str(teardown_path),
        "watchdog_receipt_path": str(watchdog_path),
    }
    terminal_path = root / "public_scene_aura_exact_residual_vast_result.json"
    _write(terminal_path, terminal)
    return prior_path, terminal_path


def test_bundle_kind_and_static_contract_are_registered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt_path, receipt = _bundle(tmp_path, monkeypatch)
    bundle = Path(receipt["bundle"]["path"])
    assert "adp_artifixer3d" in PROVIDER_RUNTIME_BUNDLE_KINDS
    with zipfile.ZipFile(bundle) as archive:
        entrypoint = archive.read(
            "provider_runtime/run_public_scene_artifixer3d.sh"
        ).decode()
        runner = archive.read(
            "provider_runtime/public_scene_artifixer3d_runner.py"
        ).decode()
    assert provider_runtime_contract_blockers(
        provider_bundle_kind="adp_artifixer3d",
        entrypoint_text=entrypoint,
        runner_text=runner,
    ) == []
    assert provider_runtime_contract_blockers(
        provider_bundle_kind="adp_artifixer3d",
        entrypoint_text=entrypoint.replace(
            "artifixer3d_runner_failed_without_result", "removed"
        ),
        runner_text=runner,
    ) == ["provider_entrypoint_missing_runtime_result_crash_fallback"]
    validated = validate_artifixer3d_bundle(receipt_path)
    assert validated["replacement_object_count"] == 2
    assert validated["task_camera_counts"] == {"task_1": 2, "task_2": 2}
    assert validated["publisher_scene_id"] == "840920"
    assert validated["forbidden_external_instance_ids"] == [47373597]
    preflight = _blueprint_bundle_preflight(
        job_dir=tmp_path / "provider_preflight",
        generated_at="2026-08-12T00:00:00Z",
        enable_blueprint_bundle=True,
        enable_isaac_smoke=False,
        provider_bundle_kind="adp_artifixer3d",
        bundle_path=bundle,
        provider_bundle_url="https://object.invalid/bundle",
        provider_output_put_url="https://object.invalid/output",
    )
    assert preflight["status"] == "passed"
    assert preflight["blockers"] == []
    shell = _probe_shell_script(
        "https://heartbeat.invalid",
        enable_blueprint_bundle=True,
        enable_isaac_smoke=False,
        provider_bundle_kind="adp_artifixer3d",
    )
    assert "run_public_scene_artifixer3d.sh" in shell
    assert "BLUEPRINT_PUBLIC_SCENE_ARTIFIXER3D_OUTPUT_DIR" in shell
    assert "public_scene_artifixer3d_runtime_result.json" in shell
    assert "final_candidate_frames" in shell
    assert "BLUEPRINT_VAST_PROVIDER_OUTPUT_UPLOAD_OK" in shell


def test_provider_bundle_validation_remains_generic_through_five_objects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt_path, _ = _bundle(tmp_path, monkeypatch, count=5)
    validated = validate_artifixer3d_bundle(receipt_path)
    assert validated["replacement_object_count"] == 5
    assert validated["task_ids"] == [f"task_{index}" for index in range(1, 6)]
    assert validated["task_camera_counts"] == {
        f"task_{index}": 2 for index in range(1, 6)
    }


def test_paid_authority_chains_prior_spend_and_is_one_shot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt_path, _ = _bundle(tmp_path, monkeypatch)
    bundle = validate_artifixer3d_bundle(receipt_path)
    prior_path, terminal_path = _prior_chain(tmp_path)
    output = tmp_path / "authority.json"
    authority = materialize_artifixer3d_paid_attempt_authority(
        bundle_receipt_path=receipt_path,
        prior_aura_authority_path=prior_path,
        prior_terminal_result_path=terminal_path,
        authorization_reference="fixture-one-shot",
        authorized_by="fixture_user",
        authorized_on="2026-08-12",
        blueprint_commit=bundle["blueprint_source_identity"]["commit"],
        max_hourly_rate_usd=1.5,
        hard_cap_usd=9.0,
        hard_ttl_seconds=21_600,
        output_path=output,
    )
    assert authority["aggregate_goal_spend_before_attempt_usd"] == 1.7
    assert authority["external_active_instance_allowlist"] == []
    assert authority["forbidden_external_instance_ids"] == [47373597]
    assert validate_artifixer3d_paid_attempt_authority(
        authority,
        prepared_bundle=bundle,
        max_hourly_rate_usd=1.5,
        hard_cap_usd=9.0,
        hard_ttl_seconds=21_600,
        allowed_active_instance_ids=[],
    )["authorization_digest"] == authority["authorization_digest"]

    predecessor_result = tmp_path / "predecessor/result.json"
    _write(
        predecessor_result,
        {
            "schema_version": "public_scene_artifixer3d_vast_run.v1",
            "status": "blocked",
            "retry_cap": 0,
            "provider_mutations_performed": 0,
            "all_staged_objects_absent": True,
            "authorization_consumption": {
                "status": "consumed",
                "authorization_digest": authority["authorization_digest"],
            },
        },
    )
    predecessor_cleanup = tmp_path / "predecessor/cleanup.json"
    _write(
        predecessor_cleanup,
        {
            "schema_version": "wam_provider_object_store_cleanup.v1",
            "all_objects_absent": True,
            "signed_url_files_removed": True,
        },
    )
    predecessor_zero = tmp_path / "predecessor/provider_zero.json"
    _write(
        predecessor_zero,
        {
            "schema_version": "artifixer3d_postblocked_provider_zero.v1",
            "attempt_authority_digest": authority["authorization_digest"],
            "provider_mutations_performed_by_attempt": 0,
            "provider_zero_confirmed": True,
            "inventory": {"api_confirmed": True, "live_resource_count": 0},
        },
    )
    successor = materialize_artifixer3d_paid_attempt_authority(
        bundle_receipt_path=receipt_path,
        prior_aura_authority_path=prior_path,
        prior_terminal_result_path=terminal_path,
        prior_artifixer_authority_path=output,
        prior_artifixer_result_path=predecessor_result,
        prior_artifixer_cleanup_path=predecessor_cleanup,
        prior_artifixer_provider_zero_path=predecessor_zero,
        authorization_reference="fixture-successor-one-shot",
        authorized_by="fixture_user",
        authorized_on="2026-08-12",
        blueprint_commit=bundle["blueprint_source_identity"]["commit"],
        max_hourly_rate_usd=1.5,
        hard_cap_usd=9.0,
        hard_ttl_seconds=21_600,
        output_path=tmp_path / "successor_authority.json",
    )
    assert successor["prior_artifixer_attempt"]["authority_digest"] == authority[
        "authorization_digest"
    ]
    assert successor["prior_artifixer_attempt"]["terminal_cost_usd"] == 0.0
    assert successor["prior_artifixer_attempt"]["lineage_cost_usd"] == 0.0
    assert successor["aggregate_goal_spend_before_attempt_usd"] == 1.7
    successor_result = tmp_path / "successor/result.json"
    _write(
        successor_result,
        {
            "schema_version": "public_scene_artifixer3d_vast_run.v1",
            "status": "blocked",
            "retry_cap": 0,
            "estimated_cost_usd": 0.2,
            "all_staged_objects_absent": True,
            "authorization_consumption": {
                "status": "consumed",
                "authorization_digest": successor["authorization_digest"],
            },
        },
    )
    successor_adapter = tmp_path / "successor/adapter.json"
    _write(
        successor_adapter,
        {
            "schema_version": "vast_provider_adapter_result.v1",
            "status": "failed",
            "provider_create_attempted": True,
            "api_call_performed": True,
            "continuing_spend_from_this_run": False,
            "vast_instance_ids": [123],
            "estimated_cost_usd": 0.2,
            "provider_attempt_classification": {
                "classification": "pre_execution_provider_null",
                "provider_bundle_started": False,
                "provider_entrypoint_started": False,
                "provider_output_returned": False,
            },
        },
    )
    successor_zero = tmp_path / "successor/provider_zero.json"
    _write(
        successor_zero,
        {
            "schema_version": "artifixer3d_postblocked_provider_zero.v1",
            "attempt_authority_digest": successor["authorization_digest"],
            "provider_mutations_performed_by_attempt": 1,
            "provider_adapter": _record(successor_adapter),
            "provider_zero_confirmed": True,
            "inventory": {"api_confirmed": True, "live_resource_count": 0},
        },
    )
    third = materialize_artifixer3d_paid_attempt_authority(
        bundle_receipt_path=receipt_path,
        prior_aura_authority_path=prior_path,
        prior_terminal_result_path=terminal_path,
        prior_artifixer_authority_path=tmp_path / "successor_authority.json",
        prior_artifixer_result_path=successor_result,
        prior_artifixer_cleanup_path=predecessor_cleanup,
        prior_artifixer_provider_zero_path=successor_zero,
        authorization_reference="fixture-third-one-shot",
        authorized_by="fixture_user",
        authorized_on="2026-08-12",
        blueprint_commit=bundle["blueprint_source_identity"]["commit"],
        max_hourly_rate_usd=1.5,
        hard_cap_usd=9.0,
        hard_ttl_seconds=21_600,
        output_path=tmp_path / "third_authority.json",
    )
    assert third["prior_artifixer_attempt"]["terminal_cost_usd"] == 0.2
    assert third["prior_artifixer_attempt"]["lineage_cost_usd"] == 0.2
    assert third["aggregate_goal_spend_before_attempt_usd"] == 1.9

    import blueprint_pipeline.public_scene_artifixer3d_vast as subject

    monkeypatch.setattr(subject, "AUTHORIZATION_CONSUMPTION_ROOT", tmp_path / "consumed")
    first = consume_artifixer3d_paid_attempt_authority_once(
        authority, blueprint_commit=authority["blueprint_commit"]
    )
    second = consume_artifixer3d_paid_attempt_authority_once(
        authority, blueprint_commit=authority["blueprint_commit"]
    )
    assert first["status"] == "consumed"
    assert second == {
        "status": "blocked",
        "blockers": ["artifixer3d_paid_attempt_authority_consumed"],
    }


def test_dry_run_is_mutation_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt_path, _ = _bundle(tmp_path, monkeypatch)
    bundle = validate_artifixer3d_bundle(receipt_path)
    import blueprint_pipeline.public_scene_artifixer3d_vast as subject

    monkeypatch.setattr(
        subject,
        "inspect_artifixer3d_container_image",
        lambda **_kwargs: {"status": "completed", "blockers": []},
    )
    result = run_artifixer3d_vast(
        job_dir=tmp_path / "job",
        paid_resource_admission_grant=None,
        execute=False,
        prepared_bundle=bundle,
        max_hourly_rate_usd=1.5,
        hard_cap_usd=9.0,
        hard_ttl_seconds=21_600,
    )
    assert PROBE_KIND == "adp-artifixer3d-exact-support"
    assert result["status"] == "dry_run_ready"
    assert result["provider_mutations_performed"] == 0
    assert result["retry_cap"] == 0


def test_paid_wrapper_uses_canary_scoped_watchdog_and_instance_labels() -> None:
    assert INSTANCE_LABEL_PREFIX == "blueprint-groot-oscar-canary-adp-artifixer3d-"
    assert INSTANCE_LABEL_PREFIX.startswith("blueprint-groot-oscar-canary-")


def test_container_registry_preflight_fails_before_paid_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="manifest unknown"
        ),
    )
    result = inspect_artifixer3d_container_image(
        image_ref="nvcr.io/nvidia/pytorch@sha256:" + "a" * 64,
        output_path=tmp_path / "registry.json",
    )
    assert result["status"] == "blocked"
    assert result["registry_manifest_available"] is False
    assert result["blockers"] == [
        "artifixer3d_container_image_not_registry_resolvable"
    ]


def test_container_registry_preflight_accepts_exact_reachable_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps(
                {
                    "schemaVersion": 2,
                    "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
                    "layers": [{"digest": "sha256:" + "b" * 64, "size": 1}],
                }
            ),
            stderr="",
        ),
    )
    result = inspect_artifixer3d_container_image(
        image_ref="nvcr.io/nvidia/pytorch@sha256:" + "a" * 64,
        output_path=tmp_path / "registry.json",
    )
    assert result["status"] == "completed"
    assert result["digest_pinned"] is True
    assert result["registry_manifest_available"] is True
    assert result["raw_registry_manifest_recorded"] is False


def test_materialize_postblocked_provider_zero_binds_closeout(tmp_path: Path) -> None:
    authority: dict[str, object] = {
        "schema_version": "public_scene_artifixer3d_paid_attempt_authority.v1",
        "authorization_digest": "",
    }
    authority["authorization_digest"] = canonical_digest(
        authority, digest_field="authorization_digest"
    )
    authority_path = tmp_path / "authority.json"
    result_path = tmp_path / "result.json"
    adapter_path = tmp_path / "adapter.json"
    cleanup_path = tmp_path / "cleanup.json"
    watchdog_path = tmp_path / "watchdog.json"
    _write(authority_path, authority)
    _write(
        result_path,
        {
            "schema_version": "public_scene_artifixer3d_vast_run.v1",
            "status": "blocked",
            "authorization_consumption": {
                "status": "consumed",
                "authorization_digest": authority["authorization_digest"],
            },
            "continuing_spend_from_this_run": False,
        },
    )
    _write(
        adapter_path,
        {"provider_create_attempted": True, "continuing_spend_from_this_run": False},
    )
    _write(
        cleanup_path,
        {"all_objects_absent": True, "signed_url_files_removed": True},
    )
    _write(
        watchdog_path,
        {
            "status": "provider_terminal",
            "provider_absence_confirmed": True,
            "final_global_inventory": {
                "api_confirmed": True,
                "live_resource_count": 0,
            },
        },
    )
    receipt = materialize_artifixer3d_postblocked_provider_zero(
        attempt_authority_path=authority_path,
        result_path=result_path,
        adapter_result_path=adapter_path,
        cleanup_path=cleanup_path,
        watchdog_path=watchdog_path,
        output_path=tmp_path / "provider_zero.json",
    )
    assert receipt["provider_mutations_performed_by_attempt"] == 1
    assert receipt["provider_zero_confirmed"] is True
    assert receipt["inventory"]["live_resource_count"] == 0
    assert receipt["receipt_digest"] == canonical_digest(
        receipt, digest_field="receipt_digest"
    )


def test_raw_result_uses_sealed_per_task_camera_count(tmp_path: Path) -> None:
    execution_root = tmp_path / "immutable_execution"
    frame_rows = []
    for index in range(3):
        path = execution_root / "tasks" / "task_variable" / f"{index:05d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"frame-{index}".encode())
        frame_rows.append(
            {
                "frame_index": index,
                "camera_id": f"camera_{index}",
                "repair_pixel_count": index + 1,
                "outside_support_changed_pixels": 0,
                **_record(path),
                "path": f"/provider/runtime_output/tasks/task_variable/{index:05d}.png",
            }
        )
    checkpoint = execution_root / "tasks" / "task_variable" / "ckpt_10.pt"
    checkpoint.write_bytes(b"checkpoint")
    execution = {
        "tasks": [
            {
                "task_id": "task_variable",
                "final_candidate_frames": frame_rows,
                "artifixer3d_checkpoint": {
                    **_record(checkpoint),
                    "path": "/provider/runtime_output/tasks/task_variable/ckpt_10.pt",
                },
                "outside_support_changed_pixels_total": 0,
            }
        ]
    }
    raw = _materialize_raw_result(
        execution=execution,
        execution_root=execution_root,
        bundle={
            "task_ids": ["task_variable"],
            "task_camera_counts": {"task_variable": 3},
            "bundle_sha256": "sha256:bundle",
            "manifest_digest": "sha256:manifest",
            "runtime_request_digest": "sha256:request",
            "replacement_object_count": 1,
        },
        closeout={"provider_zero_confirmed": True},
    )
    assert len(raw["tasks"][0]["final_candidate_frames"]) == 3
    assert raw["appearance_repair_qualified"] is False


def test_canonical_allocator_dry_run_binds_exact_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt_path, _ = _bundle(tmp_path, monkeypatch)
    bundle = validate_artifixer3d_bundle(receipt_path)
    commit = bundle["blueprint_source_identity"]["commit"]
    import blueprint_pipeline.public_scene_artifixer3d_vast as vast_subject
    import blueprint_pipeline.paid_resource_allocator as allocator

    monkeypatch.setattr(
        vast_subject,
        "inspect_artifixer3d_container_image",
        lambda **_kwargs: {"status": "completed", "blockers": []},
    )
    monkeypatch.setattr(
        allocator,
        "_control_plane_checkout_blockers",
        lambda: (
            [],
            {
                "orchestrator_source_commit": commit,
                "checkout_clean": True,
                "identity_probe_ran": True,
            },
        ),
    )
    admission = tmp_path / "allocator/admission.json"
    adapter = tmp_path / "allocator/adapter.json"
    result = allocator.main(
        [
            "gpu-canary",
            "--provider",
            "vast",
            "--probe-kind",
            PROBE_KIND,
            "--expected-source-commit",
            commit,
            "--adp-artifixer3d-bundle-receipt",
            str(receipt_path),
            "--adp-job-dir",
            str(tmp_path / "allocator/job"),
            "--adp-max-hourly-rate-usd",
            "1.5",
            "--adp-max-spend-usd",
            "9.0",
            "--adp-hard-ttl-seconds",
            "21600",
            "--admission-out",
            str(admission),
            "--adapter-output",
            str(adapter),
        ]
    )
    assert result == 0
    admitted = json.loads(admission.read_text(encoding="utf-8"))
    dry_run = json.loads(adapter.read_text(encoding="utf-8"))
    assert admitted["status"] == "admitted"
    assert admitted["allocation_binding"]["bundle_sha256"] == bundle["bundle_sha256"]
    assert admitted["allocation_binding"]["retry_cap"] == 0
    assert admitted["raw_dataset_bytes_upload_authorized"] is False
    assert dry_run["status"] == "dry_run_ready"
    assert dry_run["provider_mutations_performed"] == 0
