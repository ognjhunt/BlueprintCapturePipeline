from __future__ import annotations

import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
from types import SimpleNamespace
import urllib.error
import zipfile

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.paid_resource_admission import (
    PAID_LANE_ADMISSION_SCHEMA_VERSION,
    build_paid_lane_admission,
    require_paid_resource_admission,
)
import blueprint_pipeline.semantic_teacher_image_edit_vast as vast
from blueprint_pipeline.semantic_teacher_image_edit_vast import (
    PROBE_KIND,
    SemanticTeacherImageEditVastError,
    _bootstrap_script,
    _default_result_fetcher,
    _validate_bundle_runtime_bindings,
    run_semantic_teacher_image_edit_vast,
)
from blueprint_pipeline.semantic_teacher_image_edit_worker import (
    RUNTIME_RESULT_SCHEMA_VERSION,
)


SOURCE_COMMIT = "a" * 40
IMAGE = "registry.example/blueprint/semantic-teacher@sha256:" + "b" * 64
TOKEN = "openai_fixture_secret_value"
WORKER_CODE_DIGEST = "sha256:" + "d" * 64
INPUT_URL = "https://objects.example/bundle?signature=input-secret"
PUT_URL = "https://objects.example/output?signature=put-secret"
GET_URL = "https://objects.example/output?signature=get-secret"


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _grant():
    return require_paid_resource_admission(
        build_paid_lane_admission(resource_class="gpu_render"),
        resource_class="gpu_render",
        expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION,
    )


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _bound(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _inputs(tmp_path: Path) -> SimpleNamespace:
    bundle = tmp_path / "semantic_teacher_bundle.zip"
    registry_entry = {
        "backend_id": "fixture-hosted-editor",
        "capability": "semantic_teacher_image_edit",
        "execution": {
            "adapter_id": "openai_images_edits_v1",
            "model_snapshot": "fixture-snapshot",
            "runtime_image_identity": IMAGE,
        },
    }
    backend_digest = canonical_digest(registry_entry)
    runtime_request = {
        "schema_version": "semantic_teacher_image_edit_runtime_request.v1",
        "source_commit_sha": SOURCE_COMMIT,
        "source_packet_digest": "sha256:" + "6" * 64,
        "backend": {
            "registry_entry": registry_entry,
            "backend_entry_digest": backend_digest,
            "execution": {
                "adapter_id": "openai_images_edits_v1",
                "model_snapshot": "fixture-snapshot",
                "runtime_image_identity": IMAGE,
            },
        },
        "prompt_policy": {"version": "fixture"},
        "prompt": "remove the selected object",
        "tasks": [
            {
                "task_id": "task_a",
                "frames": [
                    {
                        "frame_index": 0,
                        "camera_id": "front",
                        "input_rgb": {},
                        "edit_mask": {},
                    }
                ],
            }
        ],
        "retry_count": 0,
        "request_digest": "",
    }
    runtime_request["request_digest"] = canonical_digest(
        runtime_request, digest_field="request_digest"
    )
    manifest = {
        "schema_version": "semantic_teacher_image_edit_provider_manifest.v1",
        "classification": "private_derived_semantic_teacher_image_edit",
        "source_commit_sha": SOURCE_COMMIT,
        "runtime_request_digest": runtime_request["request_digest"],
        "backend_entry_digest": backend_digest,
        "task_count": 1,
        "camera_count": 1,
        "automatic_retry_count": 0,
    }
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "provider_runtime/semantic_teacher_image_edit_runtime_request.v1.json",
            json.dumps(runtime_request, sort_keys=True),
        )
        archive.writestr(
            "provider_runtime/semantic_teacher_image_edit_provider_manifest.v1.json",
            json.dumps(manifest, sort_keys=True),
        )
    receipt_path = tmp_path / "bundle_receipt.json"
    receipt = {
        "schema_version": "semantic_teacher_image_edit_provider_bundle.v1",
        "status": "completed_no_upload_no_inference",
        "source_commit_sha": SOURCE_COMMIT,
        "bundle": _bound(bundle),
        "manifest_digest": "sha256:" + "1" * 64,
        "runtime_request_digest": runtime_request["request_digest"],
        "backend_entry_digest": backend_digest,
        "worker_image_digest": IMAGE,
        "runtime_image_identity": IMAGE,
        "worker_source_sha256": WORKER_CODE_DIGEST,
        "model_snapshot": "fixture-snapshot",
        "adapter_id": "openai_images_edits_v1",
        "pricing_binding_digest": "sha256:" + "7" * 64,
        "maximum_cost_per_request_usd": 0.2,
        "task_count": 1,
        "camera_count": 1,
        "rehearsal": {
            "status": "passed",
            "token_lookup_performed": False,
            "upload_performed": False,
            "provider_mutations_performed": 0,
        },
        "provider_mutations_performed": 0,
        "secret_values_stored": False,
        "raw_nonredistributable_source_bytes_included": False,
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    _write_json(receipt_path, receipt)
    authority_path = tmp_path / "authority.json"
    authority = {
        "schema_version": "semantic_teacher_image_edit_paid_authority.v1",
        "authority_kind": "explicit_user_direction_in_current_goal",
        "authority_reference": "fixture authority",
        "authorized_by": "fixture human",
        "authorized_on": "2026-08-13",
        "purpose": "one_shot_semantic_teacher_image_edit",
        "paid_execution_authorized": True,
        "source_commit_sha": SOURCE_COMMIT,
        "worker_image_digest": IMAGE,
        "worker_container_image_digest": IMAGE,
        "runtime_image_identity": IMAGE,
        "worker_source_sha256": WORKER_CODE_DIGEST,
        "model_snapshot": "fixture-snapshot",
        "adapter_id": "openai_images_edits_v1",
        "bundle": _bound(bundle),
        "bundle_receipt": _bound(receipt_path),
        "bundle_receipt_digest": receipt["receipt_digest"],
        "runtime_request_digest": receipt["runtime_request_digest"],
        "backend_entry_digest": receipt["backend_entry_digest"],
        "task_count": 1,
        "camera_count": 1,
        "maximum_paid_attempts": 1,
        "maximum_provider_allocations": 1,
        "maximum_automatic_retries": 0,
        "automatic_paid_retry_authorized": False,
        "maximum_hourly_rate_usd": 0.5,
        "hard_total_spend_cap_usd": 0.5,
        "hard_ttl_seconds": 60,
        "vast_spend_upper_bound_usd": 0.5 * 60 / 3600,
        "hosted_editor_spend_upper_bound_usd": 0.2,
        "maximum_cost_per_request_usd": 0.2,
        "maximum_editor_request_cost_usd": 0.2,
        "maximum_compute_cost_usd": 0.5 * 60 / 3600,
        "pricing_binding_digest": "sha256:" + "7" * 64,
        "aggregate_goal_spend_before_attempt_usd": 0.0,
        "aggregate_goal_spend_cap_usd": 10.0,
        "prior_spend_reconciliation": None,
        "consumption_root_kind": "host_private_atomic_single_use",
        "raw_nonredistributable_bytes_upload_authorized": False,
        "canonical_interiorgs_mutation_authorized": False,
        "authorization_digest": "",
    }
    authority["authorization_digest"] = canonical_digest(
        authority, digest_field="authorization_digest"
    )
    _write_json(authority_path, authority)
    token_path = tmp_path / "openai_token"
    token_path.write_text(TOKEN)
    token_path.chmod(0o600)
    return SimpleNamespace(
        semantic_teacher_attempt_authority=str(authority_path),
        semantic_teacher_bundle=str(bundle),
        semantic_teacher_bundle_receipt=str(receipt_path),
        semantic_teacher_job_dir=str(tmp_path / "job"),
        semantic_teacher_token_file=str(token_path),
        semantic_teacher_runtime_image_identity=IMAGE,
    )


def _preflight(tmp_path: Path) -> dict:
    return {
        "provider": "vast",
        "watchdog": {
            "status": "armed",
            "independent_process": True,
            "watchdog_pid": os.getpid(),
            "watchdog_started_epoch": 900,
            "watchdog_deadline_epoch": 10_000,
            "pod_name_prefix": "blueprint-semantic-teacher-",
            "started_instance_id_path": str(
                (tmp_path / "watchdog" / "started_instance_id").resolve()
            ),
        },
        "gpu_memory_bytes": 16 * 1024**3,
        "container_disk_bytes": 32 * 1024**3,
        "on_demand_price_usd_per_hour": 0.25,
    }


def _runtime_archive(
    *,
    status: str = "completed",
    runtime_request_digest: str = "",
    backend_entry_digest: str = "",
) -> bytes:
    frame = b"fixture-png-bytes"
    if status == "completed":
        runtime = {
            "schema_version": RUNTIME_RESULT_SCHEMA_VERSION,
            "status": "completed_unreviewed_semantic_teacher_candidates",
            "source_runtime_request_digest": runtime_request_digest,
            "backend_id": "fixture-hosted-editor",
            "backend_entry_digest": backend_entry_digest,
            "adapter_id": "openai_images_edits_v1",
            "model_snapshot": "fixture-snapshot",
            "task_count": 1,
            "request_count": 1,
            "attempted_request_count": 1,
            "successful_request_count": 1,
            "retry_count": 0,
            "tasks": [
                {
                    "task_id": "task_a",
                    "camera_count": 1,
                    "frames": [
                        {
                            "frame_index": 0,
                            "camera_id": "front",
                            "source_rgb_sha256": "sha256:" + "4" * 64,
                            "edit_mask_sha256": "sha256:" + "5" * 64,
                            "semantic_teacher_frame": {
                                "relative_path": "tasks/task_a/00000.png",
                                "size_bytes": len(frame),
                                "sha256": "sha256:"
                                + hashlib.sha256(frame).hexdigest(),
                            },
                            "visual_reviewed": False,
                            "multiview_consistency_qualified": False,
                        }
                    ],
                }
            ],
            "raw_secret_values_recorded": False,
            "canonical_source_altered": False,
            "simulator_or_policy_output_is_physical_evidence": False,
            "appearance_qualified": False,
            "result_digest": "",
        }
    else:
        runtime = {
            "schema_version": RUNTIME_RESULT_SCHEMA_VERSION,
            "status": "blocked",
            "blockers": ["fixture_worker_failure"],
            "retry_count": 0,
            "raw_secret_values_recorded": False,
            "canonical_source_altered": False,
            "appearance_qualified": False,
            "result_digest": "",
        }
    runtime["result_digest"] = canonical_digest(runtime, digest_field="result_digest")
    payload = BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            f"{RUNTIME_RESULT_SCHEMA_VERSION}.json",
            json.dumps(runtime, sort_keys=True),
        )
        archive.writestr("runtime_stdout.log", "worker stdout\n")
        archive.writestr("runtime_stderr.log", "worker stderr\n")
        if status == "completed":
            archive.writestr("tasks/task_a/00000.png", frame)
    return payload.getvalue()


class _Provider:
    name = "vast"

    def __init__(
        self,
        *,
        initially_live: bool = False,
        launch_status: str = "launched",
        inspect_absent: bool = False,
        launch_result: dict | None = None,
    ):
        self.initially_live = initially_live
        self.launch_status = launch_status
        self.live = False
        self.launch_calls = 0
        self.terminate_calls = 0
        self.seen_token = ""
        self.inspect_absent = inspect_absent
        self.inspect_calls = 0
        self.launch_result = launch_result
        self.seen_spec = None

    def billable_inventory(self, *, name_prefix: str) -> dict:
        live = self.initially_live or self.live
        return {
            "api_confirmed": True,
            "live_resource_count": 1 if live else 0,
            "resources": [{"id": "existing"}] if live else [],
        }

    def build_request(self, spec, job_dir):
        assert spec.name.startswith("blueprint-semantic-teacher-")
        assert spec.image == IMAGE
        assert spec.env["BLUEPRINT_IMAGE_EDITOR_TOKEN"] == TOKEN
        assert spec.env["BLUEPRINT_SEMANTIC_TEACHER_INPUT_BUNDLE_GET_URL"] == INPUT_URL
        self.seen_spec = spec
        self.seen_token = spec.env["BLUEPRINT_IMAGE_EDITOR_TOKEN"]
        return {"create_payload": {"env_keys": sorted(spec.env)}}

    def launch(self, job_dir, request, **kwargs):
        self.launch_calls += 1
        assert request["maximum_create_attempts"] == 1
        assert request["prelaunch_spend_guard"]["maximum_create_attempts"] == 1
        if self.launch_result is not None:
            return dict(self.launch_result)
        if self.launch_status != "launched":
            return {
                "status": self.launch_status,
                "maximum_create_attempts": 1,
                "create_attempt_count": 1,
                "allocation_created": False,
            }
        self.live = True
        return {
            "status": "launched",
            "instance_id": "42",
            "estimated_cost_usd": 0.01,
            "maximum_create_attempts": 1,
            "create_attempt_count": 1,
        }

    def terminate(self, instance_id):
        self.terminate_calls += 1
        self.live = False
        return {"status": "stopped", "instance_id": instance_id}

    def inspect(self, instance_id):
        self.inspect_calls += 1
        if self.inspect_absent:
            return {
                "status": "absent",
                "instance_id": instance_id,
                "api_confirmed": True,
                "provider_absence_confirmed": True,
                "blockers": [],
            }
        return {
            "status": "observed",
            "instance_id": instance_id,
            "api_confirmed": True,
            "provider_absence_confirmed": False,
            "blockers": [],
        }


class _ObjectStore:
    def __init__(self):
        self.stage_calls = 0
        self.cleanup_calls = 0

    def stage(self, *, job_dir, **kwargs):
        self.stage_calls += 1
        root = Path(job_dir)
        root.mkdir(parents=True)
        for name, value in (
            ("provider_bundle_url.txt", INPUT_URL),
            ("provider_output_put_url.txt", PUT_URL),
            ("provider_output_get_url.txt", GET_URL),
        ):
            path = root / name
            path.write_text(value)
            path.chmod(0o600)
        _write_json(
            root / "wam_provider_object_store_staging_manifest.json",
            {"status": "completed", "raw_secret_values_recorded": False},
        )
        return {"status": "completed", "blockers": []}

    def cleanup(self, job_dir):
        self.cleanup_calls += 1
        root = Path(job_dir)
        for name in (
            "provider_bundle_url.txt",
            "provider_output_put_url.txt",
            "provider_output_get_url.txt",
        ):
            (root / name).unlink(missing_ok=True)
        result = {
            "schema_version": "wam_provider_object_store_cleanup.v1",
            "status": "completed",
            "all_objects_absent": True,
            "signed_url_files_removed": True,
            "blockers": [],
            "raw_secret_values_recorded": False,
        }
        _write_json(root / "wam_provider_object_store_cleanup.json", result)
        return result


def _consume(monkeypatch: pytest.MonkeyPatch, calls: list[dict]) -> None:
    def consume(authority, *, source_commit_sha):
        calls.append(dict(authority))
        return {
            "status": "consumed",
            "authorization_digest": authority["authorization_digest"],
            "consumption_record_sha256": "sha256:" + "9" * 64,
            "record_location_disclosed": False,
        }

    monkeypatch.setattr(
        vast, "consume_semantic_teacher_image_edit_paid_authority_once", consume
    )


def _run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    provider: _Provider | None = None,
    archive_status: str = "completed",
    watchdog_status: str = "provider_terminal",
    store_override: _ObjectStore | None = None,
    result_fetcher=None,
    watchdog_closer=None,
):
    args = _inputs(tmp_path)
    receipt = json.loads(Path(args.semantic_teacher_bundle_receipt).read_text())
    selected_provider = provider or _Provider()
    store = store_override or _ObjectStore()
    consumption_calls: list[dict] = []
    _consume(monkeypatch, consumption_calls)
    ticks = iter(float(value) for value in range(1000, 1100))

    def bind(path: Path, instance_id: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(instance_id))

    def default_closer(**kwargs):
        return {
            "schema_version": "vast_independent_watchdog_handoff.v1",
            "status": watchdog_status,
            "instance_ids": kwargs["instance_ids"],
            "watchdog_armed_before_allocation": True,
            "provider_absence_confirmed": True,
            "raw_secret_values_recorded": False,
        }
    result = run_semantic_teacher_image_edit_vast(
        args,
        checkout_commit=SOURCE_COMMIT,
        preflight=_preflight(tmp_path),
        provider=selected_provider,
        paid_resource_admission_grant=_grant(),
        watchdog_closer=watchdog_closer or default_closer,
        object_store_stager=store.stage,
        object_store_cleaner=store.cleanup,
        result_fetcher=result_fetcher
        or (
            lambda _url: _runtime_archive(
                status=archive_status,
                runtime_request_digest=receipt["runtime_request_digest"],
                backend_entry_digest=receipt["backend_entry_digest"],
            )
        ),
        sleeper=lambda _seconds: None,
        clock=lambda: next(ticks),
        watchdog_validator=lambda _watchdog, _now, _ttl: True,
        watchdog_instance_binder=bind,
    )
    return result, selected_provider, store, consumption_calls


def test_probe_kind_and_bootstrap_are_single_attempt_and_ephemeral_secret() -> None:
    assert PROBE_KIND == "semantic-teacher-image-edit"
    script = _bootstrap_script()
    assert script.count("run_semantic_teacher_image_edit.sh") == 1
    assert 'secret_file="$secret_root/image_editor_token"' in script
    assert "unset BLUEPRINT_IMAGE_EDITOR_TOKEN" in script
    assert 'rm -f "$secret_file"' in script
    assert "runtime_stdout.log" in script
    assert "runtime_stderr.log" in script
    assert "semantic_teacher_image_edit_runtime_output.zip" in script
    assert "retry" not in script.lower()


def test_bootstrap_creates_the_bundle_parent_before_download() -> None:
    """The pinned runtime image does not guarantee that /work exists.

    A production container reached the first ``Path.write_bytes`` with no
    parent directory, failed before any OpenAI request, and then remained live
    in the provider's log-hold wrapper. The bootstrap must own its writable
    root instead of relying on image-specific filesystem state.
    """

    script = _bootstrap_script()
    create_parent = 'mkdir -p "$(dirname "$bundle_path")"'
    first_download = 'python - "$bundle_path"'
    assert script.count(create_parent) == 1
    assert script.index(create_parent) < script.index(first_download)


def test_watchdog_validator_accepts_realistic_full_ttl_handoff_delay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A full-TTL watchdog remains valid after ordinary preflight latency."""

    monkeypatch.setattr("os.kill", lambda pid, signal: None)
    assert vast._watchdog_valid(
        {
            "status": "armed",
            "independent_process": True,
            "watchdog_pid": 123,
            "watchdog_started_epoch": 1_000,
            "watchdog_deadline_epoch": 2_800,
            "pod_name_prefix": "blueprint-semantic-teacher-bound-run-",
        },
        now_epoch=1_000.9,
        hard_ttl_seconds=1_800,
    )


def test_watchdog_validator_accepts_control_plane_evidence_field_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("os.kill", lambda pid, signal: None)
    assert vast._watchdog_valid(
        {
            "status": "armed",
            "independent_process": True,
            "pid": 123,
            "started_epoch": 1_000,
            "deadline_epoch": 1_060,
            "pod_name_prefix": "blueprint-semantic-teacher-bound-run-",
        },
        now_epoch=1_005,
        hard_ttl_seconds=60,
    )


def test_watchdog_validator_rejects_expired_or_short_armed_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("os.kill", lambda pid, signal: None)
    base = {
        "status": "armed",
        "independent_process": True,
        "watchdog_pid": 123,
        "watchdog_started_epoch": 1_000,
        "watchdog_deadline_epoch": 1_060,
        "pod_name_prefix": "blueprint-semantic-teacher-bound-run-",
    }
    assert not vast._watchdog_valid(base, now_epoch=1_060, hard_ttl_seconds=60)
    assert not vast._watchdog_valid(
        {**base, "watchdog_deadline_epoch": 1_059},
        now_epoch=1_005,
        hard_ttl_seconds=60,
    )


def test_one_instance_run_retains_output_and_proves_every_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, provider, store, consumption_calls = _run(tmp_path, monkeypatch)
    assert result["status"] == "completed"
    assert result["allocation_count"] == 1
    assert result["automatic_retry_count"] == 0
    assert result["retry_cap"] == 0
    assert result["provider_mutations_performed"] == 2
    assert result["provider_zero_verified"] is True
    assert result["all_staged_objects_absent"] is True
    assert result["continuing_spend_from_this_run"] is False
    assert result["runtime_image_identity"] == IMAGE
    assert Path(result["artifact_manifest_path"]).is_file()
    assert Path(result["teardown_manifest_path"]).is_file()
    teardown_manifest = json.loads(Path(result["teardown_manifest_path"]).read_text())
    assert teardown_manifest["generated_at"].endswith("+00:00")
    assert provider.launch_calls == 1
    assert provider.terminate_calls == 1
    assert provider.seen_spec.max_hourly_rate_usd == pytest.approx(0.5)
    assert provider.seen_spec.min_gpu_ram_mb == 16_000
    assert "us" in provider.seen_spec.allowed_geolocation_country_codes
    assert "cn" not in provider.seen_spec.allowed_geolocation_country_codes
    assert store.stage_calls == 1
    assert store.cleanup_calls == 1
    assert len(consumption_calls) == 1
    job = Path(_inputs(tmp_path).semantic_teacher_job_dir)
    assert (job / "runtime_output/runtime_stdout.log").is_file()
    assert (job / "runtime_output/runtime_stderr.log").is_file()
    assert (job / "runtime_output/tasks/task_a/00000.png").is_file()
    assert (job / "semantic_teacher_image_edit_result_import.v1.json").is_file()
    billing = json.loads((job / "billing_receipt.json").read_text())
    assert billing["editor_request_cost_usd"] == 0.2
    assert billing["compute_cost_usd"] == pytest.approx(0.5 * 60 / 3600)
    assert billing["cost_usd"] == pytest.approx(0.2 + 0.5 * 60 / 3600)
    assert billing["ledger_basis"] == (
        "conservative_upper_bound_for_missing_actual_component"
    )
    persisted = b"\n".join(
        path.read_bytes() for path in job.rglob("*") if path.is_file()
    )
    for secret in (TOKEN, INPUT_URL, PUT_URL, GET_URL):
        assert secret.encode() not in persisted


def test_global_nonzero_refuses_before_consumption_staging_or_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _inputs(tmp_path)
    provider = _Provider(initially_live=True)
    store = _ObjectStore()
    consumption_calls: list[dict] = []
    _consume(monkeypatch, consumption_calls)
    with pytest.raises(
        SemanticTeacherImageEditVastError, match="provider_not_zero_before_launch"
    ):
        run_semantic_teacher_image_edit_vast(
            args,
            checkout_commit=SOURCE_COMMIT,
            preflight=_preflight(tmp_path),
            provider=provider,
            paid_resource_admission_grant=_grant(),
            watchdog_closer=lambda **_kwargs: {},
            object_store_stager=store.stage,
            object_store_cleaner=store.cleanup,
            clock=lambda: 1000.0,
        )
    assert consumption_calls == []
    assert store.stage_calls == 0
    assert provider.launch_calls == 0


def test_reopened_bundle_rejects_changed_runtime_request_order(tmp_path: Path) -> None:
    args = _inputs(tmp_path)
    receipt = json.loads(Path(args.semantic_teacher_bundle_receipt).read_text())
    authority = json.loads(Path(args.semantic_teacher_attempt_authority).read_text())
    bundle = Path(args.semantic_teacher_bundle)
    with zipfile.ZipFile(bundle) as archive:
        request = json.loads(
            archive.read(
                "provider_runtime/semantic_teacher_image_edit_runtime_request.v1.json"
            )
        )
        manifest = archive.read(
            "provider_runtime/semantic_teacher_image_edit_provider_manifest.v1.json"
        )
    request["tasks"][0]["frames"][0]["camera_id"] = "changed-camera"
    request["request_digest"] = canonical_digest(
        request, digest_field="request_digest"
    )
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "provider_runtime/semantic_teacher_image_edit_runtime_request.v1.json",
            json.dumps(request, sort_keys=True),
        )
        archive.writestr(
            "provider_runtime/semantic_teacher_image_edit_provider_manifest.v1.json",
            manifest,
        )
    with pytest.raises(
        SemanticTeacherImageEditVastError, match="runtime_binding_invalid"
    ):
        _validate_bundle_runtime_bindings(
            bundle,
            receipt=receipt,
            authority=authority,
            checkout_source_commit=SOURCE_COMMIT,
            runtime_image_identity=IMAGE,
        )


def test_worker_image_must_be_digest_bound_by_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _inputs(tmp_path)
    args.semantic_teacher_runtime_image_identity = (
        "registry.example/blueprint/semantic-teacher@sha256:" + "c" * 64
    )
    consumption_calls: list[dict] = []
    _consume(monkeypatch, consumption_calls)
    with pytest.raises(
        SemanticTeacherImageEditVastError, match="execution_bounds_invalid"
    ):
        run_semantic_teacher_image_edit_vast(
            args,
            checkout_commit=SOURCE_COMMIT,
            preflight=_preflight(tmp_path),
            provider=_Provider(),
            paid_resource_admission_grant=_grant(),
            watchdog_closer=lambda **_kwargs: {},
            clock=lambda: 1000.0,
        )
    assert consumption_calls == []


def test_failed_worker_archive_is_preserved_without_paid_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, provider, store, _calls = _run(
        tmp_path, monkeypatch, archive_status="blocked"
    )
    assert result["status"] == "blocked"
    assert result["automatic_retry_count"] == 0
    assert result["retry_cap"] == 0
    assert result["allocation_count"] == 1
    assert result["provider_zero_verified"] is True
    assert provider.launch_calls == 1
    assert provider.terminate_calls == 1
    assert store.cleanup_calls == 1
    job = Path(_inputs(tmp_path).semantic_teacher_job_dir)
    assert (job / "semantic_teacher_image_edit_runtime_output.zip").is_file()
    assert (job / f"runtime_output/{RUNTIME_RESULT_SCHEMA_VERSION}.json").is_file()
    assert (job / "runtime_output/runtime_stdout.log").is_file()
    assert (job / "runtime_output/runtime_stderr.log").is_file()


def test_confirmed_no_allocation_closes_zero_without_relaunch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = _Provider(launch_status="blocked")
    result, provider, store, _calls = _run(
        tmp_path,
        monkeypatch,
        provider=provider,
        watchdog_status="cancelled_no_allocation",
    )
    assert result["status"] == "blocked"
    assert result["allocation_count"] == 0
    assert result["automatic_retry_count"] == 0
    assert result["provider_mutations_performed"] == 0
    assert Path(result["artifact_manifest_path"]).is_file()
    assert Path(result["teardown_manifest_path"]).is_file()
    assert result["provider_zero_verified"] is True
    assert result["continuing_spend_from_this_run"] is False
    assert provider.launch_calls == 1
    assert provider.terminate_calls == 0
    assert store.cleanup_calls == 1
    job = Path(_inputs(tmp_path).semantic_teacher_job_dir)
    provider_zero = json.loads(
        (job / "no_allocation_provider_zero_receipt.json").read_text()
    )
    assert provider_zero["provider_zero_api_confirmed"] is True
    assert provider_zero["scoped_inventory"]["live_resource_count"] == 0
    assert provider_zero["global_inventory"]["live_resource_count"] == 0


def test_no_matching_offer_preserves_zero_attempt_truth_and_provider_blocker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = _Provider(
        launch_result={
            "status": "blocked",
            "blockers": ["no_vast_offer_matching_rate_and_gpu_memory"],
            "maximum_create_attempts": 1,
            "create_attempt_count": 0,
            "allocation_created": False,
        }
    )
    result, provider, store, _calls = _run(
        tmp_path,
        monkeypatch,
        provider=provider,
        watchdog_status="cancelled_no_allocation",
    )

    assert result["status"] == "blocked"
    assert result["allocation_count"] == 0
    assert result["create_attempt_count"] == 0
    assert result["provider_mutations_performed"] == 0
    assert result["provider_mutation_outcome_ambiguous"] is False
    assert "no_vast_offer_matching_rate_and_gpu_memory" in result["blockers"]
    assert "semantic_teacher_vast_instance_not_created" in result["blockers"]
    assert "semantic_teacher_vast_create_attempt_contract_invalid" not in result[
        "blockers"
    ]
    assert "semantic_teacher_independent_watchdog_not_closed" not in result[
        "blockers"
    ]
    assert result["provider_zero_verified"] is True
    assert result["continuing_spend_from_this_run"] is False
    assert provider.launch_calls == 1
    assert provider.terminate_calls == 0
    assert store.cleanup_calls == 1


def test_zero_create_attempt_with_contradictory_allocation_is_ambiguous_and_redacted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret_provider_detail = "secret=https://provider.invalid/?token=do-not-retain"
    provider = _Provider(
        launch_result={
            "status": "blocked",
            "blockers": [
                "no_vast_offer_matching_rate_and_gpu_memory",
                secret_provider_detail,
            ],
            "maximum_create_attempts": 1,
            "create_attempt_count": 0,
            "allocation_created": True,
        }
    )
    result, provider, _store, _calls = _run(
        tmp_path,
        monkeypatch,
        provider=provider,
        watchdog_status="cancelled_no_allocation",
    )

    assert result["status"] == "blocked"
    assert result["provider_mutation_outcome_ambiguous"] is True
    assert result["provider_mutations_performed"] == 1
    assert "semantic_teacher_vast_create_attempt_contract_invalid" in result[
        "blockers"
    ]
    assert "semantic_teacher_vast_create_outcome_ambiguous" in result["blockers"]
    assert "semantic_teacher_independent_watchdog_not_closed" in result["blockers"]
    assert "no_vast_offer_matching_rate_and_gpu_memory" in result["blockers"]
    assert secret_provider_detail not in result["blockers"]
    persisted = b"\n".join(
        path.read_bytes()
        for path in Path(_inputs(tmp_path).semantic_teacher_job_dir).rglob("*")
        if path.is_file()
    )
    assert secret_provider_detail.encode() not in persisted
    assert provider.terminate_calls == 0


def test_authority_consumption_block_closes_watchdog_and_double_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _inputs(tmp_path)
    provider = _Provider()
    store = _ObjectStore()
    closer_calls: list[dict] = []

    monkeypatch.setattr(
        vast,
        "consume_semantic_teacher_image_edit_paid_authority_once",
        lambda *_args, **_kwargs: {
            "status": "blocked",
            "blockers": ["semantic_teacher_authority_already_consumed"],
        },
    )

    def close(**kwargs):
        closer_calls.append(dict(kwargs))
        return {
            "schema_version": "vast_independent_watchdog_handoff.v1",
            "status": "cancelled_no_allocation",
            "instance_ids": [],
            "watchdog_armed_before_allocation": True,
            "provider_absence_confirmed": True,
            "raw_secret_values_recorded": False,
        }

    result = run_semantic_teacher_image_edit_vast(
        args,
        checkout_commit=SOURCE_COMMIT,
        preflight=_preflight(tmp_path),
        provider=provider,
        paid_resource_admission_grant=_grant(),
        watchdog_closer=close,
        object_store_stager=store.stage,
        object_store_cleaner=store.cleanup,
        clock=lambda: 1000.0,
        watchdog_validator=lambda _watchdog, _now, _ttl: True,
    )
    assert result["status"] == "blocked"
    assert result["provider_zero_verified"] is True
    assert result["continuing_spend_from_this_run"] is False
    assert result["provider_mutations_performed"] == 0
    assert provider.launch_calls == 0
    assert store.stage_calls == 0
    assert closer_calls == [
        {
            "instance_ids": [],
            "provider_teardown_completed": True,
            "provider_allocation_impossible": True,
        }
    ]
    job = Path(args.semantic_teacher_job_dir)
    assert (job / "no_allocation_provider_zero_receipt.json").is_file()
    assert json.loads((job / "independent_watchdog.json").read_text())["status"] == (
        "cancelled_no_allocation"
    )


def test_allocated_timeout_retains_gap_conservative_billing_and_canonical_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def missing(_url: str) -> bytes:
        raise FileNotFoundError("not uploaded")

    result, provider, store, _calls = _run(
        tmp_path,
        monkeypatch,
        result_fetcher=missing,
    )
    assert result["status"] == "blocked"
    assert result["allocation_count"] == 1
    assert result["automatic_retry_count"] == 0
    assert result["provider_zero_verified"] is True
    assert result["continuing_spend_from_this_run"] is False
    assert result["runtime_result_digest"] is None
    assert result["runtime_media_gap_digest"].startswith("sha256:")
    assert Path(result["artifact_manifest_path"]).is_file()
    assert Path(result["teardown_manifest_path"]).is_file()
    assert provider.launch_calls == 1
    assert provider.terminate_calls == 1
    assert store.cleanup_calls == 1

    job = Path(_inputs(tmp_path).semantic_teacher_job_dir)
    gap = json.loads(
        (
            job
            / "runtime_output"
            / "semantic_teacher_image_edit_runtime_media_gap.v1.json"
        ).read_text()
    )
    assert gap["gap_type"] == "runtime_timeout"
    assert gap["attempted_request_count_known"] is False
    assert "attempted_request_count" not in gap
    assert gap["attempted_request_count_upper_bound"] == 1
    assert gap["partial_png_inventory"] == []
    billing = json.loads((job / "billing_receipt.json").read_text())
    assert billing["status"] == "conservative_upper_bound_runtime_result_missing"
    assert billing["attempted_request_count_known"] is False
    assert "attempted_request_count" not in billing
    assert billing["attempted_request_count_upper_bound"] == 1
    assert billing["editor_request_cost_usd"] == 0.2
    assert billing["editor_request_cost_basis"] == (
        "full_authorized_upper_bound_due_to_unknown_attempt_count"
    )
    provider_zero = json.loads((job / "provider_zero_receipt.json").read_text())
    assert provider_zero["terminal_evidence_kind"] == "runtime_media_gap"
    assert provider_zero["runtime_media_gap_digest"] == gap["gap_digest"]
    assert provider_zero["continuing_spend_from_this_run"] is False
    assert (
        job / "semantic_teacher_image_edit_runtime_failure_artifacts.zip"
    ).is_file()


def test_provider_confirmed_absence_ends_output_wait_without_full_ttl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(vast, "PROVIDER_LIVENESS_POLL_SECONDS", 0.0)
    fetch_calls = 0

    def missing(_url: str) -> bytes:
        nonlocal fetch_calls
        fetch_calls += 1
        raise FileNotFoundError("not uploaded")

    provider = _Provider(inspect_absent=True)
    result, _provider, _store, _calls = _run(
        tmp_path,
        monkeypatch,
        provider=provider,
        result_fetcher=missing,
    )

    assert result["status"] == "blocked"
    assert result["blockers"] == ["semantic_teacher_provider_instance_vanished"]
    assert result["excluded_machine_ids"] == []
    assert provider.inspect_calls == vast.PROVIDER_ABSENCE_CONFIRMATIONS_REQUIRED
    assert fetch_calls == vast.PROVIDER_ABSENCE_CONFIRMATIONS_REQUIRED
    gap = json.loads(
        (
            Path(_inputs(tmp_path).semantic_teacher_job_dir)
            / "runtime_output"
            / "semantic_teacher_image_edit_runtime_media_gap.v1.json"
        ).read_text()
    )
    assert gap["gap_type"] == "provider_instance_vanished"


def test_default_result_fetcher_treats_object_store_404_as_not_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(*_args, **_kwargs):
        raise urllib.error.HTTPError(GET_URL, 404, "Not Found", {}, None)

    monkeypatch.setattr(
        "blueprint_pipeline.semantic_teacher_image_edit_vast.safe_http_request",
        missing,
    )

    with pytest.raises(FileNotFoundError, match="semantic_teacher_output_not_ready"):
        _default_result_fetcher(GET_URL)
