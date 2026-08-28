from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

from blueprint_pipeline import task_evaluation_scene_configuration_vast as scene_vast
from blueprint_pipeline.task_evaluation_scene_configuration_bundle import (
    BUNDLE_SCHEMA_VERSION,
)
from blueprint_pipeline.task_evaluation_scene_configuration_runtime_budget import (
    MAX_ATTEMPT_SPEND_USD,
    MAX_HOURLY_RATE_USD,
    MAX_PROVIDER_COMPUTE_SPEND_USD,
    REQUIRED_PARENT_TTL_SECONDS,
)
from tests.test_task_evaluation_scene_configuration_bundle import (
    _build,
    _construction_queue,
)


def test_postcreate_adapter_exception_preserves_instance_identity_and_watchdog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A finalization error cannot turn a rented instance into "unallocated"."""

    receipt = _build(tmp_path, "bundle")
    receipt_path = tmp_path / "bundle" / f"{BUNDLE_SCHEMA_VERSION}.receipt.json"
    authority_path = tmp_path / "authority.json"
    authority_path.write_text("{}", encoding="utf-8")
    authority = {
        "authority_digest": "sha256:" + "e" * 64,
        "hard_attempt_spend_cap_usd": MAX_ATTEMPT_SPEND_USD,
        "provider_compute_spend_cap_usd": MAX_PROVIDER_COMPUTE_SPEND_USD,
        "maximum_hourly_rate_usd": MAX_HOURLY_RATE_USD,
        "maximum_single_resource_ttl_seconds": REQUIRED_PARENT_TTL_SECONDS,
        "container_image": "nvcr.io/nvidia/isaac-sim@sha256:" + "b" * 64,
        "external_service_spend_caps": {
            "openai": {"maximum_cost_usd": 1.5, "maximum_requests": 32}
        },
    }
    monkeypatch.setattr(
        scene_vast,
        "validate_scene_configuration_paid_authority",
        lambda _value, **_kwargs: authority,
    )
    monkeypatch.setattr(
        scene_vast, "_provider_runtime_inputs", lambda _authority: ({}, {})
    )
    monkeypatch.setattr(
        scene_vast, "require_paid_resource_admission_grant", lambda *_a, **_k: None
    )

    def stage(*, job_dir, **_kwargs):
        staging = Path(job_dir)
        staging.mkdir(parents=True, exist_ok=True)
        for name in (
            "provider_bundle_url.txt",
            "provider_output_put_url.txt",
            "provider_output_get_url.txt",
        ):
            (staging / name).write_text(
                f"https://objects.example.test/{name}", encoding="utf-8"
            )
        bundle = Path(_kwargs["bundle_path"])
        return {
            "status": "completed",
            "provider_bundle_remote_reference": {
                "schema_version": "task_evaluation_scene_artifact_reference.v1",
                "status": "remote_verified",
                "artifact_kind": "provider-bundle",
                "uri": "s3://scene-artifacts/provider-bundle.zip",
                "digest": receipt["bundle_sha256"],
                "size_bytes": bundle.stat().st_size,
                "content_addressed_key": True,
                "remote_identity_verified": True,
                "full_byte_service_account_readback_passed": True,
                "raw_secret_values_recorded": False,
            },
        }

    monkeypatch.setattr(
        scene_vast, "stage_wam_provider_bundle_object_store", stage
    )
    watchdog_handle: types.SimpleNamespace | None = None

    def arm(**kwargs):
        nonlocal watchdog_handle
        watchdog_handle = types.SimpleNamespace(
            pod_name_prefix=kwargs["pod_name_prefix"] + "fixture-",
            started_instance_id_path=(
                Path(kwargs["job_dir"]) / "started_vast_instance_id.txt"
            ),
            deadline_epoch=9_999_999_999.0,
        )
        return {"status": "armed"}, watchdog_handle

    monkeypatch.setattr(scene_vast, "arm_independent_vast_watchdog", arm)
    close_call: dict[str, object] = {}

    def close(**kwargs):
        close_call.update(kwargs)
        return {"status": "retained_until_hard_ttl"}

    monkeypatch.setattr(scene_vast, "close_independent_vast_watchdog", close)
    monkeypatch.setattr(
        scene_vast,
        "cleanup_staged_wam_provider_objects",
        lambda _root: {"all_objects_absent": True},
    )
    monkeypatch.setattr(
        scene_vast,
        "_consume_authority_once",
        lambda _authority, **_kwargs: {"status": "consumed"},
    )
    monkeypatch.setattr(
        scene_vast,
        "_stage_owner_only_runtime_secrets",
        lambda **_kwargs: ({}, None),
    )

    def adapter(**kwargs):
        started = Path(kwargs["started_instance_id_path"])
        started.parent.mkdir(parents=True, exist_ok=True)
        started.write_text("918273\n", encoding="utf-8")
        raise RuntimeError("fixture_adapter_finalization_failed")

    monkeypatch.setattr(scene_vast, "run_vast_provider_adapter", adapter)

    def false_unallocated(*_args, **_kwargs):
        raise AssertionError("post-create failure was mislabeled unallocated")

    monkeypatch.setattr(
        scene_vast, "seal_unallocated_provider_teardown", false_unallocated
    )
    _download, upload = scene_vast._provider_transfer_byte_budget(receipt)
    required_free = scene_vast._provider_output_disk_requirements(upload)[
        "required_free_bytes_before_download"
    ]
    job = tmp_path / "job"

    result = scene_vast.run_scene_configuration_vast(
        job_dir=job,
        bundle_receipt_path=receipt_path,
        paid_attempt_authority_path=authority_path,
        paid_resource_admission_grant=object(),
        execute=True,
        scene_construction_queue_root=_construction_queue(tmp_path),
        disk_usage_provider=lambda _path: types.SimpleNamespace(free=required_free),
    )

    assert watchdog_handle is not None
    assert close_call["instance_ids"] == [918273]
    assert close_call["provider_teardown_completed"] is False
    assert close_call["provider_allocation_impossible"] is False
    assert result["status"] == "blocked"
    assert result["provider_mutations_performed"] == 1
    assert result["continuing_spend_from_this_run"] is None
    assert result["teardown_manifest_path"] is None
    assert "provider_zero_not_proven" in result["blockers"]
    assert "independent_watchdog_not_closed" in result["blockers"]
    assert any(
        blocker.startswith("vast_adapter_failed:RuntimeError")
        for blocker in result["blockers"]
    )

    adapter_receipt = json.loads(
        (job / "vast_provider_run/vast_provider_adapter_result.json").read_text(
            encoding="utf-8"
        )
    )
    assert adapter_receipt["status"] == "blocked"
    assert adapter_receipt["vast_instance_ids"] == [918273]
    assert adapter_receipt["provider_create_attempted"] is True
    assert adapter_receipt["vast_side_effects_may_have_occurred"] is True
    assert not (job / "vast_provider_run/vast_teardown_manifest.json").exists()
