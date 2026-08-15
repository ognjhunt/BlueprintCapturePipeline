from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import pytest

import blueprint_pipeline.paid_resource_allocator as allocator
import blueprint_pipeline.public_scene_simready_isaac_vast as runtime
from blueprint_pipeline.common import write_json
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.public_scene_simready_isaac_bundle import DEFAULT_IMAGE


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _bundle(tmp_path: Path, *, commit: str = "a" * 40) -> dict:
    bundle = tmp_path / "bundle.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("provider_runtime/example.txt", "bound")
    return {
        "status": "ready",
        "source_commit_sha": commit,
        "container_image": DEFAULT_IMAGE,
        "retry_cap": 0,
        "blockers": [],
        "probe_spec_sha256": "sha256:" + "b" * 64,
        "bundle_path": str(bundle),
        "bundle_sha256": _sha256(bundle),
    }


def _paid_attempt_authority(
    prepared_bundle: dict,
    *,
    bundle_receipt_sha256: str | None = "sha256:" + "c" * 64,
    hard_cap_usd: float = 3.0,
    max_hourly_rate_usd: float = 1.0,
    hard_ttl_seconds: int = 10_800,
    external_instance_allowlist: list[int] | None = None,
) -> dict:
    authority = {
        "schema_version": runtime.PAID_ATTEMPT_AUTHORITY_SCHEMA,
        "authority_kind": "explicit_user_direction_in_current_goal",
        "authority_reference": "user-message-2026-08-11-gpu-authority",
        "authorized_by": "user",
        "authorized_on": "2026-08-11",
        "purpose": "simready_native_import_probe",
        "provider": "vast",
        "paid_compute_authorized": True,
        "bundle_sha256": prepared_bundle["bundle_sha256"],
        "bundle_receipt_sha256": bundle_receipt_sha256,
        "probe_spec_sha256": prepared_bundle["probe_spec_sha256"],
        "container_image": DEFAULT_IMAGE,
        "maximum_paid_attempts": 1,
        "maximum_automatic_retries": 0,
        "automatic_paid_retry_authorized": False,
        "zero_retry": True,
        "hard_attempt_spend_cap_usd": hard_cap_usd,
        "maximum_hourly_rate_usd": max_hourly_rate_usd,
        "maximum_single_resource_ttl_seconds": hard_ttl_seconds,
        "native_simulator_import_probe_only": True,
        "physical_success_established": False,
        "candidate_policy_queried": False,
        "external_instance_allowlist": external_instance_allowlist or [],
        "authorization_digest": "",
    }
    authority["authorization_digest"] = canonical_digest(
        authority, digest_field="authorization_digest"
    )
    return authority


def _completed_execution() -> dict:
    value = {
        "schema_version": "adp009b_simready_isaac_result.v1",
        "status": "completed",
        "blockers": [],
        "native_isaac_executed": True,
        "physical_success_established": False,
        "source_target_collider_active": False,
        "replacement_count": 1,
        "probe_results": [
            {"probe": name, "passed": True}
            for name in ("drop", "slide", "tip", "gripper")
        ],
    }
    value["result_digest"] = canonical_digest(value, digest_field="result_digest")
    return value


def test_runtime_digest_accepts_only_verifiable_current_or_retained_encoding() -> None:
    current = _completed_execution()
    assert runtime._runtime_result_digest_valid(current) is True

    retained = dict(current)
    digest = retained.pop("result_digest")
    retained["result_digest"] = digest
    retained["_canonical_digest"] = digest
    assert runtime._runtime_result_digest_valid(retained) is True

    retained["replacement_count"] = 2
    assert runtime._runtime_result_digest_valid(retained) is False


def test_retained_articulated_probe_key_is_adjudicated_without_weakening_set() -> None:
    expected = frozenset({"zero_action_door_stays_shut", "positive_holds"})
    retained = {
        "status": "completed",
        "blockers": [],
        "native_isaac_executed": True,
        "physical_success_established": False,
        "source_target_collider_active": False,
        "replacement_count": 1,
        "probe_results": [
            {"name": "zero_action_door_stays_shut", "passed": True},
            {"name": "positive_holds", "passed": True},
        ],
    }

    assert runtime._execution_blockers(retained, expected) == []
    assert "simready_isaac_probe_set_invalid" in runtime._execution_blockers(
        retained, frozenset({"different_probe"})
    )


def test_retained_execution_adjudication_binds_bytes_and_cannot_override_failure(
    tmp_path: Path,
) -> None:
    execution = _completed_execution()
    execution["probe_results"] = [
        {"name": row["probe"], "passed": row["passed"]}
        for row in execution["probe_results"]
    ]
    execution.pop("result_digest")
    digest = canonical_digest(execution)
    execution["result_digest"] = digest
    execution["_canonical_digest"] = digest
    execution_path = tmp_path / "execution.json"
    write_json(execution_path, execution)

    bundle = {
        "status": "ready",
        "probe_names": ["drop", "slide", "tip", "gripper"],
        "bundle_sha256": "sha256:" + "a" * 64,
        "receipt_digest": "",
    }
    bundle["receipt_digest"] = canonical_digest(bundle, digest_field="receipt_digest")
    bundle_path = tmp_path / "bundle.json"
    write_json(bundle_path, bundle)

    receipt = runtime.adjudicate_retained_simready_isaac_execution(
        execution_path=execution_path,
        bundle_receipt_path=bundle_path,
        destination=tmp_path / "adjudication.json",
        generated_at="2026-08-10T00:00:00+00:00",
    )

    assert receipt["status"] == "passed"
    assert receipt["source_execution"]["retained_legacy_encoding"] is True
    assert receipt["source_execution"]["sha256"] == _sha256(execution_path)

    execution["probe_results"][0]["passed"] = False
    execution.pop("result_digest")
    execution.pop("_canonical_digest")
    digest = canonical_digest(execution)
    execution["result_digest"] = digest
    execution["_canonical_digest"] = digest
    write_json(execution_path, execution)
    blocked = runtime.adjudicate_retained_simready_isaac_execution(
        execution_path=execution_path,
        bundle_receipt_path=bundle_path,
        destination=tmp_path / "blocked.json",
        generated_at="2026-08-10T00:00:01+00:00",
    )
    assert blocked["status"] == "blocked"
    assert "simready_isaac_probe_failure" in blocked["blockers"]


def test_dry_run_never_stages_or_mutates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        runtime,
        "stage_wam_provider_bundle_object_store",
        lambda **kwargs: pytest.fail("dry run staged provider bytes"),
    )

    result = runtime.run_simready_isaac_vast(
        job_dir=tmp_path / "job",
        prepared_bundle=_bundle(tmp_path),
        paid_resource_admission_grant=None,
        execute=False,
    )

    assert result["status"] == "dry_run_ready"
    assert result["provider_mutations_performed"] == 0


def test_live_run_requires_all_four_native_probes_and_provider_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared_bundle = _bundle(tmp_path)
    bundle_receipt_sha256 = "sha256:" + "c" * 64
    authority = _paid_attempt_authority(
        prepared_bundle, bundle_receipt_sha256=bundle_receipt_sha256
    )
    monkeypatch.setenv("BLUEPRINT_SPEND_AUTHORITY_ROOT", str((tmp_path / "consumed").parent))

    def fake_stage(**kwargs):
        staging = Path(kwargs["job_dir"])
        staging.mkdir(parents=True)
        for name in (
            "provider_bundle_url.txt",
            "provider_output_put_url.txt",
            "provider_output_get_url.txt",
        ):
            (staging / name).write_text("https://example.invalid/bound", encoding="utf-8")
        return {"status": "completed", "blockers": []}

    def fake_adapter(**kwargs):
        output = Path(kwargs["provider_runtime_output_zip"])
        output.parent.mkdir(parents=True)
        with zipfile.ZipFile(output, "w") as archive:
            archive.writestr(
                "isaac_runtime_result.json",
                json.dumps(_completed_execution(), sort_keys=True),
            )
        write_json(
            Path(kwargs["job_dir"]) / "vast_teardown_manifest.json",
            {"continuing_spend_from_this_run": False},
        )
        return {"status": "completed", "blockers": [], "estimated_cost_usd": 0.12}

    monkeypatch.setattr(runtime, "stage_wam_provider_bundle_object_store", fake_stage)
    monkeypatch.setattr(runtime, "run_vast_provider_adapter", fake_adapter)
    monkeypatch.setattr(
        runtime,
        "cleanup_staged_wam_provider_objects",
        lambda path: {"all_objects_absent": True},
    )

    result = runtime.run_simready_isaac_vast(
        job_dir=tmp_path / "job",
        prepared_bundle=prepared_bundle,
        paid_resource_admission_grant=object(),  # type: ignore[arg-type]
        paid_attempt_authority=authority,
        bundle_receipt_sha256=bundle_receipt_sha256,
        execute=True,
    )

    assert result["status"] == "completed"
    assert result["retry_cap"] == 0
    assert result["authorization_consumption"]["status"] == "consumed"
    assert result["continuing_spend_from_this_run"] is False
    assert result["all_staged_objects_absent"] is True


def test_live_run_consumes_paid_attempt_authority_once_before_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared_bundle = _bundle(tmp_path)
    bundle_receipt_sha256 = "sha256:" + "c" * 64
    authority = _paid_attempt_authority(
        prepared_bundle, bundle_receipt_sha256=bundle_receipt_sha256
    )
    monkeypatch.setenv("BLUEPRINT_SPEND_AUTHORITY_ROOT", str((tmp_path / "consumed").parent))
    stage_calls = 0

    def fake_stage(**kwargs):
        nonlocal stage_calls
        stage_calls += 1
        return {"status": "blocked", "blockers": ["synthetic_stop_after_consumption"]}

    monkeypatch.setattr(runtime, "stage_wam_provider_bundle_object_store", fake_stage)

    first = runtime.run_simready_isaac_vast(
        job_dir=tmp_path / "job",
        prepared_bundle=prepared_bundle,
        paid_resource_admission_grant=object(),  # type: ignore[arg-type]
        paid_attempt_authority=authority,
        bundle_receipt_sha256=bundle_receipt_sha256,
        execute=True,
    )
    second = runtime.run_simready_isaac_vast(
        job_dir=tmp_path / "job",
        prepared_bundle=prepared_bundle,
        paid_resource_admission_grant=object(),  # type: ignore[arg-type]
        paid_attempt_authority=authority,
        bundle_receipt_sha256=bundle_receipt_sha256,
        execute=True,
    )

    assert first["authorization_consumption"]["status"] == "consumed"
    assert second["status"] == "blocked"
    assert second["provider_mutations_performed"] == 0
    assert "simready_isaac_paid_attempt_authority_consumed" in second["blockers"]
    assert stage_calls == 1


def test_simready_authority_binds_external_instance_allowlist(tmp_path: Path) -> None:
    prepared_bundle = _bundle(tmp_path)
    authority = _paid_attempt_authority(
        prepared_bundle, external_instance_allowlist=[31]
    )

    runtime.validate_simready_isaac_paid_attempt_authority(
        authority,
        prepared_bundle=prepared_bundle,
        bundle_receipt_sha256="sha256:" + "c" * 64,
        max_hourly_rate_usd=1.0,
        hard_cap_usd=3.0,
        hard_ttl_seconds=10_800,
        allowed_active_instance_ids=[31],
    )
    with pytest.raises(ValueError, match="external_instance_allowlist_mismatch"):
        runtime.validate_simready_isaac_paid_attempt_authority(
            authority,
            prepared_bundle=prepared_bundle,
            bundle_receipt_sha256="sha256:" + "c" * 64,
            max_hourly_rate_usd=1.0,
            hard_cap_usd=3.0,
            hard_ttl_seconds=10_800,
            allowed_active_instance_ids=[],
        )


def test_simready_authority_binds_same_goal_concurrent_instances(
    tmp_path: Path,
) -> None:
    prepared_bundle = _bundle(tmp_path)
    authority = _paid_attempt_authority(prepared_bundle)
    authority.pop("external_instance_allowlist")
    authority.update(
        {
            "active_instance_allowlist": {
                "external_provider_owned": [17],
                "same_goal_concurrent": [23],
            },
            "concurrent_goal_id": "fixture-bounded-objects",
            "same_goal_concurrent_members": [
                {
                    "instance_id": 23,
                    "paid_attempt_authority_digest": "sha256:" + "a" * 64,
                }
            ],
        }
    )
    authority["authorization_digest"] = canonical_digest(
        authority, digest_field="authorization_digest"
    )

    runtime.validate_simready_isaac_paid_attempt_authority(
        authority,
        prepared_bundle=prepared_bundle,
        bundle_receipt_sha256="sha256:" + "c" * 64,
        max_hourly_rate_usd=1.0,
        hard_cap_usd=3.0,
        hard_ttl_seconds=10_800,
        allowed_active_instance_ids=[23, 17],
    )

    authority["same_goal_concurrent_members"] = []
    authority["authorization_digest"] = canonical_digest(
        authority, digest_field="authorization_digest"
    )
    with pytest.raises(
        ValueError, match="same_goal_concurrent_allowlist_metadata_invalid"
    ):
        runtime.validate_simready_isaac_paid_attempt_authority(
            authority,
            prepared_bundle=prepared_bundle,
            bundle_receipt_sha256="sha256:" + "c" * 64,
            max_hourly_rate_usd=1.0,
            hard_cap_usd=3.0,
            hard_ttl_seconds=10_800,
            allowed_active_instance_ids=[23, 17],
        )


def _allocator_args(tmp_path: Path, receipt: Path) -> list[str]:
    return [
        "gpu-canary",
        "--probe-kind",
        runtime.PROBE_KIND,
        "--provider",
        "vast",
        "--provider-launch-request",
        str(tmp_path / "unused-request.json"),
        "--release-evidence",
        str(tmp_path / "unused-release.json"),
        "--model-cache-evidence",
        str(tmp_path / "unused-model.json"),
        "--preflight-bundle",
        str(tmp_path / "unused-preflight.json"),
        "--admission-out",
        str(tmp_path / "admission.json"),
        "--bound-request-out",
        str(tmp_path / "unused-bound.json"),
        "--adapter-output",
        str(tmp_path / "adapter.json"),
        "--pod-name",
        "adp009b-simready",
        "--expected-source-commit",
        "a" * 40,
        "--adp-simready-isaac-bundle-receipt",
        str(receipt),
        "--adp-job-dir",
        str(tmp_path / "job"),
        "--adp-max-hourly-rate-usd",
        "1.0",
        "--adp-max-spend-usd",
        "3.0",
        "--adp-hard-ttl-seconds",
        "10800",
    ]


def test_canonical_allocator_binds_exact_bundle_and_withholds_dry_run_grant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = tmp_path / "bundle_receipt.json"
    bundle_receipt = _bundle(tmp_path)
    write_json(receipt, bundle_receipt)
    observed: dict = {}
    monkeypatch.setattr(
        allocator,
        "_control_plane_checkout_blockers",
        lambda: ([], {"orchestrator_source_commit": "a" * 40, "checkout_clean": True}),
    )

    def fake_run(**kwargs):
        observed.update(kwargs)
        return {"status": "dry_run_ready"}

    monkeypatch.setattr(allocator, "run_simready_isaac_vast", fake_run)

    assert allocator.main(_allocator_args(tmp_path, receipt)) == 0
    assert observed["execute"] is False
    assert observed["paid_resource_admission_grant"] is None
    admission = json.loads((tmp_path / "admission.json").read_text(encoding="utf-8"))
    assert admission["status"] == "admitted"
    assert admission["retry_cap"] == 0
    assert admission["allocation_binding"]["bundle_sha256"] == bundle_receipt[
        "bundle_sha256"
    ]


def test_canonical_allocator_reuses_retained_simready_bad_hosts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = tmp_path / "bundle_receipt.json"
    write_json(receipt, _bundle(tmp_path))
    state_root = tmp_path / "launch-runs"
    prior = (
        state_root
        / "prior-launch"
        / "allocator"
        / "simready-isaac-job"
        / "adp009b_simready_isaac_machine_avoidlist.json"
    )
    write_json(
        prior,
        {
            "schema_version": "vast_machine_avoidlist.v1",
            "status": "completed",
            "machine_ids": [140718],
            "entries": [{"machine_id": 140718, "reason": "heartbeat_missing"}],
            "raw_secret_values_recorded": False,
        },
    )
    current_job = state_root / "current-launch" / "allocator" / "simready-isaac-job"
    args = _allocator_args(tmp_path, receipt)
    args[args.index("--adp-job-dir") + 1] = str(current_job)
    observed: dict = {}
    monkeypatch.setattr(
        allocator,
        "_control_plane_checkout_blockers",
        lambda: ([], {"orchestrator_source_commit": "a" * 40, "checkout_clean": True}),
    )
    def fake_run(**kwargs: object) -> dict[str, str]:
        observed.update(kwargs)
        attempt_avoidlist = Path(str(kwargs["machine_avoidlist_path"]))
        value = json.loads(attempt_avoidlist.read_text(encoding="utf-8"))
        value["machine_ids"].append(140719)
        write_json(attempt_avoidlist, value)
        return {"status": "dry_run_ready"}

    monkeypatch.setattr(allocator, "run_simready_isaac_vast", fake_run)

    assert allocator.main(args) == 0
    shared = (
        state_root
        / "provider-machine-avoidlists"
        / "adp009b-simready-isaac-vast-machine-avoidlist.json"
    )
    assert observed["machine_avoidlist_path"] == (
        current_job / "adp009b_simready_isaac_machine_avoidlist.json"
    )
    assert json.loads(shared.read_text(encoding="utf-8"))["machine_ids"] == [140718]
    assert json.loads(
        Path(str(observed["machine_avoidlist_path"])).read_text(encoding="utf-8")
    )["machine_ids"] == [140718, 140719]
    admission = json.loads((tmp_path / "admission.json").read_text(encoding="utf-8"))
    assert admission["allocation_binding"]["machine_avoidlist_sha256"] == (
        "sha256:" + hashlib.sha256(shared.read_bytes()).hexdigest()
    )


def test_allocator_execute_requires_paid_attempt_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = tmp_path / "bundle_receipt.json"
    write_json(receipt, _bundle(tmp_path))
    monkeypatch.setattr(
        allocator,
        "_control_plane_checkout_blockers",
        lambda: ([], {"orchestrator_source_commit": "a" * 40, "checkout_clean": True}),
    )

    rc = allocator.main([*_allocator_args(tmp_path, receipt), "--execute"])

    assert rc == 2
    result = json.loads((tmp_path / "adapter.json").read_text(encoding="utf-8"))
    assert "simready_isaac_paid_attempt_authority_missing" in result["blockers"]


def test_allocator_execute_binds_paid_attempt_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = tmp_path / "bundle_receipt.json"
    bundle_receipt = _bundle(tmp_path)
    write_json(receipt, bundle_receipt)
    receipt_sha256 = _sha256(receipt)
    authority = _paid_attempt_authority(
        bundle_receipt, bundle_receipt_sha256=receipt_sha256
    )
    authority_path = tmp_path / "attempt_authority.json"
    write_json(authority_path, authority)
    observed: dict = {}
    monkeypatch.setattr(
        allocator,
        "_control_plane_checkout_blockers",
        lambda: ([], {"orchestrator_source_commit": "a" * 40, "checkout_clean": True}),
    )

    def fake_admission(_admission, **_kwargs):
        return object()

    def fake_run(**kwargs):
        observed.update(kwargs)
        return {"status": "completed"}

    monkeypatch.setattr(allocator, "require_paid_resource_admission", fake_admission)
    monkeypatch.setattr(allocator, "run_simready_isaac_vast", fake_run)

    rc = allocator.main(
        [
            *_allocator_args(tmp_path, receipt),
            "--adp-simready-isaac-attempt-authority",
            str(authority_path),
            "--execute",
        ]
    )

    assert rc == 0
    assert observed["execute"] is True
    assert observed["paid_attempt_authority"] == authority
    assert observed["bundle_receipt_sha256"] == receipt_sha256
    admission = json.loads((tmp_path / "admission.json").read_text(encoding="utf-8"))
    assert admission["paid_attempt_authority_required_for_execute"] is True
    assert admission["paid_attempt_authority_digest"] == authority[
        "authorization_digest"
    ]
