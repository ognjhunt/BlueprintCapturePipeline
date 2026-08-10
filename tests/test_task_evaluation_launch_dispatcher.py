import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import blueprint_pipeline.task_evaluation_launch_dispatcher as dispatcher_module
from blueprint_pipeline.task_evaluation_launch_dispatcher import (
    CANONICAL_ALLOCATOR_ENTRYPOINT,
    EXECUTE_ENV,
    LAUNCH_RUN_ROOT_PLACEHOLDER,
    SECRET_PROFILE_ID_ENV,
    TaskEvaluationLaunchError,
    canonical_digest,
    dispatch_launch_request,
    load_public_launch_profile_catalog,
    process_launch_queue,
    stage_launch_request,
    validate_launch_profile,
    validate_launch_request,
)
from blueprint_pipeline.task_evaluation_launch_reconciler import reconcile_launches
from scripts.publish_task_evaluation_launch_profiles import publish_profiles


def _reference(name: str) -> dict[str, str]:
    return {
        "bundle_id": name,
        "source_kind": "interiorgs_sage",
        "uri": f"gs://blueprint-runs/{name}.json",
        "digest": "sha256:" + name[0].lower() * 64,
    }


def _path_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _profile(tmp_path: Path) -> dict:
    source = _reference("aaaa-source")
    spec = {
        "uri": "gs://blueprint-runs/evaluation-run-spec.json",
        "digest": "sha256:" + "b" * 64,
    }
    result_path = tmp_path / "allocator-result.json"
    source_manifest = tmp_path / "source-bundle-manifest.json"
    evaluation_spec = tmp_path / "evaluation-run-spec.json"
    source_manifest.write_text('{"scene":"840313"}\n', encoding="utf-8")
    evaluation_spec.write_text('{"spec":"frozen"}\n', encoding="utf-8")
    profile = {
        "schema_version": "task_evaluation_launch_profile.v1",
        "profile_id": "interiorgs-sage-franka-001",
        "program_id": "arm-decision-proof-v1",
        "source_bundle": source,
        "evaluation_run_spec": spec,
        "immutable_inputs": [
            {
                "name": "source_bundle_manifest",
                "path": str(source_manifest.resolve()),
                "digest": _path_digest(source_manifest),
            },
            {
                "name": "evaluation_run_spec",
                "path": str(evaluation_spec.resolve()),
                "digest": _path_digest(evaluation_spec),
            },
        ],
        "allocator": {
            "entrypoint": CANONICAL_ALLOCATOR_ENTRYPOINT,
            "subcommand": "gpu-canary",
            "argv": [
                "--provider-launch-request",
                str(tmp_path / "provider-launch.json"),
                "--release-evidence",
                str(tmp_path / "release.json"),
                "--model-cache-evidence",
                str(tmp_path / "cache.json"),
                "--preflight-bundle",
                str(tmp_path / "preflight.json"),
                "--admission-out",
                str(tmp_path / "admission.json"),
                "--bound-request-out",
                str(tmp_path / "bound-request.json"),
                "--adapter-output",
                str(result_path),
                "--pod-name",
                "adp-web-trigger-test",
            ],
            "max_spend_usd": 2.0,
            "hard_ttl_seconds": 7200,
            "retry_cap": 0,
        },
        "runtime_environment": {
            "BLUEPRINT_ADP009D_CAMERA_RESOLUTION": "policy",
            "BLUEPRINT_ADP009D_EPISODES": "3",
        },
        "execution_admission": {
            "live_enabled": True,
            "readiness_receipt": {
                "uri": "gs://blueprint-runs/readiness.json",
                "digest": "sha256:" + "d" * 64,
            },
            "blockers": [],
        },
        "reconciliation": {
            "required_providers": ["vast"],
            "max_guard_age_seconds": 300,
        },
        "webapp_sync": {"max_attempts": 20},
        "terminal_contract": {
            "result_path": str(result_path),
            "success_statuses": ["completed"],
            "required_values": {
                "continuing_spend_from_this_run": False,
                "retry_cap": 0,
            },
            "required_path_fields": [
                "teardown_manifest_path",
                "artifact_manifest_path",
            ],
        },
        "required_controls": {
            "canonical_allocator": CANONICAL_ALLOCATOR_ENTRYPOINT,
            "secret_profile_id": "canonical-vast-adp",
            "watchdog_required": True,
            "artifact_storage_required": True,
            "teardown_required": True,
            "provider_zero_required": True,
            "webapp_status_sync_required": True,
            "retry_cap": 0,
        },
        "claim_ceiling": "development_only",
    }
    profile["profile_digest"] = canonical_digest(profile, digest_field="profile_digest")
    return profile


def _request(profile: dict) -> dict:
    request = {
        "schema_version": "task_evaluation_launch_request.v1",
        "launch_id": "launch-interiorgs-sage-001",
        "run_id": "run-interiorgs-sage-001",
        "launch_profile_id": profile["profile_id"],
        "launch_profile_digest": profile["profile_digest"],
        "source_bundle": profile["source_bundle"],
        "evaluation_run_spec": profile["evaluation_run_spec"],
        "authorization": {
            "actor": {"id": "founder-001", "role": "admin"},
            "authorized_at": datetime.now(timezone.utc).isoformat(),
            "rights": {
                "approved": True,
                "scope": "interiorgs_sage_simulator_evaluation",
                "evidence": {
                    "uri": "firestore://taskEvaluationLaunchAuthorities/rights-001",
                    "digest": "sha256:" + "c" * 64,
                },
            },
            "spend": {
                "approved": True,
                "currency": "USD",
                "max_spend_usd": 2.0,
                "expires_at": (datetime.now(timezone.utc) + timedelta(days=1)).isoformat(),
            },
            "execution": {"approved": True},
        },
        "required_controls": profile["required_controls"],
        "claim_ceiling": "development_only",
        "idempotency_key": "launch-interiorgs-sage-001",
    }
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    return request


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def test_contract_binds_web_authority_to_pipeline_owned_profile(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    request = _request(profile)
    assert validate_launch_profile(profile) == []
    assert validate_launch_request(request) == []

    tampered = json.loads(json.dumps(request))
    tampered["source_bundle"]["uri"] = "gs://attacker/short-path.zip"
    tampered["request_digest"] = canonical_digest(tampered, digest_field="request_digest")
    assert validate_launch_request(tampered) == []
    profile_dir = tmp_path / "profiles"
    _write(profile_dir / f"{profile['profile_id']}.json", profile)
    request_path = tmp_path / "request.json"
    _write(request_path, tampered)
    calls: list[list[str]] = []
    receipt = dispatch_launch_request(
        request_path=request_path,
        profile_dir=profile_dir,
        state_root=tmp_path / "state",
        allocator_runner=lambda argv: calls.append(list(argv)) or 0,
    )
    assert receipt["status"] == "blocked"
    assert "source_bundle_profile_binding_mismatch" in receipt["blockers"]
    assert calls == []


def test_stage_is_immutable_and_idempotent(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    request = _request(profile)
    first = stage_launch_request(value=request, queue_root=tmp_path / "queue")
    second = stage_launch_request(value=request, queue_root=tmp_path / "queue")
    assert first["already_exists"] is False
    assert second["already_exists"] is True
    changed_path = Path(first["queue_path"])
    changed_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(TaskEvaluationLaunchError, match="immutable_launch_conflict"):
        stage_launch_request(value=request, queue_root=tmp_path / "queue")
    assert changed_path.is_file()


def test_stage_replay_cannot_requeue_a_terminal_launch(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    request = _request(profile)
    queue_root = tmp_path / "queue"
    first = stage_launch_request(value=request, queue_root=queue_root)
    queued_path = Path(first["queue_path"])
    completed_path = queue_root / "completed" / queued_path.name
    completed_path.parent.mkdir(parents=True)
    queued_path.replace(completed_path)

    replay = stage_launch_request(value=request, queue_root=queue_root)

    assert replay["already_exists"] is True
    assert replay["status"] == "completed"
    assert replay["queue_path"] == str(completed_path)
    assert not list((queue_root / "pending").glob("*.json"))


def test_dispatch_calls_only_canonical_allocator_and_live_closeout_is_required(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = _profile(tmp_path)
    request = _request(profile)
    profile_dir = tmp_path / "profiles"
    _write(profile_dir / f"{profile['profile_id']}.json", profile)
    request_path = tmp_path / "request.json"
    _write(request_path, request)
    calls: list[list[str]] = []

    dry = dispatch_launch_request(
        request_path=request_path,
        profile_dir=profile_dir,
        state_root=tmp_path / "dry-state",
        allocator_runner=lambda argv: calls.append(list(argv)) or 0,
    )
    assert dry["status"] == "dry_run_completed"
    assert calls[0][0] == "gpu-canary"
    assert "--execute" not in calls[0]
    assert dry["canonical_allocator"] == CANONICAL_ALLOCATOR_ENTRYPOINT

    monkeypatch.setenv(EXECUTE_ENV, "true")
    monkeypatch.setenv(SECRET_PROFILE_ID_ENV, "canonical-vast-adp")
    teardown = tmp_path / "teardown.json"
    artifacts = tmp_path / "artifacts.json"

    def live_runner(argv: list[str]) -> int:
        calls.append(list(argv))
        _write(teardown, {"continuing_spend_from_this_run": False})
        _write(artifacts, {"status": "retained"})
        _write(
            Path(profile["terminal_contract"]["result_path"]),
            {
                "status": "completed",
                "continuing_spend_from_this_run": False,
                "retry_cap": 0,
                "teardown_manifest_path": str(teardown),
                "artifact_manifest_path": str(artifacts),
            },
        )
        return 0

    live = dispatch_launch_request(
        request_path=request_path,
        profile_dir=profile_dir,
        state_root=tmp_path / "live-state",
        execute=True,
        allocator_runner=live_runner,
    )
    assert live["status"] == "completed"
    assert calls[-1][-1] == "--execute"
    assert live["terminal_evidence"]["status"] == "passed"
    assert live["provider_mutation_attempted"] is True
    assert live["agent_operator_used"] is False


def test_dispatch_renders_all_output_paths_inside_the_launch_run_root(
    tmp_path: Path,
) -> None:
    profile = _profile(tmp_path)
    profile["allocator"]["argv"] = [
        "--provider-launch-request",
        f"{LAUNCH_RUN_ROOT_PLACEHOLDER}/provider-launch.json",
        "--admission-out",
        f"{LAUNCH_RUN_ROOT_PLACEHOLDER}/allocator/admission.json",
        "--adapter-output",
        f"{LAUNCH_RUN_ROOT_PLACEHOLDER}/allocator/result.json",
    ]
    profile["terminal_contract"]["result_path"] = (
        f"{LAUNCH_RUN_ROOT_PLACEHOLDER}/allocator/result.json"
    )
    profile["profile_digest"] = canonical_digest(profile, digest_field="profile_digest")
    request = _request(profile)
    profile_dir = tmp_path / "profiles"
    _write(profile_dir / f"{profile['profile_id']}.json", profile)
    request_path = tmp_path / "request.json"
    _write(request_path, request)
    calls: list[list[str]] = []

    receipt = dispatch_launch_request(
        request_path=request_path,
        profile_dir=profile_dir,
        state_root=tmp_path / "state",
        allocator_runner=lambda argv: calls.append(list(argv)) or 0,
    )

    run_root = (tmp_path / "state" / request["launch_id"]).resolve()
    assert receipt["status"] == "dry_run_completed"
    assert LAUNCH_RUN_ROOT_PLACEHOLDER not in " ".join(calls[0])
    assert str(run_root / "provider-launch.json") in calls[0]
    assert str(run_root / "allocator" / "admission.json") in calls[0]
    assert str(run_root / "allocator" / "result.json") in calls[0]


def test_profile_rejects_unknown_runtime_placeholders(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    profile["allocator"]["argv"][-1] = "{other_run}/result.json"
    profile["terminal_contract"]["result_path"] = "{unsafe}/terminal.json"
    profile["profile_digest"] = canonical_digest(profile, digest_field="profile_digest")

    blockers = validate_launch_profile(profile)

    assert "launch_profile_allocator_argv_placeholder_invalid" in blockers
    assert "launch_profile_terminal_result_path_placeholder_invalid" in blockers


def test_default_dispatch_uses_isolated_canonical_allocator_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = _profile(tmp_path)
    request = _request(profile)
    profile_dir = tmp_path / "profiles"
    _write(profile_dir / f"{profile['profile_id']}.json", profile)
    request_path = tmp_path / "request.json"
    _write(request_path, request)
    calls: list[dict[str, object]] = []

    class Completed:
        returncode = 0
        stdout = '{"success":true}\n'
        stderr = ""

    def run(argv: list[str], **kwargs: object) -> Completed:
        calls.append({"argv": argv, **kwargs})
        return Completed()

    monkeypatch.setattr(dispatcher_module.subprocess, "run", run)
    receipt = dispatch_launch_request(
        request_path=request_path,
        profile_dir=profile_dir,
        state_root=tmp_path / "state",
    )

    assert receipt["status"] == "dry_run_completed"
    assert calls[0]["argv"][1:] == [
        "-m",
        "blueprint_pipeline.paid_resource_allocator",
        "gpu-canary",
        *profile["allocator"]["argv"],
    ]
    assert calls[0]["shell"] is False
    run_root = tmp_path / "state" / request["launch_id"]
    assert (run_root / "allocator.stdout.log").read_text(encoding="utf-8") == (
        '{"success":true}\n'
    )


def test_live_dispatch_blocks_without_independent_execute_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(EXECUTE_ENV, raising=False)
    profile = _profile(tmp_path)
    request = _request(profile)
    profile_dir = tmp_path / "profiles"
    _write(profile_dir / f"{profile['profile_id']}.json", profile)
    request_path = tmp_path / "request.json"
    _write(request_path, request)
    calls: list[list[str]] = []
    receipt = dispatch_launch_request(
        request_path=request_path,
        profile_dir=profile_dir,
        state_root=tmp_path / "state",
        execute=True,
        allocator_runner=lambda argv: calls.append(list(argv)) or 0,
    )
    assert receipt["status"] == "blocked"
    assert f"missing_env_{EXECUTE_ENV}" in receipt["blockers"]
    assert calls == []
    assert receipt["provider_mutation_attempted"] is False


def test_live_dispatch_blocks_dry_only_profile_and_secret_profile_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(EXECUTE_ENV, "true")
    monkeypatch.setenv(SECRET_PROFILE_ID_ENV, "wrong-profile")
    profile = _profile(tmp_path)
    profile["execution_admission"]["live_enabled"] = False
    profile["execution_admission"]["blockers"] = ["scripted_positive_control_not_passed"]
    profile["profile_digest"] = canonical_digest(profile, digest_field="profile_digest")
    request = _request(profile)
    profile_dir = tmp_path / "profiles"
    _write(profile_dir / f"{profile['profile_id']}.json", profile)
    request_path = tmp_path / "request.json"
    _write(request_path, request)
    calls: list[list[str]] = []

    receipt = dispatch_launch_request(
        request_path=request_path,
        profile_dir=profile_dir,
        state_root=tmp_path / "state",
        execute=True,
        allocator_runner=lambda argv: calls.append(list(argv)) or 0,
    )

    assert receipt["status"] == "blocked"
    assert "launch_profile_live_execution_disabled" in receipt["blockers"]
    assert "canonical_secret_profile_mismatch" in receipt["blockers"]
    assert calls == []


def test_profile_runtime_environment_is_scoped_to_allocator_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("BLUEPRINT_ADP009D_CAMERA_RESOLUTION", raising=False)
    profile = _profile(tmp_path)
    request = _request(profile)
    profile_dir = tmp_path / "profiles"
    _write(profile_dir / f"{profile['profile_id']}.json", profile)
    request_path = tmp_path / "request.json"
    _write(request_path, request)
    observed: list[str | None] = []

    receipt = dispatch_launch_request(
        request_path=request_path,
        profile_dir=profile_dir,
        state_root=tmp_path / "state",
        allocator_runner=lambda _argv: (
            observed.append(__import__("os").environ.get("BLUEPRINT_ADP009D_CAMERA_RESOLUTION"))
            or 0
        ),
    )

    assert receipt["status"] == "dry_run_completed"
    assert observed == ["policy"]
    assert __import__("os").environ.get("BLUEPRINT_ADP009D_CAMERA_RESOLUTION") is None


def test_profile_runtime_environment_rejects_authority_or_output_keys(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    profile["runtime_environment"] = {
        "BLUEPRINT_ADP009D_GATED_BACKBONE_AUTHORIZED": "true",
        "BLUEPRINT_ADP009D_OUTPUT_DIR": "/tmp/redirected",
    }
    profile["profile_digest"] = canonical_digest(profile, digest_field="profile_digest")

    blockers = validate_launch_profile(profile)

    assert (
        "launch_profile_runtime_environment_key_invalid:BLUEPRINT_ADP009D_GATED_BACKBONE_AUTHORIZED"
    ) in blockers
    assert "launch_profile_runtime_environment_key_invalid:BLUEPRINT_ADP009D_OUTPUT_DIR" in blockers


def test_queue_moves_failed_request_to_blocked_without_retry(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    request = _request(profile)
    queue_root = tmp_path / "queue"
    stage_launch_request(value=request, queue_root=queue_root)
    result = process_launch_queue(
        queue_root=queue_root,
        profile_dir=tmp_path / "missing-profiles",
        state_root=tmp_path / "state",
        max_messages=1,
        allocator_runner=lambda _argv: 0,
    )
    assert result["status"] == "blocked"
    assert result["processed_count"] == 1
    assert result["automatic_retry_performed"] is False
    assert not list((queue_root / "pending").glob("*.json"))
    assert len(list((queue_root / "blocked").glob("*.json"))) == 1


def test_unhandled_dispatch_failure_stays_processing_for_independent_reconciliation(
    tmp_path: Path,
) -> None:
    profile = _profile(tmp_path)
    request = _request(profile)
    profile_dir = tmp_path / "profiles"
    _write(profile_dir / f"{profile['profile_id']}.json", profile)
    queue_root = tmp_path / "queue"
    stage_launch_request(value=request, queue_root=queue_root)

    def crash_after_boundary(_argv: list[str]) -> int:
        raise RuntimeError("simulated abrupt allocator boundary failure")

    result = process_launch_queue(
        queue_root=queue_root,
        profile_dir=profile_dir,
        state_root=tmp_path / "state",
        allocator_runner=crash_after_boundary,
    )

    assert result["status"] == "blocked"
    assert result["receipts"][0]["provider_mutation_attempted"] is None
    assert result["receipts"][0]["retain_processing_for_reconciliation"] is True
    assert len(list((queue_root / "processing").glob("*.json"))) == 1
    assert not list((queue_root / "blocked").glob("*.json"))
    assert result["automatic_retry_performed"] is False


def test_reconciler_closes_stale_processing_only_after_fresh_provider_zero(
    tmp_path: Path,
) -> None:
    profile = _profile(tmp_path)
    request = _request(profile)
    queue_root = tmp_path / "queue"
    staged = stage_launch_request(value=request, queue_root=queue_root)
    processing = queue_root / "processing" / Path(staged["queue_path"]).name
    processing.parent.mkdir(parents=True)
    Path(staged["queue_path"]).replace(processing)
    run_root = tmp_path / "state" / request["launch_id"]
    old = datetime.now(timezone.utc) - timedelta(hours=3)
    _write(run_root / "launch_profile.json", profile)
    _write(
        run_root / "launch_started.json",
        {
            "schema_version": "task_evaluation_launch_started.v1",
            "started_at": old.isoformat(),
            "hard_ttl_seconds": 60,
        },
    )
    guard_path = tmp_path / "gpu-spend-guard.json"
    _write(
        guard_path,
        {
            "schema_version": "gpu_spend_guard.v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "reap_mode": True,
            "live_instance_count": 0,
            "reap_candidate_ids": [],
            "reap_results": [],
            "inventory_results": [{"provider": "vast", "status": "succeeded"}],
        },
    )

    result = reconcile_launches(
        queue_root=queue_root,
        state_root=tmp_path / "state",
        guard_report_path=guard_path,
    )

    assert result["status"] == "passed"
    assert result["launches"][0]["status"] == "provider_zero_confirmed"
    assert not processing.exists()
    assert (queue_root / "blocked" / processing.name).is_file()
    recovery = json.loads((run_root / "orphan_recovery_receipt.json").read_text())
    assert recovery["provider_zero_confirmed"] is True
    assert recovery["automatic_retry_performed"] is False


def test_reconciler_retains_stale_processing_when_provider_inventory_is_uncertain(
    tmp_path: Path,
) -> None:
    profile = _profile(tmp_path)
    request = _request(profile)
    queue_root = tmp_path / "queue"
    staged = stage_launch_request(value=request, queue_root=queue_root)
    processing = queue_root / "processing" / Path(staged["queue_path"]).name
    processing.parent.mkdir(parents=True)
    Path(staged["queue_path"]).replace(processing)
    run_root = tmp_path / "state" / request["launch_id"]
    _write(run_root / "launch_profile.json", profile)
    _write(
        run_root / "launch_started.json",
        {
            "started_at": (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat(),
            "hard_ttl_seconds": 60,
        },
    )

    result = reconcile_launches(
        queue_root=queue_root,
        state_root=tmp_path / "state",
        guard_report_path=tmp_path / "missing-guard.json",
    )

    assert result["status"] == "blocked"
    assert result["launches"][0]["status"] == "recovery_pending"
    assert processing.is_file()
    assert not (run_root / "orphan_recovery_receipt.json").exists()


def test_profile_publisher_emits_webapp_descriptor_without_allocator_arguments(
    tmp_path: Path,
) -> None:
    profile = _profile(tmp_path)
    source = tmp_path / "staging" / "profile.json"
    _write(source, profile)

    result = publish_profiles(
        profile_paths=[source],
        profile_dir=tmp_path / "published",
        webapp_catalog_out=tmp_path / "catalog.json",
    )

    assert result["status"] == "published"
    catalog = json.loads((tmp_path / "catalog.json").read_text())
    assert catalog == [
        {
            field: profile[field]
            for field in (
                "profile_id",
                "profile_digest",
                "source_bundle",
                "evaluation_run_spec",
                "required_controls",
                "execution_admission",
                "claim_ceiling",
            )
        }
    ]
    assert "allocator" not in catalog[0]
    assert "provider-launch-request" not in json.dumps(catalog)

    replay = publish_profiles(
        profile_paths=[source],
        profile_dir=tmp_path / "published",
        webapp_catalog_out=tmp_path / "catalog.json",
    )
    assert replay["profiles"][0]["created"] is False

    wrapped = load_public_launch_profile_catalog(tmp_path / "catalog.json")
    assert wrapped == {
        "schema_version": "task_evaluation_launch_profile_catalog.v1",
        "profiles": catalog,
    }


def test_public_catalog_rejects_execution_fields_symlinks_and_duplicates(
    tmp_path: Path,
) -> None:
    profile = _profile(tmp_path)
    source = tmp_path / "staging" / "profile.json"
    _write(source, profile)
    publish_profiles(
        profile_paths=[source],
        profile_dir=tmp_path / "published",
        webapp_catalog_out=tmp_path / "catalog.json",
    )
    catalog = json.loads((tmp_path / "catalog.json").read_text())
    catalog[0]["allocator"] = profile["allocator"]
    _write(tmp_path / "unsafe.json", catalog)
    with pytest.raises(TaskEvaluationLaunchError, match="public_catalog_invalid"):
        load_public_launch_profile_catalog(tmp_path / "unsafe.json")

    _write(tmp_path / "duplicate.json", [catalog[0], catalog[0]])
    with pytest.raises(TaskEvaluationLaunchError, match="public_catalog_invalid"):
        load_public_launch_profile_catalog(tmp_path / "duplicate.json")

    (tmp_path / "catalog-link.json").symlink_to(tmp_path / "catalog.json")
    with pytest.raises(TaskEvaluationLaunchError, match="public_catalog_invalid"):
        load_public_launch_profile_catalog(tmp_path / "catalog-link.json")
