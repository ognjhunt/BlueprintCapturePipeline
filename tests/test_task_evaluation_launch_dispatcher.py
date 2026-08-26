import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import blueprint_pipeline.task_evaluation_launch_dispatcher as dispatcher_module
import blueprint_pipeline.task_evaluation_launch_webapp_sync as webapp_sync_module
from blueprint_pipeline.decision_evidence_contracts import (
    cross_runtime_canonical_digest,
)
from blueprint_pipeline.task_evaluation_launch_dispatcher import (
    CANONICAL_ALLOCATOR_ENTRYPOINT,
    EXECUTE_ENV,
    LAUNCH_RECEIPT_DIGEST_CANONICALIZATION,
    LAUNCH_RUN_ROOT_PLACEHOLDER,
    SECRET_PROFILE_ID_ENV,
    TaskEvaluationLaunchError,
    canonical_digest,
    dispatch_launch_request,
    load_public_launch_profile_catalog,
    process_launch_queue,
    public_launch_profile_descriptor,
    stage_launch_request,
    validate_launch_request_against_public_catalog,
    validate_launch_profile,
    validate_launch_request,
    validate_public_launch_profile_descriptor,
)
from blueprint_pipeline.task_evaluation_launch_reconciler import reconcile_launches
from blueprint_pipeline.task_evaluation_immutable_input_resolver import (
    ImmutableInputResolutionError,
    resolve_immutable_input,
)
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
    if "source_commit" in profile:
        request["source_commit"] = profile["source_commit"]
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    return request


def _write_profile_and_request(
    tmp_path: Path, profile: dict
) -> tuple[Path, Path]:
    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)
    profile_path = profile_dir / f"{profile['profile_id']}.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(_request(profile)), encoding="utf-8")
    return profile_dir, request_path


def test_dispatcher_stages_exact_input_and_child_isolated_from_late_source_tamper(
    tmp_path: Path,
) -> None:
    profile = _profile(tmp_path)
    source_path = Path(profile["immutable_inputs"][0]["path"])
    original = source_path.read_bytes()
    profile["allocator"]["argv"].extend(["--exact-input", str(source_path)])
    profile["profile_digest"] = canonical_digest(
        profile, digest_field="profile_digest"
    )
    profile_dir, request_path = _write_profile_and_request(tmp_path, profile)
    observed: dict[str, object] = {}

    def runner(argv: list[str]) -> int:
        staged = Path(argv[argv.index("--exact-input") + 1])
        observed["staged"] = staged
        assert staged != source_path
        assert staged.read_bytes() == original
        source_path.write_bytes(b'{"scene":"tampered-after-copy"}\n')
        assert staged.read_bytes() == original
        assert resolve_immutable_input(
            source_path,
            expected_digest=_path_digest(staged),
            expected_size_bytes=len(original),
        ) == staged
        return 0

    receipt = dispatch_launch_request(
        request_path=request_path,
        profile_dir=profile_dir,
        state_root=tmp_path / "state",
        allocator_runner=runner,
    )

    assert receipt["status"] == "dry_run_completed"
    staging = json.loads(
        (
            tmp_path
            / "state"
            / "launch-interiorgs-sage-001"
            / "immutable_input_staging_receipt.json"
        ).read_text(encoding="utf-8")
    )
    assert staging["status"] == "staged"
    row = next(
        item for item in staging["inputs"] if item["name"] == "source_bundle_manifest"
    )
    assert row["allocator_argv_indices"]
    assert Path(row["staged_path"]) == observed["staged"]
    assert row["staged_digest"] == row["expected_digest"]
    assert staging["source_paths_forwarded_to_allocator"] is False


def test_child_resolver_fails_closed_for_missing_or_tampered_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = _profile(tmp_path)
    source_path = Path(profile["immutable_inputs"][0]["path"])
    expected_digest = _path_digest(source_path)
    receipt, _argv = dispatcher_module._stage_profile_immutable_inputs(
        profile=profile,
        run_root=tmp_path / "run",
        allocator_argv=[],
    )
    monkeypatch.setenv(
        "BLUEPRINT_TASK_EVALUATION_IMMUTABLE_INPUT_STAGING_RECEIPT",
        str(tmp_path / "run" / "immutable_input_staging_receipt.json"),
    )
    with pytest.raises(
        ImmutableInputResolutionError,
        match="immutable_input_staging_mapping_missing",
    ):
        resolve_immutable_input(
            tmp_path / "not-declared.json",
            expected_digest="sha256:" + "0" * 64,
            expected_size_bytes=0,
        )

    staged = Path(receipt["inputs"][0]["staged_path"])
    staged.write_bytes(b"tampered")
    with pytest.raises(
        ImmutableInputResolutionError,
        match="immutable_input_staging_target_identity_mismatch",
    ):
        resolve_immutable_input(
            source_path,
            expected_digest=expected_digest,
            expected_size_bytes=source_path.stat().st_size,
        )


def test_dispatcher_projects_declared_packet_files_and_rewrites_directory(
    tmp_path: Path,
) -> None:
    profile = _profile(tmp_path)
    packet_dir = tmp_path / "native-packet"
    packet_dir.mkdir()
    packet_files = {
        "native_task_arena_packet_receipt.v1.json": b'{"receipt":"sealed"}\n',
        "native_task_arena_packet_request.v1.json": b'{"request":"sealed"}\n',
        "native_task_arena_scene_plan.v1.json": b'{"scene":"sealed"}\n',
    }
    for name, payload in packet_files.items():
        path = packet_dir / name
        path.write_bytes(payload)
        profile["immutable_inputs"].append(
            {
                "name": name.removesuffix(".v1.json"),
                "path": str(path.resolve()),
                "digest": _path_digest(path),
            }
        )
    (packet_dir / "provider-secret.txt").write_text(
        "must-not-be-copied", encoding="utf-8"
    )
    profile["allocator"]["argv"].extend(
        ["--native-task-arena-packet", str(packet_dir.resolve())]
    )
    profile["profile_digest"] = canonical_digest(
        profile, digest_field="profile_digest"
    )
    profile_dir, request_path = _write_profile_and_request(tmp_path, profile)
    observed: dict[str, Path] = {}

    def runner(argv: list[str]) -> int:
        staged_dir = Path(argv[argv.index("--native-task-arena-packet") + 1])
        observed["staged_dir"] = staged_dir
        assert staged_dir != packet_dir
        assert not (staged_dir / "provider-secret.txt").exists()
        for name, payload in packet_files.items():
            assert (staged_dir / name).read_bytes() == payload
            (packet_dir / name).write_bytes(b'{"tampered":true}\n')
            assert (staged_dir / name).read_bytes() == payload
        return 0

    receipt = dispatch_launch_request(
        request_path=request_path,
        profile_dir=profile_dir,
        state_root=tmp_path / "state",
        allocator_runner=runner,
    )

    assert receipt["status"] == "dry_run_completed"
    staging = json.loads(
        (
            tmp_path
            / "state"
            / "launch-interiorgs-sage-001"
            / "immutable_input_staging_receipt.json"
        ).read_text(encoding="utf-8")
    )
    assert staging["directory_projection_count"] == 1
    projection = staging["directory_projections"][0]
    assert Path(projection["staged_directory"]) == observed["staged_dir"]
    assert projection["allocator_argv_indices"]
    assert {item["relative_path"] for item in projection["inputs"]} == set(
        packet_files
    )


def test_dispatcher_blocks_source_tamper_before_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = _profile(tmp_path)
    source_path = Path(profile["immutable_inputs"][0]["path"])
    profile["allocator"]["argv"].extend(["--exact-input", str(source_path)])
    profile["profile_digest"] = canonical_digest(
        profile, digest_field="profile_digest"
    )
    profile_dir, request_path = _write_profile_and_request(tmp_path, profile)
    real_stage = dispatcher_module._stage_profile_immutable_inputs

    def tamper_then_stage(**kwargs: object):  # type: ignore[no-untyped-def]
        source_path.write_bytes(b'{"scene":"tampered-before-copy"}\n')
        return real_stage(**kwargs)

    monkeypatch.setattr(
        dispatcher_module, "_stage_profile_immutable_inputs", tamper_then_stage
    )
    calls: list[list[str]] = []
    receipt = dispatch_launch_request(
        request_path=request_path,
        profile_dir=profile_dir,
        state_root=tmp_path / "state",
        allocator_runner=lambda argv: calls.append(list(argv)) or 0,
    )

    assert receipt["status"] == "blocked"
    assert any(
        "immutable_input_staging_source_digest_mismatch" in blocker
        for blocker in receipt["blockers"]
    )
    assert calls == []


def test_dispatcher_refuses_embedded_immutable_input_path(
    tmp_path: Path,
) -> None:
    profile = _profile(tmp_path)
    source_path = profile["immutable_inputs"][0]["path"]
    profile["allocator"]["argv"].append(f"--exact-input={source_path}")
    profile["profile_digest"] = canonical_digest(
        profile, digest_field="profile_digest"
    )
    profile_dir, request_path = _write_profile_and_request(tmp_path, profile)
    calls: list[list[str]] = []

    receipt = dispatch_launch_request(
        request_path=request_path,
        profile_dir=profile_dir,
        state_root=tmp_path / "state",
        allocator_runner=lambda argv: calls.append(list(argv)) or 0,
    )

    assert receipt["status"] == "blocked"
    assert any(
        "immutable_input_allocator_path_not_exactly_rewritable" in blocker
        for blocker in receipt["blockers"]
    )
    assert calls == []


def test_concurrent_immutable_input_staging_is_byte_idempotent(
    tmp_path: Path,
) -> None:
    profile = _profile(tmp_path)
    source_path = profile["immutable_inputs"][0]["path"]
    argv = ["--exact-input", source_path]
    run_root = tmp_path / "run"

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(
                lambda _index: dispatcher_module._stage_profile_immutable_inputs(
                    profile=profile,
                    run_root=run_root,
                    allocator_argv=argv,
                ),
                range(2),
            )
        )

    assert results[0][1] == results[1][1]
    assert Path(results[0][1][1]).read_bytes() == Path(source_path).read_bytes()
    receipt = json.loads(
        (run_root / "immutable_input_staging_receipt.json").read_text(
            encoding="utf-8"
        )
    )
    assert receipt["receipt_digest"] == canonical_digest(
        receipt, digest_field="receipt_digest"
    )


def _native_policy_binding() -> dict:
    return {
        "schema_version": "native_task_arena_policy_binding.v1",
        "candidate_id": "pi05_droid",
        "robot": {
            "runtime_robot_id": "franka_panda",
            "rights_embodiment_id": "franka",
            "alias_binding_digest": "sha256:" + "0" * 64,
            "config_digest": "sha256:" + "1" * 64,
        },
        "task": {
            "task_id": "task_a_washer_door_open",
            "config_digest": "sha256:" + "2" * 64,
        },
        "policy": {
            "spec_digest": "sha256:" + "3" * 64,
            "input_schema_digest": "sha256:" + "4" * 64,
            "output_schema_digest": "sha256:" + "5" * 64,
            "action_adapter": "normalized_joint_position_v1",
        },
        "runtime": {
            "arena_container_image": "docker.io/blueprint/arena@sha256:" + "6" * 64,
            "arena_container_digest_pinned": True,
            "candidate_policy_container": False,
        },
        "rights": {
            "scene_policy_readiness_digest": "sha256:" + "7" * 64,
            "candidate_rights_binding_digest": "sha256:" + "8" * 64,
        },
    }


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _webapp_sync_succeeded(receipt: dict, *, attempt_number: int = 1) -> dict:
    value = {
        "schema_version": "task_evaluation_launch_webapp_sync_result.v1",
        "status": "succeeded",
        "launch_id": receipt["launch_id"],
        "run_id": receipt["run_id"],
        "request_digest": receipt["request_digest"],
        "receipt_digest": receipt["receipt_digest"],
        "response": {
            "launch_id": receipt["launch_id"],
            "run_id": receipt["run_id"],
            "request_digest": receipt["request_digest"],
            "receipt_digest": receipt["receipt_digest"],
        },
        "attempt_number": attempt_number,
        "attempted_at": "2026-08-13T14:00:00+00:00",
        "provider_mutation_performed": False,
    }
    value["sync_result_digest"] = canonical_digest(
        value, digest_field="sync_result_digest"
    )
    return value


def _zero_guard(*, generated_at: datetime, live_instance_count: int = 0) -> dict:
    provider_zero = live_instance_count == 0
    return {
        "schema_version": "gpu_spend_guard.v1",
        "generated_at": generated_at.isoformat(),
        "reap_mode": True,
        "live_instance_count": live_instance_count,
        "total_burn_per_hour_usd": 0.0 if provider_zero else 0.5,
        "reap_candidate_ids": [],
        "reap_results": [],
        "inventory_results": [{
            "provider": "vast",
            "status": "succeeded",
            "row_count": live_instance_count,
            "required": True,
        }],
        "provider_zero_verified": provider_zero,
        "provider_zero": {
            "status": "verified" if provider_zero else "unverified",
            "required_provider_ids": ["vast"],
            "global_live_instance_count": live_instance_count,
            "global_total_burn_per_hour_usd": 0.0 if provider_zero else 0.5,
            "blockers": [] if provider_zero else ["provider_zero_live_instances_observed"],
        },
    }


def _paid_terminal_receipt(
    *, request: dict, profile: dict, teardown_path: Path
) -> dict:
    receipt = {
        "schema_version": "task_evaluation_launch_receipt.v1",
        # This intentionally remains a scientific blocker. A provider-zero
        # receipt proves resource closure only, never policy success.
        "status": "blocked",
        "launch_id": request["launch_id"],
        "run_id": request["run_id"],
        "request_digest": request["request_digest"],
        "launch_profile_digest": profile["profile_digest"],
        "execute_requested": True,
        "provider_mutation_attempted": True,
        "terminal_evidence": {
            "status": "blocked",
            "artifacts": {
                "teardown_manifest_path": {
                    "path": str(teardown_path.resolve()),
                    "exists": True,
                    "digest": _path_digest(teardown_path),
                }
            },
        },
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    return receipt


def test_live_profile_accepts_full_readback_r2_manifest_uri(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    identity = "f" * 64
    uri = f"r2://blueprint/task-evaluation/sha256/ff/{identity}.json"
    profile["source_bundle"]["uri"] = uri
    profile["evaluation_run_spec"]["uri"] = uri
    profile["execution_admission"]["readiness_receipt"]["uri"] = uri
    profile["profile_digest"] = canonical_digest(
        profile, digest_field="profile_digest"
    )

    assert validate_launch_profile(profile) == []


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


def test_native_policy_binding_binds_robot_policy_and_shared_arena_runtime(
    tmp_path: Path,
) -> None:
    profile = _profile(tmp_path)
    profile["native_policy_binding"] = _native_policy_binding()
    profile["profile_digest"] = canonical_digest(
        profile, digest_field="profile_digest"
    )
    assert validate_launch_profile(profile) == []
    request = _request(profile)
    # WebApp execution authority and frozen-candidate checkpoint rights are
    # independent authorities and must not be forced to share a digest.
    request["request_digest"] = canonical_digest(
        request, digest_field="request_digest"
    )
    profile_dir = tmp_path / "profiles"
    _write(profile_dir / f"{profile['profile_id']}.json", profile)
    request_path = tmp_path / "request.json"
    _write(request_path, request)

    receipt = dispatch_launch_request(
        request_path=request_path,
        profile_dir=profile_dir,
        state_root=tmp_path / "state",
        allocator_runner=lambda _argv: 0,
    )

    assert receipt["status"] == "dry_run_completed"


def test_native_policy_campaign_binds_the_exact_webapp_launch_id(
    tmp_path: Path,
) -> None:
    profile = _profile(tmp_path)
    binding = _native_policy_binding()
    binding["policy_campaign"] = {
        "campaign_id": "scene-840920-policy-pair-1",
        "campaign_digest": "sha256:" + "9" * 64,
        "member_id": "pi05_droid",
        "launch_id": "launch-policy-pi05-campaign-1",
        "resource_name": "blueprint-native-task-policy-pi05-" + "a" * 32,
        "sibling_member_id": "groot_n17_droid",
        "sibling_launch_id": "launch-policy-groot-campaign-1",
        "sibling_resource_name": "blueprint-native-task-policy-groot-" + "b" * 32,
    }
    profile["native_policy_binding"] = binding
    profile["profile_digest"] = canonical_digest(
        profile, digest_field="profile_digest"
    )
    assert validate_launch_profile(profile) == []
    profile_dir = tmp_path / "profiles"
    _write(profile_dir / f"{profile['profile_id']}.json", profile)

    mismatched = _request(profile)
    mismatched["request_digest"] = canonical_digest(
        mismatched, digest_field="request_digest"
    )
    mismatch_path = tmp_path / "mismatched-request.json"
    _write(mismatch_path, mismatched)
    blocked = dispatch_launch_request(
        request_path=mismatch_path,
        profile_dir=profile_dir,
        state_root=tmp_path / "blocked-state",
        allocator_runner=lambda _argv: 0,
    )
    assert blocked["status"] == "blocked"
    assert "native_policy_campaign_launch_id_mismatch" in blocked["blockers"]

    matched = _request(profile)
    matched["launch_id"] = "launch-policy-pi05-campaign-1"
    matched["idempotency_key"] = matched["launch_id"]
    matched["request_digest"] = canonical_digest(
        matched, digest_field="request_digest"
    )
    matched_path = tmp_path / "matched-request.json"
    _write(matched_path, matched)
    accepted = dispatch_launch_request(
        request_path=matched_path,
        profile_dir=profile_dir,
        state_root=tmp_path / "accepted-state",
        allocator_runner=lambda _argv: 0,
    )
    assert accepted["status"] == "dry_run_completed"


def test_native_policy_binding_refuses_tagged_or_empty_arena_container(
    tmp_path: Path,
) -> None:
    profile = _profile(tmp_path)
    profile["native_policy_binding"] = _native_policy_binding()
    profile["native_policy_binding"]["runtime"]["arena_container_image"] = (
        "docker.io/blueprint/arena:latest"
    )
    profile["profile_digest"] = canonical_digest(
        profile, digest_field="profile_digest"
    )
    assert "native_policy_binding_arena_container_invalid" in validate_launch_profile(
        profile
    )

    profile["native_policy_binding"] = _native_policy_binding()
    profile["native_policy_binding"]["runtime"]["arena_container_image"] = (
        "@sha256:" + "6" * 64
    )
    profile["profile_digest"] = canonical_digest(
        profile, digest_field="profile_digest"
    )
    assert "native_policy_binding_arena_container_invalid" in validate_launch_profile(
        profile
    )


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


def test_dispatch_rejects_a_profile_absent_from_the_published_catalog(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    request = _request(profile)
    profile_dir = tmp_path / "profiles"
    _write(profile_dir / f"{profile['profile_id']}.json", profile)
    request_path = tmp_path / "request.json"
    _write(request_path, request)
    catalog_path = tmp_path / "catalog.json"
    _write(catalog_path, [])
    calls: list[list[str]] = []

    assert validate_launch_request_against_public_catalog(
        request, catalog_path=catalog_path
    ) == ["launch_profile_not_published"]
    receipt = dispatch_launch_request(
        request_path=request_path,
        profile_dir=profile_dir,
        state_root=tmp_path / "state",
        public_catalog_path=catalog_path,
        allocator_runner=lambda argv: calls.append(list(argv)) or 0,
    )

    assert receipt["status"] == "blocked"
    assert "launch_profile_not_published" in receipt["blockers"]
    assert receipt["provider_mutation_attempted"] is False
    assert calls == []


def test_public_catalog_binds_request_fields_to_the_published_descriptor(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    request = _request(profile)
    catalog_path = tmp_path / "catalog.json"
    _write(catalog_path, [public_launch_profile_descriptor(profile)])
    request["source_bundle"] = {
        **request["source_bundle"],
        "uri": "gs://blueprint-runs/tampered-source.json",
    }
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")

    assert validate_launch_request_against_public_catalog(
        request, catalog_path=catalog_path
    ) == ["launch_profile_public_catalog_source_bundle_mismatch"]


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
        execute_launch_id=request["launch_id"],
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
    monkeypatch.delenv("BLUEPRINT_ADP009D_CAMERA_RESOLUTION", raising=False)
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
    assert calls[0]["env"]["BLUEPRINT_ADP009D_CAMERA_RESOLUTION"] == "policy"
    assert __import__("os").environ.get("BLUEPRINT_ADP009D_CAMERA_RESOLUTION") is None
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
    assert "execute_launch_id_required" in receipt["blockers"]
    assert calls == []
    assert receipt["provider_mutation_attempted"] is False


def test_native_policy_sigterm_before_admission_seals_typed_media_gap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A killed allocator used to leave neither its result nor the mandatory
    typed pre-observation media gap, while claiming a provider mutation attempt
    solely because the subprocess had been invoked."""

    monkeypatch.setenv(EXECUTE_ENV, "true")
    monkeypatch.setenv(SECRET_PROFILE_ID_ENV, "canonical-vast-adp")
    profile = _profile(tmp_path)
    profile["native_policy_binding"] = _native_policy_binding()
    profile["profile_digest"] = canonical_digest(
        profile, digest_field="profile_digest"
    )
    request = _request(profile)
    profile_dir = tmp_path / "profiles"
    _write(profile_dir / f"{profile['profile_id']}.json", profile)
    request_path = tmp_path / "request.json"
    _write(request_path, request)

    receipt = dispatch_launch_request(
        request_path=request_path,
        profile_dir=profile_dir,
        state_root=tmp_path / "state",
        execute=True,
        execute_launch_id=request["launch_id"],
        allocator_runner=lambda _argv: -15,
    )

    assert receipt["status"] == "blocked"
    assert receipt["execute_requested"] is True
    assert receipt["allocator_invoked"] is True
    assert receipt["allocator_exit_code"] == -15
    assert receipt["provider_mutation_attempted"] is False
    assert receipt["provider_mutation_evidence"] == {
        "schema_version": "task_evaluation_provider_mutation_evidence.v1",
        "status": "absent_before_paid_admission",
        "allocator_invoked": True,
        "admission_artifact_path_configured": True,
        "bound_request_artifact_path_configured": True,
        "adapter_output_artifact_path_configured": True,
        "all_boundary_artifact_paths_configured": True,
        "admission_artifact_present": False,
        "bound_request_artifact_present": False,
        "adapter_output_artifact_present": False,
        "terminal_result_artifact_present": False,
        "raw_secret_values_recorded": False,
    }
    expected_visual = {
        "status": "unavailable_before_first_observation",
        "media_gap": {
            "type": "before_first_observation",
            "reason": "allocator_terminated_before_paid_admission",
        },
    }
    assert receipt["visual_evidence"] == expected_visual
    assert receipt["terminal_evidence"]["visual_evidence"] == expected_visual
    assert "allocator_terminal_result_missing" in receipt["blockers"]
    assert "canonical_allocator_nonzero_exit" in receipt["blockers"]
    persisted = json.loads(
        (
            tmp_path
            / "state"
            / request["launch_id"]
            / "launch_receipt.json"
        ).read_text(encoding="utf-8")
    )
    assert persisted == receipt
    assert receipt["receipt_digest_canonicalization"] == (
        LAUNCH_RECEIPT_DIGEST_CANONICALIZATION
    )
    assert receipt["receipt_digest"] == cross_runtime_canonical_digest(
        receipt, digest_field="receipt_digest"
    )


def test_native_policy_post_admission_result_propagates_typed_media_gap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A terminal allocator result remains the media-gap authority after admission."""

    monkeypatch.setenv(EXECUTE_ENV, "true")
    monkeypatch.setenv(SECRET_PROFILE_ID_ENV, "canonical-vast-adp")
    profile = _profile(tmp_path)
    profile["native_policy_binding"] = _native_policy_binding()
    profile["profile_digest"] = canonical_digest(
        profile, digest_field="profile_digest"
    )
    request = _request(profile)
    profile_dir = tmp_path / "profiles"
    _write(profile_dir / f"{profile['profile_id']}.json", profile)
    request_path = tmp_path / "request.json"
    _write(request_path, request)
    expected_visual = {
        "status": "unavailable_before_first_observation",
        "media_gap": {
            "type": "before_first_observation",
            "reason": "vast_probe_interrupted_before_completion",
        },
    }

    def _runner(_argv: list[str]) -> int:
        _write(tmp_path / "admission.json", {"status": "admitted"})
        _write(tmp_path / "bound-request.json", {"status": "bound"})
        _write(
            tmp_path / "allocator-result.json",
            {
                "schema_version": "native_task_arena_vast_run.v1",
                "status": "blocked",
                "continuing_spend_from_this_run": False,
                "retry_cap": 0,
                "visual_evidence": expected_visual,
            },
        )
        return -15

    receipt = dispatch_launch_request(
        request_path=request_path,
        profile_dir=profile_dir,
        state_root=tmp_path / "state",
        execute=True,
        execute_launch_id=request["launch_id"],
        allocator_runner=_runner,
    )

    assert receipt["status"] == "blocked"
    assert receipt["allocator_invoked"] is True
    assert receipt["provider_mutation_attempted"] is True
    assert (
        receipt["provider_mutation_evidence"]["status"]
        == "allocator_boundary_artifacts_present"
    )
    assert receipt["visual_evidence"] == expected_visual
    assert receipt["terminal_evidence"]["visual_evidence"] == expected_visual
    assert receipt["receipt_digest_canonicalization"] == (
        LAUNCH_RECEIPT_DIGEST_CANONICALIZATION
    )
    assert receipt["receipt_digest"] == cross_runtime_canonical_digest(
        receipt, digest_field="receipt_digest"
    )


def test_native_policy_missing_result_does_not_invent_preobservation_after_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Once the paid admission artifact exists, a missing result cannot prove
    whether the provider reached its first observation."""

    monkeypatch.setenv(EXECUTE_ENV, "true")
    monkeypatch.setenv(SECRET_PROFILE_ID_ENV, "canonical-vast-adp")
    profile = _profile(tmp_path)
    profile["native_policy_binding"] = _native_policy_binding()
    profile["profile_digest"] = canonical_digest(
        profile, digest_field="profile_digest"
    )
    request = _request(profile)
    profile_dir = tmp_path / "profiles"
    _write(profile_dir / f"{profile['profile_id']}.json", profile)
    request_path = tmp_path / "request.json"
    _write(request_path, request)

    def _runner(_argv: list[str]) -> int:
        _write(tmp_path / "admission.json", {"status": "admitted"})
        return -15

    receipt = dispatch_launch_request(
        request_path=request_path,
        profile_dir=profile_dir,
        state_root=tmp_path / "state",
        execute=True,
        execute_launch_id=request["launch_id"],
        allocator_runner=_runner,
    )

    assert receipt["execute_requested"] is True
    assert receipt["allocator_invoked"] is True
    assert receipt["provider_mutation_attempted"] is True
    assert (
        receipt["provider_mutation_evidence"]["status"]
        == "allocator_boundary_artifacts_present"
    )
    assert receipt["provider_mutation_evidence"]["admission_artifact_present"] is True
    assert "visual_evidence" not in receipt
    assert "visual_evidence" not in receipt["terminal_evidence"]


@pytest.mark.parametrize(
    "removed_flags",
    [
        ("--admission-out",),
        ("--admission-out", "--bound-request-out", "--adapter-output"),
    ],
)
def test_native_policy_missing_boundary_paths_stays_conservative(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    removed_flags: tuple[str, ...],
) -> None:
    """Uninstrumented argv cannot prove that no paid boundary was crossed."""

    monkeypatch.setenv(EXECUTE_ENV, "true")
    monkeypatch.setenv(SECRET_PROFILE_ID_ENV, "canonical-vast-adp")
    profile = _profile(tmp_path)
    profile["native_policy_binding"] = _native_policy_binding()
    argv = profile["allocator"]["argv"]
    for flag in removed_flags:
        index = argv.index(flag)
        del argv[index : index + 2]
    profile["profile_digest"] = canonical_digest(
        profile, digest_field="profile_digest"
    )
    request = _request(profile)
    profile_dir = tmp_path / "profiles"
    _write(profile_dir / f"{profile['profile_id']}.json", profile)
    request_path = tmp_path / "request.json"
    _write(request_path, request)

    receipt = dispatch_launch_request(
        request_path=request_path,
        profile_dir=profile_dir,
        state_root=tmp_path / "state",
        execute=True,
        execute_launch_id=request["launch_id"],
        allocator_runner=lambda _argv: -15,
    )

    assert receipt["execute_requested"] is True
    assert receipt["allocator_invoked"] is True
    assert receipt["provider_mutation_attempted"] is True
    evidence = receipt["provider_mutation_evidence"]
    assert evidence["status"] == "boundary_artifact_paths_unconfigured"
    assert evidence["all_boundary_artifact_paths_configured"] is False
    for flag, field in (
        ("--admission-out", "admission_artifact_path_configured"),
        ("--bound-request-out", "bound_request_artifact_path_configured"),
        ("--adapter-output", "adapter_output_artifact_path_configured"),
    ):
        assert evidence[field] is (flag not in removed_flags)
    assert "visual_evidence" not in receipt
    assert "visual_evidence" not in receipt["terminal_evidence"]


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


def test_vast_geolocation_preference_is_an_allowed_scoped_runtime_key(
    tmp_path: Path,
) -> None:
    profile = _profile(tmp_path)
    profile["runtime_environment"] = {
        "BLUEPRINT_VAST_PREFERRED_GEOLOCATION_REGEX": "california|oregon|texas"
    }
    profile["profile_digest"] = canonical_digest(
        profile, digest_field="profile_digest"
    )

    assert validate_launch_profile(profile) == []


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


def test_paid_queue_execution_requires_an_exact_launch_id_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = _profile(tmp_path)
    request = _request(profile)
    queue_root = tmp_path / "queue"
    stage_launch_request(value=request, queue_root=queue_root)
    monkeypatch.setenv(EXECUTE_ENV, "true")
    monkeypatch.setenv(SECRET_PROFILE_ID_ENV, "canonical-vast-adp")
    calls: list[list[str]] = []

    result = process_launch_queue(
        queue_root=queue_root,
        profile_dir=tmp_path / "profiles",
        state_root=tmp_path / "state",
        execute=True,
        allocator_runner=lambda argv: calls.append(list(argv)) or 0,
    )

    assert result["status"] == "blocked"
    assert result["processed_count"] == 0
    assert result["blockers"] == ["execute_launch_id_required"]
    assert calls == []
    assert len(list((queue_root / "pending").glob("*.json"))) == 1


def test_paid_queue_execution_leaves_unscoped_launches_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = _profile(tmp_path)
    target = _request(profile)
    other = json.loads(json.dumps(target))
    other["launch_id"] = "launch-interiorgs-sage-other"
    other["run_id"] = "run-interiorgs-sage-other"
    other["idempotency_key"] = "launch-interiorgs-sage-other"
    other["request_digest"] = canonical_digest(other, digest_field="request_digest")
    queue_root = tmp_path / "queue"
    target_stage = stage_launch_request(value=target, queue_root=queue_root)
    other_stage = stage_launch_request(value=other, queue_root=queue_root)
    profile_dir = tmp_path / "profiles"
    _write(profile_dir / f"{profile['profile_id']}.json", profile)
    monkeypatch.setenv(EXECUTE_ENV, "true")
    monkeypatch.setenv(SECRET_PROFILE_ID_ENV, "canonical-vast-adp")
    teardown = tmp_path / "teardown.json"
    artifacts = tmp_path / "artifacts.json"
    calls: list[list[str]] = []

    def runner(argv: list[str]) -> int:
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

    result = process_launch_queue(
        queue_root=queue_root,
        profile_dir=profile_dir,
        state_root=tmp_path / "state",
        execute=True,
        execute_launch_id=target["launch_id"],
        max_messages=10,
        allocator_runner=runner,
    )

    assert result["status"] == "completed"
    assert result["processed_count"] == 1
    assert result["execute_launch_id"] == target["launch_id"]
    assert result["receipts"][0]["launch_id"] == target["launch_id"]
    assert result["receipts"][0]["execute_launch_id"] == target["launch_id"]
    assert calls and calls[0][-1] == "--execute"
    assert Path(target_stage["queue_path"]).name in {
        path.name for path in (queue_root / "completed").glob("*.json")
    }
    assert Path(other_stage["queue_path"]).is_file()
    bound = json.loads(
        (tmp_path / "state" / target["launch_id"] / "launch_binding.json").read_text(
            encoding="utf-8"
        )
    )
    assert bound["execute_launch_id"] == target["launch_id"]


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


def _stage_distinct_queue_requests(
    *, queue_root: Path, profile: dict, count: int
) -> list[dict]:
    requests: list[dict] = []
    for index in range(count):
        request = _request(profile)
        request["launch_id"] = f"launch-concurrent-{index}"
        request["run_id"] = f"run-concurrent-{index}"
        request["idempotency_key"] = request["launch_id"]
        request["request_digest"] = canonical_digest(
            request, digest_field="request_digest"
        )
        stage_launch_request(value=request, queue_root=queue_root)
        requests.append(request)
    return requests


def test_explicit_concurrency_overlaps_two_independently_claimed_requests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Removing the worker pool makes the barrier fail: this is an overlap test."""

    profile = _profile(tmp_path)
    queue_root = tmp_path / "queue"
    requests = _stage_distinct_queue_requests(
        queue_root=queue_root, profile=profile, count=2
    )
    barrier = threading.Barrier(2)
    active = 0
    peak_active = 0
    active_lock = threading.Lock()
    seen_launch_ids: list[str] = []

    def fake_dispatch(**kwargs) -> dict:
        nonlocal active, peak_active
        request = json.loads(Path(kwargs["request_path"]).read_text(encoding="utf-8"))
        with active_lock:
            active += 1
            peak_active = max(peak_active, active)
            seen_launch_ids.append(request["launch_id"])
        barrier.wait(timeout=5)
        with active_lock:
            active -= 1
        return {"status": "dry_run_completed", "launch_id": request["launch_id"]}

    monkeypatch.setattr(dispatcher_module, "dispatch_launch_request", fake_dispatch)

    report = process_launch_queue(
        queue_root=queue_root,
        profile_dir=tmp_path / "profiles",
        state_root=tmp_path / "state",
        max_messages=2,
        max_concurrency=2,
    )

    assert report["status"] == "completed"
    assert report["processed_count"] == 2
    assert report["max_concurrency"] == 2
    assert peak_active == 2
    assert set(seen_launch_ids) == {request["launch_id"] for request in requests}
    assert len(list((queue_root / "completed").glob("*.json"))) == 2


def test_concurrent_queue_workers_never_double_claim_the_same_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = _profile(tmp_path)
    queue_root = tmp_path / "queue"
    _stage_distinct_queue_requests(queue_root=queue_root, profile=profile, count=1)
    entered_dispatch = threading.Event()
    release_dispatch = threading.Event()
    calls = 0
    calls_lock = threading.Lock()

    def fake_dispatch(**kwargs) -> dict:
        nonlocal calls
        with calls_lock:
            calls += 1
        entered_dispatch.set()
        assert release_dispatch.wait(timeout=5)
        request = json.loads(Path(kwargs["request_path"]).read_text(encoding="utf-8"))
        return {"status": "dry_run_completed", "launch_id": request["launch_id"]}

    monkeypatch.setattr(dispatcher_module, "dispatch_launch_request", fake_dispatch)

    def process() -> dict:
        return process_launch_queue(
            queue_root=queue_root,
            profile_dir=tmp_path / "profiles",
            state_root=tmp_path / "state",
            max_messages=1,
            max_concurrency=1,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(process)
        assert entered_dispatch.wait(timeout=5)
        second = executor.submit(process)
        second_report = second.result(timeout=5)
        release_dispatch.set()
        first_report = first.result(timeout=5)

    assert calls == 1
    assert sorted(
        (first_report["processed_count"], second_report["processed_count"])
    ) == [0, 1]
    assert len(list((queue_root / "completed").glob("*.json"))) == 1


def test_one_concurrent_provider_error_does_not_strand_its_sibling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = _profile(tmp_path)
    queue_root = tmp_path / "queue"
    requests = _stage_distinct_queue_requests(
        queue_root=queue_root, profile=profile, count=2
    )
    failing_launch_id = requests[0]["launch_id"]
    barrier = threading.Barrier(2)

    def fake_dispatch(**kwargs) -> dict:
        request = json.loads(Path(kwargs["request_path"]).read_text(encoding="utf-8"))
        barrier.wait(timeout=5)
        if request["launch_id"] == failing_launch_id:
            raise RuntimeError("provider boundary failed")
        return {"status": "dry_run_completed", "launch_id": request["launch_id"]}

    monkeypatch.setattr(dispatcher_module, "dispatch_launch_request", fake_dispatch)

    report = process_launch_queue(
        queue_root=queue_root,
        profile_dir=tmp_path / "profiles",
        state_root=tmp_path / "state",
        max_messages=2,
        max_concurrency=2,
    )

    assert report["status"] == "blocked"
    assert report["processed_count"] == 2
    failed = next(row for row in report["receipts"] if row["status"] == "blocked")
    assert failed["error_type"] == "RuntimeError"
    assert failed["retain_processing_for_reconciliation"] is True
    assert len(list((queue_root / "processing").glob("*.json"))) == 1
    assert len(list((queue_root / "completed").glob("*.json"))) == 1


@pytest.mark.parametrize("max_concurrency", [0, 4, True, 1.5])
def test_dispatcher_refuses_concurrency_outside_one_to_three(
    tmp_path: Path, max_concurrency: object
) -> None:
    with pytest.raises(
        TaskEvaluationLaunchError,
        match=(
            "launch_dispatch_concurrency_invalid"
            if max_concurrency in (True, 1.5)
            else "launch_dispatch_concurrency_out_of_bounds"
        ),
    ):
        process_launch_queue(
            queue_root=tmp_path / "queue",
            profile_dir=tmp_path / "profiles",
            state_root=tmp_path / "state",
            max_concurrency=max_concurrency,  # type: ignore[arg-type]
        )


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
        _zero_guard(generated_at=datetime.now(timezone.utc)),
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


@pytest.mark.parametrize("cross_runtime_receipt", [False, True])
def test_reconciler_retains_post_teardown_provider_zero_for_paid_terminal(
    tmp_path: Path, cross_runtime_receipt: bool,
) -> None:
    profile = _profile(tmp_path)
    request = _request(profile)
    run_root = tmp_path / "state" / request["launch_id"]
    teardown_at = datetime.now(timezone.utc) - timedelta(seconds=5)
    teardown_path = run_root / "vast_teardown_manifest.json"
    _write(
        teardown_path,
        {
            "schema_version": "vast_teardown_manifest.v1",
            "generated_at": teardown_at.isoformat(),
            "status": "completed",
            "vast_instance_ids": [47482504],
            "continuing_spend_from_this_run": False,
        },
    )
    receipt = _paid_terminal_receipt(
        request=request,
        profile=profile,
        teardown_path=teardown_path,
    )
    if cross_runtime_receipt:
        receipt["receipt_digest_canonicalization"] = (
            LAUNCH_RECEIPT_DIGEST_CANONICALIZATION
        )
        receipt["receipt_digest"] = cross_runtime_canonical_digest(
            receipt, digest_field="receipt_digest"
        )
    _write(run_root / "launch_profile.json", profile)
    _write(run_root / "launch_receipt.json", receipt)
    # Keep this focused on closure evidence rather than WebApp callback setup.
    _write(run_root / "webapp_sync_succeeded.json", _webapp_sync_succeeded(receipt))
    guard_path = tmp_path / "gpu-spend-guard.json"
    observed_at = datetime.now(timezone.utc)
    _write(guard_path, _zero_guard(generated_at=observed_at - timedelta(seconds=1)))

    first = reconcile_launches(
        queue_root=tmp_path / "queue",
        state_root=tmp_path / "state",
        guard_report_path=guard_path,
        now=observed_at,
    )
    second = reconcile_launches(
        queue_root=tmp_path / "queue",
        state_root=tmp_path / "state",
        guard_report_path=guard_path,
        now=observed_at + timedelta(seconds=1),
    )

    assert first["status"] == "passed"
    assert first["terminal_provider_zero"][0]["status"] == "provider_zero_confirmed"
    assert first["terminal_provider_zero"][0]["provider_mutation_performed"] is False
    assert second["terminal_provider_zero"][0]["status"] == "provider_zero_receipt_retained"
    closure_path = run_root / "post_teardown_provider_zero_receipt.json"
    closure = json.loads(closure_path.read_text())
    assert closure["status"] == "provider_zero_confirmed"
    assert closure["provider_zero_verified"] is True
    assert closure["teardown_manifest"]["digest"] == _path_digest(teardown_path)
    assert closure["provider_zero_receipt_digest"] == canonical_digest(
        closure, digest_field="provider_zero_receipt_digest"
    )
    snapshot_path = Path(closure["independent_guard_snapshot"]["path"])
    snapshot = json.loads(snapshot_path.read_text())
    assert snapshot["guard"]["provider_zero_verified"] is True
    assert snapshot["source_guard_report_sha256"] == _path_digest(guard_path)
    assert len(list((run_root / "provider_zero_guard_snapshots").glob("*.json"))) == 1


def test_reconciler_never_retains_provider_zero_before_teardown_or_while_nonzero(
    tmp_path: Path,
) -> None:
    profile = _profile(tmp_path)
    request = _request(profile)
    run_root = tmp_path / "state" / request["launch_id"]
    teardown_at = datetime.now(timezone.utc)
    teardown_path = run_root / "vast_teardown_manifest.json"
    _write(
        teardown_path,
        {
            "schema_version": "vast_teardown_manifest.v1",
            "generated_at": teardown_at.isoformat(),
            "status": "completed",
            "continuing_spend_from_this_run": False,
        },
    )
    receipt = _paid_terminal_receipt(
        request=request,
        profile=profile,
        teardown_path=teardown_path,
    )
    _write(run_root / "launch_profile.json", profile)
    _write(run_root / "launch_receipt.json", receipt)
    _write(run_root / "webapp_sync_succeeded.json", _webapp_sync_succeeded(receipt))
    guard_path = tmp_path / "gpu-spend-guard.json"
    _write(
        guard_path,
        _zero_guard(generated_at=teardown_at - timedelta(seconds=1)),
    )

    predating = reconcile_launches(
        queue_root=tmp_path / "queue",
        state_root=tmp_path / "state",
        guard_report_path=guard_path,
        now=teardown_at + timedelta(seconds=2),
    )
    assert predating["status"] == "blocked"
    assert predating["terminal_provider_zero"][0]["status"] == "provider_zero_pending"
    assert "gpu_spend_guard_predates_teardown" in predating["terminal_provider_zero"][0]["blockers"]
    assert not (run_root / "post_teardown_provider_zero_receipt.json").exists()

    _write(
        guard_path,
        _zero_guard(generated_at=teardown_at + timedelta(seconds=1), live_instance_count=1),
    )
    nonzero = reconcile_launches(
        queue_root=tmp_path / "queue",
        state_root=tmp_path / "state",
        guard_report_path=guard_path,
        now=teardown_at + timedelta(seconds=2),
    )
    assert nonzero["status"] == "blocked"
    assert nonzero["terminal_provider_zero"][0]["provider_zero_confirmed"] is False
    assert "gpu_provider_nonzero" in nonzero["terminal_provider_zero"][0]["blockers"]
    assert not (run_root / "post_teardown_provider_zero_receipt.json").exists()


def test_reconciler_separates_preprovider_admission_rejection_from_teardown_gap(
    tmp_path: Path,
) -> None:
    profile = _profile(tmp_path)
    request = _request(profile)
    run_root = tmp_path / "state" / request["launch_id"]
    allocator_result = run_root / "allocator" / "result.json"
    _write(
        allocator_result,
        {
            "status": "blocked",
            "blockers": [
                "paid_resource_admission_has_blockers",
                "paid_resource_admission_not_admitted",
            ],
        },
    )
    receipt = {
        "schema_version": "task_evaluation_launch_receipt.v1",
        "status": "blocked",
        "launch_id": request["launch_id"],
        "run_id": request["run_id"],
        "request_digest": request["request_digest"],
        "launch_profile_digest": profile["profile_digest"],
        "execute_requested": True,
        # Dispatcher intent alone is not evidence that a provider API was
        # reached; the exact admission result below is the differentiator.
        "provider_mutation_attempted": True,
        "terminal_evidence": {
            "status": "blocked",
            "result": {
                "path": str(allocator_result.resolve()),
                "exists": True,
                "digest": _path_digest(allocator_result),
            },
            "artifacts": {},
        },
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    _write(run_root / "launch_profile.json", profile)
    _write(run_root / "launch_receipt.json", receipt)
    _write(run_root / "webapp_sync_succeeded.json", _webapp_sync_succeeded(receipt))
    guard_path = tmp_path / "gpu-spend-guard.json"
    _write(guard_path, _zero_guard(generated_at=datetime.now(timezone.utc)))

    reconciliation = reconcile_launches(
        queue_root=tmp_path / "queue",
        state_root=tmp_path / "state",
        guard_report_path=guard_path,
    )

    assert reconciliation["status"] == "passed"
    assert reconciliation["terminal_provider_zero"] == [{
        "launch_id": request["launch_id"],
        "status": "provider_zero_not_applicable_pre_provider_admission_blocked",
        "provider_zero_confirmed": None,
        "provider_zero_receipt_required": False,
        "provider_mutation_performed": False,
        "allocator_invoked": False,
        "automatic_retry_performed": False,
        "blockers": [
            "paid_resource_admission_has_blockers",
            "paid_resource_admission_not_admitted",
        ],
    }]
    assert not (run_root / "post_teardown_provider_zero_receipt.json").exists()


def test_reconciler_keeps_unknown_paid_terminal_without_teardown_pending(
    tmp_path: Path,
) -> None:
    profile = _profile(tmp_path)
    request = _request(profile)
    run_root = tmp_path / "state" / request["launch_id"]
    allocator_result = run_root / "allocator" / "result.json"
    _write(
        allocator_result,
        {"status": "blocked", "blockers": ["allocator_internal_failure"]},
    )
    receipt = {
        "schema_version": "task_evaluation_launch_receipt.v1",
        "status": "blocked",
        "launch_id": request["launch_id"],
        "run_id": request["run_id"],
        "request_digest": request["request_digest"],
        "launch_profile_digest": profile["profile_digest"],
        "execute_requested": True,
        "provider_mutation_attempted": True,
        "terminal_evidence": {
            "status": "blocked",
            "result": {
                "path": str(allocator_result.resolve()),
                "exists": True,
                "digest": _path_digest(allocator_result),
            },
            "artifacts": {},
        },
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    _write(run_root / "launch_profile.json", profile)
    _write(run_root / "launch_receipt.json", receipt)
    _write(run_root / "webapp_sync_succeeded.json", _webapp_sync_succeeded(receipt))
    guard_path = tmp_path / "gpu-spend-guard.json"
    _write(guard_path, _zero_guard(generated_at=datetime.now(timezone.utc)))

    reconciliation = reconcile_launches(
        queue_root=tmp_path / "queue",
        state_root=tmp_path / "state",
        guard_report_path=guard_path,
    )

    assert reconciliation["status"] == "blocked"
    assert reconciliation["terminal_provider_zero"][0]["status"] == "provider_zero_pending"
    assert "terminal_teardown_manifest_descriptor_missing" in reconciliation[
        "terminal_provider_zero"
    ][0]["blockers"]
    assert not (run_root / "post_teardown_provider_zero_receipt.json").exists()


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


def test_reconciler_retries_dry_terminal_receipt_sync_without_allocator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _profile(tmp_path)
    request = _request(profile)
    run_root = tmp_path / "state" / request["launch_id"]
    receipt = {
        "schema_version": "task_evaluation_launch_receipt.v1",
        "status": "dry_run_completed",
        "launch_id": request["launch_id"],
        "run_id": request["run_id"],
        "request_digest": request["request_digest"],
        "receipt_digest": "sha256:" + "e" * 64,
        "execute_requested": False,
        "provider_mutation_attempted": False,
    }
    _write(run_root / "launch_profile.json", profile)
    _write(run_root / "launch_receipt.json", receipt)
    _write(
        run_root / "webapp_sync_attempts" / "first.json",
        {
            "status": "failed",
            "attempt_number": 1,
            "reason": "timeouterror",
        },
    )
    sync_calls: list[dict] = []

    def sync_receipt(*, receipt: dict) -> dict:
        sync_calls.append(receipt)
        return {
            "schema_version": "task_evaluation_launch_webapp_sync_result.v1",
            "status": "succeeded",
            "launch_id": receipt["launch_id"],
            "run_id": receipt["run_id"],
            "request_digest": receipt["request_digest"],
            "receipt_digest": receipt["receipt_digest"],
            "response": {
                "launch_id": receipt["launch_id"],
                "run_id": receipt["run_id"],
                "request_digest": receipt["request_digest"],
                "receipt_digest": receipt["receipt_digest"],
            },
        }

    monkeypatch.setattr(webapp_sync_module, "sync_launch_receipt_to_webapp", sync_receipt)

    result = reconcile_launches(
        queue_root=tmp_path / "queue",
        state_root=tmp_path / "state",
        guard_report_path=tmp_path / "missing-guard.json",
    )

    assert sync_calls == [receipt]
    assert result["status"] == "passed"
    assert result["webapp_sync"] == [{
        "launch_id": request["launch_id"],
        "status": "webapp_sync_succeeded",
        "attempts": 2,
        "blockers": [],
        "webapp_record_bound": True,
        "website_trigger_proven": True,
        "provider_mutation_performed": False,
        "allocator_invoked": False,
        "automatic_retry_performed": False,
        "receipt": {
            "sync_result_digest": result["webapp_sync"][0]["receipt"][
                "sync_result_digest"
            ],
            "launch_id": request["launch_id"],
            "run_id": request["run_id"],
            "request_digest": request["request_digest"],
            "receipt_digest": receipt["receipt_digest"],
        },
    }]
    assert result["terminal_provider_zero"] == []
    succeeded = json.loads((run_root / "webapp_sync_succeeded.json").read_text())
    assert succeeded["attempt_number"] == 2
    assert succeeded["provider_mutation_performed"] is False
    replay = reconcile_launches(
        queue_root=tmp_path / "queue",
        state_root=tmp_path / "state",
        guard_report_path=tmp_path / "missing-guard.json",
    )
    assert replay["webapp_sync"] == result["webapp_sync"]
    assert sync_calls == [receipt]


def test_reconciler_retains_unmatched_webapp_404_without_retrying(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _profile(tmp_path)
    request = _request(profile)
    run_root = tmp_path / "state" / request["launch_id"]
    receipt = {
        "schema_version": "task_evaluation_launch_receipt.v1",
        "status": "dry_run_completed",
        "launch_id": request["launch_id"],
        "run_id": request["run_id"],
        "request_digest": request["request_digest"],
        "receipt_digest": "sha256:" + "f" * 64,
        "execute_requested": False,
        "provider_mutation_attempted": False,
    }
    _write(run_root / "launch_profile.json", profile)
    _write(run_root / "launch_receipt.json", receipt)
    sync_calls: list[dict] = []

    def sync_receipt(*, receipt: dict) -> dict:
        sync_calls.append(receipt)
        return {
            "schema_version": "task_evaluation_launch_webapp_sync_result.v1",
            "status": "failed",
            "reason": "http_error:404",
            "launch_id": receipt["launch_id"],
            "run_id": receipt["run_id"],
            "request_digest": receipt["request_digest"],
            "receipt_digest": receipt["receipt_digest"],
        }

    monkeypatch.setattr(webapp_sync_module, "sync_launch_receipt_to_webapp", sync_receipt)

    first = reconcile_launches(
        queue_root=tmp_path / "queue",
        state_root=tmp_path / "state",
        guard_report_path=tmp_path / "missing-guard.json",
    )
    second = reconcile_launches(
        queue_root=tmp_path / "queue",
        state_root=tmp_path / "state",
        guard_report_path=tmp_path / "missing-guard.json",
    )

    expected_row = {
        "launch_id": request["launch_id"],
        "status": "webapp_sync_terminal_unmatched",
        "attempts": 1,
        "blockers": ["webapp_launch_record_missing"],
        "webapp_record_bound": False,
        "website_trigger_proven": False,
        "provider_mutation_performed": False,
        "allocator_invoked": False,
        "automatic_retry_performed": False,
    }
    assert sync_calls == [receipt]
    assert first["status"] == "passed"
    assert second["status"] == "passed"
    assert first["webapp_sync"] == [expected_row]
    assert second["webapp_sync"] == [expected_row]
    assert first["allocator_invoked"] is False
    assert first["automatic_retry_performed"] is False
    unmatched = json.loads(
        (run_root / "webapp_sync_terminal_unmatched.json").read_text()
    )
    assert unmatched["receipt_digest"] == receipt["receipt_digest"]
    assert unmatched["sync_result_digest"].startswith("sha256:")
    assert unmatched["website_trigger_proven"] is False
    assert len(list((run_root / "webapp_sync_attempts").glob("*.json"))) == 1


def test_reconciler_rejects_unmatched_webapp_marker_without_bound_attempt(
    tmp_path: Path,
) -> None:
    profile = _profile(tmp_path)
    request = _request(profile)
    run_root = tmp_path / "state" / request["launch_id"]
    receipt = {
        "schema_version": "task_evaluation_launch_receipt.v1",
        "status": "dry_run_completed",
        "launch_id": request["launch_id"],
        "run_id": request["run_id"],
        "request_digest": request["request_digest"],
        "receipt_digest": "sha256:" + "d" * 64,
        "execute_requested": False,
        "provider_mutation_attempted": False,
    }
    unmatched = {
        "schema_version": "task_evaluation_launch_webapp_sync_terminal_unmatched.v1",
        "status": "terminal_unmatched",
        "launch_id": receipt["launch_id"],
        "run_id": receipt["run_id"],
        "request_digest": receipt["request_digest"],
        "receipt_digest": receipt["receipt_digest"],
        "sync_result_digest": "sha256:" + "e" * 64,
        "attempt_number": 1,
        "detected_at": "2026-08-11T00:00:00+00:00",
        "reason": "http_error:404",
        "webapp_record_bound": False,
        "website_trigger_proven": False,
        "provider_mutation_performed": False,
        "allocator_invoked": False,
        "automatic_retry_performed": False,
        "blockers": ["webapp_launch_record_missing"],
    }
    unmatched["terminal_unmatched_digest"] = canonical_digest(
        unmatched, digest_field="terminal_unmatched_digest"
    )
    _write(run_root / "launch_profile.json", profile)
    _write(run_root / "launch_receipt.json", receipt)
    _write(run_root / "webapp_sync_terminal_unmatched.json", unmatched)

    result = reconcile_launches(
        queue_root=tmp_path / "queue",
        state_root=tmp_path / "state",
        guard_report_path=tmp_path / "missing-guard.json",
    )

    assert result["status"] == "blocked"
    assert result["webapp_sync"] == [{
        "launch_id": request["launch_id"],
        "status": "webapp_sync_reconciliation_blocked",
        "blockers": ["webapp_sync_reconciliation_input_invalid"],
        "error_type": "FileNotFoundError",
        "provider_mutation_performed": False,
    }]


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
            **{
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
            },
            "required_authorization": {
                "max_spend_usd": profile["allocator"]["max_spend_usd"],
                "hard_ttl_seconds": profile["allocator"]["hard_ttl_seconds"],
            },
        }
    ]
    assert "allocator" not in catalog[0]
    assert "provider-launch-request" not in json.dumps(catalog)
    assert catalog[0]["required_authorization"]["max_spend_usd"] > 0
    assert catalog[0]["required_authorization"]["hard_ttl_seconds"] > 0

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


def test_source_commit_is_public_bound_and_propagated_to_terminal_receipt(
    tmp_path: Path,
) -> None:
    source_commit = "a" * 40
    profile = _profile(tmp_path)
    profile["source_commit"] = source_commit
    profile["profile_digest"] = canonical_digest(profile, digest_field="profile_digest")
    profile_dir, request_path = _write_profile_and_request(tmp_path, profile)
    catalog_path = tmp_path / "catalog.json"
    _write(catalog_path, [public_launch_profile_descriptor(profile)])

    request = json.loads(request_path.read_text(encoding="utf-8"))
    assert request["source_commit"] == source_commit
    assert validate_launch_request_against_public_catalog(
        request, catalog_path=catalog_path
    ) == []

    receipt = dispatch_launch_request(
        request_path=request_path,
        profile_dir=profile_dir,
        state_root=tmp_path / "state",
        public_catalog_path=catalog_path,
        allocator_runner=lambda argv: 0,
    )

    assert receipt["source_commit"] == source_commit
    assert receipt["status"] == "dry_run_completed"


def test_source_commit_fails_closed_at_every_public_launch_boundary(
    tmp_path: Path,
) -> None:
    profile = _profile(tmp_path)
    profile["source_commit"] = "a" * 40
    profile["profile_digest"] = canonical_digest(profile, digest_field="profile_digest")
    descriptor = public_launch_profile_descriptor(profile)
    catalog_path = tmp_path / "catalog.json"
    _write(catalog_path, [descriptor])
    request = _request(profile)

    malformed_profile = dict(profile, source_commit="A" * 40)
    malformed_profile["profile_digest"] = canonical_digest(
        malformed_profile, digest_field="profile_digest"
    )
    assert "launch_profile_source_commit_invalid" in validate_launch_profile(
        malformed_profile
    )

    malformed_descriptor = dict(descriptor, source_commit="main")
    assert "launch_profile_public_source_commit_invalid" in (
        validate_public_launch_profile_descriptor(malformed_descriptor)
    )

    mismatched_request = dict(request, source_commit="b" * 40)
    mismatched_request["request_digest"] = canonical_digest(
        mismatched_request, digest_field="request_digest"
    )
    assert validate_launch_request(mismatched_request) == []
    assert validate_launch_request_against_public_catalog(
        mismatched_request, catalog_path=catalog_path
    ) == ["launch_profile_public_catalog_source_commit_mismatch"]

    profile_dir = tmp_path / "profiles"
    _write(profile_dir / f"{profile['profile_id']}.json", profile)
    mismatched_path = tmp_path / "mismatched-request.json"
    _write(mismatched_path, mismatched_request)
    receipt = dispatch_launch_request(
        request_path=mismatched_path,
        profile_dir=profile_dir,
        state_root=tmp_path / "state",
    )
    assert "launch_source_commit_profile_binding_mismatch" in receipt["blockers"]


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


def test_public_catalog_accepts_more_than_one_hundred_bounded_profiles(
    tmp_path: Path,
) -> None:
    """Publishing the 101st immutable profile must not take every launch down."""

    profile = _profile(tmp_path)
    source = tmp_path / "staging" / "profile.json"
    _write(source, profile)
    publish_profiles(
        profile_paths=[source],
        profile_dir=tmp_path / "published",
        webapp_catalog_out=tmp_path / "catalog.json",
    )
    descriptor = json.loads((tmp_path / "catalog.json").read_text())[0]
    catalog = []
    for index in range(101):
        row = dict(descriptor)
        row["profile_id"] = f"profile-{index:03d}"
        row["profile_digest"] = "sha256:" + f"{index + 1:064x}"
        catalog.append(row)
    _write(tmp_path / "catalog-101.json", catalog)

    loaded = load_public_launch_profile_catalog(tmp_path / "catalog-101.json")

    assert len(loaded["profiles"]) == 101
    with pytest.raises(TaskEvaluationLaunchError, match="public_catalog_invalid"):
        load_public_launch_profile_catalog(
            tmp_path / "catalog-101.json", max_profiles=100
        )


def test_public_descriptor_requires_bounded_authorization_projection(
    tmp_path: Path,
) -> None:
    """The catalog must tell approvers the exact spend/TTL the profile demands."""

    profile = _profile(tmp_path)
    source = tmp_path / "staging" / "profile.json"
    _write(source, profile)
    publish_profiles(
        profile_paths=[source],
        profile_dir=tmp_path / "published",
        webapp_catalog_out=tmp_path / "catalog.json",
    )
    descriptor = json.loads((tmp_path / "catalog.json").read_text())[0]
    assert descriptor["required_authorization"] == {
        "max_spend_usd": profile["allocator"]["max_spend_usd"],
        "hard_ttl_seconds": profile["allocator"]["hard_ttl_seconds"],
    }
    assert validate_public_launch_profile_descriptor(descriptor) == []

    missing = {
        key: value for key, value in descriptor.items() if key != "required_authorization"
    }
    assert "launch_profile_public_descriptor_fields_invalid" in (
        validate_public_launch_profile_descriptor(missing)
    )

    smuggled = json.loads(json.dumps(descriptor))
    smuggled["required_authorization"]["argv"] = ["--execute"]
    assert "launch_profile_public_required_authorization_fields_invalid" in (
        validate_public_launch_profile_descriptor(smuggled)
    )

    nonpositive = json.loads(json.dumps(descriptor))
    nonpositive["required_authorization"]["max_spend_usd"] = 0
    assert "launch_profile_public_required_spend_invalid" in (
        validate_public_launch_profile_descriptor(nonpositive)
    )

    bool_ttl = json.loads(json.dumps(descriptor))
    bool_ttl["required_authorization"]["hard_ttl_seconds"] = True
    assert "launch_profile_public_required_ttl_invalid" in (
        validate_public_launch_profile_descriptor(bool_ttl)
    )


def test_prelaunch_skill_failure_blocks_before_canonical_allocator(tmp_path: Path) -> None:
    """A profile-bound skill failure is retained and can never become a GPU launch."""

    profile = _profile(tmp_path)
    plan = {
        "schema_version": "task_evaluation_prelaunch_skill_plan.v1",
        "program_id": "arm-decision-proof-v1",
        "plan_id": "prelaunch-skill-plan-001",
        "source_bundle": {
            "bundle_id": profile["source_bundle"]["bundle_id"],
            "digest": profile["source_bundle"]["digest"],
        },
        "steps": [
            {
                "step_id": "room-survey",
                "adapter": "interiorgs_room_survey",
                "structure_input": "structure",
                "labels_input": "labels",
                "scene_id": "scene-001",
                "target_ins_id": None,
                "timeout_seconds": 60,
            }
        ],
    }
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    plan_path = tmp_path / "prelaunch-plan.json"
    _write(plan_path, plan)
    plan_digest = _path_digest(plan_path)
    profile["immutable_inputs"].append(
        {"name": "prelaunch_skill_plan", "path": str(plan_path), "digest": plan_digest}
    )
    profile["prelaunch_skill_plan"] = {
        "plan_id": plan["plan_id"],
        "path": str(plan_path),
        "digest": plan_digest,
    }
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

    assert receipt["status"] == "blocked"
    assert calls == []
    assert receipt["provider_mutation_attempted"] is False
    assert receipt["prelaunch_skill_execution"]["status"] == "blocked"
    assert "prelaunch_skill_execution_blocked" in receipt["blockers"]
    assert (tmp_path / "state" / request["launch_id"] / "prelaunch_skills" / "execution.json").is_file()


def _standing_authorization(profile: dict, *, max_launches: int) -> dict:
    return {
        "schema_version": "task_evaluation_standing_launch_authorization.v1",
        "profile_id": profile["profile_id"],
        "profile_digest": profile["profile_digest"],
        "max_launches": max_launches,
        "max_total_spend_usd": 500.0,
        "expires_at": (datetime.now(timezone.utc) + timedelta(days=7)).isoformat(),
    }


def _require_one_use_standing_authorization(profile: dict) -> None:
    profile["standing_launch_authorization"] = {
        "schema_version": (
            "task_evaluation_standing_launch_authorization_requirement.v1"
        ),
        "required_for_live_execution": True,
        "maximum_launches": 1,
        "consumption_must_precede_allocator": True,
    }
    profile["profile_digest"] = canonical_digest(
        profile, digest_field="profile_digest"
    )


def test_a_standing_authorization_bound_to_one_launch_admits_only_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`max_launches` was declared and never counted.

    Nothing in production called ``record_launch``, so consumption stayed at
    zero forever and an authorization for one launch admitted every launch. The
    bound exists only if each admission is written down before the allocator
    runs.
    """

    monkeypatch.setenv(EXECUTE_ENV, "true")
    monkeypatch.setenv(SECRET_PROFILE_ID_ENV, "canonical-vast-adp")
    monkeypatch.delenv("BLUEPRINT_TASK_EVALUATION_STANDING_AUTHORIZATION_DIR", raising=False)
    profile = _profile(tmp_path)
    profile_dir = tmp_path / "profiles"
    _write(profile_dir / f"{profile['profile_id']}.json", profile)
    state_root = tmp_path / "control-plane" / "state"
    state_root.mkdir(parents=True)
    standing_dir = state_root.parent / "standing-authorizations"
    _write(
        standing_dir / f"{profile['profile_id']}.json",
        _standing_authorization(profile, max_launches=1),
    )
    calls: list[list[str]] = []

    receipts = []
    for index in (1, 2):
        request = _request(profile)
        request["launch_id"] = f"launch-standing-{index}"
        request["run_id"] = f"run-standing-{index}"
        request["idempotency_key"] = f"launch-standing-{index}"
        request["request_digest"] = canonical_digest(request, digest_field="request_digest")
        request_path = tmp_path / f"request-{index}.json"
        _write(request_path, request)
        receipts.append(
            dispatch_launch_request(
                request_path=request_path,
                profile_dir=profile_dir,
                state_root=state_root,
                execute=True,
                allocator_runner=lambda argv: calls.append(list(argv)) or 0,
            )
        )

    # The first launch is admitted with no copied launch id -- the point of the
    # standing authorization -- and the second is refused by its own bound.
    assert "execute_launch_id_required" not in receipts[0]["blockers"]
    assert len(calls) == 1, "the second launch must not reach the allocator"
    assert "execute_launch_id_required" in receipts[1]["blockers"]
    assert any(
        "standing_authorization_launch" in blocker for blocker in receipts[1]["blockers"]
    ), receipts[1]["blockers"]
    assert receipts[1]["provider_mutation_attempted"] is False


def test_one_use_standing_authority_is_atomic_across_distinct_website_launches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two launch ids racing at the website boundary may allocate only once."""

    monkeypatch.setenv(EXECUTE_ENV, "true")
    monkeypatch.setenv(SECRET_PROFILE_ID_ENV, "canonical-vast-adp")
    monkeypatch.delenv(
        "BLUEPRINT_TASK_EVALUATION_STANDING_AUTHORIZATION_DIR", raising=False
    )
    profile = _profile(tmp_path)
    _require_one_use_standing_authorization(profile)
    profile_dir = tmp_path / "profiles"
    _write(profile_dir / f"{profile['profile_id']}.json", profile)
    state_root = tmp_path / "control-plane" / "state"
    state_root.mkdir(parents=True)
    standing_dir = state_root.parent / "standing-authorizations"
    _write(
        standing_dir / f"{profile['profile_id']}.json",
        _standing_authorization(profile, max_launches=1),
    )
    request_paths: list[Path] = []
    for index in (1, 2):
        request = _request(profile)
        request["launch_id"] = f"joint-agent-website-launch-{index}"
        request["run_id"] = f"joint-agent-run-{index}"
        request["idempotency_key"] = request["launch_id"]
        request["request_digest"] = canonical_digest(
            request, digest_field="request_digest"
        )
        request_path = tmp_path / f"request-{index}.json"
        _write(request_path, request)
        request_paths.append(request_path)

    # Force both dispatcher processes past the old non-atomic admission read
    # before either reaches the new locked check-and-consume boundary.
    barrier = threading.Barrier(2)
    original_decision = dispatcher_module._standing_authorization_decision

    def synchronized_decision(*args, **kwargs):
        decision = original_decision(*args, **kwargs)
        barrier.wait(timeout=5)
        return decision

    monkeypatch.setattr(
        dispatcher_module, "_standing_authorization_decision", synchronized_decision
    )
    calls: list[list[str]] = []

    def allocator(argv: list[str]) -> int:
        assert len(
            list(
                (standing_dir / "consumed" / profile["profile_id"]).glob(
                    "*.json"
                )
            )
        ) == 1, "authority must be durable before allocator staging/allocation"
        calls.append(list(argv))
        return 0

    def dispatch(path: Path) -> dict:
        return dispatch_launch_request(
            request_path=path,
            profile_dir=profile_dir,
            state_root=state_root,
            execute=True,
            allocator_runner=allocator,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        receipts = list(executor.map(dispatch, request_paths))

    assert len(calls) == 1
    assert sum(receipt["allocator_invoked"] is True for receipt in receipts) == 1
    assert all(receipt["provider_mutation_attempted"] is False for receipt in receipts)
    refused = next(receipt for receipt in receipts if receipt["allocator_invoked"] is False)
    assert "standing_authorization_consumption_not_recorded" in refused["blockers"]
    assert "standing_authorization_launches_exhausted" in refused["blockers"]
    consumption_records = list(
        (standing_dir / "consumed" / profile["profile_id"]).glob("*.json")
    )
    assert len(consumption_records) == 1


def test_joint_one_use_requirement_cannot_be_bypassed_by_execute_launch_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(EXECUTE_ENV, "true")
    monkeypatch.setenv(SECRET_PROFILE_ID_ENV, "canonical-vast-adp")
    monkeypatch.delenv(
        "BLUEPRINT_TASK_EVALUATION_STANDING_AUTHORIZATION_DIR", raising=False
    )
    profile = _profile(tmp_path)
    _require_one_use_standing_authorization(profile)
    profile_dir = tmp_path / "profiles"
    _write(profile_dir / f"{profile['profile_id']}.json", profile)
    request = _request(profile)
    request_path = tmp_path / "request.json"
    _write(request_path, request)
    calls: list[list[str]] = []

    receipt = dispatch_launch_request(
        request_path=request_path,
        profile_dir=profile_dir,
        state_root=tmp_path / "state",
        execute=True,
        execute_launch_id=request["launch_id"],
        allocator_runner=lambda argv: calls.append(list(argv)) or 0,
    )

    assert receipt["provider_mutation_attempted"] is False
    assert "one_use_standing_authorization_required" in receipt["blockers"]
    assert calls == []


def test_an_unconfigured_host_still_finds_its_standing_authorizations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The deployed control plane never set the directory variable, so the
    capability could not admit anything there."""

    monkeypatch.setenv(EXECUTE_ENV, "true")
    monkeypatch.setenv(SECRET_PROFILE_ID_ENV, "canonical-vast-adp")
    monkeypatch.delenv("BLUEPRINT_TASK_EVALUATION_STANDING_AUTHORIZATION_DIR", raising=False)
    profile = _profile(tmp_path)
    profile_dir = tmp_path / "profiles"
    _write(profile_dir / f"{profile['profile_id']}.json", profile)
    state_root = tmp_path / "control-plane" / "state"
    state_root.mkdir(parents=True)
    _write(
        state_root.parent / "standing-authorizations" / f"{profile['profile_id']}.json",
        _standing_authorization(profile, max_launches=3),
    )
    request = _request(profile)
    request_path = tmp_path / "request.json"
    _write(request_path, request)
    calls: list[list[str]] = []

    receipt = dispatch_launch_request(
        request_path=request_path,
        profile_dir=profile_dir,
        state_root=state_root,
        execute=True,
        allocator_runner=lambda argv: calls.append(list(argv)) or 0,
    )

    assert "execute_launch_id_required" not in receipt["blockers"]
    assert len(calls) == 1


def test_an_unset_terminal_path_field_is_not_recorded_as_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`Path("").resolve()` is the process working directory, so a result that
    never set `teardown_manifest_path` used to produce a descriptor naming the
    release checkout -- evidence a reader could mistake for a real artifact
    that had merely gone missing."""

    monkeypatch.setenv(EXECUTE_ENV, "true")
    monkeypatch.setenv(SECRET_PROFILE_ID_ENV, "canonical-vast-adp")
    profile = _profile(tmp_path)
    request = _request(profile)
    profile_dir = tmp_path / "profiles"
    _write(profile_dir / f"{profile['profile_id']}.json", profile)
    request_path = tmp_path / "request.json"
    _write(request_path, request)
    result_path = Path(profile["terminal_contract"]["result_path"])

    def _runner(argv: list[str]) -> int:
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps({"status": "blocked", "continuing_spend_from_this_run": False}),
            encoding="utf-8",
        )
        return 2

    receipt = dispatch_launch_request(
        request_path=request_path,
        profile_dir=profile_dir,
        state_root=tmp_path / "state",
        execute=True,
        execute_launch_id=request["launch_id"],
        allocator_runner=_runner,
    )

    artifacts = receipt["terminal_evidence"]["artifacts"]
    for field in ("teardown_manifest_path", "artifact_manifest_path"):
        assert artifacts[field] == {"path": None, "exists": False, "digest": None}
        assert f"allocator_terminal_artifact_missing:{field}" in receipt["blockers"]


def test_a_standing_authorization_admits_a_queued_launch_with_no_env_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The queue refused for a missing launch id before `dispatch_launch_request`
    -- the only place that reads a standing authorization -- was ever called. So
    every paid run still needed the hand-edited env var the standing
    authorization exists to replace."""

    monkeypatch.setenv(EXECUTE_ENV, "true")
    monkeypatch.setenv(SECRET_PROFILE_ID_ENV, "canonical-vast-adp")
    monkeypatch.delenv("BLUEPRINT_TASK_EVALUATION_STANDING_AUTHORIZATION_DIR", raising=False)
    profile = _profile(tmp_path)
    profile_dir = tmp_path / "profiles"
    _write(profile_dir / f"{profile['profile_id']}.json", profile)
    state_root = tmp_path / "control-plane" / "state"
    state_root.mkdir(parents=True)
    _write(
        state_root.parent / "standing-authorizations" / f"{profile['profile_id']}.json",
        _standing_authorization(profile, max_launches=2),
    )
    queue_root = tmp_path / "queue"
    request = _request(profile)
    _write(queue_root / "pending" / f"{request['launch_id']}-abcd1234.json", request)
    calls: list[list[str]] = []

    report = process_launch_queue(
        queue_root=queue_root,
        profile_dir=profile_dir,
        state_root=state_root,
        execute=True,
        allocator_runner=lambda argv: calls.append(list(argv)) or 0,
    )

    assert report["processed_count"] == 1
    assert len(calls) == 1
    assert "--execute" in calls[0]


def test_without_either_authority_the_queue_still_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(EXECUTE_ENV, "true")
    monkeypatch.delenv("BLUEPRINT_TASK_EVALUATION_STANDING_AUTHORIZATION_DIR", raising=False)
    profile = _profile(tmp_path)
    profile_dir = tmp_path / "profiles"
    _write(profile_dir / f"{profile['profile_id']}.json", profile)
    state_root = tmp_path / "control-plane" / "state"
    state_root.mkdir(parents=True)
    queue_root = tmp_path / "queue"
    request = _request(profile)
    _write(queue_root / "pending" / f"{request['launch_id']}-abcd1234.json", request)
    calls: list[list[str]] = []

    report = process_launch_queue(
        queue_root=queue_root,
        profile_dir=profile_dir,
        state_root=state_root,
        execute=True,
        allocator_runner=lambda argv: calls.append(list(argv)) or 0,
    )

    assert report["status"] == "blocked"
    assert report["blockers"] == ["execute_launch_id_required"]
    assert calls == []


def test_an_armed_launch_id_still_scopes_the_window_to_itself(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stale armed id silently filtered every newer request out of the queue,
    which is how a fresh launch sat pending while the dispatcher reported
    `processed_count: 0`. Scoping is still correct when an id is armed -- it
    just must not be the only way in."""

    monkeypatch.setenv(EXECUTE_ENV, "true")
    monkeypatch.setenv(SECRET_PROFILE_ID_ENV, "canonical-vast-adp")
    monkeypatch.delenv("BLUEPRINT_TASK_EVALUATION_STANDING_AUTHORIZATION_DIR", raising=False)
    profile = _profile(tmp_path)
    profile_dir = tmp_path / "profiles"
    _write(profile_dir / f"{profile['profile_id']}.json", profile)
    state_root = tmp_path / "control-plane" / "state"
    state_root.mkdir(parents=True)
    _write(
        state_root.parent / "standing-authorizations" / f"{profile['profile_id']}.json",
        _standing_authorization(profile, max_launches=5),
    )
    queue_root = tmp_path / "queue"
    request = _request(profile)
    _write(queue_root / "pending" / f"{request['launch_id']}-abcd1234.json", request)
    calls: list[list[str]] = []

    report = process_launch_queue(
        queue_root=queue_root,
        profile_dir=profile_dir,
        state_root=state_root,
        execute=True,
        execute_launch_id="some-other-launch-id",
        allocator_runner=lambda argv: calls.append(list(argv)) or 0,
    )

    assert report["processed_count"] == 0
    assert calls == []


def test_a_terminal_stale_launch_id_does_not_strand_a_standing_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A prior paid window remained active in the production EnvironmentFile.

    Its launch was already terminal-blocked, but the queue filtered every
    later website request to that old identifier and repeatedly reported zero
    processed messages. A terminal one-shot scope must not override the newer
    request's own digest-bound standing authorization.
    """

    monkeypatch.setenv(EXECUTE_ENV, "true")
    monkeypatch.setenv(SECRET_PROFILE_ID_ENV, "canonical-vast-adp")
    monkeypatch.delenv("BLUEPRINT_TASK_EVALUATION_STANDING_AUTHORIZATION_DIR", raising=False)
    profile = _profile(tmp_path)
    profile_dir = tmp_path / "profiles"
    _write(profile_dir / f"{profile['profile_id']}.json", profile)
    state_root = tmp_path / "control-plane" / "state"
    state_root.mkdir(parents=True)
    _write(
        state_root.parent / "standing-authorizations" / f"{profile['profile_id']}.json",
        _standing_authorization(profile, max_launches=1),
    )
    queue_root = tmp_path / "queue"
    stale_launch_id = "launch-terminal-stale"
    _write(queue_root / "blocked" / f"{stale_launch_id}-deadbeef.json", {})
    request = _request(profile)
    _write(queue_root / "pending" / f"{request['launch_id']}-abcd1234.json", request)
    calls: list[list[str]] = []

    report = process_launch_queue(
        queue_root=queue_root,
        profile_dir=profile_dir,
        state_root=state_root,
        execute=True,
        execute_launch_id=stale_launch_id,
        allocator_runner=lambda argv: calls.append(list(argv)) or 1,
    )

    assert report["processed_count"] == 1
    assert report["execute_launch_id"] == ""
    assert report["ignored_terminal_execute_launch_id"] == stale_launch_id
    assert len(calls) == 1
    binding = json.loads(
        (state_root / request["launch_id"] / "launch_binding.json").read_text(
            encoding="utf-8"
        )
    )
    assert binding["execute_launch_id"] == ""
