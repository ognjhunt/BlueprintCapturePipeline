from pathlib import Path
import hashlib

import pytest

from blueprint_pipeline.task_evaluation_launch_dispatcher import (
    CANONICAL_ALLOCATOR_ENTRYPOINT,
    SECRET_PROFILE_ID_ENV,
    canonical_digest,
)
from blueprint_pipeline.task_evaluation_launch_supervisor import (
    LaunchSupervisorRecommendation,
    build_supervisor_snapshot,
    run_launch_supervisor,
)
from blueprint_pipeline.task_evaluation_supervisor.agents_sdk import (
    AgentsSDKInvocationResult,
)


def _write(path: Path, value: dict) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _profile(tmp_path: Path) -> dict:
    source_manifest = tmp_path / "source-bundle-manifest.json"
    evaluation_spec = tmp_path / "evaluation-run-spec.json"
    source_manifest.write_text('{"scene":"840313"}\n', encoding="utf-8")
    evaluation_spec.write_text('{"spec":"frozen"}\n', encoding="utf-8")
    profile = {
        "schema_version": "task_evaluation_launch_profile.v1",
        "profile_id": "interiorgs-sage-franka-001",
        "program_id": "arm-decision-proof-v1",
        "source_bundle": {
            "bundle_id": "scene-001",
            "source_kind": "interiorgs_sage",
            "uri": "gs://blueprint-runs/scene-001.json",
            "digest": "sha256:" + "a" * 64,
        },
        "evaluation_run_spec": {
            "uri": "gs://blueprint-runs/evaluation-run-spec.json",
            "digest": "sha256:" + "b" * 64,
        },
        "immutable_inputs": [
            {
                "name": "source_bundle_manifest",
                "path": str(source_manifest.resolve()),
                "digest": "sha256:" + hashlib.sha256(source_manifest.read_bytes()).hexdigest(),
            },
            {
                "name": "evaluation_run_spec",
                "path": str(evaluation_spec.resolve()),
                "digest": "sha256:" + hashlib.sha256(evaluation_spec.read_bytes()).hexdigest(),
            },
        ],
        "allocator": {
            "entrypoint": CANONICAL_ALLOCATOR_ENTRYPOINT,
            "subcommand": "gpu-canary",
            "argv": ["--provider-launch-request", str(tmp_path / "provider.json")],
            "max_spend_usd": 2.0,
            "hard_ttl_seconds": 7200,
            "retry_cap": 0,
        },
        "runtime_environment": {},
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
            "result_path": str(tmp_path / "result.json"),
            "success_statuses": ["completed"],
            "required_values": {"continuing_spend_from_this_run": False},
            "required_path_fields": ["teardown_manifest_path", "artifact_manifest_path"],
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


class _Invoker:
    def __init__(self, profile_id: str) -> None:
        self.profile_id = profile_id
        self.spec = None

    def invoke(self, spec, _input_text: str) -> AgentsSDKInvocationResult:
        self.spec = spec
        return AgentsSDKInvocationResult(
            output=LaunchSupervisorRecommendation(
                disposition="recommend_profile",
                summary="This is the only deterministically admissible profile.",
                recommended_profile_id=self.profile_id,
            ),
            provider="openai",
            model=spec.model,
            sdk_version="test",
            latency_seconds=0.01,
            usage={},
            cost_usd=None,
            cost_status="test",
        )


def _snapshot(tmp_path: Path) -> tuple[dict, dict]:
    profile = _profile(tmp_path)
    profile_dir = tmp_path / "profiles"
    _write(profile_dir / f"{profile['profile_id']}.json", profile)
    guard_path = tmp_path / "guard.json"
    _write(
        guard_path,
        {
            "schema_version": "gpu_spend_guard.v1",
            "status": "passed",
            "generated_at": "2026-08-10T12:00:00+00:00",
            "live_instance_count": 0,
            "total_burn_per_hour_usd": 0,
            "inventory_results": [{"provider": "vast", "status": "succeeded"}],
            "spend_admission_lock": {"admission_allowed": True},
            "blockers": [],
        },
    )
    return profile, build_supervisor_snapshot(
        profile_dir=profile_dir,
        queue_root=tmp_path / "queue",
        state_root=tmp_path / "state",
        guard_report_path=guard_path,
    )


def test_agents_sdk_supervisor_has_no_tools_or_mutation_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(SECRET_PROFILE_ID_ENV, "canonical-vast-adp")
    profile, snapshot = _snapshot(tmp_path)
    invoker = _Invoker(profile["profile_id"])

    result = run_launch_supervisor(
        snapshot=snapshot,
        output_dir=tmp_path / "supervision",
        invoker=invoker,
        enabled=True,
    )

    assert result["status"] == "completed"
    assert result["recommendation"]["recommended_profile_id"] == profile["profile_id"]
    assert result["tool_count"] == 0
    assert result["provider_mutation_performed"] is False
    assert result["allocator_invoked"] is False
    assert result["automatic_retry_performed"] is False
    assert invoker.spec.tool_bindings == ()


def test_passed_guard_with_no_blockers_stays_clean_in_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(SECRET_PROFILE_ID_ENV, "canonical-vast-adp")

    _, snapshot = _snapshot(tmp_path)

    assert snapshot["guard"]["blockers"] == []
    assert snapshot["admissible_profile_ids"] == ["interiorgs-sage-franka-001"]


def test_missing_guard_fails_closed_without_becoming_agent_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(SECRET_PROFILE_ID_ENV, "canonical-vast-adp")
    profile = _profile(tmp_path)
    profile_dir = tmp_path / "profiles"
    _write(profile_dir / f"{profile['profile_id']}.json", profile)

    snapshot = build_supervisor_snapshot(
        profile_dir=profile_dir,
        queue_root=tmp_path / "queue",
        state_root=tmp_path / "state",
        guard_report_path=tmp_path / "missing-guard.json",
    )

    assert snapshot["admissible_profile_ids"] == []
    assert snapshot["guard"]["blockers"] == [
        "gpu_spend_guard_report_unavailable"
    ]
    assert "gpu_spend_guard_report_unavailable" in snapshot["profiles"][0][
        "blockers"
    ]


def test_supervisor_rejects_non_admissible_profile_recommendation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(SECRET_PROFILE_ID_ENV, "canonical-vast-adp")
    _, snapshot = _snapshot(tmp_path)

    result = run_launch_supervisor(
        snapshot=snapshot,
        output_dir=tmp_path / "supervision",
        invoker=_Invoker("not-in-admissible-set"),
        enabled=True,
    )

    assert result["status"] == "blocked"
    assert "agent_recommended_non_admissible_profile" in result["blockers"]
    assert result["provider_mutation_performed"] is False


def test_supervisor_excludes_dry_only_profile(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    profile["execution_admission"]["live_enabled"] = False
    profile["execution_admission"]["blockers"] = ["positive_control_not_passed"]
    profile["profile_digest"] = canonical_digest(profile, digest_field="profile_digest")
    profile_dir = tmp_path / "profiles"
    _write(profile_dir / f"{profile['profile_id']}.json", profile)
    guard_path = tmp_path / "guard.json"
    _write(
        guard_path,
        {
            "status": "passed",
            "live_instance_count": 0,
            "inventory_results": [{"provider": "vast", "status": "succeeded"}],
            "spend_admission_lock": {"admission_allowed": True},
        },
    )

    snapshot = build_supervisor_snapshot(
        profile_dir=profile_dir,
        queue_root=tmp_path / "queue",
        state_root=tmp_path / "state",
        guard_report_path=guard_path,
    )

    assert snapshot["admissible_profile_ids"] == []
    assert "launch_profile_live_execution_disabled" in snapshot["profiles"][0]["blockers"]


def test_disabled_supervisor_makes_no_agent_call(tmp_path: Path) -> None:
    profile, snapshot = _snapshot(tmp_path)
    invoker = _Invoker(profile["profile_id"])

    result = run_launch_supervisor(
        snapshot=snapshot,
        output_dir=tmp_path / "supervision",
        invoker=invoker,
        enabled=False,
    )

    assert result == {
        "schema_version": "task_evaluation_launch_supervision.v1",
        "status": "disabled",
        "snapshot_digest": snapshot["snapshot_digest"],
        "agent_invoked": False,
        "provider_mutation_performed": False,
        "automatic_retry_performed": False,
    }
    assert invoker.spec is None
