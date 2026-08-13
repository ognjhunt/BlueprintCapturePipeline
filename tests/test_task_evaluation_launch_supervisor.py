import hashlib
import json
from pathlib import Path

import pytest

from blueprint_pipeline.task_evaluation_launch_dispatcher import (
    CANONICAL_ALLOCATOR_ENTRYPOINT,
    SECRET_PROFILE_ID_ENV,
    canonical_digest,
    public_launch_profile_descriptor,
)
from blueprint_pipeline.task_evaluation_launch_supervisor import (
    DEFAULT_SUPERVISOR_SNAPSHOT_MAX_BYTES,
    LaunchSupervisorRecommendation,
    SUPERVISOR_SNAPSHOT_INPUT_CEILING_BLOCKER,
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
            "provider_zero_verified": True,
            "provider_zero": {"blockers": []},
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
    assert snapshot["guard"]["provider_zero_verified"] is True
    assert snapshot["admissible_profile_ids"] == ["interiorgs-sage-franka-001"]


def test_supervisor_considers_only_profiles_in_the_published_catalog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(SECRET_PROFILE_ID_ENV, "canonical-vast-adp")
    profile, _ = _snapshot(tmp_path)
    shadow = json.loads(json.dumps(profile))
    shadow["profile_id"] = "interiorgs-sage-franka-unpublished"
    shadow["profile_digest"] = canonical_digest(shadow, digest_field="profile_digest")
    profile_dir = tmp_path / "profiles"
    _write(profile_dir / f"{shadow['profile_id']}.json", shadow)
    catalog_path = tmp_path / "published-catalog.json"
    catalog_path.write_text(
        json.dumps([public_launch_profile_descriptor(profile)], sort_keys=True) + "\n",
        encoding="utf-8",
    )

    snapshot = build_supervisor_snapshot(
        profile_dir=profile_dir,
        queue_root=tmp_path / "queue",
        state_root=tmp_path / "state",
        guard_report_path=tmp_path / "guard.json",
        public_catalog_path=catalog_path,
    )

    assert snapshot["profile_catalog"] == {
        "status": "verified",
        "published_profile_count": 1,
        "blockers": [],
    }
    assert [row["profile_id"] for row in snapshot["profiles"]] == [profile["profile_id"]]
    assert snapshot["admissible_profile_ids"] == [profile["profile_id"]]


def test_supervisor_fails_closed_when_the_configured_catalog_is_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(SECRET_PROFILE_ID_ENV, "canonical-vast-adp")
    _, _snapshot_value = _snapshot(tmp_path)
    catalog_path = tmp_path / "invalid-published-catalog.json"
    catalog_path.write_text("{}\n", encoding="utf-8")

    snapshot = build_supervisor_snapshot(
        profile_dir=tmp_path / "profiles",
        queue_root=tmp_path / "queue",
        state_root=tmp_path / "state",
        guard_report_path=tmp_path / "guard.json",
        public_catalog_path=catalog_path,
    )

    assert snapshot["profile_catalog"] == {
        "status": "blocked",
        "published_profile_count": 0,
        "blockers": ["launch_profile_public_catalog_invalid"],
    }
    assert snapshot["profiles"] == []
    assert snapshot["admissible_profile_ids"] == []


def test_supervisor_blocks_a_guard_that_explicitly_reports_nonzero_provider_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(SECRET_PROFILE_ID_ENV, "canonical-vast-adp")
    profile, _ = _snapshot(tmp_path)
    guard_path = tmp_path / "guard.json"
    guard = json.loads(guard_path.read_text(encoding="utf-8"))
    guard["provider_zero_verified"] = False
    guard["provider_zero"] = {"blockers": ["provider_zero_live_instances_observed"]}
    _write(guard_path, guard)

    snapshot = build_supervisor_snapshot(
        profile_dir=tmp_path / "profiles",
        queue_root=tmp_path / "queue",
        state_root=tmp_path / "state",
        guard_report_path=guard_path,
    )

    assert snapshot["admissible_profile_ids"] == []
    assert snapshot["guard"]["provider_zero_verified"] is False
    assert snapshot["guard"]["provider_zero_blockers"] == [
        "provider_zero_live_instances_observed"
    ]
    assert "gpu_provider_zero_not_verified" in snapshot["profiles"][0]["blockers"]


def test_supervisor_snapshot_exposes_unmatched_webapp_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(SECRET_PROFILE_ID_ENV, "canonical-vast-adp")
    profile, _ = _snapshot(tmp_path)
    run_root = tmp_path / "state" / "historical-launch"
    receipt = {
        "launch_id": "historical-launch",
        "request_digest": "sha256:" + "c" * 64,
        "status": "blocked",
        "provider_mutation_attempted": False,
    }
    _write(run_root / "launch_receipt.json", receipt)
    _write(
        run_root / "webapp_sync_terminal_unmatched.json",
        {
            "status": "terminal_unmatched",
            "webapp_record_bound": False,
            "website_trigger_proven": False,
            "blockers": ["webapp_launch_record_missing"],
        },
    )

    snapshot = build_supervisor_snapshot(
        profile_dir=tmp_path / "profiles",
        queue_root=tmp_path / "queue",
        state_root=tmp_path / "state",
        guard_report_path=tmp_path / "guard.json",
    )

    assert profile["profile_id"] in snapshot["admissible_profile_ids"]
    assert snapshot["terminal_launches"] == [{
        "launch_id": "historical-launch",
        "status": "blocked",
        "request_digest": "sha256:" + "c" * 64,
        "blockers": [],
        "provider_mutation_attempted": False,
        "terminal_evidence_status": None,
        "webapp_sync_status": "terminal_unmatched",
        "webapp_record_bound": False,
        "website_trigger_proven": False,
        "webapp_sync_blockers": ["webapp_launch_record_missing"],
    }]


def test_supervisor_snapshot_exposes_bound_website_trigger_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(SECRET_PROFILE_ID_ENV, "canonical-vast-adp")
    profile, _ = _snapshot(tmp_path)
    run_root = tmp_path / "state" / "website-launch"
    receipt = {
        "launch_id": "website-launch",
        "run_id": "website-run",
        "request_digest": "sha256:" + "c" * 64,
        "receipt_digest": "sha256:" + "d" * 64,
        "status": "completed",
        "provider_mutation_attempted": False,
    }
    attempt = {
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
        "attempt_number": 1,
        "attempted_at": "2026-08-13T14:00:00+00:00",
        "provider_mutation_performed": False,
    }
    attempt["sync_result_digest"] = canonical_digest(
        attempt, digest_field="sync_result_digest"
    )
    _write(run_root / "launch_receipt.json", receipt)
    _write(run_root / "webapp_sync_succeeded.json", attempt)

    snapshot = build_supervisor_snapshot(
        profile_dir=tmp_path / "profiles",
        queue_root=tmp_path / "queue",
        state_root=tmp_path / "state",
        guard_report_path=tmp_path / "guard.json",
    )

    assert profile["profile_id"] in snapshot["admissible_profile_ids"]
    terminal = snapshot["terminal_launches"][0]
    assert terminal["webapp_sync_status"] == "webapp_sync_succeeded"
    assert terminal["webapp_record_bound"] is True
    assert terminal["website_trigger_proven"] is True
    assert terminal["webapp_sync_blockers"] == []
    assert terminal["webapp_sync_receipt"] == {
        "sync_result_digest": attempt["sync_result_digest"],
        "launch_id": receipt["launch_id"],
        "run_id": receipt["run_id"],
        "request_digest": receipt["request_digest"],
        "receipt_digest": receipt["receipt_digest"],
    }


def test_supervisor_snapshot_bounds_old_terminal_history_before_live_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(SECRET_PROFILE_ID_ENV, "canonical-vast-adp")
    _snapshot(tmp_path)
    state_root = tmp_path / "state"
    for index in range(30):
        _write(
            state_root / f"launch-{index:03d}" / "launch_receipt.json",
            {
                "launch_id": f"launch-{index:03d}",
                "request_digest": "sha256:" + f"{index:064x}",
                "status": "blocked",
                "provider_mutation_attempted": False,
                "blockers": [f"historical-{index}-" + "x" * 2_500],
            },
        )

    snapshot = build_supervisor_snapshot(
        profile_dir=tmp_path / "profiles",
        queue_root=tmp_path / "queue",
        state_root=state_root,
        guard_report_path=tmp_path / "guard.json",
    )

    assert len(json.dumps(snapshot, sort_keys=True).encode("utf-8")) <= (
        DEFAULT_SUPERVISOR_SNAPSHOT_MAX_BYTES
    )
    history = snapshot["terminal_history"]
    assert history["selection"] == "lexicographically_latest_launch_receipts"
    assert history["total_count"] == 30
    assert history["included_count"] == len(snapshot["terminal_launches"])
    assert history["omitted_count"] == 30 - len(snapshot["terminal_launches"])
    assert history["input_byte_ceiling"] == DEFAULT_SUPERVISOR_SNAPSHOT_MAX_BYTES
    assert history["status"] == "bounded"
    assert history["omitted_count"] > 0
    assert history["omitted_terminal_rows_digest"].startswith("sha256:")
    assert snapshot["terminal_launches"][-1]["launch_id"] == "launch-029"
    assert all(row["launch_id"] != "launch-000" for row in snapshot["terminal_launches"])


def test_supervisor_refuses_an_uncompactable_snapshot_without_an_agent_call(
    tmp_path: Path,
) -> None:
    invoker = _Invoker("interiorgs-sage-franka-001")
    result = run_launch_supervisor(
        snapshot={
            "snapshot_digest": "sha256:" + "e" * 64,
            "terminal_history": {"status": "input_ceiling_exceeded"},
        },
        output_dir=tmp_path / "supervision",
        invoker=invoker,
        enabled=True,
    )

    assert result["status"] == "blocked"
    assert result["blockers"] == [SUPERVISOR_SNAPSHOT_INPUT_CEILING_BLOCKER]
    assert result["tool_count"] == 0
    assert result["provider_mutation_performed"] is False
    assert invoker.spec is None


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
