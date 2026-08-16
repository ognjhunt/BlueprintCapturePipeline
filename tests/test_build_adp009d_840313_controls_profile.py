from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from blueprint_pipeline.adp009d_physics_backend_comparison import (
    build_backend_profile,
    build_newton_canary_admission,
)
from blueprint_pipeline.spend_admission_lock import build_spend_admission_lock
from blueprint_pipeline.task_evaluation_launch_dispatcher import (
    CANONICAL_ALLOCATOR_ENTRYPOINT,
    EXECUTE_ENV,
    LAUNCH_RUN_ROOT_PLACEHOLDER,
    SECRET_PROFILE_ID_ENV,
    canonical_digest,
    dispatch_launch_request,
    process_launch_queue,
    stage_launch_request,
    validate_launch_profile,
    validate_public_launch_profile_descriptor,
)
from scripts import build_adp009d_840313_controls_profile as builder
from scripts.publish_task_evaluation_launch_profiles import publish_profiles

COMMIT = "a" * 40


def _guard(path: Path, *, status: str = "passed", instances: list | None = None) -> Path:
    rows = instances or []
    path.write_text(
        json.dumps({"status": status, "live_instance_count": len(rows), "instances": rows}),
        encoding="utf-8",
    )
    return path


def _newton_admission(env: dict, tmp_path: Path) -> Path:
    now = datetime.now(timezone.utc)
    inventory = [
        {
            "provider": provider,
            "status": "succeeded",
            "required": True,
            "credential_present": True,
            "row_count": 0,
            "blockers": [],
        }
        for provider in ("runpod", "vast", "digitalocean")
    ]
    guard = {
        "schema_version": "gpu_spend_guard.v1",
        "status": "passed",
        "generated_at": now.isoformat(),
        "provider_zero_verified": True,
        "live_instance_count": 0,
        "total_burn_per_hour_usd": 0.0,
        "blockers": [],
        "inventory_results": inventory,
        "instances": [],
    }
    env["guard"].write_text(json.dumps(guard), encoding="utf-8")
    spend_lock = build_spend_admission_lock(
        fleet_budget={"status": "passed", "total_spend_usd": 0.0, "blockers": []},
        billing_reconciliation={
            "status": "reconciled",
            "required": True,
            "billing_export_schema_version": "blueprint.provider_billing_export.v1",
            "billing_export_sha256": "sha256:" + "a" * 64,
            "billing_export_mode_octal": "0600",
            "generated_at": now.isoformat(),
            "currency": "USD",
            "scope": "blueprint_beta_100_user_cohort",
            "provider_totals_usd": {
                "runpod": 0.0,
                "vast": 0.0,
                "digitalocean": 0.0,
            },
            "blockers": [],
        },
        instances=[],
        reap_results=[],
        inventory_results=inventory,
        override_path=None,
        now=now,
    )
    admission = build_newton_canary_admission(
        authorization_evidence_ref="goal:scene-840920-newton-controls",
        spend_admission_lock=spend_lock,
        provider_zero_precheck=guard,
        max_spend_usd=2.0,
        hard_ttl_seconds=5400,
        issued_at=now,
    )
    assert admission["backend_profile_digest"] == build_backend_profile("newton")["profile_digest"]
    path = tmp_path / "newton-admission.json"
    path.write_text(json.dumps(admission), encoding="utf-8")
    return path


def _release(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    """Build one controls profile against stubbed source verification.

    The digest and checkout verifications belong to the dry builder and are
    covered there; this exercises the controls contract on top of them.
    """
    repo = tmp_path / "repo"
    manifests = repo / builder.MANIFEST_RELATIVE_ROOT
    manifests.mkdir(parents=True)
    (repo / builder.APPROVED_CAN_RELATIVE).parent.mkdir(parents=True, exist_ok=True)
    (repo / builder.APPROVED_CAN_RELATIVE).write_text("#usda 1.0\n", encoding="utf-8")
    for name in (
        "adp009d_840313_runtime_readiness.v1.json",
        builder.HARNESS_MANIFEST_RELATIVE.name,
        builder.SCENARIO_INSTANCE_RELATIVE.name,
    ):
        (manifests / name).write_text(json.dumps({"name": name}), encoding="utf-8")
    (manifests / builder.SOURCE_MANIFEST_NAME).write_text(
        json.dumps(
            {"bundle_id": builder.BUNDLE_ID, "bundle_digest": builder.EXPECTED_BUNDLE_DIGEST}
        ),
        encoding="utf-8",
    )
    (manifests / builder.EVALUATION_SPEC_NAME).write_text(
        json.dumps({"spec": "stub"}), encoding="utf-8"
    )
    monkeypatch.setattr(
        builder,
        "validate_evaluation_run_spec",
        lambda value: {"status": "passed", "spec_digest": builder.EXPECTED_SPEC_DIGEST},
    )
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / builder.SAGE_COLLISION_NAME).write_text("collision", encoding="utf-8")
    runtime_inputs = tmp_path / "runtime"
    runtime_inputs.mkdir()
    (runtime_inputs / builder.AURA_APPEARANCE_NAME).write_text("aura", encoding="utf-8")

    monkeypatch.setattr(builder, "verify_protected_main_checkout", lambda *a, **k: None)
    monkeypatch.setattr(builder, "verify_materialized_source_artifacts", lambda *a, **k: {})
    return {
        "repo": repo,
        "inputs": inputs,
        "runtime_inputs": runtime_inputs,
        "guard": _guard(tmp_path / "guard.json"),
        "out": tmp_path / "out",
    }


def _launch_request(profile: dict, *, launch_id: str) -> dict:
    request = {
        "schema_version": "task_evaluation_launch_request.v1",
        "launch_id": launch_id,
        "run_id": launch_id.replace("launch-", "run-", 1),
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
                    "uri": "firestore://taskEvaluationLaunchAuthorities/newton-001",
                    "digest": "sha256:" + "c" * 64,
                },
            },
            "spend": {
                "approved": True,
                "currency": "USD",
                "max_spend_usd": profile["allocator"]["max_spend_usd"],
                "expires_at": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
            },
            "execution": {"approved": True},
        },
        "required_controls": profile["required_controls"],
        "claim_ceiling": "development_only",
        "idempotency_key": launch_id,
    }
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    return request


def test_controls_profile_is_website_triggerable_and_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The controls canary must be launchable through the production dispatcher
    rather than a hand-written runner, while staying inside the same spend, TTL,
    retry, and claim ceilings."""
    env = _release(tmp_path, monkeypatch)
    receipt = builder.build_controls_profile_release(
        source_commit=COMMIT,
        repo_root=env["repo"],
        production_input_root=env["inputs"],
        runtime_input_root=env["runtime_inputs"],
        provider_guard_path=env["guard"],
        output_dir=env["out"],
    )

    assert receipt["status"] == "built"
    assert receipt["provider_mutation_performed"] is False
    profile = json.loads(Path(receipt["profile_path"]).read_text(encoding="utf-8"))
    assert validate_launch_profile(profile) == []

    # Website-triggerable: the dispatcher only appends --execute when this is true.
    assert profile["execution_admission"]["live_enabled"] is True
    assert profile["execution_admission"]["blockers"] == []
    # A passing controls pair is harness evidence, never a task or policy result.
    assert profile["claim_ceiling"] == "development_only"

    allocator = profile["allocator"]
    assert allocator["entrypoint"] == CANONICAL_ALLOCATOR_ENTRYPOINT
    assert allocator["max_spend_usd"] == 6.0
    assert allocator["hard_ttl_seconds"] == 5400
    assert allocator["retry_cap"] == 0
    argv = allocator["argv"]
    # Controls only: no learned candidate may be queried by this profile.
    assert "--adp009d-controls" in argv
    assert "--adp009d-policy-candidate" not in argv
    assert "--adp009d-authorize-gated-backbone" not in argv
    # The dispatcher owns the execute flag; a profile may never carry it.
    assert "--execute" not in argv
    # Per-launch outputs must not collide across website launches.
    for flag in ("--admission-out", "--adapter-output", "--adp-job-dir"):
        assert any(
            item.startswith(LAUNCH_RUN_ROOT_PLACEHOLDER)
            for item in argv[argv.index(flag) + 1 : argv.index(flag) + 2]
        )
    immutable_names = {row["name"] for row in profile["immutable_inputs"]}
    assert {
        "provider_guard",
        "allocator_preflight_request",
        "pipeline_release_evidence",
    } <= immutable_names
    for flag in ("--provider-launch-request", "--release-evidence"):
        assert Path(argv[argv.index(flag) + 1]).is_file()


def test_controls_profile_requires_provider_zero_and_present_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Its admission is its own, so each element of that admission must be real."""
    env = _release(tmp_path, monkeypatch)

    _guard(
        env["guard"],
        instances=[{"id": 1, "name": f"{builder.LANE_INSTANCE_PREFIX}1786"}],
    )
    with pytest.raises(
        builder.ProductionProfileBuildError,
        match="lane_instance_live",
    ):
        builder.build_controls_profile_release(
            source_commit=COMMIT,
            repo_root=env["repo"],
            production_input_root=env["inputs"],
            runtime_input_root=env["runtime_inputs"],
            provider_guard_path=env["guard"],
            output_dir=env["out"],
        )

    _guard(env["guard"], status="blocked")
    with pytest.raises(builder.ProductionProfileBuildError, match="provider_guard_not_passed"):
        builder.build_controls_profile_release(
            source_commit=COMMIT,
            repo_root=env["repo"],
            production_input_root=env["inputs"],
            runtime_input_root=env["runtime_inputs"],
            provider_guard_path=env["guard"],
            output_dir=env["out"],
        )

    # A concurrent operator's unrelated instance must NOT block this lane.
    _guard(
        env["guard"],
        instances=[{"id": 2, "name": "blueprint-native-deformable-asset-1786"}],
    )
    receipt = builder.build_controls_profile_release(
        source_commit=COMMIT,
        repo_root=env["repo"],
        production_input_root=env["inputs"],
        runtime_input_root=env["runtime_inputs"],
        provider_guard_path=env["guard"],
        output_dir=env["out"] / "concurrent",
    )
    assert receipt["status"] == "built"

    _guard(env["guard"])
    (env["runtime_inputs"] / builder.AURA_APPEARANCE_NAME).unlink()
    with pytest.raises(builder.ProductionProfileBuildError, match="input_missing:aura_appearance"):
        builder.build_controls_profile_release(
            source_commit=COMMIT,
            repo_root=env["repo"],
            production_input_root=env["inputs"],
            runtime_input_root=env["runtime_inputs"],
            provider_guard_path=env["guard"],
            output_dir=env["out"],
        )


def test_published_controls_catalog_exposes_no_allocator_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The public projection must still hide allocator arguments and secrets even
    though this profile is live-enabled."""
    env = _release(tmp_path, monkeypatch)
    receipt = builder.build_controls_profile_release(
        source_commit=COMMIT,
        repo_root=env["repo"],
        production_input_root=env["inputs"],
        runtime_input_root=env["runtime_inputs"],
        provider_guard_path=env["guard"],
        output_dir=env["out"],
    )
    publish_profiles(
        profile_paths=[receipt["profile_path"]],
        profile_dir=tmp_path / "published",
        webapp_catalog_out=tmp_path / "catalog.json",
    )
    catalog = json.loads((tmp_path / "catalog.json").read_text(encoding="utf-8"))

    assert len(catalog) == 1
    descriptor = catalog[0]
    assert validate_public_launch_profile_descriptor(descriptor) == []
    assert descriptor["execution_admission"]["live_enabled"] is True
    assert descriptor["required_authorization"]["max_spend_usd"] == 6.0
    serialized = json.dumps(catalog)
    assert "allocator" not in descriptor
    assert "--adp009d-controls" not in serialized
    assert "provider-launch-request" not in serialized


def test_diagnostic_mode_runs_the_pipeline_without_the_control_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The descend stall blocks every controls run before terminal success, so the
    production path has never produced a completed run. Diagnostic mode must
    exercise the same bundle, provider, and teardown path with no control pair
    and no policy — proving the pipeline, never the task."""
    env = _release(tmp_path, monkeypatch)
    receipt = builder.build_controls_profile_release(
        source_commit=COMMIT,
        repo_root=env["repo"],
        production_input_root=env["inputs"],
        runtime_input_root=env["runtime_inputs"],
        provider_guard_path=env["guard"],
        output_dir=env["out"],
        mode="diagnostic",
    )

    assert receipt["mode"] == "diagnostic"
    assert receipt["profile_id"].startswith(builder.DIAGNOSTIC_PROFILE_ID_PREFIX)
    profile = json.loads(Path(receipt["profile_path"]).read_text(encoding="utf-8"))
    assert validate_launch_profile(profile) == []

    argv = profile["allocator"]["argv"]
    # Exactly one execution mode: the allocator rejects more than one.
    assert "--adp009d-diagnostic-only" in argv
    assert "--adp009d-controls" not in argv
    assert "--adp009d-policy-candidate" not in argv
    # Same ceilings and the same claim boundary as the controls profile.
    assert profile["allocator"]["max_spend_usd"] == 6.0
    assert profile["allocator"]["retry_cap"] == 0
    assert profile["claim_ceiling"] == "development_only"

    # Distinct profile id, so a diagnostic run can never be mistaken for
    # controls evidence when the live profile is later assembled.
    controls = builder.build_controls_profile_release(
        source_commit=COMMIT,
        repo_root=env["repo"],
        production_input_root=env["inputs"],
        runtime_input_root=env["runtime_inputs"],
        provider_guard_path=env["guard"],
        output_dir=env["out"] / "controls",
        mode="controls",
    )
    assert controls["profile_id"] != receipt["profile_id"]

    with pytest.raises(builder.ProductionProfileBuildError, match="mode_invalid"):
        builder.build_controls_profile_release(
            source_commit=COMMIT,
            repo_root=env["repo"],
            production_input_root=env["inputs"],
            runtime_input_root=env["runtime_inputs"],
            provider_guard_path=env["guard"],
            output_dir=env["out"] / "bogus",
            mode="policy",
        )


def test_newton_controls_profile_binds_admission_guard_and_two_dollar_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _release(tmp_path, monkeypatch)
    admission_path = _newton_admission(env, tmp_path)

    receipt = builder.build_controls_profile_release(
        source_commit=COMMIT,
        repo_root=env["repo"],
        production_input_root=env["inputs"],
        runtime_input_root=env["runtime_inputs"],
        provider_guard_path=env["guard"],
        output_dir=env["out"],
        physics_backend="newton",
        newton_canary_admission_path=admission_path,
    )

    profile = json.loads(Path(receipt["profile_path"]).read_text(encoding="utf-8"))
    argv = profile["allocator"]["argv"]
    assert validate_launch_profile(profile) == []
    assert profile["profile_id"].startswith(builder.NEWTON_PROFILE_ID_PREFIX)
    assert profile["physics_backend"] == "newton"
    assert (
        profile["physics_backend_profile_digest"]
        == build_backend_profile("newton")["profile_digest"]
    )
    assert profile["allocator"]["max_spend_usd"] == 2.0
    assert profile["allocator"]["hard_ttl_seconds"] == 5400
    assert argv[argv.index("--adp009d-physics-backend") + 1] == "newton"
    assert argv[argv.index("--adp009d-newton-canary-admission") + 1] == str(admission_path)
    assert "--adp009d-controls" in argv
    assert "--adp009d-policy-candidate" not in argv
    assert "--adp009d-authorize-gated-backbone" not in argv
    assert "--execute" not in argv
    assert (
        receipt["newton_canary_admission_digest"]
        == json.loads(admission_path.read_text(encoding="utf-8"))["admission_digest"]
    )
    immutable_names = {row["name"] for row in profile["immutable_inputs"]}
    assert {"provider_guard", "newton_canary_admission"} <= immutable_names


def test_newton_controls_profile_fails_closed_on_missing_or_mismatched_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _release(tmp_path, monkeypatch)
    with pytest.raises(builder.ProductionProfileBuildError, match="admission_missing"):
        builder.build_controls_profile_release(
            source_commit=COMMIT,
            repo_root=env["repo"],
            production_input_root=env["inputs"],
            runtime_input_root=env["runtime_inputs"],
            provider_guard_path=env["guard"],
            output_dir=env["out"],
            physics_backend="newton",
        )

    admission_path = _newton_admission(env, tmp_path)
    guard = json.loads(env["guard"].read_text(encoding="utf-8"))
    guard["generated_at"] = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    env["guard"].write_text(json.dumps(guard), encoding="utf-8")
    with pytest.raises(builder.ProductionProfileBuildError, match="guard_mismatch"):
        builder.build_controls_profile_release(
            source_commit=COMMIT,
            repo_root=env["repo"],
            production_input_root=env["inputs"],
            runtime_input_root=env["runtime_inputs"],
            provider_guard_path=env["guard"],
            output_dir=env["out"] / "mismatch",
            physics_backend="newton",
            newton_canary_admission_path=admission_path,
        )


def test_newton_website_queue_dispatches_one_launch_exactly_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _release(tmp_path, monkeypatch)
    admission_path = _newton_admission(env, tmp_path)
    receipt = builder.build_controls_profile_release(
        source_commit=COMMIT,
        repo_root=env["repo"],
        production_input_root=env["inputs"],
        runtime_input_root=env["runtime_inputs"],
        provider_guard_path=env["guard"],
        output_dir=env["out"],
        physics_backend="newton",
        newton_canary_admission_path=admission_path,
    )
    profile = json.loads(Path(receipt["profile_path"]).read_text(encoding="utf-8"))
    launch_id = "launch-newton-controls-001"
    request = _launch_request(profile, launch_id=launch_id)
    queue_root = tmp_path / "queue"
    first = stage_launch_request(value=request, queue_root=queue_root)
    second = stage_launch_request(value=request, queue_root=queue_root)
    assert first["already_exists"] is False
    assert second["already_exists"] is True

    monkeypatch.setenv(EXECUTE_ENV, "true")
    monkeypatch.setenv(SECRET_PROFILE_ID_ENV, "canonical-vast-adp")
    calls: list[list[str]] = []

    def allocator_runner(argv: list[str]) -> int:
        calls.append(argv)
        result_path = Path(argv[argv.index("--adapter-output") + 1])
        teardown = result_path.parent / "teardown.json"
        artifacts = result_path.parent / "artifacts.json"
        teardown.parent.mkdir(parents=True, exist_ok=True)
        teardown.write_text(
            json.dumps({"continuing_spend_from_this_run": False}),
            encoding="utf-8",
        )
        artifacts.write_text(json.dumps({"status": "retained"}), encoding="utf-8")
        result_path.write_text(
            json.dumps(
                {
                    "status": "completed",
                    "continuing_spend_from_this_run": False,
                    "retry_cap": 0,
                    "teardown_manifest_path": str(teardown),
                    "artifact_manifest_path": str(artifacts),
                }
            ),
            encoding="utf-8",
        )
        return 0

    first_run = process_launch_queue(
        queue_root=queue_root,
        profile_dir=env["out"],
        state_root=tmp_path / "state",
        execute=True,
        execute_launch_id=launch_id,
        allocator_runner=allocator_runner,
    )
    second_run = process_launch_queue(
        queue_root=queue_root,
        profile_dir=env["out"],
        state_root=tmp_path / "state",
        execute=True,
        execute_launch_id=launch_id,
        allocator_runner=allocator_runner,
    )

    assert first_run["processed_count"] == 1
    assert first_run["receipts"][0]["status"] == "completed"
    assert second_run["processed_count"] == 0
    assert len(calls) == 1
    assert "--execute" in calls[0]


def test_newton_website_dispatch_fails_before_allocator_if_guard_drifts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _release(tmp_path, monkeypatch)
    admission_path = _newton_admission(env, tmp_path)
    receipt = builder.build_controls_profile_release(
        source_commit=COMMIT,
        repo_root=env["repo"],
        production_input_root=env["inputs"],
        runtime_input_root=env["runtime_inputs"],
        provider_guard_path=env["guard"],
        output_dir=env["out"],
        physics_backend="newton",
        newton_canary_admission_path=admission_path,
    )
    profile = json.loads(Path(receipt["profile_path"]).read_text(encoding="utf-8"))
    request = _launch_request(profile, launch_id="launch-newton-drift-001")
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    env["guard"].write_text('{"status":"tampered"}\n', encoding="utf-8")
    calls: list[list[str]] = []

    dispatch = dispatch_launch_request(
        request_path=request_path,
        profile_dir=env["out"],
        state_root=tmp_path / "state",
        allocator_runner=lambda argv: calls.append(list(argv)) or 0,
    )

    assert dispatch["status"] == "blocked"
    assert any("provider_guard" in blocker for blocker in dispatch["blockers"])
    assert dispatch["provider_mutation_attempted"] is False
    assert calls == []
