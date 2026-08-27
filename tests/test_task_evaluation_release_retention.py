import hashlib
import json
import os
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from blueprint_pipeline.task_evaluation_launch_catalog import build_catalog_payload
from blueprint_pipeline.task_evaluation_launch_dispatcher import (
    CANONICAL_ALLOCATOR_ENTRYPOINT,
    canonical_digest,
)
from blueprint_pipeline.task_evaluation_release_retention import (
    APPLY_ACKNOWLEDGEMENT,
    EVIDENCE_BINDING_SCHEMA_VERSION,
    ReleaseRetentionError,
    apply_release_retention_plan,
    build_release_retention_plan,
)


NOW = datetime(2026, 8, 27, 18, 0, tzinfo=timezone.utc)


def _profile(root: Path, *, profile_id: str, commit: str) -> dict:
    source_manifest = root / f"{profile_id}-source.json"
    evaluation_spec = root / f"{profile_id}-spec.json"
    source_manifest.write_text('{"scene":"839873"}\n', encoding="utf-8")
    evaluation_spec.write_text('{"spec":"frozen"}\n', encoding="utf-8")

    def digest(path: Path) -> str:
        return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()

    profile = {
        "schema_version": "task_evaluation_launch_profile.v1",
        "profile_id": profile_id,
        "program_id": "arm-decision-proof-v1",
        "source_bundle": {
            "bundle_id": "scene-839873",
            "source_kind": "interiorgs_sage",
            "uri": "s3://blueprint/scene-839873.json",
            "digest": "sha256:" + "a" * 64,
        },
        "evaluation_run_spec": {
            "uri": "s3://blueprint/evaluation-run-spec.json",
            "digest": "sha256:" + "b" * 64,
        },
        "immutable_inputs": [
            {
                "name": "source_bundle_manifest",
                "path": str(source_manifest),
                "digest": digest(source_manifest),
            },
            {
                "name": "evaluation_run_spec",
                "path": str(evaluation_spec),
                "digest": digest(evaluation_spec),
            },
        ],
        "allocator": {
            "entrypoint": CANONICAL_ALLOCATOR_ENTRYPOINT,
            "subcommand": "gpu-canary",
            "argv": [
                "--provider-launch-request",
                str(root / "provider.json"),
                "--release-evidence",
                str(root / "release.json"),
                "--model-cache-evidence",
                str(root / "cache.json"),
                "--preflight-bundle",
                str(root / "preflight.json"),
                "--admission-out",
                str(root / "admission.json"),
                "--bound-request-out",
                str(root / "bound.json"),
                "--adapter-output",
                str(root / "result.json"),
                "--pod-name",
                "blueprint-retention-test",
            ],
            "max_spend_usd": 2.0,
            "hard_ttl_seconds": 7200,
            "retry_cap": 0,
        },
        "runtime_environment": {},
        "execution_admission": {
            "live_enabled": True,
            "readiness_receipt": {
                "uri": "s3://blueprint/readiness.json",
                "digest": "sha256:" + "c" * 64,
            },
            "blockers": [],
        },
        "reconciliation": {
            "required_providers": ["vast"],
            "max_guard_age_seconds": 300,
        },
        "webapp_sync": {"max_attempts": 3},
        "terminal_contract": {
            "result_path": str(root / "result.json"),
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
        "source_commit": commit,
    }
    profile["profile_digest"] = canonical_digest(
        profile, digest_field="profile_digest"
    )
    return profile


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _managed_artifacts(root: Path, commits: list[str]) -> tuple[Path, Path]:
    release_root = root / "releases"
    runtime_root = root / "system-runtimes"
    for commit in commits:
        for path in (
            release_root / commit,
            runtime_root / "splat-render" / commit,
            runtime_root / "scene-configuration" / commit,
        ):
            path.mkdir(parents=True, exist_ok=True)
            (path / "payload.bin").write_bytes(commit.encode("ascii"))
            old = (NOW - timedelta(days=3)).timestamp()
            os.utime(path, (old, old))
    return release_root, runtime_root


def _base_state(
    tmp_path: Path,
    *,
    commits: list[str],
    active_commit: str,
) -> dict[str, object]:
    release_root, runtime_root = _managed_artifacts(tmp_path, commits)
    active_link = tmp_path / "active"
    active_link.symlink_to(release_root / active_commit)
    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()
    catalog = tmp_path / "catalog.json"
    catalog.write_text("[]\n", encoding="utf-8")
    standing = tmp_path / "standing"
    standing.mkdir()
    live_pending = tmp_path / "queue" / "pending"
    live_processing = tmp_path / "queue" / "processing"
    live_pending.mkdir(parents=True)
    live_processing.mkdir(parents=True)
    return {
        "release_root": release_root,
        "runtime_root": runtime_root,
        "active_link": active_link,
        "profile_dir": profile_dir,
        "public_catalog": catalog,
        "standing_authorization_dir": standing,
        "live_reference_roots": (live_pending, live_processing),
        "evidence_binding_root": tmp_path / "evidence-bindings",
        "minimum_age_seconds": 86400,
        "now": NOW,
    }


def _rewrite_catalog(profile_dir: Path, catalog: Path) -> None:
    catalog.write_bytes(build_catalog_payload(profile_dir))


def test_plan_protects_every_live_binding_across_all_three_managed_roots(
    tmp_path: Path,
) -> None:
    active = "a" * 40
    deploy = "b" * 40
    stale = "c" * 40
    pending = "d" * 40
    authorized = "e" * 40
    evidence = "f" * 40
    state = _base_state(
        tmp_path,
        commits=[active, deploy, stale, pending, authorized, evidence],
        active_commit=active,
    )
    profile_dir = state["profile_dir"]
    assert isinstance(profile_dir, Path)
    pending_profile = _profile(
        tmp_path, profile_id="scene-839873-pending", commit=pending
    )
    authorized_profile = _profile(
        tmp_path, profile_id="scene-839873-authorized", commit=authorized
    )
    for profile in (pending_profile, authorized_profile):
        _write_json(profile_dir / f"{profile['profile_id']}.json", profile)
    catalog = state["public_catalog"]
    assert isinstance(catalog, Path)
    _rewrite_catalog(profile_dir, catalog)
    live_pending = state["live_reference_roots"][0]  # type: ignore[index]
    _write_json(
        live_pending / "launch.json",
        {"launch_profile_id": pending_profile["profile_id"]},
    )
    standing = state["standing_authorization_dir"]
    assert isinstance(standing, Path)
    _write_json(
        standing / f"{authorized_profile['profile_id']}.json",
        {
            "schema_version": "task_evaluation_standing_launch_authorization.v1",
            "profile_id": authorized_profile["profile_id"],
            "profile_digest": authorized_profile["profile_digest"],
            "max_launches": 1,
            "max_total_spend_usd": 2.0,
            "expires_at": (NOW + timedelta(days=1)).isoformat(),
        },
    )
    evidence_root = state["evidence_binding_root"]
    assert isinstance(evidence_root, Path)
    _write_json(
        evidence_root / "qualification.json",
        {
            "schema_version": EVIDENCE_BINDING_SCHEMA_VERSION,
            "status": "required",
            "source_commit": evidence,
            "reason": "terminal qualification replay remains open",
        },
    )

    plan = build_release_retention_plan(
        **state, current_deploy_commit=deploy  # type: ignore[arg-type]
    )

    assert [row["source_commit"] for row in plan["eligible_commits"]] == [stale]
    protected = plan["protected_commits"]
    assert protected[active] == ["active_release"]
    assert protected[deploy] == ["current_deploy_candidate"]
    assert protected[pending] == [
        "live_profile_reference:scene-839873-pending"
    ]
    assert protected[authorized] == [
        "unconsumed_standing_authorization:scene-839873-authorized"
    ]
    assert protected[evidence] == ["required_evidence:qualification.json"]
    stale_artifacts = plan["eligible_commits"][0]["artifacts"]
    assert {item["kind"] for item in stale_artifacts} == {
        "control_plane_release",
        "runtime_splat_render",
        "runtime_scene_configuration",
    }
    assert plan["predicted_removed_bytes"] == sum(
        item["size_bytes"] for item in stale_artifacts
    )
    assert plan["production_artifact_mutation_performed"] is False


def test_expired_orphan_standing_authorization_is_inventory_only(
    tmp_path: Path,
) -> None:
    active = "a" * 40
    stale = "c" * 40
    state = _base_state(tmp_path, commits=[active, stale], active_commit=active)
    standing = state["standing_authorization_dir"]
    assert isinstance(standing, Path)
    profile_id = f"retired-scene-{stale}"
    _write_json(
        standing / f"{profile_id}.json",
        {
            "schema_version": "task_evaluation_standing_launch_authorization.v1",
            "profile_id": profile_id,
            "profile_digest": "sha256:" + "d" * 64,
            "max_launches": 2,
            "max_total_spend_usd": 2.0,
            "expires_at": (NOW - timedelta(days=1)).isoformat(),
        },
    )

    plan = build_release_retention_plan(
        **state, current_deploy_commit=active  # type: ignore[arg-type]
    )

    assert [row["source_commit"] for row in plan["eligible_commits"]] == [stale]
    assert plan["terminal_orphan_standing_authorizations"] == [
        {
            "path": str(standing / f"{profile_id}.json"),
            "profile_id": profile_id,
            "terminal_blockers": ["standing_authorization_expired"],
            "launches_consumed": 0,
            "spend_consumed_usd": 0.0,
            "source_release_protected": False,
        }
    ]


def test_unexpired_or_malformed_orphan_standing_authorization_still_blocks(
    tmp_path: Path,
) -> None:
    active = "a" * 40
    state = _base_state(tmp_path, commits=[active], active_commit=active)
    standing = state["standing_authorization_dir"]
    assert isinstance(standing, Path)
    profile_id = "retired-but-still-authorized"
    _write_json(
        standing / f"{profile_id}.json",
        {
            "schema_version": "task_evaluation_standing_launch_authorization.v1",
            "profile_id": profile_id,
            "profile_digest": "sha256:" + "d" * 64,
            "max_launches": 2,
            "max_total_spend_usd": 2.0,
            "expires_at": (NOW + timedelta(days=1)).isoformat(),
        },
    )

    with pytest.raises(
        ReleaseRetentionError,
        match="release_retention_unknown_standing_authorization_child",
    ):
        build_release_retention_plan(
            **state, current_deploy_commit=active  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "corruption",
    ("unknown_managed_child", "managed_symlink", "malformed_pending"),
)
def test_any_ambiguous_managed_or_live_state_blocks_the_whole_plan(
    tmp_path: Path, corruption: str
) -> None:
    active = "a" * 40
    stale = "c" * 40
    state = _base_state(
        tmp_path, commits=[active, stale], active_commit=active
    )
    release_root = state["release_root"]
    assert isinstance(release_root, Path)
    pending = state["live_reference_roots"][0]  # type: ignore[index]
    if corruption == "unknown_managed_child":
        (release_root / "latest").mkdir()
    elif corruption == "managed_symlink":
        (release_root / ("d" * 40)).symlink_to(release_root / stale)
    else:
        (pending / "broken.json").write_text("{", encoding="utf-8")

    with pytest.raises(ReleaseRetentionError, match="release_retention_"):
        build_release_retention_plan(
            **state, current_deploy_commit=active  # type: ignore[arg-type]
        )
    assert (release_root / stale).is_dir()


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def test_apply_requires_exact_reviewed_plan_and_removes_only_eligible_sha(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _git(source, "init")
    _git(source, "config", "user.email", "retention@example.invalid")
    _git(source, "config", "user.name", "Retention Test")
    (source / "value.txt").write_text("active\n", encoding="utf-8")
    _git(source, "add", "value.txt")
    _git(source, "commit", "-m", "active")
    active = _git(source, "rev-parse", "HEAD")
    (source / "value.txt").write_text("stale\n", encoding="utf-8")
    _git(source, "commit", "-am", "stale")
    stale = _git(source, "rev-parse", "HEAD")
    release_root = tmp_path / "releases"
    release_root.mkdir()
    _git(source, "worktree", "add", "--detach", str(release_root / active), active)
    _git(source, "worktree", "add", "--detach", str(release_root / stale), stale)
    runtime_root = tmp_path / "runtimes"
    for commit in (active, stale):
        for component in ("splat-render", "scene-configuration"):
            path = runtime_root / component / commit
            path.mkdir(parents=True, exist_ok=True)
            (path / "payload").write_text(commit, encoding="ascii")
    active_link = tmp_path / "active"
    active_link.symlink_to(release_root / active)
    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()
    catalog = tmp_path / "catalog.json"
    catalog.write_text("[]\n", encoding="utf-8")
    standing = tmp_path / "standing"
    standing.mkdir()
    pending = tmp_path / "pending"
    processing = tmp_path / "processing"
    pending.mkdir()
    processing.mkdir()
    evidence = tmp_path / "evidence"
    plan = build_release_retention_plan(
        release_root=release_root,
        runtime_root=runtime_root,
        active_link=active_link,
        current_deploy_commit=active,
        profile_dir=profile_dir,
        public_catalog=catalog,
        standing_authorization_dir=standing,
        live_reference_roots=(pending, processing),
        evidence_binding_root=evidence,
        minimum_age_seconds=0,
        now=NOW,
    )
    plan_path = tmp_path / "reviewed-plan.json"
    plan_path.write_text(
        json.dumps(plan, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    receipt_path = tmp_path / "apply-receipt.json"

    receipt = apply_release_retention_plan(
        dry_run_plan_path=plan_path,
        acknowledgement=APPLY_ACKNOWLEDGEMENT,
        receipt_out=receipt_path,
    )

    assert receipt["status"] == "applied"
    assert receipt["removed_bytes"] == plan["predicted_removed_bytes"]
    assert not (release_root / stale).exists()
    assert not (runtime_root / "splat-render" / stale).exists()
    assert not (runtime_root / "scene-configuration" / stale).exists()
    assert (release_root / active).is_dir()
    assert (runtime_root / "splat-render" / active).is_dir()
    assert active_link.resolve() == (release_root / active).resolve()
    assert receipt_path.is_file()


def test_apply_rejects_a_binding_added_after_dry_run_before_any_delete(
    tmp_path: Path,
) -> None:
    active = "a" * 40
    stale = "c" * 40
    state = _base_state(
        tmp_path, commits=[active, stale], active_commit=active
    )
    plan = build_release_retention_plan(
        **state, current_deploy_commit=active  # type: ignore[arg-type]
    )
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    pending = state["live_reference_roots"][0]  # type: ignore[index]
    _write_json(pending / "late.json", {"expected_production_commit": stale})

    with pytest.raises(
        ReleaseRetentionError, match="release_retention_plan_changed_since_dry_run"
    ):
        apply_release_retention_plan(
            dry_run_plan_path=plan_path,
            acknowledgement=APPLY_ACKNOWLEDGEMENT,
            receipt_out=tmp_path / "receipt.json",
        )
    release_root = state["release_root"]
    runtime_root = state["runtime_root"]
    assert isinstance(release_root, Path)
    assert isinstance(runtime_root, Path)
    assert (release_root / stale).is_dir()
    assert (runtime_root / "splat-render" / stale).is_dir()


def test_expired_or_consumed_authorization_does_not_pin_runtime_forever(
    tmp_path: Path,
) -> None:
    active = "a" * 40
    stale = "c" * 40
    state = _base_state(
        tmp_path, commits=[active, stale], active_commit=active
    )
    profile_dir = state["profile_dir"]
    catalog = state["public_catalog"]
    standing = state["standing_authorization_dir"]
    assert isinstance(profile_dir, Path)
    assert isinstance(catalog, Path)
    assert isinstance(standing, Path)
    profile = _profile(tmp_path, profile_id="expired-profile", commit=stale)
    _write_json(profile_dir / "expired-profile.json", profile)
    _rewrite_catalog(profile_dir, catalog)
    _write_json(
        standing / "expired-profile.json",
        {
            "schema_version": "task_evaluation_standing_launch_authorization.v1",
            "profile_id": "expired-profile",
            "profile_digest": profile["profile_digest"],
            "max_launches": 1,
            "max_total_spend_usd": 2.0,
            "expires_at": (NOW - timedelta(seconds=1)).isoformat(),
        },
    )

    plan = build_release_retention_plan(
        **state, current_deploy_commit=active  # type: ignore[arg-type]
    )

    assert [row["source_commit"] for row in plan["eligible_commits"]] == [stale]
