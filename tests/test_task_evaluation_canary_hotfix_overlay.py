from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest

from blueprint_pipeline.task_evaluation_canary_hotfix_overlay import (
    CanaryHotfixOverlayError,
    bind_service_group,
    resolve_service_account,
    service_account_access_blockers,
    DEFAULT_STRATEGY,
    FALLBACK_STRATEGY,
    apply_canary_hotfix_overlay,
    canary_hotfix_execution_release,
    choose_canary_iteration_strategy,
    prepare_canary_hotfix_overlay,
    install_policy_canary_dispatcher_overlay,
    run_focused_hotfix_tests,
    verify_canary_hotfix_overlay,
)


def _account(**_kwargs) -> dict:
    """The test runner's own identity plays the hardened service account."""

    return {
        "user": "runner",
        "uid": os.getuid(),
        "group": "runner",
        "gid": os.getgid(),
        "group_ids": sorted({os.getgid(), *os.getgroups()}),
    }


def _foreign_account(**_kwargs) -> dict:
    """An identity that owns nothing in the test tree and shares no group."""

    return {"user": "other", "uid": 65_001, "group": "other", "gid": 65_002, "group_ids": [65_002]}


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def _repo(tmp_path: Path) -> tuple[Path, str, str]:
    root = tmp_path / "repo"
    remote = tmp_path / "origin.git"
    root.mkdir()
    _git(root, "init", "-b", "main")
    _git(root, "config", "user.email", "canary@example.invalid")
    _git(root, "config", "user.name", "Canary Test")
    runtime = root / "src/blueprint_pipeline/openpi_droid_policy_runtime.py"
    test = root / "tests/test_openpi_droid_policy_runtime.py"
    runtime.parent.mkdir(parents=True)
    test.parent.mkdir(parents=True)
    runtime.write_text("BASE = True\n", encoding="utf-8")
    test.write_text("def test_base():\n    assert True\n", encoding="utf-8")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "base")
    base = _git(root, "rev-parse", "HEAD")
    subprocess.run(["git", "init", "--bare", str(remote)], check=True)
    _git(root, "remote", "add", "origin", str(remote))
    _git(root, "push", "-u", "origin", "main")
    _git(root, "switch", "-c", "hotfix")
    runtime.write_text("BASE = False\nHOTFIX = True\n", encoding="utf-8")
    test.write_text("def test_hotfix():\n    assert True\n", encoding="utf-8")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "hotfix")
    patch = _git(root, "rev-parse", "HEAD")
    _git(root, "push", "-u", "origin", "hotfix")
    return root, base, patch


def test_provider_runtime_only_patch_uses_overlay_by_default(tmp_path: Path) -> None:
    root, base, patch = _repo(tmp_path)

    routed = choose_canary_iteration_strategy(
        repo_root=root, base_commit=base, patch_commit=patch
    )

    assert routed["strategy"] == DEFAULT_STRATEGY
    assert routed["unsupported_paths"] == []
    assert routed["evidence_grade_ceiling"] == "development_only"
    assert routed["normal_deployment_required_for_promotion"] is True


def test_unsupported_surface_falls_back_to_exact_main_deploy(tmp_path: Path) -> None:
    root, base, _patch = _repo(tmp_path)
    (root / "pyproject.toml").write_text("[project]\nname='changed'\n", encoding="utf-8")
    _git(root, "add", "pyproject.toml")
    _git(root, "commit", "-m", "dependency surface")
    patch = _git(root, "rev-parse", "HEAD")
    _git(root, "push", "origin", "hotfix")

    routed = choose_canary_iteration_strategy(
        repo_root=root, base_commit=base, patch_commit=patch
    )

    assert routed["strategy"] == FALLBACK_STRATEGY
    assert routed["unsupported_paths"] == ["pyproject.toml"]
    assert "canary_hotfix_unsupported_surface_changed" in routed["blockers"]


def test_canary_worker_overlay_targets_the_executed_provider_runner(tmp_path: Path) -> None:
    root, base, _patch = _repo(tmp_path)
    worker = root / "src/blueprint_pipeline/native_task_arena_policy_canary_worker.py"
    worker.write_text("HOTFIX = True\n", encoding="utf-8")
    _git(root, "add", str(worker.relative_to(root)))
    _git(root, "commit", "-m", "worker hotfix")
    patch = _git(root, "rev-parse", "HEAD")
    _git(root, "push", "origin", "hotfix")
    failure = tmp_path / "worker-failure.json"
    failure.write_text("{}\n", encoding="utf-8")
    tests = run_focused_hotfix_tests(
        repo_root=root,
        base_commit=base,
        patch_commit=patch,
        commands=[[sys.executable, "-c", "assert True"]],
        exact_failure_input=failure,
    )
    plan = prepare_canary_hotfix_overlay(
        repo_root=root,
        output_dir=tmp_path / "worker-overlay",
        base_commit=base,
        patch_commit=patch,
        test_receipt=tests,
        account_resolver=_account,
    )

    destinations = {
        row["destination"] for row in plan["manifest"]["source_inventory"]
    }
    assert "adp_arena_provider_runner.py" in destinations


def test_overlay_seals_exact_failure_test_and_applies_only_to_staging(
    tmp_path: Path,
) -> None:
    root, base, patch = _repo(tmp_path)
    failure = tmp_path / "failed-policy-spec.json"
    failure.write_text(json.dumps({"schema_version": "canary.v1"}), encoding="utf-8")
    tests = run_focused_hotfix_tests(
        repo_root=root,
        base_commit=base,
        patch_commit=patch,
        commands=[[sys.executable, "-c", "assert True"]],
        exact_failure_input=failure,
    )
    output = tmp_path / "overlay"
    plan = prepare_canary_hotfix_overlay(
        repo_root=root,
        output_dir=output,
        base_commit=base,
        patch_commit=patch,
        test_receipt=tests,
        account_resolver=_account,
    )
    archive = Path(plan["archive_path"])
    verified = verify_canary_hotfix_overlay(archive)
    execution_release = canary_hotfix_execution_release(verified)
    staging = tmp_path / "provider-runtime"
    staging.mkdir()
    destination = staging / "openpi_droid_policy_runtime.py"
    destination.write_text("BASE = True\n", encoding="utf-8")

    applied = apply_canary_hotfix_overlay(
        archive_path=archive, provider_runtime_root=staging
    )

    assert verified["patch_commit"] == patch
    assert destination.read_text(encoding="utf-8") == "BASE = False\nHOTFIX = True\n"
    assert (staging / "signed_hotfix_overlay.v1.json").is_file()
    assert applied["status"] == "applied_to_staging_bundle"
    assert applied["active_release_mutation_performed"] is False
    assert applied["provider_mutation_performed"] is False
    assert applied["evidence_grade_ceiling"] == "development_only"
    assert execution_release["mode"] == DEFAULT_STRATEGY
    assert execution_release["patch_commit"] == patch
    drop_in = tmp_path / "systemd/dispatcher.service.d/96-hotfix.conf"
    installation = install_policy_canary_dispatcher_overlay(
        plan_path=output / "task_evaluation_canary_hotfix_overlay_plan.v1.json",
        drop_in_path=drop_in,
        receipt_path=tmp_path / "installation.json",
        account_resolver=_account,
    )
    assert str(archive) in drop_in.read_text(encoding="utf-8")
    assert installation["status"] == "installed_for_next_policy_canary_dispatch"
    assert installation["active_release_mutation_performed"] is False


def test_prepare_binds_the_service_group_and_seals_readability_into_the_plan(
    tmp_path: Path,
) -> None:
    root, base, patch = _repo(tmp_path)
    failure = tmp_path / "failed.json"
    failure.write_text("{}", encoding="utf-8")
    tests = run_focused_hotfix_tests(
        repo_root=root,
        base_commit=base,
        patch_commit=patch,
        commands=[[sys.executable, "-c", "assert True"]],
        exact_failure_input=failure,
    )
    output = tmp_path / "overlay"
    chowns: list[tuple[str, int, int]] = []

    plan = prepare_canary_hotfix_overlay(
        repo_root=root,
        output_dir=output,
        base_commit=base,
        patch_commit=patch,
        test_receipt=tests,
        account_resolver=_account,
        chown=lambda path, uid, gid: chowns.append((Path(path).name, uid, gid)),
    )

    archive = Path(plan["archive_path"])
    assert plan["service_access"]["status"] == "readable_by_service_account"
    assert plan["service_access"]["service_user"] == "runner"
    assert plan["service_access"]["verified_names"] == [archive.name]
    assert {row["name"]: row["mode"] for row in plan["service_access"]["group_bindings"]} == {
        output.name: "0750",
        archive.name: "0640",
    }
    assert (output.stat().st_mode & 0o777) == 0o750
    assert (archive.stat().st_mode & 0o777) == 0o640
    for name in (
        "task_evaluation_canary_hotfix_overlay_plan.v1.json",
        "task_evaluation_canary_hotfix_test_receipt.v1.json",
    ):
        assert ((output / name).stat().st_mode & 0o777) == 0o640
    assert {name for name, _uid, _gid in chowns} == {
        output.name,
        archive.name,
        "task_evaluation_canary_hotfix_overlay_plan.v1.json",
        "task_evaluation_canary_hotfix_test_receipt.v1.json",
    }
    assert all(uid == -1 and gid == os.getgid() for _name, uid, gid in chowns)
    assert plan["plan_digest"] == canonical_digest(plan, digest_field="plan_digest")


def test_install_refuses_an_overlay_the_service_account_cannot_open(
    tmp_path: Path,
) -> None:
    root, base, patch = _repo(tmp_path)
    failure = tmp_path / "failed.json"
    failure.write_text("{}", encoding="utf-8")
    tests = run_focused_hotfix_tests(
        repo_root=root,
        base_commit=base,
        patch_commit=patch,
        commands=[[sys.executable, "-c", "assert True"]],
        exact_failure_input=failure,
    )
    output = tmp_path / "overlay"
    prepare_canary_hotfix_overlay(
        repo_root=root,
        output_dir=output,
        base_commit=base,
        patch_commit=patch,
        test_receipt=tests,
        account_resolver=_account,
    )
    drop_in = tmp_path / "systemd/dispatcher.service.d/96-hotfix.conf"

    # The exact production failure: a root-only overlay directory (0750
    # root:root) that the blueprint service account cannot traverse.
    with pytest.raises(CanaryHotfixOverlayError) as unreadable:
        install_policy_canary_dispatcher_overlay(
            plan_path=output / "task_evaluation_canary_hotfix_overlay_plan.v1.json",
            drop_in_path=drop_in,
            receipt_path=tmp_path / "installation.json",
            account_resolver=_foreign_account,
        )

    assert "canary_hotfix_overlay_unreadable_by_service_account" in unreadable.value.blockers
    assert any(
        blocker == f"canary_hotfix_service_account_cannot_traverse:{output.name}"
        for blocker in unreadable.value.blockers
    )
    assert not drop_in.exists()

    # Once the directory is traversable by others but the archive is not
    # readable by them, the archive itself is the named blocker.
    output.chmod(0o755)
    with pytest.raises(CanaryHotfixOverlayError) as archive_unreadable:
        install_policy_canary_dispatcher_overlay(
            plan_path=output / "task_evaluation_canary_hotfix_overlay_plan.v1.json",
            drop_in_path=drop_in,
            receipt_path=tmp_path / "installation.json",
            account_resolver=_foreign_account,
        )
    assert (
        "canary_hotfix_service_account_cannot_read:task_evaluation_canary_hotfix_overlay.zip"
        in archive_unreadable.value.blockers
    )
    assert not drop_in.exists()

    installation = install_policy_canary_dispatcher_overlay(
        plan_path=output / "task_evaluation_canary_hotfix_overlay_plan.v1.json",
        drop_in_path=drop_in,
        receipt_path=tmp_path / "installation.json",
        account_resolver=_account,
    )
    assert installation["service_access"]["status"] == "readable_by_service_account"
    assert drop_in.is_file()


def test_service_account_access_blockers_walk_every_ancestor(tmp_path: Path) -> None:
    nested = tmp_path / "outer" / "inner"
    nested.mkdir(parents=True)
    target = nested / "overlay.zip"
    target.write_bytes(b"zip")
    (tmp_path / "outer").chmod(0o700)
    nested.chmod(0o750)
    target.chmod(0o640)

    def owned(blockers: list[str]) -> list[str]:
        # pytest's own temporary ancestors are private to the runner; only the
        # tree this test built is under assertion.
        return [row for row in blockers if row.rsplit(":", 1)[-1] in {"outer", "inner", "overlay.zip"}]

    assert service_account_access_blockers(target, account=_account()) == []
    assert owned(service_account_access_blockers(target, account=_foreign_account())) == [
        "canary_hotfix_service_account_cannot_traverse:outer",
        "canary_hotfix_service_account_cannot_traverse:inner",
        "canary_hotfix_service_account_cannot_read:overlay.zip",
    ]
    (tmp_path / "outer").chmod(0o755)
    nested.chmod(0o755)
    target.chmod(0o644)
    assert owned(service_account_access_blockers(target, account=_foreign_account())) == []
    assert service_account_access_blockers(
        tmp_path / "outer" / "missing.zip", account=_account()
    ) == ["canary_hotfix_service_account_cannot_stat:missing.zip"]


def test_service_account_resolution_and_group_binding_fail_closed(tmp_path: Path) -> None:
    class _Passwd:
        pw_uid = 1000
        pw_gid = 1000

    class _Group:
        gr_gid = 2000
        gr_mem = ("blueprint",)

    account = resolve_service_account(
        user="blueprint",
        group="blueprint",
        getpwnam=lambda _name: _Passwd(),
        getgrnam=lambda _name: _Group(),
        getgrall=lambda: [_Group()],
    )
    assert account == {
        "user": "blueprint",
        "uid": 1000,
        "group": "blueprint",
        "gid": 2000,
        "group_ids": [1000, 2000],
    }

    def unknown(_name: str) -> object:
        raise KeyError(_name)

    with pytest.raises(CanaryHotfixOverlayError, match="canary_hotfix_service_account_unknown"):
        resolve_service_account(getpwnam=unknown, getgrnam=unknown, getgrall=lambda: [])

    target = tmp_path / "artifact.zip"
    target.write_bytes(b"zip")

    def refused(path: Path, uid: int, gid: int) -> None:
        raise PermissionError(str(path))

    with pytest.raises(
        CanaryHotfixOverlayError, match="canary_hotfix_service_group_binding_failed"
    ):
        bind_service_group([target], account=_account(), chown=refused)
    link = tmp_path / "link.zip"
    link.symlink_to(target)
    with pytest.raises(CanaryHotfixOverlayError, match="canary_hotfix_service_binding_symlink"):
        bind_service_group([link], account=_account(), chown=lambda *_args: None)
