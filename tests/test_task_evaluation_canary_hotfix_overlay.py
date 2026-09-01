from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from blueprint_pipeline.task_evaluation_canary_hotfix_overlay import (
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
    )
    assert str(archive) in drop_in.read_text(encoding="utf-8")
    assert installation["status"] == "installed_for_next_policy_canary_dispatch"
    assert installation["active_release_mutation_performed"] is False
