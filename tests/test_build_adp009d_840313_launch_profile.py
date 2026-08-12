from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from blueprint_pipeline.task_evaluation_launch_dispatcher import (
    LAUNCH_RUN_ROOT_PLACEHOLDER,
    validate_launch_profile,
)
from scripts import build_adp009d_840313_launch_profile as builder


REPO = Path(__file__).resolve().parents[1]


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def test_builder_emits_exact_dry_profile_with_per_launch_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commit = "a" * 40
    monkeypatch.setattr(builder, "verify_protected_main_checkout", lambda *_args: None)
    raw_inputs: list[dict[str, str]] = []
    for index, name in enumerate(
        ("appearance_3dgs", "semantic_metadata", "scene_structure", "sage_usdz", "static_collision_geometry")
    ):
        path = tmp_path / "raw" / f"artifact-{index}.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"artifact-{index}".encode())
        raw_inputs.append({"name": name, "path": str(path), "digest": _digest(path)})
    monkeypatch.setattr(
        builder,
        "verify_materialized_source_artifacts",
        lambda *_args: raw_inputs,
    )

    receipt = builder.build_profile_release(
        source_commit=commit,
        repo_root=REPO,
        production_input_root=tmp_path / "production-inputs",
        provider_guard_path=tmp_path / "provider-guard.json",
        output_dir=tmp_path / "release",
    )

    profile = json.loads(Path(receipt["profile_path"]).read_text())
    preflight = json.loads(Path(receipt["preflight_request_path"]).read_text())
    assert receipt["status"] == "built"
    assert receipt["profile_id"] == f"{builder.PROFILE_ID_PREFIX}-{commit}"
    assert profile["profile_id"] == receipt["profile_id"]
    assert receipt["live_execution_enabled"] is False
    assert receipt["provider_mutation_performed"] is False
    assert validate_launch_profile(profile) == []
    assert profile["source_bundle"]["digest"] == builder.EXPECTED_BUNDLE_DIGEST
    assert profile["evaluation_run_spec"]["digest"] == builder.EXPECTED_SPEC_DIGEST
    assert profile["execution_admission"] == {
        "live_enabled": False,
        "readiness_receipt": {
            "uri": (
                f"{builder.RAW_GITHUB_ROOT}/{commit}/"
                f"{builder.MANIFEST_RELATIVE_ROOT.as_posix()}/{builder.READINESS_NAME}"
            ),
            "digest": builder.EXPECTED_READINESS_DIGEST,
        },
        "blockers": [
            "exact_adp009d_runtime_adapter_not_on_protected_main",
            "scripted_positive_control_not_passed",
            "allocator_artifact_manifest_not_emitted",
        ],
    }
    allocator_text = json.dumps(profile["allocator"])
    assert LAUNCH_RUN_ROOT_PLACEHOLDER in allocator_text
    assert str(tmp_path / "release" / "allocator") not in allocator_text
    assert profile["allocator"]["max_spend_usd"] == 6.0
    assert profile["allocator"]["hard_ttl_seconds"] == 5400
    assert profile["allocator"]["retry_cap"] == 0
    assert preflight["required_provider_zero"] == ["digitalocean", "runpod", "vast"]
    assert preflight["live_execution_authorized"] is False


def test_dry_profile_identity_changes_with_allocator_source_commit() -> None:
    first = builder.profile_id_for_source_commit("a" * 40)
    second = builder.profile_id_for_source_commit("b" * 40)

    assert first == f"{builder.PROFILE_ID_PREFIX}-{'a' * 40}"
    assert second == f"{builder.PROFILE_ID_PREFIX}-{'b' * 40}"
    assert first != second


def test_builder_fails_closed_on_non_main_or_dirty_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commit = "b" * 40

    def clean_git(_repo: Path, *args: str) -> str:
        if args == ("status", "--porcelain"):
            return ""
        if args == ("merge-base", "--is-ancestor", commit, "origin/main"):
            return ""
        return commit

    monkeypatch.setattr(builder, "_git", clean_git)
    builder.verify_protected_main_checkout(tmp_path, commit)

    monkeypatch.setattr(
        builder,
        "_git",
        lambda _repo, *args: "M file" if args == ("status", "--porcelain") else commit,
    )
    with pytest.raises(
        builder.ProductionProfileBuildError,
        match="checkout_not_exact_clean_main",
    ):
        builder.verify_protected_main_checkout(tmp_path, commit)


def test_builder_rejects_an_invalid_release_identity_before_running_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def git_must_not_run(_repo: Path, *args: str) -> str:
        raise AssertionError(args)

    monkeypatch.setattr(builder, "_git", git_must_not_run)

    with pytest.raises(
        builder.ProductionProfileBuildError,
        match="checkout_not_exact_clean_main",
    ):
        builder.verify_protected_main_checkout(tmp_path, "not-a-sha")


def test_builder_accepts_a_clean_release_commit_after_main_advances(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release_commit = "c" * 40
    newer_main_commit = "d" * 40

    def release_git(_repo: Path, *args: str) -> str:
        if args == ("status", "--porcelain"):
            return ""
        if args == ("merge-base", "--is-ancestor", release_commit, "origin/main"):
            return ""
        if args == ("rev-parse", "HEAD"):
            return release_commit
        if args == ("rev-parse", "origin/main"):
            return newer_main_commit
        raise AssertionError(args)

    monkeypatch.setattr(builder, "_git", release_git)

    builder.verify_protected_main_checkout(tmp_path, release_commit)


def test_materialized_source_verifier_rejects_size_or_digest_drift(tmp_path: Path) -> None:
    root = tmp_path / builder.BUNDLE_ID
    root.mkdir()
    path = root / "appearance.ply"
    path.write_bytes(b"exact-source")
    bundle = {
        "materialized_artifacts": [
            {
                "role": f"role-{index}",
                "production_path": f"/var/lib/blueprint/{builder.BUNDLE_ID}/artifact-{index}",
                "sha256": _digest(path),
                "size_bytes": path.stat().st_size,
            }
            for index in range(5)
        ]
    }
    for index in range(5):
        (root / f"artifact-{index}").write_bytes(path.read_bytes())

    verified = builder.verify_materialized_source_artifacts(bundle, root)
    assert len(verified) == 5

    bundle["materialized_artifacts"][0]["size_bytes"] += 1
    with pytest.raises(
        builder.ProductionProfileBuildError,
        match="production_profile_source_artifact_invalid",
    ):
        builder.verify_materialized_source_artifacts(bundle, root)
