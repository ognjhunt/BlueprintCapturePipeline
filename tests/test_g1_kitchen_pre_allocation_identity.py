from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from typing import Any

from blueprint_pipeline.g1_kitchen_worker_image_evidence import (
    assemble_worker_image_runtime_evidence,
    validate_worker_image_runtime_evidence,
)

IMAGE_HASH = "d" * 64
SOURCE_COMMIT = "b" * 40
DIRTY_PATCH = "e" * 64

SEALED_HEALTHCHECK_PATH = (
    Path(__file__).resolve().parents[1]
    / "deploy"
    / "docker"
    / "robot_eval_worker"
    / "groot_oscar_closed_loop"
    / "groot_oscar_closed_loop_image_healthcheck.py"
)


def _load_sealed_healthcheck_module():
    spec = importlib.util.spec_from_file_location(
        "groot_oscar_closed_loop_image_healthcheck", SEALED_HEALTHCHECK_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sealed_runtime_metadata() -> dict[str, Any]:
    module = _load_sealed_healthcheck_module()
    return module.build_runtime_metadata(
        env={
            "BLUEPRINT_WORKER_IMAGE_FAMILY": "isaac-eval-worker",
            "BLUEPRINT_WORKER_IMAGE_VARIANT": "groot-oscar-closed-loop",
            "BLUEPRINT_SIMULATOR_FRAMEWORK": "isaac_sim",
            "BLUEPRINT_ISAAC_SIM_MAJOR_VERSION": "6",
        },
        isaac_imported=True,
        g1_exists=True,
        asset_binding_valid=True,
        official_assets_exist=True,
        source_commit=SOURCE_COMMIT,
        dirty_patch=DIRTY_PATCH,
    )


def _canary(name: str) -> dict[str, Any]:
    canary = {
        "status": "passed",
        "image_digest": f"sha256:{IMAGE_HASH}",
        "provider_allocation_id": "do-1",
        "run_id": "run-1",
        "attempt_id": "attempt-1",
        "launch_nonce": "nonce-1",
    }
    if name == "review":
        canary.update({"width": 640, "height": 480})
    return canary


def _worker_evidence() -> dict[str, Any]:
    return assemble_worker_image_runtime_evidence(
        image_digest=f"sha256:{IMAGE_HASH}",
        source_commit=SOURCE_COMMIT,
        source_dirty_patch_sha256=DIRTY_PATCH,
        build_healthcheck={
            "status": "passed",
            "runtime_metadata": _sealed_runtime_metadata(),
        },
        fast_canary=_canary("fast"),
        review_canary=_canary("review"),
        teardown={"api_confirmed": True, "terminal_state": "terminated"},
        final_inventory={"api_confirmed": True, "live_resource_count": 0},
    )


def test_sealed_healthcheck_payload_passes_real_evidence_validator() -> None:
    metadata = _sealed_runtime_metadata()
    assert metadata["configured_g1_usd_exists"] is True
    assert metadata["configured_g1_asset_binding_valid"] is True
    evidence = _worker_evidence()
    validation = validate_worker_image_runtime_evidence(
        evidence,
        expected_image_digest=f"sha256:{IMAGE_HASH}",
        expected_source_commit=SOURCE_COMMIT,
        expected_dirty_patch_sha256=DIRTY_PATCH,
    )
    assert validation["blockers"] == []
    assert validation["status"] == "passed"


def test_sealed_healthcheck_claims_stay_distinct() -> None:
    module = _load_sealed_healthcheck_module()
    metadata = module.build_runtime_metadata(
        env={
            "BLUEPRINT_WORKER_IMAGE_FAMILY": "isaac-eval-worker",
            "BLUEPRINT_SIMULATOR_FRAMEWORK": "isaac_sim",
            "BLUEPRINT_ISAAC_SIM_MAJOR_VERSION": "6",
        },
        isaac_imported=True,
        g1_exists=True,
        asset_binding_valid=False,
        official_assets_exist=True,
        source_commit=SOURCE_COMMIT,
        dirty_patch=DIRTY_PATCH,
    )
    assert metadata["configured_g1_usd_exists"] is True
    assert metadata["configured_g1_asset_binding_valid"] is False


# ---------------------------------------------------------------------------
# Source-identity clarity for release image builds (2026-07-11): a clean tree
# produces a NON-EMPTY canonical patch digest, and only the dirty flag may be
# read as evidence of a dirty build.
# ---------------------------------------------------------------------------


def test_canonical_clean_patch_digest_matches_hashing_scheme() -> None:
    from blueprint_pipeline.g1_kitchen_bundle_compatibility import (
        CANONICAL_CLEAN_SOURCE_DIRTY_PATCH_SHA256,
    )

    assert (
        CANONICAL_CLEAN_SOURCE_DIRTY_PATCH_SHA256
        == hashlib.sha256(b"staged\0" + b"unstaged\0").hexdigest()
    )
    # The exact value the 2026-07-11 b15624 build manifest recorded; it was a
    # CLEAN build, not a dirty one.
    assert CANONICAL_CLEAN_SOURCE_DIRTY_PATCH_SHA256 == (
        "1bcd3e1a2eac27ef549049e0162ccce08efc4d13d765f6146d1191494c365454"
    )


def test_clean_and_dirty_source_tree_identity_end_to_end(tmp_path: Path) -> None:
    import subprocess

    from blueprint_pipeline.g1_kitchen_bundle_compatibility import (
        CANONICAL_CLEAN_SOURCE_DIRTY_PATCH_SHA256,
        build_source_tree_identity,
    )

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@e.st"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    (repo / "a.txt").write_text("a\n", encoding="utf-8")
    subprocess.run(["git", "add", "a.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=repo, check=True)

    clean = build_source_tree_identity(repo)
    assert clean["dirty"] is False
    assert clean["untracked_file_count"] == 0
    assert clean["source_dirty_patch_sha256"] == (CANONICAL_CLEAN_SOURCE_DIRTY_PATCH_SHA256)
    assert clean["canonical_clean_patch_sha256"] == (CANONICAL_CLEAN_SOURCE_DIRTY_PATCH_SHA256)

    (repo / "b.txt").write_text("untracked\n", encoding="utf-8")
    dirty = build_source_tree_identity(repo)
    assert dirty["dirty"] is True
    assert dirty["untracked_file_count"] == 1
    assert dirty["source_dirty_patch_sha256"] != (CANONICAL_CLEAN_SOURCE_DIRTY_PATCH_SHA256)


def test_release_image_source_identity_gate_accepts_clean_blocks_dirty() -> None:
    from blueprint_pipeline.g1_kitchen_bundle_compatibility import (
        CANONICAL_CLEAN_SOURCE_DIRTY_PATCH_SHA256,
        evaluate_release_image_source_identity,
    )

    clean_identity = {
        "source_commit": SOURCE_COMMIT,
        "source_dirty_patch_sha256": CANONICAL_CLEAN_SOURCE_DIRTY_PATCH_SHA256,
        "dirty": False,
        "untracked_file_count": 0,
        "identity_includes_staged_unstaged_and_untracked": True,
    }
    accepted = evaluate_release_image_source_identity(clean_identity, push_requested=True)
    assert accepted["status"] == "passed"
    assert accepted["source_worktree_dirty"] is False
    assert accepted["patch_sha256_is_canonical_clean"] is True
    assert accepted["dirty_build_override_used"] is False

    dirty_identity = dict(clean_identity, dirty=True, source_dirty_patch_sha256="f" * 64)
    rejected = evaluate_release_image_source_identity(dirty_identity, push_requested=True)
    assert rejected["status"] == "blocked"
    assert rejected["blockers"] == ["release_image_requires_clean_source_worktree"]
    assert rejected["patch_sha256_is_canonical_clean"] is False

    # Local-only (no push) dirty builds stay allowed for iteration.
    local = evaluate_release_image_source_identity(dirty_identity, push_requested=False)
    assert local["status"] == "passed"

    # The debug override is explicit and recorded; never silent.
    overridden = evaluate_release_image_source_identity(
        dirty_identity, push_requested=True, allow_dirty_release_build=True
    )
    assert overridden["status"] == "passed"
    assert overridden["dirty_build_override_used"] is True


def test_build_script_records_full_source_identity_and_gates_dirty_push() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "build_push_groot_oscar_closed_loop_image.sh"
    ).read_text(encoding="utf-8")
    assert "evaluate_release_image_source_identity" in script
    assert "source_worktree_dirty" in script
    assert "untracked_file_count" in script
    assert "identity_includes_staged_unstaged_and_untracked" in script
    assert "canonical_clean_patch_sha256" in script
    assert "source_identity_release_gate" in script
    assert "BLUEPRINT_ALLOW_DIRTY_GROOT_OSCAR_CLOSED_LOOP_RELEASE_BUILD" in script
    assert '"resolved_digest_ref": registry.get("resolved_digest_ref")' in script
    assert '"layer_count": registry.get("layer_count")' in script
    assert '"total_compressed_size_bytes": registry.get(' in script
    assert '"largest_layer_size_bytes": registry.get(' in script
    assert '"registry_manifest_diagnostic": {' in script
    assert '"build_time_healthcheck": {' in script
    assert 'registry.get("checkpoint_ownership_copyup_detected") is False' in script
    assert "groot_oscar_closed_loop_registry_diagnostic_command_failed" in script
    assert 'git -C "$repo_root" archive --format=tar "$source_commit"' in script
    assert '--metadata-file "$local_build_metadata_file"' in script
    assert '--metadata-file "$publish_build_metadata_file"' in script
    assert 'build_metadata.get("containerimage.digest")' in script
    assert "groot_oscar_closed_loop_buildx_registry_digest_mismatch" in script
    assert "groot_oscar_closed_loop_image_build_requires_clean_source_worktree" in script
    assert '"local_build_completed"' in script
