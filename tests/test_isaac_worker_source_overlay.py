from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

from blueprint_pipeline.g1_kitchen_bundle_compatibility import (
    CANONICAL_CLEAN_SOURCE_DIRTY_PATCH_SHA256,
)
from blueprint_pipeline.isaac_worker_source_overlay import (
    BUILD_METHOD,
    DEFAULT_BASE_IMAGE_DIGEST,
    main,
    prepare_source_overlay,
    verify_registry_overlay,
)


def _head(repo: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _clean_fixture_repo(tmp_path: Path) -> Path:
    source = Path(__file__).resolve().parents[1]
    repo = tmp_path / "repo"
    for relative in (
        Path("src/blueprint_pipeline/__init__.py"),
        Path("src/blueprint_pipeline/reconstruction_isaac_bootstrap.py"),
        Path("pyproject.toml"),
        Path("README.md"),
        Path("LICENSE"),
    ):
        destination = repo / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, destination)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Blueprint Test",
            "-c",
            "user.email=blueprint-test@example.invalid",
            "commit",
            "-q",
            "-m",
            "fixture",
        ],
        cwd=repo,
        check=True,
    )
    return repo


def test_source_overlay_is_reproducible_and_hides_lower_source(tmp_path: Path) -> None:
    repo = _clean_fixture_repo(tmp_path)
    first = prepare_source_overlay(
        repo_root=repo,
        output_dir=tmp_path / "first",
        base_image_digest=DEFAULT_BASE_IMAGE_DIGEST,
        target_image_ref="docker.io/example/isaac-worker:nurec-exact-source",
        source_commit=_head(repo),
    )
    second = prepare_source_overlay(
        repo_root=repo,
        output_dir=tmp_path / "second",
        base_image_digest=DEFAULT_BASE_IMAGE_DIGEST,
        target_image_ref="docker.io/example/isaac-worker:nurec-exact-source",
        source_commit=_head(repo),
    )

    assert first["status"] == "ready"
    assert first["layer_sha256"] == second["layer_sha256"]
    assert first["source_manifest_sha256"] == second["source_manifest_sha256"]
    assert first["source_dirty_patch_sha256"] == CANONICAL_CLEAN_SOURCE_DIRTY_PATCH_SHA256
    with tarfile.open(first["layer_path"], "r:gz") as archive:
        names = archive.getnames()
        receipt = json.load(
            archive.extractfile("opt/blueprint/isaac_worker_source_overlay.v1.json")
        )
    assert "app/src/.wh..wh..opq" in names
    assert "app/src/blueprint_pipeline/reconstruction_isaac_bootstrap.py" in names
    assert "app/pyproject.toml" in names
    assert receipt["build_method"] == BUILD_METHOD
    assert receipt["source_manifest_sha256"] == first["source_manifest_sha256"]
    assert receipt["runtime_canary_required"] is True


def test_source_overlay_refuses_dirty_or_mutable_inputs(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    blocked = prepare_source_overlay(
        repo_root=repo,
        output_dir=tmp_path,
        base_image_digest="docker.io/example/isaac:latest",
        target_image_ref="docker.io/example/isaac-worker:latest",
        source_commit="f" * 40,
    )
    assert blocked["status"] == "blocked"
    assert "isaac_worker_source_overlay_base_not_digest_pinned" in blocked["blockers"]
    assert "isaac_worker_source_overlay_target_not_versioned" in blocked["blockers"]
    assert "isaac_worker_source_overlay_source_commit_mismatch" in blocked["blockers"]


def _plan() -> dict:
    source_commit = "a" * 40
    value = {
        "schema_version": "isaac_worker_source_overlay.v1",
        "status": "ready",
        "blockers": [],
        "build_method": BUILD_METHOD,
        "source_commit": source_commit,
        "source_dirty_patch_sha256": CANONICAL_CLEAN_SOURCE_DIRTY_PATCH_SHA256,
        "base_image_digest": DEFAULT_BASE_IMAGE_DIGEST,
        "target_image_ref": "docker.io/example/isaac-worker:nurec-exact-source",
        "source_manifest_sha256": "sha256:" + "b" * 64,
        "expected_final_environment": {
            "BLUEPRINT_SOURCE_COMMIT": source_commit,
            "BLUEPRINT_SOURCE_DIRTY_PATCH_SHA256": CANONICAL_CLEAN_SOURCE_DIRTY_PATCH_SHA256,
            "BLUEPRINT_WORKER_IMAGE_FAMILY": "isaac-eval-worker",
            "BLUEPRINT_ISAAC_SIM_MAJOR_VERSION": "6",
            "BLUEPRINT_WORKER_IMAGE_BUILD_METHOD": BUILD_METHOD,
            "BLUEPRINT_WORKER_BASE_IMAGE_DIGEST": DEFAULT_BASE_IMAGE_DIGEST,
            "BLUEPRINT_WORKER_SOURCE_MANIFEST_SHA256": "sha256:" + "b" * 64,
        },
    }
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest

    value["plan_digest"] = canonical_digest(value, digest_field="plan_digest")
    return value


def test_registry_overlay_verification_binds_layer_prefix_and_runtime_config() -> None:
    plan = _plan()
    base_layers = [{"digest": "sha256:" + "c" * 64, "size": 10}]
    source_layer = "sha256:" + "d" * 64
    environment = [f"{key}={value}" for key, value in plan["expected_final_environment"].items()]
    environment.append(f"BLUEPRINT_WORKER_SOURCE_LAYER_DIGEST={source_layer}")
    result = verify_registry_overlay(
        plan=plan,
        base_manifest={"layers": base_layers},
        final_manifest={"layers": [*base_layers, {"digest": source_layer, "size": 20}]},
        final_config={
            "architecture": "amd64",
            "os": "linux",
            "config": {
                "Env": environment,
                "User": "blueprint",
                "WorkingDir": "/workspace",
                "Entrypoint": ["blueprint-run-robot-eval-worker"],
            },
        },
        resolved_image_digest="sha256:" + "e" * 64,
    )
    assert result["status"] == "verified"
    assert result["base_layers_preserved_exactly"] is True
    assert result["runtime_canary_completed"] is False

    tampered = copy.deepcopy(plan)
    tampered["expected_final_environment"]["BLUEPRINT_SOURCE_COMMIT"] = "f" * 40
    blocked = verify_registry_overlay(
        plan=tampered,
        base_manifest={"layers": base_layers},
        final_manifest={"layers": [*base_layers, {"digest": source_layer, "size": 20}]},
        final_config={
            "architecture": "amd64",
            "os": "linux",
            "config": {
                "Env": environment,
                "User": "blueprint",
                "WorkingDir": "/workspace",
                "Entrypoint": ["blueprint-run-robot-eval-worker"],
            },
        },
        resolved_image_digest="sha256:" + "e" * 64,
    )
    assert blocked["status"] == "blocked"
    assert "isaac_worker_source_overlay_plan_invalid" in blocked["blockers"]


def test_verify_registry_cli_writes_atomic_result(tmp_path: Path) -> None:
    plan = _plan()
    base_layers = [{"digest": "sha256:" + "c" * 64, "size": 10}]
    source_layer = "sha256:" + "d" * 64
    environment = [
        f"{key}={value}" for key, value in plan["expected_final_environment"].items()
    ]
    environment.append(f"BLUEPRINT_WORKER_SOURCE_LAYER_DIGEST={source_layer}")
    values = {
        "plan.json": plan,
        "base.json": {"layers": base_layers},
        "final.json": {
            "layers": [*base_layers, {"digest": source_layer, "size": 20}]
        },
        "config.json": {
            "architecture": "amd64",
            "os": "linux",
            "config": {
                "Env": environment,
                "User": "blueprint",
                "WorkingDir": "/workspace",
                "Entrypoint": ["blueprint-run-robot-eval-worker"],
            },
        },
    }
    for name, value in values.items():
        (tmp_path / name).write_text(json.dumps(value), encoding="utf-8")
    output = tmp_path / "result.json"

    assert (
        main(
            [
                "verify-registry",
                "--plan",
                str(tmp_path / "plan.json"),
                "--base-manifest",
                str(tmp_path / "base.json"),
                "--final-manifest",
                str(tmp_path / "final.json"),
                "--final-config",
                str(tmp_path / "config.json"),
                "--resolved-digest",
                "sha256:" + "e" * 64,
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "verified"
