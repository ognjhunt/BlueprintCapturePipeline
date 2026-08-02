from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
import zipfile

import pytest

from blueprint_pipeline.measurement_isaac_vast_bundle import (
    MeasurementIsaacVastBundleError,
    compile_measurement_isaac_physx_input_bundle,
    validate_measurement_isaac_physx_input_bundle_receipt,
)


ROOT = Path(__file__).parents[1]
QUALIFICATION_SPLIT_DIGEST = "sha256:" + "f" * 64
CONTROLLER_SCOPE_DIGEST = "sha256:" + "e" * 64


def _clean_source(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    (source / "src").mkdir(parents=True)
    shutil.copytree(ROOT / "src/blueprint_pipeline", source / "src/blueprint_pipeline")
    (source / "scripts").mkdir()
    for name in (
        "measurement_isaac_physx_rigid_worker.py",
        "run_measurement_isaac_physx_bundle.py",
    ):
        shutil.copyfile(ROOT / "scripts" / name, source / "scripts" / name)
    fixture = source / "tests/fixtures/measurement_capture_to_geometry_contact_v1"
    fixture.mkdir(parents=True)
    shutil.copyfile(
        ROOT / "tests/fixtures/measurement_capture_to_geometry_contact_v1/corpus.json",
        fixture / "corpus.json",
    )
    subprocess.run(["git", "init", "-q"], cwd=source, check=True)
    subprocess.run(["git", "config", "user.email", "fixture@example.com"], cwd=source, check=True)
    subprocess.run(["git", "config", "user.name", "Fixture"], cwd=source, check=True)
    subprocess.run(["git", "add", "."], cwd=source, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=source, check=True)
    return source


def test_measurement_isaac_bundle_is_clean_commit_bound_and_deterministic(tmp_path: Path) -> None:
    source = _clean_source(tmp_path)
    first_path = tmp_path / "first.zip"
    second_path = tmp_path / "second.zip"
    kwargs = {
        "repo_root": source,
        "corpus_path": (
            source / "tests/fixtures/measurement_capture_to_geometry_contact_v1/corpus.json"
        ),
        "qualification_split_digest": QUALIFICATION_SPLIT_DIGEST,
        "controller_scope_digest": CONTROLLER_SCOPE_DIGEST,
    }
    first = compile_measurement_isaac_physx_input_bundle(
        **kwargs,
        output_path=first_path,
    )
    second = compile_measurement_isaac_physx_input_bundle(
        **kwargs,
        output_path=second_path,
    )
    assert first["input_bundle_digest"] == second["input_bundle_digest"]
    assert first["execution_request_digests"] == second["execution_request_digests"]
    assert first["request_count"] == 2
    assert first["rtx_openusd_runtime_preflight_required"] is True
    assert first["rtx_renderer"] == "RayTracedLighting"
    assert first["rtx_smoke_resolution"] == [64, 64]
    assert first["rtx_required_output_kinds"] == [
        "rgb",
        "depth",
        "semantic_segmentation",
    ]
    assert first["provider_allocation_performed"] is False
    assert validate_measurement_isaac_physx_input_bundle_receipt(first) == first
    with zipfile.ZipFile(first_path) as archive:
        names = set(archive.namelist())
        assert "bundle_manifest.json" in names
        assert "requests/001.json" in names
        assert "requests/002.json" in names
        assert "scripts/run_measurement_isaac_physx_bundle.py" in names
        assert "src/blueprint_pipeline/core/common.py" in names
        manifest = json.loads(archive.read("bundle_manifest.json"))
    assert manifest["source_commit_sha"] == first["source_commit_sha"]
    assert manifest["runtime_image_digest"] == first["runtime_image_digest"]
    assert manifest["rtx_openusd_runtime_preflight_required"] is True
    assert manifest["rtx_renderer"] == "RayTracedLighting"
    assert manifest["rtx_smoke_resolution"] == [64, 64]
    assert manifest["rtx_required_output_kinds"] == first["rtx_required_output_kinds"]
    assert manifest["paid_execution_authorized_by_bundle"] is False


def test_measurement_isaac_bundle_refuses_dirty_source(tmp_path: Path) -> None:
    source = _clean_source(tmp_path)
    marker = source / "src/blueprint_pipeline/dirty_fixture.py"
    marker.write_text("DIRTY = True\n", encoding="utf-8")
    with pytest.raises(MeasurementIsaacVastBundleError, match="source_not_clean_commit"):
        compile_measurement_isaac_physx_input_bundle(
            repo_root=source,
            corpus_path=(
                source / "tests/fixtures/measurement_capture_to_geometry_contact_v1/corpus.json"
            ),
            qualification_split_digest=QUALIFICATION_SPLIT_DIGEST,
            controller_scope_digest=CONTROLLER_SCOPE_DIGEST,
            output_path=tmp_path / "blocked.zip",
        )
