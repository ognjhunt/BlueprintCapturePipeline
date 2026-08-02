from __future__ import annotations

import copy
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

import jsonschema
import pytest

from blueprint_pipeline.measurement_dlo_lab_runtime_release import (
    RUNTIME_IMAGE,
    build_measurement_dlo_lab_runtime_release,
    validate_measurement_dlo_lab_runtime_release,
)
from blueprint_pipeline.measurement_dlo_lab_vast_bundle import (
    MeasurementDloLabVastBundleError,
    compile_measurement_dlo_lab_input_bundle,
    validate_measurement_dlo_lab_input_bundle_receipt,
)


ROOT = Path(__file__).parents[1]
CORPUS = Path("tests/fixtures/measurement_dlo_lab_cable_v1/corpus.json")
QUALIFICATION_SPLIT_DIGEST = "sha256:" + "f" * 64
CONTROLLER_SCOPE_DIGEST = "sha256:" + "e" * 64


def _clean_source(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    (source / "src").mkdir(parents=True)
    shutil.copytree(ROOT / "src/blueprint_pipeline", source / "src/blueprint_pipeline")
    (source / "scripts").mkdir()
    shutil.copyfile(
        ROOT / "scripts/run_measurement_dlo_lab_bundle.py",
        source / "scripts/run_measurement_dlo_lab_bundle.py",
    )
    fixture = source / CORPUS.parent
    fixture.mkdir(parents=True)
    shutil.copyfile(ROOT / CORPUS, fixture / "corpus.json")
    subprocess.run(["git", "init", "-q"], cwd=source, check=True)
    subprocess.run(["git", "config", "user.email", "fixture@example.com"], cwd=source, check=True)
    subprocess.run(["git", "config", "user.name", "Fixture"], cwd=source, check=True)
    subprocess.run(["git", "add", "."], cwd=source, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=source, check=True)
    return source


def _compile(source: Path, output: Path) -> dict:
    return compile_measurement_dlo_lab_input_bundle(
        repo_root=source,
        corpus_path=source / CORPUS,
        qualification_split_digest=QUALIFICATION_SPLIT_DIGEST,
        controller_scope_digest=CONTROLLER_SCOPE_DIGEST,
        output_path=output,
    )


def test_runtime_release_is_exact_image_source_and_cuda_bound() -> None:
    release = build_measurement_dlo_lab_runtime_release()
    assert release["runtime_image_digest"] == RUNTIME_IMAGE
    assert "@sha256:" in RUNTIME_IMAGE
    assert release["dlo_lab_source_commit"] == ("c5026a9416b03c6bc5186eba13cd4ffd4c0e7796")
    assert release["required_backend"] == "cuda"
    assert release["cpu_fallback_allowed"] is False
    assert release["benchmark_assets_required"] is False
    assert release["production_route_eligible"] is False
    assert validate_measurement_dlo_lab_runtime_release(release) == release
    schema = json.loads(
        (
            ROOT / "docs/schemas/measurement_dlo_lab_runtime_release.v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    jsonschema.validate(release, schema)


def test_bundle_is_clean_commit_bound_deterministic_and_schema_checked(tmp_path: Path) -> None:
    source = _clean_source(tmp_path)
    first_path = tmp_path / "first.zip"
    second_path = tmp_path / "second.zip"
    first = _compile(source, first_path)
    second = _compile(source, second_path)
    assert first["input_bundle_digest"] == second["input_bundle_digest"]
    assert first["execution_request_digests"] == second["execution_request_digests"]
    assert first["request_count"] == 2
    assert first["required_backend"] == "cuda"
    assert first["replay_count"] == 2
    assert first["provider_allocation_performed"] is False
    assert validate_measurement_dlo_lab_input_bundle_receipt(first) == first
    with zipfile.ZipFile(first_path) as archive:
        names = set(archive.namelist())
        assert "bundle_manifest.json" in names
        assert "requests/001.json" in names
        assert "requests/002.json" in names
        assert "scripts/run_measurement_dlo_lab_bundle.py" in names
        assert "src/blueprint_pipeline/measurement_dlo_lab_cable_adapter.py" in names
        manifest = json.loads(archive.read("bundle_manifest.json"))
    assert manifest["source_commit_sha"] == first["source_commit_sha"]
    assert manifest["dlo_lab_source_commit"] == first["dlo_lab_source_commit"]
    assert manifest["runtime_image_digest"] == RUNTIME_IMAGE
    assert manifest["paid_execution_authorized_by_bundle"] is False
    assert all("digest" in row for row in manifest["source_files"])
    schema = json.loads(
        (
            ROOT / "docs/schemas/measurement_dlo_lab_cable_development_corpus.v1.schema.json"
        ).read_text()
    )
    jsonschema.validate(json.loads((source / CORPUS).read_text()), schema)


def test_bundle_refuses_dirty_source_and_receipt_authority_tampering(tmp_path: Path) -> None:
    source = _clean_source(tmp_path)
    marker = source / "src/blueprint_pipeline/dirty.py"
    marker.write_text("DIRTY = True\n", encoding="utf-8")
    with pytest.raises(MeasurementDloLabVastBundleError, match="source_not_clean_commit"):
        _compile(source, tmp_path / "blocked.zip")

    marker.unlink()
    receipt = _compile(source, tmp_path / "valid.zip")
    tampered = copy.deepcopy(receipt)
    tampered["paid_execution_authorized_by_bundle"] = True
    with pytest.raises(
        MeasurementDloLabVastBundleError,
        match="paid_execution_authorized_by_bundle_invalid",
    ):
        validate_measurement_dlo_lab_input_bundle_receipt(tampered)
