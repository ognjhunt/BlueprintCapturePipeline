from __future__ import annotations

import json
from pathlib import Path
import subprocess
import zipfile

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.public_scene_artifixer3d_bundle import (
    ArtiFixer3DBundleError,
    DEFAULT_IMAGE,
    SCHEMA_VERSION,
    build_artifixer3d_bundle,
    materialize_artifixer3d_use_attestation,
)
from blueprint_pipeline.public_scene_artifixer3d_candidate_inputs import (
    materialize_artifixer3d_candidate_inputs,
)
from tests.test_public_scene_artifixer3d_candidate_inputs import _preflight


def _git(command: list[str], root: Path) -> str:
    return subprocess.run(
        command, cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def _source(tmp_path: Path) -> tuple[Path, str, str]:
    root = tmp_path / "source"
    root.mkdir()
    _git(["git", "init", "-q"], root)
    _git(["git", "config", "user.name", "Fixture"], root)
    _git(["git", "config", "user.email", "fixture@example.test"], root)
    (root / "LICENSE").write_text("Apache-2.0 fixture\n", encoding="utf-8")
    (root / "default_negative_prompt.pt").write_bytes(b"fixture")
    (root / "model_eval").mkdir()
    (root / "model_eval" / "run_inference.py").write_text(
        "# fixture\n", encoding="utf-8"
    )
    _git(["git", "add", "."], root)
    _git(["git", "commit", "-qm", "fixture"], root)
    return (
        root,
        _git(["git", "rev-parse", "HEAD"], root),
        _git(["git", "rev-parse", "HEAD^{tree}"], root),
    )


def _candidate(tmp_path: Path) -> Path:
    preflight = _preflight(tmp_path / "inputs", count=2, cameras_per_task=2)
    output = tmp_path / "candidate"
    receipt = materialize_artifixer3d_candidate_inputs(
        calibrated_residual_preflight_path=preflight,
        output_root=output,
    )
    return output / f"{receipt['schema_version']}.json"


def _repository(tmp_path: Path) -> Path:
    source_repo = Path(__file__).resolve().parents[1]
    root = tmp_path / "repository"
    (root / "scripts").mkdir(parents=True)
    for name in (
        "run_public_scene_artifixer3d.sh",
        "public_scene_artifixer3d_runner.py",
    ):
        (root / "scripts" / name).write_bytes(
            (source_repo / "scripts" / name).read_bytes()
        )
    _git(["git", "init", "-q"], root)
    _git(["git", "config", "user.name", "Fixture"], root)
    _git(["git", "config", "user.email", "fixture@example.test"], root)
    _git(["git", "add", "."], root)
    _git(["git", "commit", "-qm", "fixture"], root)
    return root


def _attestation(candidate: Path, path: Path) -> Path:
    materialize_artifixer3d_use_attestation(
        candidate_inputs_receipt_path=candidate,
        output_path=path,
        authorized_by="fixture_user",
    )
    return path


def test_seals_two_task_bundle_and_rehearses_exact_entrypoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, commit, tree = _source(tmp_path)
    import blueprint_pipeline.public_scene_artifixer3d_bundle as subject

    monkeypatch.setattr(subject, "ARTIFIXER_COMMIT", commit)
    monkeypatch.setattr(subject, "ARTIFIXER_TREE", tree)
    candidate = _candidate(tmp_path)
    receipt = build_artifixer3d_bundle(
        candidate_inputs_receipt_path=candidate,
        use_attestation_path=_attestation(candidate, tmp_path / "attestation.json"),
        artifixer_source_directory=source,
        output_root=tmp_path / "bundle",
        repository_root=_repository(tmp_path),
        allowed_active_instance_ids=[12, 9, 12],
        artifixer3d_steps=10,
    )

    assert receipt["schema_version"] == SCHEMA_VERSION
    assert receipt["status"] == "sealed_rehearsal_passed_no_upload_no_execution"
    assert receipt["replacement_object_count"] == 2
    assert receipt["allowed_active_instance_ids"] == [9, 12]
    assert receipt["container_image"] == DEFAULT_IMAGE
    assert receipt["provider_mutations_performed"] == 0
    assert receipt["local_rehearsal"]["status"] == "passed"
    assert receipt["local_rehearsal"]["paid_inference_performed"] is False
    assert receipt["receipt_digest"] == canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    bundle = Path(receipt["bundle"]["path"])
    with zipfile.ZipFile(bundle) as archive:
        names = set(archive.namelist())
        request = json.loads(
            archive.read("provider_runtime/artifixer3d_runtime_request.json")
        )
        manifest = json.loads(
            archive.read("provider_runtime/artifixer3d_bundle_manifest.json")
        )
    assert "provider_runtime/run_public_scene_artifixer3d.sh" in names
    assert "provider_runtime/public_scene_artifixer3d_runner.py" in names
    assert not any(name.endswith("artifixer-1.3b.pt") for name in names)
    assert request["source_object_restoration_permitted"] is False
    assert request["outside_exact_support_changed_pixels_permitted"] == 0
    assert manifest["contains_raw_dataset_bytes"] is False
    assert manifest["contains_model_weights"] is False


def test_rejects_tampered_candidate_or_dirty_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, commit, tree = _source(tmp_path)
    import blueprint_pipeline.public_scene_artifixer3d_bundle as subject

    monkeypatch.setattr(subject, "ARTIFIXER_COMMIT", commit)
    monkeypatch.setattr(subject, "ARTIFIXER_TREE", tree)
    candidate = _candidate(tmp_path)
    attestation = _attestation(candidate, tmp_path / "attestation.json")
    value = json.loads(candidate.read_text(encoding="utf-8"))
    value["repair_target_semantics"][
        "source_washer_or_notebook_restoration_permitted"
    ] = True
    candidate.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ArtiFixer3DBundleError, match="candidate_receipt_invalid"):
        build_artifixer3d_bundle(
            candidate_inputs_receipt_path=candidate,
            use_attestation_path=attestation,
            artifixer_source_directory=source,
            output_root=tmp_path / "bundle-a",
            repository_root=_repository(tmp_path),
        )

    candidate = _candidate(tmp_path / "other")
    attestation = _attestation(candidate, tmp_path / "other-attestation.json")
    (source / "untracked.txt").write_text("dirty", encoding="utf-8")
    with pytest.raises(ArtiFixer3DBundleError, match="source_invalid"):
        build_artifixer3d_bundle(
            candidate_inputs_receipt_path=candidate,
            use_attestation_path=attestation,
            artifixer_source_directory=source,
            output_root=tmp_path / "bundle-b",
            repository_root=_repository(tmp_path / "other"),
        )
