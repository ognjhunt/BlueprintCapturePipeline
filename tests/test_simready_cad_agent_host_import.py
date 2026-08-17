from __future__ import annotations

import json
import importlib.util
from concurrent.futures import ThreadPoolExecutor
import os
import shutil
import subprocess
import sys
import stat
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest
from PIL import Image

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.cad_agent_review_media import (
    materialize_cad_agent_visual_comparison,
    seal_cad_agent_visual_reference_review,
    selected_cad_agent_visual_review,
    validate_cad_agent_visual_reference_review,
)
from blueprint_pipeline.simready_cad_agent_contract import (
    INSPECTION_SCHEMA_VERSION,
    file_record,
    seal_cad_agent_execution_receipt,
    seal_cad_agent_output,
    seal_cad_agent_matrix,
    seal_cad_agent_reference_manifest,
    seal_cad_agent_request,
    validate_cad_agent_output,
)
from blueprint_pipeline.simready_cad_agent_host_import import (
    SimReadyCadAgentHostImportError,
    materialize_cad_visual_review_host_rematerialization as _materialize_review,
    materialize_simready_cad_agent_host_import as _materialize_import,
    validate_cad_visual_review_host_rematerialization,
    validate_simready_cad_agent_host_import,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
FREEZES = {
    "task_a_washer_door_open": REPO_ROOT
    / "docs/arm_decision_proof_v1/manifests/third_scene_840920_task_a_freeze.v1.json",
    "task_b_notebook_relocation": REPO_ROOT
    / "docs/arm_decision_proof_v1/manifests/third_scene_840920_task_b_freeze.v1.json",
}
ASSETS = {
    "task_a_washer_door_open": "840920_simready_washer_candidate",
    "task_b_notebook_relocation": "840920_simready_notebook_candidate",
}


def materialize_simready_cad_agent_host_import(**kwargs):
    return _materialize_import(owner_uid=os.getuid(), owner_gid=os.getgid(), **kwargs)


def materialize_cad_visual_review_host_rematerialization(**kwargs):
    source_review_path = Path(kwargs["source_visual_review_path"])
    source_review = json.loads(source_review_path.read_text())
    source_record = file_record(source_review_path)
    kwargs.setdefault(
        "expected_source_visual_review_digest", source_review["review_digest"]
    )
    kwargs.setdefault("expected_source_visual_review_sha256", source_record["sha256"])
    kwargs.setdefault(
        "expected_source_visual_review_size_bytes", source_record["size_bytes"]
    )
    return _materialize_review(owner_uid=os.getuid(), owner_gid=os.getgid(), **kwargs)


def _validate_review_receipt(receipt: dict, source_review_path: Path):
    source_review = json.loads(source_review_path.read_text())
    source_record = file_record(source_review_path)
    return validate_cad_visual_review_host_rematerialization(
        receipt,
        expected_source_visual_review_digest=source_review["review_digest"],
        expected_source_visual_review_sha256=source_record["sha256"],
        expected_source_visual_review_size_bytes=source_record["size_bytes"],
    )


def _write(path: Path, payload: str | bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, bytes):
        path.write_bytes(payload)
    else:
        path.write_text(payload, encoding="utf-8")
    return path


def _png(path: Path, color: tuple[int, int, int]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 24), color).save(path)
    return path


def _backend(root: Path, backend_id: str) -> dict:
    archive = root / "sources" / f"{backend_id}.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        with ZipFile(archive, "w", compression=ZIP_DEFLATED) as bundle:
            bundle.comment = b"1" * 40
            bundle.writestr("LICENSE", "MIT\n")
    return {
        "backend_id": backend_id,
        "execution_mode": (
            "codex_skill_step_first"
            if backend_id == "earthtojake_text_to_cad"
            else "codex_agent_direct_repo_route"
        ),
        "agent_authored_geometry": True,
        "deterministic_geometry_generator_used": False,
        "graph_geometry_used_for_cad_authoring": False,
        "deterministic_format_conversion_only": True,
        "repository_url": (
            "https://github.com/earthtojake/text-to-cad"
            if backend_id == "earthtojake_text_to_cad"
            else "https://github.com/Pan-Chera/Multi-Agent-CAD"
        ),
        "commit": "1" * 40,
        "tree": "2" * 40,
        "source_archive": file_record(archive),
        "license": "MIT",
        "model_id": "fixture-agent-model",
    }


def _candidate(
    root: Path,
    *,
    task_id: str,
    backend_id: str,
    slot: int,
    reference_manifest_path: Path | None = None,
) -> tuple[Path, dict]:
    asset_id = ASSETS[task_id]
    candidate = root / task_id / backend_id
    brief = _write(candidate / "brief.md", f"Author {asset_id}\n")
    if reference_manifest_path is None:
        reference = _png(candidate / "reference.png", (20 * slot, 40, 60))
        manifest = seal_cad_agent_reference_manifest(
            scene_id="840920",
            objects=[
                {
                    "replacement_slot": slot,
                    "task_id": task_id,
                    "asset_id": asset_id,
                    "task_freeze_path": FREEZES[task_id],
                    "reference_image_paths": [reference],
                }
            ],
        )
        reference_manifest_path = candidate / "reference_manifest.v1.json"
        reference_manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
    request = seal_cad_agent_request(
        request_id=f"840920-{slot}-{backend_id}",
        scene_id="840920",
        task_id=task_id,
        asset_id=asset_id,
        replacement_slot=slot,
        backend=_backend(root, backend_id),
        task_freeze_path=FREEZES[task_id],
        cad_brief_path=brief,
        metric_envelope_mm=(
            [600.112, 604.104004, 847.564026]
            if slot == 1
            else [356.260028, 429.631028, 299.764008]
        ),
        reference_manifest_path=reference_manifest_path,
    )
    generator = _write(candidate / "candidate.py", "# agent-authored source\n")
    step = _write(candidate / "candidate.step", f"STEP-{task_id}".encode())
    snapshot = _png(candidate / "candidate_iso.png", (60, 80, 100))
    envelope = request["inputs"]["metric_envelope_mm"]
    inspection = {
        "schema_version": INSPECTION_SCHEMA_VERSION,
        "status": "passed",
        "step": file_record(step),
        "measured_envelope_mm": envelope,
        "measured_center_mm": [0.0, 0.0, envelope[2] / 2.0],
        "topology": {"face_count": 6, "edge_count": 12},
        "inspector": {
            "backend": "build123d_import_step",
            "build123d_version": "fixture",
            "ocp_version": "fixture",
            "python_version": "fixture",
            "module_source": file_record(
                REPO_ROOT / "src/blueprint_pipeline/simready_cad_agent_contract.py"
            ),
        },
        "claim_boundary": {
            "step_bytes_reopened": True,
            "metric_envelope_measured": True,
            "visual_quality_qualified": False,
            "simready_qualified": False,
            "physical_equivalence": False,
        },
        "receipt_digest": "",
    }
    inspection["receipt_digest"] = canonical_digest(
        inspection, digest_field="receipt_digest"
    )
    inspection_path = candidate / "inspection.v1.json"
    inspection_path.write_text(json.dumps(inspection, sort_keys=True) + "\n")
    execution = seal_cad_agent_execution_receipt(
        request=request,
        generator_source_path=generator,
        cad_brief_path=brief,
        output_step_path=step,
        event_rows=[{"event": "agent_authored_step", "status": "passed"}],
    )
    execution_path = candidate / "execution.v1.json"
    execution_path.write_text(json.dumps(execution, sort_keys=True) + "\n")
    output = seal_cad_agent_output(
        request=request,
        generator_source_path=generator,
        step_path=step,
        inspection_receipt_path=inspection_path,
        snapshot_paths=[snapshot],
        execution_receipt_path=execution_path,
        measured_envelope_mm=envelope,
        actual_cost_usd=(
            0.0 if backend_id == "earthtojake_text_to_cad" else 0.75
        ),
    )
    output_path = candidate / "cad_agent_output.v1.json"
    output_path.write_text(json.dumps(output, sort_keys=True) + "\n")
    return output_path, output


def _staged_fixture(
    tmp_path: Path,
    *,
    task_id: str = "task_a_washer_door_open",
    backend_id: str = "earthtojake_text_to_cad",
    slot: int = 1,
) -> tuple[Path, Path, Path, dict]:
    historical = tmp_path / "historical-laptop" / "third_scene_dual_task_e2e"
    source_path, output = _candidate(
        historical, task_id=task_id, backend_id=backend_id, slot=slot
    )
    staged = tmp_path / "host-stage"
    shutil.copytree(historical, staged)
    staged_receipt = staged / source_path.relative_to(historical)
    return historical, staged, staged_receipt, output


def _mappings(historical: Path, staged: Path):
    return [(str(historical), staged), (str(REPO_ROOT), REPO_ROOT)]


def _all_file_records(value):
    records = []
    if isinstance(value, dict):
        if {"path", "sha256", "size_bytes"}.issubset(value):
            records.append(value)
        for item in value.values():
            records.extend(_all_file_records(item))
    elif isinstance(value, list):
        for item in value:
            records.extend(_all_file_records(item))
    return records


@pytest.mark.parametrize(
    ("task_id", "backend_id", "slot"),
    [
        ("task_a_washer_door_open", "earthtojake_text_to_cad", 1),
        ("task_b_notebook_relocation", "pan_chera_multi_agent_cad", 2),
    ],
)
def test_imports_task_candidate_and_every_transitive_binding_to_host_paths(
    tmp_path: Path, task_id: str, backend_id: str, slot: int
) -> None:
    historical, staged, source_receipt, source = _staged_fixture(
        tmp_path, task_id=task_id, backend_id=backend_id, slot=slot
    )
    destination = tmp_path / "host-resident" / f"{task_id}-{backend_id}"
    receipt = materialize_simready_cad_agent_host_import(
        source_receipt_path=source_receipt,
        destination_root=destination,
        source_prefix_mappings=_mappings(historical, staged),
    )

    assert validate_simready_cad_agent_host_import(receipt) == receipt
    imported_path = Path(receipt["imported_cad_agent_output"]["path"])
    imported = json.loads(imported_path.read_text())
    assert validate_cad_agent_output(imported, verify_files=True) == imported
    assert imported["request"]["task_id"] == task_id
    assert imported["request"]["backend"]["backend_id"] == backend_id
    assert imported["claim_boundary"] == source["claim_boundary"]
    assert imported["artifacts"]["step"]["sha256"] == source["artifacts"]["step"][
        "sha256"
    ]
    assert imported["receipt_digest"] != source["receipt_digest"]
    for record in _all_file_records(imported):
        path = Path(record["path"])
        assert path.is_relative_to(destination)
        assert path.is_file() and not path.is_symlink()
    serialized = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in destination.rglob("*.json")
    )
    assert str(historical) not in serialized
    assert str(REPO_ROOT) not in serialized
    assert receipt["geometry_bytes_modified"] is False
    assert receipt["geometry_generated"] is False
    assert receipt["ownership"]["owner_uid"] == os.getuid()
    assert receipt["ownership"]["owner_gid"] == os.getgid()
    assert receipt["ownership"]["service_account_readback_passed"] is True
    roles = {row["role"] for row in receipt["bindings"]}
    assert "artifacts.inspection_receipt.inspector.module_source" in roles
    assert "execution.execution_receipt.output_step" in roles
    assert "request.inputs.reference_manifest" in roles
    assert all(
        row["source_bytes_preserved_exactly"] is True
        for row in receipt["bindings"]
        if "source_schema_version" not in row
    )


def test_rejects_unmapped_absolute_bound_path_before_destination_mutation(
    tmp_path: Path,
) -> None:
    historical, staged, source_receipt, _source = _staged_fixture(tmp_path)
    destination = tmp_path / "imported"
    with pytest.raises(
        SimReadyCadAgentHostImportError,
        match="source_prefix_unmapped_or_ambiguous",
    ):
        materialize_simready_cad_agent_host_import(
            source_receipt_path=source_receipt,
            destination_root=destination,
            source_prefix_mappings=[(str(historical), staged)],
        )
    assert not destination.exists()


def test_rejects_overlapping_source_prefix_mappings(tmp_path: Path) -> None:
    historical, staged, source_receipt, _source = _staged_fixture(tmp_path)
    with pytest.raises(
        SimReadyCadAgentHostImportError, match="source_prefix_mapping_ambiguous"
    ):
        materialize_simready_cad_agent_host_import(
            source_receipt_path=source_receipt,
            destination_root=tmp_path / "imported",
            source_prefix_mappings=[
                (str(historical), staged),
                (str(historical / "task_a_washer_door_open"), staged),
            ],
        )


def test_rejects_symlinked_source_mapping_root(tmp_path: Path) -> None:
    historical, staged, source_receipt, _source = _staged_fixture(tmp_path)
    root_link = tmp_path / "stage-link"
    root_link.symlink_to(staged, target_is_directory=True)
    with pytest.raises(
        SimReadyCadAgentHostImportError,
        match="source_prefix_mapping_invalid",
    ):
        materialize_simready_cad_agent_host_import(
            source_receipt_path=source_receipt,
            destination_root=tmp_path / "imported",
            source_prefix_mappings=[
                (str(historical), root_link),
                (str(REPO_ROOT), REPO_ROOT),
            ],
        )


def test_rejects_digest_drift_and_symlink_sources(tmp_path: Path) -> None:
    historical, staged, source_receipt, source = _staged_fixture(tmp_path)
    step_relative = Path(source["artifacts"]["step"]["path"]).relative_to(historical)
    staged_step = staged / step_relative
    staged_step.write_bytes(b"drift")
    with pytest.raises(SimReadyCadAgentHostImportError, match="source_digest_drift"):
        materialize_simready_cad_agent_host_import(
            source_receipt_path=source_receipt,
            destination_root=tmp_path / "drifted",
            source_prefix_mappings=_mappings(historical, staged),
        )
    assert not (tmp_path / "drifted").exists()

    staged_step.unlink()
    staged_step.symlink_to(staged / "does-not-matter")
    with pytest.raises(
        SimReadyCadAgentHostImportError, match="source_symlink_or_missing"
    ):
        materialize_simready_cad_agent_host_import(
            source_receipt_path=source_receipt,
            destination_root=tmp_path / "symlinked",
            source_prefix_mappings=_mappings(historical, staged),
        )


def test_rejects_intermediate_symlink_even_when_it_resolves_inside_mapping_root(
    tmp_path: Path,
) -> None:
    historical, staged, source_receipt, _source = _staged_fixture(tmp_path)
    task_dir = staged / "task_a_washer_door_open"
    real_task_dir = staged / "task_a_washer_door_open-real"
    task_dir.rename(real_task_dir)
    task_dir.symlink_to(real_task_dir.name, target_is_directory=True)
    with pytest.raises(
        SimReadyCadAgentHostImportError, match="source_symlink_or_missing"
    ):
        materialize_simready_cad_agent_host_import(
            source_receipt_path=source_receipt,
            destination_root=tmp_path / "symlink-component",
            source_prefix_mappings=_mappings(historical, staged),
        )
def test_rejects_self_consistent_relative_path_traversal(tmp_path: Path) -> None:
    historical, staged, source_receipt, _source = _staged_fixture(tmp_path)
    value = json.loads(source_receipt.read_text())
    value["request"]["backend"]["source_archive"]["path"] = "../escape.zip"
    value["request"]["request_digest"] = canonical_digest(
        value["request"], digest_field="request_digest"
    )
    value["request_digest"] = value["request"]["request_digest"]
    value["receipt_digest"] = canonical_digest(value, digest_field="receipt_digest")
    source_receipt.write_text(json.dumps(value, sort_keys=True) + "\n")
    with pytest.raises(SimReadyCadAgentHostImportError, match="source_path_traversal"):
        materialize_simready_cad_agent_host_import(
            source_receipt_path=source_receipt,
            destination_root=tmp_path / "traversal",
            source_prefix_mappings=_mappings(historical, staged),
        )


def test_rejects_self_consistent_unknown_absolute_metadata_path(tmp_path: Path) -> None:
    historical, staged, source_receipt, _source = _staged_fixture(tmp_path)
    value = json.loads(source_receipt.read_text())
    value["unexpected_future_metadata"] = {
        "unadapted_absolute_path": "/Users/other-user/private/source.png"
    }
    value["receipt_digest"] = canonical_digest(value, digest_field="receipt_digest")
    source_receipt.write_text(json.dumps(value, sort_keys=True) + "\n")
    destination = tmp_path / "unknown-absolute"
    with pytest.raises(
        SimReadyCadAgentHostImportError,
        match="unadapted_absolute_path",
    ):
        materialize_simready_cad_agent_host_import(
            source_receipt_path=source_receipt,
            destination_root=destination,
            source_prefix_mappings=_mappings(historical, staged),
        )
    assert not destination.exists()


def test_rejects_unknown_absolute_path_inside_nested_inspection_receipt(
    tmp_path: Path,
) -> None:
    historical, staged, source_receipt, source = _staged_fixture(tmp_path)
    inspection_source = Path(source["artifacts"]["inspection_receipt"]["path"])
    inspection_path = staged / inspection_source.relative_to(historical)
    inspection = json.loads(inspection_path.read_text())
    inspection["unexpected_future_metadata"] = {
        "unadapted_absolute_path": "/Users/other-user/private/inspection.bin"
    }
    inspection["receipt_digest"] = canonical_digest(
        inspection, digest_field="receipt_digest"
    )
    inspection_path.write_text(json.dumps(inspection, sort_keys=True) + "\n")
    output = json.loads(source_receipt.read_text())
    inspection_record = file_record(inspection_path)
    inspection_record["path"] = str(inspection_source)
    output["artifacts"]["inspection_receipt"] = inspection_record
    output["receipt_digest"] = canonical_digest(output, digest_field="receipt_digest")
    source_receipt.write_text(json.dumps(output, sort_keys=True) + "\n")
    with pytest.raises(
        SimReadyCadAgentHostImportError, match="unadapted_absolute_path"
    ):
        materialize_simready_cad_agent_host_import(
            source_receipt_path=source_receipt,
            destination_root=tmp_path / "nested-unknown",
            source_prefix_mappings=_mappings(historical, staged),
        )


def test_rejects_symlink_source_receipt_and_existing_destination(tmp_path: Path) -> None:
    historical, staged, source_receipt, _source = _staged_fixture(tmp_path)
    symlink = tmp_path / "source-link.json"
    symlink.symlink_to(source_receipt)
    with pytest.raises(
        SimReadyCadAgentHostImportError, match="source_receipt_invalid"
    ):
        materialize_simready_cad_agent_host_import(
            source_receipt_path=symlink,
            destination_root=tmp_path / "from-link",
            source_prefix_mappings=_mappings(historical, staged),
        )
    destination = tmp_path / "existing"
    destination.mkdir()
    with pytest.raises(SimReadyCadAgentHostImportError, match="destination_exists"):
        materialize_simready_cad_agent_host_import(
            source_receipt_path=source_receipt,
            destination_root=destination,
            source_prefix_mappings=_mappings(historical, staged),
        )


def test_import_receipt_rejects_self_consistent_claim_or_binding_tamper(
    tmp_path: Path,
) -> None:
    historical, staged, source_receipt, _source = _staged_fixture(tmp_path)
    receipt = materialize_simready_cad_agent_host_import(
        source_receipt_path=source_receipt,
        destination_root=tmp_path / "imported",
        source_prefix_mappings=_mappings(historical, staged),
    )
    claim_tamper = json.loads(json.dumps(receipt))
    claim_tamper["geometry_bytes_modified"] = True
    claim_tamper["import_digest"] = canonical_digest(
        claim_tamper, digest_field="import_digest"
    )
    with pytest.raises(SimReadyCadAgentHostImportError, match="receipt_invalid"):
        validate_simready_cad_agent_host_import(claim_tamper)

    binding_tamper = json.loads(json.dumps(receipt))
    binding_tamper["bindings"][0]["output"]["path"] = str(
        tmp_path / "outside.bin"
    )
    binding_tamper["import_digest"] = canonical_digest(
        binding_tamper, digest_field="import_digest"
    )
    with pytest.raises(SimReadyCadAgentHostImportError, match="binding_invalid"):
        validate_simready_cad_agent_host_import(binding_tamper)
    coverage_tamper = json.loads(json.dumps(receipt))
    coverage_tamper["bindings"] = coverage_tamper["bindings"][1:]
    coverage_tamper["binding_count"] = len(coverage_tamper["bindings"])
    coverage_tamper["import_digest"] = canonical_digest(
        coverage_tamper, digest_field="import_digest"
    )
    with pytest.raises(
        SimReadyCadAgentHostImportError, match="binding_coverage_invalid"
    ):
        validate_simready_cad_agent_host_import(coverage_tamper)


def test_import_receipt_rejects_service_account_unreadable_mode(tmp_path: Path) -> None:
    historical, staged, source_receipt, _source = _staged_fixture(tmp_path)
    receipt = materialize_simready_cad_agent_host_import(
        source_receipt_path=source_receipt,
        destination_root=tmp_path / "imported",
        source_prefix_mappings=_mappings(historical, staged),
    )
    bound_path = Path(receipt["bindings"][0]["output"]["path"])
    bound_path.chmod(0o600)
    assert stat.S_IMODE(bound_path.stat().st_mode) == 0o600
    with pytest.raises(SimReadyCadAgentHostImportError, match="binding_invalid"):
        validate_simready_cad_agent_host_import(receipt)


def _source_review_chain(
    tmp_path: Path, *, task_ids: tuple[str, ...] = tuple(FREEZES)
):
    historical = tmp_path / "historical-laptop" / "third_scene_dual_task_e2e"
    references = {
        task_id: _png(
            historical / "shared" / f"reference-{slot}.png",
            (30 * slot, 50, 70),
        )
        for slot, task_id in enumerate(task_ids, start=1)
    }
    manifest = seal_cad_agent_reference_manifest(
        scene_id="840920",
        objects=[
            {
                "replacement_slot": slot,
                "task_id": task_id,
                "asset_id": ASSETS[task_id],
                "task_freeze_path": FREEZES[task_id],
                "reference_image_paths": [references[task_id]],
            }
            for slot, task_id in enumerate(task_ids, start=1)
        ],
    )
    manifest_path = historical / "shared" / "reference_manifest.v1.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
    outputs = []
    output_paths = []
    for slot, task_id in enumerate(task_ids, start=1):
        for backend_id in (
            "earthtojake_text_to_cad",
            "pan_chera_multi_agent_cad",
        ):
            output_path, output = _candidate(
                historical,
                task_id=task_id,
                backend_id=backend_id,
                slot=slot,
                reference_manifest_path=manifest_path,
            )
            output_paths.append(output_path)
            outputs.append(output)
    matrix = seal_cad_agent_matrix(
        objects=[
            {
                "replacement_slot": slot,
                "task_id": task_id,
                "asset_id": ASSETS[task_id],
                "candidates": [
                    output
                    for output in outputs
                    if output["request"]["task_id"] == task_id
                ],
            }
            for slot, task_id in enumerate(task_ids, start=1)
        ]
    )
    matrix_path = historical / "shared" / "cad_matrix.v1.json"
    matrix_path.write_text(json.dumps(matrix, sort_keys=True) + "\n")
    media = materialize_cad_agent_visual_comparison(
        matrix_path=matrix_path,
        output_dir=historical / "shared" / "review_media",
        title="Scene 840920 fixture review",
    )
    decisions = []
    for row in media["rows"]:
        reference_digests = [item["sha256"] for item in row["reference_images"]]
        for candidate in row["candidates"]:
            decisions.append(
                {
                    "replacement_slot": row["replacement_slot"],
                    "task_id": row["task_id"],
                    "asset_id": row["asset_id"],
                    "backend_id": candidate["backend_id"],
                    "cad_agent_output_receipt_digest": candidate[
                        "output_receipt_digest"
                    ],
                    "reference_signature": row["reference_signature"],
                    "reviewed_reference_image_digests": reference_digests,
                    "review_status": (
                        "conditionally_admitted_for_construction"
                        if candidate["backend_id"] == "earthtojake_text_to_cad"
                        else "rejected_visible_mismatch"
                    ),
                    "observed_feature_findings": [
                        {
                            "feature_id": "bounded_exterior",
                            "status": "matched",
                            "evidence_reference_image_digests": reference_digests,
                        }
                    ],
                    "visible_mismatch_codes": (
                        []
                        if candidate["backend_id"] == "earthtojake_text_to_cad"
                        else ["fixture_visible_mismatch"]
                    ),
                    "generated_candidate_content_labels": [
                        "unseen_geometry_is_generated_candidate"
                    ],
                }
            )
    review_path = historical / "shared" / "visual_review.v1.json"
    seal_cad_agent_visual_reference_review(
        review_media_receipt_path=(
            historical / "shared" / "review_media" / "cad_agent_visual_comparison.v1.json"
        ),
        reviewer={
            "reviewer_kind": "codex_visual_review",
            "reviewer_id": "fixture-reviewer",
            "visual_input_mode": (
                "all_manifest_bound_reference_frames_and_candidate_snapshots"
            ),
        },
        candidate_decisions=decisions,
        output_path=review_path,
    )
    staged = tmp_path / "host-stage"
    shutil.copytree(historical, staged)
    staged_outputs = [staged / path.relative_to(historical) for path in output_paths]
    staged_review = staged / review_path.relative_to(historical)
    return historical, staged, staged_outputs, staged_review


def _expected_candidates(source_outputs: list[Path]) -> list[dict]:
    rows = []
    for path in source_outputs:
        output = json.loads(path.read_text())
        request = output["request"]
        rows.append(
            {
                "replacement_slot": request["replacement_slot"],
                "task_id": request["task_id"],
                "asset_id": request["asset_id"],
                "backend_id": request["backend"]["backend_id"],
                "source_receipt_digest": output["receipt_digest"],
            }
        )
    return rows


def test_rematerializes_exhaustive_visual_review_for_all_four_imported_candidates(
    tmp_path: Path,
) -> None:
    historical, staged, source_outputs, source_review = _source_review_chain(tmp_path)
    mappings = _mappings(historical, staged)
    import_receipts = []
    artifact_root = tmp_path / "shared-import-artifacts"
    for index, source_output in enumerate(source_outputs):
        imported = materialize_simready_cad_agent_host_import(
            source_receipt_path=source_output,
            destination_root=tmp_path / "imports" / str(index),
            source_prefix_mappings=mappings,
            artifact_root=artifact_root,
        )
        import_receipts.append(
            Path(imported["destination_root"])
            / "simready_cad_agent_host_import.v1.json"
        )
    destination = tmp_path / "host-review"
    receipt = materialize_cad_visual_review_host_rematerialization(
        cad_host_import_receipt_paths=import_receipts,
        source_visual_review_path=source_review,
        destination_root=destination,
        source_prefix_mappings=mappings,
        expected_candidates=_expected_candidates(source_outputs),
    )

    assert _validate_review_receipt(receipt, source_review) == receipt
    review_path = Path(receipt["outputs"]["visual_review"]["path"])
    review = json.loads(review_path.read_text())
    assert validate_cad_agent_visual_reference_review(review) == review
    assert receipt["candidate_count"] == 4
    selected_key = next(
        row
        for row in receipt["candidate_digest_rebindings"]
        if row["task_id"] == "task_a_washer_door_open"
        and row["backend_id"] == "earthtojake_text_to_cad"
    )
    decision = selected_cad_agent_visual_review(
        file_record(review_path),
        scene_id="840920",
        task_id=selected_key["task_id"],
        asset_id=selected_key["asset_id"],
        backend_id=selected_key["backend_id"],
        cad_agent_output_receipt_digest=selected_key["imported_receipt_digest"],
    )
    assert decision["review_status"] == "conditionally_admitted_for_construction"
    tampered = json.loads(json.dumps(receipt))
    tampered["candidate_digest_rebindings"][0]["imported_receipt_digest"] = (
        "sha256:" + "f" * 64
    )
    tampered["rematerialization_digest"] = canonical_digest(
        tampered, digest_field="rematerialization_digest"
    )
    with pytest.raises(
        SimReadyCadAgentHostImportError, match="rebinding_join_invalid"
    ):
        _validate_review_receipt(tampered, source_review)
    source_tamper = json.loads(json.dumps(receipt))
    source_tamper["source_visual_review"] = {"garbage": "accepted"}
    source_tamper["rematerialization_digest"] = canonical_digest(
        source_tamper, digest_field="rematerialization_digest"
    )
    with pytest.raises(
        SimReadyCadAgentHostImportError, match="receipt_invalid"
    ):
        _validate_review_receipt(source_tamper, source_review)
    plausible_tamper = json.loads(json.dumps(receipt))
    plausible_tamper["source_visual_review"].update(
        {
            "review_digest": "sha256:" + "c" * 64,
            "sha256": "sha256:" + "d" * 64,
            "size_bytes": 12345,
        }
    )
    plausible_tamper["rematerialization_digest"] = canonical_digest(
        plausible_tamper, digest_field="rematerialization_digest"
    )
    persisted_chain = (
        destination / "simready_cad_visual_review_host_rematerialization.v1.json"
    )
    persisted_chain.write_text(json.dumps(plausible_tamper, sort_keys=True) + "\n")
    persisted_chain.chmod(0o640)
    with pytest.raises(
        SimReadyCadAgentHostImportError, match="receipt_invalid"
    ):
        _validate_review_receipt(plausible_tamper, source_review)
    serialized = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in destination.rglob("*.json")
    )
    assert str(historical) not in serialized
    assert str(REPO_ROOT) not in serialized


def test_concurrent_candidate_imports_publish_shared_artifacts_atomically(
    tmp_path: Path,
) -> None:
    historical, staged, source_outputs, _source_review = _source_review_chain(tmp_path)
    mappings = _mappings(historical, staged)
    artifact_root = tmp_path / "shared-import-artifacts"

    def run(index: int):
        return materialize_simready_cad_agent_host_import(
            source_receipt_path=source_outputs[index],
            destination_root=tmp_path / "imports" / str(index),
            source_prefix_mappings=mappings,
            artifact_root=artifact_root,
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        receipts = list(executor.map(run, range(4)))
    assert len(receipts) == 4
    assert all(validate_simready_cad_agent_host_import(row) == row for row in receipts)


@pytest.mark.skipif(importlib.util.find_spec("pxr") is None, reason="OpenUSD unavailable")
def test_imported_candidate_and_rebound_review_feed_visual_binding_sealer(
    tmp_path: Path,
) -> None:
    from blueprint_pipeline.agent_cad_graph_visual_composition import (
        seal_agent_cad_visual_binding,
    )
    from blueprint_pipeline.cad_agent_mesh_projection import (
        PACKET_SCHEMA_VERSION,
        materialize_mesh_usd_projection,
    )
    from tests.test_agent_cad_graph_visual_composition import (
        _graph_authoring_receipt,
        _identity,
    )

    historical, staged, source_outputs, source_review = _source_review_chain(tmp_path)
    mappings = _mappings(historical, staged)
    artifact_root = tmp_path / "shared-import-artifacts"
    import_receipts = []
    for index, source_output in enumerate(source_outputs):
        imported = materialize_simready_cad_agent_host_import(
            source_receipt_path=source_output,
            destination_root=tmp_path / "imports" / str(index),
            source_prefix_mappings=mappings,
            artifact_root=artifact_root,
        )
        import_receipts.append(
            Path(imported["destination_root"])
            / "simready_cad_agent_host_import.v1.json"
        )
    review_receipt = materialize_cad_visual_review_host_rematerialization(
        cad_host_import_receipt_paths=import_receipts,
        source_visual_review_path=source_review,
        destination_root=tmp_path / "host-review",
        source_prefix_mappings=mappings,
        expected_candidates=_expected_candidates(source_outputs),
    )
    selected_import = json.loads(import_receipts[0].read_text())
    selected_output_path = Path(
        selected_import["imported_cad_agent_output"]["path"]
    )
    selected_output = json.loads(selected_output_path.read_text())
    step_record = selected_output["artifacts"]["step"]
    packet = {
        "schema_version": PACKET_SCHEMA_VERSION,
        "geometry_authority": "exact_agent_authored_step",
        "deterministic_geometry_generator_used": False,
        "conversion_only": True,
        "step": step_record,
        "linear_tolerance_mm": 0.2,
        "angular_tolerance_rad": 0.1,
        "mesh_count": 2,
        "meshes": [
            {
                "prim_path": "/Asset/links/body/geometry/shell",
                "link_id": "body",
                "solid_id": "shell",
                "assembly_transform_applied": True,
                "points_mm": [[0, 0, 0], [100, 0, 0], [0, 100, 0]],
                "triangles": [[0, 1, 2]],
                "agent_authored_display_color_rgba": [0.8, 0.8, 0.8, 1.0],
            },
            {
                "prim_path": "/Asset/links/door/geometry/rim",
                "link_id": "door",
                "solid_id": "rim",
                "assembly_transform_applied": True,
                "points_mm": [[0, 0, 10], [50, 0, 10], [0, 50, 10]],
                "triangles": [[0, 1, 2]],
                "agent_authored_display_color_rgba": [0.1, 0.1, 0.1, 1.0],
            },
        ],
        "claim_boundary": {
            "cad_authored_by_projection": False,
            "appearance_working_copy_only": True,
            "collision_authority": False,
            "physics_authority": False,
            "simready_qualified": False,
        },
        "packet_digest": "",
    }
    packet["packet_digest"] = canonical_digest(packet, digest_field="packet_digest")
    packet_path = tmp_path / "projection" / "mesh_packet.json"
    packet_path.parent.mkdir(parents=True)
    packet_path.write_text(json.dumps(packet, sort_keys=True) + "\n")
    projection = materialize_mesh_usd_projection(
        packet_path=packet_path,
        output_usd_path=tmp_path / "projection" / "projection.usda",
    )
    projection_path = tmp_path / "projection" / "projection.receipt.json"
    projection_path.write_text(json.dumps(projection, sort_keys=True) + "\n")
    graph = _graph_authoring_receipt(tmp_path / "graph")
    binding = seal_agent_cad_visual_binding(
        graph_authoring_receipt_path=graph,
        cad_agent_output_receipt_path=selected_output_path,
        cad_agent_visual_review_path=review_receipt["outputs"]["visual_review"][
            "path"
        ],
        mesh_projection_receipt_path=projection_path,
        link_bindings=[
            {
                "agent_link_id": "body",
                "graph_link_id": "body",
                "transform_mode": "explicit_rigid_transform",
                "T_graph_link_from_agent_asset": _identity(),
            },
            {
                "agent_link_id": "door",
                "graph_link_id": "door",
                "transform_mode": "explicit_rigid_transform",
                "T_graph_link_from_agent_asset": _identity(),
            },
        ],
        unmapped_graph_link_reasons={},
        output_path=tmp_path / "binding.json",
    )
    assert binding["cad_agent_output_receipt_digest"] == selected_output[
        "receipt_digest"
    ]


def test_selected_candidates_alone_cannot_rematerialize_exhaustive_review(
    tmp_path: Path,
) -> None:
    historical, staged, source_outputs, source_review = _source_review_chain(tmp_path)
    mappings = _mappings(historical, staged)
    import_receipts = []
    artifact_root = tmp_path / "shared-import-artifacts"
    for index, source_output in enumerate(source_outputs[:2]):
        imported = materialize_simready_cad_agent_host_import(
            source_receipt_path=source_output,
            destination_root=tmp_path / "imports" / str(index),
            source_prefix_mappings=mappings,
            artifact_root=artifact_root,
        )
        import_receipts.append(
            Path(imported["destination_root"])
            / "simready_cad_agent_host_import.v1.json"
        )
    destination = tmp_path / "incomplete-review"
    with pytest.raises(
        SimReadyCadAgentHostImportError,
        match="all_source_candidates_required",
    ):
        materialize_cad_visual_review_host_rematerialization(
            cad_host_import_receipt_paths=import_receipts,
            source_visual_review_path=source_review,
            destination_root=destination,
            source_prefix_mappings=mappings,
            expected_candidates=_expected_candidates(source_outputs),
        )
    assert not destination.exists()


def test_truncated_self_consistent_source_review_cannot_shrink_expected_campaign(
    tmp_path: Path,
) -> None:
    historical, staged, source_outputs, source_review = _source_review_chain(
        tmp_path, task_ids=("task_a_washer_door_open",)
    )
    mappings = _mappings(historical, staged)
    artifact_root = tmp_path / "shared-import-artifacts"
    import_receipts = []
    for index, source_output in enumerate(source_outputs):
        imported = materialize_simready_cad_agent_host_import(
            source_receipt_path=source_output,
            destination_root=tmp_path / "imports" / str(index),
            source_prefix_mappings=mappings,
            artifact_root=artifact_root,
        )
        import_receipts.append(
            Path(imported["destination_root"])
            / "simready_cad_agent_host_import.v1.json"
        )
    expected = _expected_candidates(source_outputs)
    with pytest.raises(
        SimReadyCadAgentHostImportError,
        match="expected_four_candidate_set_invalid",
    ):
        materialize_cad_visual_review_host_rematerialization(
            cad_host_import_receipt_paths=import_receipts,
            source_visual_review_path=source_review,
            destination_root=tmp_path / "caller-shrunk-review",
            source_prefix_mappings=mappings,
            expected_candidates=expected,
        )
    expected.extend(
        {
            "replacement_slot": 2,
            "task_id": "task_b_notebook_relocation",
            "asset_id": "840920_simready_notebook_candidate",
            "backend_id": backend,
            "source_receipt_digest": "sha256:" + character * 64,
        }
        for backend, character in (
            ("earthtojake_text_to_cad", "a"),
            ("pan_chera_multi_agent_cad", "b"),
        )
    )
    destination = tmp_path / "truncated-review"
    with pytest.raises(
        SimReadyCadAgentHostImportError,
        match="expected_candidate_set_mismatch",
    ):
        materialize_cad_visual_review_host_rematerialization(
            cad_host_import_receipt_paths=import_receipts,
            source_visual_review_path=source_review,
            destination_root=destination,
            source_prefix_mappings=mappings,
            expected_candidates=expected,
        )
    assert not destination.exists()


@pytest.mark.parametrize(
    ("script_name", "required_flags"),
    [
        (
            "import_simready_cad_agent_output.py",
            (
                "--source-map",
                "--artifact-root",
                "--source-receipt",
                "--owner-user",
                "--owner-group",
            ),
        ),
        (
            "rematerialize_cad_visual_review_host.py",
            (
                "--source-map",
                "--cad-host-import-receipt",
                "--source-visual-review",
                "--expected-candidate",
                "--expected-source-visual-review-digest",
                "--expected-source-visual-review-sha256",
                "--owner-user",
                "--owner-group",
            ),
        ),
    ],
)
def test_host_import_clis_expose_exact_repeatable_mapping_contract(
    script_name: str, required_flags: tuple[str, ...]
) -> None:
    completed = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / script_name), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    for flag in required_flags:
        assert flag in completed.stdout
