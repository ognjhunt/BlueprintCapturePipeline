from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.simready_cad_agent_contract import (
    MATRIX_SCHEMA_VERSION,
    INSPECTION_SCHEMA_VERSION,
    SimReadyCadAgentContractError,
    file_record,
    materialize_cad_agent_reference_binding_audit,
    seal_cad_agent_execution_receipt,
    seal_cad_agent_matrix,
    seal_cad_agent_output,
    seal_cad_agent_reference_manifest,
    seal_cad_agent_request,
    validate_cad_agent_matrix,
    validate_cad_agent_reference_binding_audit,
    validate_cad_agent_output,
    validate_cad_agent_reference_manifest,
    validate_cad_agent_request,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
FREEZE_A = (
    REPO_ROOT
    / "docs/arm_decision_proof_v1/manifests/third_scene_840920_task_a_freeze.v1.json"
)
FREEZE_B = (
    REPO_ROOT
    / "docs/arm_decision_proof_v1/manifests/third_scene_840920_task_b_freeze.v1.json"
)


def _write(path: Path, payload: str | bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, bytes):
        path.write_bytes(payload)
    else:
        path.write_text(payload, encoding="utf-8")
    return path


def _backend(tmp_path: Path, backend_id: str) -> dict:
    archive = tmp_path / f"{backend_id}.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive, "w", compression=ZIP_DEFLATED) as source:
        source.comment = b"1" * 40
        source.writestr("LICENSE", "MIT\n")
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
        "source_archive": {
            "path": str(archive.resolve()),
            "size_bytes": archive.stat().st_size,
            "sha256": "sha256:"
            + __import__("hashlib").sha256(archive.read_bytes()).hexdigest(),
        },
        "license": "MIT",
        "model_id": (
            "codex_gpt_5_6_luna"
            if backend_id == "earthtojake_text_to_cad"
            else "gpt-5.6"
        ),
    }


def _request(
    tmp_path: Path,
    *,
    backend_id: str = "earthtojake_text_to_cad",
    slot: int = 1,
    freeze_path: Path = FREEZE_A,
    task_id: str = "task_a_washer_door_open",
    asset_id: str = "840920_simready_washer_candidate",
):
    brief = _write(tmp_path / f"brief-{slot}-{backend_id}.md", "CAD brief")
    image = _write(tmp_path / f"reference-{slot}.png", b"png-reference")
    reference_manifest = seal_cad_agent_reference_manifest(
        scene_id="fixture_scene",
        objects=[
            {
                "replacement_slot": slot,
                "task_id": task_id,
                "asset_id": asset_id,
                "task_freeze_path": freeze_path,
                "reference_image_paths": [image],
            }
        ],
    )
    reference_manifest_path = tmp_path / f"references-{slot}-{backend_id}.json"
    reference_manifest_path.write_text(
        json.dumps(reference_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return seal_cad_agent_request(
        request_id=f"request-{slot}-{backend_id}",
        scene_id="fixture_scene",
        task_id=task_id,
        asset_id=asset_id,
        replacement_slot=slot,
        backend=_backend(tmp_path, backend_id),
        task_freeze_path=freeze_path,
        cad_brief_path=brief,
        metric_envelope_mm=[600.112, 604.104004, 847.564026],
        reference_manifest_path=reference_manifest_path,
    )


def _output(tmp_path: Path, request: dict, suffix: str) -> dict:
    generator = _write(tmp_path / f"candidate-{suffix}.py", "def gen_step(): pass\n")
    step = _write(tmp_path / f"candidate-{suffix}.step", b"STEP")
    inspection = tmp_path / f"inspection-{suffix}.json"
    snapshot = _write(tmp_path / f"snapshot-{suffix}.png", b"PNG")
    inspection_payload = {
        "schema_version": INSPECTION_SCHEMA_VERSION,
        "status": "passed",
        "step": file_record(step),
        "measured_envelope_mm": [600.112, 604.104004, 847.564026],
        "measured_center_mm": [0.0, 0.0, 423.782013],
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
    }
    inspection_payload["receipt_digest"] = canonical_digest(
        inspection_payload, digest_field="receipt_digest"
    )
    _write(inspection, __import__("json").dumps(inspection_payload))
    execution = tmp_path / f"execution-{suffix}.json"
    execution_payload = seal_cad_agent_execution_receipt(
        request=request,
        generator_source_path=generator,
        cad_brief_path=request["inputs"]["cad_brief"]["path"],
        output_step_path=step,
        event_rows=[{"event": "agent_authored_step", "status": "passed"}],
    )
    _write(execution, __import__("json").dumps(execution_payload))
    return seal_cad_agent_output(
        request=request,
        generator_source_path=generator,
        step_path=step,
        inspection_receipt_path=inspection,
        snapshot_paths=[snapshot],
        execution_receipt_path=execution,
        measured_envelope_mm=[600.112, 604.104004, 847.564026],
        actual_cost_usd=(
            0.0
            if request["backend"]["backend_id"] == "earthtojake_text_to_cad"
            else 0.75
        ),
    )


def test_earthtojake_and_multi_agent_requests_share_agent_only_contract(tmp_path: Path):
    earth = _request(tmp_path / "earth")
    mac = _request(tmp_path / "mac", backend_id="pan_chera_multi_agent_cad")

    assert validate_cad_agent_request(earth) == earth
    assert validate_cad_agent_request(mac) == mac
    assert earth["execution_budget"]["hard_cap_usd"] == 0.0
    assert mac["execution_budget"]["hard_cap_usd"] == 1.0
    assert all(
        request["backend"]["deterministic_geometry_generator_used"] is False
        for request in (earth, mac)
    )
    assert earth["inputs"]["reference_binding_source"] == "manifest_derived"
    assert earth["inputs"]["reference_manifest"]["path"]
    assert earth["inputs"]["reference_manifest_object_digest"].startswith("sha256:")


def test_request_validator_rejects_unbound_reference_images(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path / "request")
    request["inputs"].pop("reference_manifest")
    request["inputs"].pop("reference_binding_source")
    request["inputs"].pop("reference_manifest_object_digest")
    request["request_digest"] = canonical_digest(
        request, digest_field="request_digest"
    )

    with pytest.raises(SimReadyCadAgentContractError) as excinfo:
        validate_cad_agent_request(request)

    assert "cad_agent_request_reference_manifest_invalid" in excinfo.value.codes
    assert "cad_agent_request_reference_binding_invalid" in excinfo.value.codes
    assert (
        "cad_agent_request_reference_manifest_object_digest_invalid"
        in excinfo.value.codes
    )


def test_request_sealer_rejects_manual_reference_images_without_manifest(
    tmp_path: Path,
) -> None:
    brief = _write(tmp_path / "brief.md", "CAD brief")
    image = _write(tmp_path / "reference.png", b"png-reference")

    with pytest.raises(SimReadyCadAgentContractError) as excinfo:
        seal_cad_agent_request(
            request_id="manual-reference-rejected",
            scene_id="fixture_scene",
            task_id="task_a_washer_door_open",
            asset_id="840920_simready_washer_candidate",
            replacement_slot=1,
            backend=_backend(tmp_path, "earthtojake_text_to_cad"),
            task_freeze_path=FREEZE_A,
            cad_brief_path=brief,
            reference_image_paths=[image],
            metric_envelope_mm=[600.112, 604.104004, 847.564026],
        )
    assert "cad_agent_request_reference_manifest_required" in excinfo.value.codes


def test_request_sealer_rejects_off_manifest_reference_override(
    tmp_path: Path,
) -> None:
    brief = _write(tmp_path / "brief.md", "CAD brief")
    admitted = _write(tmp_path / "admitted.png", b"admitted-reference")
    off_manifest = _write(tmp_path / "off-manifest.png", b"wrong-reference")
    reference_manifest = seal_cad_agent_reference_manifest(
        scene_id="fixture_scene",
        objects=[
            {
                "replacement_slot": 1,
                "task_id": "task_a_washer_door_open",
                "asset_id": "840920_simready_washer_candidate",
                "task_freeze_path": FREEZE_A,
                "reference_image_paths": [admitted],
            }
        ],
    )
    reference_manifest_path = tmp_path / "references.json"
    reference_manifest_path.write_text(
        json.dumps(reference_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    with pytest.raises(SimReadyCadAgentContractError) as excinfo:
        seal_cad_agent_request(
            request_id="off-manifest-reference-rejected",
            scene_id="fixture_scene",
            task_id="task_a_washer_door_open",
            asset_id="840920_simready_washer_candidate",
            replacement_slot=1,
            backend=_backend(tmp_path, "earthtojake_text_to_cad"),
            task_freeze_path=FREEZE_A,
            cad_brief_path=brief,
            reference_image_paths=[off_manifest],
            metric_envelope_mm=[600.112, 604.104004, 847.564026],
            reference_manifest_path=reference_manifest_path,
        )
    assert "cad_agent_request_manual_reference_images_forbidden" in excinfo.value.codes


def test_reference_manifest_accepts_five_replacement_objects(tmp_path: Path) -> None:
    objects = []
    for slot in range(1, 6):
        objects.append(
            {
                "replacement_slot": slot,
                "task_id": f"task_{slot}",
                "asset_id": f"asset_{slot}",
                "task_freeze_path": FREEZE_A,
                "reference_image_paths": [
                    _write(tmp_path / f"reference-{slot}.png", f"ref-{slot}".encode())
                ],
            }
        )
    manifest = seal_cad_agent_reference_manifest(
        scene_id="fixture_scene", objects=objects
    )
    assert validate_cad_agent_reference_manifest(manifest) == manifest
    assert len(manifest["objects"]) == 5


def test_reference_manifest_rejects_six_replacement_objects(tmp_path: Path) -> None:
    objects = []
    for slot in range(1, 7):
        objects.append(
            {
                "replacement_slot": slot,
                "task_id": f"task_{slot}",
                "asset_id": f"asset_{slot}",
                "task_freeze_path": FREEZE_A,
                "reference_image_paths": [
                    _write(tmp_path / f"reference-{slot}.png", f"ref-{slot}".encode())
                ],
            }
        )
    with pytest.raises(SimReadyCadAgentContractError) as excinfo:
        seal_cad_agent_reference_manifest(scene_id="fixture_scene", objects=objects)
    assert "cad_agent_reference_manifest_object_count_invalid" in excinfo.value.codes


def test_reference_manifest_rejects_swapped_object_reference_binding(
    tmp_path: Path,
) -> None:
    task_a_ref = _write(tmp_path / "washer.png", b"washer-reference")
    task_b_ref = _write(tmp_path / "notebook.png", b"notebook-reference")
    manifest = seal_cad_agent_reference_manifest(
        scene_id="fixture_scene",
        objects=[
            {
                "replacement_slot": 1,
                "task_id": "task_a_washer_door_open",
                "asset_id": "840920_simready_washer_candidate",
                "task_freeze_path": FREEZE_A,
                "reference_image_paths": [task_a_ref],
            },
            {
                "replacement_slot": 2,
                "task_id": "task_b_notebook_relocation",
                "asset_id": "840920_simready_notebook_candidate",
                "task_freeze_path": FREEZE_B,
                "reference_image_paths": [task_b_ref],
            },
        ],
    )
    assert validate_cad_agent_reference_manifest(manifest) == manifest
    manifest_path = tmp_path / "references.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    brief = _write(tmp_path / "brief.md", "CAD brief")

    with pytest.raises(SimReadyCadAgentContractError) as excinfo:
        seal_cad_agent_request(
            request_id="swapped-reference-rejected",
            scene_id="fixture_scene",
            task_id="task_a_washer_door_open",
            asset_id="840920_simready_washer_candidate",
            replacement_slot=2,
            backend=_backend(tmp_path, "earthtojake_text_to_cad"),
            task_freeze_path=FREEZE_A,
            cad_brief_path=brief,
            metric_envelope_mm=[600.112, 604.104004, 847.564026],
            reference_manifest_path=manifest_path,
        )
    assert "cad_agent_reference_manifest_object_missing" in excinfo.value.codes


@pytest.mark.parametrize("slot", [0, 6])
def test_request_rejects_slots_outside_one_to_five(tmp_path: Path, slot: int):
    brief = _write(tmp_path / "brief.md", "CAD brief")
    image = _write(tmp_path / "reference.png", b"png-reference")
    reference_manifest = seal_cad_agent_reference_manifest(
        scene_id="fixture_scene",
        objects=[
            {
                "replacement_slot": 1,
                "task_id": "task_a_washer_door_open",
                "asset_id": "840920_simready_washer_candidate",
                "task_freeze_path": FREEZE_A,
                "reference_image_paths": [image],
            }
        ],
    )
    reference_manifest_path = tmp_path / "references.json"
    reference_manifest_path.write_text(
        json.dumps(reference_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    with pytest.raises(SimReadyCadAgentContractError) as excinfo:
        seal_cad_agent_request(
            request_id=f"request-{slot}",
            scene_id="fixture_scene",
            task_id="task_a_washer_door_open",
            asset_id="840920_simready_washer_candidate",
            replacement_slot=slot,
            backend=_backend(tmp_path, "earthtojake_text_to_cad"),
            task_freeze_path=FREEZE_A,
            cad_brief_path=brief,
            metric_envelope_mm=[600.112, 604.104004, 847.564026],
            reference_manifest_path=reference_manifest_path,
        )
    assert "cad_agent_request_replacement_slot_invalid" in excinfo.value.codes


def test_request_rejects_any_deterministic_cad_backend(tmp_path: Path):
    request = _request(tmp_path)
    request["backend"]["backend_id"] = "deterministic_graph_to_cad"
    request["backend"]["execution_mode"] = "codex_skill_step_first"
    request["request_digest"] = canonical_digest(
        request, digest_field="request_digest"
    )

    with pytest.raises(SimReadyCadAgentContractError) as excinfo:
        validate_cad_agent_request(request)
    assert "cad_agent_request_backend_invalid" in excinfo.value.codes


def test_multi_agent_cad_keeps_api_fallback_admitted(tmp_path: Path):
    request = _request(tmp_path, backend_id="pan_chera_multi_agent_cad")
    request["backend"]["execution_mode"] = "openai_compatible_api_aider_first"
    request["request_digest"] = canonical_digest(
        request, digest_field="request_digest"
    )

    assert validate_cad_agent_request(request) == request


def test_request_rejects_graph_geometry_as_cad_authority(tmp_path: Path):
    request = _request(tmp_path)
    request["inputs"]["collision_graph_is_geometry_authority"] = True
    request["request_digest"] = canonical_digest(
        request, digest_field="request_digest"
    )

    with pytest.raises(SimReadyCadAgentContractError) as excinfo:
        validate_cad_agent_request(request)
    assert "cad_agent_request_graph_geometry_forbidden" in excinfo.value.codes


def test_output_requires_metric_envelope_and_agent_execution(tmp_path: Path):
    request = _request(tmp_path / "request")
    output = _output(tmp_path / "output", request, "earth")
    assert validate_cad_agent_output(output) == output

    output["measured_envelope_mm"][2] += 2.0
    output["receipt_digest"] = canonical_digest(
        output, digest_field="receipt_digest"
    )
    with pytest.raises(SimReadyCadAgentContractError) as excinfo:
        validate_cad_agent_output(output)
    assert "cad_agent_output_metric_envelope_mismatch" in excinfo.value.codes


def test_same_object_compares_two_agent_backends(tmp_path: Path):
    earth_request = _request(tmp_path / "earth")
    mac_request = _request(
        tmp_path / "mac", backend_id="pan_chera_multi_agent_cad"
    )
    matrix = seal_cad_agent_matrix(
        objects=[
            {
                "replacement_slot": 1,
                "task_id": earth_request["task_id"],
                "asset_id": earth_request["asset_id"],
                "candidates": [
                    _output(tmp_path / "earth-out", earth_request, "earth"),
                    _output(tmp_path / "mac-out", mac_request, "mac"),
                ],
            }
        ]
    )
    assert validate_cad_agent_matrix(matrix) == matrix


def test_reference_binding_audit_binds_outputs_to_manifest(tmp_path: Path) -> None:
    request = _request(tmp_path / "request")
    output = _output(tmp_path / "output", request, "earth")
    output_path = tmp_path / "output.json"
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    manifest_path = Path(request["inputs"]["reference_manifest"]["path"])

    audit = materialize_cad_agent_reference_binding_audit(
        scene_id=request["scene_id"],
        reference_manifest_path=manifest_path,
        cad_agent_output_paths=[output_path],
    )

    assert validate_cad_agent_reference_binding_audit(audit) == audit
    assert audit["replacement_object_count"] == 1
    assert audit["candidate_rows"][0]["binding_status"] == "manifest_bound"
    assert audit["candidate_rows"][0]["cad_agent_output_artifacts_verified"] is True
    assert audit["historical_requests_rewritten"] is False
    assert audit["cad_output_receipts_resealed"] is False


def test_reference_binding_audit_records_receipt_reseal_without_geometry_regen(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path / "request")
    output = _output(tmp_path / "output", request, "earth")
    output_path = tmp_path / "output.json"
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    manifest_path = Path(request["inputs"]["reference_manifest"]["path"])

    audit = materialize_cad_agent_reference_binding_audit(
        scene_id=request["scene_id"],
        reference_manifest_path=manifest_path,
        cad_agent_output_paths=[output_path],
        historical_requests_rewritten=True,
        cad_output_receipts_resealed=True,
    )

    assert validate_cad_agent_reference_binding_audit(audit) == audit
    assert audit["historical_requests_rewritten"] is True
    assert audit["cad_output_receipts_resealed"] is True
    assert audit["agent_outputs_regenerated"] is False
    audit["agent_outputs_regenerated"] = True
    audit["audit_digest"] = canonical_digest(audit, digest_field="audit_digest")
    with pytest.raises(SimReadyCadAgentContractError) as excinfo:
        validate_cad_agent_reference_binding_audit(audit)
    assert (
        "cad_agent_reference_binding_audit_claim_boundary_invalid"
        in excinfo.value.codes
    )


def test_reference_binding_audit_can_bind_historical_output_without_artifact_reopen(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path / "request")
    output = _output(tmp_path / "output", request, "earth")
    output_path = tmp_path / "output.json"
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    Path(output["artifacts"]["generator_source"]["path"]).write_text(
        "mutated after historical receipt\n", encoding="utf-8"
    )
    manifest_path = Path(request["inputs"]["reference_manifest"]["path"])

    with pytest.raises(SimReadyCadAgentContractError):
        materialize_cad_agent_reference_binding_audit(
            scene_id=request["scene_id"],
            reference_manifest_path=manifest_path,
            cad_agent_output_paths=[output_path],
        )

    audit = materialize_cad_agent_reference_binding_audit(
        scene_id=request["scene_id"],
        reference_manifest_path=manifest_path,
        cad_agent_output_paths=[output_path],
        verify_cad_output_artifact_files=False,
    )

    assert validate_cad_agent_reference_binding_audit(audit) == audit
    assert audit["candidate_rows"][0]["cad_agent_output_artifacts_verified"] is False


def test_reference_binding_audit_rejects_swapped_manifest_reference(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path / "request")
    output = _output(tmp_path / "output", request, "earth")
    output_path = tmp_path / "output.json"
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    wrong_reference = _write(tmp_path / "wrong.png", b"wrong-reference")
    swapped_manifest = seal_cad_agent_reference_manifest(
        scene_id=request["scene_id"],
        objects=[
            {
                "replacement_slot": request["replacement_slot"],
                "task_id": request["task_id"],
                "asset_id": request["asset_id"],
                "task_freeze_path": request["inputs"]["task_freeze"]["path"],
                "reference_image_paths": [wrong_reference],
            }
        ],
    )
    swapped_manifest_path = tmp_path / "swapped-references.json"
    swapped_manifest_path.write_text(
        json.dumps(swapped_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    with pytest.raises(SimReadyCadAgentContractError) as excinfo:
        materialize_cad_agent_reference_binding_audit(
            scene_id=request["scene_id"],
            reference_manifest_path=swapped_manifest_path,
            cad_agent_output_paths=[output_path],
        )

    assert any(
        code.startswith("cad_agent_reference_binding_audit_manifest_join_invalid")
        for code in excinfo.value.codes
    )


def test_matrix_rejects_six_replacement_objects_before_outputs(tmp_path: Path):
    matrix = {
        "schema_version": MATRIX_SCHEMA_VERSION,
        "objects": [
            {
                "replacement_slot": slot,
                "task_id": f"task-{slot}",
                "asset_id": f"asset-{slot}",
                "candidates": [],
            }
            for slot in range(1, 7)
        ],
        "maximum_replacement_objects": 5,
        "required_backends_per_object": sorted(
            ["earthtojake_text_to_cad", "pan_chera_multi_agent_cad"]
        ),
        "deterministic_cad_backends_admitted": False,
    }
    matrix["matrix_digest"] = canonical_digest(
        matrix, digest_field="matrix_digest"
    )
    with pytest.raises(SimReadyCadAgentContractError) as excinfo:
        validate_cad_agent_matrix(matrix)
    assert "cad_agent_matrix_object_count_invalid" in excinfo.value.codes


def test_matrix_requires_both_agent_versions_for_every_object(tmp_path: Path):
    request = _request(tmp_path / "request")
    matrix = {
        "schema_version": MATRIX_SCHEMA_VERSION,
        "objects": [
            {
                "replacement_slot": 1,
                "task_id": request["task_id"],
                "asset_id": request["asset_id"],
                "candidates": [_output(tmp_path / "output", request, "earth")],
            }
        ],
        "maximum_replacement_objects": 5,
        "required_backends_per_object": sorted(
            ["earthtojake_text_to_cad", "pan_chera_multi_agent_cad"]
        ),
        "deterministic_cad_backends_admitted": False,
    }
    matrix["matrix_digest"] = canonical_digest(
        matrix, digest_field="matrix_digest"
    )
    with pytest.raises(SimReadyCadAgentContractError) as excinfo:
        validate_cad_agent_matrix(matrix)
    assert "cad_agent_matrix_candidates_invalid" in excinfo.value.codes
