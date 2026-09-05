from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.dual_task_rehearsal_contract import validate_task_freeze, validate_scene_freeze
from blueprint_pipeline.public_scene_removal_selection import (
    ADAPTER, materialize_public_scene_removal_selections,
    validate_removal_scene_selection, validate_removal_task_selection,
    validate_source_preparation_task_selection_set,
)
from tests.test_task_evaluation_scene_configuration_submission import (
    production_fixture as _fixture, _sha, SHA,
)


def _write(path: Path, value: dict, field: str | None = None) -> None:
    if field:
        value[field] = canonical_digest(value, digest_field=field)
    path.write_text(json.dumps(value), encoding="utf-8")


def _source_fixture(root: Path) -> dict:
    fixture = _fixture(root)
    install_path = fixture["installation_receipt"]
    install = json.loads(install_path.read_text())
    rights = install_path.parent / "rights.json"
    _write(rights, {
        "schema_version": "public_scene_rights_authority.v1",
        "status": "accepted_for_declared_local_import_only",
        "agent_accepted_terms": False,
        "authorized_source_sha256": [row["sha256"] for row in install["files"]],
    })
    for row in install["files"]:
        row["rights_receipt_ids"] = ["fixture-source-rights"]
    install["files"].insert(0, {
        "kind": "rights_receipt", "receipt_id": "fixture-source-rights",
        "relative_path": rights.name, "sha256": _sha(rights), "size_bytes": rights.stat().st_size,
    })
    _write(install_path, install, "receipt_digest")
    prepared_path = fixture["source_preparation"]
    prepared = json.loads(prepared_path.read_text())
    frame_path = prepared_path.parent / "shared_frame_candidate.json"
    sources = {row.get("role"): row for row in install["files"]}
    identities = [json.loads((prepared_path.parent / f"source_identity_{i:02d}.json").read_text())
                  for i in range(2)]
    frame = {
        "schema_version": "interiorgs_sage_shared_frame_candidate.v1",
        "source_digests": {
            "interiorgs_labels": sources["semantic_metadata"]["sha256"],
            "sage_collision_usd": sources["collision_usd"]["sha256"],
        },
        "correspondences": [{
            "interiorgs_instance_id": value["target"]["interiorgs_instance_id"],
            "semantic_label": value["target"]["semantic_label"],
            "sage_prim_path": value["whole_object_matches"][0]["prim_path"],
            "identity_receipt_digest": value["receipt_digest"],
        } for value in identities],
    }
    _write(frame_path, frame, "receipt_digest")
    for row in prepared["artifacts"]:
        path = prepared_path.parent / row["relative_path"]
        row["sha256"], row["size_bytes"] = _sha(path), path.stat().st_size
    prepared["source_installation_digest"] = install["receipt_digest"]
    _write(prepared_path, prepared, "receipt_digest")
    return fixture


def _selections(root: Path) -> tuple[dict, dict, dict]:
    fixture = _source_fixture(root)
    result = materialize_public_scene_removal_selections(
        task_request_path=fixture["task_request"],
        installation_receipt_path=fixture["installation_receipt"],
        publisher_intake_path=fixture["publisher_intake"],
        source_preparation_receipt_path=fixture["source_preparation"],
        expected_production_commit=SHA, output_root=root / "removal-selections",
    )
    task = json.loads(Path(result["task_selection"]["path"]).read_text())
    scene = json.loads(Path(result["scene_selection"]["path"]).read_text())
    return fixture, result, {"task": task, "scene": scene}


def test_source_removal_selection_never_fabricates_robot_qualification(tmp_path: Path) -> None:
    _, result, values = _selections(tmp_path)
    task = validate_removal_task_selection(values["task"])
    validate_removal_scene_selection(values["scene"])
    assert task["source_object"]["instance_id"] == "115"
    assert "franka_placement_packet_digest" not in task["source_object"]
    assert "visibility_receipt_digest" not in task["source_object"]
    assert task["evaluation_authorized"] is False
    assert task["robot_reachability_established"] is False
    assert task["candidate_policy_queried"] is False
    assert result["raw_source_uploaded"] is False
    assert validate_source_preparation_task_selection_set([task])["task_count"] == 1
    with pytest.raises(ValueError, match="task_freeze_schema_invalid"):
        validate_task_freeze(task)
    with pytest.raises(ValueError, match="scene_freeze_schema_invalid"):
        validate_scene_freeze(values["scene"])


def test_source_removal_selection_rechecks_exact_raw_bytes(tmp_path: Path) -> None:
    fixture, _, values = _selections(tmp_path)
    installation = json.loads(fixture["installation_receipt"].read_text())
    row = next(row for row in installation["files"] if row.get("role") == "semantic_metadata")
    (fixture["installation_receipt"].parent / row["relative_path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="input_bytes_mismatch"):
        validate_removal_task_selection(values["task"])


def test_source_removal_selection_rejects_resigned_target_switch(tmp_path: Path) -> None:
    _, _, values = _selections(tmp_path)
    task = values["task"]
    task["source_object"]["instance_id"] = "85"
    task["task_freeze_digest"] = canonical_digest(task, digest_field="task_freeze_digest")
    with pytest.raises(ValueError, match="task_source_mismatch"):
        validate_removal_task_selection(task)


def test_source_removal_selection_rejects_resigned_evaluation_authority(tmp_path: Path) -> None:
    _, _, values = _selections(tmp_path)
    task = values["task"]
    task["evaluation_authorized"] = True
    task["task_freeze_digest"] = canonical_digest(task, digest_field="task_freeze_digest")
    with pytest.raises(ValueError, match="task_invalid"):
        validate_removal_task_selection(task)


def test_source_removal_selection_requires_installed_rights(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    with pytest.raises(ValueError, match="rights_missing"):
        materialize_public_scene_removal_selections(
            task_request_path=fixture["task_request"],
            installation_receipt_path=fixture["installation_receipt"],
            publisher_intake_path=fixture["publisher_intake"],
            source_preparation_receipt_path=fixture["source_preparation"],
            expected_production_commit=SHA, output_root=tmp_path / "forbidden",
        )
    assert not (tmp_path / "forbidden").exists()


def test_calibrated_renderer_accepts_distinct_source_removal_adapter() -> None:
    from blueprint_pipeline.public_scene_inpainting_inputs import build_public_scene_inpainting_input_request
    from tests.test_public_scene_inpainting_inputs import _request

    request = _request()
    request["schema_version"] = "public_scene_interiorgs_edit_input_request.v2"
    request["scene"] = {
        "source_adapter": ADAPTER,
        **{key: key + ".json" for key in (
            "scene_freeze_path", "task_freeze_path", "standard_splat_conversion_receipt_path",
            "standard_splat_path", "labels_path", "structure_path", "registered_frame_receipt_path",
        )},
    }
    result = build_public_scene_inpainting_input_request(request)
    assert result["scene"]["source_adapter"] == ADAPTER


def test_sam_inputs_accept_real_removal_selection_without_robot_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from blueprint_pipeline.public_scene_sam31_task_inputs import materialize_public_scene_sam31_task_inputs
    from tests.test_public_scene_sam31_task_inputs import _fixture as sam_fixture

    source_root = tmp_path / "selection-inputs"
    source_root.mkdir()
    _, result, values = _selections(source_root)
    sam = sam_fixture(tmp_path)
    receipt = json.loads(sam["receipt"].read_text())
    receipt["scene"]["task_id"] = values["task"]["task_id"]
    receipt["scene"]["target_instance_id"] = values["task"]["source_object"]["instance_id"]
    receipt["source_admission"] = {
        "adapter": ADAPTER,
        "task_freeze_digest": values["task"]["task_freeze_digest"],
        "scene_freeze_digest": values["task"]["scene_freeze_digest"],
    }
    _write(sam["receipt"], receipt, "receipt_digest")

    def encode(*, output_path: Path, **_: object) -> list[str]:
        output_path.write_bytes(b"synthetic-lossless-sequence")
        return ["fixture-ffv1"]
    monkeypatch.setattr("blueprint_pipeline.public_scene_sam31_task_inputs._encode_lossless_sequence", encode)
    packet = materialize_public_scene_sam31_task_inputs(
        calibrated_view_receipt_path=sam["receipt"], task_freeze_path=result["task_selection"]["path"],
        provider_profile_path=sam["profile"], prompts_path=sam["prompts"],
        output_root=tmp_path / "sam-output", ffmpeg_executable=sam["ffmpeg"],
    )
    assert packet["task_id"] == values["task"]["task_id"]
    assert packet["paid_execution_started"] is False
