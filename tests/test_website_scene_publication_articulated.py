"""Articulated website preview must reopen retained part evidence."""

import hashlib
import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_scene_configuration_publication import (
    TaskEvaluationSceneConfigurationPublicationError,
)
from blueprint_pipeline.website_scene_publication import publication_inputs


PROVIDER_STAGE = Path(
    "/workspace/task_evaluation_scene_configuration_provider_bundle/runtime_output/stages/stage-3"
)


def _record(path: Path, *, provider_path: Path | None = None) -> dict:
    data = path.read_bytes()
    return {
        "path": str(provider_path or path),
        "digest": "sha256:" + hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
    }


def _write(path: Path, value: dict, *, digest_field: str) -> dict:
    value[digest_field] = canonical_digest(value, digest_field=digest_field)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return value


def test_articulated_preview_reopens_part_results_and_labels_partial_view(tmp_path):
    stage = tmp_path / "stages/stage-3"
    artifacts = []

    def add(role: str, path: Path) -> None:
        artifacts.append({"role": role, **_record(path)})

    appearance_asset = tmp_path / "appearance.usdc"
    appearance_asset.write_bytes(b"appearance")
    collision_asset = tmp_path / "collision.usda"
    collision_asset.write_bytes(b"collision")
    add("configured_appearance_without_source_object", appearance_asset)
    add("configured_collision_without_source_object", collision_asset)
    source = _write(tmp_path / "source.json", {"schema_version": "source.v1"}, digest_field="digest")
    appearance = _write(tmp_path / "appearance.json", {
        "schema_version": "website_native_appearance.v1",
        "status": "native_appearance_authored",
        "appearance_removal_performed": False,
        "renderer_qualified": False,
        "physical_measurement_proven": False,
        "binding": {"preparation_digest": source["digest"]},
        "artifact": {"digest": _record(appearance_asset)["digest"]},
    }, digest_field="digest")
    collision = _write(tmp_path / "collision.json", {
        "schema_version": "website_prepared_background_result.v1",
        "status": "prepared_background_reused",
        "source_prim_excision_performed": False,
        "background_bytes_unchanged": True,
        "physical_measurement_proven": False,
        "preparation_digest": source["digest"],
        "collision_digest": _record(collision_asset)["digest"],
    }, digest_field="digest")
    add("website_background_appearance_receipt", tmp_path / "appearance.json")
    add("website_background_reuse_receipt", tmp_path / "collision.json")
    assert appearance["digest"] and collision["digest"]

    parts = {}
    part_records = {}
    for part_id in ("carcass", "drawer"):
        relative = Path("producer/authoring/parts") / part_id
        image = stage / relative / "perspective.png"
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"\x89PNG\r\n" + part_id.encode())
        image_record = _record(image, provider_path=PROVIDER_STAGE / relative / image.name)
        result = _write(stage / relative / "result.json", {
            "status": "candidate_authored_pending_native_qualification",
            "review_images": [{"path": image_record["path"],
                               "sha256": image_record["digest"],
                               "size_bytes": image_record["size_bytes"]}],
        }, digest_field="result_digest")
        parts[part_id] = result
        part_records[part_id] = _record(
            stage / relative / "result.json",
            provider_path=PROVIDER_STAGE / relative / "result.json",
        )

    authored = _write(stage / "producer/authoring/result.json", {
        "schema_version": "task_object_astra_articulated_authoring_result.v1",
        "status": "parts_authored_pending_native_qualification",
        "physical_equivalence_proven": False,
        "plan": {"parts": {"carcass": {"link_role": "carcass"},
                           "drawer": {"link_role": "task_part"}}},
        "parts": parts,
    }, digest_field="result_digest")
    authored_record = _record(
        stage / "producer/authoring/result.json",
        provider_path=PROVIDER_STAGE / "producer/authoring/result.json",
    )
    receipt = _write(stage / "adapter/replacement_authoring_receipt.json", {
        "asset_kind": "articulated_assembly",
        "authoring_backend": "astra_cad_blender_v1",
        "replacement_identity": {"id": "asset-1", "version": "v1"},
        "astra_authoring_result": authored_record,
        "part_authoring_results": part_records,
    }, digest_field="result_digest")
    add("replacement_authoring_receipt", stage / "adapter/replacement_authoring_receipt.json")
    envelope = {
        "request": {"scene": {"rights": {"public_display_authorization": False}}},
        "recipe": {"subject_identity": receipt["replacement_identity"]},
        "materialized_references": [{"contract_path": "scene.source_manifest",
                                     **_record(tmp_path / "source.json"),
                                     "materialized_path": str(tmp_path / "source.json"),
                                     "full_byte_service_account_readback_passed": True}],
    }
    output = tmp_path / "preview"
    output.mkdir()
    result, selection = publication_inputs(
        envelope=envelope, stage_results=[{"output_artifacts": artifacts}], output_root=output,
    )
    assert "carcass" in selection["rationale"]
    assert "one part" in selection["rationale"]
    assert result["configured_task_thumbnail"].read_bytes().endswith(b"carcass")
    assert json.loads(result["appearance_visual_review_receipt"].read_text())["authoring_result_digest"] == authored["result_digest"]

    (stage / "producer/authoring/parts/carcass/result.json").write_text("{}")
    with pytest.raises(TaskEvaluationSceneConfigurationPublicationError):
        publication_inputs(
            envelope=envelope, stage_results=[{"output_artifacts": artifacts}], output_root=output,
        )
