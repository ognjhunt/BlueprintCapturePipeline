import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_scene_configuration_publication import _sha256_and_size
from blueprint_pipeline.website_scene_publication import publication_inputs, STATUS


def case(root):
    artifacts = []

    def write(name, value, role=None, field="digest"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(value, dict):
            value[field] = canonical_digest(value, digest_field=field)
            path.write_text(json.dumps(value))
        else:
            path.write_bytes(value)
        digest, size = _sha256_and_size(path)
        row = {"path": str(path), "digest": digest, "size_bytes": size}
        if role:
            artifacts.append({"role": role, **row})
        return row

    prep = {}
    source = write("prep.json", prep)
    appearance = write(
        "appearance.usdc", b"appearance", "configured_appearance_without_source_object"
    )
    collision = write("collision.usda", b"collision", "configured_collision_without_source_object")
    write(
        "appearance.json",
        {
            "schema_version": "website_native_appearance.v1",
            "status": "native_appearance_authored",
            "binding": {"preparation_digest": prep["digest"]},
            "appearance_removal_performed": False,
            "renderer_qualified": False,
            "physical_measurement_proven": False,
            "artifact": appearance,
        },
        "website_background_appearance_receipt",
    )
    write(
        "collision.json",
        {
            "schema_version": "website_prepared_background_result.v1",
            "status": "prepared_background_reused",
            "preparation_digest": prep["digest"],
            "source_prim_excision_performed": False,
            "background_bytes_unchanged": True,
            "physical_measurement_proven": False,
            "collision_digest": collision["digest"],
        },
        "website_background_reuse_receipt",
    )
    provider = Path(
        "/workspace/task_evaluation_scene_configuration_provider_bundle/runtime_output/stages/stage-3"
    )

    def portable(row):
        return {**row, "path": str(provider / Path(row["path"]).relative_to(root / "stage-3"))}

    image = write("stage-3/producer/render.png", b"retained-image")
    authored = write(
        "stage-3/producer/result.json",
        {
            "status": "candidate_authored_pending_native_qualification",
            "object_id": "object",
            "physical_equivalence_proven": False,
            "review_images": [portable(image)],
        },
        field="result_digest",
    )
    write(
        "stage-3/adapter/receipt.json",
        {
            "authoring_backend": "astra_cad_blender_v1",
            "replacement_identity": {"id": "object"},
            "astra_authoring_result": portable(authored),
        },
        "replacement_authoring_receipt",
        "result_digest",
    )
    envelope = {
        "request": {"scene": {"website_native_inputs": {}, "rights": {}}},
        "recipe": {"subject_identity": {"id": "object"}},
        "materialized_references": [
            {
                **source,
                "materialized_path": source["path"],
                "contract_path": "scene.source_manifest",
                "full_byte_service_account_readback_passed": True,
            }
        ],
    }
    out = root / "publication"
    out.mkdir()
    return dict(envelope=envelope, stage_results=[{"output_artifacts": artifacts}], output_root=out)


def test_website_uses_exact_retained_preview_without_legacy_removal_claim(tmp_path):
    kwargs = case(tmp_path)
    artifacts, selection = publication_inputs(**kwargs)
    assert artifacts["configured_task_thumbnail"].read_bytes() == b"retained-image"
    assert selection["appearance_review_status"] == STATUS
    assert selection["reviewer"]["kind"] == "system"
    assert (
        json.loads(artifacts["appearance_removal_receipt"].read_text())[
            "appearance_removal_performed"
        ]
        is False
    )


@pytest.mark.parametrize("fault", ["image", "source", "appearance", "authoring", "public"])
def test_rejects_changed_artifacts_or_public_upgrade(tmp_path, fault):
    kwargs = case(tmp_path)
    if fault == "public":
        kwargs["envelope"]["request"]["scene"]["rights"]["public_display_authorization"] = {
            "status": "authorized"
        }
    else:
        target = {
            "image": "stage-3/producer/render.png",
            "source": "prep.json",
            "appearance": "appearance.usdc",
            "authoring": "stage-3/producer/result.json",
        }[fault]
        (tmp_path / target).write_bytes(b"changed")
    with pytest.raises(RuntimeError):
        publication_inputs(**kwargs)


PNG = b"\x89PNG\r\n\x1a\n"


def artisan_case(root):
    """The real ARTISAN layout (capture-447ae4f4, 2026-11-03): build and review rounds, open and crop views."""
    kwargs = case(root)
    stage = root / "stage-3"
    run = "producer/artisan_runtime/workspace/data/cache/runs/pl_test/workspace"
    renders = stage / run / "renders"
    for name, data in {
        "P4_build_r01/01/perspective.png": PNG + b"build",
        "P5_review_r01/01/perspective__r01c01.png": PNG + b"review-closed",
        "P5_review_r01/01/perspective_open__r01c01.png": PNG + b"review-open",
        "P5_review_r01/01/front__r01c01.png": PNG + b"review-front",
        "P5_review_r01/02/perspective__r01c01_crop__r01c02.png": PNG + b"crop",
    }.items():
        path = renders / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    provider = "/workspace/task_evaluation_scene_configuration_provider_bundle/runtime_output/stages/stage-3/"
    receipt = {
        "schema_version": "task_object_artisan_candidate.v1",
        "backend": "artisan_v1",
        "status": "authored_candidate_pending_qualification",
        "asset_kind": "rigid_body",
        "replacement_identity": {"id": "object"},
        "physical_equivalence_proven": False,
        "upstream_terminal": {"paths": {"root": provider + run}},
    }
    receipt["result_digest"] = canonical_digest(receipt, digest_field="result_digest")
    path = stage / "adapter/receipt.json"
    path.write_text(json.dumps(receipt))
    digest, size = _sha256_and_size(path)
    for row in kwargs["stage_results"][0]["output_artifacts"]:
        if row["role"] == "replacement_authoring_receipt":
            row.update(digest=digest, size_bytes=size)
    return kwargs, renders, receipt


def test_artisan_receipt_publishes_its_last_review_render(tmp_path):
    kwargs, _, receipt = artisan_case(tmp_path)
    artifacts, selection = publication_inputs(**kwargs)
    assert artifacts["configured_task_thumbnail"].read_bytes() == PNG + b"review-closed"
    assert selection["camera_id"] == "perspective__r01c01"
    assert "ARTISAN review render (P5_review_r01" in selection["rationale"]
    assert "not the captured room" in selection["rationale"]
    preview = json.loads(artifacts["appearance_visual_review_receipt"].read_text())
    assert preview["authoring_receipt_digest"] == preview["authoring_result_digest"] == receipt["result_digest"]


def test_artisan_build_render_only_when_no_review_round(tmp_path):
    kwargs, renders, _ = artisan_case(tmp_path)
    for path in sorted((renders / "P5_review_r01").rglob("*"), reverse=True):
        path.rmdir() if path.is_dir() else path.unlink()
    (renders / "P5_review_r01").rmdir()
    artifacts, selection = publication_inputs(**kwargs)
    assert artifacts["configured_task_thumbnail"].read_bytes() == PNG + b"build"
    assert "ARTISAN build render (P4_build_r01" in selection["rationale"]


@pytest.mark.parametrize("fault", ["no_renders", "not_png", "symlink", "outside", "schema", "identity"])
def test_artisan_refuses_unbound_or_unretained_renders(tmp_path, fault):
    kwargs, renders, receipt = artisan_case(tmp_path)
    image = renders / "P5_review_r01/01/perspective__r01c01.png"
    if fault == "no_renders":
        for path in sorted(renders.rglob("*"), reverse=True):
            path.rmdir() if path.is_dir() else path.unlink()
    elif fault == "not_png":
        image.write_bytes(b"not an image")
    elif fault == "symlink":
        outside = tmp_path / "outside.png"
        outside.write_bytes(PNG + b"outside")
        image.unlink()
        image.symlink_to(outside)
    else:
        if fault == "outside":
            receipt["upstream_terminal"]["paths"]["root"] = "/workspace/elsewhere"
        elif fault == "schema":
            receipt["schema_version"] = "task_object_artisan_sdk_candidate.v1"
        else:
            receipt["replacement_identity"] = {"id": "another-object"}
        receipt["result_digest"] = canonical_digest(receipt, digest_field="result_digest")
        path = tmp_path / "stage-3/adapter/receipt.json"
        path.write_text(json.dumps(receipt))
        digest, size = _sha256_and_size(path)
        for row in kwargs["stage_results"][0]["output_artifacts"]:
            if row["role"] == "replacement_authoring_receipt":
                row.update(digest=digest, size_bytes=size)
    with pytest.raises(RuntimeError):
        publication_inputs(**kwargs)
