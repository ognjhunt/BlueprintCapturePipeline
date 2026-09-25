"""Publish website background reuse without inventing legacy ArtiFixer evidence."""

from pathlib import Path
import json
import shutil

from .decision_evidence_contracts import canonical_digest, canonical_json

STATUS = "prepared_scene_ungraded"
WARNING = "Generated task-object preview; scene appearance ungraded"
TRUTH_SOURCE = "website_prepared_background_with_generated_task_object"


def publication_inputs(*, envelope, stage_results, output_root):
    from .task_evaluation_scene_configuration_publication import (
        _artifact,
        _sha256_and_size,
        _materialized_reference_file,
        MAX_TASK_THUMBNAIL_SIZE_BYTES,
        TaskEvaluationSceneConfigurationPublicationError,
    )

    def require(ok):
        if not ok:
            raise TaskEvaluationSceneConfigurationPublicationError(
                "website_publication_evidence_invalid"
            )

    require(not envelope["request"]["scene"].get("rights", {}).get("public_display_authorization"))
    artifacts = {}
    for old, actual in (
        ("appearance_removal_receipt", "website_background_appearance_receipt"),
        ("collision_excision_receipt", "website_background_reuse_receipt"),
    ):
        artifacts[old] = _artifact(stage_results, role=actual)[1]
    appearance = json.loads(artifacts["appearance_removal_receipt"].read_text())
    collision = json.loads(artifacts["collision_excision_receipt"].read_text())
    _, source = _materialized_reference_file(envelope, contract_path="scene.source_manifest")
    preparation = json.loads(source.read_text())
    require(
        appearance.get("schema_version") == "website_native_appearance.v1"
        and appearance.get("status") == "native_appearance_authored"
        and appearance.get("appearance_removal_performed") is False
        and appearance.get("renderer_qualified") is False
        and collision.get("schema_version") == "website_prepared_background_result.v1"
        and collision.get("status") == "prepared_background_reused"
        and collision.get("source_prim_excision_performed") is False
        and collision.get("background_bytes_unchanged") is True
        and preparation.get("digest") == canonical_digest(preparation, digest_field="digest")
        and collision.get("preparation_digest") == preparation.get("digest")
        and appearance.get("binding", {}).get("preparation_digest")
        == collision.get("preparation_digest")
        and all(
            row.get("physical_measurement_proven") is False
            and row.get("digest") == canonical_digest(row, digest_field="digest")
            for row in (appearance, collision)
        )
    )
    for role, record in (
        ("configured_appearance_without_source_object", appearance["artifact"]),
        ("configured_collision_without_source_object", {"digest": collision["collision_digest"]}),
    ):
        row, _ = _artifact(stage_results, role=role)
        require(row["digest"] == record["digest"])

    _, receipt_path = _artifact(stage_results, role="replacement_authoring_receipt")
    receipt = json.loads(receipt_path.read_text())
    require(
        receipt.get("authoring_backend") == "astra_cad_blender_v1"
        and receipt.get("replacement_identity") == envelope["recipe"]["subject_identity"]
        and receipt.get("result_digest") == canonical_digest(receipt, digest_field="result_digest")
    )
    # Resolve only exact paths inside this stage's retained archive. Never read
    # the control plane's mutable /workspace or search for a matching basename.
    stage_root = receipt_path.parent.parent.resolve()
    provider_stage = Path(
        "/workspace/task_evaluation_scene_configuration_provider_bundle/runtime_output/stages/stage-3"
    )

    def retained(record):
        try:
            relative = Path(record["path"]).relative_to(provider_stage)
            path = stage_root / relative
            require(
                ".." not in relative.parts
                and not path.is_symlink()
                and path.resolve().is_relative_to(stage_root)
                and path.is_file()
            )
            require(
                _sha256_and_size(path)
                == (record.get("digest", record.get("sha256")), record["size_bytes"])
            )
            return path
        except (KeyError, ValueError, OSError) as exc:
            raise TaskEvaluationSceneConfigurationPublicationError(
                "website_publication_retained_artifact_invalid"
            ) from exc

    authored = json.loads(retained(receipt["astra_authoring_result"]).read_text())
    require(
        authored.get("result_digest") == canonical_digest(authored, digest_field="result_digest")
        and authored.get("physical_equivalence_proven") is False
    )
    rationale = "First retained task-object studio render. Does not show or qualify the captured room."
    if receipt.get("asset_kind") == "articulated_assembly":
        parts = authored.get("parts")
        plan_parts = (authored.get("plan") or {}).get("parts")
        receipt_parts = receipt.get("part_authoring_results")
        require(
            authored.get("schema_version") == "task_object_astra_articulated_authoring_result.v1"
            and authored.get("status") == "parts_authored_pending_native_qualification"
            and isinstance(parts, dict) and bool(parts)
            and isinstance(plan_parts, dict) and set(plan_parts) == set(parts)
            and isinstance(receipt_parts, dict) and set(receipt_parts) == set(parts)
        )
        for part_id, part in parts.items():
            require(
                json.loads(retained(receipt_parts[part_id]).read_text()) == part
                and part.get("status") == "candidate_authored_pending_native_qualification"
                and part.get("result_digest") == canonical_digest(part, digest_field="result_digest")
            )
        fixed_parts = [part_id for part_id, spec in plan_parts.items()
                       if spec.get("link_role") in {"carcass", "fixed_part"}]
        require(len(fixed_parts) == 1 and bool(parts[fixed_parts[0]].get("review_images")))
        selected_part = fixed_parts[0]
        image = retained(parts[selected_part]["review_images"][0])
        rationale = (f"First retained {selected_part} studio render of the articulated assembly. "
                     "It shows one part, not the assembled object or captured room.")
    else:
        require(
            authored.get("status") == "candidate_authored_pending_native_qualification"
            and authored.get("object_id") == receipt["replacement_identity"]["id"]
            and bool(authored.get("review_images"))
        )
        image = retained(authored["review_images"][0])
    require(0 < image.stat().st_size <= MAX_TASK_THUMBNAIL_SIZE_BYTES and image.suffix == ".png")
    thumbnail = Path(output_root) / "generated_task_object.png"
    shutil.copyfile(image, thumbnail)
    digest, _ = _sha256_and_size(thumbnail)
    selection = {
        "camera_id": image.stem,
        "frame_digest": digest,
        "rationale": rationale,
        "reviewer": {
            "kind": "system",
            "identity": "website-task-object-preview",
            "runtime": "retained_artifact_selection",
            "model": "none",
        },
        "appearance_review_status": STATUS,
    }
    preview = {
        "schema_version": "website_task_object_preview.v1",
        "status": STATUS,
        "authoring_receipt_digest": receipt["result_digest"],
        "authoring_result_digest": authored["result_digest"],
        "selection": selection,
        "warning_label": WARNING,
        "scene_appearance_graded": False,
        "capture_or_physical_evidence": False,
    }
    preview["digest"] = canonical_digest(preview, digest_field="digest")
    preview_path = Path(output_root) / "website_task_object_preview.json"
    preview_path.write_text(canonical_json(preview) + "\n")
    artifacts.update(
        appearance_visual_review_receipt=preview_path, configured_task_thumbnail=thumbnail
    )
    return artifacts, selection
