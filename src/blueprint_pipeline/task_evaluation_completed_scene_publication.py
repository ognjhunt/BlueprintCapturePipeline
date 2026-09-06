"""Reopen owner source admission before publishing a completed-scene package."""
from __future__ import annotations

from pathlib import Path

from .task_evaluation_completed_scene_source import owned_upload
from .task_evaluation_owner_source_store import source_uri
from .task_evaluation_scene_configuration_submission_inputs import checked_file, read, sha
from .task_evaluation_scene_owner_authority import reopen_scene_intent
from .task_evaluation_scene_progression_state import require

SCHEMA = "task_evaluation_completed_scene_source_intake.v1"
RELATIVE_PATH = "provenance/owner_source_intake.v1.json"


def verified_owner_source_inventory(root: Path, manifest: dict) -> set[tuple[str, str, int]]:
    """A package cannot invent a publisher or authorize its own source bytes."""
    value = read(root / RELATIVE_PATH)
    require(value.get("schema_version") == SCHEMA, "completed_source_intake_invalid")
    intent = reopen_scene_intent(value.get("scene_intent_authority"))
    ref = value.get("source_binding")
    require(isinstance(ref, dict), "completed_source_binding_missing")
    binding = read(checked_file(ref["path"], ref), digest_field="binding_digest")
    task = read(root / "provenance/completed_task_request.v1.json")
    request = intent["request"]
    require(value.get("intent_digest") == intent["intent_digest"]
            and binding.get("intent_digest") == intent["intent_digest"]
            and binding.get("task_digest") == intent["task_content_digest"]
            and binding.get("owner") == request["owner"]
            and task.get("scene_intent_authority") == value["scene_intent_authority"]
            and task.get("source_binding") == ref
            and task["task_identity"]["id"] == request["task"]["task_id"]
            and task["expected_production_commit"] == manifest["source_commit"],
            "completed_source_owner_binding_mismatch")
    expected = set()
    sources = [{**request["source"], "rights_reference": request["consent"]["rights_reference"]}]
    if request["source"]["kind"] == "gaussian_splat":
        sources.append(request["source"]["collision_mesh"])
    for source in sources:
        verified = owned_upload(source, request["owner"], {})
        require(verified is not None, "completed_source_upload_missing")
        _, asset = verified
        path = asset["object_path"]
        digest, size = sha(path), path.stat().st_size
        expected.add((source_uri(digest, asset["source_original_filename"]), digest, size))
    if task.get("splat_normalization") is not None:
        ref = task["splat_normalization"]
        path = checked_file(ref["path"], ref)
        normalization = read(path, digest_field="normalization_digest")
        require(normalization.get("schema_version") == "task_evaluation_completed_splat_normalization.v1"
                and normalization.get("source_digest") == binding["references"]["primary"]["sha256"]
                and normalization.get("declared_coordinate_frame") == binding["coordinate_frame"]
                and normalization.get("source_bytes_unchanged") is True
                and normalization.get("physical_scale_measured") is False
                and normalization.get("reconstruction_performed") is False, "completed_splat_normalization_binding_invalid")
        output = normalization["output"]
        checked_file(path.parent / output["relative_path"], output)
        expected.add((source_uri(output["sha256"], "normalized.ply"), output["sha256"], output["size_bytes"]))
    recorded = value.get("artifacts")
    require(isinstance(recorded, list) and len(recorded) == len(expected)
            and all(isinstance(row, dict) and set(row) == {"uri", "digest", "size_bytes"}
                    for row in recorded)
            and {(row["uri"], row["digest"], row["size_bytes"]) for row in recorded} == expected,
            "completed_source_inventory_mismatch")
    return expected
