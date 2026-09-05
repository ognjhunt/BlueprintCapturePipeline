"""Admit full source content carried by a deleted/retained Gaussian partition."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .sam31_contribution_disclosure import validate_full_source_disclosure
from .sealed_camera_render import _standard_ply_vertex_count
from .task_evaluation_scene_configuration_sam31_plan import HOST_ROOTS
from .task_evaluation_scene_configuration_submission_inputs import checked_file, sha

PURPOSE = "configured_scene_partitioned_source_processing"
SCHEMA = "configured_scene_partitioned_source_disclosure.v1"


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise ValueError("configuration_partition_disclosure_" + code)


def _record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": sha(path), "size_bytes": path.stat().st_size}


def prepare_partition_disclosure(*, task_authority: Mapping[str, Any], conversion_path: Path,
        standard_splat_path: Path, original_source_path: Path, deleted_path: Path, retained_path: Path,
        expected_source_commit: str, publisher_scene_id: str) -> dict[str, Any]:
    proof = validate_full_source_disclosure(task_authority=task_authority,
        conversion_path=conversion_path, standard_splat_path=standard_splat_path,
        original_source_path=original_source_path, expected_source_commit=expected_source_commit,
        publisher_scene_id=publisher_scene_id, approved_roots=HOST_ROOTS, purpose=PURPOSE)
    count = proof["source_binding"]["source_gaussian_count"]
    deleted_count, retained_count = (_standard_ply_vertex_count(path) for path in (deleted_path, retained_path))
    _require(type(deleted_count) is int and deleted_count > 0 and type(retained_count) is int
             and retained_count > 0 and deleted_count + retained_count == count, "partition_counts_invalid")
    value = {"schema_version": SCHEMA, "status": "explicit_full_source_partition_disclosure_verified",
        "purpose": PURPOSE, "source_commit": expected_source_commit,
        "publisher_scene_id": publisher_scene_id, "full_source_disclosure": proof,
        "source_inputs": {"conversion": _record(conversion_path), "standard": _record(standard_splat_path),
                          "original": _record(original_source_path)},
        "partitions": {"source_object_candidate": {**_record(deleted_path), "gaussian_count": deleted_count},
                       "retained_scene_without_source_object": {**_record(retained_path), "gaussian_count": retained_count}},
        "full_source_scene_content_in_provider_packet": True,
        "original_downloaded_file_in_provider_packet": False,
        "frame_permission_used_as_full_source_authority": False,
        "provider_training_authorized": False, "public_redistribution_authorized": False,
        "disclosure_digest": ""}
    value["disclosure_digest"] = canonical_digest(value, digest_field="disclosure_digest")
    return value


def require_partition_disclosure(*, render: Mapping[str, Any], configuration: Mapping[str, Any],
        expected_source_commit: str) -> dict[str, Any]:
    """Reopen admission and both actual payload files before bundle staging."""
    proof = render.get("full_source_scene_content_disclosure")
    _require(isinstance(proof, Mapping) and proof.get("schema_version") == SCHEMA
             and proof.get("disclosure_digest") == canonical_digest(proof, digest_field="disclosure_digest"),
             "proof_missing_or_invalid")
    sources, parts = proof["source_inputs"], proof["partitions"]
    paths = {name: checked_file(row["path"], row) for name, row in sources.items()}
    partition_paths = {name: checked_file(row["path"], row) for name, row in parts.items()}
    for name, row in parts.items():
        staged_row = render["derived_gaussian_cutout"][name]
        staged = checked_file(staged_row["path"], {
            "sha256": staged_row.get("sha256") or staged_row.get("digest"),
            "size_bytes": staged_row["size_bytes"]})
        _require(sha(staged) == row["sha256"] and staged.stat().st_size == row["size_bytes"], "partition_bytes_changed")
    reopened = prepare_partition_disclosure(task_authority=configuration["human_authority"],
        conversion_path=paths["conversion"], standard_splat_path=paths["standard"], original_source_path=paths["original"],
        deleted_path=partition_paths["source_object_candidate"],
        retained_path=partition_paths["retained_scene_without_source_object"],
        expected_source_commit=expected_source_commit, publisher_scene_id=configuration["source_object"]["scene_id"])
    _require(reopened == proof and render.get("full_source_scene_content_in_provider_packet") is True,
             "proof_binding_changed")
    return reopened
