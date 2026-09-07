"""Reuse raw image-edit candidates with their original request and billing lineage."""
from __future__ import annotations

import hashlib
import io
import json
import shutil
from pathlib import Path
from collections.abc import Mapping

from .decision_evidence_contracts import canonical_digest


RETAINED_FILE_FIELDS = ("source_runtime_request", "source_runtime_result", "candidate")


def materialize_retained_selection(*, source_request_path: Path, source_result_path: Path,
                                  source_output_root: Path, camera_ids: list[str], output_root: Path) -> Path:
    """Retain selected raw candidates and original receipts without making calls."""
    request = json.loads(source_request_path.read_text())
    result = json.loads(source_result_path.read_text())
    if output_root.exists() or not camera_ids or len(set(camera_ids)) != len(camera_ids):
        raise ValueError("semantic_teacher_retained_selection_output_invalid")
    request_frames = {f["camera_id"]: f for t in request["tasks"] for f in t["frames"]}
    results = {f["camera_id"]: (t["task_id"], f) for t in result["tasks"] for f in t["frames"]}
    sources = []
    for camera_id in camera_ids:
        task_id, frame = results[camera_id]
        sources.append((task_id, camera_id, _bound(source_output_root, frame["semantic_teacher_frame"])))
    output_root.mkdir(parents=True)
    def copy(source, relative):
        target = output_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        return {"relative_path": relative, "size_bytes": target.stat().st_size,
                "sha256": "sha256:" + hashlib.sha256(target.read_bytes()).hexdigest()}
    old_request = copy(source_request_path, "original-request.json")
    old_result = copy(source_result_path, "original-result.json")
    rows = [{"task_id": task, "camera_id": camera, "source_runtime_request": old_request,
             "source_runtime_result": old_result, "candidate": copy(path, f"candidates/{index:04d}.png")}
            for index, (task, camera, path) in enumerate(sources)]
    selection = {"schema_version": "semantic_teacher_retained_candidate_selection.v1",
                 "status": "selected_unreviewed_candidates", "candidates": rows}
    selection["selection_digest"] = canonical_digest(selection, digest_field="selection_digest")
    path = output_root / "selection.json"
    path.write_text(json.dumps(selection, sort_keys=True) + "\n")
    load_retained_selection(selection_path=path, render={"derived_frames": [
        {"camera_id": camera, "digest": request_frames[camera]["input_rgb"]["sha256"]} for camera in camera_ids]})
    return path


def load_retained_selection(*, selection_path: Path, render: Mapping) -> list:
    """Read an operator-selected historical candidate inventory for exact frames."""
    selection = json.loads(selection_path.read_text())
    if (selection.get("schema_version") != "semantic_teacher_retained_candidate_selection.v1"
            or selection.get("selection_digest") != canonical_digest(selection, digest_field="selection_digest")
            or not isinstance(selection.get("candidates"), list)):
        raise ValueError("semantic_teacher_retained_selection_invalid")
    by_camera = {row["camera_id"]: row for row in render["derived_frames"]}
    rows = []
    seen = set()
    for row in selection["candidates"]:
        if row["camera_id"] in seen or row["camera_id"] not in by_camera:
            raise ValueError("semantic_teacher_retained_selection_camera_invalid")
        seen.add(row["camera_id"])
        copied = {"task_id": row["task_id"], "camera_id": row["camera_id"]}
        for field in RETAINED_FILE_FIELDS:
            path = _bound(selection_path.parent, row[field])
            copied[field] = {"path": str(path.resolve()), "size_bytes": row[field]["size_bytes"],
                "digest": row[field]["sha256"]}
        original = json.loads(Path(copied["source_runtime_request"]["path"]).read_text())
        matches = [f for t in original["tasks"] if t["task_id"] == row["task_id"]
                   for f in t["frames"] if f["camera_id"] == row["camera_id"]]
        current = by_camera[row["camera_id"]]
        if (len(matches) != 1 or not _matches_staged_rgb(current, matches[0]["input_rgb"]["sha256"])):
            raise ValueError("semantic_teacher_retained_selection_source_changed")
        # Validate the original request, result and output together before the
        # bundle is eligible for paid admission. The runtime checks again using
        # its newly constructed request and corrected mask.
        validation_request = {**original, "retained_candidates": [row]}
        load_retained_candidates(request=validation_request, request_root=selection_path.parent)
        rows.append(copied)
    return rows


def _matches_staged_rgb(current: Mapping, expected_digest: str) -> bool:
    """Reproduce the canonical RGB PNG staging, after verifying source bytes."""
    digest = current.get("digest", current.get("sha256"))
    if digest == expected_digest:
        return True
    source = Path(str(current.get("path") or ""))
    if (source.is_symlink() or not source.is_file()
            or source.stat().st_size != current.get("size_bytes")
            or "sha256:" + hashlib.sha256(source.read_bytes()).hexdigest() != digest):
        return False
    from PIL import Image
    encoded = io.BytesIO()
    with Image.open(source) as original:
        original.convert("RGB").save(encoded, format="PNG")
    return "sha256:" + hashlib.sha256(encoded.getvalue()).hexdigest() == expected_digest


def attach_retained_candidates(*, runtime_request_path: Path, candidates: list) -> None:
    """Stage byte-bound historical evidence and sign the new request's reuse plan."""
    if not candidates:
        return
    request = json.loads(runtime_request_path.read_text())
    root = runtime_request_path.parent
    staged = []
    for index, row in enumerate(candidates):
        copied = {"task_id": row["task_id"], "camera_id": row["camera_id"]}
        for field in RETAINED_FILE_FIELDS:
            record = row[field]
            source = Path(record["path"])
            if (source.is_symlink() or not source.is_file()
                    or source.stat().st_size != record["size_bytes"]
                    or "sha256:" + hashlib.sha256(source.read_bytes()).hexdigest()
                    != record.get("sha256", record.get("digest"))):
                raise ValueError("semantic_teacher_retained_candidate_bytes_invalid")
            target = root / "retained" / str(index) / (field + (".png" if field == "candidate" else ".json"))
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            copied[field] = {"relative_path": target.relative_to(root).as_posix(),
                "size_bytes": target.stat().st_size,
                "sha256": "sha256:" + hashlib.sha256(target.read_bytes()).hexdigest()}
        staged.append(copied)
    request["retained_candidates"] = staged
    load_retained_candidates(request=request, request_root=root)
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    runtime_request_path.write_text(json.dumps(request, sort_keys=True) + "\n")


def _bound(root: Path, row: Mapping) -> Path:
    relative = Path(str(row.get("relative_path") or ""))
    path = root / relative
    if (relative.is_absolute() or not relative.parts or ".." in relative.parts
            or path.is_symlink() or not path.is_file()
            or path.stat().st_size != row.get("size_bytes")
            or "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest() != row.get("sha256")):
        raise ValueError("semantic_teacher_retained_candidate_bytes_invalid")
    return path


def load_retained_candidates(*, request: Mapping, request_root: Path) -> dict:
    """Validate all reuse before any new provider call; never reuse preservation rows."""
    selected = request.get("retained_candidates", [])
    if not isinstance(selected, list):
        raise ValueError("semantic_teacher_retained_candidate_inventory_invalid")
    current = {(t["task_id"], f["camera_id"]): f for t in request["tasks"] for f in t["frames"]}
    admitted = {}
    for row in selected:
        if not isinstance(row, Mapping):
            raise ValueError("semantic_teacher_retained_candidate_inventory_invalid")
        key = (row.get("task_id"), row.get("camera_id"))
        if key not in current or key in admitted:
            raise ValueError("semantic_teacher_retained_candidate_identity_invalid")
        documents = []
        for field, digest_field, schema in (
            ("source_runtime_request", "request_digest", "semantic_teacher_image_edit_runtime_request.v1"),
            ("source_runtime_result", "result_digest", "semantic_teacher_image_edit_runtime_result.v1"),
        ):
            value = json.loads(_bound(request_root, row[field]).read_text())
            if (value.get("schema_version") != schema
                    or value.get(digest_field) != canonical_digest(value, digest_field=digest_field)):
                raise ValueError("semantic_teacher_retained_candidate_receipt_invalid")
            documents.append(value)
        original_request, original_result = documents
        previous_frames = [f for t in original_request["tasks"] if t["task_id"] == key[0]
                           for f in t["frames"] if f["camera_id"] == key[1]]
        results = [f for t in original_result["tasks"] if t["task_id"] == key[0]
                   for f in t["frames"] if f["camera_id"] == key[1]]
        if len(previous_frames) != 1 or len(results) != 1:
            raise ValueError("semantic_teacher_retained_candidate_identity_invalid")
        before, result = previous_frames[0], results[0]
        frame = current[key]
        if (original_result.get("source_runtime_request_digest") != original_request["request_digest"]
                or original_result.get("status") != "completed_unreviewed_semantic_teacher_candidates"
                or original_result.get("model_snapshot") != request["backend"]["execution"]["model_snapshot"]
                or original_result.get("backend_id") != request["backend"]["registry_entry"]["backend_id"]
                or before.get("frame_role", "semantic_edit") != "semantic_edit"
                or frame.get("frame_role", "semantic_edit") != "semantic_edit"
                or result.get("terminal_state") != "completed_unreviewed_candidate"
                or before["input_rgb"]["sha256"] != frame["input_rgb"]["sha256"]
                or result.get("source_rgb_sha256") != before["input_rgb"]["sha256"]
                or result.get("edit_mask_sha256") != before["edit_mask"]["sha256"]):
            raise ValueError("semantic_teacher_retained_candidate_binding_invalid")
        candidate = _bound(request_root, row["candidate"])
        if any(row["candidate"][field] != result["semantic_teacher_frame"][field]
               for field in ("size_bytes", "sha256")):
            raise ValueError("semantic_teacher_retained_candidate_output_mismatch")
        admitted[key] = {"path": candidate, "lineage": {
            "source_runtime_request_digest": original_request["request_digest"],
            "source_runtime_result_digest": original_result["result_digest"],
            "original_edit_mask_sha256": before["edit_mask"]["sha256"],
            "current_repair_support_sha256": frame["edit_mask"]["sha256"],
            "original_provider_usage": result.get("provider_usage"),
            "original_computed_editor_cost_usd": result.get("computed_editor_cost_usd"),
            "source_evidence": dict(row),
            "new_image_generation_performed": False,
            "visual_review_required": True,
        }}
    return admitted
