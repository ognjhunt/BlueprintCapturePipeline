"""Retain the previous attempt's reviewer-accepted image edits for byte-identical inputs.

InteriorGS 840938 (2026-09-13) paid for sixteen semantic-teacher edits and two
reviews on every retry (about $1.90 an attempt) although the source frames,
masks and backend never changed between attempts. The retained-candidate
contract (``semantic_teacher_candidate_reuse``) already binds a raw edit to its
original request, result and source digests; what was missing was anyone
building that selection from the last attempt.

Discovery scans the semantic-pretraining workspaces on this host, newest
first, for one whose sealed runtime request used the same backend snapshot and
the same ``input_rgb`` bytes for every camera the current request names. Its
last independent review (after the bounded repair when one ran) decides which
cameras are retained: only frames the reviewer accepted, and only when the
reviewed frame digest matches the sealed frame that the workspace's own
locality seal or repair merge attributes to that camera. Rejected views are
edited afresh; nothing here calls a provider.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .semantic_teacher_candidate_reuse import (
    load_retained_selection,
    materialize_retained_selection_from_sources,
)

SCHEMA = "semantic_teacher_retained_candidate_discovery.v1"
DISCOVERY_ENV = "BLUEPRINT_SCENE_CONFIGURATION_RETAINED_CANDIDATE_DISCOVERY"
DISCOVERY_ROOT_ENV = "BLUEPRINT_SCENE_CONFIGURATION_RETAINED_CANDIDATE_DISCOVERY_ROOT"
RUNTIME = "output/released_artifixer_runtime"
REQUEST = "semantic_teacher_packet/semantic_teacher_image_edit_runtime_request.v1.json"
RESULT = "semantic_teacher_output/semantic_teacher_image_edit_runtime_result.v1.json"
SEAL = "semantic_teacher_exact_support_locality_seal/task_evaluation_artifixer_semantic_locality_seal.v1.json"
REVIEW_1 = "semantic_target_review/independent_visual_review/task_evaluation_artifixer_ai_visual_review_execution.v1.json"
REVIEW_2 = "semantic_target_recovery/semantic_target_review_after_repair/independent_visual_review/task_evaluation_artifixer_ai_visual_review_execution.v1.json"
MERGE = "semantic_target_recovery/selective_semantic_repair_merged/task_evaluation_artifixer_selective_repair_merge.v1.json"
REPAIR_REQUEST = "semantic_target_recovery/selective_semantic_repair_request/semantic_teacher_image_edit_runtime_request.v1.json"
REPAIR_RESULT = "semantic_target_recovery/selective_semantic_repair_output/semantic_teacher_image_edit_runtime_result.v1.json"
ACCEPTANCE_FIELDS = ("source_object_absent", "repair_is_locally_plausible", "preserves_non_target_content")
MAX_WORKSPACES = 64


def _sealed(path: Path, field: str) -> dict[str, Any] | None:
    if path.is_symlink() or not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(value, Mapping) or value.get(field) != canonical_digest(dict(value), digest_field=field):
        return None
    return dict(value)


def _request_frames(request: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(f["camera_id"]): f for t in request.get("tasks") or [] for f in t.get("frames") or []}


def _backend(request: Mapping[str, Any]) -> tuple[str, str]:
    backend = request.get("backend") or {}
    return (
        str((backend.get("execution") or {}).get("model_snapshot") or ""),
        str((backend.get("registry_entry") or {}).get("backend_id") or ""),
    )


def _accepted_cameras(review: Mapping[str, Any]) -> dict[str, str]:
    """camera_id -> reviewed frame digest for frames the reviewer fully accepted."""
    accepted: dict[str, str] = {}
    for row in review.get("frames") or []:
        if not isinstance(row, Mapping) or row.get("decision") != "accepted":
            continue
        if not all(row.get(field) is True for field in ACCEPTANCE_FIELDS):
            continue
        camera = str(row.get("camera_id") or "")
        digest = str(row.get("frame_sha256") or "")
        if camera and digest.startswith("sha256:"):
            accepted[camera] = digest
    return accepted


def _workspace_plan(runtime: Path, current: Mapping[str, Mapping[str, Any]], backend: tuple[str, str]) -> dict[str, Any] | None:
    """Which cameras this workspace can supply, with the digest chain that justifies each."""
    request = _sealed(runtime / REQUEST, "request_digest")
    result = _sealed(runtime / RESULT, "result_digest")
    if request is None or result is None or _backend(request) != backend:
        return None
    if result.get("status") != "completed_unreviewed_semantic_teacher_candidates":
        return None
    previous = _request_frames(request)
    if any(camera not in previous or previous[camera].get("input_rgb", {}).get("sha256") != frame.get("input_rgb", {}).get("sha256")
           for camera, frame in current.items()):
        return None
    review = _sealed(runtime / REVIEW_2, "execution_digest")
    merged = _sealed(runtime / MERGE, "merge_digest") if review is not None else None
    if review is not None and merged is None:
        return None
    if review is None:
        review = _sealed(runtime / REVIEW_1, "execution_digest")
        if review is None:
            return None
    if review.get("status") != "completed":
        return None
    sealed_by_camera: dict[str, tuple[str, str]] = {}
    if merged is not None:
        for row in merged.get("frame_inventory") or []:
            source = "repair" if row.get("role") == "selectively_repaired_semantic_frame" else "base"
            sealed_by_camera[str(row.get("camera_id"))] = (str(row.get("sha256")), source)
    else:
        seal = _sealed(runtime / SEAL, "receipt_digest")
        if seal is None:
            return None
        for row in seal.get("frames") or []:
            sealed_by_camera[str(row.get("camera_id"))] = (str((row.get("sealed_semantic_teacher") or {}).get("sha256")), "base")
    accepted = _accepted_cameras(review)
    base_cameras, repair_cameras, skipped = [], [], []
    result_cameras = _request_frames(result)
    repair_result = _sealed(runtime / REPAIR_RESULT, "result_digest") if merged is not None else None
    repair_cameras_available = _request_frames(repair_result) if repair_result else {}
    for camera in current:
        if camera not in accepted:
            skipped.append({"camera_id": camera, "reason": "not_accepted_by_last_review"})
            continue
        sealed = sealed_by_camera.get(camera)
        if sealed is None or sealed[0] != accepted[camera]:
            skipped.append({"camera_id": camera, "reason": "reviewed_frame_digest_unbound"})
            continue
        if sealed[1] == "repair":
            if camera not in repair_cameras_available:
                skipped.append({"camera_id": camera, "reason": "repair_result_missing"})
                continue
            repair_cameras.append(camera)
        else:
            if camera not in result_cameras:
                skipped.append({"camera_id": camera, "reason": "base_result_missing"})
                continue
            base_cameras.append(camera)
    if not base_cameras and not repair_cameras:
        return {"retained": [], "skipped": skipped, "review_execution_digest": review["execution_digest"]}
    sources = []
    if base_cameras:
        sources.append({"request_path": runtime / REQUEST, "result_path": runtime / RESULT,
                        "output_root": runtime / "semantic_teacher_output", "camera_ids": sorted(base_cameras)})
    if repair_cameras:
        sources.append({"request_path": runtime / REPAIR_REQUEST, "result_path": runtime / REPAIR_RESULT,
                        "output_root": runtime / "semantic_target_recovery/selective_semantic_repair_output",
                        "camera_ids": sorted(repair_cameras)})
    return {"retained": sorted(base_cameras + repair_cameras), "skipped": skipped, "sources": sources,
            "review_execution_digest": review["execution_digest"],
            "chain": "repair_merge" if merged is not None else "locality_seal"}


def discover_retained_candidates(*, runtime_request_path: Path, render: Mapping[str, Any],
                                 workspace_root: Path, output_root: Path) -> dict[str, Any]:
    """Build and validate a retained selection from the newest matching workspace."""
    request = json.loads(Path(runtime_request_path).read_text(encoding="utf-8"))
    current = _request_frames(request)
    backend = _backend(request)
    receipt: dict[str, Any] = {"schema_version": SCHEMA, "status": "no_matching_workspace",
                               "workspace_root": str(workspace_root), "examined": [], "candidates_retained": 0}
    root = Path(workspace_root)
    if root.is_symlink() or not root.is_dir() or not current:
        receipt["status"] = "discovery_root_unavailable" if current else "no_current_frames"
        return {**receipt, "candidates": []}
    workspaces = [p for p in root.iterdir() if p.is_dir() and not p.is_symlink() and not p.name.startswith(".")]
    workspaces.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for workspace in workspaces[:MAX_WORKSPACES]:
        runtime = workspace / RUNTIME
        try:
            plan = _workspace_plan(runtime, current, backend)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            receipt["examined"].append({"workspace": workspace.name, "status": f"unreadable:{type(exc).__name__}"})
            continue
        if plan is None:
            receipt["examined"].append({"workspace": workspace.name, "status": "not_applicable"})
            continue
        if not plan["retained"]:
            receipt["examined"].append({"workspace": workspace.name, "status": "no_accepted_frames", "skipped": plan["skipped"]})
            continue
        selection_root = Path(output_root) / workspace.name[:16]
        selection_path = materialize_retained_selection_from_sources(sources=plan["sources"], output_root=selection_root)
        candidates = load_retained_selection(selection_path=selection_path, render=render)
        receipt.update({
            "status": "retained_from_previous_attempt",
            "source_workspace": str(workspace),
            "source_review_execution_digest": plan["review_execution_digest"],
            "digest_chain": plan["chain"],
            "retained_camera_ids": plan["retained"],
            "skipped": plan["skipped"],
            "selection_path": str(selection_path),
            "selection_sha256": "sha256:" + hashlib.sha256(selection_path.read_bytes()).hexdigest(),
            "candidates_retained": len(candidates),
            "provider_call_performed": False,
        })
        receipt["examined"].append({"workspace": workspace.name, "status": "selected"})
        break
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    Path(output_root).mkdir(parents=True, exist_ok=True)
    (Path(output_root) / "discovery.json").write_text(json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8")
    return {**receipt, "candidates": candidates if receipt["status"] == "retained_from_previous_attempt" else []}


def discovery_enabled(values: Mapping[str, Any] | None = None) -> bool:
    source = values if values is not None else os.environ
    return str(source.get(DISCOVERY_ENV, "1")).strip().lower() not in {"0", "false", "no", "off"}
