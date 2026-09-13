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
import tempfile
import shutil
import zipfile
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
#: A successful edit stage archives its workspace into the launch's
#: ``api_pretraining_capsule.zip`` and removes the expanded copy, so a retry
#: after a later (GPU-stage) failure finds its accepted edits only there.
CAPSULE_ROOT_ENV = "BLUEPRINT_SCENE_CONFIGURATION_RETAINED_CANDIDATE_CAPSULE_ROOT"
DEFAULT_CAPSULE_ROOT = "/var/lib/blueprint/pipeline-control-plane/task-evaluation-launch-runs"
CAPSULE_RELATIVE = "allocator/scene-configuration-job/api_pretraining_capsule.zip"
CAPSULE_RECEIPT_RELATIVE = "allocator/scene-configuration-job/api_pretraining_receipt.json"
CAPSULE_MEMBER_PREFIX = "output/released_artifixer_runtime/"
CAPSULE_MEMBER_DIRS = (
    "semantic_teacher_packet/", "semantic_teacher_output/", "semantic_teacher_exact_support_locality_seal/",
    "semantic_target_review/", "semantic_target_recovery/",
)
MAX_CAPSULES = 16
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


def _backend(request: Mapping[str, Any]) -> tuple[str, str, str]:
    backend = request.get("backend") or {}
    return (
        str((backend.get("execution") or {}).get("model_snapshot") or ""),
        str((backend.get("registry_entry") or {}).get("backend_id") or ""),
        str((backend.get("execution") or {}).get("mask_encoding") or ""),
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


def _workspace_plan(runtime: Path, current: Mapping[str, Mapping[str, Any]], backend: tuple[str, str, str], prompt: str | None = None) -> dict[str, Any] | None:
    """Which cameras this workspace can supply, with the digest chain that justifies each."""
    request = _sealed(runtime / REQUEST, "request_digest")
    result = _sealed(runtime / RESULT, "result_digest")
    if request is None or result is None or _backend(request) != backend or request.get("prompt") != prompt:
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
    sealed_by_camera: dict[str, tuple[str, str, str]] = {}
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
    repair_request = _sealed(runtime / REPAIR_REQUEST, "request_digest") if merged is not None else None
    repair_requested = _request_frames(repair_request) if repair_request else {}
    for camera in current:
        if camera not in accepted:
            skipped.append({"camera_id": camera, "reason": "not_accepted_by_last_review"})
            continue
        sealed = sealed_by_camera.get(camera)
        if sealed is None or sealed[0] != accepted[camera]:
            skipped.append({"camera_id": camera, "reason": "reviewed_frame_digest_unbound"})
            continue
        source_request = repair_requested if sealed[1] == "repair" else previous
        if (source_request.get(camera, {}).get("edit_mask", {}).get("sha256")
                != current[camera].get("edit_mask", {}).get("sha256")):
            skipped.append({"camera_id": camera, "reason": "edit_mask_changed"})
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _capsule_runtime(launch_root: Path, extract_root: Path) -> Path | None:
    """Extract the reviewed edit-stage files of a sealed capsule; None when the capsule is not trustworthy."""
    receipt_path = launch_root / CAPSULE_RECEIPT_RELATIVE
    capsule_path = launch_root / CAPSULE_RELATIVE
    receipt = _sealed(receipt_path, "receipt_digest")
    if receipt is None or capsule_path.is_symlink() or not capsule_path.is_file():
        return None
    if receipt.get("capsule_sha256") != _sha256_file(capsule_path) or receipt.get("capsule_bytes") != capsule_path.stat().st_size:
        return None
    extract_root.mkdir(parents=True, exist_ok=True)
    target = Path(tempfile.mkdtemp(prefix=receipt["capsule_sha256"][7:] + "-", dir=extract_root))
    with zipfile.ZipFile(capsule_path) as zipped:
        for member in zipped.infolist():
            name = member.filename
            if member.is_dir() or not name.startswith(CAPSULE_MEMBER_PREFIX):
                continue
            relative = name[len(CAPSULE_MEMBER_PREFIX):]
            if not relative.startswith(CAPSULE_MEMBER_DIRS) or ".." in relative.split("/") or relative.startswith("/"):
                continue
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with zipped.open(member) as source, destination.open("wb") as sink:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    sink.write(chunk)
    return target


def _capsule_launches(capsule_root: Path) -> list[Path]:
    root = Path(capsule_root)
    if root.is_symlink() or not root.is_dir():
        return []
    launches = [p for p in root.iterdir()
                if p.is_dir() and not p.is_symlink() and (p / CAPSULE_RELATIVE).is_file()]
    launches.sort(key=lambda p: (p / CAPSULE_RELATIVE).stat().st_mtime, reverse=True)
    return launches[:MAX_CAPSULES]


def discover_retained_candidates(*, runtime_request_path: Path, render: Mapping[str, Any],
                                 workspace_root: Path, output_root: Path,
                                 capsule_root: Path | None = None) -> dict[str, Any]:
    """Build and validate a retained selection from the newest matching workspace or capsule."""
    request = json.loads(Path(runtime_request_path).read_text(encoding="utf-8"))
    current = _request_frames(request)
    backend = _backend(request)
    receipt: dict[str, Any] = {"schema_version": SCHEMA, "status": "no_matching_workspace",
                               "workspace_root": str(workspace_root),
                               "capsule_root": str(capsule_root) if capsule_root else None,
                               "examined": [], "candidates_retained": 0}
    root = Path(workspace_root)
    if not current:
        receipt["status"] = "no_current_frames"
        return {**receipt, "candidates": []}
    entries: list[tuple[float, str, Path]] = []
    if not root.is_symlink() and root.is_dir():
        for path in root.iterdir():
            if path.is_dir() and not path.is_symlink() and not path.name.startswith("."):
                entries.append((path.stat().st_mtime, "workspace", path))
    else:
        receipt["examined"].append({"workspace_root": str(root), "status": "discovery_root_unavailable"})
    if capsule_root is not None:
        for launch in _capsule_launches(Path(capsule_root)):
            entries.append(((launch / CAPSULE_RELATIVE).stat().st_mtime, "capsule", launch))
    if not entries:
        receipt["status"] = "discovery_root_unavailable"
        return {**receipt, "candidates": []}
    entries.sort(key=lambda row: row[0], reverse=True)
    candidates: list = []
    for _mtime, kind, workspace in entries[:MAX_WORKSPACES]:
        try:
            if kind == "capsule":
                runtime = _capsule_runtime(workspace, Path(output_root) / "capsules")
                if runtime is None:
                    receipt["examined"].append({"capsule": workspace.name, "status": "capsule_not_trustworthy"})
                    continue
            else:
                runtime = workspace / RUNTIME
            plan = _workspace_plan(runtime, current, backend, request.get("prompt"))
        except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
            receipt["examined"].append({kind: workspace.name, "status": f"unreadable:{type(exc).__name__}"})
            continue
        if plan is None:
            receipt["examined"].append({kind: workspace.name, "status": "not_applicable"})
            continue
        if not plan["retained"]:
            receipt["examined"].append({kind: workspace.name, "status": "no_accepted_frames", "skipped": plan["skipped"]})
            continue
        Path(output_root).mkdir(parents=True, exist_ok=True)
        selection_parent = Path(tempfile.mkdtemp(prefix="selection-", dir=output_root))
        try:
            selection_path = materialize_retained_selection_from_sources(
                sources=plan["sources"], output_root=selection_parent / "candidate")
            candidates = load_retained_selection(selection_path=selection_path, render=render)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            shutil.rmtree(selection_parent)
            receipt["examined"].append({kind: workspace.name, "status": f"candidate_ineligible:{type(exc).__name__}"})
            continue
        receipt.update({
            "status": "retained_from_previous_attempt",
            "source_workspace": str(workspace),
            "source_kind": kind,
            "source_review_execution_digest": plan["review_execution_digest"],
            "digest_chain": plan["chain"],
            "retained_camera_ids": plan["retained"],
            "skipped": plan["skipped"],
            "selection_path": str(selection_path),
            "selection_sha256": "sha256:" + hashlib.sha256(selection_path.read_bytes()).hexdigest(),
            "candidates_retained": len(candidates),
            "provider_call_performed": False,
        })
        receipt["examined"].append({kind: workspace.name, "status": "selected"})
        break
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    Path(output_root).mkdir(parents=True, exist_ok=True)
    (Path(output_root) / "discovery.json").write_text(json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8")
    return {**receipt, "candidates": candidates if receipt["status"] == "retained_from_previous_attempt" else []}


def discovery_enabled(values: Mapping[str, Any] | None = None) -> bool:
    source = values if values is not None else os.environ
    return str(source.get(DISCOVERY_ENV, "1")).strip().lower() not in {"0", "false", "no", "off"}
