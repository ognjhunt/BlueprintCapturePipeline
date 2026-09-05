"""Append-only SAM precursor progress and digest-bound no-spend resumption."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from .decision_evidence_contracts import canonical_digest
from .task_evaluation_launch_preparation_queue import write_launch_preparation_record_exclusive

WAITING_STATE = "awaiting_source_preparation"
PROGRESS_SCHEMA = "task_evaluation_sam31_preparation_progress.v1"
RESUME_SCHEMA = "task_evaluation_sam31_preparation_resume.v1"
_WAIT_STATUSES = {"waiting_for_child", "awaiting_human_review"}
_SAFE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


class Sam31PreparationQueueError(ValueError):
    """A precursor checkpoint or resume signal failed its immutable bindings."""


class Sam31PreparationWait(Exception):
    def __init__(self, progress: dict):
        self.progress = progress


def _require(value: bool, code: str) -> None:
    if not value:
        raise Sam31PreparationQueueError("sam31_preparation_" + code)


def _read(path: Path) -> dict:
    _require(not any(p.is_symlink() for p in (path, *path.parents))
             and path.is_file() and path.stat().st_size <= 4 * 1024 * 1024,
             "record_path_invalid")
    value = json.loads(path.read_text())
    _require(isinstance(value, dict), "record_invalid")
    return value


def _filename(preparation_id: str, digest: str) -> str:
    _require(isinstance(preparation_id, str) and _SAFE.fullmatch(preparation_id) is not None
             and isinstance(digest, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is not None,
             "identity_invalid")
    return f"{preparation_id}-{digest.removeprefix('sha256:')}.json"


def ensure_progress_roots(root: Path) -> None:
    for name in (WAITING_STATE, "source-progress", "source-resume-pending",
                 "source-resume-completed", "source-resume-blocked"):
        path = root / name
        _require(not path.is_symlink(), "queue_path_invalid")
        path.mkdir(parents=True, exist_ok=True)


def verify_evidence_reference(row: Mapping[str, Any], roots: Sequence[Path]) -> Path:
    _require(isinstance(row, Mapping), "evidence_invalid")
    raw = row.get("path")
    _require(isinstance(raw, str) and Path(raw).is_absolute() and ".." not in Path(raw).parts,
             "evidence_path_invalid")
    path = Path(raw)
    _require(not any(p.is_symlink() for p in (path, *path.parents))
             and path.is_file()
             and any(path.resolve().is_relative_to(root.resolve()) for root in roots),
             "evidence_path_invalid")
    digest = hashlib.sha256()
    count = 0
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
            count += len(chunk)
    size = row.get("size_bytes")
    _require(type(size) is int and size > 0 and count == size
             and "sha256:" + digest.hexdigest() == row.get("sha256", row.get("digest")),
             "evidence_readback_mismatch")
    return path


def load_progress(root: Path, filename: str, request_digest: str) -> dict | None:
    directory = root / "source-progress" / Path(filename).stem
    if not directory.exists():
        return None
    _require(not directory.is_symlink(), "progress_path_invalid")
    prior = None
    for index, path in enumerate(sorted(directory.glob("*.json")), 1):
        value = _read(path)
        _require(value.get("schema_version") == PROGRESS_SCHEMA
                 and value.get("request_digest") == request_digest
                 and value.get("sequence") == index
                 and value.get("previous_progress_digest") == (prior["progress_digest"] if prior else None)
                 and value.get("progress_digest") == canonical_digest(value, digest_field="progress_digest"),
                 "progress_chain_invalid")
        prior = value
    return prior


def _progress(root: Path, envelope: Mapping[str, Any], advancement: dict) -> dict:
    request = envelope["request"]
    filename = _filename(request["preparation_id"], envelope["request_digest"])
    prior = load_progress(root, filename, envelope["request_digest"])
    resumed_from = envelope.get("resume_signal_digest")
    if (prior is not None and prior.get("advancement") == advancement
            and prior.get("resume_signal_digest") == resumed_from):
        return prior
    value = {
        "schema_version": PROGRESS_SCHEMA, "preparation_id": request["preparation_id"],
        "request_digest": envelope["request_digest"], "run_id": request["run_id"],
        "source_commit": request["expected_production_commit"], "status": advancement["status"],
        "sequence": prior["sequence"] + 1 if prior else 1,
        "previous_progress_digest": prior["progress_digest"] if prior else None,
        "advancement": advancement, "resume_signal_digest": resumed_from,
        "provider_mutation_performed": False,
        "paid_execution_requested": False,
    }
    value["progress_digest"] = canonical_digest(value, digest_field="progress_digest")
    directory = root / "source-progress" / Path(filename).stem
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{value['sequence']:06d}-{value['progress_digest'].removeprefix('sha256:')}.json"
    try:
        write_launch_preparation_record_exclusive(path, value)
    except FileExistsError:
        _require(_read(path) == value, "progress_conflict")
    return value


def _validate_signal(root: Path, signal: dict, roots: Sequence[Path]) -> tuple[Path, dict]:
    _require(signal.get("schema_version") == RESUME_SCHEMA
             and signal.get("signal_digest") == canonical_digest(signal, digest_field="signal_digest"),
             "resume_signal_invalid")
    filename = _filename(signal.get("preparation_id"), signal.get("request_digest"))
    waiting = root / WAITING_STATE / filename
    envelope = _read(waiting)
    _require(envelope.get("request_digest") == signal["request_digest"]
             and envelope.get("envelope_digest") == canonical_digest(envelope, digest_field="envelope_digest"),
             "resume_parent_invalid")
    prior = load_progress(root, filename, signal["request_digest"])
    _require(prior is not None and prior["progress_digest"] == signal.get("progress_digest")
             and prior.get("source_commit") == signal.get("source_commit"), "resume_progress_mismatch")
    expected = "human_review" if prior["status"] == "awaiting_human_review" else "child_result"
    _require(prior["status"] in _WAIT_STATUSES and signal.get("kind") == expected,
             "resume_kind_invalid")
    evidence_path = verify_evidence_reference(signal.get("evidence_ref"), roots)
    if expected == "human_review":
        review = _read(evidence_path)
        candidate_ref = prior["advancement"].get("review_candidate_ref")
        candidate_path = verify_evidence_reference(candidate_ref, roots)
        candidate = _read(candidate_path)
        _require(review.get("schema_version") == "public_scene_sam31_track_selection_review.v1"
                 and review.get("status") == "selected_tracks_human_review_accepted"
                 and review.get("all_selected_tracks_accepted") is True
                 and review.get("agent_selected_tracks_without_human_review") is False
                 and review.get("receipt_digest") == canonical_digest(review, digest_field="receipt_digest")
                 and review.get("candidate", {}).get("candidate_digest") == candidate.get("candidate_digest")
                 and candidate.get("candidate_digest") == canonical_digest(candidate, digest_field="candidate_digest")
                 and review.get("candidate", {}).get("sha256") ==
                     candidate_ref.get("sha256", candidate_ref.get("digest"))
                 and review.get("candidate", {}).get("size_bytes") == candidate_ref.get("size_bytes"),
                 "human_review_resume_invalid")
    return waiting, signal


def stage_resume_signal(
    *, queue_root: str | Path, preparation_id: str, request_digest: str,
    progress_digest: str, source_commit: str, kind: str, evidence_ref: Mapping[str, Any],
    approved_roots: Sequence[Path],
) -> dict:
    """Called only by an authenticated human adapter or the server-owned child watcher.

    This is a wake-up signal, not human acceptance or permission to spend.
    The production driver and mask consumer must still run their full validators.
    """
    root = Path(queue_root)
    ensure_progress_roots(root)
    signal = {"schema_version": RESUME_SCHEMA, "preparation_id": preparation_id,
              "request_digest": request_digest, "progress_digest": progress_digest,
              "source_commit": source_commit, "kind": kind, "evidence_ref": dict(evidence_ref)}
    signal["signal_digest"] = canonical_digest(signal, digest_field="signal_digest")
    _validate_signal(root, signal, approved_roots)
    path = root / "source-resume-pending" / f"{signal['signal_digest'].removeprefix('sha256:')}.json"
    try:
        write_launch_preparation_record_exclusive(path, signal)
    except FileExistsError:
        _require(_read(path) == signal, "resume_signal_conflict")
    return signal


def resume_waiting_preparations(
    *, queue_root: Path, approved_roots: Sequence[Path], max_messages: int,
) -> list[dict]:
    ensure_progress_roots(queue_root)
    results = []
    for path in sorted((queue_root / "source-resume-pending").glob("*.json"))[:max_messages]:
        try:
            signal = _read(path)
            waiting, signal = _validate_signal(queue_root, signal, approved_roots)
            pending = queue_root / "pending" / waiting.name
            _require(not pending.exists() and not pending.is_symlink(), "resume_pending_conflict")
            # Retain the verified signal before waking the immutable parent.
            completed_root = queue_root / "source-resume-completed" / waiting.stem
            completed_root.mkdir(parents=True, exist_ok=True)
            destination = completed_root / path.name
            try:
                write_launch_preparation_record_exclusive(destination, signal)
            except FileExistsError:
                _require(_read(destination) == signal, "resume_signal_conflict")
            os.replace(waiting, pending)
            path.unlink()
            results.append({"status": "resumed", "signal_digest": signal["signal_digest"]})
        except (Sam31PreparationQueueError, OSError, ValueError) as exc:
            failure = {"schema_version": "task_evaluation_sam31_preparation_resume_failure.v1",
                       "status": "rejected", "blocker": str(exc)}
            failure["failure_digest"] = canonical_digest(failure, digest_field="failure_digest")
            target = queue_root / "source-resume-blocked" / path.name
            if not target.exists():
                os.replace(path, target)
                write_launch_preparation_record_exclusive(
                    target.with_suffix(".failure.json"), failure)
            results.append(failure)
    return results


def _resume_context(root: Path, prior: dict | None) -> dict | None:
    if prior is None:
        return None
    matches = []
    filename = _filename(prior["preparation_id"], prior["request_digest"])
    completed_root = root / "source-resume-completed" / Path(filename).stem
    for path in completed_root.glob("*.json"):
        value = _read(path)
        if value.get("progress_digest") == prior["progress_digest"]:
            _require(value.get("signal_digest") == canonical_digest(value, digest_field="signal_digest"),
                     "resume_signal_invalid")
            matches.append(value)
    _require(len(matches) <= 1, "resume_signal_ambiguous")
    return matches[0] if matches else None


def advance_sam31_for_preparation(
    *, queue_root: Path, envelope_context: dict, approved_roots: Sequence[Path], advancer=None,
) -> dict:
    """Validate the production driver's evidence; never execute paid work here."""
    ensure_progress_roots(queue_root)
    request = envelope_context["request"]
    filename = _filename(request["preparation_id"], envelope_context["request_digest"])
    prior = load_progress(queue_root, filename, envelope_context["request_digest"])
    resume = _resume_context(queue_root, prior)
    if advancer is None:
        from .task_evaluation_scene_configuration_sam31_preparation_driver import advance_sam31_preparation
        advancer = advance_sam31_preparation
    if resume is not None:
        verify_evidence_reference(resume["evidence_ref"], approved_roots)
    context = {**envelope_context, "queue_root": str(queue_root),
               "expected_source_commit": request["expected_production_commit"],
               "prior_progress": prior, "validated_resume_receipt": resume}
    advancement = advancer(context)
    _require(isinstance(advancement, dict) and advancement.get("status") in _WAIT_STATUSES | {"ready"},
             "driver_result_invalid")
    refs = advancement.get("evidence_refs")
    _require(isinstance(refs, list) and (bool(refs) or advancement["status"] == "waiting_for_child"),
             "driver_evidence_missing")
    for row in refs:
        verify_evidence_reference(row, approved_roots)
    if advancement["status"] == "awaiting_human_review":
        _require(advancement.get("reviewer_kind") == "human"
                 and envelope_context["stage_one_configuration"].get("sam31_review_kind") == "human",
                 "human_pause_not_explicit")
        verify_evidence_reference(advancement.get("review_candidate_ref"), approved_roots)
    if advancement["status"] == "ready":
        _require(len(refs) >= 5
                 and isinstance(advancement.get("sam31_exact_mask_inputs"), dict)
                 and isinstance(advancement.get("sam31_preparation_result"), dict),
                 "ready_evidence_incomplete")
        if prior is not None and prior["status"] == "awaiting_human_review":
            _require(resume is not None and resume.get("kind") == "human_review",
                     "human_review_resume_required")
    progress = _progress(queue_root, {**envelope_context,
        "resume_signal_digest": resume["signal_digest"] if resume else None}, advancement)
    if advancement["status"] in _WAIT_STATUSES:
        raise Sam31PreparationWait(progress)
    return advancement
