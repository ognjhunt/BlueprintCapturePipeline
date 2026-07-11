"""Immutable selection generations and provider-attempt input lineage."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .common import utc_now_iso
from .kitchen_random_task_selection import select_random_task
from .g1_kitchen_run_index import append_run_index_event
from .g1_kitchen_worker_image_evidence import validate_worker_image_runtime_evidence


SUPERSESSION_SCHEMA_VERSION = "kitchen_task_selection_supersession.v1"
ATTEMPT_INPUT_SCHEMA_VERSION = "g1_kitchen_attempt_input_manifest.v1"
ACTIVE_POINTER_SCHEMA_VERSION = "kitchen_task_active_selection_pointer.v1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON object required: {path}")
    return dict(value)


def _exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _atomic_json_replace(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def create_selection_generation(
    *,
    generations_dir: str | Path,
    registry_path: str | Path,
    preflight_manifest_path: str | Path,
    kitchen_usd: str | Path,
    seed: int,
    invalidation_paths=(),
    selection_id: str | None = None,
) -> dict[str, Any]:
    """Create one immutable selection directory; never write task files at root."""
    root = Path(generations_dir).expanduser().resolve()
    identifier = selection_id or f"selection-{int(seed):016x}-{secrets.token_hex(6)}"
    if not identifier or Path(identifier).name != identifier or identifier in {".", ".."}:
        raise ValueError("selection_id must be one safe path component")
    generation = root / identifier
    generation.mkdir(parents=True, exist_ok=False)
    try:
        payload = select_random_task(
            registry_path=registry_path,
            preflight_manifest_path=preflight_manifest_path,
            kitchen_usd=kitchen_usd,
            out_dir=generation,
            seed=seed,
            invalidation_paths=invalidation_paths,
        )
    except Exception:
        # The directory is new and contains only this failed generation. Preserve
        # any written bytes for forensics and mark it ineligible instead of reuse.
        _exclusive_json(
            generation / "generation_failed.json",
            {"status": "ineligible", "generated_at": utc_now_iso()},
        )
        raise
    selection_path = generation / "random_task_selection.json"
    return {
        "status": "created",
        "selection_id": identifier,
        "generation_dir": str(generation),
        "selection_path": str(selection_path),
        "selection_sha256": sha256_file(selection_path),
        "selected_task_id": payload.get("selected_task_id"),
    }


def activate_selection_generation(
    *,
    run_dir: str | Path,
    generation: Mapping[str, Any],
    active_from_attempt_id: str,
    prior_pointer_path: str | Path | None = None,
    invalidation_path: str | Path | None = None,
) -> dict[str, Any]:
    """Append supersession evidence and update a checksum-bound human pointer."""
    run_root = Path(run_dir).expanduser().resolve()
    selection_path = Path(str(generation.get("selection_path") or "")).resolve()
    observed_sha = sha256_file(selection_path)
    if observed_sha != str(generation.get("selection_sha256") or ""):
        raise ValueError("selection generation checksum mismatch")
    prior = _load(prior_pointer_path) if prior_pointer_path else {}
    prior_sha = str(prior.get("selection_sha256") or "") or None
    invalidation_sha = sha256_file(invalidation_path) if invalidation_path else None
    if bool(prior_sha) != bool(invalidation_sha):
        raise ValueError("replacement requires both prior selection and invalidation")
    event = {
        "schema_version": SUPERSESSION_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "prior_selection_sha256": prior_sha,
        "invalidation_sha256": invalidation_sha,
        "invalidation_reason": (
            str(_load(invalidation_path).get("reason") or "") if invalidation_path else None
        ),
        "replacement_selection_sha256": observed_sha,
        "replacement_selection_id": generation.get("selection_id"),
        "active_from_attempt_id": str(active_from_attempt_id),
    }
    log = run_root / "selection_supersession.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    pointer = {
        "schema_version": ACTIVE_POINTER_SCHEMA_VERSION,
        "selection_id": generation.get("selection_id"),
        "selection_path": os.path.relpath(selection_path, run_root),
        "selection_sha256": observed_sha,
        "active_from_attempt_id": str(active_from_attempt_id),
        "supersession_log_path": os.path.relpath(log, run_root),
        "supersession_log_sha256": sha256_file(log),
        "evidence_policy": "pointer_requires_attempt_input_manifest",
    }
    pointer_path = run_root / "active_selection_pointer.json"
    _atomic_json_replace(pointer_path, pointer)
    append_run_index_event(
        run_root=run_root,
        event_type="selection_superseded",
        run_id=run_root.name,
        attempt_id=str(active_from_attempt_id),
        artifact_paths=[selection_path, log, pointer_path],
        detail=event,
    )
    return {**pointer, "pointer_path": str(pointer_path), "event": event}


def allocate_attempt_id(*, run_dir: str | Path, run_id: str) -> dict[str, Any]:
    """Atomically allocate the next attempt directory across concurrent callers."""
    root = Path(run_dir).expanduser().resolve() / "attempts"
    root.mkdir(parents=True, exist_ok=True)
    for number in range(1, 1_000_000):
        attempt_id = f"{run_id}-attempt-{number:06d}"
        attempt_dir = root / attempt_id
        try:
            attempt_dir.mkdir()
        except FileExistsError:
            continue
        _exclusive_json(
            attempt_dir / "attempt_allocation.json",
            {
                "schema_version": "g1_kitchen_attempt_allocation.v1",
                "allocated_at": utc_now_iso(),
                "run_id": run_id,
                "attempt_id": attempt_id,
            },
        )
        append_run_index_event(
            run_root=Path(run_dir),
            event_type="attempt_allocated",
            run_id=run_id,
            attempt_id=attempt_id,
            artifact_paths=[attempt_dir / "attempt_allocation.json"],
        )
        return {"attempt_id": attempt_id, "attempt_dir": str(attempt_dir)}
    raise RuntimeError("attempt id space exhausted")


def build_attempt_input_manifest(
    *,
    run_id: str,
    attempt_id: str,
    launch_nonce: str,
    provider: str,
    artifacts: Mapping[str, str | Path],
    image_digest: str,
    source_commit: str,
    source_dirty_patch_sha256: str,
) -> dict[str, Any]:
    """Bind every spend-bearing attempt to exact immutable input bytes."""
    required = {
        "selection",
        "scenario",
        "route",
        "task_success_contract",
        "kitchen_inventory",
        "bundle",
        "worker_image_runtime_evidence",
    }
    missing = sorted(required - set(artifacts))
    if missing:
        raise ValueError(f"attempt input artifacts missing:{','.join(missing)}")
    refs: dict[str, Any] = {}
    for name, raw_path in artifacts.items():
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        refs[name] = {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
    selection = _load(artifacts["selection"])
    task_contract = _load(artifacts["task_success_contract"])
    scenario = _load(artifacts["scenario"])
    route = _load(artifacts["route"])
    selected_task_id = str(selection.get("selected_task_id") or "")
    if str(task_contract.get("task_id") or "") != selected_task_id:
        raise ValueError("task success contract task does not match active selection")
    if str(task_contract.get("source_selection_sha256") or "") != refs["selection"]["sha256"]:
        raise ValueError("task success contract selection checksum mismatch")
    if str(scenario.get("source_selection_sha256") or "") != refs["selection"]["sha256"]:
        raise ValueError("scenario selection checksum mismatch")
    if str(route.get("source_selection_sha256") or "") != refs["selection"]["sha256"]:
        raise ValueError("route selection checksum mismatch")
    if len(source_dirty_patch_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in source_dirty_patch_sha256.lower()
    ):
        raise ValueError("source dirty patch sha256 required")
    image_hash = str(image_digest).lower().removeprefix("sha256:")
    if len(image_hash) != 64 or any(char not in "0123456789abcdef" for char in image_hash):
        raise ValueError("immutable image digest required")
    worker_evidence = _load(artifacts["worker_image_runtime_evidence"])
    worker_validation = validate_worker_image_runtime_evidence(
        worker_evidence,
        expected_image_digest="sha256:" + image_hash,
        expected_source_commit=str(source_commit),
        expected_dirty_patch_sha256=source_dirty_patch_sha256,
    )
    if worker_validation["status"] != "passed":
        raise ValueError(
            "worker image runtime evidence invalid:"
            + ",".join(worker_validation["blockers"])
        )
    for field, value in {
        "run_id": run_id,
        "attempt_id": attempt_id,
        "launch_nonce": launch_nonce,
        "provider": provider,
        "source_commit": source_commit,
    }.items():
        if not str(value or "").strip():
            raise ValueError(f"attempt input identity missing:{field}")
    return {
        "schema_version": ATTEMPT_INPUT_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "run_id": str(run_id),
        "attempt_id": str(attempt_id),
        "launch_nonce": str(launch_nonce),
        "provider": str(provider),
        "selected_task_id": selected_task_id,
        "source_commit": str(source_commit),
        "source_dirty_patch_sha256": source_dirty_patch_sha256,
        "image_digest": str(image_digest),
        "artifacts": refs,
        "compatibility": {
            "selection_schema": selection.get("schema_version"),
            "attempt_input_schema": ATTEMPT_INPUT_SCHEMA_VERSION,
            "closure_schema": "g1_kitchen_attempt_closure.v1",
            "controller_fk_schema": "gear_sonic_controller_fk_execution.v1",
            "completion_schema": "oscar_task_completion_evaluator_request.v1",
            "strict_scorer_schema": "strict_action_aware_consistency_contract.v1",
            "worker_image_runtime_evidence_schema": (
                "g1_kitchen_worker_image_runtime_evidence.v1"
            ),
        },
    }


def write_attempt_input_manifest(*, attempt_dir: str | Path, manifest: Mapping[str, Any]) -> Path:
    target = Path(attempt_dir) / "attempt_input_manifest.json"
    _exclusive_json(target, manifest)
    return target
