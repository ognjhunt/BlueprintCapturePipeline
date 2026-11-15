"""Read the exact completed controls source without importing its advancer.

The terminal bytes, authenticated WebApp binding, and post-teardown provider-zero
receipt must satisfy the unchanged worker contract before a source is returned.
"""
from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest
from .task_evaluation_launch_evidence_contracts import (
    LAUNCH_RECEIPT_DIGEST_CANONICALIZATION,
    validated_succeeded_webapp_sync_row,
)
from .validation_file_digests import sha256_file


class TaskEvaluationConfiguredControlsProgressionWorkerError(RuntimeError):
    """The automatic progression worker refused an unsafe transition."""


def _load(path: Path, *, blocker: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TaskEvaluationConfiguredControlsProgressionWorkerError(blocker) from exc
    if path.is_symlink() or not isinstance(value, Mapping):
        raise TaskEvaluationConfiguredControlsProgressionWorkerError(blocker)
    return dict(value)


def _sha256(path: Path) -> str:
    return sha256_file(path)


def validate_source(run_root: Path, *, load_json: Callable[..., dict[str, Any]] = _load,
                    sha256: Callable[[Path], str] = _sha256,
                    is_file: Callable[[Path], bool] | None = None) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    receipt = load_json(
        run_root / "launch_receipt.json", blocker="configured_controls_worker_launch_receipt_invalid"
    )
    expected = (
        cross_runtime_canonical_digest(receipt, digest_field="receipt_digest")
        if receipt.get("receipt_digest_canonicalization")
        == LAUNCH_RECEIPT_DIGEST_CANONICALIZATION
        else canonical_digest(receipt, digest_field="receipt_digest")
    )
    terminal = receipt.get("terminal_evidence")
    result_artifact = terminal.get("result") if isinstance(terminal, Mapping) else None
    if (
        receipt.get("schema_version") != "task_evaluation_launch_receipt.v1"
        or receipt.get("status") != "completed"
        or receipt.get("receipt_digest") != expected
        or not isinstance(terminal, Mapping)
        or terminal.get("status") != "passed"
        or not isinstance(terminal.get("scene_configuration"), Mapping)
        or not isinstance(result_artifact, Mapping)
        or result_artifact.get("exists") is not True
    ):
        raise TaskEvaluationConfiguredControlsProgressionWorkerError(
            "configured_controls_worker_qualifying_terminal_missing"
        )
    result_path = Path(str(result_artifact.get("path") or "")).expanduser()
    present = (is_file(result_path) if is_file is not None
               else not result_path.is_symlink() and result_path.is_file())
    if not present or sha256(result_path) != result_artifact.get("digest"):
        raise TaskEvaluationConfiguredControlsProgressionWorkerError(
            "configured_controls_worker_terminal_artifact_invalid"
        )
    sync = load_json(
        run_root / "webapp_sync_succeeded.json",
        blocker="configured_controls_worker_webapp_sync_missing",
    )
    try:
        validated_succeeded_webapp_sync_row(receipt=receipt, attempt=sync)
    except Exception as exc:
        raise TaskEvaluationConfiguredControlsProgressionWorkerError(
            "configured_controls_worker_webapp_sync_invalid"
        ) from exc
    zero = load_json(
        run_root / "post_teardown_provider_zero_receipt.json",
        blocker="configured_controls_worker_post_teardown_provider_zero_missing",
    )
    if (
        zero.get("schema_version") != "task_evaluation_post_teardown_provider_zero.v1"
        or zero.get("status") != "provider_zero_confirmed"
        or zero.get("provider_zero_verified") is not True
        or zero.get("continuing_spend_from_this_run") is not False
        or zero.get("allocator_invoked") is not False
        or zero.get("provider_mutation_performed") is not False
        or zero.get("automatic_retry_performed") is not False
        or zero.get("blockers") != []
        or zero.get("provider_zero_receipt_digest")
        != canonical_digest(zero, digest_field="provider_zero_receipt_digest")
        or any(
            zero.get(field) != receipt.get(field)
            for field in ("launch_id", "run_id", "request_digest", "receipt_digest", "launch_profile_digest")
        )
    ):
        raise TaskEvaluationConfiguredControlsProgressionWorkerError(
            "configured_controls_worker_post_teardown_provider_zero_invalid"
        )
    return load_json(result_path, blocker="configured_controls_worker_terminal_result_invalid"), receipt, zero


def _write_immutable(path: Path, value: Mapping[str, Any]) -> None:
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
    try:
        with path.open("xb") as stream:
            stream.write(payload)
        path.chmod(0o440)
    except FileExistsError:
        if path.is_symlink() or path.read_bytes() != payload:
            raise TaskEvaluationConfiguredControlsProgressionWorkerError(
                "configured_controls_worker_immutable_conflict"
            )


def _validate_source(run_root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    return validate_source(run_root, load_json=_load, sha256=_sha256)


def validate_retained_source(run_root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """``validate_source`` of an offloaded run, read from the bytes its pointer kept; raises like it.

    Stage 8: a re-plan then needs no restore. A pointer without the records raises the same refusals.
    """
    import hashlib

    # The read-only leaf: the pointer's own checked bytes, the launch receipt among them; nothing here can
    # reach a worker (``tests/test_controls_read_evidence.py``).
    from .scene_run_cold_pointer import retained_member

    def raw(path: Path) -> bytes | None:
        try:
            relative = path.relative_to(run_root).as_posix()
        except ValueError:
            return None
        return retained_member(run_root, relative)

    def load(path: Path, *, blocker: str) -> dict[str, Any]:
        data = raw(path)
        try:
            value = json.loads(data) if data is not None else None
        except ValueError:
            value = None
        if not isinstance(value, Mapping):
            raise TaskEvaluationConfiguredControlsProgressionWorkerError(blocker)
        return dict(value)

    def digest(path: Path) -> str:
        data = raw(path)
        return "sha256:" + hashlib.sha256(data).hexdigest() if data is not None else ""

    return validate_source(run_root, load_json=load, sha256=digest, is_file=lambda path: raw(path) is not None)
