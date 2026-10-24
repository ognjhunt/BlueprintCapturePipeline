"""Retain verified later cleanup without rewriting a failed native attempt."""
from __future__ import annotations
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .native_task_arena_paid_authority_io import read as _read, record as _record


def _recovered_result(original: Mapping[str, Any], recovery: Mapping[str, Any]) -> dict[str, Any]:
    result = {**original, "all_staged_objects_absent": True,
              "post_cleanup_recovery": dict(recovery)}
    if "result_digest" in original:
        if original["result_digest"] != canonical_digest(original, digest_field="result_digest"):
            raise ValueError("native_task_arena_original_result_digest_invalid")
        result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    return result


def _validate_post_cleanup_recovery(result: Mapping[str, Any]) -> None:
    from .native_task_arena_paid_authority import _aware_time, _bound_record

    recovery = result.get("post_cleanup_recovery")
    if recovery is None:
        return
    if not isinstance(recovery, Mapping):
        raise ValueError("native_task_arena_post_cleanup_recovery_invalid")
    original_path, _ = _bound_record(
        recovery.get("original_result"), "native_task_arena_original_result_unbound"
    )
    cleanup_path, cleanup_record = _bound_record(
        recovery.get("cleanup"), "native_task_arena_recovered_cleanup_unbound"
    )
    original = _read(original_path, "native_task_arena_original_result_invalid")
    cleanup = _read(cleanup_path, "native_task_arena_recovered_cleanup_invalid")
    expected = _recovered_result(original, recovery)
    if (dict(result) != expected
            or original.get("schema_version") != "native_task_arena_vast_run.v1"
            or original.get("status") != "blocked"
            or "post_cleanup_recovery" in original
            or original.get("continuing_spend_from_this_run") is not False
            or cleanup_path != Path(str(original.get("object_store_cleanup_path") or "")).resolve()
            or cleanup.get("schema_version") != "wam_provider_object_store_cleanup.v1"
            or cleanup.get("status") != "completed"
            or cleanup.get("all_objects_absent") is not True
            or cleanup.get("signed_url_files_removed") is not True
            or _aware_time(cleanup.get("generated_at"), "cleanup_time_invalid")
            < _aware_time(original.get("generated_at"), "result_time_invalid")
            or cleanup_record != _record(cleanup_path)):
        raise ValueError("native_task_arena_post_cleanup_recovery_invalid")


def materialize_native_task_arena_post_cleanup_closeout(
    *, authority_path: str | Path, result_path: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    """Bind later exact-object cleanup without rewriting the failed attempt.

    All scientific outcomes, blockers and nonzero costs stay byte-equivalent in
    the derived result. Ordinary provider-zero validation still checks teardown,
    watchdog, provider inventory and the consumed single-use authority.
    """
    from .native_task_arena_paid_authority import materialize_native_task_arena_provider_zero

    original_path = Path(result_path).resolve()
    original = _read(original_path, "native_task_arena_original_result_invalid")
    cleanup_path = Path(str(original.get("object_store_cleanup_path") or "")).resolve()
    result = _recovered_result(original, {
                  "original_result": _record(original_path),
                  "cleanup": _record(cleanup_path),
              })
    _validate_post_cleanup_recovery(result)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    result_file = output / "native_task_arena_post_cleanup_result.json"
    write_json(result_file, result)
    return materialize_native_task_arena_provider_zero(
        authority_path=authority_path, result_path=result_file,
        output_path=output / "native_task_arena_provider_zero.json",
    )
