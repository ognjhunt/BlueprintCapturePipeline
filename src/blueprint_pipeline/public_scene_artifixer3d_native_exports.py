"""Seal host-resident ArtiFixer native exports for paired-native consumption."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
from typing import Any

from .common import write_json
from .decision_evidence_contracts import canonical_digest


RAW_RESULT_SCHEMA = "public_scene_artifixer3d_raw_result.v1"
EXPORT_SCHEMA = "public_scene_artifixer3d_native_appearance_export.v1"
EXPORT_STATUS = (
    "native_appearance_candidates_exported_pending_native_import_and_multiview_review"
)


class ArtiFixerNativeExportError(ValueError):
    """Raised when provider-native output cannot be sealed for host consumption."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _read(path: Path, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtiFixerNativeExportError(code) from exc
    if not isinstance(value, dict):
        raise ArtiFixerNativeExportError(code)
    return value


def _digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and value.startswith("sha256:")
        and len(value) == 71
        and all(character in "0123456789abcdef" for character in value[7:])
    )


def _bound_file(record: Any, code: str) -> Path:
    if not isinstance(record, Mapping):
        raise ArtiFixerNativeExportError(code)
    path = Path(str(record.get("path") or "")).expanduser().resolve()
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != record.get("size_bytes")
        or _sha256(path) != record.get("sha256")
    ):
        raise ArtiFixerNativeExportError(code)
    return path


def validate_artifixer3d_native_appearance_export(
    receipt_path: str | Path,
) -> dict[str, Any]:
    """Reopen one host-resident receipt and every exported source byte."""

    path = Path(receipt_path).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise ArtiFixerNativeExportError("artifixer3d_native_export_receipt_missing")
    value = _read(path, "artifixer3d_native_export_receipt_invalid")
    if (
        value.get("schema_version") != EXPORT_SCHEMA
        or value.get("status") != EXPORT_STATUS
        or value.get("export_digest")
        != canonical_digest(value, digest_field="export_digest")
        or not _digest(value.get("source_export_digest"))
        or value.get("host_path_rebased_from_provider_runtime_output") is not True
        or value.get("generated_output_is_capture_or_physical_evidence") is not False
        or value.get("native_import_qualified") is not False
    ):
        raise ArtiFixerNativeExportError("artifixer3d_native_export_receipt_invalid")
    source_raw = value.get("source_raw_result")
    if not isinstance(source_raw, Mapping) or not _digest(source_raw.get("result_digest")):
        raise ArtiFixerNativeExportError("artifixer3d_native_export_source_invalid")
    raw_path = _bound_file(
        source_raw, "artifixer3d_native_export_source_invalid"
    )
    raw = _read(raw_path, "artifixer3d_native_export_source_invalid")
    if (
        raw.get("schema_version") != RAW_RESULT_SCHEMA
        or raw.get("result_digest") != source_raw.get("result_digest")
        or raw.get("result_digest") != canonical_digest(raw, digest_field="result_digest")
    ):
        raise ArtiFixerNativeExportError("artifixer3d_native_export_source_invalid")
    task_id = str(value.get("task_id") or "")
    source_tasks = [
        task
        for task in raw.get("tasks") or []
        if isinstance(task, Mapping) and task.get("task_id") == task_id
    ]
    expected_native = dict(value)
    for field in (
        "task_id",
        "source_raw_result",
        "host_path_rebased_from_provider_runtime_output",
        "export_digest",
    ):
        expected_native.pop(field, None)
    if len(source_tasks) != 1 or source_tasks[0].get("native_appearance") != expected_native:
        raise ArtiFixerNativeExportError("artifixer3d_native_export_source_invalid")
    for field in ("standard_gaussian_ply", "isaac_nurec_usdz"):
        _bound_file(value.get(field), "artifixer3d_native_export_file_invalid")
    return value


def materialize_artifixer3d_native_appearance_exports(
    *, raw_result_path: str | Path, output_root: str | Path
) -> list[dict[str, Any]]:
    """Rebase provider paths into canonical host receipts, one per completed task."""

    raw_path = Path(raw_result_path).expanduser().resolve()
    if raw_path.is_symlink() or not raw_path.is_file():
        raise ArtiFixerNativeExportError("artifixer3d_native_export_raw_result_missing")
    raw = _read(raw_path, "artifixer3d_native_export_raw_result_invalid")
    closeout = raw.get("provider_closeout")
    tasks = raw.get("tasks")
    if (
        raw.get("schema_version") != RAW_RESULT_SCHEMA
        or raw.get("result_digest") != canonical_digest(raw, digest_field="result_digest")
        or raw.get("status")
        != "raw_artifixer3d_review_frames_ready_for_external_visual_and_multiview_review"
        or not isinstance(closeout, Mapping)
        or closeout.get("provider_zero_confirmed") is not True
        or not isinstance(tasks, list)
        or not tasks
    ):
        raise ArtiFixerNativeExportError("artifixer3d_native_export_raw_result_invalid")

    raw_record = {**_record(raw_path), "result_digest": raw["result_digest"]}
    receipts: list[tuple[str, dict[str, Any]]] = []
    seen: set[str] = set()
    for task in tasks:
        if not isinstance(task, Mapping):
            raise ArtiFixerNativeExportError("artifixer3d_native_export_task_invalid")
        task_id = str(task.get("task_id") or "")
        native = task.get("native_appearance")
        if not task_id or task_id in seen or not isinstance(native, Mapping):
            raise ArtiFixerNativeExportError("artifixer3d_native_export_task_invalid")
        seen.add(task_id)
        if (
            native.get("schema_version") != EXPORT_SCHEMA
            or native.get("status") != EXPORT_STATUS
            or not _digest(native.get("source_export_digest"))
            or "export_digest" in native
            or native.get("generated_output_is_capture_or_physical_evidence") is not False
            or native.get("native_import_qualified") is not False
        ):
            raise ArtiFixerNativeExportError("artifixer3d_native_export_task_invalid")
        for field in ("standard_gaussian_ply", "isaac_nurec_usdz"):
            _bound_file(native.get(field), "artifixer3d_native_export_file_invalid")
        receipt = {
            **dict(native),
            "task_id": task_id,
            "source_raw_result": raw_record,
            "host_path_rebased_from_provider_runtime_output": True,
            "export_digest": "",
        }
        receipt["export_digest"] = canonical_digest(
            receipt, digest_field="export_digest"
        )
        receipts.append((task_id, receipt))

    destination = Path(output_root).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise ArtiFixerNativeExportError("artifixer3d_native_export_output_exists")
    destination.mkdir(parents=True)
    outputs: list[dict[str, Any]] = []
    try:
        for index, (task_id, receipt) in enumerate(receipts):
            path = destination / f"task_{index:03d}.native_appearance_export.v1.json"
            write_json(path, receipt)
            reopened = validate_artifixer3d_native_appearance_export(path)
            outputs.append(
                {
                    "task_id": task_id,
                    **_record(path),
                    "export_digest": reopened["export_digest"],
                    "source_export_digest": reopened["source_export_digest"],
                }
            )
    except Exception:
        for path in destination.glob("*"):
            if path.is_file() and not path.is_symlink():
                path.unlink()
        destination.rmdir()
        raise
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-result", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    outputs = materialize_artifixer3d_native_appearance_exports(
        raw_result_path=args.raw_result, output_root=args.output_root
    )
    print(json.dumps({"exports": outputs}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ArtiFixerNativeExportError",
    "materialize_artifixer3d_native_appearance_exports",
    "validate_artifixer3d_native_appearance_export",
]
