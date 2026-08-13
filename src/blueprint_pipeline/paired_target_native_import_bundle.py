"""Freeze a 1-5 object co-present native-import probe bundle.

The paired-target render request already binds the exact repaired appearance,
collision scene, calibrated cameras, and the complete co-present SimReady set.
Native task construction cannot consume those replacements until Isaac has
actually imported each exact USD.  This module packages that smallest prior
gate without allocating a provider or claiming that Isaac executed.

Only the agent-authored SimReady USD bytes are staged.  No source capture,
canonical InteriorGS, repaired NuRec, or learned-policy input is included.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
from typing import Any, Mapping
import zipfile

from .adp009d_native_microcheck_bundle import DEFAULT_IMAGE
from .common import ensure_dir, write_json
from .decision_evidence_contracts import canonical_digest
from .dual_task_rehearsal_contract import MAX_REPLACEMENT_OBJECTS


SCHEMA_VERSION = "paired_target_native_import_provider_bundle.v1"
REQUEST_SCHEMA_VERSION = "paired_target_native_import_request.v1"
SOURCE_SCHEMA_VERSION = "paired_target_native_render_request.v1"
RESULT_SCHEMA_VERSION = "paired_target_native_import_runtime_result.v1"
RESULT_FILENAME = "paired_target_native_import_runtime_result.v1.json"


class PairedTargetNativeImportBundleError(ValueError):
    """Stable fail-closed bundle preparation failure."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read_mapping(path: str | Path, code: str) -> tuple[Path, dict[str, Any]]:
    candidate = Path(path).expanduser().resolve()
    try:
        value = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PairedTargetNativeImportBundleError(code) from exc
    if candidate.is_symlink() or not isinstance(value, dict):
        raise PairedTargetNativeImportBundleError(code)
    return candidate, value


def _verified_source_file(record: Any, code: str) -> Path:
    if not isinstance(record, Mapping):
        raise PairedTargetNativeImportBundleError(code)
    candidate = Path(str(record.get("path") or "")).expanduser().resolve()
    if (
        candidate.is_symlink()
        or not candidate.is_file()
        or candidate.suffix.lower() not in {".usd", ".usda", ".usdc"}
        or candidate.stat().st_size != record.get("size_bytes")
        or _sha256(candidate) != record.get("sha256")
    ):
        raise PairedTargetNativeImportBundleError(code)
    return candidate


def _safe_identifier(value: Any, code: str) -> str:
    text = str(value or "")
    if (
        not text
        or PurePosixPath(text).name != text
        or any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            for character in text
        )
    ):
        raise PairedTargetNativeImportBundleError(code)
    return text


def _verified_registration(record: Any) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise PairedTargetNativeImportBundleError(
            "paired_target_native_import_asset_frame_registration_invalid"
        )
    digest = record.get("registration_digest")
    if (
        not isinstance(digest, str)
        or not digest.startswith("sha256:")
        or len(digest) != 71
        or any(character not in "0123456789abcdef" for character in digest[7:])
    ):
        raise PairedTargetNativeImportBundleError(
            "paired_target_native_import_asset_frame_registration_invalid"
        )
    return dict(record)


def _source_replacement_set(source: Mapping[str, Any]) -> list[dict[str, Any]]:
    tasks = source.get("tasks")
    if (
        source.get("schema_version") != SOURCE_SCHEMA_VERSION
        or source.get("receipt_digest") != canonical_digest(source, digest_field="receipt_digest")
        or source.get("status") != "native_render_requests_materialized_pending_isaac_execution"
        or source.get("native_isaac_executed") is not False
        or source.get("provider_allocation_performed") is not False
        or source.get("generated_output_is_capture_or_physical_evidence") is not False
        or not isinstance(tasks, list)
        or not 1 <= len(tasks) <= MAX_REPLACEMENT_OBJECTS
        or source.get("replacement_object_count") != len(tasks)
    ):
        raise PairedTargetNativeImportBundleError(
            "paired_target_native_import_source_request_invalid"
        )

    canonical_set: list[dict[str, Any]] | None = None
    task_ids: set[str] = set()
    for task in tasks:
        if not isinstance(task, Mapping):
            raise PairedTargetNativeImportBundleError(
                "paired_target_native_import_source_task_invalid"
            )
        task_id = _safe_identifier(
            task.get("task_id"), "paired_target_native_import_source_task_invalid"
        )
        if task_id in task_ids:
            raise PairedTargetNativeImportBundleError(
                "paired_target_native_import_source_task_invalid"
            )
        task_ids.add(task_id)
        replacements = task.get("co_present_replacements")
        if not isinstance(replacements, list) or len(replacements) != len(tasks):
            raise PairedTargetNativeImportBundleError(
                "paired_target_native_import_replacement_set_invalid"
            )
        rows: list[dict[str, Any]] = []
        for row in replacements:
            if not isinstance(row, Mapping):
                raise PairedTargetNativeImportBundleError(
                    "paired_target_native_import_replacement_set_invalid"
                )
            replacement_task_id = _safe_identifier(
                row.get("task_id"), "paired_target_native_import_replacement_set_invalid"
            )
            asset_id = _safe_identifier(
                row.get("asset_id"), "paired_target_native_import_replacement_set_invalid"
            )
            asset = row.get("visual_usd")
            asset_path = _verified_source_file(
                asset, "paired_target_native_import_replacement_asset_invalid"
            )
            registration = _verified_registration(row.get("asset_frame_registration"))
            rows.append(
                {
                    "task_id": replacement_task_id,
                    "asset_id": asset_id,
                    "source": dict(asset),
                    "source_path": asset_path,
                    "asset_frame_registration": registration,
                    "task_subject": row.get("task_subject") is True,
                    "passive_co_present": row.get("passive_co_present") is True,
                }
            )
        if sum(row["task_subject"] for row in rows) != 1:
            raise PairedTargetNativeImportBundleError(
                "paired_target_native_import_subject_count_invalid"
            )
        if next(row for row in rows if row["task_subject"])["task_id"] != task_id:
            raise PairedTargetNativeImportBundleError(
                "paired_target_native_import_subject_binding_invalid"
            )
        if any(row["passive_co_present"] is row["task_subject"] for row in rows):
            raise PairedTargetNativeImportBundleError(
                "paired_target_native_import_passive_binding_invalid"
            )
        identity = [
            (row["task_id"], row["asset_id"], row["source"]["sha256"], row["source"]["size_bytes"])
            for row in rows
        ]
        if canonical_set is None:
            canonical_set = rows
        else:
            prior_identity = [
                (
                    row["task_id"],
                    row["asset_id"],
                    row["source"]["sha256"],
                    row["source"]["size_bytes"],
                )
                for row in canonical_set
            ]
            if identity != prior_identity:
                raise PairedTargetNativeImportBundleError(
                    "paired_target_native_import_replacement_set_mismatch"
                )
    assert canonical_set is not None
    if {row["task_id"] for row in canonical_set} != task_ids:
        raise PairedTargetNativeImportBundleError(
            "paired_target_native_import_replacement_task_set_mismatch"
        )
    if len({row["asset_id"] for row in canonical_set}) != len(canonical_set):
        raise PairedTargetNativeImportBundleError(
            "paired_target_native_import_replacement_asset_ids_duplicate"
        )
    return canonical_set


def _write_zip_member(archive: zipfile.ZipFile, name: str, source: Path) -> None:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    with source.open("rb") as input_stream, archive.open(info, "w") as output:
        shutil.copyfileobj(input_stream, output, length=1024 * 1024)


def build_paired_target_native_import_bundle(
    *,
    native_render_request_path: str | Path,
    runner_path: str | Path,
    output_root: str | Path,
    implementation_commit: str,
) -> dict[str, Any]:
    """Build one immutable co-present import bundle without provider mutation."""

    if len(implementation_commit) != 40 or any(
        character not in "0123456789abcdef" for character in implementation_commit
    ):
        raise PairedTargetNativeImportBundleError("paired_target_native_import_commit_invalid")
    source_path, source = _read_mapping(
        native_render_request_path,
        "paired_target_native_import_source_request_invalid",
    )
    replacements = _source_replacement_set(source)
    runner = Path(runner_path).expanduser().resolve()
    if runner.is_symlink() or not runner.is_file() or runner.suffix != ".py":
        raise PairedTargetNativeImportBundleError("paired_target_native_import_runner_invalid")
    destination = Path(output_root).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise PairedTargetNativeImportBundleError("paired_target_native_import_output_exists")
    ensure_dir(destination)
    runtime = destination / "provider_runtime"
    assets_dir = runtime / "assets"
    ensure_dir(assets_dir)
    try:
        replacement_rows: list[dict[str, Any]] = []
        for index, row in enumerate(replacements):
            suffix = row["source_path"].suffix.lower()
            filename = f"replacement_{index:02d}{suffix}"
            staged = assets_dir / filename
            shutil.copyfile(row["source_path"], staged)
            if (
                staged.stat().st_size != row["source"]["size_bytes"]
                or _sha256(staged) != row["source"]["sha256"]
            ):
                raise PairedTargetNativeImportBundleError(
                    "paired_target_native_import_asset_copy_mismatch"
                )
            replacement_rows.append(
                {
                    "index": index,
                    "task_id": row["task_id"],
                    "asset_id": row["asset_id"],
                    "relative_path": f"assets/{filename}",
                    "size_bytes": staged.stat().st_size,
                    "sha256": _sha256(staged),
                    "asset_frame_registration_digest": row["asset_frame_registration"][
                        "registration_digest"
                    ],
                }
            )
        request: dict[str, Any] = {
            "schema_version": REQUEST_SCHEMA_VERSION,
            "status": "frozen_pending_native_isaac_import",
            "program_id": "arm-decision-proof-v1",
            "scene_id": str(source.get("scene_id") or ""),
            "source_native_render_request": {
                "path": str(source_path),
                "size_bytes": source_path.stat().st_size,
                "sha256": _sha256(source_path),
                "receipt_digest": source["receipt_digest"],
            },
            "replacement_count": len(replacement_rows),
            "maximum_replacement_objects": MAX_REPLACEMENT_OBJECTS,
            "replacements": replacement_rows,
            "candidate_policy_queried": False,
            "native_isaac_executed": False,
            "physical_equivalence_claimed": False,
            "request_digest": "",
        }
        request["request_digest"] = canonical_digest(request, digest_field="request_digest")
        request_file = runtime / "paired_target_native_import_request.v1.json"
        write_json(request_file, request)
        runner_file = runtime / "run_paired_target_native_import_probe.py"
        shutil.copyfile(runner, runner_file)
        entrypoint = runtime / "run_paired_target_native_import_probe.sh"
        entrypoint.write_text(
            "#!/usr/bin/env bash\n"
            "set +e\n"
            'RUNTIME_DIR="$(cd "$(dirname "$0")" && pwd)"\n'
            'OUT_DIR="${BLUEPRINT_PAIRED_TARGET_IMPORT_OUTPUT_DIR:-$RUNTIME_DIR/../runtime_output}"\n'
            'mkdir -p "$OUT_DIR"\n'
            '/isaac-sim/python.sh "$RUNTIME_DIR/run_paired_target_native_import_probe.py" '
            '--request "$RUNTIME_DIR/paired_target_native_import_request.v1.json" '
            '--output-root "$OUT_DIR"\n'
            "runner_rc=$?\n"
            'if [ ! -s "$OUT_DIR/paired_target_native_import_runtime_result.v1.json" ]; then\n'
            "/isaac-sim/python.sh - \"$OUT_DIR/paired_target_native_import_runtime_result.v1.json\" <<'PY'\n"
            "import hashlib, json, sys\n"
            "from pathlib import Path\n"
            "path=Path(sys.argv[1])\n"
            'value={"schema_version":"paired_target_native_import_runtime_result.v1","status":"blocked","native_isaac_executed":False,"all_replacements_import_qualified":False,"candidate_policy_queried":False,"physical_equivalence_claimed":False,"blockers":["paired_target_native_import_runner_failed_without_runtime_result"],"result_digest":""}\n'
            'payload=dict(value); payload.pop("result_digest",None)\n'
            'value["result_digest"]="sha256:"+hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()\n'
            'path.write_text(json.dumps(value,indent=2,sort_keys=True)+"\\n",encoding="utf-8")\n'
            "PY\n"
            "fi\n"
            "exit $runner_rc\n",
            encoding="utf-8",
        )
        entrypoint.chmod(0o755)
        inventory = [
            {
                "relative_path": path.relative_to(runtime).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in sorted(runtime.rglob("*"))
            if path.is_file()
        ]
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": "ready",
            "implementation_commit": implementation_commit,
            "source_commit_sha": implementation_commit,
            "container_image": DEFAULT_IMAGE,
            "source_request_digest": source["receipt_digest"],
            "request_digest": request["request_digest"],
            "probe_spec_sha256": request["request_digest"],
            "replacement_count": len(replacement_rows),
            "replacements": replacement_rows,
            "input_files": inventory,
            "expected_output_filename": RESULT_FILENAME,
            "expected_output_schema": RESULT_SCHEMA_VERSION,
            "provider_allocation_performed": False,
            "paid_execution_authorized_by_bundle": False,
            "candidate_policy_queried": False,
            "retry_cap": 0,
            "provider_zero_required_before_and_after": True,
            "raw_nonredistributable_bytes_included": False,
            "canonical_interiorgs_included_or_mutated": False,
            "claim_ceiling": "immutable_native_import_probe_input_only",
            "input_digest": "",
        }
        manifest["input_digest"] = canonical_digest(manifest, digest_field="input_digest")
        write_json(runtime / "paired_target_native_import_bundle_manifest.v1.json", manifest)
        bundle = destination / "paired_target_native_import_provider_bundle.zip"
        with zipfile.ZipFile(bundle, "w", allowZip64=True) as archive:
            for path in sorted(runtime.rglob("*")):
                if path.is_file():
                    _write_zip_member(archive, path.relative_to(destination).as_posix(), path)
        receipt = {
            **manifest,
            "bundle_path": str(bundle),
            "bundle_size_bytes": bundle.stat().st_size,
            "bundle_sha256": _sha256(bundle),
            "receipt_digest": "",
        }
        receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
        write_json(destination / "paired_target_native_import_bundle_receipt.v1.json", receipt)
        return receipt
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise


def validate_paired_target_native_import_bundle(
    receipt_path: str | Path,
) -> dict[str, Any]:
    """Reopen one immutable 1--5 replacement bundle and verify every staged byte."""

    receipt_file, receipt = _read_mapping(
        receipt_path, "paired_target_native_import_bundle_receipt_invalid"
    )
    bundle = Path(str(receipt.get("bundle_path") or "")).expanduser().resolve()
    rows = receipt.get("input_files")
    if (
        receipt.get("schema_version") != SCHEMA_VERSION
        or receipt.get("receipt_digest") != canonical_digest(receipt, digest_field="receipt_digest")
        or receipt.get("status") != "ready"
        or receipt.get("source_commit_sha") != receipt.get("implementation_commit")
        or not isinstance(rows, list)
        or not 1 <= int(receipt.get("replacement_count") or 0) <= MAX_REPLACEMENT_OBJECTS
        or receipt.get("retry_cap") != 0
        or receipt.get("provider_zero_required_before_and_after") is not True
        or receipt.get("raw_nonredistributable_bytes_included") is not False
        or receipt.get("canonical_interiorgs_included_or_mutated") is not False
        or bundle.is_symlink()
        or not bundle.is_file()
        or bundle.stat().st_size != receipt.get("bundle_size_bytes")
        or _sha256(bundle) != receipt.get("bundle_sha256")
    ):
        raise PairedTargetNativeImportBundleError(
            "paired_target_native_import_bundle_receipt_invalid"
        )
    declared = {
        str(row.get("relative_path") or ""): dict(row) for row in rows if isinstance(row, Mapping)
    }
    if len(declared) != len(rows) or any(
        not path or PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts
        for path in declared
    ):
        raise PairedTargetNativeImportBundleError(
            "paired_target_native_import_bundle_inventory_invalid"
        )
    try:
        with zipfile.ZipFile(bundle) as archive:
            if archive.testzip() is not None:
                raise ValueError("zip_integrity")
            members = {
                name.removeprefix("provider_runtime/"): name
                for name in archive.namelist()
                if name.startswith("provider_runtime/") and not name.endswith("/")
            }
            manifest_name = "paired_target_native_import_bundle_manifest.v1.json"
            if set(members) != {*declared, manifest_name}:
                raise ValueError("inventory_mismatch")
            for relative, record in declared.items():
                body = archive.read(members[relative])
                if len(body) != record.get("size_bytes") or "sha256:" + hashlib.sha256(
                    body
                ).hexdigest() != record.get("sha256"):
                    raise ValueError("member_digest")
            manifest = json.loads(archive.read(f"provider_runtime/{manifest_name}").decode("utf-8"))
            if not isinstance(manifest, dict):
                raise ValueError("manifest_shape")
            expected_manifest = dict(receipt)
            for field in (
                "bundle_path",
                "bundle_size_bytes",
                "bundle_sha256",
                "receipt_digest",
            ):
                expected_manifest.pop(field, None)
            if manifest != expected_manifest:
                raise ValueError("manifest_mismatch")
    except (OSError, ValueError, KeyError, json.JSONDecodeError, zipfile.BadZipFile) as exc:
        raise PairedTargetNativeImportBundleError(
            "paired_target_native_import_bundle_archive_invalid"
        ) from exc
    return {**receipt, "receipt_path": str(receipt_file)}


__all__ = [
    "PairedTargetNativeImportBundleError",
    "REQUEST_SCHEMA_VERSION",
    "RESULT_FILENAME",
    "RESULT_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "build_paired_target_native_import_bundle",
    "validate_paired_target_native_import_bundle",
]
