"""Portable, source-bound input bundle for the current-reference policy canary."""

from __future__ import annotations

import json
import re
import shutil
import zipfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from .common import ensure_dir, write_json
from .openpi_current_reference_droid_policy_runtime import (
    CURRENT_REFERENCE_INVENTORY_FILES,
)
from .openpi_current_reference_observation import (
    GENERATED_OBSERVATION_SCHEMA,
    validate_current_reference_policy_observation_manifest,
)
from .policy_ranking_thesis import canonical_sha256, file_sha256


INPUT_SCHEMA_VERSION = "openpi_current_reference_gpu_input_bundle.v1"
INPUT_RECEIPT_SCHEMA_VERSION = "openpi_current_reference_gpu_input_bundle_receipt.v1"
MAX_INPUT_BYTES = 32 * 1024**2
_COMMIT = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SOURCE_URL = re.compile(
    r"https://codeload\.github\.com/ognjhunt/BlueprintCapturePipeline/"
    r"tar\.gz/(?P<commit>[0-9a-f]{40})"
)


def _read_object(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"current_reference_bundle_json_not_object:{path}")
    return dict(value)


def _portable_initial_observation(*, source_manifest: Path, staging_root: Path) -> dict[str, Any]:
    payload = _read_object(source_manifest)
    try:
        payload = validate_current_reference_policy_observation_manifest(payload)
    except ValueError as exc:
        raise ValueError("current_reference_bundle_initial_observation_invalid") from exc
    recorded = str(payload["manifest_sha256"])
    portable = dict(payload)
    views = portable.get("views")
    state = portable.get("state")
    if not isinstance(views, Mapping) or not isinstance(state, Mapping):
        raise ValueError("current_reference_bundle_initial_observation_files_invalid")
    portable_views: dict[str, Any] = {}
    portable_state: dict[str, Any] = {}
    for view_id, raw_row in views.items():
        if not isinstance(raw_row, Mapping):
            raise ValueError("current_reference_bundle_view_row_invalid")
        row = dict(raw_row)
        source = Path(str(row.get("frame_path") or "")).expanduser().resolve()
        expected = str(row.get("frame_sha256") or "")
        if not source.is_file() or source.is_symlink() or file_sha256(source) != expected:
            raise ValueError("current_reference_bundle_view_file_invalid")
        relative = Path("files") / source.name
        destination = staging_root / relative
        if destination.exists():
            raise ValueError("current_reference_bundle_duplicate_filename")
        shutil.copyfile(source, destination)
        row["frame_path"] = relative.as_posix()
        portable_views[str(view_id)] = row
    for state_id, raw_row in state.items():
        if not isinstance(raw_row, Mapping):
            raise ValueError("current_reference_bundle_state_row_invalid")
        row = dict(raw_row)
        source = Path(str(row.get("path") or "")).expanduser().resolve()
        expected = str(row.get("sha256") or "")
        if not source.is_file() or source.is_symlink() or file_sha256(source) != expected:
            raise ValueError("current_reference_bundle_state_file_invalid")
        relative = Path("files") / source.name
        destination = staging_root / relative
        if destination.exists():
            raise ValueError("current_reference_bundle_duplicate_filename")
        shutil.copyfile(source, destination)
        row["path"] = relative.as_posix()
        portable_state[str(state_id)] = row
    portable["views"] = portable_views
    portable["state"] = portable_state
    portable["portable_bundle_derivative_of_manifest_sha256"] = recorded
    portable.pop("manifest_sha256", None)
    portable["manifest_sha256"] = canonical_sha256(portable)
    return portable


def build_current_reference_gpu_input_bundle(
    *,
    source_freeze_path: str | Path,
    checkpoint_inventory_dir: str | Path,
    initial_observation_manifest_path: str | Path,
    runtime_source_commit: str,
    runtime_source_archive_url: str,
    runtime_source_archive_sha256: str,
    image_source_commit: str,
    output_zip: str | Path,
) -> dict[str, Any]:
    """Build the small signed bundle; checkpoint weights remain public remote inputs."""

    runtime_commit = runtime_source_commit.strip().lower()
    image_commit = image_source_commit.strip().lower()
    source_match = _SOURCE_URL.fullmatch(runtime_source_archive_url.strip())
    if (
        not _COMMIT.fullmatch(runtime_commit)
        or not _COMMIT.fullmatch(image_commit)
        or source_match is None
        or source_match.group("commit") != runtime_commit
        or not _SHA256.fullmatch(runtime_source_archive_sha256.strip().lower())
    ):
        raise ValueError("current_reference_bundle_runtime_source_identity_invalid")
    output = Path(output_zip).expanduser().resolve()
    staging = output.with_name(f".{output.name}.staging")
    if staging.exists() or output.exists():
        raise FileExistsError("current_reference_bundle_output_exists")
    ensure_dir(staging / "files")
    ensure_dir(staging / "checkpoint_inventories")
    try:
        source_freeze = Path(source_freeze_path).expanduser().resolve()
        shutil.copyfile(source_freeze, staging / "source_freeze.json")
        inventory_root = Path(checkpoint_inventory_dir).expanduser().resolve()
        for name in CURRENT_REFERENCE_INVENTORY_FILES.values():
            source = inventory_root / name
            if not source.is_file() or source.is_symlink():
                raise ValueError(f"current_reference_bundle_inventory_missing:{name}")
            shutil.copyfile(source, staging / "checkpoint_inventories" / name)
        portable = _portable_initial_observation(
            source_manifest=Path(initial_observation_manifest_path).expanduser().resolve(),
            staging_root=staging,
        )
        write_json(staging / "initial_observation_manifest.json", portable)
        if portable.get("schema_version") == GENERATED_OBSERVATION_SCHEMA:
            observation_source = portable.get("source")
            if not isinstance(observation_source, Mapping):
                raise ValueError("current_reference_bundle_generated_observation_source_invalid")
            policy_ids = [str(observation_source.get("policy_id") or "")]
            purpose = "label_free_current_reference_same_policy_requery"
            same_candidate_policy_id: str | None = policy_ids[0]
        else:
            policy_ids = sorted(CURRENT_REFERENCE_INVENTORY_FILES)
            purpose = "label_free_current_reference_real_policy_identity_canary"
            same_candidate_policy_id = None
        file_rows = [
            {
                "path": path.relative_to(staging).as_posix(),
                "sha256": file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        ]
        manifest: dict[str, Any] = {
            "schema_version": INPUT_SCHEMA_VERSION,
            "purpose": purpose,
            "runtime_source": {
                "repository": "https://github.com/ognjhunt/BlueprintCapturePipeline",
                "commit": runtime_commit,
                "archive_url": runtime_source_archive_url.strip(),
                "archive_sha256": runtime_source_archive_sha256.strip().lower(),
                "overlay_required": True,
            },
            "image_source_commit": image_commit,
            "policy_ids": policy_ids,
            "observation_schema": portable["schema_version"],
            "same_candidate_policy_id": same_candidate_policy_id,
            "requests_per_policy": 1,
            "raw_3dgs_included": False,
            "redistribution_authorized": False,
            "label_free": True,
            "confirmation_eligible": False,
            "physical_outcome_included": False,
            "checkpoint_weights_included": False,
            "files": file_rows,
        }
        manifest["manifest_sha256"] = canonical_sha256(manifest)
        write_json(staging / "manifest.json", manifest)
        output.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(staging.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(staging).as_posix())
        if output.stat().st_size > MAX_INPUT_BYTES:
            raise ValueError("current_reference_bundle_too_large")
        receipt = {
            "schema_version": INPUT_RECEIPT_SCHEMA_VERSION,
            "bundle_path": str(output),
            "bundle_sha256": file_sha256(output),
            "bundle_size_bytes": output.stat().st_size,
            "manifest": manifest,
        }
        receipt["manifest_sha256"] = canonical_sha256(receipt)
        return receipt
    except Exception:
        output.unlink(missing_ok=True)
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def extract_current_reference_gpu_input_bundle(
    *, bundle_path: str | Path, expected_bundle_sha256: str, output_dir: str | Path
) -> dict[str, Any]:
    bundle = Path(bundle_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if (
        not bundle.is_file()
        or bundle.is_symlink()
        or bundle.stat().st_size > MAX_INPUT_BYTES
        or file_sha256(bundle) != expected_bundle_sha256
    ):
        raise ValueError("current_reference_bundle_missing_unsafe_or_hash_mismatch")
    if output.exists():
        raise FileExistsError("current_reference_bundle_extract_output_exists")
    output.mkdir(parents=True)
    try:
        with zipfile.ZipFile(bundle) as archive:
            members = archive.infolist()
            names: set[str] = set()
            for member in members:
                pure = PurePosixPath(member.filename)
                if (
                    pure.is_absolute()
                    or ".." in pure.parts
                    or member.filename in names
                    or member.flag_bits & 0x1
                ):
                    raise ValueError("current_reference_bundle_member_unsafe")
                names.add(member.filename)
            if "manifest.json" not in names:
                raise ValueError("current_reference_bundle_manifest_missing")
            archive.extractall(output)
        manifest = _read_object(output / "manifest.json")
        declared = str(manifest.get("manifest_sha256") or "")
        payload = dict(manifest)
        payload.pop("manifest_sha256", None)
        if manifest.get("schema_version") != INPUT_SCHEMA_VERSION or declared != canonical_sha256(
            payload
        ):
            raise ValueError("current_reference_bundle_manifest_invalid")
        rows = manifest.get("files")
        if not isinstance(rows, list):
            raise ValueError("current_reference_bundle_file_inventory_invalid")
        expected_names = {"manifest.json"}
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError("current_reference_bundle_file_row_invalid")
            relative = PurePosixPath(str(row.get("path") or ""))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("current_reference_bundle_file_path_invalid")
            path = output.joinpath(*relative.parts)
            if (
                not path.is_file()
                or path.is_symlink()
                or path.stat().st_size != row.get("size_bytes")
                or file_sha256(path) != row.get("sha256")
            ):
                raise ValueError("current_reference_bundle_file_verification_failed")
            expected_names.add(relative.as_posix())
        actual_names = {
            path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()
        }
        if actual_names != expected_names:
            raise ValueError("current_reference_bundle_file_allowlist_mismatch")
        return {
            "manifest": manifest,
            "source_freeze_path": output / "source_freeze.json",
            "checkpoint_inventory_dir": output / "checkpoint_inventories",
            "initial_observation_manifest_path": output / "initial_observation_manifest.json",
        }
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise


__all__ = [
    "INPUT_RECEIPT_SCHEMA_VERSION",
    "INPUT_SCHEMA_VERSION",
    "MAX_INPUT_BYTES",
    "build_current_reference_gpu_input_bundle",
    "extract_current_reference_gpu_input_bundle",
]
