"""Build one immutable provider bundle for native deformable-asset preparation.

This is a transport boundary only.  It packages an already inspected and
metric-frozen source package, the pinned released-runtime source packet, and
the task-neutral native worker.  A ready bundle does not qualify the cook.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import zipfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from .adp_isaac_lab_arena_vast import DEFAULT_IMAGE
from .decision_evidence_contracts import canonical_digest, canonical_json
from .native_deformable_asset_preparation import (
    PACKAGE_RECEIPT_FILENAME,
    PACKAGE_SCHEMA_VERSION,
)
from .native_task_runtime_source_packet import verify_native_task_runtime_source_packet


PROBE_KIND = "native-deformable-asset-preparation"
PROVIDER_BUNDLE_KIND = "native_deformable_asset"
SCHEMA_VERSION = "native_deformable_asset_provider_bundle.v1"
EXPECTED_OUTPUT_FILENAME = "native_deformable_asset_vast_execution.v1.json"
RESULT_SCHEMA_VERSION = "native_deformable_asset_vast_execution.v1"
_MAX_INPUT_FILE_BYTES = 512 * 1024 * 1024
_RUNTIME_MODULES = (
    "common.py",
    "decision_evidence_contracts.py",
    "external_simready_deformable_asset.py",
    "native_deformable_asset_preparation.py",
    "native_deformable_asset_preparation_worker.py",
    "native_deformable_asset_stage_adapter.py",
    "native_task_arena_import_scope.py",
    "native_task_entity_asset_authoring_bundle.py",
    "native_task_entity_contract.py",
    "native_task_runtime_source_packet.py",
    "native_task_runtime_source_provision.py",
    "task_entity_asset_candidate.py",
)


class NativeDeformableAssetProviderBundleError(ValueError):
    """Stable provider-bundle boundary failure."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(item) for item in errors if str(item))))
        super().__init__(";".join(self.errors))


def _sha256_bytes(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _snapshot_regular_file(
    path: Path,
    *,
    maximum_bytes: int,
    error: str,
) -> bytes:
    """Read one regular-file leaf exactly once without following a symlink."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise NativeDeformableAssetProviderBundleError([error]) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= maximum_bytes:
            raise NativeDeformableAssetProviderBundleError([error])
        content = bytearray()
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, maximum_bytes + 1 - len(content)))
            if not chunk:
                break
            content.extend(chunk)
            if len(content) > maximum_bytes:
                raise NativeDeformableAssetProviderBundleError([error])
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ) or len(content) != before.st_size:
            raise NativeDeformableAssetProviderBundleError([error])
        return bytes(content)
    finally:
        os.close(descriptor)


def _json(path: Path, *, error: str) -> tuple[dict[str, Any], bytes]:
    content = _snapshot_regular_file(path, maximum_bytes=16 * 1024 * 1024, error=error)
    try:
        value = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NativeDeformableAssetProviderBundleError([error]) from exc
    if not isinstance(value, dict):
        raise NativeDeformableAssetProviderBundleError([error])
    return value, content


def _verified_source_package(
    receipt_path: str | Path,
) -> tuple[dict[str, Any], bytes, dict[str, bytes]]:
    path = Path(receipt_path).expanduser().resolve()
    receipt, receipt_bytes = _json(path, error="native_deformable_provider_source_receipt_invalid")
    errors: list[str] = []
    if receipt.get("schema_version") != PACKAGE_SCHEMA_VERSION or receipt.get(
        "receipt_digest"
    ) != canonical_digest(receipt, digest_field="receipt_digest"):
        errors.append("native_deformable_provider_source_receipt_invalid")
    root = path.parent
    if Path(str(receipt.get("package_root") or "")).expanduser().resolve() != root:
        errors.append("native_deformable_provider_source_root_mismatch")
    rows = receipt.get("files")
    if not isinstance(rows, list) or not rows:
        errors.append("native_deformable_provider_source_files_invalid")
        rows = []
    seen: set[str] = set()
    snapshots: dict[str, bytes] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            errors.append("native_deformable_provider_source_file_invalid")
            continue
        relative = PurePosixPath(str(row.get("package_path") or ""))
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.as_posix() in {"", "."}
            or relative.as_posix() in seen
        ):
            errors.append("native_deformable_provider_source_file_invalid")
            continue
        seen.add(relative.as_posix())
        source = root.joinpath(*relative.parts)
        try:
            content = _snapshot_regular_file(
                source,
                maximum_bytes=_MAX_INPUT_FILE_BYTES,
                error=(
                    "native_deformable_provider_source_file_identity_mismatch:"
                    + relative.as_posix()
                ),
            )
        except NativeDeformableAssetProviderBundleError:
            errors.append(
                f"native_deformable_provider_source_file_identity_mismatch:{relative.as_posix()}"
            )
            continue
        if len(content) != row.get("size_bytes") or _sha256_bytes(content) != row.get("sha256"):
            errors.append(
                f"native_deformable_provider_source_file_identity_mismatch:{relative.as_posix()}"
            )
            continue
        snapshots[relative.as_posix()] = content
    if PACKAGE_RECEIPT_FILENAME != path.name:
        errors.append("native_deformable_provider_source_receipt_name_invalid")
    if errors:
        raise NativeDeformableAssetProviderBundleError(errors)
    return receipt, receipt_bytes, snapshots


def _runtime_source_container_mismatch(
    runtime_receipt: Mapping[str, Any], container_image: str
) -> bool:
    """Return True when a dependency-free packet depends on a different image."""

    wheels = runtime_receipt.get("runtime_dependency_wheels")
    if isinstance(wheels, list) and wheels:
        return False
    paired_stack = runtime_receipt.get("paired_stack")
    if not isinstance(paired_stack, Mapping):
        return True
    required_image = str(paired_stack.get("simulator_runtime_image") or "").strip()
    return not required_image or required_image != container_image


def _entrypoint(*, plan_digest: str) -> str:
    digest = json.dumps(plan_digest)
    output_name = json.dumps(EXPECTED_OUTPUT_FILENAME)
    return f"""#!/usr/bin/env bash
set +e
RUNTIME_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="${{BLUEPRINT_ADP_ARENA_OUTPUT_DIR:-$RUNTIME_DIR/../runtime_output}}"
mkdir -p "$OUT_DIR"
export PYTHONPATH="$RUNTIME_DIR${{PYTHONPATH:+:$PYTHONPATH}}"
echo "BLUEPRINT_WAM_RUNTIME_PHASE:native_deformable_asset:runtime_sources:started"
/isaac-sim/python.sh -m blueprint_pipeline.native_task_runtime_source_provision \
  --source-receipt "$RUNTIME_DIR/native_task_runtime_sources/native_task_runtime_source_packet.v1.json" \
  --source-packet "$RUNTIME_DIR/native_task_runtime_sources/native_task_runtime_sources.zip" \
  --extraction-dir "$RUNTIME_DIR/provisioned_runtime_sources" \
  --output "$OUT_DIR/native_task_runtime_source_provisioning.v1.json" \
  --simulator-root /isaac-sim
provision_rc=$?
if [ $provision_rc -eq 0 ]; then
  . "$RUNTIME_DIR/provisioned_runtime_sources/native_task_runtime_environment.sh"
  echo "BLUEPRINT_WAM_RUNTIME_PHASE:native_deformable_asset:worker:started"
  /isaac-sim/python.sh -m blueprint_pipeline.native_deformable_asset_preparation_worker \
    --plan "$RUNTIME_DIR/input_package/native_deformable_asset_preparation_plan.v1.json" \
    --expected-plan-digest {digest} \
    --package-root "$RUNTIME_DIR/input_package" \
    --output-root "$OUT_DIR/prepared_asset" \
    --isaaclab-source-root "$RUNTIME_DIR/provisioned_runtime_sources/runtime_sources/isaaclab" \
    --terminal-output "$OUT_DIR/native_deformable_asset_preparation_worker_terminal.v1.json"
  worker_rc=$?
else
  worker_rc=2
fi
/isaac-sim/python.sh - "$OUT_DIR" "$provision_rc" "$worker_rc" <<'PY'
import json
import sys
from pathlib import Path
out = Path(sys.argv[1])
provision_rc = int(sys.argv[2])
worker_rc = int(sys.argv[3])
terminal_path = out / "native_deformable_asset_preparation_worker_terminal.v1.json"
terminal = json.loads(terminal_path.read_text()) if terminal_path.is_file() else {{}}
ok = provision_rc == 0 and worker_rc == 0 and str(terminal.get("status", "")).startswith("worker_payload_materialized")
result = {{
    "schema_version": "native_deformable_asset_vast_execution.v1",
    "status": "completed" if ok else "blocked",
    "blockers": [] if ok else (["native_deformable_asset_runtime_source_provisioning_failed"] if provision_rc else list(terminal.get("errors") or ["native_deformable_asset_worker_failed"])),
    "candidate_policy_queried": False,
    "candidate_outcomes_accessed": False,
    "native_isaac_executed": bool(provision_rc == 0),
    "worker_terminal_status": terminal.get("status"),
    "worker_result_digest": terminal.get("worker_result_digest"),
    "provider_zero_required_after_return": True,
    "native_qualification_requires_trusted_return_verification": True,
}}
(out / {output_name}).write_text(json.dumps(result, sort_keys=True, indent=2) + "\\n")
PY
result_rc=$?
if [ $result_rc -ne 0 ]; then
  echo "native_deformable_asset_worker_failed_without_runtime_result"
  exit $result_rc
fi
if [ $worker_rc -ne 0 ]; then
  echo "native_deformable_asset_process_exited_without_result"
fi
exit $worker_rc
"""


def _write_deterministic_zip(root: Path, output: Path) -> None:
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source in sorted(path for path in root.rglob("*") if path.is_file() and path != output):
            info = zipfile.ZipInfo(source.relative_to(root).as_posix())
            info.date_time = (1980, 1, 1, 0, 0, 0)
            info.external_attr = (0o755 if source.stat().st_mode & stat.S_IXUSR else 0o644) << 16
            archive.writestr(info, source.read_bytes(), compress_type=zipfile.ZIP_DEFLATED)


def build_native_deformable_asset_provider_bundle(
    *,
    job_dir: str | Path,
    source_package_receipt_path: str | Path,
    runtime_source_packet_receipt_path: str | Path,
    implementation_commit: str,
    package_source_root: str | Path,
    container_image: str = DEFAULT_IMAGE,
    runtime_source_packet_verifier: Callable[..., Mapping[str, Any]] = (
        verify_native_task_runtime_source_packet
    ),
) -> dict[str, Any]:
    """Build the exact paid-free bundle consumed by the canonical Vast lane."""

    if (
        not isinstance(implementation_commit, str)
        or len(implementation_commit) != 40
        or any(character not in "0123456789abcdef" for character in implementation_commit)
    ):
        raise NativeDeformableAssetProviderBundleError(
            ["native_deformable_provider_implementation_commit_invalid"]
        )
    image_digest = container_image.rpartition("@sha256:")[2]
    if (
        not isinstance(container_image, str)
        or len(image_digest) != 64
        or any(character not in "0123456789abcdef" for character in image_digest)
    ):
        raise NativeDeformableAssetProviderBundleError(
            ["native_deformable_provider_container_image_unpinned"]
        )
    source_receipt, source_receipt_bytes, source_snapshots = _verified_source_package(
        source_package_receipt_path
    )
    runtime_receipt = dict(runtime_source_packet_verifier(runtime_source_packet_receipt_path))
    packet = Path(str(runtime_receipt.get("verified_packet_path") or "")).resolve()
    packet_bytes = _snapshot_regular_file(
        packet,
        maximum_bytes=_MAX_INPUT_FILE_BYTES,
        error="native_deformable_provider_runtime_source_packet_invalid",
    )
    runtime_receipt_bytes = _snapshot_regular_file(
        Path(runtime_source_packet_receipt_path).expanduser().resolve(),
        maximum_bytes=16 * 1024 * 1024,
        error="native_deformable_provider_runtime_source_packet_invalid",
    )
    if runtime_receipt.get("redistribution_permitted") is not True or _sha256_bytes(
        packet_bytes
    ) != runtime_receipt.get("packet_sha256"):
        raise NativeDeformableAssetProviderBundleError(
            ["native_deformable_provider_runtime_source_packet_invalid"]
        )
    if _runtime_source_container_mismatch(runtime_receipt, container_image):
        raise NativeDeformableAssetProviderBundleError(
            ["native_deformable_provider_runtime_source_container_mismatch"]
        )
    plan_digest = str(source_receipt.get("plan_digest") or "")
    package_sources = Path(package_source_root).expanduser().resolve()
    module_snapshots = {
        name: _snapshot_regular_file(
            package_sources / name,
            maximum_bytes=16 * 1024 * 1024,
            error="native_deformable_provider_runtime_module_missing",
        )
        for name in _RUNTIME_MODULES
    }
    job = Path(job_dir).expanduser().resolve()
    if job.exists():
        raise NativeDeformableAssetProviderBundleError(["native_deformable_provider_output_exists"])
    runtime = job / "provider_runtime"
    package = runtime / "blueprint_pipeline"
    try:
        package.mkdir(parents=True)
    except FileExistsError as exc:
        raise NativeDeformableAssetProviderBundleError(
            ["native_deformable_provider_output_exists"]
        ) from exc
    (package / "__init__.py").write_text("", encoding="utf-8")
    for name, content in module_snapshots.items():
        (package / name).write_bytes(content)
    input_root = runtime / "input_package"
    for relative_name, content in source_snapshots.items():
        relative = PurePosixPath(relative_name)
        destination = input_root.joinpath(*relative.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
    (input_root / PACKAGE_RECEIPT_FILENAME).write_bytes(source_receipt_bytes)
    runtime_sources = runtime / "native_task_runtime_sources"
    runtime_sources.mkdir()
    (runtime_sources / "native_task_runtime_source_packet.v1.json").write_bytes(
        runtime_receipt_bytes
    )
    (runtime_sources / "native_task_runtime_sources.zip").write_bytes(packet_bytes)
    entrypoint = runtime / "run_native_deformable_asset_provider_runtime.sh"
    entrypoint.write_text(_entrypoint(plan_digest=plan_digest), encoding="utf-8")
    entrypoint.chmod(entrypoint.stat().st_mode | stat.S_IXUSR)
    transport_entrypoint = runtime / "run_adp_arena_provider_runtime.sh"
    transport_entrypoint.write_bytes(entrypoint.read_bytes())
    transport_entrypoint.chmod(transport_entrypoint.stat().st_mode | stat.S_IXUSR)
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "ready",
        "provider_bundle_kind": PROVIDER_BUNDLE_KIND,
        "execution_mode": "asset_preparation_canary",
        "implementation_commit": implementation_commit,
        "container_image": container_image,
        "source_package_receipt_digest": source_receipt["receipt_digest"],
        "source_package_content_digest": source_receipt["package_content_digest"],
        "plan_digest": plan_digest,
        "runtime_source_packet_receipt_digest": runtime_receipt["receipt_digest"],
        "runtime_source_packet_sha256": runtime_receipt["packet_sha256"],
        "runtime_entrypoint": "provider_runtime/run_native_deformable_asset_provider_runtime.sh",
        "transport_entrypoint": "provider_runtime/run_adp_arena_provider_runtime.sh",
        "expected_output_filename": EXPECTED_OUTPUT_FILENAME,
        "candidate_policy_queried": False,
        "candidate_outcomes_accessed": False,
        "native_cook_qualified": False,
        "provider_mutations_performed": 0,
        "manifest_digest": "",
    }
    manifest["manifest_digest"] = canonical_digest(manifest, digest_field="manifest_digest")
    manifest_path = runtime / "native_deformable_asset_provider_manifest.json"
    manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    bundle_path = job / "native_deformable_asset_provider_bundle.v1.zip"
    _write_deterministic_zip(job, bundle_path)
    receipt = {
        **manifest,
        "bundle_path": str(bundle_path),
        "bundle_sha256": _sha256(bundle_path),
        "bundle_size_bytes": bundle_path.stat().st_size,
        "input_digest": canonical_digest(
            {
                "source_package_receipt_digest": source_receipt["receipt_digest"],
                "runtime_source_packet_receipt_digest": runtime_receipt["receipt_digest"],
                "implementation_commit": implementation_commit,
                "container_image": container_image,
            }
        ),
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    (job / "native_deformable_asset_provider_bundle_receipt.v1.json").write_text(
        canonical_json(receipt) + "\n", encoding="utf-8"
    )
    return receipt


def load_verified_native_deformable_asset_provider_bundle(
    receipt_path: str | Path,
    *,
    expected_implementation_commit: str,
    expected_source_package_receipt_digest: str,
    expected_runtime_source_packet_receipt_digest: str,
) -> dict[str, Any]:
    """Replay one dry-run bundle receipt before consuming paid authority."""

    receipt, _ = _json(
        Path(receipt_path).expanduser().resolve(),
        error="native_deformable_provider_bundle_receipt_invalid",
    )
    bundle = Path(str(receipt.get("bundle_path") or "")).expanduser().resolve()
    errors: list[str] = []
    if (
        receipt.get("schema_version") != SCHEMA_VERSION
        or receipt.get("status") != "ready"
        or receipt.get("provider_bundle_kind") != PROVIDER_BUNDLE_KIND
        or receipt.get("execution_mode") != "asset_preparation_canary"
        or receipt.get("candidate_policy_queried") is not False
        or receipt.get("candidate_outcomes_accessed") is not False
        or receipt.get("native_cook_qualified") is not False
        or receipt.get("receipt_digest") != canonical_digest(receipt, digest_field="receipt_digest")
    ):
        errors.append("native_deformable_provider_bundle_receipt_invalid")
    if receipt.get("implementation_commit") != expected_implementation_commit:
        errors.append("native_deformable_provider_bundle_commit_mismatch")
    if receipt.get("source_package_receipt_digest") != expected_source_package_receipt_digest:
        errors.append("native_deformable_provider_bundle_source_mismatch")
    if (
        receipt.get("runtime_source_packet_receipt_digest")
        != expected_runtime_source_packet_receipt_digest
    ):
        errors.append("native_deformable_provider_bundle_runtime_source_mismatch")
    try:
        bundle_bytes = _snapshot_regular_file(
            bundle,
            maximum_bytes=_MAX_INPUT_FILE_BYTES,
            error="native_deformable_provider_bundle_bytes_mismatch",
        )
    except NativeDeformableAssetProviderBundleError:
        bundle_bytes = b""
    if len(bundle_bytes) != receipt.get("bundle_size_bytes") or _sha256_bytes(
        bundle_bytes
    ) != receipt.get("bundle_sha256"):
        errors.append("native_deformable_provider_bundle_bytes_mismatch")
    else:
        try:
            with zipfile.ZipFile(io.BytesIO(bundle_bytes)) as archive:
                manifest = json.loads(
                    archive.read("provider_runtime/native_deformable_asset_provider_manifest.json")
                )
        except (KeyError, UnicodeDecodeError, json.JSONDecodeError, zipfile.BadZipFile):
            errors.append("native_deformable_provider_bundle_manifest_invalid")
        else:
            receipt_only = {
                "bundle_path",
                "bundle_sha256",
                "bundle_size_bytes",
                "input_digest",
                "receipt_digest",
            }
            expected_manifest = {
                key: value for key, value in receipt.items() if key not in receipt_only
            }
            if (
                not isinstance(manifest, dict)
                or manifest != expected_manifest
                or manifest.get("manifest_digest")
                != canonical_digest(manifest, digest_field="manifest_digest")
            ):
                errors.append("native_deformable_provider_bundle_manifest_invalid")
    expected_input_digest = canonical_digest(
        {
            "source_package_receipt_digest": expected_source_package_receipt_digest,
            "runtime_source_packet_receipt_digest": (expected_runtime_source_packet_receipt_digest),
            "implementation_commit": expected_implementation_commit,
            "container_image": receipt.get("container_image"),
        }
    )
    if receipt.get("input_digest") != expected_input_digest:
        errors.append("native_deformable_provider_bundle_input_digest_mismatch")
    if errors:
        raise NativeDeformableAssetProviderBundleError(errors)
    return receipt


__all__ = [
    "EXPECTED_OUTPUT_FILENAME",
    "NativeDeformableAssetProviderBundleError",
    "PROBE_KIND",
    "PROVIDER_BUNDLE_KIND",
    "RESULT_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "build_native_deformable_asset_provider_bundle",
    "load_verified_native_deformable_asset_provider_bundle",
]
