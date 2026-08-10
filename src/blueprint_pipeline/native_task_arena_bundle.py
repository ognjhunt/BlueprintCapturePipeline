"""Build a deterministic provider bundle from a sealed native-task packet.

The historical ADP-009D bundle builder constructs the first canned-beverage
scene while it packages it.  Reusing that builder for a different task would
silently construct the wrong collider derivative and ship first-scene receipts.
This builder has the narrower, reusable job the runtime needs: reverify an
already materialized :mod:`native_task_arena_packet`, copy it byte-for-byte,
ship a selected worker plus its flat Python modules, and bind the pinned Arena
image and expected result name.

No simulator or provider is imported here.  A ready bundle proves transport
integrity only; native application and task outcomes remain worker evidence.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import stat
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from .adp_isaac_lab_arena_vast import DEFAULT_IMAGE
from .common import ensure_dir, utc_now_iso, write_json
from .decision_evidence_contracts import canonical_digest
from .native_task_arena_packet import RECEIPT_SCHEMA_VERSION
from .native_task_runtime_source_packet import (
    verify_native_task_runtime_source_packet,
)


SCHEMA_VERSION = "native_task_arena_provider_bundle.v1"
DEFAULT_EXPECTED_OUTPUT_FILENAME = "native_task_arena_construction_result.v1.json"


class NativeTaskArenaBundleError(ValueError):
    """Stable packet/bundle failures before any provider mutation."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _has_symlink_component(path: Path, *, root: Path) -> bool:
    current = root
    for part in path.relative_to(root).parts:
        current /= part
        if current.is_symlink():
            return True
    return False


def _read_mapping(path: Path, *, error: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise NativeTaskArenaBundleError([error]) from exc
    if not isinstance(value, Mapping):
        raise NativeTaskArenaBundleError([error])
    return dict(value)


def _verified_packet(packet_dir: str | Path) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    raw_root = Path(packet_dir).expanduser()
    if raw_root.is_symlink():
        raise NativeTaskArenaBundleError(["native_task_arena_bundle_packet_invalid"])
    root = raw_root.resolve()
    if not root.is_dir():
        raise NativeTaskArenaBundleError(["native_task_arena_bundle_packet_invalid"])
    receipt_path = root / "native_task_arena_packet_receipt.v1.json"
    receipt = _read_mapping(
        receipt_path, error="native_task_arena_bundle_packet_receipt_invalid"
    )
    errors: list[str] = []
    if receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION:
        errors.append("native_task_arena_bundle_packet_receipt_invalid")
    if receipt.get("receipt_digest") != canonical_digest(
        receipt, digest_field="receipt_digest"
    ):
        errors.append("native_task_arena_bundle_packet_receipt_digest_invalid")

    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        relative = path.relative_to(root)
        if _has_symlink_component(path, root=root) or not path.is_file():
            errors.append(
                f"native_task_arena_bundle_packet_file_invalid:{relative.as_posix()}"
            )
            continue
        rows.append(
            {
                "relative_path": relative.as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    by_path = {row["relative_path"]: row for row in rows}
    required = {
        "native_task_arena_packet_request.v1.json",
        "native_task_runtime_contract.v1.json",
        "native_task_arena_scene_plan.v1.json",
        "native_task_arena_packet_receipt.v1.json",
    }
    for missing in sorted(required - set(by_path)):
        errors.append(f"native_task_arena_bundle_packet_file_missing:{missing}")

    for artifact in receipt.get("artifacts") or []:
        if not isinstance(artifact, Mapping):
            errors.append("native_task_arena_bundle_packet_artifact_invalid")
            continue
        relative = str(artifact.get("relative_path") or "")
        observed = by_path.get(relative)
        if (
            observed is None
            or observed["sha256"] != artifact.get("sha256")
            or observed["size_bytes"] != artifact.get("size_bytes")
        ):
            errors.append(
                f"native_task_arena_bundle_packet_artifact_identity_mismatch:{relative}"
            )
    for binding in receipt.get("source_bindings") or []:
        if not isinstance(binding, Mapping):
            errors.append("native_task_arena_bundle_packet_source_binding_invalid")
            continue
        relative = str(binding.get("staged_relative_path") or "")
        pure = PurePosixPath(relative)
        observed = by_path.get(relative)
        if (
            pure.is_absolute()
            or ".." in pure.parts
            or observed is None
            or observed["sha256"] != binding.get("staged_sha256")
            or observed["size_bytes"] != binding.get("staged_size_bytes")
        ):
            errors.append(
                f"native_task_arena_bundle_packet_asset_identity_mismatch:{binding.get('semantic_role') or relative}"
            )
    if errors:
        raise NativeTaskArenaBundleError(errors)
    return root, receipt, rows


def _entrypoint(
    *, expected_output_filename: str, runtime_source_packet_required: bool
) -> str:
    quoted = json.dumps(str(expected_output_filename))
    source_required = "true" if runtime_source_packet_required else "false"
    return f'''#!/usr/bin/env bash
set +e
RUNTIME_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="${{BLUEPRINT_ADP_ARENA_OUTPUT_DIR:-$RUNTIME_DIR/../runtime_output}}"
mkdir -p "$OUT_DIR"
echo "BLUEPRINT_WAM_RUNTIME_PHASE:native_task_arena:runtime_sources:started"
SOURCE_RECEIPT="$RUNTIME_DIR/native_task_runtime_sources/native_task_runtime_source_packet.v1.json"
SOURCE_PACKET="$RUNTIME_DIR/native_task_runtime_sources/native_task_runtime_sources.zip"
if [ -f "$SOURCE_RECEIPT" ] && [ -f "$SOURCE_PACKET" ]; then
  cd "$RUNTIME_DIR"
  /isaac-sim/python.sh -m blueprint_pipeline.native_task_runtime_source_provision \
    --source-receipt "$SOURCE_RECEIPT" \
    --source-packet "$SOURCE_PACKET" \
    --extraction-dir "$RUNTIME_DIR/provisioned_runtime_sources" \
    --output "$OUT_DIR/native_task_runtime_source_provisioning.v1.json" \
    --simulator-root /isaac-sim
  provision_rc=$?
  cd "$RUNTIME_DIR"
elif {source_required}; then
  provision_rc=2
else
  provision_rc=0
fi
if [ $provision_rc -ne 0 ]; then
  echo "BLUEPRINT_WAM_RUNTIME_PHASE:native_task_arena:runtime_sources:blocked"
  /isaac-sim/python.sh - "$OUT_DIR" <<'PY'
import json
import sys
from pathlib import Path
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
name = {quoted}
(out / name).write_text(json.dumps({{
    "schema_version": "native_task_arena_construction_result.v1",
    "status": "blocked",
    "blockers": ["native_task_runtime_source_provisioning_failed"],
    "candidate_policy_queried": False,
    "candidate_outcomes_accessed": False,
    "native_isaac_executed": False,
    "provider_zero_required_after_return": True
}}, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
PY
  exit $provision_rc
fi
echo "BLUEPRINT_WAM_RUNTIME_PHASE:native_task_arena:runtime_sources:completed"
echo "BLUEPRINT_WAM_RUNTIME_PHASE:native_task_arena:media_toolchain:started"
if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
  DEBIAN_FRONTEND=noninteractive apt-get update -qq >"$OUT_DIR/media_toolchain_install.log" 2>&1 && \
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg >>"$OUT_DIR/media_toolchain_install.log" 2>&1
fi
if command -v ffmpeg >/dev/null 2>&1 && command -v ffprobe >/dev/null 2>&1; then
  echo "BLUEPRINT_WAM_RUNTIME_PHASE:native_task_arena:media_toolchain:completed"
else
  echo "BLUEPRINT_WAM_RUNTIME_PHASE:native_task_arena:media_toolchain:blocked"
fi
/isaac-sim/python.sh "$RUNTIME_DIR/adp_arena_provider_runner.py"
runner_rc=$?
if [ $runner_rc -ne 0 ] && [ ! -f "$OUT_DIR/{expected_output_filename}" ]; then
/isaac-sim/python.sh - "$OUT_DIR" <<'PY'
import json
import sys
from pathlib import Path
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
name = {quoted}
(out / name).write_text(json.dumps({{
    "schema_version": "native_task_arena_construction_result.v1",
    "status": "blocked",
    "blockers": [
        "native_task_arena_worker_failed_without_runtime_result",
        "native_task_arena_process_exited_without_result"
    ],
    "candidate_policy_queried": False,
    "candidate_outcomes_accessed": False,
    "provider_zero_required_after_return": True
}}, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
PY
fi
exit $runner_rc
'''


def build_native_task_arena_bundle(
    *,
    job_dir: str | Path,
    packet_dir: str | Path,
    worker_source: str | Path,
    runtime_module_sources: Sequence[str | Path],
    implementation_commit: str,
    execution_mode: str = "construction_canary",
    policy_candidate_id: str | None = None,
    expected_output_filename: str = DEFAULT_EXPECTED_OUTPUT_FILENAME,
    container_image: str = DEFAULT_IMAGE,
    runtime_source_packet_receipt: str | Path | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Package one exact native-task packet without reconstructing its scene."""

    if len(implementation_commit) != 40 or any(
        character not in "0123456789abcdef" for character in implementation_commit
    ):
        raise NativeTaskArenaBundleError(
            ["native_task_arena_bundle_implementation_commit_invalid"]
        )
    if execution_mode not in {"construction_canary", "controls", "policy"}:
        raise NativeTaskArenaBundleError(
            ["native_task_arena_bundle_execution_mode_invalid"]
        )
    if (execution_mode == "policy") is not bool(str(policy_candidate_id or "").strip()):
        raise NativeTaskArenaBundleError(
            ["native_task_arena_bundle_policy_binding_invalid"]
        )
    pure_output = PurePosixPath(str(expected_output_filename))
    if (
        pure_output.name != str(expected_output_filename)
        or str(expected_output_filename) in {"", ".", ".."}
    ):
        raise NativeTaskArenaBundleError(
            ["native_task_arena_bundle_output_filename_invalid"]
        )
    image = str(container_image).strip()
    if "@sha256:" not in image or len(image.rsplit("@sha256:", 1)[-1]) != 64:
        raise NativeTaskArenaBundleError(
            ["native_task_arena_bundle_container_image_not_digest_pinned"]
        )

    packet_root, packet_receipt, packet_rows = _verified_packet(packet_dir)
    runtime_source_receipt: dict[str, Any] | None = None
    if runtime_source_packet_receipt is not None:
        runtime_source_receipt = verify_native_task_runtime_source_packet(
            runtime_source_packet_receipt
        )
    worker = Path(worker_source).expanduser().resolve()
    if not worker.is_file():
        raise NativeTaskArenaBundleError(["native_task_arena_bundle_worker_missing"])
    modules: list[Path] = []
    module_names: set[str] = set()
    for source in runtime_module_sources:
        module = Path(source).expanduser().resolve()
        if not module.is_file():
            raise NativeTaskArenaBundleError(
                [f"native_task_arena_bundle_runtime_module_missing:{module.name}"]
            )
        if module.name in module_names or module.name == "adp_arena_provider_runner.py":
            raise NativeTaskArenaBundleError(
                [f"native_task_arena_bundle_runtime_module_duplicate:{module.name}"]
            )
        module_names.add(module.name)
        modules.append(module)

    job = Path(job_dir).expanduser().resolve()
    if job.exists():
        shutil.rmtree(job)
    runtime = job / "provider_runtime"
    packet_destination = runtime / "native_task_packet"
    ensure_dir(runtime)
    shutil.copytree(packet_root, packet_destination, symlinks=False)
    shutil.copy2(worker, runtime / "adp_arena_provider_runner.py")
    package = runtime / "blueprint_pipeline"
    ensure_dir(package)
    (package / "__init__.py").write_text("", encoding="utf-8")
    module_rows: list[dict[str, Any]] = []
    for module in sorted(modules, key=lambda path: path.name):
        destination = package / module.name
        shutil.copy2(module, destination)
        module_rows.append(
            {
                "relative_path": f"blueprint_pipeline/{module.name}",
                "size_bytes": destination.stat().st_size,
                "sha256": _sha256(destination),
            }
        )
    if runtime_source_receipt is not None:
        runtime_sources = runtime / "native_task_runtime_sources"
        ensure_dir(runtime_sources)
        source_receipt_path = Path(runtime_source_packet_receipt).expanduser().resolve()
        source_packet_path = Path(runtime_source_receipt["verified_packet_path"])
        shutil.copy2(
            source_receipt_path,
            runtime_sources / "native_task_runtime_source_packet.v1.json",
        )
        shutil.copy2(
            source_packet_path,
            runtime_sources / "native_task_runtime_sources.zip",
        )
    entrypoint = runtime / "run_adp_arena_provider_runtime.sh"
    entrypoint.write_text(
        _entrypoint(
            expected_output_filename=expected_output_filename,
            runtime_source_packet_required=runtime_source_receipt is not None,
        ),
        encoding="utf-8",
    )
    entrypoint.chmod(
        entrypoint.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
    )

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at or utc_now_iso(),
        "status": "ready",
        "program_id": "arm-decision-proof-v1",
        "execution_mode": execution_mode,
        "implementation_commit": implementation_commit,
        "container_image": image,
        "packet_receipt_digest": packet_receipt["receipt_digest"],
        "arena_scene_plan_digest": packet_receipt["arena_scene_plan_digest"],
        "runtime_contract_digest": packet_receipt["runtime_contract_digest"],
        "scenario_instance_digest": packet_receipt["scenario_instance_digest"],
        "packet_files": packet_rows,
        "packet_file_count": len(packet_rows),
        "worker_source_sha256": _sha256(worker),
        "runtime_modules": module_rows,
        "runtime_source_packet": (
            {
                "receipt_digest": runtime_source_receipt["receipt_digest"],
                "packet_sha256": runtime_source_receipt["packet_sha256"],
                "packet_size_bytes": runtime_source_receipt["packet_size_bytes"],
                "install_roots": runtime_source_receipt["install_roots"],
                "runtime_dependency_wheels": runtime_source_receipt[
                    "runtime_dependency_wheels"
                ],
                "redistribution_permitted": runtime_source_receipt[
                    "redistribution_permitted"
                ],
            }
            if runtime_source_receipt is not None
            else None
        ),
        "runtime_entrypoint": "provider_runtime/run_adp_arena_provider_runtime.sh",
        "expected_output_filename": str(expected_output_filename),
        "policy_candidate_id": policy_candidate_id,
        "candidate_policy_queried": False,
        "candidate_outcomes_accessed": False,
        "packet_bytes_mutated": False,
        "scene_reconstructed_by_bundle": False,
        "native_application_claimed": False,
        "retry_cap": 0,
        "provider_zero_required_after_return": True,
        "blockers": [],
        "input_digest": "",
    }
    manifest["input_digest"] = canonical_digest(
        manifest, digest_field="input_digest"
    )
    write_json(runtime / "adp_arena_provider_manifest.json", manifest)
    bundle_path = job / "native_task_arena_provider_bundle.zip"
    with zipfile.ZipFile(bundle_path, "w", allowZip64=True) as archive:
        for path in sorted(runtime.rglob("*")):
            if not path.is_file():
                continue
            info = zipfile.ZipInfo(
                path.relative_to(job).as_posix(), date_time=(1980, 1, 1, 0, 0, 0)
            )
            info.create_system = 3
            info.external_attr = (path.stat().st_mode & 0xFFFF) << 16
            archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_STORED)
    receipt = {
        **manifest,
        "bundle_path": str(bundle_path),
        "bundle_size_bytes": bundle_path.stat().st_size,
        "bundle_sha256": _sha256(bundle_path),
    }
    write_json(job / "native_task_arena_provider_bundle_receipt.v1.json", receipt)
    return receipt


__all__ = [
    "DEFAULT_EXPECTED_OUTPUT_FILENAME",
    "NativeTaskArenaBundleError",
    "SCHEMA_VERSION",
    "build_native_task_arena_bundle",
]
