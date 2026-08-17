"""Build one deterministic semantic-teacher image-edit provider bundle.

The archive contains only packet-derived calibrated PNGs, exact edit masks,
registry and human-rights provenance, an immutable runtime request, and the
transport worker. It excludes publisher raw bytes and stores no credentials.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from collections.abc import Mapping
import re
import shutil
import stat
import subprocess
from typing import Any, Sequence
import zipfile

from .decision_evidence_contracts import canonical_digest, canonical_json
from .fresh_scene_semantic_teacher_image_edit import (
    PACKET_SCHEMA_VERSION,
    RIGHTS_SCHEMA_VERSION,
)
from .image_editor_backend_registry import (
    REGISTRY_SCHEMA_VERSION,
    SEMANTIC_TEACHER_IMAGE_EDIT_CAPABILITY,
    ImageEditorRegistryError,
    load_registry,
)
from .provider_bundle_rehearsal import rehearse_provider_bundle_entrypoint
from .semantic_teacher_image_edit_worker import (
    MAX_PARALLEL_REQUESTS,
    PRODUCTION_MAX_PARALLEL_REQUESTS,
    RUNTIME_REQUEST_SCHEMA_VERSION,
    USAGE_TOKEN_FIELDS,
)


BUNDLE_RECEIPT_SCHEMA_VERSION = "semantic_teacher_image_edit_provider_bundle.v1"
MANIFEST_SCHEMA_VERSION = "semantic_teacher_image_edit_provider_manifest.v1"
ENTRYPOINT = "provider_runtime/run_semantic_teacher_image_edit.sh"
WORKER = "provider_runtime/semantic_teacher_image_edit_worker.py"
RUNTIME_REQUEST = "provider_runtime/semantic_teacher_image_edit_runtime_request.v1.json"
MANIFEST = "provider_runtime/semantic_teacher_image_edit_provider_manifest.v1.json"
MAX_TASKS = 5
MAX_CAMERAS_PER_TASK = 256
MAX_MEMBER_COUNT = 3_000
MAX_MEMBER_BYTES = 128 * 1024 * 1024
MAX_TOTAL_BYTES = 2 * 1024 * 1024 * 1024


class SemanticTeacherImageEditBundleError(ValueError):
    """The provider bundle or one transitive input failed validation."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read(path: Path, *, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SemanticTeacherImageEditBundleError(code) from exc
    if path.is_symlink() or not isinstance(value, dict):
        raise SemanticTeacherImageEditBundleError(code)
    return value


def _record(path: Path, *, root: Path | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }
    result["relative_path" if root is not None else "path"] = (
        path.relative_to(root).as_posix() if root is not None else str(path)
    )
    return result


def _bound_absolute(record: Any, *, code: str) -> Path:
    if not isinstance(record, Mapping):
        raise SemanticTeacherImageEditBundleError(code)
    path = Path(str(record.get("path") or "")).expanduser().resolve()
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != record.get("size_bytes")
        or _sha256(path) != record.get("sha256")
    ):
        raise SemanticTeacherImageEditBundleError(code)
    return path


def _bound_relative(root: Path, record: Any, *, code: str) -> Path:
    if not isinstance(record, Mapping):
        raise SemanticTeacherImageEditBundleError(code)
    relative = Path(str(record.get("relative_path") or ""))
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise SemanticTeacherImageEditBundleError(code)
    path = (root / relative).resolve()
    if root != path and root not in path.parents:
        raise SemanticTeacherImageEditBundleError(code)
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != record.get("size_bytes")
        or _sha256(path) != record.get("sha256")
    ):
        raise SemanticTeacherImageEditBundleError(code)
    return path


def _copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if _sha256(source) != _sha256(destination):
        raise SemanticTeacherImageEditBundleError(
            "semantic_teacher_bundle_copy_digest_mismatch"
        )


def _zip_tree(source: Path, destination: Path) -> None:
    count = 0
    total = 0
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for item in sorted(source.rglob("*")):
            if item.is_dir() and not item.is_symlink():
                continue
            if item.is_symlink() or not item.is_file():
                raise SemanticTeacherImageEditBundleError(
                    "semantic_teacher_bundle_member_invalid"
                )
            count += 1
            size = item.stat().st_size
            total += size
            if (
                count > MAX_MEMBER_COUNT
                or size > MAX_MEMBER_BYTES
                or total > MAX_TOTAL_BYTES
            ):
                raise SemanticTeacherImageEditBundleError(
                    "semantic_teacher_bundle_size_limit_exceeded"
                )
            info = zipfile.ZipInfo(
                item.relative_to(source).as_posix(), date_time=(1980, 1, 1, 0, 0, 0)
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            with item.open("rb") as input_stream, archive.open(info, "w") as output:
                shutil.copyfileobj(input_stream, output, length=1024 * 1024)


def _repository_identity(repository_root: Path, worker_source: Path) -> str:
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        relative = worker_source.relative_to(repository_root).as_posix()
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", relative],
            cwd=repository_root,
            check=True,
            capture_output=True,
            timeout=10,
        )
        committed_worker = subprocess.run(
            ["git", "show", f"HEAD:{relative}"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise SemanticTeacherImageEditBundleError(
            "semantic_teacher_bundle_repository_identity_invalid"
        ) from exc
    if status.stdout or committed_worker != worker_source.read_bytes():
        raise SemanticTeacherImageEditBundleError(
            "semantic_teacher_bundle_repository_not_clean_committed"
        )
    return head.stdout.strip()


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def _safe_component(value: Any) -> str:
    component = str(value or "")
    if (
        not component
        or component != component.strip()
        or not component.isprintable()
        or component in {".", ".."}
        or "/" in component
        or "\\" in component
    ):
        return ""
    return component


def build_semantic_teacher_image_edit_provider_bundle(
    *,
    packet_path: str | Path,
    repository_root: str | Path,
    expected_source_commit: str,
    output_root: str | Path,
    max_parallel_requests: int = PRODUCTION_MAX_PARALLEL_REQUESTS,
) -> dict[str, Any]:
    """Build and zero-cost rehearse the exact hosted-editor archive."""

    packet_file = Path(packet_path).expanduser().resolve()
    packet = _read(packet_file, code="semantic_teacher_bundle_packet_unreadable")
    tasks = packet.get("tasks")
    if (
        isinstance(max_parallel_requests, bool)
        or not isinstance(max_parallel_requests, int)
        or not 1 <= max_parallel_requests <= MAX_PARALLEL_REQUESTS
        or packet.get("schema_version") != PACKET_SCHEMA_VERSION
        or packet.get("status")
        != "semantic_teacher_image_edit_packet_prepared_no_upload_no_execution"
        or packet.get("packet_digest")
        != canonical_digest(packet, digest_field="packet_digest")
        or not isinstance(tasks, list)
        or not 1 <= len(tasks) <= MAX_TASKS
        or packet.get("task_count") != len(tasks)
        or packet.get("retry_count") != 0
        or packet.get("provider_mutations_performed") != 0
        or packet.get("raw_nonredistributable_source_bytes_included") is not False
        or packet.get("private_derived_calibrated_frames_included") is not True
    ):
        raise SemanticTeacherImageEditBundleError(
            "semantic_teacher_bundle_packet_invalid"
        )
    repository = Path(repository_root).expanduser().resolve()
    worker_source = repository / "src/blueprint_pipeline/semantic_teacher_image_edit_worker.py"
    if (
        repository.is_symlink()
        or not worker_source.is_file()
        or len(expected_source_commit) != 40
        or _repository_identity(repository, worker_source) != expected_source_commit
    ):
        raise SemanticTeacherImageEditBundleError(
            "semantic_teacher_bundle_source_commit_invalid"
        )
    registry_path = _bound_absolute(
        packet.get("backend_registry"), code="semantic_teacher_bundle_registry_unbound"
    )
    rights_path = _bound_absolute(
        packet.get("rights_attestation"), code="semantic_teacher_bundle_rights_unbound"
    )
    backend = packet.get("backend")
    if not isinstance(backend, Mapping) or not isinstance(backend.get("execution"), Mapping):
        raise SemanticTeacherImageEditBundleError(
            "semantic_teacher_bundle_backend_invalid"
        )
    execution = dict(backend["execution"])
    pricing = execution.get("pricing_binding")
    runtime_image_identity = execution.get("runtime_image_identity")
    supported_output_sizes = execution.get("supported_output_sizes")
    usage_rates = (
        pricing.get("usd_per_million_tokens") if isinstance(pricing, Mapping) else None
    )
    backend_entry = backend.get("registry_entry")
    backend_digest = backend.get("backend_entry_digest")
    registry_payload = _read(
        registry_path, code="semantic_teacher_bundle_registry_invalid"
    )
    try:
        registry = load_registry(registry_path)
    except ImageEditorRegistryError as exc:
        raise SemanticTeacherImageEditBundleError(
            "semantic_teacher_bundle_registry_invalid"
        ) from exc
    backend_id = (
        str(backend_entry.get("backend_id") or "")
        if isinstance(backend_entry, Mapping)
        else ""
    )
    if (
        registry_payload.get("schema_version") != REGISTRY_SCHEMA_VERSION
        or not isinstance(backend_entry, Mapping)
        or backend_entry.get("capability")
        != SEMANTIC_TEACHER_IMAGE_EDIT_CAPABILITY
        or backend_id not in registry
        or registry[backend_id] != dict(backend_entry)
        or registry[backend_id].get("execution") != execution
        or backend_digest != canonical_digest(backend_entry)
        or backend_digest
        != (packet.get("backend_registry") or {}).get("selected_backend_entry_digest")
        or execution.get("adapter_id") != "openai_images_edits_v1"
        or execution.get("transport_kind") != "hosted_image_edit"
        or not isinstance(runtime_image_identity, str)
        or re.fullmatch(r"\S+@sha256:[0-9a-f]{64}", runtime_image_identity) is None
        or not isinstance(pricing, Mapping)
        or not isinstance(supported_output_sizes, list)
        or not supported_output_sizes
        or any(
            not isinstance(size, str)
            or re.fullmatch(r"[1-9][0-9]*x[1-9][0-9]*", size) is None
            for size in supported_output_sizes
        )
        or not isinstance(pricing.get("usage_required"), bool)
        or not isinstance(usage_rates, Mapping)
        or any(
            isinstance(usage_rates.get(field, 0), bool)
            or not isinstance(usage_rates.get(field, 0), (int, float))
            or not math.isfinite(float(usage_rates.get(field, 0)))
            or float(usage_rates.get(field, 0)) < 0
            for field in USAGE_TOKEN_FIELDS
        )
        or pricing.get("kind") not in {"immutable_registry_rate", "fresh_quote"}
        or pricing.get("billing_unit") != "per_request"
        or pricing.get("currency") != "USD"
        or not str(pricing.get("pricing_identity") or "").strip()
        or isinstance(pricing.get("max_cost_per_request_usd"), bool)
        or not isinstance(pricing.get("max_cost_per_request_usd"), (int, float))
        or not math.isfinite(float(pricing["max_cost_per_request_usd"]))
        or float(pricing["max_cost_per_request_usd"]) <= 0
        or (
            pricing.get("kind") == "fresh_quote"
            and (
                not str(pricing.get("quote_id") or "").strip()
                or pricing.get("quote_digest")
                != canonical_digest(pricing, digest_field="quote_digest")
            )
        )
    ):
        raise SemanticTeacherImageEditBundleError(
            "semantic_teacher_bundle_backend_invalid"
        )
    rights = _read(rights_path, code="semantic_teacher_bundle_rights_invalid")
    rights_external = execution.get("external_disclosure_required") is True
    if (
        rights.get("schema_version") != RIGHTS_SCHEMA_VERSION
        or rights.get("status") != "accepted_for_private_derived_semantic_edit"
        or rights.get("attestation_digest")
        != canonical_digest(rights, digest_field="attestation_digest")
        or rights.get("attestation_digest")
        != (packet.get("rights_attestation") or {}).get("attestation_digest")
        or rights.get("backend_id") != backend_id
        or rights.get("backend_entry_digest") != backend_digest
        or rights.get("provider_id") != execution.get("provider_id")
        or rights.get("model_snapshot") != execution.get("model_snapshot")
        or rights.get("raw_nonredistributable_source_bytes_included") is not False
        or rights.get("issued_by_agent") is not False
        or not str(rights.get("accepted_by") or "").strip()
        or not str(rights.get("accepted_on") or "").strip()
        or (
            rights_external
            and (
                rights.get("private_derived_frame_disclosure_authorized") is not True
                or rights.get("provider_retention_terms_accepted") is not True
                or rights.get("provider_training_terms_accepted") is not True
            )
        )
        or (
            not rights_external
            and rights.get("local_private_derived_use_authorized") is not True
        )
    ):
        raise SemanticTeacherImageEditBundleError(
            "semantic_teacher_bundle_rights_invalid"
        )
    output = Path(output_root).expanduser().resolve()
    if output.is_symlink() or (output.exists() and any(output.iterdir())):
        raise SemanticTeacherImageEditBundleError(
            "semantic_teacher_bundle_output_not_empty"
        )
    output.mkdir(parents=True, exist_ok=True)
    stage = output / "stage"
    runtime = stage / "provider_runtime"
    input_root = runtime / "input"
    runtime.mkdir(parents=True)
    _copy(worker_source, runtime / Path(WORKER).name)
    _copy(registry_path, input_root / "image_editor_backend_registry.json")
    _copy(rights_path, input_root / "human_rights_attestation.json")
    packet_root = packet_file.parent
    runtime_tasks: list[dict[str, Any]] = []
    task_ids: set[str] = set()
    total_cameras = 0
    for task in tasks:
        task_id = _safe_component(
            task.get("task_id") if isinstance(task, Mapping) else None
        )
        frames = task.get("frames") if isinstance(task, Mapping) else None
        if (
            not task_id
            or task_id in task_ids
            or not isinstance(frames, list)
            or not 1 <= len(frames) <= MAX_CAMERAS_PER_TASK
            or task.get("camera_count") != len(frames)
        ):
            raise SemanticTeacherImageEditBundleError(
                "semantic_teacher_bundle_task_set_invalid"
            )
        task_ids.add(task_id)
        runtime_frames: list[dict[str, Any]] = []
        camera_ids: set[str] = set()
        for index, frame in enumerate(frames):
            camera_id = _safe_component(
                frame.get("camera_id") if isinstance(frame, Mapping) else None
            )
            if (
                not camera_id
                or camera_id in camera_ids
                or frame.get("frame_index") != index
                or not isinstance(frame.get("width"), int)
                or not isinstance(frame.get("height"), int)
                or f"{frame.get('width')}x{frame.get('height')}"
                not in supported_output_sizes
            ):
                raise SemanticTeacherImageEditBundleError(
                    "semantic_teacher_bundle_camera_set_invalid"
                )
            camera_ids.add(camera_id)
            source = _bound_relative(
                packet_root,
                frame.get("staged_input_rgb"),
                code="semantic_teacher_bundle_source_frame_unbound",
            )
            mask = _bound_relative(
                packet_root,
                frame.get("staged_edit_mask"),
                code="semantic_teacher_bundle_edit_mask_unbound",
            )
            source_target = input_root / "tasks" / task_id / "input_frames" / f"{index:05d}.png"
            mask_target = input_root / "tasks" / task_id / "edit_masks" / f"{index:05d}.png"
            _copy(source, source_target)
            _copy(mask, mask_target)
            runtime_frames.append(
                {
                    "frame_index": index,
                    "camera_id": camera_id,
                    "input_rgb": _record(source_target, root=runtime),
                    "edit_mask": _record(mask_target, root=runtime),
                }
            )
        total_cameras += len(runtime_frames)
        runtime_tasks.append({"task_id": task_id, "frames": runtime_frames})
    if packet.get("request_count") != total_cameras:
        raise SemanticTeacherImageEditBundleError(
            "semantic_teacher_bundle_camera_count_invalid"
        )
    runtime_request: dict[str, Any] = {
        "schema_version": RUNTIME_REQUEST_SCHEMA_VERSION,
        "source_commit_sha": expected_source_commit,
        "source_packet_digest": packet["packet_digest"],
        "backend": {
            "registry_entry": dict(backend_entry),
            "backend_entry_digest": backend_digest,
            "execution": execution,
        },
        "prompt_policy": backend.get("prompt_policy"),
        "prompt": backend.get("prompt"),
        "tasks": runtime_tasks,
        "max_parallel_requests": max_parallel_requests,
        "retry_count": 0,
        "request_digest": "",
    }
    runtime_request["request_digest"] = canonical_digest(
        runtime_request, digest_field="request_digest"
    )
    _write(runtime / Path(RUNTIME_REQUEST).name, runtime_request)
    provenance = {
        "schema_version": "semantic_teacher_image_edit_bundle_provenance.v1",
        "source_packet_digest": packet["packet_digest"],
        "backend_registry": _record(input_root / "image_editor_backend_registry.json", root=runtime),
        "backend_entry_digest": backend_digest,
        "pricing_binding_digest": canonical_digest(pricing),
        "maximum_cost_per_request_usd": pricing["max_cost_per_request_usd"],
        "model_snapshot": execution["model_snapshot"],
        "worker_sha256": _sha256(runtime / Path(WORKER).name),
        "runtime_image_identity": runtime_image_identity,
        "rights_attestation": _record(input_root / "human_rights_attestation.json", root=runtime),
        "rights_attestation_digest": (packet.get("rights_attestation") or {}).get(
            "attestation_digest"
        ),
        "raw_nonredistributable_source_bytes_included": False,
        "private_derived_inputs_only": True,
    }
    _write(input_root / "semantic_teacher_provenance.json", provenance)
    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "classification": "private_derived_semantic_teacher_image_edit",
        "source_commit_sha": expected_source_commit,
        "runtime_request_digest": runtime_request["request_digest"],
        "backend_entry_digest": backend_digest,
        "pricing_binding_digest": canonical_digest(pricing),
        "maximum_cost_per_request_usd": pricing["max_cost_per_request_usd"],
        "model_snapshot": execution["model_snapshot"],
        "adapter_id": execution["adapter_id"],
        "runtime_image_identity": runtime_image_identity,
        "task_count": len(runtime_tasks),
        "camera_count": total_cameras,
        "automatic_retry_count": 0,
        "maximum_parallel_requests": max_parallel_requests,
        "entrypoint": ENTRYPOINT,
        "worker": _record(runtime / Path(WORKER).name, root=stage),
        "environment": {
            "required_secret_names": ["BLUEPRINT_IMAGE_EDITOR_TOKEN"],
            "secret_values_stored": False,
            "output_root_variable": "BLUEPRINT_SEMANTIC_TEACHER_OUTPUT_DIR",
        },
        "output_allowlist": [
            "tasks/*/*.png",
            "semantic_teacher_image_edit_runtime_result.v1.json",
            "runtime_stdout.log",
            "runtime_stderr.log",
            "billing_receipt.json",
            "object_store_cleanup.json",
            "independent_watchdog.json",
            "secret_redaction_receipt.json",
            "provider_zero_receipt.json",
        ],
        "raw_nonredistributable_source_bytes_included": False,
        "canonical_interiorgs_included_or_mutated": False,
        "manifest_digest": "",
    }
    manifest["manifest_digest"] = canonical_digest(manifest, digest_field="manifest_digest")
    _write(runtime / Path(MANIFEST).name, manifest)
    entrypoint = runtime / Path(ENTRYPOINT).name
    entrypoint.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
umask 077
runtime_dir="$(cd "$(dirname "$0")" && pwd)"
output_dir="${BLUEPRINT_SEMANTIC_TEACHER_OUTPUT_DIR:-${BLUEPRINT_ADP_CONTENT_AGENTS_OUTPUT_DIR:-}}"
if [[ -z "$output_dir" ]]; then
  echo "semantic_teacher_output_dir_missing" >&2
  exit 2
fi
mkdir -p "$output_dir"
if [[ "${BLUEPRINT_PROVIDER_BUNDLE_REHEARSAL:-0}" == "1" ]]; then
  printf '%s\n' '{"gpu_runtime_started":false,"paid_inference_performed":false,"provider_mutations_performed":0,"status":"passed","token_lookup_performed":false,"upload_performed":false}' > "$output_dir/provider_bundle_rehearsal.json"
  exit 0
fi
if [[ -z "${BLUEPRINT_IMAGE_EDITOR_TOKEN:-}" ]]; then
  echo "semantic_teacher_token_missing" >&2
  exit 3
fi
exec python "$runtime_dir/semantic_teacher_image_edit_worker.py" \
  --runtime-request "$runtime_dir/semantic_teacher_image_edit_runtime_request.v1.json" \
  --output-root "$output_dir" \
  --token-env BLUEPRINT_IMAGE_EDITOR_TOKEN
""",
        encoding="utf-8",
    )
    entrypoint.chmod(0o755)
    bundle = output / "semantic_teacher_image_edit_provider_bundle.zip"
    _zip_tree(stage, bundle)
    rehearsal = rehearse_provider_bundle_entrypoint(
        bundle_path=bundle,
        entrypoint_relative_path=ENTRYPOINT,
        evidence_path=output / "provider_bundle_rehearsal.json",
    )
    receipt: dict[str, Any] = {
        "schema_version": BUNDLE_RECEIPT_SCHEMA_VERSION,
        "status": "completed_no_upload_no_inference",
        "source_commit_sha": expected_source_commit,
        "bundle": _record(bundle),
        "manifest_digest": manifest["manifest_digest"],
        "runtime_request_digest": runtime_request["request_digest"],
        "backend_entry_digest": backend_digest,
        "pricing_binding_digest": canonical_digest(pricing),
        "maximum_cost_per_request_usd": pricing["max_cost_per_request_usd"],
        "model_snapshot": execution["model_snapshot"],
        "runtime_image_identity": runtime_image_identity,
        "worker_image_digest": runtime_image_identity,
        "worker_source_sha256": manifest["worker"]["sha256"],
        "adapter_id": execution["adapter_id"],
        "task_count": len(runtime_tasks),
        "camera_count": total_cameras,
        "maximum_parallel_requests": max_parallel_requests,
        "rehearsal": rehearsal,
        "provider_mutations_performed": 0,
        "secret_values_stored": False,
        "raw_nonredistributable_source_bytes_included": False,
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    _write(output / f"{BUNDLE_RECEIPT_SCHEMA_VERSION}.json", receipt)
    shutil.rmtree(stage)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--max-parallel-requests",
        type=int,
        choices=range(1, MAX_PARALLEL_REQUESTS + 1),
        default=PRODUCTION_MAX_PARALLEL_REQUESTS,
    )
    args = parser.parse_args(argv)
    receipt = build_semantic_teacher_image_edit_provider_bundle(
        packet_path=args.packet,
        repository_root=args.repository_root,
        expected_source_commit=args.expected_source_commit,
        output_root=args.output_root,
        max_parallel_requests=args.max_parallel_requests,
    )
    print(canonical_json(receipt))
    return 0


__all__ = [
    "BUNDLE_RECEIPT_SCHEMA_VERSION",
    "ENTRYPOINT",
    "MANIFEST_SCHEMA_VERSION",
    "SemanticTeacherImageEditBundleError",
    "build_semantic_teacher_image_edit_provider_bundle",
    "main",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
