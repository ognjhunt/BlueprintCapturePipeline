"""Build the policy-free native Arena rigid-destination qualification bundle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .decision_evidence_contracts import canonical_digest
from .native_task_arena_bundle import build_native_task_arena_bundle
from .native_task_arena_execution_contract import (
    DESTINATION_QUALIFICATION_RUNTIME_MODULE_NAMES,
)
from .native_task_isaaclab_launch import NATIVE_TASK_ARENA_IMAGE
from .task_evaluation_rigid_destination_native_observation import (
    validate_rigid_destination_native_probe_request,
)


PROBE_KIND = "native-task-arena-destination-qualification"
RESULT_FILENAME = "task_evaluation_rigid_destination_native_observation.v1.json"
EXECUTION_MODE = "destination_qualification"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _load(path: Path, *, blocker: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(blocker) from exc
    if path.is_symlink() or not isinstance(value, Mapping):
        raise ValueError(blocker)
    return dict(value)


def destination_qualification_runtime_sources() -> tuple[Path, ...]:
    package = Path(__file__).resolve().parent
    return tuple(
        package / name for name in DESTINATION_QUALIFICATION_RUNTIME_MODULE_NAMES
    )


def build_native_task_arena_destination_qualification_bundle(
    *,
    job_dir: str | Path,
    packet_dir: str | Path,
    runtime_source_packet_receipt: str | Path,
    probe_request_path: str | Path,
    configured_scene_support_plane_path: str | Path,
    destination_static_qualification_path: str | Path,
    destination_native_import_qualification_path: str | Path,
    destination_geometry_path: str | Path,
    implementation_commit: str,
    container_image: str = NATIVE_TASK_ARENA_IMAGE,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Seal the exact packet, release, request, and prerequisite receipts."""

    packet = Path(packet_dir).expanduser().resolve()
    request_path = Path(probe_request_path).expanduser().resolve()
    support_plane_path = Path(configured_scene_support_plane_path).expanduser().resolve()
    static_path = Path(destination_static_qualification_path).expanduser().resolve()
    native_path = Path(destination_native_import_qualification_path).expanduser().resolve()
    geometry_path = Path(destination_geometry_path).expanduser().resolve()
    request = validate_rigid_destination_native_probe_request(
        _load(request_path, blocker="destination_qualification_probe_request_invalid")
    )
    packet_receipt = _load(
        packet / "native_task_arena_packet_receipt.v1.json",
        blocker="destination_qualification_packet_receipt_invalid",
    )
    source_bindings = packet_receipt.get("source_bindings")
    by_role = {
        str(row.get("semantic_role") or ""): row
        for row in source_bindings or []
        if isinstance(row, Mapping)
    }
    manifest_digest = "sha256:" + container_image.rsplit("@sha256:", 1)[-1]
    if (
        request["execution_commit"] != implementation_commit
        or request["container_identity"]
        != {"image": container_image, "digest": manifest_digest}
        or by_role.get("scene_collision", {}).get("staged_sha256")
        != request["configured_scene_collision_digest"]
        or by_role.get("task_support", {}).get("staged_sha256")
        != request["destination_asset_digest"]
        or _sha256(support_plane_path)
        != request["configured_scene_support_plane_digest"]
        or _sha256(static_path)
        != request["destination_static_qualification_digest"]
        or _sha256(native_path)
        != request["destination_native_import_qualification_digest"]
        or _load(
            geometry_path, blocker="destination_qualification_geometry_invalid"
        ).get("geometry_digest")
        != request["destination_geometry_digest"]
    ):
        raise ValueError("destination_qualification_input_binding_invalid")
    return build_native_task_arena_bundle(
        job_dir=job_dir,
        packet_dir=packet,
        runtime_source_packet_receipt=runtime_source_packet_receipt,
        worker_source=(
            Path(__file__).resolve().parent
            / "native_task_arena_destination_qualification_worker.py"
        ),
        runtime_module_sources=destination_qualification_runtime_sources(),
        implementation_commit=implementation_commit,
        execution_mode=EXECUTION_MODE,
        expected_output_filename=RESULT_FILENAME,
        container_image=container_image,
        bound_runtime_inputs={
            "rigid_destination_native_probe_request.v1.json": request_path,
            "configured_scene_support_plane.v1.json": support_plane_path,
            "destination_static_qualification.v1.json": static_path,
            "destination_native_import_qualification.v1.json": native_path,
            "destination_geometry.v1.json": geometry_path,
        },
        generated_at=generated_at,
    )


def load_verified_native_task_arena_destination_qualification_bundle(
    receipt_path: str | Path,
    *,
    expected_implementation_commit: str,
    expected_packet_receipt_digest: str | None = None,
    expected_runtime_source_packet_digest: str | None = None,
) -> dict[str, Any]:
    """Reverify the exact already-built bundle without mutating it."""

    receipt = _load(
        Path(receipt_path).expanduser().resolve(),
        blocker="destination_qualification_bundle_receipt_invalid",
    )
    bundle = Path(str(receipt.get("bundle_path") or "")).expanduser().resolve()
    manifest = {
        key: value
        for key, value in receipt.items()
        if key not in {"bundle_path", "bundle_size_bytes", "bundle_sha256"}
    }
    if (
        receipt.get("status") != "ready"
        or receipt.get("execution_mode") != EXECUTION_MODE
        or receipt.get("expected_output_filename") != RESULT_FILENAME
        or receipt.get("implementation_commit") != expected_implementation_commit
        or receipt.get("candidate_policy_queried") is not False
        or receipt.get("policy_candidate_id") is not None
        or (
            expected_packet_receipt_digest is not None
            and receipt.get("packet_receipt_digest")
            != expected_packet_receipt_digest
        )
        or (
            expected_runtime_source_packet_digest is not None
            and (receipt.get("runtime_source_packet") or {}).get("receipt_digest")
            != expected_runtime_source_packet_digest
        )
        or receipt.get("input_digest")
        != canonical_digest(manifest, digest_field="input_digest")
        or not bundle.is_file()
        or receipt.get("bundle_size_bytes") != bundle.stat().st_size
        or receipt.get("bundle_sha256") != _sha256(bundle)
    ):
        raise ValueError("destination_qualification_bundle_receipt_invalid")
    return receipt


__all__ = [
    "EXECUTION_MODE",
    "PROBE_KIND",
    "RESULT_FILENAME",
    "build_native_task_arena_destination_qualification_bundle",
    "destination_qualification_runtime_sources",
    "load_verified_native_task_arena_destination_qualification_bundle",
]


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--packet-dir", required=True)
    parser.add_argument("--runtime-source-packet-receipt", required=True)
    parser.add_argument("--probe-request", required=True)
    parser.add_argument("--configured-scene-support-plane", required=True)
    parser.add_argument("--destination-static-qualification", required=True)
    parser.add_argument("--destination-native-import-qualification", required=True)
    parser.add_argument("--destination-geometry", required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--container-image", default=NATIVE_TASK_ARENA_IMAGE)
    parser.add_argument("--generated-at")
    args = parser.parse_args(argv)
    receipt = build_native_task_arena_destination_qualification_bundle(
        job_dir=args.job_dir,
        packet_dir=args.packet_dir,
        runtime_source_packet_receipt=args.runtime_source_packet_receipt,
        probe_request_path=args.probe_request,
        configured_scene_support_plane_path=args.configured_scene_support_plane,
        destination_static_qualification_path=args.destination_static_qualification,
        destination_native_import_qualification_path=(
            args.destination_native_import_qualification
        ),
        destination_geometry_path=args.destination_geometry,
        implementation_commit=args.implementation_commit,
        container_image=args.container_image,
        **({"generated_at": args.generated_at} if args.generated_at else {}),
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - production CLI
    raise SystemExit(main())
