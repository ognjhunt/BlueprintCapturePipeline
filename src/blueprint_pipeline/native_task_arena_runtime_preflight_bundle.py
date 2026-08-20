"""Build and reverify the no-motion native Arena GPU preflight bundle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .native_task_arena_bundle import build_native_task_arena_bundle
from .native_task_arena_construction_bundle import construction_runtime_sources
from .native_task_isaaclab_launch import NATIVE_TASK_ARENA_IMAGE


PROBE_KIND = "native-task-arena-runtime-preflight"
RESULT_FILENAME = "native_task_arena_runtime_preflight.v1.json"
RESULT_SCHEMA_VERSION = "native_task_arena_runtime_preflight.v1"


def _sources() -> tuple[Path, ...]:
    package = Path(__file__).resolve().parent
    return (
        *construction_runtime_sources(),
        package / "native_task_arena_construction_worker.py",
    )


def build_native_task_arena_runtime_preflight_bundle(
    *,
    job_dir: str | Path,
    packet_dir: str | Path,
    runtime_source_packet_receipt: str | Path,
    implementation_commit: str,
    generated_at: str | None = None,
) -> dict[str, Any]:
    return build_native_task_arena_bundle(
        job_dir=job_dir,
        packet_dir=packet_dir,
        runtime_source_packet_receipt=runtime_source_packet_receipt,
        worker_source=(
            Path(__file__).resolve().parent
            / "native_task_arena_runtime_preflight_worker.py"
        ),
        runtime_module_sources=_sources(),
        implementation_commit=implementation_commit,
        execution_mode="runtime_preflight",
        expected_output_filename=RESULT_FILENAME,
        container_image=NATIVE_TASK_ARENA_IMAGE,
        generated_at=generated_at,
    )


def load_verified_native_task_arena_runtime_preflight_bundle(
    receipt_path: str | Path,
    *,
    expected_implementation_commit: str,
    expected_packet_receipt_digest: str | None = None,
    expected_runtime_source_packet_digest: str | None = None,
) -> dict[str, Any]:
    path = Path(receipt_path).expanduser().resolve()
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("native_task_arena_runtime_preflight_receipt_invalid") from exc
    bundle = Path(str(receipt.get("bundle_path") or "")).expanduser().resolve()
    manifest = {
        key: value
        for key, value in receipt.items()
        if key not in {"bundle_path", "bundle_size_bytes", "bundle_sha256"}
    }
    errors: list[str] = []
    if (
        receipt.get("schema_version") != "native_task_arena_provider_bundle.v1"
        or receipt.get("status") != "ready"
        or receipt.get("execution_mode") != "runtime_preflight"
        or receipt.get("expected_output_filename") != RESULT_FILENAME
        or receipt.get("policy_candidate_id") is not None
        or receipt.get("candidate_policy_queried") is not False
    ):
        errors.append("native_task_arena_runtime_preflight_contract_invalid")
    if receipt.get("implementation_commit") != expected_implementation_commit:
        errors.append("native_task_arena_runtime_preflight_commit_mismatch")
    if receipt.get("container_image") != NATIVE_TASK_ARENA_IMAGE:
        errors.append("native_task_arena_runtime_preflight_image_mismatch")
    if expected_packet_receipt_digest and (
        receipt.get("packet_receipt_digest") != expected_packet_receipt_digest
    ):
        errors.append("native_task_arena_runtime_preflight_packet_mismatch")
    source = receipt.get("runtime_source_packet") or {}
    if expected_runtime_source_packet_digest and (
        source.get("receipt_digest") != expected_runtime_source_packet_digest
    ):
        errors.append("native_task_arena_runtime_preflight_sources_mismatch")
    if receipt.get("input_digest") != canonical_digest(
        manifest, digest_field="input_digest"
    ):
        errors.append("native_task_arena_runtime_preflight_input_digest_invalid")
    digest = hashlib.sha256(bundle.read_bytes()).hexdigest() if bundle.is_file() else ""
    if (
        not bundle.is_file()
        or receipt.get("bundle_size_bytes") != bundle.stat().st_size
        or receipt.get("bundle_sha256") != "sha256:" + digest
    ):
        errors.append("native_task_arena_runtime_preflight_bundle_identity_invalid")
    if errors:
        raise ValueError(";".join(sorted(set(errors))))
    return receipt


__all__ = [
    "PROBE_KIND",
    "RESULT_FILENAME",
    "RESULT_SCHEMA_VERSION",
    "build_native_task_arena_runtime_preflight_bundle",
    "load_verified_native_task_arena_runtime_preflight_bundle",
]


def main(argv: list[str] | None = None) -> int:
    """Build the immutable no-motion preflight bundle; rents nothing."""

    import argparse

    parser = argparse.ArgumentParser(
        description="Build the immutable native Arena runtime preflight bundle."
    )
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--packet-dir", required=True)
    parser.add_argument("--runtime-source-packet-receipt", required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--generated-at")
    args = parser.parse_args(argv)
    try:
        receipt = build_native_task_arena_runtime_preflight_bundle(
            job_dir=args.job_dir,
            packet_dir=args.packet_dir,
            runtime_source_packet_receipt=args.runtime_source_packet_receipt,
            implementation_commit=args.implementation_commit,
            **({"generated_at": args.generated_at} if args.generated_at else {}),
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "blockers": [f"{type(exc).__name__}:{exc}"],
                    "provider_mutation_performed": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt.get("status") == "ready" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
