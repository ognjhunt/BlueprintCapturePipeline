"""Build and reverify the no-motion native Arena GPU preflight bundle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
import zipfile

from .decision_evidence_contracts import canonical_digest
from .native_task_arena_bundle import build_native_task_arena_bundle, verify_native_task_arena_packet
from .native_task_arena_execution_contract import RUNTIME_PREFLIGHT_MODULE_NAMES
from .native_task_runtime_source_packet import verify_native_task_runtime_source_packet
from .native_task_isaaclab_launch import NATIVE_TASK_ARENA_IMAGE


PROBE_KIND = "native-task-arena-runtime-preflight"
RESULT_FILENAME = "native_task_arena_runtime_preflight.v1.json"
RESULT_SCHEMA_VERSION = "native_task_arena_runtime_preflight.v1"


def _sources() -> tuple[Path, ...]:
    package = Path(__file__).resolve().parent
    return tuple(package / name for name in RUNTIME_PREFLIGHT_MODULE_NAMES)


def build_native_task_arena_runtime_preflight_bundle(
    *,
    job_dir: str | Path,
    packet_dir: str | Path,
    runtime_source_packet_receipt: str | Path,
    implementation_commit: str,
    generated_at: str | None = None,
    composition_request_path: str | Path | None = None,
    composition_scene_plan_path: str | Path | None = None,
) -> dict[str, Any]:
    source = verify_native_task_runtime_source_packet(runtime_source_packet_receipt)
    image = _profile_image(source)
    package = Path(__file__).resolve().parent
    worker = package / "native_task_arena_runtime_preflight_worker.py"
    modules = _sources()
    bound_inputs = {}
    if composition_request_path is not None or composition_scene_plan_path is not None:
        from .native_task_composition_diagnostic import PASSES, validate_request, verify_record

        if composition_request_path is None or composition_scene_plan_path is None:
            raise ValueError("composition_request_and_scene_plan_required")
        request_path = Path(composition_request_path)
        plan_path = Path(composition_scene_plan_path)
        request = validate_request(json.loads(request_path.read_text()))
        verify_record(plan_path, request["resolved_scene_plan"])
        plan = json.loads(plan_path.read_text())
        _, packet, _ = verify_native_task_arena_packet(packet_dir)
        if (request["passes"] != list(PASSES)
                or request.get("require_composition_gate") not in (None, False)
                or plan.get("plan_digest") != canonical_digest(plan, digest_field="plan_digest")
                or plan["plan_digest"] != request.get("resolved_scene_plan_digest")
                or plan["scenario"]["seed"] != request.get("seed")
                or packet["receipt_digest"] != request.get("packet_receipt_digest")):
            raise ValueError("composition_plan_binding_invalid")
        worker = package / "native_task_composition_worker.py"
        # Reuse the already-qualified AOV writer's complete import closure.
        # No policy provisioning or checkpoint bytes are added.
        bound_inputs = {"composition_request.json": request_path,
                        "composition_scene_plan.json": plan_path}
    return build_native_task_arena_bundle(
        job_dir=job_dir,
        packet_dir=packet_dir,
        runtime_source_packet_receipt=runtime_source_packet_receipt,
        worker_source=worker,
        runtime_module_sources=modules,
        implementation_commit=implementation_commit,
        execution_mode="runtime_preflight",
        expected_output_filename=RESULT_FILENAME,
        container_image=image,
        bound_runtime_inputs=bound_inputs,
        generated_at=generated_at,
    )


def _profile_image(source: dict[str, Any]) -> str:
    if source.get("runtime_profile") == "isaaclab_3_ga":
        from .native_task_isaaclab_ga import IMAGE
        if source.get("container_image") != IMAGE:
            raise ValueError("ga_runtime_image_source_mismatch")
        return IMAGE
    return NATIVE_TASK_ARENA_IMAGE


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
    try:
        with zipfile.ZipFile(bundle) as archive:
            source_path = "provider_runtime/native_task_runtime_sources/native_task_runtime_source_packet.v1.json"
            if archive.getinfo(source_path).file_size > 4_000_000:
                raise ValueError("native_task_arena_runtime_preflight_source_receipt_invalid")
            embedded_source = json.loads(archive.read(source_path))
    except (OSError, zipfile.BadZipFile, KeyError, json.JSONDecodeError) as exc:
        raise ValueError("native_task_arena_runtime_preflight_source_receipt_invalid") from exc
    if (not isinstance(embedded_source, dict)
            or embedded_source.get("receipt_digest") != source.get("receipt_digest")
            or embedded_source.get("receipt_digest") != canonical_digest(embedded_source, digest_field="receipt_digest")
            or receipt.get("container_image") != _profile_image(embedded_source)):
        raise ValueError("native_task_arena_runtime_preflight_image_source_mismatch")
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
    parser.add_argument("--composition-request")
    parser.add_argument("--composition-scene-plan")
    args = parser.parse_args(argv)
    try:
        receipt = build_native_task_arena_runtime_preflight_bundle(
            job_dir=args.job_dir,
            packet_dir=args.packet_dir,
            runtime_source_packet_receipt=args.runtime_source_packet_receipt,
            implementation_commit=args.implementation_commit,
            composition_request_path=args.composition_request,
            composition_scene_plan_path=args.composition_scene_plan,
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
