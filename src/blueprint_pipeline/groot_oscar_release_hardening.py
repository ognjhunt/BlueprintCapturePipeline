"""Hermetic contracts for the official GR00T+OSCAR release process."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")

# These are admission ceilings, not size targets.  They are deliberately just
# above the measured 2026-07-14 closure (47,101,357,226 total;
# 14,083,497,680 largest layer) so a release cannot silently grow while the
# layer-reduction work proceeds.  Reducing either constant is a closure change
# and must be accompanied by a fresh registry diagnostic.
MAX_IMAGE_COMPRESSED_BYTES = 48_000_000_000
MAX_IMAGE_LAYER_BYTES = 15_000_000_000
LARGE_IMAGE_PRELOAD_THRESHOLD_BYTES = 20 * 1024**3


@dataclass(frozen=True)
class DiskAdmission:
    available_bytes: int
    image_compressed_bytes: int
    image_unpacked_bytes: int
    build_scratch_multiplier: float = 1.25
    registry_scan_scratch_multiplier: float = 1.15
    reserve_bytes: int = 20 * 1024**3

    @property
    def required_bytes(self) -> int:
        return int(
            self.image_unpacked_bytes * self.build_scratch_multiplier
            + self.image_compressed_bytes * self.registry_scan_scratch_multiplier
            + self.reserve_bytes
        )

    def evidence(self) -> dict[str, Any]:
        passed = self.available_bytes >= self.required_bytes
        return {
            "schema_version": "groot_oscar_disk_admission.v1",
            "status": "passed" if passed else "blocked",
            "available_bytes": self.available_bytes,
            "required_bytes": self.required_bytes,
            "reserve_bytes": self.reserve_bytes,
            "scan_source": "registry_digest",
            "blockers": [] if passed else ["insufficient_disk_for_build_and_registry_scan"],
        }


def syft_registry_scan_command(digest_ref: str, output_path: str) -> list[str]:
    """Return a scan command that can never auto-select the Docker daemon."""

    name, separator, digest = digest_ref.rpartition("@")
    if not name or separator != "@" or not _DIGEST.fullmatch(digest):
        raise ValueError("registry_scan_requires_immutable_digest_ref")
    return ["syft", f"registry:{digest_ref}", "-o", f"spdx-json={output_path}"]


def validate_spdx_document(payload: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    if not str(payload.get("spdxVersion") or "").startswith("SPDX-"):
        blockers.append("spdx_version_missing")
    if not str(payload.get("documentNamespace") or "").strip():
        blockers.append("spdx_document_namespace_missing")
    if not isinstance(payload.get("packages"), list) or not payload.get("packages"):
        blockers.append("spdx_packages_missing")
    return blockers


def validate_provenance_digest(payload: Mapping[str, Any], expected_digest: str) -> list[str]:
    if not _DIGEST.fullmatch(expected_digest):
        return ["expected_digest_invalid"]
    serialized = json.dumps(payload, sort_keys=True)
    return [] if expected_digest in serialized else ["provenance_digest_mismatch"]


def validate_buildkit_provenance_binding(
    provenance_attestation: Mapping[str, Any],
    image_index: Mapping[str, Any],
    runnable_digest: str,
) -> list[str]:
    """Bind BuildKit's predicate to the runnable OCI manifest descriptor.

    ``imagetools inspect --format '{{json .Provenance}}'`` returns the SLSA
    predicate without an in-toto subject envelope. BuildKit records the subject
    binding on the OCI attestation-manifest descriptor instead, so both the
    actual predicate and that descriptor must validate together.
    """

    blockers: list[str] = []
    if not _DIGEST.fullmatch(runnable_digest):
        return ["buildkit_runnable_digest_invalid"]
    slsa = provenance_attestation.get("SLSA")
    if not isinstance(slsa, Mapping):
        blockers.append("buildkit_provenance_slsa_predicate_missing")
    else:
        if slsa.get("buildType") != "https://mobyproject.org/buildkit@v1":
            blockers.append("buildkit_provenance_build_type_invalid")
        invocation = slsa.get("invocation")
        invocation = invocation if isinstance(invocation, Mapping) else {}
        environment = invocation.get("environment")
        environment = environment if isinstance(environment, Mapping) else {}
        if environment.get("platform") != "linux/amd64":
            blockers.append("buildkit_provenance_platform_invalid")

    manifests = image_index.get("manifests")
    manifests = manifests if isinstance(manifests, list) else []
    runnable = [
        descriptor
        for descriptor in manifests
        if isinstance(descriptor, Mapping)
        and descriptor.get("digest") == runnable_digest
        and isinstance(descriptor.get("platform"), Mapping)
        and descriptor["platform"].get("os") == "linux"
        and descriptor["platform"].get("architecture") == "amd64"
    ]
    if len(runnable) != 1:
        blockers.append("buildkit_runnable_manifest_descriptor_mismatch")
    bound_attestations = [
        descriptor
        for descriptor in manifests
        if isinstance(descriptor, Mapping)
        and isinstance(descriptor.get("annotations"), Mapping)
        and descriptor["annotations"].get("vnd.docker.reference.type") == "attestation-manifest"
        and descriptor["annotations"].get("vnd.docker.reference.digest") == runnable_digest
    ]
    if not bound_attestations:
        blockers.append("buildkit_attestation_subject_binding_missing")
    return blockers


def _load_json_object(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected_json_object:{path}")
    return payload


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_thin_release_supply_chain(
    *,
    digest_ref: str,
    registry_diagnostic_path: str | Path,
    buildx_metadata_path: str | Path,
    spdx_path: str | Path,
    buildkit_sbom_path: str | Path,
    buildkit_provenance_path: str | Path,
    buildkit_index_path: str | Path,
    provenance_output_path: str | Path,
    layer_report_output_path: str | Path,
    manifest_output_path: str | Path,
) -> dict[str, Any]:
    """Validate and bind the canonical thin release's supply-chain evidence."""

    _, separator, expected_digest = digest_ref.rpartition("@")
    blockers: list[str] = []
    if separator != "@" or not _DIGEST.fullmatch(expected_digest):
        blockers.append("thin_release_digest_ref_invalid")

    registry = _load_json_object(registry_diagnostic_path)
    metadata = _load_json_object(buildx_metadata_path)
    spdx = _load_json_object(spdx_path)
    buildkit_sbom = _load_json_object(buildkit_sbom_path)
    buildkit_provenance = _load_json_object(buildkit_provenance_path)
    image_index = _load_json_object(buildkit_index_path)

    if registry.get("resolved_digest_ref") != digest_ref:
        blockers.append("registry_diagnostic_digest_ref_mismatch")
    metadata_digest = metadata.get("containerimage.digest")
    if metadata_digest is None and isinstance(metadata.get("containerimage.descriptor"), Mapping):
        metadata_digest = metadata["containerimage.descriptor"].get("digest")
    if metadata_digest != expected_digest:
        blockers.append("buildx_metadata_digest_mismatch")
    blockers.extend(validate_spdx_document(spdx))
    if not buildkit_sbom:
        blockers.append("buildkit_sbom_attestation_empty")
    runnable_digest = str(registry.get("runnable_child_digest") or "")
    blockers.extend(
        validate_buildkit_provenance_binding(
            buildkit_provenance,
            image_index,
            runnable_digest,
        )
    )

    layer_report = build_layer_report(registry.get("layers") or [])
    if layer_report["status"] != "passed":
        blockers.extend(layer_report["blockers"])
    Path(layer_report_output_path).write_text(
        json.dumps(layer_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    provenance = {
        "schema_version": "groot_oscar_thin_release_provenance.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "resolved_digest_ref": digest_ref,
        "resolved_digest": expected_digest,
        "runnable_child_digest": runnable_digest,
        "buildx_metadata_sha256": _file_sha256(buildx_metadata_path),
        "registry_diagnostic_sha256": _file_sha256(registry_diagnostic_path),
        "spdx_sha256": _file_sha256(spdx_path),
        "buildkit_sbom_attestation_sha256": _file_sha256(buildkit_sbom_path),
        "buildkit_provenance_attestation_sha256": _file_sha256(buildkit_provenance_path),
        "buildkit_attestation_index_sha256": _file_sha256(buildkit_index_path),
    }
    Path(provenance_output_path).write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    blockers.extend(validate_provenance_digest(provenance, expected_digest))

    manifest = {
        "schema_version": "groot_oscar_thin_release_supply_chain.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if not blockers else "blocked",
        "resolved_digest_ref": digest_ref,
        "scan_source": "registry_digest",
        "syft_daemon_export_allowed": False,
        "artifacts": {
            "spdx": {"path": str(spdx_path), "sha256": _file_sha256(spdx_path)},
            "provenance": {
                "path": str(provenance_output_path),
                "sha256": _file_sha256(provenance_output_path),
            },
            "layer_report": {
                "path": str(layer_report_output_path),
                "sha256": _file_sha256(layer_report_output_path),
            },
        },
        "blockers": sorted(set(blockers)),
    }
    Path(manifest_output_path).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def validate_registry_mirror_equivalence(
    source: Mapping[str, Any], mirror: Mapping[str, Any]
) -> dict[str, Any]:
    """Require identical per-platform content, not merely matching tags."""

    source_platforms = dict(source.get("platform_digests") or {})
    mirror_platforms = dict(mirror.get("platform_digests") or {})
    passed = bool(source_platforms) and source_platforms == mirror_platforms
    return {
        "schema_version": "registry_mirror_equivalence.v1",
        "status": "passed" if passed else "blocked",
        "source_manifest_digest": source.get("manifest_digest"),
        "mirror_manifest_digest": mirror.get("manifest_digest"),
        "platform_digests": source_platforms,
        "equivalent_platform_content": passed,
        "blockers": [] if passed else ["registry_platform_manifest_digest_mismatch"],
        "claim_boundary": "Registry equivalence is image-content proof, not runtime proof.",
    }


def build_regional_mirror_plan(
    *,
    source_digest_ref: str,
    project_id: str,
    repository: str,
    locations: Sequence[str],
    compressed_size_bytes: int,
    storage_usd_per_gb_month: float,
) -> dict[str, Any]:
    """Build a no-mutation mirror plan with explicit recurring-cost exposure."""

    name, separator, digest = source_digest_ref.rpartition("@")
    if not name or separator != "@" or not _DIGEST.fullmatch(digest):
        raise ValueError("regional_mirror_requires_immutable_source_digest")
    if not project_id or not repository or not locations:
        raise ValueError("regional_mirror_destination_incomplete")
    image_name = name.rsplit("/", 1)[-1].split(":", 1)[0]
    gib = compressed_size_bytes / 1024**3
    monthly = gib * storage_usd_per_gb_month * len(locations)
    destinations = []
    for location in locations:
        target = f"{location}-docker.pkg.dev/{project_id}/{repository}/{image_name}:{digest[7:19]}"
        destinations.append(
            {
                "location": location,
                "target": target,
                "copy_command": ["crane", "copy", source_digest_ref, target],
                "required_post_copy_gate": "registry_mirror_equivalence.v1",
            }
        )
    return {
        "schema_version": "groot_oscar_regional_mirror_plan.v1",
        "status": "planned_not_executed",
        "source_digest_ref": source_digest_ref,
        "destinations": destinations,
        "estimated_storage_usd_per_month": round(monthly, 4),
        "idle_paid_compute_required": False,
        "delete_policy": "retain_only_active_release_closures",
        "claim_boundary": "A mirror plan is not a copied or equivalent image.",
    }


def build_layer_report(
    layers: Sequence[Mapping[str, Any]],
    *,
    max_total_compressed_bytes: int = MAX_IMAGE_COMPRESSED_BYTES,
    max_layer_bytes: int = MAX_IMAGE_LAYER_BYTES,
) -> dict[str, Any]:
    def layer_size(layer: Mapping[str, Any]) -> int:
        # Registry diagnostics use OCI's `size`; internal callers historically
        # used `size_bytes`. Accept both without silently converting a present
        # registry layer to zero bytes.
        value = layer.get("size_bytes")
        if value is None:
            value = layer.get("size")
        return int(value or 0)

    normalized = [
        {
            "digest": str(layer.get("digest") or ""),
            "size_bytes": layer_size(layer),
            "created_by": str(layer.get("created_by") or ""),
        }
        for layer in layers
    ]
    normalized.sort(key=lambda row: row["size_bytes"], reverse=True)
    total = sum(row["size_bytes"] for row in normalized)
    largest = max((row["size_bytes"] for row in normalized), default=0)
    blockers: list[str] = []
    if not normalized or total <= 0:
        blockers.append("image_layer_inventory_empty")
    if total > max_total_compressed_bytes:
        blockers.append("image_total_compressed_size_budget_exceeded")
    if largest > max_layer_bytes:
        blockers.append("image_largest_layer_size_budget_exceeded")
    duplicate_digests = sorted(
        digest
        for digest in {row["digest"] for row in normalized if row["digest"]}
        if sum(row["digest"] == digest for row in normalized) > 1
    )
    categories = {
        "isaac_sim_base": 0,
        "oscar_python_cuda_runtime": 0,
        "wbc_build_and_cuda_toolchain": 0,
        "groot_python_cuda_runtime": 0,
        "sealed_model_checkpoints": 0,
        "other": 0,
    }
    for row in normalized:
        command = row["created_by"]
        if "COPY . /isaac-sim/" in command:
            category = "isaac_sim_base"
        elif "uv venv /opt/oscar-venv" in command:
            category = "oscar_python_cuda_runtime"
        elif "scripts/install_deps.sh" in command and "just build" in command:
            category = "wbc_build_and_cuda_toolchain"
        elif "uv venv /opt/gr00t-venv" in command:
            category = "groot_python_cuda_runtime"
        elif "snapshot_download" in command and "SONIC_CHECKPOINT_REPO" in command:
            category = "sealed_model_checkpoints"
        else:
            category = "other"
        categories[category] += row["size_bytes"]
    return {
        "schema_version": "groot_oscar_image_layer_report.v1",
        "status": "passed" if not blockers else "blocked",
        "layer_count": len(normalized),
        "total_compressed_size_bytes": total,
        "largest_layer_size_bytes": largest,
        "max_total_compressed_size_bytes": max_total_compressed_bytes,
        "max_layer_size_bytes": max_layer_bytes,
        "requires_preloaded_host": total >= LARGE_IMAGE_PRELOAD_THRESHOLD_BYTES,
        "largest_layers": normalized[:20],
        "duplicate_layer_digests": duplicate_digests,
        "compressed_bytes_by_build_role": categories,
        "measured_optimization_candidates": [
            {
                "role": "wbc_build_and_cuda_toolchain",
                "required_change": "build WBC in a separate stage and copy only runtime artifacts",
                "runtime_gpu_abi_test_required": True,
            },
            {
                "role": "oscar_and_groot_python_cuda_runtimes",
                "required_change": (
                    "deduplicate only byte-identical CUDA wheels; keep incompatible torch ABIs isolated"
                ),
                "runtime_gpu_abi_test_required": True,
            },
            {
                "role": "sealed_model_checkpoints",
                "required_change": (
                    "retain offline immutable blobs but preload them in the host image or regional cache"
                ),
                "runtime_gpu_abi_test_required": False,
            },
        ],
        "optimization_rules": {
            "remove_build_only_dependencies": True,
            "deduplicate_model_and_framework_files": True,
            "external_blobs_must_be_digest_pinned": True,
            "offline_execution_required": True,
            "hidden_runtime_downloads_forbidden": True,
        },
        "blockers": blockers,
    }


STARTUP_MILESTONES = (
    "vm_allocation",
    "driver_ready",
    "container_runtime_ready",
    "image_pull",
    "container_start",
    "health",
    "isaac_startup",
    "policy_ready",
    "first_simulator_step",
    "first_learned_action",
    "first_frame",
    "artifact_upload",
)


def record_startup_milestone(
    timing: Mapping[str, Any], milestone: str, elapsed_seconds: float
) -> dict[str, float]:
    """Append one monotonic startup milestone without overwriting evidence."""

    if milestone not in STARTUP_MILESTONES:
        raise ValueError(f"startup_milestone_unknown:{milestone}")
    if milestone in timing:
        raise ValueError(f"startup_milestone_duplicate:{milestone}")
    expected_index = len(timing)
    if expected_index >= len(STARTUP_MILESTONES) or STARTUP_MILESTONES[expected_index] != milestone:
        raise ValueError(f"startup_milestone_out_of_order:{milestone}")
    if (
        isinstance(elapsed_seconds, bool)
        or not isinstance(elapsed_seconds, (int, float))
        or not math.isfinite(float(elapsed_seconds))
        or float(elapsed_seconds) < 0
    ):
        raise ValueError(f"startup_milestone_time_invalid:{milestone}")
    normalized: dict[str, float] = {}
    for key, value in timing.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0
        ):
            raise ValueError(f"startup_milestone_existing_time_invalid:{key}")
        normalized[str(key)] = float(value)
    previous = max(normalized.values(), default=-1.0)
    if float(elapsed_seconds) < previous:
        raise ValueError(f"startup_milestone_time_reversed:{milestone}")
    return {
        **normalized,
        milestone: float(elapsed_seconds),
    }


def validate_startup_timing(timing: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    previous = -1.0
    for milestone in STARTUP_MILESTONES:
        value = timing.get(milestone)
        if value is None:
            blockers.append(f"startup_timing_missing:{milestone}")
            continue
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0
        ):
            blockers.append(f"startup_timing_invalid:{milestone}")
            continue
        if float(value) < previous:
            blockers.append(f"startup_timing_out_of_order:{milestone}")
        previous = float(value)
    return blockers


def evaluate_release_slos(
    timing: Mapping[str, Any], *, cached_worker_ready_target_seconds: int = 300
) -> dict[str, Any]:
    blockers = validate_startup_timing(timing)
    policy_ready = timing.get("policy_ready")
    if isinstance(policy_ready, (int, float)) and policy_ready > cached_worker_ready_target_seconds:
        blockers.append("cached_worker_ready_slo_missed")
    return {
        "schema_version": "groot_oscar_release_slo.v1",
        "status": "passed" if not blockers else "blocked",
        "cached_worker_ready_target_seconds": cached_worker_ready_target_seconds,
        "measured_cached_worker_ready_seconds": policy_ready,
        "failure_classification_target_seconds": 180,
        "opaque_waits_allowed": False,
        "closure_change_required_for_image_rebuild": True,
        "documentation_only_change_rebuilds_image": False,
        "blockers": blockers,
    }


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate-thin-release")
    validate.add_argument("--digest-ref", required=True)
    validate.add_argument("--registry-diagnostic", required=True)
    validate.add_argument("--buildx-metadata", required=True)
    validate.add_argument("--spdx", required=True)
    validate.add_argument("--buildkit-sbom", required=True)
    validate.add_argument("--buildkit-provenance", required=True)
    validate.add_argument("--buildkit-index", required=True)
    validate.add_argument("--provenance-output", required=True)
    validate.add_argument("--layer-report-output", required=True)
    validate.add_argument("--manifest-output", required=True)
    args = parser.parse_args(argv)
    result = validate_thin_release_supply_chain(
        digest_ref=args.digest_ref,
        registry_diagnostic_path=args.registry_diagnostic,
        buildx_metadata_path=args.buildx_metadata,
        spdx_path=args.spdx,
        buildkit_sbom_path=args.buildkit_sbom,
        buildkit_provenance_path=args.buildkit_provenance,
        buildkit_index_path=args.buildkit_index,
        provenance_output_path=args.provenance_output,
        layer_report_output_path=args.layer_report_output,
        manifest_output_path=args.manifest_output,
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(_main())
