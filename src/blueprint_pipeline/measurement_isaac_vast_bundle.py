"""Compile a clean-commit, deterministic Isaac/PhysX measurement input bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import stat
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence
import zipfile

from .decision_evidence_contracts import canonical_digest
from .measurement_adapter_execution import build_measurement_adapter_execution_request
from .measurement_adapter_runtime import build_measurement_adapter_descriptor
from .measurement_geometry_contact_development_suite import _file_digest, _load_corpus
from .measurement_isaac_physx_rigid_adapter import (
    IMPLEMENTATION_ID,
    IMPLEMENTATION_VERSION,
    PROTOCOL_ID,
    implementation_digest,
)
from .measurement_isaac_runtime_release import (
    ISAAC_VERSION,
    RUNTIME_IMAGE,
    build_measurement_isaac_runtime_release,
)
from .measurement_qualification_benchmarks import (
    build_benchmark_case_manifest,
    build_qualification_benchmark_spec,
)


BUNDLE_SCHEMA_VERSION = "measurement_isaac_physx_rtx_input_bundle.v3"
RECEIPT_SCHEMA_VERSION = "measurement_isaac_physx_rtx_input_bundle_receipt.v3"
RTX_REQUIRED_OUTPUT_KINDS = ("rgb", "depth", "semantic_segmentation")
RUNNER_RELATIVE_PATH = Path("scripts/run_measurement_isaac_physx_bundle.py")
WORKER_RELATIVE_PATH = Path("scripts/measurement_isaac_physx_rigid_worker.py")


class MeasurementIsaacVastBundleError(ValueError):
    pass


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _git_identity(root: Path) -> str:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if head.returncode != 0 or status.returncode != 0 or status.stdout.strip():
        raise MeasurementIsaacVastBundleError("measurement_isaac_source_not_clean_commit")
    commit = head.stdout.strip().lower()
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise MeasurementIsaacVastBundleError("measurement_isaac_source_commit_invalid")
    return commit


def _requests(
    corpus_path: Path,
    *,
    qualification_split_digest: str,
    controller_scope_digest: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    corpus = _load_corpus(corpus_path)
    corpus_digest = _file_digest(corpus_path)
    spec = build_qualification_benchmark_spec(
        benchmark_id="capture-to-geometry-and-contact",
        benchmark_version="development-isaac-physx-tgs-rigid-contact-1",
        method_ids=["isaac-sim-6-physx"],
        development_split_digest=corpus_digest,
        qualification_split_digest=qualification_split_digest,
        capture_bundle_digests=[corpus_digest],
        robot_controller_digests=[controller_scope_digest],
        acceptance_thresholds={
            "maximum_mean_absolute_error": 1.0,
            "maximum_mismatch_rate": 0.0,
            "maximum_harmful_false_negative_rate": 0.0,
            "minimum_coverage": 2 / 9,
        },
        compute_budget={"usd": 0.0, "maximum_duration_seconds": 900},
        minimum_repeated_trials=2,
    )
    descriptor = build_measurement_adapter_descriptor("isaac-sim-6-physx")
    requests = []
    for index, raw_case in enumerate(corpus["cases"]):
        row = dict(raw_case)
        case_id = str(row.pop("case_id"))
        case = build_benchmark_case_manifest(
            spec,
            case_id=f"{case_id}--isaac-sim-6-physx",
            split="development",
            input_artifact_digests=[corpus_digest],
            task_class="rigid_pick_place",
            material_regime="synthetic_rigid_body_drop",
            operating_point={
                **dict(corpus["shared_operating_point"]),
                "adapter_protocol": PROTOCOL_ID,
                **row,
            },
        )
        requests.append(
            build_measurement_adapter_execution_request(
                descriptor,
                spec,
                case,
                execution_id=f"isaac-physx-rigid-contact-{index + 1:03d}-{case_id}",
                implementation_id=IMPLEMENTATION_ID,
                implementation_version=IMPLEMENTATION_VERSION,
                implementation_digest=implementation_digest(),
                backend_id="isaac-physx-cpu-tgs-rigid",
                precision="float32",
                seed=47,
                solver_settings={
                    "solver_type": "TGS",
                    "broadphase_type": "SAP",
                    "gpu_dynamics": False,
                    "enhanced_determinism": True,
                    "position_iterations": 8,
                    "velocity_iterations": 2,
                },
                timeout_seconds=300,
            )
        )
    return corpus, requests


def _zip_bytes(archive: zipfile.ZipFile, name: str, payload: bytes) -> None:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    archive.writestr(info, payload)


def compile_measurement_isaac_physx_input_bundle(
    *,
    repo_root: str | Path,
    corpus_path: str | Path,
    qualification_split_digest: str,
    controller_scope_digest: str,
    output_path: str | Path,
    rtx_required_output_kinds: Sequence[str] = RTX_REQUIRED_OUTPUT_KINDS,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    commit = _git_identity(root)
    corpus_file = Path(corpus_path).resolve()
    if root != corpus_file and root not in corpus_file.parents:
        raise MeasurementIsaacVastBundleError("measurement_isaac_corpus_outside_source")
    corpus, requests = _requests(
        corpus_file,
        qualification_split_digest=qualification_split_digest,
        controller_scope_digest=controller_scope_digest,
    )
    runtime_release = build_measurement_isaac_runtime_release()
    source_files = sorted((root / "src/blueprint_pipeline").rglob("*.py"))
    source_files.extend([root / RUNNER_RELATIVE_PATH, root / WORKER_RELATIVE_PATH])
    if not source_files or any(not path.is_file() or path.is_symlink() for path in source_files):
        raise MeasurementIsaacVastBundleError("measurement_isaac_source_files_invalid")
    source_records = [
        {"path": path.relative_to(root).as_posix(), "digest": _sha256(path)}
        for path in source_files
    ]
    request_records = [
        {
            "path": f"requests/{index + 1:03d}.json",
            "execution_request_digest": request["execution_request_digest"],
        }
        for index, request in enumerate(requests)
    ]
    required_outputs = tuple(
        dict.fromkeys(str(item).strip() for item in rtx_required_output_kinds if str(item).strip())
    )
    if (
        not required_outputs
        or "rgb" not in required_outputs
        or not set(required_outputs) <= set(RTX_REQUIRED_OUTPUT_KINDS)
    ):
        raise MeasurementIsaacVastBundleError("measurement_isaac_rtx_output_kinds_invalid")
    manifest = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "source_commit_sha": commit,
        "runtime_image_digest": RUNTIME_IMAGE,
        "runtime_release_digest": runtime_release["runtime_release_digest"],
        "isaac_sim_version": ISAAC_VERSION,
        "corpus_id": corpus["corpus_id"],
        "corpus_digest": _file_digest(corpus_file),
        "qualification_split_digest": qualification_split_digest,
        "controller_scope_digest": controller_scope_digest,
        "request_files": request_records,
        "source_files": source_records,
        "runner_path": RUNNER_RELATIVE_PATH.as_posix(),
        "worker_path": WORKER_RELATIVE_PATH.as_posix(),
        "rtx_openusd_runtime_preflight_required": True,
        "rtx_renderer": "RayTracedLighting",
        "rtx_smoke_resolution": [64, 64],
        "rtx_required_output_kinds": list(required_outputs),
        "development_only": True,
        "held_out": False,
        "qualification_labels_included": False,
        "paid_execution_authorized_by_bundle": False,
        "r7_admission_created": False,
        "physical_success_established": False,
    }
    manifest["bundle_manifest_digest"] = canonical_digest(
        manifest, digest_field="bundle_manifest_digest"
    )
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", allowZip64=True) as archive:
        _zip_bytes(
            archive,
            "bundle_manifest.json",
            (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(),
        )
        for index, request in enumerate(requests):
            _zip_bytes(
                archive,
                f"requests/{index + 1:03d}.json",
                (json.dumps(request, indent=2, sort_keys=True) + "\n").encode(),
            )
        for path in source_files:
            _zip_bytes(archive, path.relative_to(root).as_posix(), path.read_bytes())
    receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "source_commit_sha": commit,
        "runtime_image_digest": RUNTIME_IMAGE,
        "runtime_release_digest": runtime_release["runtime_release_digest"],
        "bundle_manifest_digest": manifest["bundle_manifest_digest"],
        "input_bundle_digest": _sha256(output),
        "input_bundle_size_bytes": output.stat().st_size,
        "execution_request_digests": [row["execution_request_digest"] for row in request_records],
        "request_count": len(requests),
        "rtx_openusd_runtime_preflight_required": True,
        "rtx_renderer": "RayTracedLighting",
        "rtx_smoke_resolution": [64, 64],
        "rtx_required_output_kinds": list(required_outputs),
        "raw_secret_values_recorded": False,
        "provider_allocation_performed": False,
        "paid_execution_authorized_by_bundle": False,
        "proof_effect": "none",
        "claim_ceiling": "immutable_development_input_bundle_only",
    }
    receipt["bundle_receipt_digest"] = canonical_digest(
        receipt, digest_field="bundle_receipt_digest"
    )
    return validate_measurement_isaac_physx_input_bundle_receipt(receipt)


def validate_measurement_isaac_physx_input_bundle_receipt(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    receipt = json.loads(json.dumps(dict(value)))
    errors = []
    if receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION:
        errors.append("measurement_isaac_bundle_receipt_schema_invalid")
    if receipt.get("runtime_image_digest") != RUNTIME_IMAGE:
        errors.append("measurement_isaac_bundle_receipt_image_invalid")
    if receipt.get("request_count") != 2:
        errors.append("measurement_isaac_bundle_receipt_request_count_invalid")
    if (
        receipt.get("rtx_openusd_runtime_preflight_required") is not True
        or receipt.get("rtx_renderer") != "RayTracedLighting"
        or receipt.get("rtx_smoke_resolution") != [64, 64]
        or not isinstance(receipt.get("rtx_required_output_kinds"), list)
        or not receipt.get("rtx_required_output_kinds")
        or "rgb" not in receipt.get("rtx_required_output_kinds")
        or not set(receipt.get("rtx_required_output_kinds")) <= set(RTX_REQUIRED_OUTPUT_KINDS)
    ):
        errors.append("measurement_isaac_bundle_receipt_rtx_contract_invalid")
    for key, expected in (
        ("raw_secret_values_recorded", False),
        ("provider_allocation_performed", False),
        ("paid_execution_authorized_by_bundle", False),
        ("proof_effect", "none"),
        ("claim_ceiling", "immutable_development_input_bundle_only"),
    ):
        if receipt.get(key) != expected:
            errors.append(f"measurement_isaac_bundle_receipt_{key}_invalid")
    if receipt.get("bundle_receipt_digest") != canonical_digest(
        receipt, digest_field="bundle_receipt_digest"
    ):
        errors.append("measurement_isaac_bundle_receipt_digest_mismatch")
    if errors:
        raise MeasurementIsaacVastBundleError(";".join(sorted(set(errors))))
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--qualification-split-digest", required=True)
    parser.add_argument("--controller-scope-digest", required=True)
    parser.add_argument(
        "--rtx-output-kind",
        action="append",
        choices=list(RTX_REQUIRED_OUTPUT_KINDS),
        default=None,
    )
    parser.add_argument("--bundle-output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = compile_measurement_isaac_physx_input_bundle(
        repo_root=args.repo_root,
        corpus_path=args.corpus,
        qualification_split_digest=args.qualification_split_digest,
        controller_scope_digest=args.controller_scope_digest,
        output_path=args.bundle_output,
        rtx_required_output_kinds=args.rtx_output_kind or RTX_REQUIRED_OUTPUT_KINDS,
    )
    args.receipt_output.parent.mkdir(parents=True, exist_ok=True)
    args.receipt_output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BUNDLE_SCHEMA_VERSION",
    "MeasurementIsaacVastBundleError",
    "RECEIPT_SCHEMA_VERSION",
    "RTX_REQUIRED_OUTPUT_KINDS",
    "compile_measurement_isaac_physx_input_bundle",
    "main",
    "validate_measurement_isaac_physx_input_bundle_receipt",
]
