"""Compile a clean-commit DLO-Lab CUDA development input bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import stat
import subprocess
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from .decision_evidence_contracts import canonical_digest
from .measurement_adapter_execution import build_measurement_adapter_execution_request
from .measurement_adapter_runtime import build_measurement_adapter_descriptor
from .measurement_dlo_lab_cable_adapter import (
    EXPECTED_SOURCE_COMMIT,
    IMPLEMENTATION_ID,
    IMPLEMENTATION_VERSION,
    implementation_digest,
)
from .measurement_dlo_lab_runtime_release import (
    RUNTIME_IMAGE,
    build_measurement_dlo_lab_runtime_release,
)
from .measurement_qualification_benchmarks import (
    build_benchmark_case_manifest,
    build_qualification_benchmark_spec,
)


BUNDLE_SCHEMA_VERSION = "measurement_dlo_lab_cuda_input_bundle.v1"
RECEIPT_SCHEMA_VERSION = "measurement_dlo_lab_cuda_input_bundle_receipt.v1"
RUNNER_RELATIVE_PATH = Path("scripts/run_measurement_dlo_lab_bundle.py")


class MeasurementDloLabVastBundleError(ValueError):
    pass


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _git_identity(root: Path) -> str:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=False
    )
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if head.returncode != 0 or status.returncode != 0 or status.stdout.strip():
        raise MeasurementDloLabVastBundleError("measurement_dlo_lab_source_not_clean_commit")
    commit = head.stdout.strip().lower()
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise MeasurementDloLabVastBundleError("measurement_dlo_lab_source_commit_invalid")
    return commit


def _load_corpus(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MeasurementDloLabVastBundleError("measurement_dlo_lab_corpus_unreadable") from exc
    if not isinstance(value, Mapping) or value.get("schema_version") != (
        "measurement_dlo_lab_cable_development_corpus.v1"
    ):
        raise MeasurementDloLabVastBundleError("measurement_dlo_lab_corpus_invalid")
    if (
        value.get("development_only") is not True
        or value.get("synthetic_fixture") is not True
        or value.get("held_out") is not False
        or value.get("physical_measurements_included") is not False
        or value.get("qualification_labels_included") is not False
        or value.get("r5_evidence") is not False
        or value.get("r6_decision") is not False
        or value.get("r7_admission") is not False
        or not isinstance(value.get("cases"), list)
        or len(value["cases"]) != 2
    ):
        raise MeasurementDloLabVastBundleError("measurement_dlo_lab_corpus_boundary_invalid")
    return dict(value)


def _requests(
    corpus_path: Path, *, qualification_split_digest: str, controller_scope_digest: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    corpus = _load_corpus(corpus_path)
    corpus_digest = _sha256(corpus_path)
    spec = build_qualification_benchmark_spec(
        benchmark_id="capture-to-deformation",
        benchmark_version="development-dlo-lab-cuda-cable-1",
        method_ids=["dlo-lab"],
        development_split_digest=corpus_digest,
        qualification_split_digest=qualification_split_digest,
        capture_bundle_digests=[corpus_digest],
        robot_controller_digests=[controller_scope_digest],
        acceptance_thresholds={
            "maximum_mean_absolute_error": 1.0,
            "maximum_mismatch_rate": 0.0,
            "maximum_harmful_false_negative_rate": 0.0,
            "minimum_coverage": 0.5,
        },
        compute_budget={"usd": 1.0, "maximum_duration_seconds": 1800},
        minimum_repeated_trials=2,
        lane="cable",
    )
    descriptor = build_measurement_adapter_descriptor("dlo-lab")
    requests = []
    for index, raw_case in enumerate(corpus["cases"]):
        row = dict(raw_case)
        case_id = str(row.pop("case_id"))
        case = build_benchmark_case_manifest(
            spec,
            case_id=case_id,
            split="development",
            input_artifact_digests=[corpus_digest],
            task_class="cable_hose_routing",
            material_regime="synthetic_parameterized_rod",
            operating_point={**dict(corpus["shared_operating_point"]), **row},
        )
        requests.append(
            build_measurement_adapter_execution_request(
                descriptor,
                spec,
                case,
                execution_id=f"dlo-lab-cuda-cable-{index + 1:03d}-{case_id}",
                implementation_id=IMPLEMENTATION_ID,
                implementation_version=IMPLEMENTATION_VERSION,
                implementation_digest=implementation_digest(),
                backend_id="dlo-lab-genesis-cuda",
                precision="float64",
                seed=31,
                solver_settings={
                    "backend": "cuda",
                    "import_diagnostic": "audit_exception_first_case_only",
                    "native_diagnostic": "gdb_first_case_only",
                    "replay_count": 2,
                    "source_commit": EXPECTED_SOURCE_COMMIT,
                },
                timeout_seconds=1200,
            )
        )
    return corpus, requests


def _zip_bytes(archive: zipfile.ZipFile, name: str, payload: bytes) -> None:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    archive.writestr(info, payload)


def compile_measurement_dlo_lab_input_bundle(
    *,
    repo_root: str | Path,
    corpus_path: str | Path,
    qualification_split_digest: str,
    controller_scope_digest: str,
    output_path: str | Path,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    commit = _git_identity(root)
    corpus_file = Path(corpus_path).resolve()
    if root != corpus_file and root not in corpus_file.parents:
        raise MeasurementDloLabVastBundleError("measurement_dlo_lab_corpus_outside_source")
    corpus, requests = _requests(
        corpus_file,
        qualification_split_digest=qualification_split_digest,
        controller_scope_digest=controller_scope_digest,
    )
    runtime_release = build_measurement_dlo_lab_runtime_release()
    source_files = sorted((root / "src/blueprint_pipeline").rglob("*.py"))
    source_files.append(root / RUNNER_RELATIVE_PATH)
    if not source_files or any(not path.is_file() or path.is_symlink() for path in source_files):
        raise MeasurementDloLabVastBundleError("measurement_dlo_lab_source_files_invalid")
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
    manifest = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "source_commit_sha": commit,
        "runtime_image_digest": RUNTIME_IMAGE,
        "runtime_release_digest": runtime_release["runtime_release_digest"],
        "dlo_lab_source_repository": runtime_release["dlo_lab_source_repository"],
        "dlo_lab_source_commit": EXPECTED_SOURCE_COMMIT,
        "corpus_id": corpus["corpus_id"],
        "corpus_digest": _sha256(corpus_file),
        "qualification_split_digest": qualification_split_digest,
        "controller_scope_digest": controller_scope_digest,
        "request_files": request_records,
        "source_files": source_records,
        "runner_path": RUNNER_RELATIVE_PATH.as_posix(),
        "required_backend": "cuda",
        "replay_count": 2,
        "development_only": True,
        "synthetic_fixture": True,
        "held_out": False,
        "physical_measurements_included": False,
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
        "dlo_lab_source_commit": EXPECTED_SOURCE_COMMIT,
        "bundle_manifest_digest": manifest["bundle_manifest_digest"],
        "input_bundle_digest": _sha256(output),
        "input_bundle_size_bytes": output.stat().st_size,
        "execution_request_digests": [row["execution_request_digest"] for row in request_records],
        "request_count": len(requests),
        "required_backend": "cuda",
        "replay_count": 2,
        "raw_secret_values_recorded": False,
        "provider_allocation_performed": False,
        "paid_execution_authorized_by_bundle": False,
        "proof_effect": "none",
        "claim_ceiling": "immutable_development_input_bundle_only",
    }
    receipt["bundle_receipt_digest"] = canonical_digest(
        receipt, digest_field="bundle_receipt_digest"
    )
    return validate_measurement_dlo_lab_input_bundle_receipt(receipt)


def validate_measurement_dlo_lab_input_bundle_receipt(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    receipt = json.loads(json.dumps(dict(value)))
    errors: list[str] = []
    if receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION:
        errors.append("measurement_dlo_lab_bundle_receipt_schema_invalid")
    if receipt.get("runtime_image_digest") != RUNTIME_IMAGE:
        errors.append("measurement_dlo_lab_bundle_receipt_image_invalid")
    if receipt.get("dlo_lab_source_commit") != EXPECTED_SOURCE_COMMIT:
        errors.append("measurement_dlo_lab_bundle_receipt_source_invalid")
    if (
        receipt.get("request_count") != 2
        or len(receipt.get("execution_request_digests") or []) != 2
    ):
        errors.append("measurement_dlo_lab_bundle_receipt_requests_invalid")
    if receipt.get("required_backend") != "cuda" or receipt.get("replay_count") != 2:
        errors.append("measurement_dlo_lab_bundle_receipt_runtime_contract_invalid")
    for key, expected in (
        ("raw_secret_values_recorded", False),
        ("provider_allocation_performed", False),
        ("paid_execution_authorized_by_bundle", False),
        ("proof_effect", "none"),
        ("claim_ceiling", "immutable_development_input_bundle_only"),
    ):
        if receipt.get(key) != expected:
            errors.append(f"measurement_dlo_lab_bundle_receipt_{key}_invalid")
    if receipt.get("bundle_receipt_digest") != canonical_digest(
        receipt, digest_field="bundle_receipt_digest"
    ):
        errors.append("measurement_dlo_lab_bundle_receipt_digest_mismatch")
    if errors:
        raise MeasurementDloLabVastBundleError(";".join(sorted(set(errors))))
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--qualification-split-digest", required=True)
    parser.add_argument("--controller-scope-digest", required=True)
    parser.add_argument("--bundle-output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = compile_measurement_dlo_lab_input_bundle(
        repo_root=args.repo_root,
        corpus_path=args.corpus,
        qualification_split_digest=args.qualification_split_digest,
        controller_scope_digest=args.controller_scope_digest,
        output_path=args.bundle_output,
    )
    args.receipt_output.parent.mkdir(parents=True, exist_ok=True)
    args.receipt_output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BUNDLE_SCHEMA_VERSION",
    "MeasurementDloLabVastBundleError",
    "RECEIPT_SCHEMA_VERSION",
    "compile_measurement_dlo_lab_input_bundle",
    "main",
    "validate_measurement_dlo_lab_input_bundle_receipt",
]
