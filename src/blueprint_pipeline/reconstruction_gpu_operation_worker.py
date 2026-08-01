"""Execute one admitted pose or trainer bundle inside the pinned GPU worker."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence

from .common import write_json
from .native_360_colmap_runner import execute_native_360_colmap_plan
from .reconstruction_gaussian_trainer import run_gaussian_reconstruction_training
from .reconstruction_gpu_operation_bundle import (
    ReconstructionGpuOperationBundleError,
    extract_reconstruction_gpu_operation_bundle,
)
from .reconstruction_gpu_operation_output import (
    compile_reconstruction_gpu_operation_output_bundle,
)
from .reconstruction_worker_contracts import (
    ReconstructionWorkerContractError,
    build_pose_estimation_result,
    build_training_result,
)


class ReconstructionGpuOperationWorkerError(ValueError):
    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__("; ".join(self.codes))


def _load_object(path: Path, *, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconstructionGpuOperationWorkerError([code]) from exc
    if not isinstance(value, Mapping):
        raise ReconstructionGpuOperationWorkerError([code])
    return dict(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _single_role(
    manifest: Mapping[str, Any], *, role: str, root: Path
) -> tuple[dict[str, Any], Path]:
    members = manifest.get("artifact_members")
    members = members if isinstance(members, list) else []
    rows = [dict(row) for row in members if isinstance(row, Mapping) and row.get("role") == role]
    if len(rows) != 1:
        raise ReconstructionGpuOperationWorkerError(
            ["reconstruction_operation_worker_role_cardinality_invalid"]
        )
    portable = PurePosixPath(str(rows[0].get("archive_path") or ""))
    target = root.joinpath(*portable.parts)
    if (
        target.is_symlink()
        or not target.is_file()
        or root not in target.resolve().parents
        or _sha256(target) != rows[0].get("digest")
    ):
        raise ReconstructionGpuOperationWorkerError(
            ["reconstruction_operation_worker_role_artifact_invalid"]
        )
    return rows[0], target


def _verify_output_files(result: Mapping[str, Any], root: Path) -> None:
    references = list(result.get("output_digests") or []) + list(
        result.get("checkpoint_references") or []
    )
    errors: list[str] = []
    for row in references:
        if not isinstance(row, Mapping):
            errors.append("reconstruction_operation_worker_output_reference_invalid")
            continue
        portable = PurePosixPath(str(row.get("artifact_id") or ""))
        if (
            not portable.parts
            or portable.is_absolute()
            or any(part in {"", ".", ".."} for part in portable.parts)
        ):
            errors.append("reconstruction_operation_worker_output_reference_invalid")
            continue
        target = root.joinpath(*portable.parts)
        if (
            target.is_symlink()
            or not target.is_file()
            or root not in target.resolve().parents
            or _sha256(target) != row.get("digest")
        ):
            errors.append("reconstruction_operation_worker_output_binding_invalid")
    if errors:
        raise ReconstructionGpuOperationWorkerError(errors)


def execute_reconstruction_gpu_operation_bundle(
    *,
    bundle_path: str | Path,
    bundle_receipt: Mapping[str, Any],
    materialization_root: str | Path,
    output_root: str | Path,
    pose_executor: Callable[..., Mapping[str, Any]] = execute_native_360_colmap_plan,
    trainer_executor: Callable[..., Mapping[str, Any]] = run_gaussian_reconstruction_training,
) -> dict[str, Any]:
    """Materialize, dispatch, and validate one registered scientific operation."""

    try:
        extraction = extract_reconstruction_gpu_operation_bundle(
            bundle_path=bundle_path,
            bundle_receipt=bundle_receipt,
            output_root=materialization_root,
        )
    except ReconstructionGpuOperationBundleError as exc:
        raise ReconstructionGpuOperationWorkerError(
            [f"reconstruction_operation_worker_bundle_invalid:{code}" for code in exc.codes]
        ) from exc
    materialized = (
        Path(materialization_root).resolve()
        / extraction["operation_input_bundle_digest"].removeprefix("sha256:")
    )
    manifest = _load_object(
        materialized / "bundle_manifest.json",
        code="reconstruction_operation_worker_manifest_invalid",
    )
    request = _load_object(
        materialized / "operation_request.json",
        code="reconstruction_operation_worker_request_invalid",
    )
    output = Path(output_root)
    if output.is_symlink():
        raise ReconstructionGpuOperationWorkerError(
            ["reconstruction_operation_worker_output_root_symlink_forbidden"]
        )
    output.mkdir(parents=True, exist_ok=True)
    output = output.resolve()
    operation = extraction["operation"]
    try:
        if operation == "pose_canary":
            _, plan_path = _single_role(
                manifest, role="pose_execution_plan", root=materialized
            )
            plan = _load_object(
                plan_path,
                code="reconstruction_operation_worker_pose_plan_invalid",
            )
            emitted = dict(
                pose_executor(
                    plan=plan,
                    input_root=materialized / "inputs",
                    artifact_root=output,
                    timestamp=str(request.get("timestamp") or ""),
                )
            )
            result = build_pose_estimation_result(emitted)
            request_digest = request.get("pose_estimation_request_digest")
            result_digest_field = "pose_estimation_result_digest"
            result_root = output / (
                "native_colmap_execution_"
                + str(plan.get("native_360_colmap_execution_plan_digest") or "")[7:23]
            )
        elif operation == "trainer_canary":
            _, export_path = _single_role(
                manifest, role="dataset_export", root=materialized
            )
            emitted = dict(
                trainer_executor(
                    training_request=request,
                    dataset_export=_load_object(
                        export_path,
                        code="reconstruction_operation_worker_dataset_export_invalid",
                    ),
                    artifact_root=materialized / "inputs",
                    output_root=output,
                )
            )
            result = build_training_result(emitted)
            request_digest = request.get("reconstruction_training_request_digest")
            result_digest_field = "reconstruction_training_result_digest"
            result_root = output / str(request_digest or "")[7:23]
        else:
            raise ReconstructionGpuOperationWorkerError(
                ["reconstruction_operation_worker_unregistered_operation"]
            )
    except ReconstructionWorkerContractError as exc:
        raise ReconstructionGpuOperationWorkerError(
            [f"reconstruction_operation_worker_result_invalid:{code}" for code in exc.codes]
        ) from exc
    if (
        result.get("schema_version") != manifest.get("expected_runtime_result_schema")
        or result.get(
            "pose_estimation_request_digest"
            if operation == "pose_canary"
            else "reconstruction_training_request_digest"
        )
        != request_digest
        or result.get("heldout_labels_included") is not False
        or result.get("candidate_self_graded") is not False
        or result.get(result_digest_field) is None
    ):
        raise ReconstructionGpuOperationWorkerError(
            ["reconstruction_operation_worker_result_binding_invalid"]
        )
    _verify_output_files(result, result_root)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--bundle-receipt", type=Path, required=True)
    parser.add_argument("--materialization-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output-bundle", type=Path, required=True)
    args = parser.parse_args(argv)
    bundle_receipt = _load_object(
        args.bundle_receipt,
        code="reconstruction_operation_worker_bundle_receipt_invalid",
    )
    result = execute_reconstruction_gpu_operation_bundle(
        bundle_path=args.bundle,
        bundle_receipt=bundle_receipt,
        materialization_root=args.materialization_root,
        output_root=args.output_root,
    )
    materialized = (
        args.materialization_root.resolve()
        / str(bundle_receipt.get("operation_input_bundle_digest") or "").removeprefix(
            "sha256:"
        )
    )
    operation_request = _load_object(
        materialized / "operation_request.json",
        code="reconstruction_operation_worker_request_invalid",
    )
    output_bundle = compile_reconstruction_gpu_operation_output_bundle(
        operation=str(bundle_receipt.get("operation") or ""),
        operation_request=operation_request,
        runtime_result=result,
        operation_output_root=args.output_root,
        output_path=args.output_bundle,
    )
    write_json(args.result, result)
    print(
        json.dumps(
            {
                "schema_version": result["schema_version"],
                "status": result["status"],
                "operation_output_bundle_digest": output_bundle[
                    "operation_output_bundle_digest"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "ReconstructionGpuOperationWorkerError",
    "execute_reconstruction_gpu_operation_bundle",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
