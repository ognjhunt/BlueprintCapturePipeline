"""Canonical V3.2 capture to Postshot/Splatfacto 3DGS execution boundary.

The module performs the complete non-paid, deterministic half of the path:

1. validate a Capture Raw Contract V3.2 iPhone/LiDAR bundle;
2. freeze candidate/validation/hidden observations from decoded PTS;
3. bind retained RGB to raw ARKit poses and intrinsics;
4. back-project captured high-confidence depth into initialization points;
5. export one immutable candidate-only COLMAP dataset; and
6. precommit Postshot as the primary trainer and Splatfacto as the comparison.

Worker execution is represented by a small injected runner boundary so the
same plan can be exercised hermetically without pretending that macOS executed
Postshot or that a CPU fixture executed CUDA training. Production workers must
return the same byte-hashed result contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .arkit_depth_surface_compiler import (
    build_arkit_depth_surface_compilation_request,
    compile_arkit_depth_surface,
)
from .decision_evidence_contracts import canonical_digest, canonical_json
from .canonical_3dgs_evaluation import (
    compile_canonical_3dgs_hidden_evaluator_input,
    compile_canonical_3dgs_proxy_hidden_evaluator_input,
)
from .gaussian_splat_decode import read_standard_3dgs_ply
from .local_reconstruction_adapters import LocalArkitMetricScaffoldAdapter
from .reconstruction_colmap_dataset import (
    bind_colmap_initialization_surface,
    export_colmap_training_dataset,
)


PLAN_SCHEMA = "canonical_3dgs_execution_plan.v1"
ARM_RESULT_SCHEMA = "canonical_3dgs_arm_result.v1"
CAMPAIGN_RESULT_SCHEMA = "canonical_3dgs_campaign_result.v1"
PREPARATION_SCHEMA = "canonical_v32_3dgs_preparation.v1"
PROXY_PREPARATION_SCHEMA = "canonical_arkitscenes_proxy_3dgs_preparation.v1"
SOURCE_ADMISSION_SCHEMA = "canonical_3dgs_source_admission.v1"

POSTSHOT_METHOD = "jawset_postshot_splat3_v1"
SPLATFACTO_METHOD = "nerfstudio_splatfacto_v1_1_5"


class Canonical3DGSPipelineError(ValueError):
    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__(";".join(self.codes))


ArmRunner = Callable[[Mapping[str, Any], Path, Path], Mapping[str, Any]]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 71
        and value.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in value[7:])
    )


def canonical_3dgs_worker_package_digest(
    package_root: str | Path | None = None,
) -> str:
    """Digest every shipped Python source byte used by the worker package."""

    root = (
        Path(package_root).expanduser().resolve()
        if package_root is not None
        else Path(__file__).resolve().parent
    )
    if not root.is_dir() or root.is_symlink():
        raise Canonical3DGSPipelineError(["worker_package_root_invalid"])
    members = [
        {
            "relative_path": path.relative_to(root).as_posix(),
            "digest": _sha256(path),
        }
        for path in sorted(root.rglob("*.py"))
        if path.is_file() and not path.is_symlink() and "__pycache__" not in path.parts
    ]
    if not members:
        raise Canonical3DGSPipelineError(["worker_package_sources_missing"])
    return canonical_digest(
        {
            "schema_version": "canonical_3dgs_worker_python_package.v1",
            "members": members,
        }
    )


def canonical_3dgs_worker_wheel_package_digest(wheel_path: str | Path) -> str:
    """Calculate the same source digest directly from a built pure-Python wheel."""

    wheel = Path(wheel_path).expanduser().resolve()
    if wheel.suffix != ".whl" or not wheel.is_file() or wheel.is_symlink():
        raise Canonical3DGSPipelineError(["worker_wheel_invalid"])
    try:
        with zipfile.ZipFile(wheel) as archive:
            members = []
            for info in sorted(archive.infolist(), key=lambda row: row.filename):
                path = PurePosixPath(info.filename)
                if (
                    len(path.parts) < 2
                    or path.parts[0] != "blueprint_pipeline"
                    or path.suffix != ".py"
                ):
                    continue
                members.append(
                    {
                        "relative_path": PurePosixPath(*path.parts[1:]).as_posix(),
                        "digest": "sha256:" + hashlib.sha256(archive.read(info)).hexdigest(),
                    }
                )
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        raise Canonical3DGSPipelineError(["worker_wheel_invalid"]) from exc
    if not members or len({row["relative_path"] for row in members}) != len(members):
        raise Canonical3DGSPipelineError(["worker_wheel_sources_invalid"])
    return canonical_digest(
        {
            "schema_version": "canonical_3dgs_worker_python_package.v1",
            "members": members,
        }
    )


def _load_json(path: Path, *, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Canonical3DGSPipelineError([code]) from exc
    if not isinstance(value, Mapping):
        raise Canonical3DGSPipelineError([code])
    return dict(value)


def _safe_child(root: Path, relative_path: Any, *, code: str) -> Path:
    text = str(relative_path or "").replace("\\", "/")
    relative = PurePosixPath(text)
    if (
        not text
        or relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise Canonical3DGSPipelineError([code])
    resolved_root = root.resolve()
    candidate = resolved_root.joinpath(*relative.parts).resolve()
    if resolved_root not in candidate.parents or not candidate.is_file() or candidate.is_symlink():
        raise Canonical3DGSPipelineError([code])
    return candidate


def _write_immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (canonical_json(dict(value)) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise Canonical3DGSPipelineError(["immutable_artifact_conflict"])
        return
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _artifact_reference(value: Mapping[str, Any], artifact_type: str) -> Mapping[str, Any]:
    for row in value.get("artifact_references") or []:
        if isinstance(row, Mapping) and row.get("artifact_type") == artifact_type:
            return row
    raise Canonical3DGSPipelineError([f"artifact_reference_missing:{artifact_type}"])


def _load_referenced_json(
    root: Path,
    reference: Mapping[str, Any],
    *,
    logical_digest_field: str,
    code: str,
) -> tuple[Path, dict[str, Any]]:
    path = _safe_child(root, reference.get("relative_path"), code=code)
    if _sha256(path) != reference.get("artifact_digest"):
        raise Canonical3DGSPipelineError([f"{code}:byte_digest_mismatch"])
    value = _load_json(path, code=code)
    if value.get(logical_digest_field) != reference.get("content_digest"):
        raise Canonical3DGSPipelineError([f"{code}:logical_digest_mismatch"])
    return path, value


def _verify_colmap_dataset(dataset: Mapping[str, Any], dataset_root: Path) -> None:
    errors: list[str] = []
    if dataset.get("schema_version") != "colmap_training_dataset_export_result.v1":
        errors.append("colmap_result_schema_invalid")
    recorded = dataset.get("colmap_training_dataset_export_result_digest")
    if recorded != canonical_digest(
        dataset, digest_field="colmap_training_dataset_export_result_digest"
    ):
        errors.append("colmap_result_digest_invalid")
    if (
        dataset.get("status") != "exported_candidate_only_colmap_text_dataset"
        or dataset.get("hidden_heldout_pixels_included") is not False
        or dataset.get("trainer_self_grading_permitted") is not False
        or dataset.get("raw_input_poses_modified") is not False
        or not _digest(dataset.get("source_capture_digest"))
        or not _digest(dataset.get("colmap_training_dataset_digest"))
        or int(dataset.get("image_count") or 0) < 3
        or int(dataset.get("initialization_point_count") or 0) < 1
    ):
        errors.append("colmap_dataset_not_training_ready")
    output_artifacts = dataset.get("output_artifacts")
    if not isinstance(output_artifacts, list) or len(output_artifacts) < 4:
        errors.append("colmap_output_artifacts_missing")
        output_artifacts = []
    seen_paths: set[str] = set()
    image_digests: list[dict[str, str]] = []
    sparse_digests: dict[str, str] = {}
    for row in output_artifacts:
        if not isinstance(row, Mapping):
            errors.append("colmap_output_artifact_invalid")
            continue
        try:
            path = _safe_child(
                dataset_root,
                row.get("relative_path"),
                code="colmap_output_artifact_unsafe",
            )
        except Canonical3DGSPipelineError as exc:
            errors.extend(exc.codes)
            continue
        relative_path = path.relative_to(dataset_root.resolve()).as_posix()
        if relative_path in seen_paths:
            errors.append("colmap_output_artifact_duplicate")
        seen_paths.add(relative_path)
        digest = _sha256(path)
        if digest != row.get("digest"):
            errors.append("colmap_output_artifact_digest_mismatch")
        if row.get("artifact_type") == "candidate_image" and relative_path.startswith(
            "images/"
        ):
            image_digests.append({"artifact_id": path.name, "digest": digest})
        elif row.get("artifact_type") == "colmap_sparse_text" and relative_path.startswith(
            "sparse/0/"
        ):
            sparse_digests[path.name] = digest
        else:
            errors.append("colmap_output_artifact_type_or_path_invalid")
    expected_paths = {
        path.relative_to(dataset_root.resolve()).as_posix()
        for base in (dataset_root / "images", dataset_root / "sparse" / "0")
        if base.is_dir()
        for path in base.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    if seen_paths != expected_paths:
        errors.append("colmap_output_artifact_inventory_mismatch")
    request_digest = (
        dataset.get("parent_artifact_or_event", {}).get("request_digest")
        if isinstance(dataset.get("parent_artifact_or_event"), Mapping)
        else None
    )
    if not _digest(request_digest):
        errors.append("colmap_parent_request_digest_invalid")
    elif dataset.get("colmap_training_dataset_digest") != canonical_digest(
        {
            "images": image_digests,
            "sparse": dict(sorted(sparse_digests.items())),
            "request_digest": request_digest,
        }
    ):
        errors.append("colmap_training_dataset_digest_invalid")
    if errors:
        raise Canonical3DGSPipelineError(errors)


def build_canonical_3dgs_source_admission(
    *,
    source_profile: str,
    source_capture_identity: str,
    source_capture_digest: str,
    source_artifact_commit_sha: str,
    frozen_split_digest: str,
    colmap_training_dataset_digest: str,
    hidden_evaluator_input_digest: str,
    raw_contract_3_2_proven: bool,
    world_frame: str,
    coordinate_frame_declaration: Mapping[str, Any],
    metric_scale_status: str,
    authority_used: Mapping[str, Any],
    input_artifacts: Sequence[Mapping[str, Any]],
    claim_limitations: Sequence[str],
    timestamp: str,
) -> dict[str, Any]:
    """Admit one exact source only for candidate training and held-out evaluation."""

    errors: list[str] = []
    if source_profile not in {"blueprint_raw_v3_2", "public_dataset_arkitscenes_proxy"}:
        errors.append("canonical_3dgs_source_profile_invalid")
    if not str(source_capture_identity or "").strip() or not _digest(source_capture_digest):
        errors.append("canonical_3dgs_source_identity_invalid")
    if len(source_artifact_commit_sha) != 40 or any(
        character not in "0123456789abcdef" for character in source_artifact_commit_sha
    ):
        errors.append("canonical_3dgs_source_artifact_commit_invalid")
    for value, code in (
        (frozen_split_digest, "canonical_3dgs_source_split_digest_invalid"),
        (colmap_training_dataset_digest, "canonical_3dgs_source_dataset_digest_invalid"),
        (hidden_evaluator_input_digest, "canonical_3dgs_source_evaluator_digest_invalid"),
    ):
        if not _digest(value):
            errors.append(code)
    if raw_contract_3_2_proven != (source_profile == "blueprint_raw_v3_2"):
        errors.append("canonical_3dgs_source_raw_contract_claim_invalid")
    if not str(world_frame or "").strip() or not isinstance(
        coordinate_frame_declaration, Mapping
    ):
        errors.append("canonical_3dgs_source_coordinate_frame_invalid")
    if metric_scale_status not in {
        "sensor_metric_unvalidated",
        "independently_validated_metric",
        "not_established",
        "unknown",
    }:
        errors.append("canonical_3dgs_source_metric_scale_status_invalid")
    if not isinstance(authority_used, Mapping) or authority_used.get(
        "local_processing_authorized"
    ) is not True:
        errors.append("canonical_3dgs_source_local_authority_missing")
    normalized_artifacts: list[dict[str, str]] = []
    for row in input_artifacts:
        if not isinstance(row, Mapping) or not str(row.get("artifact_id") or "").strip() or not _digest(
            row.get("digest")
        ):
            errors.append("canonical_3dgs_source_input_artifact_invalid")
            continue
        normalized_artifacts.append(
            {"artifact_id": str(row["artifact_id"]), "digest": str(row["digest"])}
        )
    if not normalized_artifacts:
        errors.append("canonical_3dgs_source_input_artifacts_missing")
    if errors:
        raise Canonical3DGSPipelineError(errors)
    result = {
        "schema_version": SOURCE_ADMISSION_SCHEMA,
        "status": "admitted_candidate_training_source",
        "source_profile": source_profile,
        "source_capture_identity": source_capture_identity,
        "source_capture_digest": source_capture_digest,
        "source_artifact_commit_sha": source_artifact_commit_sha,
        "frozen_split_digest": frozen_split_digest,
        "colmap_training_dataset_digest": colmap_training_dataset_digest,
        "hidden_evaluator_input_digest": hidden_evaluator_input_digest,
        "candidate_hidden_pixel_access": False,
        "raw_contract_3_2_proven": raw_contract_3_2_proven,
        "world_frame": world_frame,
        "coordinate_frame_declaration": dict(coordinate_frame_declaration),
        "metric_scale_status": metric_scale_status,
        "metric_scale_independently_validated": (
            metric_scale_status == "independently_validated_metric"
        ),
        "authority_used": dict(authority_used),
        "input_artifacts": sorted(normalized_artifacts, key=lambda row: row["artifact_id"]),
        "claim_limitations": sorted(set(str(value) for value in claim_limitations)),
        "provider_upload_authorized_by_source_admission": False,
        "paid_compute_authorized_by_source_admission": False,
        "proof_effect": "candidate_training_source_admission_only",
        "claim_ceiling": "appearance_reconstruction_candidate",
        "timestamp": timestamp,
    }
    result["canonical_3dgs_source_admission_digest"] = canonical_digest(
        result, digest_field="canonical_3dgs_source_admission_digest"
    )
    return result


def _standard_splat_profile(splat: Any) -> dict[str, Any]:
    xyz = np.asarray(splat.xyz, dtype=np.float64)
    if xyz.shape != (splat.count, 3) or not np.isfinite(xyz).all():
        raise Canonical3DGSPipelineError(["runner_standard_3dgs_positions_invalid"])
    rest_count = 0 if splat.sh_rest is None else int(splat.sh_rest.shape[1])
    coefficient_count = 1 + rest_count // 3
    sh_degree = int(round(coefficient_count**0.5)) - 1
    if rest_count % 3 or (sh_degree + 1) ** 2 != coefficient_count:
        raise Canonical3DGSPipelineError(["runner_standard_3dgs_sh_invalid"])
    return {
        "representation": "standard_3dgs_ply",
        "splat_count": int(splat.count),
        "sh_degree": sh_degree,
        "bounds": {
            "aabb_min": [float(value) for value in xyz.min(axis=0)],
            "aabb_max": [float(value) for value in xyz.max(axis=0)],
            "robust_min": [float(value) for value in np.quantile(xyz, 0.005, axis=0)],
            "robust_max": [float(value) for value in np.quantile(xyz, 0.995, axis=0)],
            "robust_percentile": 0.99,
        },
        "global_decimation_applied": False,
        "removal_reasons": [],
    }


def prepare_canonical_v32_training_dataset(
    *,
    capture_root: str | Path,
    output_root: str | Path,
    intake_id: str,
    capture_digest: str,
    rights_and_retention: Mapping[str, Any],
    maximum_frames: int = 60,
) -> dict[str, Any]:
    """Compile a strict V3.2 bundle into one depth-seeded COLMAP dataset."""

    raw_root = Path(capture_root).expanduser().resolve()
    derived_root = Path(output_root).expanduser().resolve()
    derived_root.mkdir(parents=True, exist_ok=True)
    scaffold_result = LocalArkitMetricScaffoldAdapter().execute(
        intake_id=intake_id,
        capture_digest=capture_digest,
        capture_root=raw_root,
        output_root=derived_root,
        rights_and_retention=rights_and_retention,
        maximum_frames=maximum_frames,
    )
    references = scaffold_result["asset_references"]
    scaffold_path = _safe_child(
        derived_root,
        references["metric_scaffold"]["relative_path"],
        code="metric_scaffold_path_invalid",
    )
    scaffold = _load_json(scaffold_path, code="metric_scaffold_invalid")
    export_path = _safe_child(
        derived_root,
        references["arkit_reconstruction_dataset_export"]["relative_path"],
        code="arkit_export_path_invalid",
    )
    arkit_export = _load_json(export_path, code="arkit_export_invalid")
    if (
        arkit_export.get("arkit_reconstruction_dataset_export_digest")
        != references["arkit_reconstruction_dataset_export"]["digest"]
        or arkit_export.get("source_capture_digest") != capture_digest
        or arkit_export.get("hidden_heldout_pixels_included") is not False
    ):
        raise Canonical3DGSPipelineError(["arkit_export_binding_invalid"])

    export_reference_root = export_path.parents[1]
    colmap_reference = _artifact_reference(
        arkit_export, "colmap_training_dataset_export_request.v1"
    )
    _, colmap_request = _load_referenced_json(
        export_reference_root,
        colmap_reference,
        logical_digest_field="colmap_training_dataset_export_request_digest",
        code="colmap_request_invalid",
    )
    calibration_reference = _artifact_reference(
        arkit_export, "camera_calibration_manifest.v1"
    )
    _, calibration = _load_referenced_json(
        export_reference_root,
        calibration_reference,
        logical_digest_field="calibration_digest",
        code="camera_calibration_invalid",
    )
    observation_reference = _artifact_reference(
        arkit_export, "camera_observation_manifest.v1"
    )
    _, observations = _load_referenced_json(
        export_reference_root,
        observation_reference,
        logical_digest_field="camera_observation_digest",
        code="camera_observations_invalid",
    )

    source_commit = str(arkit_export.get("source_commit_sha") or "")
    depth_request = build_arkit_depth_surface_compilation_request(
        stable_run_identity=f"{intake_id}-observed-depth-seed",
        source_capture_identity=intake_id,
        source_capture_digest=capture_digest,
        source_commit_sha=source_commit,
        metric_scaffold=scaffold,
        metric_scaffold_digest=references["metric_scaffold"]["digest"],
        camera_observation_manifest=observations,
        camera_calibration_manifest=calibration,
        artifact_root=raw_root,
        authority_used=rights_and_retention,
        timestamp=str(arkit_export["timestamp"]),
    )
    surface_reference_root = derived_root / "initialization"
    surface_result = compile_arkit_depth_surface(
        source_artifact=depth_request,
        artifact_root=raw_root,
        output_root=surface_reference_root / "arkit_depth_surface",
        output_reference_root=surface_reference_root,
    )
    initialized_request = bind_colmap_initialization_surface(
        source_artifact=colmap_request,
        surface_compilation_result=surface_result,
    )

    dataset_manifest_path = _safe_child(
        derived_root,
        references["reconstruction_dataset_manifest"]["relative_path"],
        code="reconstruction_dataset_manifest_path_invalid",
    )
    # Candidate image references are relative to the frozen dataset directory
    # that owns the dataset manifest, not the outer frame-extraction root.
    decoded_artifact_root = dataset_manifest_path.parent
    reconstruction_dataset_manifest = _load_json(
        dataset_manifest_path, code="reconstruction_dataset_manifest_invalid"
    )
    evaluator_input_root = derived_root / "evaluator_input"
    evaluator_input = compile_canonical_3dgs_hidden_evaluator_input(
        capture_root=raw_root,
        reconstruction_dataset_manifest=reconstruction_dataset_manifest,
        # Dataset artifact references are rooted one level above the
        # content-addressed frozen_dataset_* directory that owns the manifest.
        dataset_artifact_root=dataset_manifest_path.parent.parent,
        output_root=evaluator_input_root,
        source_commit_sha=source_commit,
        authority_used=rights_and_retention,
        timestamp=str(arkit_export["timestamp"]),
    )
    trainer_input_root = derived_root / "trainer_input"
    dataset = export_colmap_training_dataset(
        source_artifact=initialized_request,
        artifact_root=decoded_artifact_root,
        initialization_artifact_root=surface_reference_root,
        output_root=trainer_input_root,
    )
    dataset_root = trainer_input_root / str(dataset["relative_path"])
    _verify_colmap_dataset(dataset, dataset_root)

    source_admission = build_canonical_3dgs_source_admission(
        source_profile="blueprint_raw_v3_2",
        source_capture_identity=intake_id,
        source_capture_digest=capture_digest,
        source_artifact_commit_sha=source_commit,
        frozen_split_digest=dataset["frozen_split_digest"],
        colmap_training_dataset_digest=dataset["colmap_training_dataset_digest"],
        hidden_evaluator_input_digest=evaluator_input[
            "canonical_3dgs_hidden_evaluator_input_digest"
        ],
        raw_contract_3_2_proven=True,
        world_frame="canonical_arkit_world",
        coordinate_frame_declaration=dict(dataset.get("coordinate_frame_declaration") or {}),
        metric_scale_status=str(dataset.get("metric_scale_status") or "sensor_metric_unvalidated"),
        authority_used=rights_and_retention,
        input_artifacts=[
            {
                "artifact_id": "arkit_reconstruction_dataset_export",
                "digest": arkit_export["arkit_reconstruction_dataset_export_digest"],
            },
            {
                "artifact_id": "arkit_depth_surface_compilation_result",
                "digest": surface_result["arkit_depth_surface_compilation_result_digest"],
            },
            {
                "artifact_id": "arkit_observed_surface",
                "digest": surface_result["surface_asset"]["digest"],
            },
            {
                "artifact_id": "colmap_training_dataset_export_result",
                "digest": dataset["colmap_training_dataset_export_result_digest"],
            },
            {
                "artifact_id": "hidden_evaluator_input",
                "digest": evaluator_input["canonical_3dgs_hidden_evaluator_input_digest"],
            },
        ],
        claim_limitations=list(dataset.get("warnings") or [])
        + list(dataset.get("blockers") or []),
        timestamp=str(arkit_export["timestamp"]),
    )
    _write_immutable_json(
        derived_root / "canonical_3dgs_source_admission.json", source_admission
    )

    preparation = {
        "schema_version": PREPARATION_SCHEMA,
        "status": "training_dataset_ready",
        "source_profile": "blueprint_raw_v3_2",
        "canonical_3dgs_source_admission_digest": source_admission[
            "canonical_3dgs_source_admission_digest"
        ],
        "source_capture_identity": intake_id,
        "source_capture_digest": capture_digest,
        "pipeline_source_commit_sha": source_commit,
        "raw_contract_version": "3.2.0",
        "raw_contract_3_2_proven": True,
        "raw_capture_authority_preserved": True,
        "metric_scaffold_result_digest": scaffold_result[
            "reconstruction_result_digest"
        ],
        "arkit_reconstruction_dataset_export_digest": arkit_export[
            "arkit_reconstruction_dataset_export_digest"
        ],
        "arkit_depth_surface_compilation_result_digest": surface_result[
            "arkit_depth_surface_compilation_result_digest"
        ],
        "colmap_training_dataset_export_result_digest": dataset[
            "colmap_training_dataset_export_result_digest"
        ],
        "colmap_training_dataset_digest": dataset["colmap_training_dataset_digest"],
        "frozen_split_digest": dataset["frozen_split_digest"],
        "dataset_relative_path": dataset_root.relative_to(derived_root).as_posix(),
        "image_count": dataset["image_count"],
        "initialization_point_count": dataset["initialization_point_count"],
        "pose_binding": (
            "raw_arkit_pose_baseline"
            if dataset.get("pose_refinement_executed") is not True
            else "qualified_refined_pose_candidate"
        ),
        "world_frame": source_admission["world_frame"],
        "coordinate_frame_declaration": source_admission[
            "coordinate_frame_declaration"
        ],
        "metric_scale_status": source_admission["metric_scale_status"],
        "hidden_heldout_pixels_included": False,
        "hidden_evaluator_input_digest": evaluator_input[
            "canonical_3dgs_hidden_evaluator_input_digest"
        ],
        "hidden_evaluator_input_relative_path": (
            evaluator_input_root / "canonical_3dgs_hidden_evaluator_input.json"
        ).relative_to(derived_root).as_posix(),
        "trainer_self_grading_permitted": False,
        "warnings": list(dataset.get("warnings") or [])
        + list(dataset.get("blockers") or []),
        "proof_effect": "trainer_input_materialization_only",
        "claim_ceiling": "reconstruction_training_request",
        "timestamp": str(arkit_export["timestamp"]),
    }
    preparation["canonical_v32_3dgs_preparation_digest"] = canonical_digest(
        preparation, digest_field="canonical_v32_3dgs_preparation_digest"
    )
    _write_immutable_json(derived_root / "canonical_v32_3dgs_preparation.json", preparation)
    _write_immutable_json(derived_root / "colmap_training_dataset_export_result.json", dataset)
    return {"preparation": preparation, "dataset": dataset, "dataset_root": dataset_root}


def prepare_canonical_arkitscenes_proxy_training_dataset(
    *,
    proxy_root: str | Path,
    source_artifact_root: str | Path,
    output_root: str | Path,
    source_commit_sha: str,
) -> dict[str, Any]:
    """Admit one explicit ARKitScenes proxy and rebuild its canonical worker input.

    The caller supplies the exact content-addressed proxy directory.  No newest-
    file or scene-id search is performed, and legacy COLMAP exports are not
    trusted: the current exporter re-materializes and rehashes every candidate
    image and sparse text artifact.
    """

    proxy = Path(proxy_root).expanduser().resolve()
    source_root = Path(source_artifact_root).expanduser().resolve()
    derived_root = Path(output_root).expanduser().resolve()
    if source_root != proxy and source_root not in proxy.parents:
        raise Canonical3DGSPipelineError(["arkitscenes_proxy_root_binding_invalid"])
    if len(source_commit_sha) != 40 or any(
        character not in "0123456789abcdef" for character in source_commit_sha
    ):
        raise Canonical3DGSPipelineError(["source_commit_sha_invalid"])

    compilation_path = proxy / "arkitscenes_raw_proxy_compilation.json"
    reconstruction_path = next(
        iter(sorted(proxy.glob("frozen_dataset_*/reconstruction_dataset_manifest.json"))),
        None,
    )
    if reconstruction_path is None or len(
        list(proxy.glob("frozen_dataset_*/reconstruction_dataset_manifest.json"))
    ) != 1:
        raise Canonical3DGSPipelineError(["arkitscenes_proxy_dataset_manifest_ambiguous"])
    observations_path = proxy / "camera_observations_proxy.json"
    evaluator_scaffold_path = proxy / "evaluator_hidden" / "metric_scaffold_proxy.json"
    surface_result_path = (
        proxy / "observed_surface_proxy_v1" / "arkit_depth_surface_proxy_result.json"
    )
    compilation = _load_json(compilation_path, code="arkitscenes_proxy_compilation_invalid")
    reconstruction = _load_json(
        reconstruction_path, code="arkitscenes_proxy_dataset_manifest_invalid"
    )
    observations = _load_json(
        observations_path, code="arkitscenes_proxy_camera_observations_invalid"
    )
    evaluator_scaffold = _load_json(
        evaluator_scaffold_path, code="arkitscenes_proxy_evaluator_scaffold_invalid"
    )
    surface_result = _load_json(
        surface_result_path, code="arkitscenes_proxy_surface_result_invalid"
    )
    capture_digest = compilation.get("source_capture_digest")
    split_digest = compilation.get("train_heldout_split_digest")
    output_digests = compilation.get("output_digests")
    errors: list[str] = []
    if (
        compilation.get("schema_version") != "arkitscenes_raw_proxy_compilation.v1"
        or compilation.get("arkitscenes_proxy_compilation_digest")
        != canonical_digest(compilation, digest_field="arkitscenes_proxy_compilation_digest")
        or compilation.get("raw_contract_3_2_proven") is not False
        or compilation.get("hidden_heldout_pixels_exposed_to_candidate") is not False
        or not _digest(capture_digest)
        or not _digest(split_digest)
        or not isinstance(output_digests, Mapping)
    ):
        errors.append("arkitscenes_proxy_compilation_binding_invalid")
    if (
        reconstruction.get("schema_version") != "reconstruction_dataset_manifest.v1"
        or reconstruction.get("dataset_manifest_digest")
        != canonical_digest(reconstruction, digest_field="dataset_manifest_digest")
        or reconstruction.get("source_capture_digest") != capture_digest
        or reconstruction.get("train_heldout_split_digest") != split_digest
        or not isinstance(output_digests, Mapping)
        or output_digests.get("dataset_manifest_digest")
        != reconstruction.get("dataset_manifest_digest")
    ):
        errors.append("arkitscenes_proxy_dataset_binding_invalid")
    if (
        observations.get("schema_version")
        not in {"camera_observation_manifest.v1", "arkitscenes_camera_observations_proxy.v1"}
        or observations.get("camera_observation_digest")
        != canonical_digest(observations, digest_field="camera_observation_digest")
        or observations.get("capture_digest") != capture_digest
        or observations.get("split_digest") != split_digest
        or observations.get("hidden_heldout_pixels_included") is not False
        or observations.get("candidate_may_access_hidden_heldout") is not False
    ):
        errors.append("arkitscenes_proxy_camera_observation_binding_invalid")
    if (
        surface_result.get("arkit_depth_surface_compilation_result_digest")
        != canonical_digest(
            surface_result,
            digest_field="arkit_depth_surface_compilation_result_digest",
        )
        or surface_result.get("source_capture_digest") != capture_digest
        or surface_result.get("train_heldout_split_digest") != split_digest
        or surface_result.get("hidden_heldout_observations_accessed") is not False
        or surface_result.get("generated_fill_used") is not False
        or surface_result.get("raw_arkit_poses_modified") is not False
    ):
        errors.append("arkitscenes_proxy_surface_binding_invalid")
    authority = compilation.get("authority_used")
    if not isinstance(authority, Mapping) or authority.get("local_processing_authorized") is not True:
        errors.append("arkitscenes_proxy_authority_invalid")
    if errors:
        raise Canonical3DGSPipelineError(errors)

    references = reconstruction.get("artifact_references")
    candidate_reference = (
        references.get("candidate_dataset_manifest")
        if isinstance(references, Mapping)
        else None
    )
    if not isinstance(candidate_reference, Mapping):
        raise Canonical3DGSPipelineError(["arkitscenes_proxy_candidate_reference_missing"])
    candidate_path = _safe_child(
        proxy,
        candidate_reference.get("relative_path"),
        code="arkitscenes_proxy_candidate_path_invalid",
    )
    if _sha256(candidate_path) != candidate_reference.get("digest"):
        raise Canonical3DGSPipelineError(["arkitscenes_proxy_candidate_byte_digest_mismatch"])
    candidate = _load_json(candidate_path, code="arkitscenes_proxy_candidate_invalid")
    if (
        candidate.get("candidate_dataset_digest")
        != canonical_digest(candidate, digest_field="candidate_dataset_digest")
        or candidate.get("capture_digest") != capture_digest
        or candidate.get("split_digest") != split_digest
        or candidate.get("heldout_pixels_included") is not False
    ):
        raise Canonical3DGSPipelineError(["arkitscenes_proxy_candidate_binding_invalid"])

    surface_asset = surface_result.get("surface_asset")
    if not isinstance(surface_asset, Mapping):
        raise Canonical3DGSPipelineError(["arkitscenes_proxy_surface_asset_missing"])
    surface_path = _safe_child(
        source_root,
        surface_asset.get("relative_path"),
        code="arkitscenes_proxy_surface_asset_path_invalid",
    )
    if _sha256(surface_path) != surface_asset.get("digest"):
        raise Canonical3DGSPipelineError(["arkitscenes_proxy_surface_asset_digest_mismatch"])

    timestamp = str(compilation.get("timestamp") or "")
    request = {
        "schema_version": "colmap_training_dataset_export_request.v1",
        "stable_run_identity": (
            f"canonical-{compilation['source_capture_identity']}-{source_commit_sha[:12]}"
        ),
        "source_capture_digest": capture_digest,
        "source_commit_sha": source_commit_sha,
        "reconstruction_dataset_digest": reconstruction["dataset_manifest_digest"],
        "frozen_split_digest": split_digest,
        "camera_observation_manifest": observations,
        "camera_calibration_manifest": None,
        "candidate_dataset_manifest": candidate,
        "metric_scaffold_digest": surface_result["camera_calibration_binding"][
            "metric_scaffold_digest"
        ],
        "maximum_initialization_points": 100_000,
        "coordinate_frame_declaration": dict(
            surface_result.get("coordinate_frame_declaration") or {}
        ),
        "units": "meters",
        "metric_scale_status": "sensor_metric_unvalidated",
        "authority_used": dict(authority),
        "timestamp": timestamp,
        "blockers": [
            "initialization_surface_not_bound",
            "pose_refinement_not_executed",
            "coordinate_frame_qualification_required",
            "metric_scale_independent_validation_required",
        ],
    }
    request["colmap_training_dataset_export_request_digest"] = canonical_digest(
        request, digest_field="colmap_training_dataset_export_request_digest"
    )
    initialized_request = bind_colmap_initialization_surface(
        source_artifact=request,
        surface_compilation_result=surface_result,
    )
    evaluator_input_root = derived_root / "evaluator_input"
    evaluator_input = compile_canonical_3dgs_proxy_hidden_evaluator_input(
        reconstruction_dataset_manifest=reconstruction,
        dataset_artifact_root=proxy,
        evaluator_metric_scaffold=evaluator_scaffold,
        output_root=evaluator_input_root,
        source_commit_sha=source_commit_sha,
        authority_used=authority,
        timestamp=timestamp,
    )
    trainer_input_root = derived_root / "trainer_input"
    dataset = export_colmap_training_dataset(
        source_artifact=initialized_request,
        artifact_root=proxy,
        initialization_artifact_root=source_root,
        output_root=trainer_input_root,
    )
    dataset_root = trainer_input_root / str(dataset["relative_path"])
    _verify_colmap_dataset(dataset, dataset_root)

    source_admission = build_canonical_3dgs_source_admission(
        source_profile="public_dataset_arkitscenes_proxy",
        source_capture_identity=str(compilation["source_capture_identity"]),
        source_capture_digest=str(capture_digest),
        source_artifact_commit_sha=str(compilation["source_commit_sha"]),
        frozen_split_digest=str(split_digest),
        colmap_training_dataset_digest=dataset["colmap_training_dataset_digest"],
        hidden_evaluator_input_digest=evaluator_input[
            "canonical_3dgs_hidden_evaluator_input_digest"
        ],
        raw_contract_3_2_proven=False,
        world_frame="arkitscenes_official_loader_world",
        coordinate_frame_declaration=dict(
            surface_result.get("coordinate_frame_declaration") or {}
        ),
        metric_scale_status="sensor_metric_unvalidated",
        authority_used=authority,
        input_artifacts=[
            {
                "artifact_id": "arkitscenes_proxy_compilation",
                "digest": compilation["arkitscenes_proxy_compilation_digest"],
            },
            {
                "artifact_id": "reconstruction_dataset_manifest",
                "digest": reconstruction["dataset_manifest_digest"],
            },
            {
                "artifact_id": "observed_surface",
                "digest": surface_asset["digest"],
            },
            {
                "artifact_id": "colmap_training_dataset_export_result",
                "digest": dataset["colmap_training_dataset_export_result_digest"],
            },
            {
                "artifact_id": "hidden_evaluator_input",
                "digest": evaluator_input["canonical_3dgs_hidden_evaluator_input_digest"],
            },
        ],
        claim_limitations=[
            *list(compilation.get("blockers") or []),
            *list(compilation.get("warnings") or []),
            *list(dataset.get("blockers") or []),
            *list(dataset.get("warnings") or []),
        ],
        timestamp=timestamp,
    )
    _write_immutable_json(
        derived_root / "canonical_3dgs_source_admission.json", source_admission
    )
    preparation = {
        "schema_version": PROXY_PREPARATION_SCHEMA,
        "status": "training_dataset_ready",
        "source_profile": "public_dataset_arkitscenes_proxy",
        "canonical_3dgs_source_admission_digest": source_admission[
            "canonical_3dgs_source_admission_digest"
        ],
        "source_capture_identity": compilation["source_capture_identity"],
        "source_capture_digest": capture_digest,
        "pipeline_source_commit_sha": source_commit_sha,
        "raw_contract_version": None,
        "raw_contract_3_2_proven": False,
        "raw_capture_authority_preserved": True,
        "proxy_compilation_digest": compilation["arkitscenes_proxy_compilation_digest"],
        "arkit_depth_surface_compilation_result_digest": surface_result[
            "arkit_depth_surface_compilation_result_digest"
        ],
        "colmap_training_dataset_export_result_digest": dataset[
            "colmap_training_dataset_export_result_digest"
        ],
        "colmap_training_dataset_digest": dataset["colmap_training_dataset_digest"],
        "frozen_split_digest": dataset["frozen_split_digest"],
        "dataset_relative_path": dataset_root.relative_to(derived_root).as_posix(),
        "image_count": dataset["image_count"],
        "initialization_point_count": dataset["initialization_point_count"],
        "pose_binding": "arkitscenes_official_trajectory_exact_timestamp",
        "world_frame": "arkitscenes_official_loader_world",
        "coordinate_frame_declaration": source_admission[
            "coordinate_frame_declaration"
        ],
        "metric_scale_status": "sensor_metric_unvalidated",
        "hidden_heldout_pixels_included": False,
        "hidden_evaluator_input_digest": evaluator_input[
            "canonical_3dgs_hidden_evaluator_input_digest"
        ],
        "hidden_evaluator_input_relative_path": (
            evaluator_input_root / "canonical_3dgs_hidden_evaluator_input.json"
        ).relative_to(derived_root).as_posix(),
        "trainer_self_grading_permitted": False,
        "warnings": source_admission["claim_limitations"],
        "proof_effect": "trainer_input_materialization_only",
        "claim_ceiling": "reconstruction_training_request",
        "timestamp": timestamp,
    }
    preparation["canonical_v32_3dgs_preparation_digest"] = canonical_digest(
        preparation, digest_field="canonical_v32_3dgs_preparation_digest"
    )
    _write_immutable_json(
        derived_root / "canonical_arkitscenes_proxy_3dgs_preparation.json", preparation
    )
    _write_immutable_json(derived_root / "colmap_training_dataset_export_result.json", dataset)
    return {
        "preparation": preparation,
        "source_admission": source_admission,
        "dataset": dataset,
        "dataset_root": dataset_root,
    }


def build_canonical_3dgs_execution_plan(
    *,
    preparation: Mapping[str, Any],
    dataset: Mapping[str, Any],
    dataset_root: str | Path,
    source_commit_sha: str,
    timestamp: str | None = None,
) -> dict[str, Any]:
    root = Path(dataset_root).expanduser().resolve()
    _verify_colmap_dataset(dataset, root)
    if (
        preparation.get("schema_version") not in {PREPARATION_SCHEMA, PROXY_PREPARATION_SCHEMA}
        or preparation.get("canonical_v32_3dgs_preparation_digest")
        != canonical_digest(
            preparation, digest_field="canonical_v32_3dgs_preparation_digest"
        )
        or preparation.get("colmap_training_dataset_digest")
        != dataset.get("colmap_training_dataset_digest")
        or not _digest(preparation.get("canonical_3dgs_source_admission_digest"))
        or preparation.get("source_profile")
        not in {"blueprint_raw_v3_2", "public_dataset_arkitscenes_proxy"}
        or preparation.get("raw_contract_3_2_proven")
        != (preparation.get("source_profile") == "blueprint_raw_v3_2")
        or preparation.get("pipeline_source_commit_sha") != source_commit_sha
    ):
        raise Canonical3DGSPipelineError(["preparation_dataset_binding_invalid"])
    if len(source_commit_sha) != 40 or any(
        character not in "0123456789abcdef" for character in source_commit_sha
    ):
        raise Canonical3DGSPipelineError(["source_commit_sha_invalid"])
    input_artifacts = [
        {"relative_path": row["relative_path"], "digest": row["digest"]}
        for row in dataset.get("output_artifacts") or []
    ]
    plan = {
        "schema_version": PLAN_SCHEMA,
        "status": "ready_for_authorized_workers",
        "source_capture_digest": dataset["source_capture_digest"],
        "source_commit_sha": source_commit_sha,
        "worker_python_package_digest": canonical_3dgs_worker_package_digest(),
        "source_profile": preparation.get("source_profile"),
        "canonical_3dgs_source_admission_digest": preparation.get(
            "canonical_3dgs_source_admission_digest"
        ),
        "raw_contract_3_2_proven": preparation.get("raw_contract_3_2_proven") is True,
        "canonical_v32_3dgs_preparation_digest": preparation[
            "canonical_v32_3dgs_preparation_digest"
        ],
        "colmap_training_dataset_digest": dataset["colmap_training_dataset_digest"],
        "colmap_training_dataset_export_result_digest": dataset[
            "colmap_training_dataset_export_result_digest"
        ],
        "frozen_split_digest": dataset["frozen_split_digest"],
        "hidden_evaluator_input_digest": preparation[
            "hidden_evaluator_input_digest"
        ],
        "input_artifacts": input_artifacts,
        "image_count": dataset["image_count"],
        "initialization_point_count": dataset["initialization_point_count"],
        "pose_binding": preparation["pose_binding"],
        "world_frame": preparation.get("world_frame", "canonical_arkit_world"),
        "coordinate_frame_declaration": dict(
            preparation.get("coordinate_frame_declaration") or {}
        ),
        "metric_scale_status": str(
            preparation.get("metric_scale_status") or "sensor_metric_unvalidated"
        ),
        "primary_method_id": POSTSHOT_METHOD,
        "comparison_method_ids": [SPLATFACTO_METHOD],
        "arms": [
            {
                "arm_id": "postshot-primary",
                "role": "primary",
                "method_id": POSTSHOT_METHOD,
                "runtime": {
                    "platform": "windows",
                    "product": "Jawset Postshot CLI",
                    "profile": "Splat3",
                    "installer_and_license_must_pass_worker_admission": True,
                },
                "train_argv_template": [
                    "postshot-cli.exe",
                    "<runtime-auth-flags>",
                    "train",
                    "--import",
                    "<candidate-colmap-dataset>",
                    "--profile",
                    "Splat3",
                    "--no-recenter-points",
                    "--max-image-size",
                    "0",
                    "--output",
                    "<output-project.psht>",
                    "--export-splat",
                    "<output-splat.ply>",
                ],
                "required_outputs": ["standard_3dgs_ply", "postshot_project", "training_log"],
            },
            {
                "arm_id": "splatfacto-comparison",
                "role": "comparison",
                "method_id": SPLATFACTO_METHOD,
                "runtime": {
                    "platform": "linux",
                    "product": "Nerfstudio Splatfacto",
                    "version": "1.1.5",
                    "gsplat_version": "1.4.0",
                },
                "train_argv_template": [
                    "ns-train",
                    "splatfacto",
                    "--max-num-iterations",
                    "30000",
                    "--machine.seed",
                    "42",
                    "--pipeline.model.cull_alpha_thresh=0.005",
                    "--pipeline.model.stop_split_at",
                    "15000",
                    "--data",
                    "<candidate-colmap-dataset>",
                    "colmap",
                    "--colmap-path",
                    "sparse/0",
                    "--downscale-factor",
                    "1",
                    "--orientation-method",
                    "none",
                    "--center-method",
                    "none",
                    "--auto-scale-poses",
                    "False",
                    "--assume-colmap-world-coordinate-convention",
                    "False",
                    "--eval-mode",
                    "all",
                ],
                "export_argv_template": [
                    "ns-export",
                    "gaussian-splat",
                    "--load-config",
                    "<config.yml>",
                    "--output-dir",
                    "<export-directory>",
                ],
                "required_outputs": ["standard_3dgs_ply", "nerfstudio_config", "training_log"],
            },
        ],
        "same_candidate_dataset_required": True,
        "hidden_heldout_pixels_included": False,
        "trainer_self_grading_permitted": False,
        "independent_heldout_evaluation_required": True,
        "quality_winner": None,
        "proof_effect": "authorized_candidate_training_plan_only",
        "claim_ceiling": "appearance_reconstruction_candidate",
        "timestamp": timestamp or str(preparation["timestamp"]),
    }
    plan["canonical_3dgs_execution_plan_digest"] = canonical_digest(
        plan, digest_field="canonical_3dgs_execution_plan_digest"
    )
    return plan


def execute_canonical_3dgs_plan(
    *,
    plan: Mapping[str, Any],
    dataset_root: str | Path,
    output_root: str | Path,
    runners: Mapping[str, ArmRunner],
    require_external_worker_controls: bool = False,
) -> dict[str, Any]:
    """Execute both precommitted arms through trusted worker bindings."""

    data_root = verify_canonical_3dgs_plan_inputs(plan=plan, dataset_root=dataset_root)
    destination = Path(output_root).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for arm_value in plan.get("arms") or []:
        arm = dict(arm_value)
        arm_id = str(arm.get("arm_id") or "")
        runner = runners.get(arm_id)
        if runner is None:
            raise Canonical3DGSPipelineError([f"trusted_runner_missing:{arm_id}"])
        run_root = destination / arm_id
        if run_root.is_symlink():
            raise Canonical3DGSPipelineError([f"run_root_symlink_forbidden:{arm_id}"])
        run_root.mkdir(parents=True, exist_ok=True)
        receipt = dict(runner(arm, data_root, run_root))
        worker_control_binding = _validate_worker_control_binding(
            receipt=receipt,
            run_root=run_root,
            arm_id=arm_id,
            plan=plan,
            required=require_external_worker_controls,
        )
        receipt_plan_digest = receipt.get("canonical_3dgs_execution_plan_digest")
        if receipt_plan_digest not in {
            None,
            plan["canonical_3dgs_execution_plan_digest"],
        }:
            raise Canonical3DGSPipelineError([f"runner_plan_digest_mismatch:{arm_id}"])
        exit_code = receipt.get("exit_code")
        if isinstance(exit_code, bool) or not isinstance(exit_code, int):
            raise Canonical3DGSPipelineError([f"runner_exit_code_invalid:{arm_id}"])
        artifacts: list[dict[str, Any]] = []
        kinds: set[str] = set()
        splat_profile: dict[str, Any] | None = None
        for row in receipt.get("artifacts") or []:
            if not isinstance(row, Mapping):
                raise Canonical3DGSPipelineError([f"runner_artifact_invalid:{arm_id}"])
            path = _safe_child(
                run_root,
                row.get("relative_path"),
                code=f"runner_artifact_unsafe:{arm_id}",
            )
            digest = _sha256(path)
            if row.get("digest") not in {None, digest}:
                raise Canonical3DGSPipelineError([f"runner_artifact_digest_mismatch:{arm_id}"])
            kind = str(row.get("kind") or "")
            if kind == "standard_3dgs_ply":
                try:
                    splat = read_standard_3dgs_ply(path)
                except (OSError, TypeError, ValueError) as exc:
                    raise Canonical3DGSPipelineError(
                        [f"runner_standard_3dgs_invalid:{arm_id}"]
                    ) from exc
                if splat.count < 1:
                    raise Canonical3DGSPipelineError(
                        [f"runner_standard_3dgs_empty:{arm_id}"]
                    )
                splat_profile = _standard_splat_profile(splat)
            kinds.add(kind)
            artifacts.append(
                {
                    "kind": kind,
                    "relative_path": path.relative_to(run_root).as_posix(),
                    "digest": digest,
                    "size_bytes": path.stat().st_size,
                }
            )
        required = set(str(value) for value in arm.get("required_outputs") or [])
        status = "succeeded" if exit_code == 0 and required <= kinds else "failed"
        result = {
            "schema_version": ARM_RESULT_SCHEMA,
            "status": status,
            "arm_id": arm_id,
            "role": arm["role"],
            "method_id": arm["method_id"],
            "source_capture_digest": plan["source_capture_digest"],
            "source_commit_sha": plan["source_commit_sha"],
            "canonical_3dgs_execution_plan_digest": plan[
                "canonical_3dgs_execution_plan_digest"
            ],
            "colmap_training_dataset_digest": plan["colmap_training_dataset_digest"],
            "frozen_split_digest": plan["frozen_split_digest"],
            "runtime_identity": dict(receipt.get("runtime_identity") or {}),
            "worker_control_binding": worker_control_binding,
            "command_digest": canonical_digest(
                {"argv": receipt.get("argv") or [], "method_id": arm["method_id"]}
            ),
            "exit_code": exit_code,
            "artifacts": artifacts,
            "standard_3dgs_profile": splat_profile,
            "hidden_heldout_pixels_included": False,
            "candidate_self_graded": False,
            "quality_claimed": False,
            "proof_effect": "appearance_asset_candidate_only",
            "claim_ceiling": "appearance_reconstruction",
            "blockers": [] if status == "succeeded" else ["trainer_execution_or_output_failed"],
            "timestamp": str(receipt.get("timestamp") or plan["timestamp"]),
        }
        result["canonical_3dgs_arm_result_digest"] = canonical_digest(
            result, digest_field="canonical_3dgs_arm_result_digest"
        )
        _write_immutable_json(run_root / "canonical_3dgs_arm_result.json", result)
        results.append(result)
    return _finalize_campaign(plan=plan, destination=destination, results=results)


def _validate_worker_control_binding(
    *,
    receipt: Mapping[str, Any],
    run_root: Path,
    arm_id: str,
    plan: Mapping[str, Any],
    required: bool,
) -> dict[str, Any]:
    """Revalidate external-worker authority without treating it as quality proof."""

    if not required:
        return {
            "status": "trusted_in_process_runner",
            "external_worker_admission_proven": False,
            "provider_zero_required_after_execution": False,
            "proof_effect": "test_execution_binding_only",
        }

    # These imports stay local because transport validation itself imports this
    # module's plan verifier.
    from .canonical_3dgs_admission import (
        Canonical3DGSAdmissionError,
        require_canonical_3dgs_worker_admission,
    )
    from .canonical_3dgs_transport import (
        Canonical3DGSTransportError,
        validate_canonical_3dgs_transport_receipt,
    )

    errors: list[str] = []
    if receipt.get("canonical_3dgs_worker_receipt_digest") != canonical_digest(
        receipt, digest_field="canonical_3dgs_worker_receipt_digest"
    ):
        errors.append(f"worker_receipt_digest_mismatch:{arm_id}")
    try:
        transport_path = _safe_child(
            run_root,
            receipt.get("transport_receipt_relative_path"),
            code=f"worker_transport_receipt_path_invalid:{arm_id}",
        )
        transport = validate_canonical_3dgs_transport_receipt(
            _load_json(transport_path, code=f"worker_transport_receipt_invalid:{arm_id}")
        )
    except (Canonical3DGSPipelineError, Canonical3DGSTransportError):
        transport = {}
        errors.append(f"worker_transport_receipt_invalid:{arm_id}")
    expected_transport = {
        "canonical_3dgs_execution_plan_digest": plan.get(
            "canonical_3dgs_execution_plan_digest"
        ),
        "colmap_training_dataset_digest": plan.get(
            "colmap_training_dataset_digest"
        ),
        "source_capture_digest": plan.get("source_capture_digest"),
        "frozen_split_digest": plan.get("frozen_split_digest"),
    }
    for key, expected in expected_transport.items():
        if transport.get(key) != expected:
            errors.append(f"worker_transport_binding_mismatch:{arm_id}:{key}")
    try:
        admission_path = _safe_child(
            run_root,
            receipt.get("worker_admission_relative_path"),
            code=f"worker_admission_path_invalid:{arm_id}",
        )
        admission = require_canonical_3dgs_worker_admission(
            _load_json(admission_path, code=f"worker_admission_invalid:{arm_id}"),
            arm_id=arm_id,
            plan_digest=str(plan.get("canonical_3dgs_execution_plan_digest") or ""),
            dataset_digest=str(plan.get("colmap_training_dataset_digest") or ""),
            transport_bundle_digest=str(transport.get("transport_bundle_digest") or ""),
            worker_package_digest=str(plan.get("worker_python_package_digest") or ""),
            observed_now=datetime.fromisoformat(
                str(receipt.get("timestamp") or "").replace("Z", "+00:00")
            ),
        )
    except (Canonical3DGSPipelineError, Canonical3DGSAdmissionError, ValueError):
        admission = {}
        errors.append(f"worker_admission_invalid:{arm_id}")
    expected_receipt = {
        "canonical_3dgs_execution_plan_digest": plan.get(
            "canonical_3dgs_execution_plan_digest"
        ),
        "transport_bundle_digest": transport.get("transport_bundle_digest"),
        "transport_receipt_digest": transport.get("receipt_digest"),
        "canonical_3dgs_worker_admission_digest": admission.get(
            "canonical_3dgs_worker_admission_digest"
        ),
        "allocation_binding_digest": admission.get("allocation_binding_digest"),
        "provider_zero_required_after_execution": True,
    }
    for key, expected in expected_receipt.items():
        if receipt.get(key) != expected:
            errors.append(f"worker_receipt_binding_mismatch:{arm_id}:{key}")
    runtime_identity = receipt.get("runtime_identity")
    if not isinstance(runtime_identity, Mapping) or runtime_identity.get(
        "worker_python_package_digest"
    ) != plan.get("worker_python_package_digest"):
        errors.append(f"worker_runtime_package_digest_mismatch:{arm_id}")
    if not isinstance(runtime_identity, Mapping) or runtime_identity.get(
        "source_commit_sha_bound_by_plan"
    ) != plan.get("source_commit_sha"):
        errors.append(f"worker_runtime_source_commit_binding_mismatch:{arm_id}")
    if not isinstance(runtime_identity, Mapping) or runtime_identity.get(
        "trainer_runtime_digest"
    ) != admission.get("trainer_runtime_digest"):
        errors.append(f"worker_trainer_runtime_digest_mismatch:{arm_id}")
    if not isinstance(runtime_identity, Mapping) or runtime_identity.get(
        "trainer_runtime_version"
    ) != admission.get("trainer_runtime_version"):
        errors.append(f"worker_trainer_runtime_version_mismatch:{arm_id}")
    if errors:
        raise Canonical3DGSPipelineError(errors)
    return {
        "status": "external_worker_admission_bound",
        "external_worker_admission_proven": True,
        "transport_bundle_digest": transport["transport_bundle_digest"],
        "transport_receipt_digest": transport["receipt_digest"],
        "worker_admission_digest": admission[
            "canonical_3dgs_worker_admission_digest"
        ],
        "allocation_binding_digest": admission["allocation_binding_digest"],
        "trainer_runtime_digest": admission["trainer_runtime_digest"],
        "trainer_runtime_version": admission["trainer_runtime_version"],
        "authority_id": admission["authority_id"],
        "max_spend_usd": admission["max_spend_usd"],
        "hard_ttl_seconds": admission["hard_ttl_seconds"],
        "retry_cap": 0,
        "provider_zero_required_after_execution": True,
        "provider_zero_verified_after_execution": False,
        "proof_effect": "worker_execution_authority_only",
    }


def verify_canonical_3dgs_plan_inputs(
    *, plan: Mapping[str, Any], dataset_root: str | Path
) -> Path:
    """Revalidate the plan and every training byte before a worker starts."""

    if (
        plan.get("schema_version") != PLAN_SCHEMA
        or plan.get("canonical_3dgs_execution_plan_digest")
        != canonical_digest(plan, digest_field="canonical_3dgs_execution_plan_digest")
        or plan.get("worker_python_package_digest")
        != canonical_3dgs_worker_package_digest()
        or plan.get("source_profile")
        not in {"blueprint_raw_v3_2", "public_dataset_arkitscenes_proxy"}
        or not _digest(plan.get("canonical_3dgs_source_admission_digest"))
        or not _digest(plan.get("hidden_evaluator_input_digest"))
        or not str(plan.get("world_frame") or "").strip()
        or not isinstance(plan.get("coordinate_frame_declaration"), Mapping)
        or plan.get("metric_scale_status")
        not in {
            "sensor_metric_unvalidated",
            "independently_validated_metric",
            "not_established",
            "unknown",
        }
        or plan.get("hidden_heldout_pixels_included") is not False
        or plan.get("trainer_self_grading_permitted") is not False
    ):
        raise Canonical3DGSPipelineError(["execution_plan_invalid"])
    data_root = Path(dataset_root).expanduser().resolve()
    for row in plan.get("input_artifacts") or []:
        path = _safe_child(
            data_root,
            row.get("relative_path"),
            code="execution_input_artifact_unsafe",
        )
        if _sha256(path) != row.get("digest"):
            raise Canonical3DGSPipelineError(["execution_input_artifact_digest_mismatch"])
    return data_root


def _camera_axis_projection(plan: Mapping[str, Any]) -> str:
    if plan.get("source_profile") == "public_dataset_arkitscenes_proxy":
        return "source_opencv_preserved_no_axis_flip"
    return "arkit_to_opencv_explicit_yz_flip"


def _finalize_campaign(
    *, plan: Mapping[str, Any], destination: Path, results: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    primary = next((row for row in results if row["role"] == "primary"), None)
    comparisons = [row for row in results if row["role"] == "comparison"]
    completed = primary is not None and primary["status"] == "succeeded" and all(
        row["status"] == "succeeded" for row in comparisons
    )
    coordinate_basis_digest = canonical_digest(
        {
            "source_capture_digest": plan["source_capture_digest"],
            "colmap_training_dataset_digest": plan["colmap_training_dataset_digest"],
            "pose_binding": plan["pose_binding"],
            "world_frame": plan.get("world_frame"),
            "coordinate_frame_declaration": plan.get("coordinate_frame_declaration"),
            "camera_axis_projection": _camera_axis_projection(plan),
            "units": "meters",
            "metric_scale_status": plan.get("metric_scale_status"),
        }
    )
    fidelity_bindings = []
    for row in results:
        splat_artifacts = [
            artifact
            for artifact in row.get("artifacts") or []
            if artifact.get("kind") == "standard_3dgs_ply"
        ]
        if len(splat_artifacts) != 1 or not isinstance(
            row.get("standard_3dgs_profile"), Mapping
        ):
            continue
        fidelity_bindings.append(
            {
                "candidate_arm_id": row["arm_id"],
                "candidate_method_id": row["method_id"],
                "candidate_role": row["role"],
                "candidate_result_digest": row["canonical_3dgs_arm_result_digest"],
                "asset_digest": splat_artifacts[0]["digest"],
                "coordinate_basis_digest": coordinate_basis_digest,
                **dict(row["standard_3dgs_profile"]),
                "source_appearance_immutable": True,
                "reference_frame_comparison_status": "required_not_executed",
            }
        )
    external_controls = [
        dict(row.get("worker_control_binding") or {}) for row in results
    ]
    campaign = {
        "schema_version": CAMPAIGN_RESULT_SCHEMA,
        "status": (
            "candidates_ready_for_independent_evaluation"
            if completed
            else "candidate_training_incomplete"
        ),
        "source_capture_digest": plan["source_capture_digest"],
        "source_commit_sha": plan["source_commit_sha"],
        "source_profile": plan.get("source_profile"),
        "canonical_3dgs_source_admission_digest": plan.get(
            "canonical_3dgs_source_admission_digest"
        ),
        "world_frame": plan.get("world_frame"),
        "coordinate_frame_declaration": dict(
            plan.get("coordinate_frame_declaration") or {}
        ),
        "metric_scale_status": plan.get("metric_scale_status"),
        "metric_scale_independently_validated": (
            plan.get("metric_scale_status") == "independently_validated_metric"
        ),
        "canonical_3dgs_execution_plan_digest": plan[
            "canonical_3dgs_execution_plan_digest"
        ],
        "colmap_training_dataset_digest": plan["colmap_training_dataset_digest"],
        "frozen_split_digest": plan["frozen_split_digest"],
        "hidden_evaluator_input_digest": plan["hidden_evaluator_input_digest"],
        "primary_method_id": plan["primary_method_id"],
        "primary_result_digest": (
            primary["canonical_3dgs_arm_result_digest"] if primary else None
        ),
        "comparison_result_digests": [
            row["canonical_3dgs_arm_result_digest"] for row in comparisons
        ],
        "all_arms_used_identical_candidate_dataset": all(
            row["colmap_training_dataset_digest"]
            == plan["colmap_training_dataset_digest"]
            for row in results
        ),
        "hidden_heldout_pixels_included": False,
        "independent_heldout_evaluation_status": "required_not_executed",
        "appearance_fidelity_candidate_bindings": fidelity_bindings,
        "execution_control_summary": {
            "all_external_workers_admitted": bool(external_controls)
            and all(
                row.get("status") == "external_worker_admission_bound"
                for row in external_controls
            ),
            "control_modes": sorted(
                {str(row.get("status") or "unknown") for row in external_controls}
            ),
            "allocation_binding_digests": sorted(
                str(row["allocation_binding_digest"])
                for row in external_controls
                if row.get("allocation_binding_digest")
            ),
            "provider_zero_required_after_execution": any(
                row.get("provider_zero_required_after_execution") is True
                for row in external_controls
            ),
            "provider_zero_verified_after_execution": False,
            "resource_closeout_is_quality_evidence": False,
        },
        "next_quality_gate": {
            "schema_version": "appearance_fidelity_qualification.v1",
            "required_metrics": ["ssim", "psnr_db", "lpips"],
            "native_3dgs_exact_camera_render_required": True,
            "site_task_specific_thresholds_required": True,
            "default_thresholds_assumed": False,
            "selection_allowed_before_measurement": False,
        },
        "quality_winner": None,
        "raw_capture_authority_upgraded": False,
        "metric_collision_or_physical_claim_upgraded": False,
        "proof_effect": "candidate_generation_only",
        "claim_ceiling": "appearance_reconstruction_candidates",
        "timestamp": plan["timestamp"],
    }
    campaign["canonical_3dgs_campaign_result_digest"] = canonical_digest(
        campaign, digest_field="canonical_3dgs_campaign_result_digest"
    )
    _write_immutable_json(destination / "canonical_3dgs_campaign_result.json", campaign)
    return {"campaign": campaign, "arm_results": list(results)}


def finalize_canonical_3dgs_receipts(
    *, plan: Mapping[str, Any], dataset_root: str | Path, results_root: str | Path
) -> dict[str, Any]:
    """Normalize already-executed platform-worker receipts into one campaign."""

    root = Path(results_root).expanduser().resolve()

    def receipt_runner(arm: Mapping[str, Any], _: Path, run_root: Path) -> Mapping[str, Any]:
        receipt = _load_json(
            run_root / "worker_receipt.json",
            code=f"worker_receipt_invalid:{arm.get('arm_id')}",
        )
        return receipt

    return execute_canonical_3dgs_plan(
        plan=plan,
        dataset_root=dataset_root,
        output_root=root,
        runners={
            "postshot-primary": receipt_runner,
            "splatfacto-comparison": receipt_runner,
        },
        require_external_worker_controls=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-profile",
        choices=("blueprint_raw_v3_2", "public_dataset_arkitscenes_proxy"),
        default="blueprint_raw_v3_2",
    )
    parser.add_argument("--capture-root")
    parser.add_argument("--proxy-root")
    parser.add_argument("--source-artifact-root")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--intake-id")
    parser.add_argument("--capture-digest")
    parser.add_argument("--source-commit-sha", required=True)
    parser.add_argument("--maximum-frames", type=int, default=60)
    arguments = parser.parse_args(argv)
    if arguments.source_profile == "blueprint_raw_v3_2":
        if not all((arguments.capture_root, arguments.intake_id, arguments.capture_digest)):
            parser.error(
                "blueprint_raw_v3_2 requires --capture-root, --intake-id, and --capture-digest"
            )
        prepared = prepare_canonical_v32_training_dataset(
            capture_root=arguments.capture_root,
            output_root=arguments.output_root,
            intake_id=arguments.intake_id,
            capture_digest=arguments.capture_digest,
            rights_and_retention={
                "local_processing_authorized": True,
                "provider_upload_authorized": False,
                "paid_compute_authorized": False,
            },
            maximum_frames=arguments.maximum_frames,
        )
    else:
        if not all((arguments.proxy_root, arguments.source_artifact_root)):
            parser.error(
                "public_dataset_arkitscenes_proxy requires --proxy-root and "
                "--source-artifact-root"
            )
        prepared = prepare_canonical_arkitscenes_proxy_training_dataset(
            proxy_root=arguments.proxy_root,
            source_artifact_root=arguments.source_artifact_root,
            output_root=arguments.output_root,
            source_commit_sha=arguments.source_commit_sha,
        )
    plan = build_canonical_3dgs_execution_plan(
        preparation=prepared["preparation"],
        dataset=prepared["dataset"],
        dataset_root=prepared["dataset_root"],
        source_commit_sha=arguments.source_commit_sha,
    )
    destination = Path(arguments.output_root).resolve() / "canonical_3dgs_execution_plan.json"
    _write_immutable_json(destination, plan)
    print(
        json.dumps(
            {
                "status": plan["status"],
                "plan": str(destination),
                "plan_digest": plan["canonical_3dgs_execution_plan_digest"],
                "primary_method_id": plan["primary_method_id"],
                "comparison_method_ids": plan["comparison_method_ids"],
            },
            indent=2,
        )
    )
    return 0


def finalize_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Finalize canonical 3DGS worker receipts")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--results-root", required=True)
    arguments = parser.parse_args(argv)
    plan = _load_json(Path(arguments.plan), code="execution_plan_invalid")
    result = finalize_canonical_3dgs_receipts(
        plan=plan,
        dataset_root=arguments.dataset_root,
        results_root=arguments.results_root,
    )
    print(json.dumps(result["campaign"], indent=2, sort_keys=True))
    return 0 if result["campaign"]["status"] == "candidates_ready_for_independent_evaluation" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARM_RESULT_SCHEMA",
    "CAMPAIGN_RESULT_SCHEMA",
    "Canonical3DGSPipelineError",
    "PLAN_SCHEMA",
    "POSTSHOT_METHOD",
    "PREPARATION_SCHEMA",
    "SPLATFACTO_METHOD",
    "build_canonical_3dgs_execution_plan",
    "build_canonical_3dgs_source_admission",
    "execute_canonical_3dgs_plan",
    "finalize_canonical_3dgs_receipts",
    "prepare_canonical_v32_training_dataset",
    "prepare_canonical_arkitscenes_proxy_training_dataset",
    "verify_canonical_3dgs_plan_inputs",
]
