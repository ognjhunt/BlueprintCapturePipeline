"""Evaluator-owned exact-camera input for canonical V3.2 3DGS campaigns."""

from __future__ import annotations

import hashlib
import argparse
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .decision_evidence_contracts import canonical_digest, canonical_json
from .appearance_fidelity import build_appearance_fidelity_qualification
from .heldout_appearance_evaluation_v2 import (
    build_heldout_appearance_evaluation_request_v2,
    evaluate_heldout_appearance_v2,
)
from .sealed_camera_render import render_splat_at_exact_cameras


EVALUATOR_INPUT_SCHEMA = "canonical_3dgs_hidden_evaluator_input.v1"
QUALITY_COMPARISON_SCHEMA = "canonical_3dgs_quality_comparison.v1"
ARKIT_TO_OPENCV = np.diag([1.0, -1.0, -1.0])


class Canonical3DGSEvaluationError(ValueError):
    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__(";".join(self.codes))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _safe_file(root: Path, value: Any, *, code: str) -> Path:
    path = PurePosixPath(str(value or "").replace("\\", "/"))
    if (
        not str(value or "")
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise Canonical3DGSEvaluationError([code])
    candidate = root.joinpath(*path.parts)
    if candidate.is_symlink():
        raise Canonical3DGSEvaluationError([code])
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise Canonical3DGSEvaluationError([code]) from exc
    if not resolved.is_file() or (resolved != root and root not in resolved.parents):
        raise Canonical3DGSEvaluationError([code])
    return resolved


def _load_json(path: Path, *, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Canonical3DGSEvaluationError([code]) from exc
    if not isinstance(value, Mapping):
        raise Canonical3DGSEvaluationError([code])
    return dict(value)


def _load_jsonl(path: Path, *, code: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(code)
            rows.append(dict(value))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise Canonical3DGSEvaluationError([code]) from exc
    return rows


def _reference(
    dataset_manifest: Mapping[str, Any], name: str, dataset_root: Path
) -> tuple[Path, dict[str, Any]]:
    references = dataset_manifest.get("artifact_references")
    reference = references.get(name) if isinstance(references, Mapping) else None
    if not isinstance(reference, Mapping):
        raise Canonical3DGSEvaluationError([f"evaluator_reference_missing:{name}"])
    path = _safe_file(
        dataset_root,
        reference.get("relative_path"),
        code=f"evaluator_reference_path_invalid:{name}",
    )
    if _sha256(path) != reference.get("digest"):
        raise Canonical3DGSEvaluationError(
            [f"evaluator_reference_byte_digest_mismatch:{name}"]
        )
    return path, _load_json(path, code=f"evaluator_reference_json_invalid:{name}")


def _immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (canonical_json(dict(value)) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise Canonical3DGSEvaluationError(["evaluator_immutable_conflict"])
        return
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _copy_immutable(source: Path, destination: Path, expected_digest: str) -> None:
    if _sha256(source) != expected_digest:
        raise Canonical3DGSEvaluationError(["evaluator_hidden_frame_digest_mismatch"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if _sha256(destination) != expected_digest:
            raise Canonical3DGSEvaluationError(["evaluator_immutable_conflict"])
        return
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _opencv_camera_to_world(value: Any) -> list[list[float]]:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise Canonical3DGSEvaluationError(["evaluator_arkit_pose_invalid"])
    converted = matrix.copy()
    converted[:3, :3] = matrix[:3, :3] @ ARKIT_TO_OPENCV
    return [[float(item) for item in row] for row in converted]


def compile_canonical_3dgs_hidden_evaluator_input(
    *,
    capture_root: str | Path,
    reconstruction_dataset_manifest: Mapping[str, Any],
    dataset_artifact_root: str | Path,
    output_root: str | Path,
    source_commit_sha: str,
    authority_used: Mapping[str, Any],
    timestamp: str,
) -> dict[str, Any]:
    """Materialize hidden RGB and exact cameras outside the trainer transport."""

    capture = Path(capture_root).expanduser().resolve()
    dataset_root = Path(dataset_artifact_root).expanduser().resolve()
    hidden_path, hidden = _reference(
        reconstruction_dataset_manifest,
        "hidden_heldout_evaluator_manifest",
        dataset_root,
    )
    _, split = _reference(
        reconstruction_dataset_manifest,
        "frozen_split_manifest",
        dataset_root,
    )
    if (
        hidden.get("schema_version") != "hidden_heldout_evaluator_manifest.v1"
        or hidden.get("access_scope") != "independent_evaluator_only"
        or hidden.get("candidate_method_access_allowed") is not False
        or hidden.get("hidden_heldout_digest")
        != canonical_digest(hidden, digest_field="hidden_heldout_digest")
        or split.get("split_digest")
        != canonical_digest(split, digest_field="split_digest")
        or hidden.get("split_digest") != split.get("split_digest")
    ):
        raise Canonical3DGSEvaluationError(["evaluator_hidden_split_invalid"])
    hidden_rows = hidden.get("frames")
    if not isinstance(hidden_rows, list) or not hidden_rows:
        raise Canonical3DGSEvaluationError(["evaluator_hidden_frames_missing"])
    sync_rows = _load_jsonl(capture / "sync_map.jsonl", code="evaluator_sync_map_invalid")
    poses = _load_jsonl(capture / "arkit/poses.jsonl", code="evaluator_poses_invalid")
    intrinsics_value = _load_json(
        capture / "arkit/session_intrinsics.json",
        code="evaluator_intrinsics_invalid",
    )
    intrinsics = intrinsics_value.get("intrinsics")
    if not isinstance(intrinsics, Mapping):
        raise Canonical3DGSEvaluationError(["evaluator_intrinsics_invalid"])
    try:
        camera_intrinsics = {
            key: float(intrinsics[key])
            for key in ("fx", "fy", "cx", "cy")
        } | {
            "width": int(intrinsics["width"]),
            "height": int(intrinsics["height"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise Canonical3DGSEvaluationError(["evaluator_intrinsics_invalid"]) from exc
    pose_by_id = {str(row.get("frame_id")): row for row in poses}
    destination = Path(output_root).expanduser().resolve()
    cameras: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in hidden_rows:
        if not isinstance(raw, Mapping):
            raise Canonical3DGSEvaluationError(["evaluator_hidden_frame_invalid"])
        camera_id = str(raw.get("frame_id") or "")
        timestamp_value = float(raw.get("t_video_sec"))
        matches = [
            row
            for row in sync_rows
            if abs(float(row.get("t_video_sec")) - timestamp_value) <= 1e-6
            and row.get("sync_status") == "encoded_decoded_pts_match"
        ]
        if not camera_id or camera_id in seen or len(matches) != 1:
            raise Canonical3DGSEvaluationError(
                [f"evaluator_hidden_camera_sync_invalid:{camera_id or 'unknown'}"]
            )
        seen.add(camera_id)
        pose = pose_by_id.get(str(matches[0].get("pose_frame_id") or ""))
        if not isinstance(pose, Mapping):
            raise Canonical3DGSEvaluationError(
                [f"evaluator_hidden_pose_missing:{camera_id}"]
            )
        source = _safe_file(
            hidden_path.parent.parent,
            raw.get("evaluator_relative_path"),
            code=f"evaluator_hidden_frame_path_invalid:{camera_id}",
        )
        relative = f"reference_frames/{camera_id}.png"
        _copy_immutable(source, destination / relative, str(raw.get("frame_digest") or ""))
        cameras.append(
            {
                "camera_id": camera_id,
                "trajectory": "author_heldout",
                "t_video_sec": timestamp_value,
                "capture_pose_frame_id": str(matches[0].get("pose_frame_id")),
                "T_world_camera_provider_frame": _opencv_camera_to_world(
                    pose.get("T_world_camera")
                ),
                "intrinsics": camera_intrinsics,
                "reference_relative_path": relative,
                "reference_digest": str(raw.get("frame_digest")),
                "excluded_from_training": True,
            }
        )
    result = {
        "schema_version": EVALUATOR_INPUT_SCHEMA,
        "status": "ready_for_independent_exact_camera_evaluation",
        "source_capture_identity": reconstruction_dataset_manifest[
            "source_capture_identity"
        ],
        "source_capture_digest": reconstruction_dataset_manifest[
            "source_capture_digest"
        ],
        "source_commit_sha": source_commit_sha,
        "reconstruction_dataset_digest": reconstruction_dataset_manifest[
            "dataset_manifest_digest"
        ],
        "frozen_split_digest": split["split_digest"],
        "hidden_heldout_digest": hidden["hidden_heldout_digest"],
        "camera_axis_convention": "opencv_x_right_y_down_z_forward",
        "world_frame": "canonical_arkit_world",
        "units": "meters",
        "cameras": cameras,
        "camera_count": len(cameras),
        "candidate_access_allowed": False,
        "trainer_transport_contains_this_manifest": False,
        "trainer_transport_contains_hidden_pixels": False,
        "authority_used": dict(authority_used),
        "proof_effect": "evaluator_input_only",
        "claim_ceiling": "independent_appearance_evaluation_request",
        "timestamp": timestamp,
    }
    result["canonical_3dgs_hidden_evaluator_input_digest"] = canonical_digest(
        result, digest_field="canonical_3dgs_hidden_evaluator_input_digest"
    )
    _immutable_json(destination / "canonical_3dgs_hidden_evaluator_input.json", result)
    return result


def _validated_evaluator_input(value: Mapping[str, Any]) -> dict[str, Any]:
    result = json.loads(canonical_json(dict(value)))
    if (
        result.get("schema_version") != EVALUATOR_INPUT_SCHEMA
        or result.get("status") != "ready_for_independent_exact_camera_evaluation"
        or result.get("canonical_3dgs_hidden_evaluator_input_digest")
        != canonical_digest(
            result, digest_field="canonical_3dgs_hidden_evaluator_input_digest"
        )
        or result.get("candidate_access_allowed") is not False
        or result.get("trainer_transport_contains_hidden_pixels") is not False
        or not isinstance(result.get("cameras"), list)
        or not result["cameras"]
        or result.get("camera_count") != len(result["cameras"])
    ):
        raise Canonical3DGSEvaluationError(["canonical_quality_evaluator_input_invalid"])
    return result


def _validated_campaign(value: Mapping[str, Any]) -> dict[str, Any]:
    campaign = json.loads(canonical_json(dict(value)))
    if (
        campaign.get("schema_version") != "canonical_3dgs_campaign_result.v1"
        or campaign.get("status") != "candidates_ready_for_independent_evaluation"
        or campaign.get("canonical_3dgs_campaign_result_digest")
        != canonical_digest(
            campaign, digest_field="canonical_3dgs_campaign_result_digest"
        )
        or campaign.get("quality_winner") is not None
        or campaign.get("hidden_heldout_pixels_included") is not False
    ):
        raise Canonical3DGSEvaluationError(["canonical_quality_campaign_invalid"])
    return campaign


def _safe_result_artifact(root: Path, arm_id: str, value: Any, digest: str) -> Path:
    arm_root = root / arm_id
    path = _safe_file(
        arm_root,
        value,
        code=f"canonical_quality_candidate_path_invalid:{arm_id}",
    )
    if _sha256(path) != digest:
        raise Canonical3DGSEvaluationError(
            [f"canonical_quality_candidate_digest_mismatch:{arm_id}"]
        )
    return path


def _aggregate_digest(rows: Sequence[Mapping[str, Any]], key: str) -> str:
    return canonical_digest(
        {"digests": [str(row[key]) for row in sorted(rows, key=lambda item: str(item["view_id"]))]}
    )


RenderExecutor = Callable[..., Mapping[str, Any]]
AppearanceEvaluator = Callable[..., Mapping[str, Any]]


def evaluate_canonical_3dgs_campaign(
    *,
    campaign_result: Mapping[str, Any],
    results_root: str | Path,
    evaluator_input: Mapping[str, Any],
    evaluator_root: str | Path,
    thresholds: Mapping[str, Any],
    lpips_model: Mapping[str, Any],
    output_root: str | Path,
    renderer: RenderExecutor = render_splat_at_exact_cameras,
    appearance_evaluator: AppearanceEvaluator = evaluate_heldout_appearance_v2,
    evaluator_identity: str = "blueprint-independent-heldout-evaluator-v2",
    evaluator_provider_identity: str = "blueprint-local-independent-evaluator",
) -> dict[str, Any]:
    """Render both candidates at identical hidden cameras and qualify/rank them."""

    campaign = _validated_campaign(campaign_result)
    evaluator = _validated_evaluator_input(evaluator_input)
    if (
        evaluator["source_capture_digest"] != campaign["source_capture_digest"]
        or evaluator["frozen_split_digest"] != campaign["frozen_split_digest"]
    ):
        raise Canonical3DGSEvaluationError(["canonical_quality_capture_or_split_mismatch"])
    required_thresholds = {
        "minimum_mean_psnr_db",
        "minimum_mean_global_ssim",
        "minimum_mean_windowed_ssim",
        "maximum_mean_absolute_error",
        "maximum_mean_lpips",
    }
    if set(thresholds) != required_thresholds or not all(
        not isinstance(value, bool) and isinstance(value, (int, float))
        for value in thresholds.values()
    ):
        raise Canonical3DGSEvaluationError(["canonical_quality_thresholds_invalid"])
    if (
        lpips_model.get("model_id") != "lpips_alex_v0.1"
        or not str(lpips_model.get("checkpoint_digest") or "").startswith("sha256:")
    ):
        raise Canonical3DGSEvaluationError(["canonical_quality_lpips_model_invalid"])
    results = Path(results_root).expanduser().resolve()
    evaluation_root = Path(evaluator_root).expanduser().resolve()
    destination = Path(output_root).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    bindings = {
        str(row["candidate_method_id"]): dict(row)
        for row in campaign.get("appearance_fidelity_candidate_bindings") or []
        if isinstance(row, Mapping)
    }
    reports: list[dict[str, Any]] = []
    qualifications: list[dict[str, Any]] = []
    for arm_id in ("postshot-primary", "splatfacto-comparison"):
        result_path = results / arm_id / "canonical_3dgs_arm_result.json"
        arm = _load_json(result_path, code=f"canonical_quality_arm_result_invalid:{arm_id}")
        if (
            arm.get("status") != "succeeded"
            or arm.get("canonical_3dgs_arm_result_digest")
            != canonical_digest(arm, digest_field="canonical_3dgs_arm_result_digest")
            or arm.get("frozen_split_digest") != campaign["frozen_split_digest"]
        ):
            raise Canonical3DGSEvaluationError(
                [f"canonical_quality_arm_result_binding_invalid:{arm_id}"]
            )
        splats = [
            row
            for row in arm.get("artifacts") or []
            if isinstance(row, Mapping) and row.get("kind") == "standard_3dgs_ply"
        ]
        binding = bindings.get(str(arm.get("method_id")))
        if len(splats) != 1 or not isinstance(binding, Mapping):
            raise Canonical3DGSEvaluationError(
                [f"canonical_quality_splat_binding_missing:{arm_id}"]
            )
        splat = _safe_result_artifact(
            results,
            arm_id,
            splats[0]["relative_path"],
            str(splats[0]["digest"]),
        )
        if binding.get("asset_digest") != splats[0]["digest"]:
            raise Canonical3DGSEvaluationError(
                [f"canonical_quality_campaign_asset_mismatch:{arm_id}"]
            )
        candidate_root = destination / arm_id / "render"
        render_manifest = dict(
            renderer(
                splat_path=splat,
                cameras=evaluator["cameras"],
                output_dir=candidate_root,
                provider_splat_import_receipt_digest=arm[
                    "canonical_3dgs_arm_result_digest"
                ],
                alignment_digest=binding["coordinate_basis_digest"],
                camera_set_label="canonical_v32_frozen_hidden",
            )
        )
        rendered = {
            str(row["camera_id"]): row
            for row in render_manifest.get("renders") or []
            if isinstance(row, Mapping)
        }
        if set(rendered) != {str(row["camera_id"]) for row in evaluator["cameras"]}:
            raise Canonical3DGSEvaluationError(
                [f"canonical_quality_render_set_mismatch:{arm_id}"]
            )
        request = {
            "schema_version": "heldout_appearance_evaluation_request.v2",
            "stable_run_identity": f"canonical-3dgs-quality-{campaign['canonical_3dgs_campaign_result_digest'][7:31]}-{arm_id}",
            "source_capture_identity": evaluator["source_capture_identity"],
            "source_capture_digest": evaluator["source_capture_digest"],
            "reconstruction_dataset_digest": evaluator["reconstruction_dataset_digest"],
            "frozen_split_digest": evaluator["frozen_split_digest"],
            "candidate_reconstruction_result_digest": arm[
                "canonical_3dgs_arm_result_digest"
            ],
            "candidate_method_id": arm["method_id"],
            "candidate_provider_identity": f"trainer:{arm_id}",
            "evaluator_identity": evaluator_identity,
            "evaluator_provider_identity": evaluator_provider_identity,
            "evaluator_implementation_digest": _sha256(Path(__file__)),
            "source_commit_sha": campaign["source_commit_sha"],
            "candidate_root": str(candidate_root),
            "evaluator_root": str(evaluation_root),
            "coordinate_frame_declaration": {
                "world_frame": evaluator["world_frame"],
                "camera_axis_convention": evaluator["camera_axis_convention"],
                "units": evaluator["units"],
            },
            "split_frozen_before_training": True,
            "thresholds_frozen_before_evaluation": True,
            "candidate_had_hidden_access": False,
            "candidate_selected_heldout": False,
            "candidate_self_grading": False,
            "lpips_required": True,
            "lpips_model": dict(lpips_model),
            "thresholds": dict(thresholds),
            "pairs": [
                {
                    "view_id": row["camera_id"],
                    "trajectory": row["trajectory"],
                    "split": "held_out",
                    "excluded_from_training": True,
                    "real_view_relative_path": row["reference_relative_path"],
                    "real_view_digest": row["reference_digest"],
                    "candidate_render_relative_path": rendered[row["camera_id"]][
                        "relative_path"
                    ],
                    "candidate_render_digest": rendered[row["camera_id"]]["digest"],
                }
                for row in evaluator["cameras"]
            ],
            "authority_used": dict(evaluator["authority_used"]),
            "timestamp": campaign["timestamp"],
        }
        request = build_heldout_appearance_evaluation_request_v2(request)
        report = dict(
            appearance_evaluator(source_artifact=request, output_root=destination / arm_id)
        )
        _immutable_json(destination / arm_id / "heldout_evaluation_request.json", request)
        _immutable_json(destination / arm_id / "heldout_evaluation_report.json", report)
        aggregate = report.get("by_trajectory", {}).get("author_heldout")
        if not isinstance(aggregate, Mapping):
            raise Canonical3DGSEvaluationError(
                [f"canonical_quality_report_aggregate_missing:{arm_id}"]
            )
        profile = {
            key: binding[key]
            for key in (
                "asset_digest",
                "coordinate_basis_digest",
                "representation",
                "splat_count",
                "sh_degree",
                "bounds",
            )
        }
        renderer_identity = dict(render_manifest.get("renderer_identity") or {})
        qualification = build_appearance_fidelity_qualification(
            {
                "schema_version": "appearance_fidelity_qualification.v1",
                "source_appearance": dict(profile),
                "render_input": {
                    **profile,
                    "removal_reasons": [],
                    "global_decimation_applied": False,
                },
                "renderer": {
                    "renderer_id": str(render_manifest.get("rendered_by") or "native_3dgs_renderer"),
                    "implementation_digest": canonical_digest(
                        {
                            "harness_digest": renderer_identity.get("harness_digest"),
                            "render_entry_digest": renderer_identity.get("render_entry_digest"),
                        }
                    ),
                    "runtime_digest": canonical_digest(renderer_identity),
                    "native_3dgs": True,
                    "full_anisotropic_gaussians": True,
                    "maximum_sh_degree": int(binding["sh_degree"]),
                },
                "reference_frame_comparison": {
                    "status": "completed",
                    "source_frame_digest": _aggregate_digest(
                        request["pairs"], "real_view_digest"
                    ),
                    "rendered_frame_digest": _aggregate_digest(
                        request["pairs"], "candidate_render_digest"
                    ),
                    "camera_spec_digest": evaluator[
                        "canonical_3dgs_hidden_evaluator_input_digest"
                    ],
                    "camera_basis_digest": binding["coordinate_basis_digest"],
                    "metrics": {
                        "ssim": float(aggregate["mean_windowed_ssim"]),
                        "psnr_db": float(aggregate["mean_psnr_db"]),
                        "lpips": float(aggregate["mean_lpips"]),
                    },
                },
                "qualification_policy": {
                    "minimum_retained_fraction": 1.0,
                    "minimum_ssim": float(thresholds["minimum_mean_windowed_ssim"]),
                    "minimum_psnr_db": float(thresholds["minimum_mean_psnr_db"]),
                    "maximum_lpips": float(thresholds["maximum_mean_lpips"]),
                },
            }
        )
        _immutable_json(
            destination / arm_id / "appearance_fidelity_qualification.json",
            qualification,
        )
        reports.append({"arm_id": arm_id, "report": report, "aggregate": dict(aggregate)})
        qualifications.append({"arm_id": arm_id, "qualification": qualification})
    qualified = [
        row
        for row in qualifications
        if row["qualification"]["status"] == "qualified"
    ]
    report_by_arm = {row["arm_id"]: row for row in reports}
    qualified.sort(
        key=lambda row: (
            float(report_by_arm[row["arm_id"]]["aggregate"]["mean_lpips"]),
            -float(report_by_arm[row["arm_id"]]["aggregate"]["mean_psnr_db"]),
            -float(report_by_arm[row["arm_id"]]["aggregate"]["mean_windowed_ssim"]),
            row["arm_id"],
        )
    )
    winner = qualified[0]["arm_id"] if qualified else None
    comparison = {
        "schema_version": QUALITY_COMPARISON_SCHEMA,
        "status": "quality_winner_selected" if winner else "abstained_no_qualified_candidate",
        "canonical_3dgs_campaign_result_digest": campaign[
            "canonical_3dgs_campaign_result_digest"
        ],
        "canonical_3dgs_hidden_evaluator_input_digest": evaluator[
            "canonical_3dgs_hidden_evaluator_input_digest"
        ],
        "frozen_split_digest": campaign["frozen_split_digest"],
        "thresholds": dict(thresholds),
        "lpips_model": dict(lpips_model),
        "candidate_reports": [
            {
                "arm_id": row["arm_id"],
                "visual_heldout_evaluation_report_digest": row["report"][
                    "visual_heldout_evaluation_report_digest"
                ],
                "aggregate": row["aggregate"],
                "appearance_fidelity_qualification_digest": next(
                    item["qualification"]["appearance_fidelity_qualification_digest"]
                    for item in qualifications
                    if item["arm_id"] == row["arm_id"]
                ),
                "appearance_fidelity_status": next(
                    item["qualification"]["status"]
                    for item in qualifications
                    if item["arm_id"] == row["arm_id"]
                ),
            }
            for row in reports
        ],
        "quality_winner": winner,
        "selection_policy": "qualified_only_then_lpips_psnr_windowed_ssim",
        "candidate_hidden_pixel_access": False,
        "trainer_self_grading": False,
        "raw_capture_authority_upgraded": False,
        "metric_collision_or_physical_claim_upgraded": False,
        "proof_effect": "independent_appearance_quality_selection_only",
        "claim_ceiling": "appearance_reconstruction_quality",
        "timestamp": campaign["timestamp"],
    }
    comparison["canonical_3dgs_quality_comparison_digest"] = canonical_digest(
        comparison, digest_field="canonical_3dgs_quality_comparison_digest"
    )
    _immutable_json(destination / "canonical_3dgs_quality_comparison.json", comparison)
    return comparison


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--evaluator-input", required=True)
    parser.add_argument("--evaluator-root", required=True)
    parser.add_argument("--thresholds", required=True)
    parser.add_argument("--lpips-model", required=True)
    parser.add_argument("--output-root", required=True)
    arguments = parser.parse_args(argv)
    loaded = []
    for value in (
        arguments.campaign,
        arguments.evaluator_input,
        arguments.thresholds,
        arguments.lpips_model,
    ):
        try:
            loaded.append(json.loads(Path(value).read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as exc:
            raise Canonical3DGSEvaluationError(
                ["canonical_quality_cli_input_invalid"]
            ) from exc
    result = evaluate_canonical_3dgs_campaign(
        campaign_result=loaded[0],
        results_root=arguments.results_root,
        evaluator_input=loaded[1],
        evaluator_root=arguments.evaluator_root,
        thresholds=loaded[2],
        lpips_model=loaded[3],
        output_root=arguments.output_root,
    )
    print(canonical_json(result))
    return 0 if result["quality_winner"] is not None else 2


__all__ = [
    "EVALUATOR_INPUT_SCHEMA",
    "QUALITY_COMPARISON_SCHEMA",
    "Canonical3DGSEvaluationError",
    "compile_canonical_3dgs_hidden_evaluator_input",
    "evaluate_canonical_3dgs_campaign",
]


if __name__ == "__main__":
    raise SystemExit(main())
