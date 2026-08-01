"""Independent held-out appearance evaluation, v2.

Extends the v1 deterministic evaluator with standard windowed SSIM, an
explicitly pinned LPIPS lane, and per-trajectory reporting (author-designated
held-out versus independent short trajectory are never averaged together).
The v1 request/report contracts remain untouched for replay compatibility.

Rules preserved from v1: the candidate never sees reference pixels, the
evaluator must be provider-independent of the candidate, thresholds are frozen
before evaluation, digests bind every consumed byte, and the report ceiling is
appearance reconstruction only.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from .decision_evidence_contracts import canonical_digest


HELDOUT_V2_REQUEST_SCHEMA_VERSION = "heldout_appearance_evaluation_request.v2"
HELDOUT_V2_REPORT_SCHEMA_VERSION = "visual_heldout_evaluation_report.v2"

TRAJECTORIES = ("author_heldout", "independent_short")
THRESHOLD_FIELDS = {
    "minimum_mean_psnr_db",
    "minimum_mean_global_ssim",
    "minimum_mean_windowed_ssim",
    "maximum_mean_absolute_error",
    "maximum_mean_lpips",
}
SSIM_WINDOW = 11
SSIM_SIGMA = 1.5
SSIM_K1 = 0.01
SSIM_K2 = 0.03


class HeldoutAppearanceV2Error(ValueError):
    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__("; ".join(self.codes))


def _is_digest(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 71 and text.startswith("sha256:") and all(
        character in "0123456789abcdef" for character in text[7:]
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _safe_path(root: Path, value: Any, *, label: str) -> Path:
    raw = str(value or "").strip().replace("\\", "/")
    relative = PurePosixPath(raw)
    if not raw or relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise HeldoutAppearanceV2Error([f"{label}_path_invalid"])
    lexical = root / Path(*relative.parts)
    if lexical.is_symlink():
        raise HeldoutAppearanceV2Error([f"{label}_symlink_forbidden"])
    resolved = lexical.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise HeldoutAppearanceV2Error([f"{label}_path_escape"]) from exc
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise HeldoutAppearanceV2Error([f"{label}_missing"])
    return resolved


def _rgb(path: Path, *, label: str) -> np.ndarray:
    try:
        with Image.open(path) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
    except (OSError, ValueError) as exc:
        raise HeldoutAppearanceV2Error([f"{label}_unreadable"]) from exc
    if array.size == 0 or not np.isfinite(array).all():
        raise HeldoutAppearanceV2Error([f"{label}_invalid_pixels"])
    return array


def _global_ssim(left: np.ndarray, right: np.ndarray) -> float:
    left_mean, right_mean = float(np.mean(left)), float(np.mean(right))
    left_variance = float(np.mean(np.square(left - left_mean)))
    right_variance = float(np.mean(np.square(right - right_mean)))
    covariance = float(np.mean((left - left_mean) * (right - right_mean)))
    c1, c2 = SSIM_K1**2, SSIM_K2**2
    denominator = (left_mean**2 + right_mean**2 + c1) * (left_variance + right_variance + c2)
    if denominator == 0.0:
        return 1.0
    return float(((2.0 * left_mean * right_mean + c1) * (2.0 * covariance + c2)) / denominator)


def _gaussian_kernel() -> np.ndarray:
    offsets = np.arange(SSIM_WINDOW, dtype=np.float64) - (SSIM_WINDOW - 1) / 2.0
    kernel = np.exp(-np.square(offsets) / (2.0 * SSIM_SIGMA**2))
    return kernel / kernel.sum()


_KERNEL = _gaussian_kernel()


def _filter_valid(channel: np.ndarray) -> np.ndarray:
    """Separable Gaussian filter with 'valid' boundary handling (canonical SSIM)."""

    rows = np.apply_along_axis(
        lambda line: np.convolve(line, _KERNEL, mode="valid"), 1, channel
    )
    return np.apply_along_axis(
        lambda line: np.convolve(line, _KERNEL, mode="valid"), 0, rows
    )


def windowed_ssim(left: np.ndarray, right: np.ndarray) -> float:
    """Standard Wang et al. SSIM: 11x11 Gaussian window, sigma 1.5, L=1."""

    if left.shape != right.shape or left.ndim != 3:
        raise HeldoutAppearanceV2Error(["windowed_ssim_shape_mismatch"])
    if min(left.shape[0], left.shape[1]) < SSIM_WINDOW:
        raise HeldoutAppearanceV2Error(["windowed_ssim_image_too_small"])
    c1, c2 = SSIM_K1**2, SSIM_K2**2
    channel_means = []
    for channel_index in range(left.shape[2]):
        a = left[..., channel_index]
        b = right[..., channel_index]
        mu_a = _filter_valid(a)
        mu_b = _filter_valid(b)
        mu_aa = _filter_valid(a * a)
        mu_bb = _filter_valid(b * b)
        mu_ab = _filter_valid(a * b)
        variance_a = mu_aa - mu_a * mu_a
        variance_b = mu_bb - mu_b * mu_b
        covariance = mu_ab - mu_a * mu_b
        ssim_map = ((2 * mu_a * mu_b + c1) * (2 * covariance + c2)) / (
            (mu_a**2 + mu_b**2 + c1) * (variance_a + variance_b + c2)
        )
        channel_means.append(float(np.mean(ssim_map)))
    return float(np.mean(channel_means))


class _LpipsRuntime:
    """Pinned LPIPS runtime wrapper; import failures surface as typed blockers."""

    def __init__(self, model_id: str, checkpoint_digest: str, backbone_digest: str | None):
        try:
            import inspect
            import os

            import lpips  # type: ignore
            import torch  # type: ignore
        except ImportError as exc:
            raise HeldoutAppearanceV2Error(["lpips_runtime_unavailable"]) from exc
        if model_id != "lpips_alex_v0.1":
            raise HeldoutAppearanceV2Error(["lpips_model_id_unsupported"])
        weights_path = Path(os.path.dirname(inspect.getfile(lpips))) / "weights" / "v0.1" / "alex.pth"
        if not weights_path.is_file():
            raise HeldoutAppearanceV2Error(["lpips_checkpoint_missing"])
        observed = "sha256:" + hashlib.sha256(weights_path.read_bytes()).hexdigest()
        if observed != checkpoint_digest:
            raise HeldoutAppearanceV2Error(["lpips_checkpoint_digest_mismatch"])
        self._torch = torch
        torch.set_grad_enabled(False)
        self.network = lpips.LPIPS(net="alex", verbose=False).eval()
        backbone = Path.home() / ".cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth"
        self.backbone_digest = (
            "sha256:" + hashlib.sha256(backbone.read_bytes()).hexdigest()
            if backbone.is_file()
            else None
        )
        if backbone_digest is not None and self.backbone_digest != backbone_digest:
            raise HeldoutAppearanceV2Error(["lpips_backbone_digest_mismatch"])
        self.torch_version = str(torch.__version__)
        self.checkpoint_digest = observed

    def distance(self, left: np.ndarray, right: np.ndarray) -> float:
        torch = self._torch
        def to_tensor(array: np.ndarray):
            tensor = torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1)[None]
            return tensor.to(torch.float32) * 2.0 - 1.0

        with torch.no_grad():
            value = self.network(to_tensor(left), to_tensor(right))
        return float(value.reshape(()).item())


def build_heldout_appearance_evaluation_request_v2(value: Mapping[str, Any]) -> dict[str, Any]:
    request = dict(value)
    errors: list[str] = []
    if request.get("schema_version") != HELDOUT_V2_REQUEST_SCHEMA_VERSION:
        errors.append("heldout_v2_request_schema_invalid")
    for key in (
        "source_capture_digest",
        "reconstruction_dataset_digest",
        "frozen_split_digest",
        "candidate_reconstruction_result_digest",
        "evaluator_implementation_digest",
    ):
        if not _is_digest(request.get(key)):
            errors.append(f"heldout_v2_request_{key}_invalid")
    commit = str(request.get("source_commit_sha") or "")
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        errors.append("heldout_v2_request_source_commit_sha_invalid")
    for key in (
        "stable_run_identity",
        "source_capture_identity",
        "candidate_method_id",
        "candidate_provider_identity",
        "evaluator_identity",
        "evaluator_provider_identity",
        "candidate_root",
        "evaluator_root",
        "timestamp",
    ):
        if not str(request.get(key) or "").strip():
            errors.append(f"heldout_v2_request_{key}_missing")
    if not isinstance(request.get("coordinate_frame_declaration"), Mapping):
        errors.append("heldout_v2_request_coordinate_frame_declaration_invalid")
    if not isinstance(request.get("authority_used"), Mapping):
        errors.append("heldout_v2_request_authority_used_invalid")
    for key, expected in (
        ("split_frozen_before_training", True),
        ("thresholds_frozen_before_evaluation", True),
        ("candidate_had_hidden_access", False),
        ("candidate_selected_heldout", False),
        ("candidate_self_grading", False),
    ):
        if request.get(key) is not expected:
            errors.append(f"heldout_v2_request_{key}_not_{str(expected).lower()}")
    if request.get("candidate_provider_identity") == request.get("evaluator_provider_identity"):
        errors.append("heldout_v2_request_evaluator_not_independent")
    lpips_required = request.get("lpips_required")
    if not isinstance(lpips_required, bool):
        errors.append("heldout_v2_request_lpips_required_invalid")
    lpips_model = request.get("lpips_model")
    if lpips_required:
        if (
            not isinstance(lpips_model, Mapping)
            or not str(lpips_model.get("model_id") or "").strip()
            or not _is_digest(lpips_model.get("checkpoint_digest"))
        ):
            errors.append("heldout_v2_request_lpips_model_invalid")
    thresholds = request.get("thresholds")
    if not isinstance(thresholds, Mapping) or set(thresholds) != THRESHOLD_FIELDS:
        errors.append("heldout_v2_request_threshold_fields_invalid")
    else:
        for key in THRESHOLD_FIELDS:
            number = thresholds.get(key)
            if key == "maximum_mean_lpips" and number is None and not lpips_required:
                continue
            if (
                isinstance(number, bool)
                or not isinstance(number, (int, float))
                or not math.isfinite(float(number))
            ):
                errors.append(f"heldout_v2_request_threshold_invalid:{key}")
    pairs = request.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        errors.append("heldout_v2_request_pairs_missing")
    else:
        seen: set[str] = set()
        trajectories_present: set[str] = set()
        for index, raw_pair in enumerate(pairs):
            if not isinstance(raw_pair, Mapping):
                errors.append(f"heldout_v2_request_pair_invalid:{index}")
                continue
            view_id = str(raw_pair.get("view_id") or "").strip()
            if not view_id or view_id in seen:
                errors.append(f"heldout_v2_request_view_id_invalid:{index}")
            seen.add(view_id)
            if raw_pair.get("trajectory") not in TRAJECTORIES:
                errors.append(f"heldout_v2_request_trajectory_invalid:{view_id or index}")
            else:
                trajectories_present.add(raw_pair["trajectory"])
            if raw_pair.get("split") != "held_out" or raw_pair.get("excluded_from_training") is not True:
                errors.append(f"heldout_v2_request_pair_not_heldout:{view_id or index}")
            for key in ("real_view_digest", "candidate_render_digest"):
                if not _is_digest(raw_pair.get(key)):
                    errors.append(f"heldout_v2_request_pair_digest_invalid:{view_id or index}:{key}")
            for key in ("real_view_relative_path", "candidate_render_relative_path"):
                path = PurePosixPath(str(raw_pair.get(key) or "").replace("\\", "/"))
                if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
                    errors.append(f"heldout_v2_request_pair_path_invalid:{view_id or index}:{key}")
    supplied_digest = request.pop("heldout_appearance_evaluation_request_digest", None)
    request["heldout_appearance_evaluation_request_digest"] = canonical_digest(
        request, digest_field="heldout_appearance_evaluation_request_digest"
    )
    if supplied_digest is not None and supplied_digest != request[
        "heldout_appearance_evaluation_request_digest"
    ]:
        errors.append("heldout_v2_request_digest_mismatch")
    if errors:
        raise HeldoutAppearanceV2Error(errors)
    return request


def _aggregate(rows: list[dict[str, Any]], *, lpips_computed: bool) -> dict[str, Any]:
    finite_psnr = [float(row["psnr_db"]) for row in rows if row["psnr_db"] != "infinity"]
    mean_psnr = float("inf") if len(finite_psnr) < len(rows) else float(np.mean(finite_psnr))
    aggregate = {
        "view_count": len(rows),
        "mean_psnr_db": "infinity" if math.isinf(mean_psnr) else round(mean_psnr, 6),
        "mean_global_ssim": round(float(np.mean([row["global_ssim"] for row in rows])), 8),
        "mean_windowed_ssim": round(
            float(np.mean([row["windowed_ssim"] for row in rows])), 8
        ),
        "mean_absolute_error": round(
            float(np.mean([row["mean_absolute_error"] for row in rows])), 8
        ),
    }
    aggregate["mean_lpips"] = (
        round(float(np.mean([row["lpips"] for row in rows])), 8) if lpips_computed else None
    )
    return aggregate


def _thresholds_pass(aggregate: Mapping[str, Any], thresholds: Mapping[str, Any]) -> bool:
    mean_psnr = (
        float("inf") if aggregate["mean_psnr_db"] == "infinity" else float(aggregate["mean_psnr_db"])
    )
    passed = (
        mean_psnr >= float(thresholds["minimum_mean_psnr_db"])
        and float(aggregate["mean_global_ssim"]) >= float(thresholds["minimum_mean_global_ssim"])
        and float(aggregate["mean_windowed_ssim"])
        >= float(thresholds["minimum_mean_windowed_ssim"])
        and float(aggregate["mean_absolute_error"])
        <= float(thresholds["maximum_mean_absolute_error"])
    )
    if thresholds.get("maximum_mean_lpips") is not None:
        passed = passed and (
            aggregate.get("mean_lpips") is not None
            and float(aggregate["mean_lpips"]) <= float(thresholds["maximum_mean_lpips"])
        )
    return bool(passed)


def evaluate_heldout_appearance_v2(
    *, source_artifact: Mapping[str, Any], output_root: str | Path
) -> dict[str, Any]:
    """Evaluate candidate renders per trajectory without leaking hidden pixels."""

    del output_root  # The caller owns artifact persistence.
    request = build_heldout_appearance_evaluation_request_v2(source_artifact)
    candidate_root = Path(request["candidate_root"]).expanduser().resolve()
    evaluator_root = Path(request["evaluator_root"]).expanduser().resolve()
    if candidate_root == evaluator_root:
        raise HeldoutAppearanceV2Error(["heldout_v2_roots_not_isolated"])
    lpips_runtime: _LpipsRuntime | None = None
    if request["lpips_required"]:
        model = request["lpips_model"]
        lpips_runtime = _LpipsRuntime(
            str(model["model_id"]),
            str(model["checkpoint_digest"]),
            model.get("backbone_digest"),
        )
    rows: list[dict[str, Any]] = []
    for pair in request["pairs"]:
        view_id = str(pair["view_id"])
        real_path = _safe_path(
            evaluator_root, pair["real_view_relative_path"], label="heldout_real_view"
        )
        candidate_path = _safe_path(
            candidate_root, pair["candidate_render_relative_path"], label="candidate_render"
        )
        if _sha256_file(real_path) != pair["real_view_digest"]:
            raise HeldoutAppearanceV2Error([f"heldout_real_digest_mismatch:{view_id}"])
        if _sha256_file(candidate_path) != pair["candidate_render_digest"]:
            raise HeldoutAppearanceV2Error([f"candidate_render_digest_mismatch:{view_id}"])
        real = _rgb(real_path, label="heldout_real_view")
        candidate = _rgb(candidate_path, label="candidate_render")
        if real.shape != candidate.shape:
            raise HeldoutAppearanceV2Error([f"heldout_view_shape_mismatch:{view_id}"])
        if float(candidate.std()) < 1e-9:
            raise HeldoutAppearanceV2Error([f"candidate_render_blank:{view_id}"])
        difference = real - candidate
        mse = float(np.mean(np.square(difference)))
        psnr = float("inf") if mse == 0.0 else float(10.0 * math.log10(1.0 / mse))
        row = {
            "view_id": view_id,
            "trajectory": pair["trajectory"],
            "real_view_digest": pair["real_view_digest"],
            "candidate_render_digest": pair["candidate_render_digest"],
            "psnr_db": "infinity" if math.isinf(psnr) else round(psnr, 6),
            "global_ssim": round(_global_ssim(real, candidate), 8),
            "windowed_ssim": round(windowed_ssim(real, candidate), 8),
            "mean_absolute_error": round(float(np.mean(np.abs(difference))), 8),
        }
        if lpips_runtime is not None:
            row["lpips"] = round(lpips_runtime.distance(real, candidate), 8)
        rows.append(row)
    thresholds = dict(request["thresholds"])
    by_trajectory: dict[str, Any] = {}
    all_pass = True
    for trajectory in TRAJECTORIES:
        trajectory_rows = [row for row in rows if row["trajectory"] == trajectory]
        if not trajectory_rows:
            by_trajectory[trajectory] = {"view_count": 0, "thresholds_passed": None}
            continue
        aggregate = _aggregate(trajectory_rows, lpips_computed=lpips_runtime is not None)
        aggregate["thresholds_passed"] = _thresholds_pass(aggregate, thresholds)
        by_trajectory[trajectory] = aggregate
        all_pass = all_pass and aggregate["thresholds_passed"]
    measured = [
        trajectory
        for trajectory in TRAJECTORIES
        if by_trajectory[trajectory]["view_count"] > 0
    ]
    if not measured:
        raise HeldoutAppearanceV2Error(["heldout_v2_no_trajectory_measured"])
    report = {
        "schema_version": HELDOUT_V2_REPORT_SCHEMA_VERSION,
        "stable_run_identity": request["stable_run_identity"],
        "source_capture_identity": request["source_capture_identity"],
        "source_capture_digest": request["source_capture_digest"],
        "reconstruction_dataset_digest": request["reconstruction_dataset_digest"],
        "frozen_split_digest": request["frozen_split_digest"],
        "candidate_reconstruction_result_digest": request[
            "candidate_reconstruction_result_digest"
        ],
        "evaluation_request_digest": request["heldout_appearance_evaluation_request_digest"],
        "candidate_method_id": request["candidate_method_id"],
        "candidate_provider_identity": request["candidate_provider_identity"],
        "evaluator_identity": request["evaluator_identity"],
        "evaluator_provider_identity": request["evaluator_provider_identity"],
        "evaluator_implementation_digest": request["evaluator_implementation_digest"],
        "source_commit_sha": request["source_commit_sha"],
        "coordinate_frame_declaration": dict(request["coordinate_frame_declaration"]),
        "metric_definitions": {
            "windowed_ssim": "wang2004_gaussian_11x11_sigma1.5_valid_region_L1",
            "global_ssim": "repository_deterministic_global_equivalent",
            "lpips": "lpips_alex_v0.1" if lpips_runtime is not None else None,
        },
        "lpips_runtime": (
            {
                "model_id": "lpips_alex_v0.1",
                "checkpoint_digest": lpips_runtime.checkpoint_digest,
                "backbone_digest": lpips_runtime.backbone_digest,
                "torch_version": lpips_runtime.torch_version,
            }
            if lpips_runtime is not None
            else None
        ),
        "rows": rows,
        "by_trajectory": by_trajectory,
        "measured_trajectories": measured,
        "thresholds": thresholds,
        "all_measured_trajectories_passed": bool(all_pass),
        "status": (
            "passed_appearance_only" if all_pass else "rejected_appearance_quality"
        ),
        "heldout_observation_count": len(rows),
        "candidate_had_hidden_access": False,
        "candidate_selected_heldout": False,
        "candidate_self_graded": False,
        "cost_usd": 0.0,
        "authority_used": dict(request.get("authority_used") or {}),
        "warnings": [
            "trajectory_aggregates_are_reported_separately_and_both_must_pass",
        ],
        "blockers": [] if all_pass else ["heldout_appearance_thresholds_not_met"],
        "proof_effect": "independent_heldout_appearance_evaluation_only",
        "claim_ceiling": "appearance_reconstruction",
        "metric_scale_proven": False,
        "metric_geometry_proven": False,
        "collision_geometry_proven": False,
        "physics_readiness_proven": False,
        "physical_task_success_proven": False,
        "deployment_readiness_proven": False,
        "parent_artifact_or_event": {
            "candidate_reconstruction_result_digest": request[
                "candidate_reconstruction_result_digest"
            ],
            "frozen_split_digest": request["frozen_split_digest"],
        },
        "timestamp": request["timestamp"],
    }
    report["visual_heldout_evaluation_report_digest"] = canonical_digest(
        report, digest_field="visual_heldout_evaluation_report_digest"
    )
    return report


def build_visual_heldout_evaluation_report_v2(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a recorded v2 evaluator report for deterministic replay."""

    report = dict(value)
    errors: list[str] = []
    if report.get("schema_version") != HELDOUT_V2_REPORT_SCHEMA_VERSION:
        errors.append("heldout_v2_report_schema_invalid")
    rows = report.get("rows")
    if not isinstance(rows, list) or not rows:
        errors.append("heldout_v2_report_rows_missing")
        rows = []
    if report.get("heldout_observation_count") != len(rows):
        errors.append("heldout_v2_report_count_mismatch")
    lpips_computed = isinstance(report.get("lpips_runtime"), Mapping)
    thresholds = report.get("thresholds")
    by_trajectory = report.get("by_trajectory")
    if not isinstance(thresholds, Mapping) or set(thresholds) != THRESHOLD_FIELDS:
        errors.append("heldout_v2_report_thresholds_invalid")
    elif not isinstance(by_trajectory, Mapping):
        errors.append("heldout_v2_report_by_trajectory_invalid")
    else:
        all_pass = True
        for trajectory in TRAJECTORIES:
            trajectory_rows = [
                row
                for row in rows
                if isinstance(row, Mapping) and row.get("trajectory") == trajectory
            ]
            recorded = by_trajectory.get(trajectory)
            if not trajectory_rows:
                if not isinstance(recorded, Mapping) or recorded.get("view_count") != 0:
                    errors.append(f"heldout_v2_report_trajectory_mismatch:{trajectory}")
                continue
            expected = _aggregate(
                [dict(row) for row in trajectory_rows], lpips_computed=lpips_computed
            )
            expected["thresholds_passed"] = _thresholds_pass(expected, thresholds)
            if not isinstance(recorded, Mapping) or dict(recorded) != expected:
                errors.append(f"heldout_v2_report_trajectory_recomputation_mismatch:{trajectory}")
            all_pass = all_pass and expected["thresholds_passed"]
        expected_status = "passed_appearance_only" if all_pass else "rejected_appearance_quality"
        expected_blockers = [] if all_pass else ["heldout_appearance_thresholds_not_met"]
        if (
            report.get("all_measured_trajectories_passed") is not all_pass
            or report.get("status") != expected_status
            or report.get("blockers") != expected_blockers
        ):
            errors.append("heldout_v2_report_status_threshold_mismatch")
    if (
        not str(report.get("candidate_provider_identity") or "").strip()
        or report.get("candidate_provider_identity")
        == report.get("evaluator_provider_identity")
    ):
        errors.append("heldout_v2_report_evaluator_not_independent")
    for key in (
        "candidate_had_hidden_access",
        "candidate_selected_heldout",
        "candidate_self_graded",
        "metric_scale_proven",
        "metric_geometry_proven",
        "collision_geometry_proven",
        "physics_readiness_proven",
        "physical_task_success_proven",
        "deployment_readiness_proven",
    ):
        if report.get(key) is not False:
            errors.append(f"heldout_v2_report_forbidden_claim:{key}")
    if (
        report.get("proof_effect") != "independent_heldout_appearance_evaluation_only"
        or report.get("claim_ceiling") != "appearance_reconstruction"
    ):
        errors.append("heldout_v2_report_claim_boundary_invalid")
    expected_digest = canonical_digest(
        report, digest_field="visual_heldout_evaluation_report_digest"
    )
    if report.get("visual_heldout_evaluation_report_digest") != expected_digest:
        errors.append("heldout_v2_report_digest_mismatch")
    if errors:
        raise HeldoutAppearanceV2Error(errors)
    return report


__all__ = [
    "HELDOUT_V2_REPORT_SCHEMA_VERSION",
    "HELDOUT_V2_REQUEST_SCHEMA_VERSION",
    "HeldoutAppearanceV2Error",
    "build_heldout_appearance_evaluation_request_v2",
    "build_visual_heldout_evaluation_report_v2",
    "evaluate_heldout_appearance_v2",
    "windowed_ssim",
]
