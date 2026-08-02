"""Register a derived visual splat frame to an external collision-mesh frame.

Scan providers may export appearance and mesh assets with different axis signs
or origins. This compiler enumerates the 24 right-handed axis conventions,
estimates provider-scale consistency, and scores symmetric trimmed nearest-point
error. It emits a candidate registration only when the best convention clears
frozen error, scale, and ambiguity gates. It never proves metric scale or contact.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial import cKDTree

from .decision_evidence_contracts import canonical_digest
from .external_scene_collision_candidate import _flatten_glb
from .gaussian_splat_decode import read_standard_3dgs_ply


REQUEST_SCHEMA = "external_scene_frame_registration_request.v1"
RESULT_SCHEMA = "external_scene_frame_registration_result.v1"
COMPOSED_BINDING_SCHEMA = "registered_scene_task_target_binding.v1"


class ExternalSceneFrameRegistrationError(ValueError):
    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__(";".join(self.codes))


def _digest(value: Any) -> bool:
    text = str(value or "")
    return (
        len(text) == 71
        and text.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in text[7:])
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _axis_rotations() -> list[np.ndarray]:
    rotations: list[np.ndarray] = []
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            rotation = np.zeros((3, 3), dtype=np.float64)
            for output_axis, input_axis in enumerate(permutation):
                rotation[output_axis, input_axis] = signs[output_axis]
            if np.linalg.det(rotation) > 0.5:
                rotations.append(rotation)
    return rotations


def build_external_scene_frame_registration_request(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        request = json.loads(json.dumps(dict(value), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ExternalSceneFrameRegistrationError(["frame_registration_request_not_json"]) from exc
    supplied = request.pop("request_digest", None)
    errors: list[str] = []
    if request.get("schema_version") != REQUEST_SCHEMA:
        errors.append("frame_registration_request_schema_invalid")
    for key in ("appearance_scene_digest", "analysis_splat_digest", "collision_source_digest"):
        if not _digest(request.get(key)):
            errors.append(f"frame_registration_{key}_invalid")
    if request.get("appearance_up_axis") not in {"Y", "Z"}:
        errors.append("frame_registration_appearance_up_axis_invalid")
    if request.get("collision_source_up_axis") != "Y":
        errors.append("frame_registration_collision_up_axis_invalid")
    for key, low, high in (
        ("minimum_opacity", 0.0, 1.0),
        ("trim_fraction", 0.5, 0.95),
        ("maximum_trimmed_rmse_scene_units", 0.01, 10.0),
        ("minimum_runner_up_ratio", 1.0, 10.0),
        ("minimum_scale_ratio", 0.1, 1.0),
        ("maximum_scale_ratio", 1.0, 10.0),
    ):
        item = request.get(key)
        if (
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            or not low <= float(item) <= high
        ):
            errors.append(f"frame_registration_{key}_invalid")
    cap = request.get("sample_cap_per_asset")
    if not isinstance(cap, int) or isinstance(cap, bool) or not 1000 <= cap <= 100_000:
        errors.append("frame_registration_sample_cap_invalid")
    if request.get("candidate_may_self_qualify") is not False:
        errors.append("frame_registration_self_qualification_forbidden")
    expected = canonical_digest(request, digest_field="request_digest")
    if supplied is not None and supplied != expected:
        errors.append("frame_registration_request_digest_mismatch")
    if errors:
        raise ExternalSceneFrameRegistrationError(errors)
    request["request_digest"] = expected
    return request


def _sample(points: np.ndarray, *, cap: int, seed: int) -> np.ndarray:
    if len(points) <= cap:
        return np.asarray(points, dtype=np.float64)
    generator = np.random.default_rng(seed)
    return np.asarray(points[generator.choice(len(points), cap, replace=False)], dtype=np.float64)


def _trimmed_rmse(distances: np.ndarray, fraction: float) -> float:
    cutoff = float(np.quantile(distances, fraction))
    retained = distances[distances <= cutoff]
    return float(np.sqrt(np.mean(np.square(retained))))


def register_external_scene_frames(
    *, analysis_splat_path: str | Path, collision_glb_path: str | Path, request: Mapping[str, Any]
) -> dict[str, Any]:
    admitted = build_external_scene_frame_registration_request(request)
    splat_path = Path(analysis_splat_path).resolve(strict=True)
    glb_path = Path(collision_glb_path).resolve(strict=True)
    if (
        splat_path.suffix.lower() != ".ply"
        or _sha256(splat_path) != admitted["analysis_splat_digest"]
    ):
        raise ExternalSceneFrameRegistrationError(["frame_registration_splat_binding_invalid"])
    if (
        glb_path.suffix.lower() != ".glb"
        or _sha256(glb_path) != admitted["collision_source_digest"]
    ):
        raise ExternalSceneFrameRegistrationError(["frame_registration_glb_binding_invalid"])
    splat = read_standard_3dgs_ply(splat_path)
    appearance = np.asarray(splat.xyz, dtype=np.float64)[
        np.asarray(splat.opacity_sigmoid) >= float(admitted["minimum_opacity"])
    ]
    collision, _, _ = _flatten_glb(glb_path)
    cap = int(admitted["sample_cap_per_asset"])
    appearance = _sample(appearance, cap=cap, seed=506)
    collision = _sample(np.asarray(collision), cap=cap, seed=507)
    if len(appearance) < 1000 or len(collision) < 1000:
        raise ExternalSceneFrameRegistrationError(["frame_registration_support_insufficient"])
    collision_tree = cKDTree(collision)
    trim = float(admitted["trim_fraction"])
    rows: list[dict[str, Any]] = []
    for rotation in _axis_rotations():
        transformed = appearance @ rotation.T
        source_extent = np.percentile(transformed, 99, axis=0) - np.percentile(
            transformed, 1, axis=0
        )
        target_extent = np.percentile(collision, 99, axis=0) - np.percentile(collision, 1, axis=0)
        scale = float(np.median(target_extent / np.maximum(source_extent, 1e-9)))
        transformed *= scale
        translation = np.median(collision, axis=0) - np.median(transformed, axis=0)
        transformed += translation
        forward_distances = collision_tree.query(transformed, k=1)[0]
        reverse_distances = cKDTree(transformed).query(collision, k=1)[0]
        forward_rmse = _trimmed_rmse(forward_distances, trim)
        reverse_rmse = _trimmed_rmse(reverse_distances, trim)
        rows.append(
            {
                "score": math.sqrt((forward_rmse**2 + reverse_rmse**2) / 2.0),
                "scale": scale,
                "rotation": rotation,
                "translation": translation,
                "forward_median": float(np.median(forward_distances)),
                "reverse_median": float(np.median(reverse_distances)),
            }
        )
    rows.sort(key=lambda row: float(row["score"]))
    best, runner_up = rows[:2]
    runner_up_ratio = float(runner_up["score"]) / max(float(best["score"]), 1e-12)
    blockers: list[str] = []
    if float(best["score"]) > float(admitted["maximum_trimmed_rmse_scene_units"]):
        blockers.append("frame_registration_error_above_threshold")
    if runner_up_ratio < float(admitted["minimum_runner_up_ratio"]):
        blockers.append("frame_registration_axis_solution_ambiguous")
    if (
        not float(admitted["minimum_scale_ratio"])
        <= float(best["scale"])
        <= float(admitted["maximum_scale_ratio"])
    ):
        blockers.append("frame_registration_provider_scale_ratio_out_of_range")
    rotation = np.asarray(best["rotation"], dtype=np.float64)
    translation = np.asarray(best["translation"], dtype=np.float64)
    scale = float(best["scale"])
    source_to_glb = np.eye(4, dtype=np.float64)
    source_to_glb[:3, :3] = scale * rotation
    source_to_glb[:3, 3] = translation
    glb_to_collision_stage = np.asarray(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0, 0, 0, 1]],
        dtype=np.float64,
    )
    source_to_collision_stage = glb_to_collision_stage @ source_to_glb
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "candidate_registered" if not blockers else "abstained",
        "request_digest": admitted["request_digest"],
        "appearance_scene_digest": admitted["appearance_scene_digest"],
        "analysis_splat_digest": admitted["analysis_splat_digest"],
        "collision_source_digest": admitted["collision_source_digest"],
        "sample_counts": {"appearance": len(appearance), "collision": len(collision)},
        "selected_axis_rotation": rotation.astype(int).tolist(),
        "estimated_scale_ratio": round(scale, 9),
        "translation_collision_source_units": [round(float(item), 9) for item in translation],
        "source_to_collision_stage_matrix": [
            round(float(item), 12) for item in source_to_collision_stage.reshape(-1)
        ],
        "symmetric_trimmed_rmse_scene_units": round(float(best["score"]), 9),
        "runner_up_ratio": round(runner_up_ratio, 9),
        "forward_median_distance_scene_units": round(float(best["forward_median"]), 9),
        "reverse_median_distance_scene_units": round(float(best["reverse_median"]), 9),
        "thresholds": {
            "trim_fraction": trim,
            "maximum_trimmed_rmse_scene_units": admitted["maximum_trimmed_rmse_scene_units"],
            "minimum_runner_up_ratio": admitted["minimum_runner_up_ratio"],
            "scale_ratio_range": [
                admitted["minimum_scale_ratio"],
                admitted["maximum_scale_ratio"],
            ],
        },
        "blockers": sorted(blockers),
        "metric_scale_proven": False,
        "collision_contact_proven": False,
        "candidate_may_self_qualify": False,
        "proof_effect": "derived_cross_export_frame_registration_candidate",
        "claim_ceiling": "appearance_to_collision_frame_binding_candidate",
    }
    result["scene_frame_binding_digest"] = canonical_digest(
        result, digest_field="scene_frame_binding_digest"
    )
    return result


def compose_registered_target_binding(
    *, target_binding: Mapping[str, Any], frame_registration: Mapping[str, Any]
) -> dict[str, Any]:
    if frame_registration.get("status") != "candidate_registered" or not _digest(
        frame_registration.get("scene_frame_binding_digest")
    ):
        raise ExternalSceneFrameRegistrationError(["target_binding_frame_registration_unavailable"])
    if target_binding.get("status") != "candidate_bound" or not _digest(
        target_binding.get("binding_evidence_digest")
    ):
        raise ExternalSceneFrameRegistrationError(["target_binding_source_candidate_invalid"])
    if target_binding.get("source_scene_digest") != frame_registration.get(
        "appearance_scene_digest"
    ):
        raise ExternalSceneFrameRegistrationError(["target_binding_scene_digest_mismatch"])
    position = np.asarray(target_binding.get("position_scene"), dtype=np.float64)
    matrix = np.asarray(
        frame_registration.get("source_to_collision_stage_matrix"), dtype=np.float64
    ).reshape(4, 4)
    transformed = matrix @ np.asarray([*position, 1.0], dtype=np.float64)
    scale = float(frame_registration["estimated_scale_ratio"])
    result = {
        "schema_version": COMPOSED_BINDING_SCHEMA,
        "status": "candidate_bound_in_collision_stage",
        "source_scene_digest": target_binding["source_scene_digest"],
        "collision_source_digest": frame_registration["collision_source_digest"],
        "source_binding_evidence_digest": target_binding["binding_evidence_digest"],
        "scene_frame_binding_digest": frame_registration["scene_frame_binding_digest"],
        "method": "rendered_depth_backprojection",
        "position_collision_stage": [round(float(item), 9) for item in transformed[:3]],
        "spatial_uncertainty_collision_stage_units": round(
            float(target_binding["spatial_uncertainty_scene_units"]) * scale, 9
        ),
        "metric_scale_proven": False,
        "collision_contact_proven": False,
        "proof_effect": "composed_visual_to_collision_stage_target_candidate",
        "claim_ceiling": "task_target_binding_candidate",
    }
    result["binding_evidence_digest"] = canonical_digest(
        result, digest_field="binding_evidence_digest"
    )
    return result


def transform_camera_specs_to_collision_stage(
    *, cameras: Sequence[Mapping[str, Any]], frame_registration: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Transform digest-bound appearance cameras into the packaged collision stage."""

    if frame_registration.get("status") != "candidate_registered":
        raise ExternalSceneFrameRegistrationError(["camera_frame_registration_unavailable"])
    matrix = np.asarray(
        frame_registration.get("source_to_collision_stage_matrix"), dtype=np.float64
    ).reshape(4, 4)
    linear = matrix[:3, :3]
    transformed: list[dict[str, Any]] = []
    for row in cameras:
        spec = row.get("spec") if isinstance(row, Mapping) else None
        if not isinstance(spec, Mapping):
            raise ExternalSceneFrameRegistrationError(["camera_spec_invalid"])
        position = np.asarray(spec.get("pos"), dtype=np.float64)
        target = np.asarray(spec.get("target"), dtype=np.float64)
        up = np.asarray(spec.get("up"), dtype=np.float64)
        if any(
            value.shape != (3,) or not np.isfinite(value).all() for value in (position, target, up)
        ):
            raise ExternalSceneFrameRegistrationError(["camera_spec_vector_invalid"])
        transformed_position = matrix @ np.asarray([*position, 1.0])
        transformed_target = matrix @ np.asarray([*target, 1.0])
        transformed_up = linear @ up
        norm = float(np.linalg.norm(transformed_up))
        if norm <= 1e-9:
            raise ExternalSceneFrameRegistrationError(["camera_spec_up_degenerate"])
        transformed_up /= norm
        transformed.append(
            {
                "id": str(row.get("id") or ""),
                "spec": {
                    "pos": [round(float(item), 9) for item in transformed_position[:3]],
                    "target": [round(float(item), 9) for item in transformed_target[:3]],
                    "fov": float(spec.get("fov")),
                    "up": [round(float(item), 9) for item in transformed_up],
                },
            }
        )
    return transformed


__all__ = [
    "COMPOSED_BINDING_SCHEMA",
    "REQUEST_SCHEMA",
    "RESULT_SCHEMA",
    "ExternalSceneFrameRegistrationError",
    "build_external_scene_frame_registration_request",
    "compose_registered_target_binding",
    "register_external_scene_frames",
    "transform_camera_specs_to_collision_stage",
]
