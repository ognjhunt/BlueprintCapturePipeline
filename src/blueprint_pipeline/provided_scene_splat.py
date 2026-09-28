"""Inspect a completed 3DGS asset with the existing decoder and room-survey seam."""
from __future__ import annotations

from dataclasses import replace
import io
import math

import numpy as np

from .decision_evidence_contracts import canonical_digest
from .gaussian_splat_decode import _parse_ply_header, read_standard_3dgs_ply
from .splat_scene_analysis import analyze_scene
from .task_evaluation_scene_configuration_submission_inputs import sha
from .task_evaluation_scene_progression_state import safe_path, require

MAX_BYTES = 512 * 1024 * 1024
MAX_GAUSSIANS = 5_000_000


def inspect_splat(path, *, coordinate_frame_declaration):
    path = safe_path(path)
    require(path.is_file() and 0 < path.stat().st_size <= MAX_BYTES, "provided_splat_size_invalid")
    scale = coordinate_frame_declaration.get("meters_per_unit")
    axis = coordinate_frame_declaration.get("up_axis")
    require(type(scale) in (int, float) and math.isfinite(scale) and 0 < scale <= 1000
            and axis in {"Y", "Z"}, "provided_splat_declared_frame_required")
    with path.open("rb") as stream:
        header = stream.read(1024 * 1024)
    fmt, count, properties, offset = _parse_ply_header(io.BytesIO(header))
    names = [row[1] for row in properties]
    require(fmt == "binary_little_endian" and 0 < count <= MAX_GAUSSIANS
            and 14 <= len(properties) <= 256 and len(names) == len(set(names))
            and path.stat().st_size == offset + count * len(properties) * 4,
            "provided_splat_layout_invalid")
    splat = read_standard_3dgs_ply(path)
    require(all(np.isfinite(value).all() for value in (splat.xyz, splat.opacity, splat.f_dc, splat.scales, splat.quats))
            and (splat.sh_rest is None or np.isfinite(splat.sh_rest).all())
            and (np.linalg.norm(splat.quats, axis=1) > 0).all(), "provided_splat_values_invalid")
    # This is an analysis projection only. The retained input bytes are never
    # transformed, sampled, rewritten, or presented as calibrated source views.
    survey_input = replace(splat, xyz=splat.xyz * scale)
    survey = analyze_scene(survey_input, up_axis=1 if axis == "Y" else 2).to_dict()
    value = {"schema_version": "provided_scene_splat_inspection.v1", "asset_digest": sha(path),
        "source_kind": "provided_scene_splat", "retained_gaussian_count": count,
        "properties": names, "scene_survey": survey,
        "coordinate_system": {"declared_meters_per_unit": scale, "declared_up_axis": axis,
                              "physical_scale_measured": False},
        "whole_retained_splat_surveyed": True, "survey_is_method_input": False,
        "captured_observations_supplied": False, "renderer_qualified": False,
        "collision_qualified": False, "physics_qualified": False,
        "unseen_or_uncaptured_regions_recovered": False}
    value["inspection_digest"] = canonical_digest(value, digest_field="inspection_digest")
    return value
