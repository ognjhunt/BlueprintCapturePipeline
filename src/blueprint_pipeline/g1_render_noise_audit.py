"""G1 textured-robot render-noise audit: variant plan, frame metrics, gates, interpretation.

Isolates the visual-quality regression seen on textured/physically shaded G1 arms in close
robot-POV manipulation frames (heavy path-tracing grain, dark denoiser blotches, unreadable
hands) versus the much cleaner untextured white proxy. The audit holds scene, task, stance,
camera, arm pose, resolution, and seed constant while changing exactly one material/render
variable per comparison, so the failing part of the textured render path can be identified:
missing texture assets, sample starvation, denoiser behavior, PBR material response,
lighting underexposure, or camera/pose clipping.

This module is pure Python (numpy + Pillow only; no Isaac/omni imports) so the variant plan,
image statistics, gates, and interpretation rules are unit-testable and reusable both on the
GPU worker and for local re-analysis of collected artifacts. The GPU-side variant renderer
lives in ``scripts/run_isaac_g1_kitchen_parity_eval.py`` (``--render-noise-audit``) and reuses
the normal dynamic path: task string -> target resolution -> task stance -> robot pose ->
camera contract -> render variants. No kitchen/site coordinates are hardcoded here.

Claim boundary: this is a simulator/render-quality audit only. It does not prove physical
robot readiness, task success, contact correctness, policy quality, or WAM rank fidelity.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from .common import ensure_dir, utc_now_iso, write_json
except ImportError:  # flat bundle copy on the GPU worker: no package context
    from datetime import datetime, timezone

    def ensure_dir(path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)

    def utc_now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def write_json(path: Path, payload: Mapping[str, Any]) -> None:
        ensure_dir(path.parent)
        path.write_text(json.dumps(dict(payload), indent=2), encoding="utf-8")

AUDIT_MANIFEST_SCHEMA_VERSION = "textured_robot_render_noise_audit_manifest.v1"
MATERIAL_RESOLUTION_SCHEMA_VERSION = "robot_material_resolution_manifest.v1"
RENDER_SETTINGS_MANIFEST_SCHEMA_VERSION = "render_settings_manifest.v1"
VARIANT_PLAN_SCHEMA_VERSION = "g1_render_noise_audit_variant_plan.v1"
FRAME_STATS_SCHEMA_VERSION = "g1_render_noise_audit_frame_stats.v1"
WAM_SEED_MEDIA_CONTRACT_SCHEMA_VERSION = "g1_seed_frame_robot_material_contract.v1"

AUDIT_MANIFEST_NAME = "textured_robot_render_noise_audit_manifest.json"
MATERIAL_RESOLUTION_MANIFEST_NAME = "robot_material_resolution_manifest.json"
RENDER_SETTINGS_MANIFEST_NAME = "render_settings_manifest.json"
CAMERA_CONTRACT_NAME = "camera_contract.json"
WORKER_RUN_MANIFEST_NAME = "audit_run_manifest.json"
CONTACT_SHEET_NAME = "render_noise_audit_contact_sheet.png"
FRAME_STATS_NAME = "render_noise_audit_frame_stats.json"
AUDIT_SUBDIR_NAME = "render_noise_audit"
VARIANTS_SUBDIR_NAME = "variants"
VARIANT_FRAME_NAME = "frame_raw.png"
VARIANT_MANIFEST_NAME = "variant_manifest.json"

# Robot material modes an artifact may claim for a seed frame. ``textured_unverified`` exists
# so a run whose texture references did NOT all resolve can never silently upgrade itself to
# ``verified_textured``.
ROBOT_MATERIAL_MODE_VERIFIED_TEXTURED = "verified_textured"
ROBOT_MATERIAL_MODE_TEXTURED_UNVERIFIED = "textured_unverified"
ROBOT_MATERIAL_MODE_SIMPLIFIED_DIFFUSE = "simplified_diffuse"
ROBOT_MATERIAL_MODE_WHITE_PROXY = "white_proxy"
WAM_ALLOWED_ROBOT_MATERIAL_MODES = (
    ROBOT_MATERIAL_MODE_VERIFIED_TEXTURED,
    ROBOT_MATERIAL_MODE_SIMPLIFIED_DIFFUSE,
    ROBOT_MATERIAL_MODE_WHITE_PROXY,
)

# Existing pipeline labels (kitchen_task_scaling_preflight and the parity runner) mapped onto
# the audit's explicit modes. "preserve authored" alone is NOT texture-verification evidence.
LEGACY_ROBOT_MATERIAL_MODE_MAP = {
    "neutral_matte_untextured_g1": ROBOT_MATERIAL_MODE_WHITE_PROXY,
    "preserve_authored_g1_materials_when_available": ROBOT_MATERIAL_MODE_TEXTURED_UNVERIFIED,
}

# What the variant renderer binds on the robot subtree for each variant.
VARIANT_MATERIAL_WHITE_PROXY = "white_proxy"
VARIANT_MATERIAL_TEXTURED_ORIGINAL = "textured_original"
VARIANT_MATERIAL_SIMPLIFIED_DIFFUSE = "simplified_diffuse"

RENDER_BUDGET_CURRENT_DEFAULT = "current_default"
RENDER_BUDGET_HIGH = "high"

# --- metric thresholds (frame-level; luma in [0, 255]) ---
DARK_PIXEL_LUMA_MAX = 32.0
NEAR_BLACK_PIXEL_LUMA_MAX = 16.0
BRIGHT_PIXEL_LUMA_MIN = 224.0
EDGE_GRADIENT_MIN = 18.0  # matches wam_generated_video_review edge_density convention
BLACK_WEDGE_CELL_LUMA_MAX = 24.0
BLACK_WEDGE_GRID_MAX_CELLS = 64  # longest grid side for the border-wedge connected-component scan

# --- gate thresholds ---
GATE_MIN_CENTER_MEAN_LUMA = 30.0
GATE_MAX_CENTER_DARK_PIXEL_RATIO = 0.70
GATE_MAX_BLACK_EDGE_WEDGE_RATIO = 0.18
GATE_MIN_EDGE_DENSITY_RATIO_TO_PROXY = 0.25
# Denoised textured must not be darker or lower-structure than raw textured beyond tolerance.
GATE_DENOISER_MAX_MEAN_LUMA_DROP = 18.0
GATE_DENOISER_MIN_EDGE_DENSITY_RATIO = 0.50

# Advisory noise grading from the high-frequency noise estimate (Immerkaer residual sigma).
NOISE_GRADE_CLEAN_MAX = 3.0
NOISE_GRADE_MODERATE_MAX = 8.0

AUDIT_CLAIM_BOUNDARY = (
    "Simulator/render-quality audit only. Frame statistics, gates, and diagnoses describe "
    "renderer/material/lighting behavior for review seed media. They do not prove physical "
    "robot readiness, task success, contact correctness, policy quality, or WAM rank fidelity."
)

TEXTURED_MATERIAL_UNVERIFIED_BLOCKER = "textured_robot_material_unverified_texture_refs_missing_or_absent"


@dataclass(frozen=True)
class RenderNoiseAuditVariant:
    """One row of the audit matrix: exactly one material/render lever vs its comparison peer."""

    variant_id: str
    label: str
    robot_material: str
    denoiser_enabled: bool
    render_budget: str
    lighting_boost: bool
    purpose: str
    exploratory: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "label": self.label,
            "robot_material": self.robot_material,
            "denoiser_enabled": self.denoiser_enabled,
            "render_budget": self.render_budget,
            "lighting_boost": self.lighting_boost,
            "purpose": self.purpose,
            "exploratory": self.exploratory,
        }


def build_default_variant_matrix() -> tuple[RenderNoiseAuditVariant, ...]:
    """The spec's minimum variant matrix (A-G) on one dynamic scene/camera/pose."""
    return (
        RenderNoiseAuditVariant(
            variant_id="A", label="white_proxy_denoised_default_budget",
            robot_material=VARIANT_MATERIAL_WHITE_PROXY, denoiser_enabled=True,
            render_budget=RENDER_BUDGET_CURRENT_DEFAULT, lighting_boost=False,
            purpose="known clean proxy baseline"),
        RenderNoiseAuditVariant(
            variant_id="B", label="textured_raw_default_budget",
            robot_material=VARIANT_MATERIAL_TEXTURED_ORIGINAL, denoiser_enabled=False,
            render_budget=RENDER_BUDGET_CURRENT_DEFAULT, lighting_boost=False,
            purpose="raw textured-noise baseline"),
        RenderNoiseAuditVariant(
            variant_id="C", label="textured_denoised_default_budget",
            robot_material=VARIANT_MATERIAL_TEXTURED_ORIGINAL, denoiser_enabled=True,
            render_budget=RENDER_BUDGET_CURRENT_DEFAULT, lighting_boost=False,
            purpose="denoiser regression check"),
        RenderNoiseAuditVariant(
            variant_id="D", label="textured_raw_high_budget",
            robot_material=VARIANT_MATERIAL_TEXTURED_ORIGINAL, denoiser_enabled=False,
            render_budget=RENDER_BUDGET_HIGH, lighting_boost=False,
            purpose="test sample starvation"),
        RenderNoiseAuditVariant(
            variant_id="E", label="textured_denoised_high_budget",
            robot_material=VARIANT_MATERIAL_TEXTURED_ORIGINAL, denoiser_enabled=True,
            render_budget=RENDER_BUDGET_HIGH, lighting_boost=False,
            purpose="test denoiser with enough samples"),
        RenderNoiseAuditVariant(
            variant_id="F", label="simplified_diffuse_denoised_default_budget",
            robot_material=VARIANT_MATERIAL_SIMPLIFIED_DIFFUSE, denoiser_enabled=True,
            render_budget=RENDER_BUDGET_CURRENT_DEFAULT, lighting_boost=False,
            purpose="test whether PBR/specular maps are unstable"),
        RenderNoiseAuditVariant(
            variant_id="G", label="textured_denoised_default_budget_bright_lighting",
            robot_material=VARIANT_MATERIAL_TEXTURED_ORIGINAL, denoiser_enabled=True,
            render_budget=RENDER_BUDGET_CURRENT_DEFAULT, lighting_boost=True,
            purpose="test shadow/underexposure"),
    )


# Pairwise comparisons that each isolate exactly one changed variable. Any other cross-variant
# comparison is exploratory and must not be the main pass/fail evidence.
SINGLE_VARIABLE_COMPARISON_PAIRS: tuple[dict[str, Any], ...] = (
    {"pair": ("A", "C"), "isolates": "robot_material_white_proxy_vs_textured"},
    {"pair": ("B", "C"), "isolates": "denoiser_at_default_budget"},
    {"pair": ("D", "E"), "isolates": "denoiser_at_high_budget"},
    {"pair": ("B", "D"), "isolates": "render_budget_raw"},
    {"pair": ("C", "E"), "isolates": "render_budget_denoised"},
    {"pair": ("C", "F"), "isolates": "pbr_specular_maps_vs_simplified_diffuse"},
    {"pair": ("C", "G"), "isolates": "task_lighting_brightness"},
)

_VARIANT_LEVERS = ("robot_material", "denoiser_enabled", "render_budget", "lighting_boost")


def variant_lever_deltas(a: RenderNoiseAuditVariant, b: RenderNoiseAuditVariant) -> tuple[str, ...]:
    return tuple(lever for lever in _VARIANT_LEVERS if getattr(a, lever) != getattr(b, lever))


def default_execution_order(variants: Sequence[RenderNoiseAuditVariant]) -> list[str]:
    """Material application on the worker is monotonic (authored -> simplified overrides ->
    white-proxy overrides), so overrides never need to be un-authored mid-run: all
    textured-original variants render first, then simplified diffuse, then white proxy."""
    rank = {
        VARIANT_MATERIAL_TEXTURED_ORIGINAL: 0,
        VARIANT_MATERIAL_SIMPLIFIED_DIFFUSE: 1,
        VARIANT_MATERIAL_WHITE_PROXY: 2,
    }
    ordered = sorted(
        variants,
        key=lambda v: (rank.get(v.robot_material, 3), v.variant_id),
    )
    return [v.variant_id for v in ordered]


def build_variant_plan(
    variants: Sequence[RenderNoiseAuditVariant] | None = None,
) -> dict[str, Any]:
    variant_list = list(variants) if variants is not None else list(build_default_variant_matrix())
    plan = {
        "schema_version": VARIANT_PLAN_SCHEMA_VERSION,
        "variants": [v.to_dict() for v in variant_list],
        "single_variable_comparison_pairs": [
            {"pair": list(entry["pair"]), "isolates": entry["isolates"]}
            for entry in SINGLE_VARIABLE_COMPARISON_PAIRS
            if all(any(v.variant_id == vid for v in variant_list) for vid in entry["pair"])
        ],
        "execution_order": default_execution_order(variant_list),
        "execution_order_reason": (
            "material overrides are applied monotonically (textured first, then simplified "
            "diffuse, then white proxy) so authored materials never need to be restored mid-run"
        ),
        "claim_boundary": AUDIT_CLAIM_BOUNDARY,
    }
    plan["blockers"] = validate_variant_plan(plan)
    return plan


def parse_variant_plan(plan: Mapping[str, Any]) -> list[RenderNoiseAuditVariant]:
    variants: list[RenderNoiseAuditVariant] = []
    for row in plan.get("variants") or []:
        if not isinstance(row, Mapping):
            continue
        variants.append(RenderNoiseAuditVariant(
            variant_id=str(row.get("variant_id")),
            label=str(row.get("label") or row.get("variant_id")),
            robot_material=str(row.get("robot_material")),
            denoiser_enabled=bool(row.get("denoiser_enabled")),
            render_budget=str(row.get("render_budget") or RENDER_BUDGET_CURRENT_DEFAULT),
            lighting_boost=bool(row.get("lighting_boost")),
            purpose=str(row.get("purpose") or ""),
            exploratory=bool(row.get("exploratory")),
        ))
    return variants


def validate_variant_plan(plan: Mapping[str, Any]) -> list[str]:
    """Structural honesty checks: unique ids, known levers, and every declared single-variable
    comparison pair differing in exactly one lever (otherwise it must be exploratory)."""
    blockers: list[str] = []
    variants = parse_variant_plan(plan)
    ids = [v.variant_id for v in variants]
    if len(ids) != len(set(ids)):
        blockers.append("variant_plan_duplicate_variant_ids")
    by_id = {v.variant_id: v for v in variants}
    known_materials = {
        VARIANT_MATERIAL_WHITE_PROXY,
        VARIANT_MATERIAL_TEXTURED_ORIGINAL,
        VARIANT_MATERIAL_SIMPLIFIED_DIFFUSE,
    }
    known_budgets = {RENDER_BUDGET_CURRENT_DEFAULT, RENDER_BUDGET_HIGH}
    for v in variants:
        if v.robot_material not in known_materials:
            blockers.append(f"variant_plan_unknown_robot_material:{v.variant_id}")
        if v.render_budget not in known_budgets:
            blockers.append(f"variant_plan_unknown_render_budget:{v.variant_id}")
    for entry in plan.get("single_variable_comparison_pairs") or []:
        pair = list((entry or {}).get("pair") or [])
        if len(pair) != 2 or any(vid not in by_id for vid in pair):
            blockers.append(f"variant_plan_comparison_pair_unresolved:{pair}")
            continue
        deltas = variant_lever_deltas(by_id[pair[0]], by_id[pair[1]])
        if len(deltas) != 1:
            blockers.append(
                f"variant_plan_comparison_pair_not_single_variable:{pair[0]}:{pair[1]}:{','.join(deltas)}"
            )
    return blockers


# ============================ frame statistics ============================

def _luma_array(image: Any) -> Any:
    import numpy as np

    if isinstance(image, (str, Path)):
        from PIL import Image

        with Image.open(image) as img:
            arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    else:
        arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 3:
        if arr.shape[2] >= 3:
            arr = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]
        else:
            arr = arr[:, :, 0]
    return arr


def estimate_high_frequency_noise(luma: Any) -> float:
    """Immerkaer's fast noise-variance estimate: the mean absolute response of the
    zero-mean Laplacian-difference kernel [[1,-2,1],[-2,4,-2],[1,-2,1]] scaled to a sigma.
    Structured edges contribute, but at matched scene/pose the estimate cleanly separates
    path-tracing speckle from converged frames."""
    import numpy as np

    arr = np.asarray(luma, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 3 or arr.shape[1] < 3:
        return 0.0
    center = arr[1:-1, 1:-1]
    response = (
        4.0 * center
        - 2.0 * (arr[:-2, 1:-1] + arr[2:, 1:-1] + arr[1:-1, :-2] + arr[1:-1, 2:])
        + arr[:-2, :-2] + arr[:-2, 2:] + arr[2:, :-2] + arr[2:, 2:]
    )
    height, width = center.shape
    sigma = math.sqrt(math.pi / 2.0) / (6.0 * width * height) * float(np.abs(response).sum())
    return float(sigma)


def black_edge_wedge_ratio(luma: Any) -> float:
    """Fraction of the frame covered by the largest dark connected region touching the frame
    border, computed on a coarse block-mean grid. Large values indicate the black wedges seen
    with camera self-occlusion/clipping into doors, panels, or the robot's own head mesh."""
    import numpy as np

    arr = np.asarray(luma, dtype=np.float32)
    if arr.ndim != 2 or arr.size == 0:
        return 0.0
    height, width = arr.shape
    scale = max(1, int(math.ceil(max(height, width) / float(BLACK_WEDGE_GRID_MAX_CELLS))))
    grid_h = height // scale
    grid_w = width // scale
    if grid_h < 2 or grid_w < 2:
        grid = arr  # tiny frame: scan raw pixels
        grid_h, grid_w = arr.shape
    else:
        grid = (
            arr[: grid_h * scale, : grid_w * scale]
            .reshape(grid_h, scale, grid_w, scale)
            .mean(axis=(1, 3))
        )
    dark = grid < BLACK_WEDGE_CELL_LUMA_MAX
    if not bool(dark.any()):
        return 0.0
    visited = np.zeros_like(dark, dtype=bool)
    best = 0
    for start_r in range(grid_h):
        for start_c in range(grid_w):
            if not dark[start_r, start_c] or visited[start_r, start_c]:
                continue
            stack = [(start_r, start_c)]
            visited[start_r, start_c] = True
            size = 0
            touches_border = False
            while stack:
                r, c = stack.pop()
                size += 1
                if r == 0 or c == 0 or r == grid_h - 1 or c == grid_w - 1:
                    touches_border = True
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < grid_h and 0 <= nc < grid_w and dark[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            if touches_border:
                best = max(best, size)
    return float(best) / float(grid_h * grid_w)


def compute_frame_stats(image: Any) -> dict[str, Any]:
    """Per-frame statistics for one audit variant PNG (or HxWx3 array)."""
    import numpy as np

    luma = _luma_array(image)
    height, width = int(luma.shape[0]), int(luma.shape[1])
    total = float(luma.size) or 1.0
    hist, _ = np.histogram(luma, bins=256, range=(0.0, 255.0))
    probs = hist.astype(np.float64) / total
    nonzero = probs[probs > 0]
    entropy_bits = float(-(nonzero * np.log2(nonzero)).sum()) if nonzero.size else 0.0
    gy, gx = np.gradient(luma)
    grad_mag = np.hypot(gx, gy)

    def _region(region: Any) -> dict[str, float]:
        rtotal = float(region.size) or 1.0
        rhist, _ = np.histogram(region, bins=256, range=(0.0, 255.0))
        rprobs = rhist.astype(np.float64) / rtotal
        rnonzero = rprobs[rprobs > 0]
        r_gy, r_gx = np.gradient(region)
        r_grad = np.hypot(r_gx, r_gy)
        return {
            "mean_luma": round(float(region.mean()), 4),
            "std_luma": round(float(region.std()), 4),
            "dark_pixel_ratio": round(float((region < DARK_PIXEL_LUMA_MAX).sum() / rtotal), 6),
            "edge_density": round(float((r_grad > EDGE_GRADIENT_MIN).sum() / rtotal), 6),
            "entropy_bits": round(
                float(-(rnonzero * np.log2(rnonzero)).sum()) if rnonzero.size else 0.0, 4
            ),
        }

    center = luma[height // 4: max(height // 4 + 1, 3 * height // 4),
                  width // 4: max(width // 4 + 1, 3 * width // 4)]
    return {
        "schema_version": FRAME_STATS_SCHEMA_VERSION,
        "width": width,
        "height": height,
        "mean_luma": round(float(luma.mean()), 4),
        "std_luma": round(float(luma.std()), 4),
        "luma_min": round(float(luma.min()), 4),
        "luma_max": round(float(luma.max()), 4),
        "luma_range": round(float(luma.max() - luma.min()), 4),
        "dark_pixel_ratio": round(float((luma < DARK_PIXEL_LUMA_MAX).sum() / total), 6),
        "near_black_pixel_ratio": round(float((luma < NEAR_BLACK_PIXEL_LUMA_MAX).sum() / total), 6),
        "bright_pixel_ratio": round(float((luma > BRIGHT_PIXEL_LUMA_MIN).sum() / total), 6),
        "entropy_bits": round(entropy_bits, 4),
        "edge_density": round(float((grad_mag > EDGE_GRADIENT_MIN).sum() / total), 6),
        "high_frequency_noise_estimate": round(estimate_high_frequency_noise(luma), 4),
        "black_edge_wedge_ratio": round(black_edge_wedge_ratio(luma), 6),
        "center_crop": _region(center),
    }


def noise_grade(stats: Mapping[str, Any]) -> str:
    value = float(stats.get("high_frequency_noise_estimate") or 0.0)
    if value <= NOISE_GRADE_CLEAN_MAX:
        return "clean"
    if value <= NOISE_GRADE_MODERATE_MAX:
        return "moderate"
    return "noisy"


# ============================ material resolution ============================

def summarize_material_resolution(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize the worker's raw robot-material traversal into the resolution manifest.

    ``raw`` rows come from the GPU worker's USD traversal of the robot subtree:
    materials with their shader ids and texture asset references (authored path, resolved
    path, existence), plus gprim/mesh counts and unbound-gprim counts.
    """
    materials = [m for m in (raw.get("materials") or []) if isinstance(m, Mapping)]
    texture_refs: list[dict[str, Any]] = []
    missing_refs: list[dict[str, Any]] = []
    shader_ids: set[str] = set()
    for material in materials:
        for sid in material.get("shader_ids") or []:
            if sid:
                shader_ids.add(str(sid))
        for ref in material.get("texture_refs") or []:
            if not isinstance(ref, Mapping):
                continue
            row = {
                "material_path": material.get("path"),
                "input": ref.get("input"),
                "authored_path": ref.get("authored_path"),
                "resolved_path": ref.get("resolved_path"),
                "exists": bool(ref.get("exists")),
            }
            texture_refs.append(row)
            if not row["exists"]:
                missing_refs.append(row)
    gprim_count = int(raw.get("gprim_count") or 0)
    unbound = int(raw.get("gprims_without_material") or 0)
    blockers: list[str] = []
    if gprim_count == 0:
        blockers.append("robot_visual_mesh_missing")
    if unbound > 0:
        blockers.append("robot_material_bindings_missing")
    if missing_refs:
        blockers.append("robot_texture_references_missing")
    textured_evidence = bool(texture_refs) and not missing_refs
    return {
        "schema_version": MATERIAL_RESOLUTION_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "status": "completed" if not blockers else "blocked",
        "robot_prim_path": raw.get("robot_prim_path"),
        "robot_asset_uri": raw.get("robot_asset_uri"),
        "resolved_visual_asset": raw.get("resolved_visual_asset"),
        "robot_visual_mesh_count": int(raw.get("mesh_count") or 0),
        "robot_gprim_count": gprim_count,
        "robot_material_count": len(materials),
        "material_binding_missing_count": unbound,
        "texture_reference_count": len(texture_refs),
        "texture_reference_missing_count": len(missing_refs),
        "missing_texture_references": missing_refs,
        "texture_references": texture_refs,
        "shader_ids": sorted(shader_ids),
        "materials": list(materials),
        "textured_material_evidence_present": textured_evidence,
        "blockers": blockers,
        "claim_boundary": (
            "Records USD material/texture resolution state on the render worker only. It does "
            "not prove material visual correctness, calibrated appearance, or task evidence."
        ),
    }


def classify_robot_material_mode(
    *, requested_material: str, material_resolution: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Honest robot-material-mode label for one variant/seed frame.

    ``textured_original`` may only be labeled ``verified_textured`` when the material
    resolution manifest proves texture references exist AND none are missing; otherwise the
    output stays ``textured_unverified`` with an explicit blocker.
    """
    requested = str(requested_material)
    if requested == VARIANT_MATERIAL_WHITE_PROXY:
        return {"robot_material_mode": ROBOT_MATERIAL_MODE_WHITE_PROXY, "blockers": []}
    if requested == VARIANT_MATERIAL_SIMPLIFIED_DIFFUSE:
        return {"robot_material_mode": ROBOT_MATERIAL_MODE_SIMPLIFIED_DIFFUSE, "blockers": []}
    if requested != VARIANT_MATERIAL_TEXTURED_ORIGINAL:
        return {
            "robot_material_mode": ROBOT_MATERIAL_MODE_TEXTURED_UNVERIFIED,
            "blockers": [f"unknown_requested_robot_material:{requested}"],
        }
    resolution = material_resolution or {}
    evidence = bool(resolution.get("textured_material_evidence_present"))
    missing = int(resolution.get("texture_reference_missing_count") or 0)
    ref_count = int(resolution.get("texture_reference_count") or 0)
    if evidence and missing == 0 and ref_count > 0:
        return {"robot_material_mode": ROBOT_MATERIAL_MODE_VERIFIED_TEXTURED, "blockers": []}
    return {
        "robot_material_mode": ROBOT_MATERIAL_MODE_TEXTURED_UNVERIFIED,
        "blockers": [TEXTURED_MATERIAL_UNVERIFIED_BLOCKER],
    }


def normalize_legacy_robot_material_mode(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text in (
        ROBOT_MATERIAL_MODE_VERIFIED_TEXTURED,
        ROBOT_MATERIAL_MODE_TEXTURED_UNVERIFIED,
        ROBOT_MATERIAL_MODE_SIMPLIFIED_DIFFUSE,
        ROBOT_MATERIAL_MODE_WHITE_PROXY,
    ):
        return text
    return LEGACY_ROBOT_MATERIAL_MODE_MAP.get(text)


# ============================ gates ============================

def evaluate_variant_gates(
    *,
    variant: RenderNoiseAuditVariant,
    stats: Mapping[str, Any],
    proxy_stats: Mapping[str, Any] | None,
    raw_textured_stats: Mapping[str, Any] | None,
    visibility: Mapping[str, Any] | None,
    material_resolution: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Seed-frame gates for one variant against the spec's checklist.

    ``visibility`` carries the pose-constant arm/end-effector/target visibility record from the
    worker (projected arm-link geometry plus optional robot-pixel mask evidence).
    """
    def _stat(mapping: Mapping[str, Any], key: str, default: float) -> float:
        value = mapping.get(key)
        return default if value is None else float(value)

    center = stats.get("center_crop") or {}
    vis = visibility or {}
    gates: dict[str, Any] = {}
    gates["both_arms_visible"] = bool(vis.get("left_arm_visible")) and bool(vis.get("right_arm_visible"))
    gates["both_end_effectors_visible"] = bool(vis.get("both_end_effectors_visible"))
    gates["target_visible"] = bool(vis.get("target_in_frame"))
    gates["no_large_black_edge_wedge"] = (
        _stat(stats, "black_edge_wedge_ratio", 0.0) <= GATE_MAX_BLACK_EDGE_WEDGE_RATIO
    )
    gates["no_camera_self_occlusion_suspected"] = gates["no_large_black_edge_wedge"]
    gates["luma_not_collapsed_in_task_region"] = (
        _stat(center, "mean_luma", 0.0) >= GATE_MIN_CENTER_MEAN_LUMA
        and _stat(center, "dark_pixel_ratio", 1.0) <= GATE_MAX_CENTER_DARK_PIXEL_RATIO
    )
    if proxy_stats is not None and variant.robot_material != VARIANT_MATERIAL_WHITE_PROXY:
        proxy_edges = _stat(proxy_stats, "edge_density", 0.0)
        if proxy_edges > 0.0:
            ratio = _stat(stats, "edge_density", 0.0) / proxy_edges
            gates["edge_structure_preserved_vs_proxy"] = ratio >= GATE_MIN_EDGE_DENSITY_RATIO_TO_PROXY
            gates["edge_density_ratio_to_proxy"] = round(ratio, 4)
        else:
            gates["edge_structure_preserved_vs_proxy"] = True
            gates["edge_density_ratio_to_proxy"] = None
    denoiser_regression = None
    if (
        variant.robot_material == VARIANT_MATERIAL_TEXTURED_ORIGINAL
        and variant.denoiser_enabled
        and raw_textured_stats is not None
    ):
        luma_drop = _stat(raw_textured_stats, "mean_luma", 0.0) - _stat(stats, "mean_luma", 0.0)
        raw_edges = _stat(raw_textured_stats, "edge_density", 0.0)
        edge_ratio = _stat(stats, "edge_density", 0.0) / raw_edges if raw_edges > 0.0 else 1.0
        # A speckle-dominated raw reference inflates edge_density with noise, so the
        # structure half of the regression check is only meaningful against a raw frame
        # that is not itself graded noisy; the darkness half is always measurable.
        raw_reference_noisy = noise_grade(raw_textured_stats) == "noisy"
        lower_structure = (
            edge_ratio < GATE_DENOISER_MIN_EDGE_DENSITY_RATIO if not raw_reference_noisy else False
        )
        denoiser_regression = {
            "mean_luma_drop_from_raw": round(luma_drop, 4),
            "edge_density_ratio_to_raw": round(edge_ratio, 4),
            "raw_reference_noisy_structure_check_skipped": raw_reference_noisy,
            "darker_than_raw_beyond_tolerance": luma_drop > GATE_DENOISER_MAX_MEAN_LUMA_DROP,
            "lower_structure_than_raw_beyond_tolerance": lower_structure,
        }
        gates["denoised_not_darker_or_lower_structure_than_raw"] = not (
            denoiser_regression["darker_than_raw_beyond_tolerance"]
            or denoiser_regression["lower_structure_than_raw_beyond_tolerance"]
        )
    material_mode = classify_robot_material_mode(
        requested_material=variant.robot_material,
        material_resolution=material_resolution,
    )
    blockers = [
        f"gate_failed:{name}" for name, value in gates.items()
        if isinstance(value, bool) and not value
    ]
    blockers.extend(material_mode["blockers"])
    return {
        "variant_id": variant.variant_id,
        "gates": gates,
        "denoiser_regression": denoiser_regression,
        "robot_material_mode": material_mode["robot_material_mode"],
        "noise_grade": noise_grade(stats),
        "passed": not any(b.startswith("gate_failed:") for b in blockers),
        "blockers": blockers,
    }


# ============================ interpretation rules ============================

def _variant_clean(stats: Mapping[str, Any] | None, gate: Mapping[str, Any] | None) -> bool | None:
    if stats is None:
        return None
    grade_ok = noise_grade(stats) != "noisy"
    luma_ok = True
    if gate is not None:
        luma_ok = bool((gate.get("gates") or {}).get("luma_not_collapsed_in_task_region", True))
    return grade_ok and luma_ok


def interpret_audit(
    *,
    stats_by_id: Mapping[str, Mapping[str, Any]],
    gates_by_id: Mapping[str, Mapping[str, Any]],
    material_resolution: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Apply the spec's interpretation rules over whichever variants actually ran.

    Returns every matched finding plus a single ``primary_diagnosis`` chosen by priority:
    camera/pose problems and missing assets invalidate the material comparison entirely, so
    they outrank sampling, denoiser, PBR-response, and lighting findings.
    """
    def stats(vid: str) -> Mapping[str, Any] | None:
        return stats_by_id.get(vid)

    def gate(vid: str) -> Mapping[str, Any] | None:
        return gates_by_id.get(vid)

    def clean(vid: str) -> bool | None:
        return _variant_clean(stats(vid), gate(vid))

    findings: list[dict[str, Any]] = []

    ran = [vid for vid in stats_by_id]
    wedge_or_arms_failures = []
    for vid in ran:
        g = (gate(vid) or {}).get("gates") or {}
        if g.get("no_large_black_edge_wedge") is False or g.get("both_arms_visible") is False:
            wedge_or_arms_failures.append(vid)
    if ran and len(wedge_or_arms_failures) == len(ran):
        findings.append({
            "rule": "camera_pose_clipping",
            "matched": True,
            "evidence": (
                "all executed variants show a large black edge wedge or missing arms; the issue "
                f"is camera/pose/clipping, not material texture (variants: {sorted(ran)})"
            ),
        })

    missing = int((material_resolution or {}).get("texture_reference_missing_count") or 0)
    if missing > 0:
        findings.append({
            "rule": "missing_texture_assets",
            "matched": True,
            "evidence": (
                f"{missing} robot texture reference(s) failed to resolve on the worker; the "
                "issue is asset packaging or worker asset resolution"
            ),
        })

    if clean("D") and clean("E") and (clean("B") is False or clean("C") is False):
        findings.append({
            "rule": "render_budget_sample_starvation",
            "matched": True,
            "evidence": (
                "high-sample textured renders (D/E) are clean while default-budget textured "
                "renders (B/C) are not; the issue is render budget/cold-start sampling, not "
                "texture correctness"
            ),
        })

    e_gate = gate("E") or {}
    e_regression = e_gate.get("denoiser_regression") or {}
    if clean("D") and (
        e_regression.get("darker_than_raw_beyond_tolerance")
        or e_regression.get("lower_structure_than_raw_beyond_tolerance")
        or clean("E") is False
    ):
        findings.append({
            "rule": "denoiser_path_failure",
            "matched": True,
            "evidence": (
                "the high-sample raw textured render (D) is clean but the denoised high-sample "
                "render (E) is dark/blotchy or lower-structure; the denoiser path is likely failing"
            ),
        })

    if clean("F") and clean("C") is False:
        findings.append({
            "rule": "pbr_specular_material_response",
            "matched": True,
            "evidence": (
                "simplified-diffuse (F) is clean while the full PBR textured variant (C) is not "
                "at the same budget/denoiser; likely metallic/specular map response under current "
                "lighting/sample budget"
            ),
        })

    if clean("G") and clean("C") is False:
        findings.append({
            "rule": "lighting_underexposure",
            "matched": True,
            "evidence": (
                "brighter task lighting (G) is clean while default lighting (C) is not; the "
                "robot/fridge gap is underexposed at the default lighting"
            ),
        })

    proxy_gate = gate("A") or {}
    textured_failed = any(clean(vid) is False for vid in ("B", "C", "D", "E"))
    if proxy_gate.get("passed") and textured_failed:
        findings.append({
            "rule": "white_proxy_bounded_workaround_available",
            "matched": True,
            "evidence": (
                "the white proxy baseline (A) passes while textured variants fail; Blueprint may "
                "proceed with white_proxy as a bounded seed-frame workaround but must not claim "
                "textured robot visual fidelity"
            ),
        })
    if proxy_gate and not proxy_gate.get("passed"):
        findings.append({
            "rule": "proxy_baseline_failed",
            "matched": True,
            "evidence": (
                "the white proxy baseline (A) itself fails the seed-frame gates; fix scene/"
                "camera/stance issues before interpreting material variants"
            ),
        })

    priority = (
        "camera_pose_clipping",
        "missing_texture_assets",
        "proxy_baseline_failed",
        "denoiser_path_failure",
        "render_budget_sample_starvation",
        "pbr_specular_material_response",
        "lighting_underexposure",
        "white_proxy_bounded_workaround_available",
    )
    matched_rules = {f["rule"] for f in findings}
    primary = next((rule for rule in priority if rule in matched_rules), None)
    required_for_rules = ("A", "B", "C", "D", "E", "F", "G")
    missing_variants = [vid for vid in required_for_rules if vid not in stats_by_id]
    return {
        "findings": findings,
        "primary_diagnosis": primary or "inconclusive_needs_more_evidence",
        "missing_variants": missing_variants,
        "claim_boundary": AUDIT_CLAIM_BOUNDARY,
    }


# ============================ WAM seed-media contract ============================

def build_wam_seed_media_contract(
    *,
    robot_material_mode: str,
    seed_frame_visual_quality_status: str | None,
    noise_grade_value: str | None = None,
    visual_smoke_passed: bool | None = None,
) -> dict[str, Any]:
    """Whether a seed frame may feed WAM conditioning, and under which explicit label.

    - ``white_proxy`` / ``simplified_diffuse`` are allowed for short-term WAM plumbing only
      with the simplified-robot boundary recorded.
    - noisy textured frames are allowed only when visual smoke accepted them AND the noisy/
      textured status is recorded in the WAM input manifest.
    - ``textured_unverified`` may never be presented as textured material fidelity.
    """
    mode = str(robot_material_mode)
    status = str(seed_frame_visual_quality_status or "")
    blockers: list[str] = []
    if mode not in (
        ROBOT_MATERIAL_MODE_VERIFIED_TEXTURED,
        ROBOT_MATERIAL_MODE_TEXTURED_UNVERIFIED,
        ROBOT_MATERIAL_MODE_SIMPLIFIED_DIFFUSE,
        ROBOT_MATERIAL_MODE_WHITE_PROXY,
    ):
        blockers.append(f"unknown_robot_material_mode:{mode}")
    if status != "completed":
        blockers.append("seed_frame_visual_quality_status_not_completed")
    if mode == ROBOT_MATERIAL_MODE_TEXTURED_UNVERIFIED:
        blockers.append(TEXTURED_MATERIAL_UNVERIFIED_BLOCKER)
    noisy = str(noise_grade_value or "") == "noisy"
    if noisy:
        if visual_smoke_passed is not True:
            blockers.append("noisy_textured_seed_requires_visual_smoke_acceptance")
    allowed = not blockers and mode in WAM_ALLOWED_ROBOT_MATERIAL_MODES
    return {
        "schema_version": WAM_SEED_MEDIA_CONTRACT_SCHEMA_VERSION,
        "robot_material_mode": mode,
        "seed_frame_visual_quality_status": status or None,
        "noise_grade": noise_grade_value,
        "noisy_textured_seed": noisy,
        "visual_smoke_passed": visual_smoke_passed,
        "wam_conditioning_allowed": allowed,
        "simplified_robot_visual_proxy": mode in (
            ROBOT_MATERIAL_MODE_SIMPLIFIED_DIFFUSE,
            ROBOT_MATERIAL_MODE_WHITE_PROXY,
        ),
        "textured_robot_visual_fidelity_claimed": mode == ROBOT_MATERIAL_MODE_VERIFIED_TEXTURED,
        "blockers": blockers,
        "claim_boundary": (
            "Seed-media material labeling contract only: it bounds what the WAM input manifest "
            "may claim about robot appearance. It is not task success, policy quality, or rank "
            "fidelity evidence."
        ),
    }


# ============================ contact sheet ============================

def write_contact_sheet(
    entries: Sequence[Mapping[str, Any]], out_path: Path,
    *, thumb_width: int = 320, thumb_height: int = 220, columns: int = 4,
) -> dict[str, Any]:
    """Side-by-side variant comparison PNG: one labeled thumbnail per audit variant."""
    from PIL import Image, ImageDraw

    drawable = [e for e in entries if e.get("png_path") and Path(str(e["png_path"])).is_file()]
    if not drawable:
        return {"status": "blocked", "blockers": ["no_variant_frames_for_contact_sheet"]}
    cols = max(1, min(columns, len(drawable)))
    rows = int(math.ceil(len(drawable) / float(cols)))
    label_h = 34
    cell_w, cell_h = thumb_width, thumb_height + label_h
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), (235, 238, 241))
    draw = ImageDraw.Draw(sheet)
    for index, entry in enumerate(drawable):
        col, row = index % cols, index // cols
        x0, y0 = col * cell_w, row * cell_h
        draw.rectangle([x0, y0, x0 + cell_w - 1, y0 + label_h - 1], fill=(28, 34, 42))
        draw.text((x0 + 6, y0 + 4), str(entry.get("label") or ""), fill=(240, 240, 240))
        sub = str(entry.get("sublabel") or "")
        if sub:
            draw.text((x0 + 6, y0 + 18), sub, fill=(170, 200, 230))
        try:
            with Image.open(str(entry["png_path"])) as img:
                thumb = img.convert("RGB")
                thumb.thumbnail((thumb_width, thumb_height))
                sheet.paste(thumb, (x0 + (cell_w - thumb.width) // 2,
                                    y0 + label_h + (thumb_height - thumb.height) // 2))
        except Exception:  # noqa: BLE001 - a bad PNG must not sink the whole sheet
            draw.text((x0 + 6, y0 + label_h + 8), "unreadable frame", fill=(120, 20, 20))
    ensure_dir(out_path.parent)
    sheet.save(out_path)
    return {
        "status": "completed",
        "path": str(out_path),
        "frame_count": len(drawable),
        "width": sheet.width,
        "height": sheet.height,
    }


# ============================ run analysis ============================

REQUIRED_RECORDED_INPUT_KEYS = (
    "task",
    "target_resolution",
    "stance_plan",
    "placement_validation",
    "camera_contract",
    "robot_asset",
    "render_settings",
    "lighting_summary",
    "runtime_metadata",
)


def _find_audit_dir(run_dir: Path) -> Path | None:
    candidates = [run_dir, run_dir / AUDIT_SUBDIR_NAME]
    candidates.extend(sorted(run_dir.glob(f"*/{AUDIT_SUBDIR_NAME}")))
    for candidate in candidates:
        if (candidate / WORKER_RUN_MANIFEST_NAME).is_file():
            return candidate
    return None


def _read_json_file(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    return data if isinstance(data, dict) else None


def analyze_render_noise_audit_run(
    run_dir: str | Path,
    out_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Compute frame stats, gates, interpretation, and the final audit manifest for one
    collected audit run (worker output tree). Pure local re-analysis: no GPU required."""
    run_dir = Path(run_dir)
    out_root = Path(out_dir) if out_dir is not None else None
    manifest: dict[str, Any] = {
        "schema_version": AUDIT_MANIFEST_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "status": "blocked",
        "run_dir": str(run_dir),
        "blockers": [],
        "claim_boundary": AUDIT_CLAIM_BOUNDARY,
    }
    audit_dir = _find_audit_dir(run_dir)
    if audit_dir is None:
        manifest["blockers"].append("audit_worker_run_manifest_missing")
        if out_root is not None:
            write_json(out_root / AUDIT_MANIFEST_NAME, manifest)
        return manifest
    out_root = out_root or audit_dir
    worker = _read_json_file(audit_dir / WORKER_RUN_MANIFEST_NAME) or {}
    material_resolution = _read_json_file(audit_dir / MATERIAL_RESOLUTION_MANIFEST_NAME)
    render_settings = _read_json_file(audit_dir / RENDER_SETTINGS_MANIFEST_NAME)
    camera_contract = _read_json_file(audit_dir / CAMERA_CONTRACT_NAME)

    plan_payload = worker.get("variant_plan") if isinstance(worker.get("variant_plan"), Mapping) else None
    variants = parse_variant_plan(plan_payload) if plan_payload else list(build_default_variant_matrix())
    by_id = {v.variant_id: v for v in variants}

    visibility = worker.get("arm_visibility") if isinstance(worker.get("arm_visibility"), Mapping) else None

    stats_by_id: dict[str, dict[str, Any]] = {}
    variant_records: list[dict[str, Any]] = []
    for record in worker.get("variant_results") or []:
        if not isinstance(record, Mapping):
            continue
        vid = str(record.get("variant_id") or "")
        variant = by_id.get(vid)
        if variant is None:
            manifest["blockers"].append(f"variant_result_without_plan_entry:{vid}")
            continue
        png_rel = str(record.get("frame_png") or f"{VARIANTS_SUBDIR_NAME}/{vid}/{VARIANT_FRAME_NAME}")
        png_path = (audit_dir / png_rel) if not Path(png_rel).is_absolute() else Path(png_rel)
        row: dict[str, Any] = {
            "variant_id": vid,
            "variant": variant.to_dict(),
            "worker_record": dict(record),
            "frame_png": str(png_path),
        }
        if png_path.is_file():
            try:
                row["frame_stats"] = compute_frame_stats(png_path)
                stats_by_id[vid] = row["frame_stats"]
            except Exception as exc:  # noqa: BLE001
                row["frame_stats_error"] = repr(exc)
                manifest["blockers"].append(f"variant_frame_stats_failed:{vid}")
        else:
            row["frame_stats"] = None
            manifest["blockers"].append(f"variant_frame_missing:{vid}")
        variant_records.append(row)

    proxy_stats = next(
        (stats_by_id[v.variant_id] for v in variants
         if v.robot_material == VARIANT_MATERIAL_WHITE_PROXY and v.variant_id in stats_by_id),
        None,
    )

    def _raw_reference(variant: RenderNoiseAuditVariant) -> Mapping[str, Any] | None:
        # Denoiser regression compares against the raw textured variant at the SAME budget.
        for other in variants:
            if (
                other.robot_material == VARIANT_MATERIAL_TEXTURED_ORIGINAL
                and not other.denoiser_enabled
                and other.render_budget == variant.render_budget
                and not other.lighting_boost
            ):
                return stats_by_id.get(other.variant_id)
        return None

    gates_by_id: dict[str, dict[str, Any]] = {}
    for row in variant_records:
        vid = row["variant_id"]
        if vid not in stats_by_id:
            continue
        variant = by_id[vid]
        gates_by_id[vid] = evaluate_variant_gates(
            variant=variant,
            stats=stats_by_id[vid],
            proxy_stats=proxy_stats,
            raw_textured_stats=_raw_reference(variant),
            visibility=visibility,
            material_resolution=material_resolution,
        )
        row["gates"] = gates_by_id[vid]

    interpretation = interpret_audit(
        stats_by_id=stats_by_id,
        gates_by_id=gates_by_id,
        material_resolution=material_resolution,
    )

    recorded_inputs = {
        "task": worker.get("task"),
        "target_resolution": worker.get("target_resolution"),
        "stance_plan": worker.get("stance_plan_summary") or worker.get("stance_plan"),
        "placement_validation": worker.get("placement_validation"),
        "camera_contract": camera_contract,
        "robot_asset": worker.get("robot_asset"),
        "render_settings": render_settings,
        "lighting_summary": (render_settings or {}).get("lighting_summary") or worker.get("lighting_summary"),
        "runtime_metadata": (render_settings or {}).get("runtime_metadata") or worker.get("runtime_metadata"),
    }
    missing_inputs = [key for key in REQUIRED_RECORDED_INPUT_KEYS if not recorded_inputs.get(key)]
    if missing_inputs:
        manifest["blockers"].extend(f"required_recorded_input_missing:{key}" for key in missing_inputs)

    frame_stats_rows = [
        {"variant_id": row["variant_id"], "frame_stats": row.get("frame_stats")}
        for row in variant_records
    ]
    write_json(out_root / FRAME_STATS_NAME, {
        "schema_version": FRAME_STATS_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "frames": frame_stats_rows,
    })

    sheet_entries = []
    for row in variant_records:
        stats = row.get("frame_stats") or {}
        sublabel = ""
        if stats:
            sublabel = (
                f"luma={stats.get('mean_luma')} noise={stats.get('high_frequency_noise_estimate')} "
                f"edges={stats.get('edge_density')}"
            )
        sheet_entries.append({
            "png_path": row.get("frame_png"),
            "label": f"{row['variant_id']}: {by_id[row['variant_id']].label}",
            "sublabel": sublabel,
        })
    contact_sheet = write_contact_sheet(sheet_entries, out_root / CONTACT_SHEET_NAME)

    executed = sorted(stats_by_id)
    manifest.update({
        "audit_dir": str(audit_dir),
        "variant_plan": plan_payload or build_variant_plan(variants),
        "variants_executed": executed,
        "variant_results": variant_records,
        "arm_visibility": visibility,
        "gates": gates_by_id,
        "interpretation": interpretation,
        "material_resolution": material_resolution,
        "render_settings": render_settings,
        "camera_contract": camera_contract,
        "recorded_inputs": recorded_inputs,
        "contact_sheet": contact_sheet,
        "frame_stats_path": str(out_root / FRAME_STATS_NAME),
        "worker_run_manifest_path": str(audit_dir / WORKER_RUN_MANIFEST_NAME),
    })
    if material_resolution is None:
        manifest["blockers"].append("robot_material_resolution_manifest_missing")
    if not executed:
        manifest["blockers"].append("no_variant_frames_analyzed")
    # Completion is honest only when every planned variant produced an analyzable frame;
    # partial runs still get stats/gates/interpretation but stay blocked.
    manifest["status"] = "completed" if not manifest["blockers"] else "blocked"
    write_json(out_root / AUDIT_MANIFEST_NAME, manifest)
    return manifest


# ============================ CLI ============================

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="G1 textured-robot render-noise audit: variant plan + local analysis",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    plan_parser = sub.add_parser("plan", help="write the default audit variant plan JSON")
    plan_parser.add_argument("--out", required=True)
    analyze_parser = sub.add_parser(
        "analyze", help="analyze a collected audit run dir (frame stats + gates + manifest)",
    )
    analyze_parser.add_argument("--run-dir", required=True)
    analyze_parser.add_argument("--out-dir", default=None)
    args = parser.parse_args(argv)
    if args.command == "plan":
        plan = build_variant_plan()
        write_json(Path(args.out), plan)
        print(f"[g1-render-noise-audit] plan={args.out} variants={len(plan['variants'])}")
        return 0
    manifest = analyze_render_noise_audit_run(args.run_dir, out_dir=args.out_dir)
    print(f"[g1-render-noise-audit] status={manifest['status']}")
    print(f"[g1-render-noise-audit] primary={((manifest.get('interpretation') or {}).get('primary_diagnosis'))}")
    out_root = Path(args.out_dir) if args.out_dir else Path(manifest.get("audit_dir") or args.run_dir)
    print(f"[g1-render-noise-audit] manifest={out_root / AUDIT_MANIFEST_NAME}")
    return 0 if manifest["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
