"""Bootstrap InteriorGS-style sidecars from a bare 3DGS PLY — no labels required.

When a splat arrives with NO ``labels.json`` / ``structure.json`` (any capture
that isn't an InteriorGS export), the placement lane still needs an object
catalog, wall geometry, and candidate tasks. This module generates all three
from the splat alone, on CPU plus a vision-language call:

1.  Decode the compressed PLY (cached, canonical ``splat-transform`` CLI).
2.  Estimate the floor plane from the splat mass (lowest prominent z-mode).
3.  Plan INTERIOR cameras (a look-outward star + pull-back diagonals — orbit
    rings sit outside the walls and see nothing for indoor scans).
4.  Render the views with the local headless Spark harness (no GPU).
5.  Detect objects per view — default detector is Gemini 2D boxes (the same
    google-genai integration the repo already uses); SAM3 slots in on GPU
    workers via the injectable ``detector``.
6.  Depth per view comes from the splat itself (:mod:`splat_depth` point
    z-buffer) — no DA3 needed when the geometry IS the reconstruction.
7.  Unproject + multi-view fuse (``min_views >= 2`` kills single-view
    hallucinations) into world AABBs.
8.  Write ``labels.bootstrap.json`` (InteriorGS labels schema — the existing
    :class:`InteriorGSSceneSpatialIndex` loads it unchanged) and
    ``task_targets.bootstrap.json`` (pick-up / open-close prompts per object).

Wall geometry and room connectivity for the probe come from splat occupancy at
runtime (:func:`splat_occupancy.wall_boxes_from_splat` /
``FloorOccupancyGrid.region_of_fn``) rather than a synthesized structure.json —
occupancy is the honest source; polygonizing it would only add error.

Truth boundary: generated labels are VLM detections lifted through
reconstruction depth — approximate boxes with detector-grade labels, clearly
marked ``bootstrap_generated``. They gate placement previews, not capture truth.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np

from .common import ensure_dir, utc_now_iso, write_json
from .gaussian_splat_decode import (
    SplatData,
    convert_to_standard_ply,
    read_standard_3dgs_ply,
)
from .scene_placement import SceneObject
from .scene_placement.perception_fusion import MultiViewPerceptionSceneSpatialIndex
from .scene_placement.stance_cameras import to_splat_render_specs
from .scene_placement.target_resolver import (
    _canonical_group_for_token,
    is_openable_target,
)
from .splat_depth import depth_provider_for_camera

BOOTSTRAP_SCHEMA_VERSION = "splat_scene_bootstrap.v1"
RENDER_HARNESS_REL = "tools/splat_render/render_splat.mjs"

# Detector contract: (png_path, camera) -> [{label, bbox_px:(x0,y0,x1,y1), confidence}]
DetectFn = Callable[[Path, Mapping[str, object]], List[dict]]

DEFAULT_MODEL_CASCADE = (
    "gemini-3-flash-preview",
    "gemini-2.5-flash",
)

_DETECT_PROMPT = (
    "You are labeling objects in an indoor scene for robot manipulation planning. "
    "Detect EVERY distinct physical object visible in this image: furniture "
    "(table, chair, bed, cabinet, sofa), appliances (fridge, microwave, stove, "
    "kettle, rice cooker), fixtures (sink, faucet, door, window, toilet), and "
    "manipulable items (pot, bowl, cup, vase, lamp, trash can). Skip walls, "
    "floors, ceilings, and reflections.\n\n"
    'Respond with STRICT JSON only: a list like '
    '[{"label": "<lowercase singular noun>", "box_2d": [ymin, xmin, ymax, xmax], '
    '"confidence": <0..1>}] where box_2d coordinates are integers normalized to '
    "0-1000 (Gemini spatial convention)."
)

# Objects bigger than this on any axis are furniture/structure, not pick-up-able.
_PICKUP_MAX_EXTENT_M = 0.6
_PICKUP_MAX_Z_M = 1.6
# Detections below this confidence are dropped before fusion — VLM detectors
# hedge with low-confidence guesses that only pollute the catalog.
DEFAULT_MIN_DETECTION_CONFIDENCE = 0.65


def canonicalize_detection_label(label: str) -> str:
    """Map a free-form VLM label onto its canonical fixture group when known.

    The same physical object gets different names from different viewpoints
    ("desk" vs "table", "shelf" vs "cabinet"); multi-view fusion clusters by
    label, so without canonicalization cross-view matches never form. Tokens
    with a synonym group collapse to the group name; unknown labels pass
    through normalized (lowercase, underscores).
    """
    tokens = str(label or "").strip().lower().replace("_", " ").split()
    for token in tokens:
        group = _canonical_group_for_token(token)
        if group:
            return group
    return "_".join(tokens)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


# ----------------------------- floor + camera planning -----------------------------

def estimate_floor_z_from_points(
    z_values: np.ndarray,
    *,
    bin_m: float = 0.05,
    band_m: float = 2.0,
    prominence: float = 0.35,
) -> float:
    """Lowest prominent mode of splat-center z — same estimator as chunk bounds.

    Under-floor reconstruction fuzz makes minima/percentiles land too low; the
    floor is where the mass piles up (see
    :meth:`gaussian_splat_decode.CompressedSplatChunkBounds.floor_z_estimate`).
    """
    z = np.asarray(z_values, dtype=np.float64)
    if z.size == 0:
        return 0.0
    lo = float(np.percentile(z, 0.5))
    band = z[(z >= lo) & (z <= lo + band_m)]
    if band.size == 0:
        return lo
    edges = np.arange(lo, lo + band_m + bin_m, bin_m)
    hist, edges = np.histogram(band, bins=edges)
    if hist.size == 0 or hist.max() == 0:
        return lo
    threshold = max(1.0, prominence * float(hist.max()))
    for i, count in enumerate(hist):
        if count >= threshold:
            return float(0.5 * (edges[i] + edges[i + 1]))
    return float(np.percentile(band, 5.0))


def plan_interior_views(
    splat: SplatData,
    *,
    floor_z: float,
    n_azimuths: int = 8,
    n_pullbacks: int = 4,
    eye_height: float = 1.4,
    vfov_deg: float = 65.0,
    width: int = 1024,
    height: int = 768,
    min_opacity: float = 0.3,
) -> List[Dict[str, object]]:
    """Interior camera star: look OUTWARD from the scene core, plus pull-backs.

    Orbit rings (the perception default) circle OUTSIDE the bounds — correct for
    a single object, useless for a room scan whose walls face inward. Instead:
    ``n_azimuths`` cameras stand near the robust scene center looking outward
    and slightly down (they sweep the perimeter furniture), and ``n_pullbacks``
    cameras stand off-center looking back ACROSS the center (they cover the
    core the star cameras stand on).
    """
    opacity = splat.opacity_sigmoid
    pts = splat.xyz[opacity >= float(min_opacity)]
    if pts.shape[0] == 0:
        raise ValueError("no opaque splat centers to plan views from")
    # Robust-bounds MIDPOINT over the floor footprint, not all splat mass:
    # furniture/appliance blobs can be much denser than the floor and drag even
    # percentile extents toward one corner. Floor-band points best describe the
    # interior footprint the cameras can stand in; fall back to all points only
    # when the reconstruction has no usable floor band.
    floor_band = pts[
        (pts[:, 2] >= floor_z - 0.05)
        & (pts[:, 2] <= floor_z + 0.15)
    ]
    footprint = floor_band if floor_band.shape[0] >= 16 else pts
    x_lo, x_hi = (float(v) for v in np.percentile(footprint[:, 0], (3, 97)))
    y_lo, y_hi = (float(v) for v in np.percentile(footprint[:, 1], (3, 97)))
    cx = 0.5 * (x_lo + x_hi)
    cy = 0.5 * (y_lo + y_hi)
    radius = max(0.5 * (x_hi - x_lo), 0.5 * (y_hi - y_lo), 1.0)
    eye_z = floor_z + float(eye_height)
    look_z = floor_z + 0.8
    cams: List[Dict[str, object]] = []
    for k in range(max(1, int(n_azimuths))):
        theta = 2.0 * math.pi * k / max(1, int(n_azimuths))
        target = (cx + 0.85 * radius * math.cos(theta), cy + 0.85 * radius * math.sin(theta), look_z)
        cams.append(
            {
                "id": f"star_{k:02d}",
                "eye": (cx, cy, eye_z),
                "target": target,
                "up": (0.0, 0.0, 1.0),
                "vfov": math.radians(vfov_deg),
                "width": int(width),
                "height": int(height),
            }
        )
    for k in range(max(0, int(n_pullbacks))):
        theta = 2.0 * math.pi * (k + 0.5) / max(1, int(n_pullbacks))
        ex = cx + 0.55 * radius * math.cos(theta)
        ey = cy + 0.55 * radius * math.sin(theta)
        target = (cx - 0.4 * radius * math.cos(theta), cy - 0.4 * radius * math.sin(theta), look_z)
        cams.append(
            {
                "id": f"pullback_{k:02d}",
                "eye": (ex, ey, eye_z),
                "target": target,
                "up": (0.0, 0.0, 1.0),
                "vfov": math.radians(vfov_deg),
                "width": int(width),
                "height": int(height),
            }
        )
    return cams


# ----------------------------- Gemini detector -----------------------------

def _extract_json_list(text: str) -> list:
    try:
        payload = json.loads(text)
    except Exception:  # noqa: BLE001
        import re

        match = re.search(r"\[.*\]", text or "", re.DOTALL)
        if not match:
            return []
        try:
            payload = json.loads(match.group(0))
        except Exception:  # noqa: BLE001
            return []
    if isinstance(payload, dict):  # some models wrap: {"objects": [...]}
        for value in payload.values():
            if isinstance(value, list):
                return value
        return []
    return payload if isinstance(payload, list) else []


def detections_from_gemini_boxes(
    records: Sequence[Mapping[str, Any]],
    *,
    width: int,
    height: int,
) -> List[dict]:
    """Convert Gemini ``box_2d`` records ([ymin,xmin,ymax,xmax], 0-1000) to bbox_px."""
    out: List[dict] = []
    for rec in records:
        if not isinstance(rec, Mapping):
            continue
        box = rec.get("box_2d") or rec.get("bbox") or rec.get("box")
        label = str(rec.get("label", "") or "").strip().lower()
        if not label or not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        try:
            ymin, xmin, ymax, xmax = (float(v) for v in box)
        except (TypeError, ValueError):
            continue
        x0, x1 = sorted((xmin / 1000.0 * width, xmax / 1000.0 * width))
        y0, y1 = sorted((ymin / 1000.0 * height, ymax / 1000.0 * height))
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        conf = rec.get("confidence", 0.8)
        try:
            conf = max(0.0, min(1.0, float(conf)))
        except (TypeError, ValueError):
            conf = 0.8
        out.append({"label": label, "bbox_px": (x0, y0, x1, y1), "confidence": conf})
    return out


def _gemini_detect_image(
    image_bytes: bytes,
    *,
    models: Sequence[str] = DEFAULT_MODEL_CASCADE,
) -> str:
    """Send one rendered view to Gemini, return raw JSON text (repo-standard call)."""
    api_key = (os.getenv("GOOGLE_GENAI_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("missing_GOOGLE_GENAI_API_KEY")
    from google import genai  # type: ignore

    client = genai.Client(api_key=api_key)
    image_part = genai.types.Part.from_bytes(data=image_bytes, mime_type="image/png")
    last_exc: Optional[Exception] = None
    for model in models:
        try:
            response = client.models.generate_content(
                model=model,
                contents=[image_part, _DETECT_PROMPT],
                config={"response_mime_type": "application/json"},  # no thinking_config (silent fail)
            )
            text = str(getattr(response, "text", "") or "").strip()
            if text:
                return text
        except Exception as exc:  # noqa: BLE001 - cascade to the next model
            last_exc = exc
            continue
    if last_exc is not None:
        raise last_exc
    return ""


def gemini_view_detector(png_path: Path, camera: Mapping[str, object]) -> List[dict]:
    """Gemini 2D boxes on one rendered view."""
    raw = _gemini_detect_image(Path(png_path).read_bytes())
    records = _extract_json_list(raw)
    return detections_from_gemini_boxes(
        records,
        width=int(camera["width"]),  # type: ignore[arg-type]
        height=int(camera["height"]),  # type: ignore[arg-type]
    )


def _openai_detect_image(image_bytes: bytes, *, model: str = "gpt-4o-mini") -> str:
    """OpenAI vision fallback detector (plain REST, no SDK dependency).

    Same prompt and the same normalized ``box_2d`` output contract as the Gemini
    call, so one parser serves both. Used when no working Gemini key is present.
    """
    import base64
    import urllib.request

    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("missing_OPENAI_API_KEY")
    payload = {
        "model": model,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,"
                            + base64.b64encode(image_bytes).decode("ascii")
                        },
                    },
                    {
                        "type": "text",
                        "text": _DETECT_PROMPT
                        + '\n\nWrap the list in an object: {"objects": [...]}.',
                    },
                ],
            }
        ],
    }
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return str(body["choices"][0]["message"]["content"] or "")


def openai_view_detector(png_path: Path, camera: Mapping[str, object]) -> List[dict]:
    """OpenAI-backed :data:`DetectFn` (same output contract as the Gemini one)."""
    raw = _openai_detect_image(Path(png_path).read_bytes())
    records = _extract_json_list(raw)
    return detections_from_gemini_boxes(
        records,
        width=int(camera["width"]),  # type: ignore[arg-type]
        height=int(camera["height"]),  # type: ignore[arg-type]
    )


def default_view_detector(png_path: Path, camera: Mapping[str, object]) -> List[dict]:
    """Default :data:`DetectFn`: Gemini first, OpenAI vision as the fallback.

    Tries Gemini only when a key is configured; any Gemini failure (revoked key,
    quota, transient) falls through to OpenAI when ITS key is configured. Raises
    the original error when no fallback is possible, so the per-view report
    shows the real cause instead of a generic miss.
    """
    has_gemini = bool(
        (os.getenv("GOOGLE_GENAI_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip()
    )
    has_openai = bool((os.getenv("OPENAI_API_KEY") or "").strip())
    if has_gemini:
        try:
            return gemini_view_detector(png_path, camera)
        except Exception:
            if not has_openai:
                raise
    if has_openai:
        return openai_view_detector(png_path, camera)
    raise RuntimeError("no_vlm_detector_key_available (GOOGLE_GENAI_API_KEY or OPENAI_API_KEY)")


# ----------------------------- render the views -----------------------------

def render_views_with_spark(
    splat_path: Path,
    cameras: Sequence[Mapping[str, object]],
    out_dir: Path,
    *,
    repo_root: Path | None = None,
    node: str = "node",
    timeout_seconds: int = 1200,
) -> dict:
    """Render the planned views with the local Spark harness; fail-closed."""
    root = repo_root or _repo_root()
    harness = root / RENDER_HARNESS_REL
    if not harness.is_file():
        return {"status": "blocked", "blockers": ["splat_render_harness_missing"]}
    ensure_dir(out_dir)
    by_id = {str(cam["id"]): cam for cam in cameras}
    specs = to_splat_render_specs({cid: cam for cid, cam in by_id.items()})
    cameras_path = out_dir / "bootstrap_cameras.json"
    cameras_path.write_text(json.dumps(specs, indent=2))
    width = int(cameras[0]["width"])  # type: ignore[arg-type]
    height = int(cameras[0]["height"])  # type: ignore[arg-type]
    cmd = [
        node, str(harness),
        "--splat", str(splat_path),
        "--out", str(out_dir),
        "--cameras", str(cameras_path),
        "--width", str(width),
        "--height", str(height),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_seconds)
    except FileNotFoundError:
        return {"status": "blocked", "blockers": ["node_runtime_unavailable"]}
    except subprocess.TimeoutExpired:
        return {"status": "blocked", "blockers": ["splat_render_timeout"]}
    manifest: dict = {}
    try:
        manifest = json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception:  # noqa: BLE001
        manifest = {}
    rendered = {
        str(entry.get("id")): str(entry.get("path"))
        for entry in manifest.get("cameras", [])
        if entry.get("path")
    }
    harness_ok = proc.returncode == 0 and manifest.get("status") in {"completed", "ok"}
    if not harness_ok:
        # The harness can die AFTER writing every frame (e.g. a browser-close
        # timeout). The PNGs on disk are the actual evidence: if every planned
        # view exists and is non-trivial, the render succeeded — record the
        # harness hiccup honestly instead of discarding 20+ minutes of output.
        on_disk = {
            str(cam["id"]): str(out_dir / f"{cam['id']}.png")
            for cam in cameras
            if (out_dir / f"{cam['id']}.png").is_file()
            and (out_dir / f"{cam['id']}.png").stat().st_size > 10_000
        }
        if len(on_disk) == len(cameras):
            return {
                "status": "completed",
                "rendered": on_disk,
                "note": "harness_exit_unclean_but_all_views_on_disk",
                "returncode": proc.returncode,
                "stderr_tail": (proc.stderr or "")[-800:],
            }
        return {
            "status": "blocked",
            "blockers": ["splat_render_failed"],
            "returncode": proc.returncode,
            "views_on_disk": len(on_disk),
            "views_planned": len(cameras),
            "stderr_tail": (proc.stderr or "")[-2000:],
        }
    return {"status": "completed", "rendered": rendered, "harness_manifest": manifest}


# ----------------------------- sidecar synthesis -----------------------------

def _aabb_corners(bbox_min, bbox_max) -> List[dict]:
    x0, y0, z0 = bbox_min
    x1, y1, z1 = bbox_max
    return [
        {"x": x, "y": y, "z": z}
        for z in (z0, z1)
        for x, y in ((x0, y0), (x0, y1), (x1, y1), (x1, y0))
    ]


def labels_payload_from_objects(objects: Sequence[SceneObject]) -> List[dict]:
    """Fused world AABBs -> the InteriorGS labels.json shape (loader-compatible)."""
    payload: List[dict] = []
    for i, obj in enumerate(objects, start=1):
        payload.append(
            {
                "ins_id": str(i),
                "label": obj.label or f"object_{i}",
                "bounding_box": _aabb_corners(obj.bbox_min, obj.bbox_max),
            }
        )
    return payload


def task_targets_payload_from_objects(
    objects: Sequence[SceneObject],
    *,
    floor_z: float,
    facility_name: str = "Bootstrapped splat scene",
) -> dict:
    """Candidate pick-up / open-close prompts per detected object."""
    tasks: List[dict] = [
        {"task_id": "pick_place_manipulation", "source": "splat_bootstrap", "scene_type": "indoor"},
        {"task_id": "open_close_access_points", "source": "splat_bootstrap", "scene_type": "indoor"},
    ]
    for i, obj in enumerate(objects, start=1):
        name = f"{obj.label}_{i}"
        dx, dy, dz = obj.size()
        if is_openable_target(obj):
            tasks.append(
                {"task_id": f"Open and close {name}", "source": "splat_bootstrap_prompt",
                 "scene_type": "indoor"}
            )
        elif (
            max(dx, dy, dz) <= _PICKUP_MAX_EXTENT_M
            and obj.centroid[2] <= floor_z + _PICKUP_MAX_Z_M
        ):
            tasks.append(
                {"task_id": f"Pick up {name} and place it in the target zone",
                 "source": "splat_bootstrap_prompt", "scene_type": "indoor"}
            )
    return {
        "bootstrap_generated": True,
        "generated_at": utc_now_iso(),
        "facility_name": facility_name,
        "source": "splat_bootstrap",
        "scene_type": "indoor",
        "tasks": tasks,
    }


# ----------------------------- orchestration -----------------------------

def bootstrap_scene_sidecars(
    splat_path: str | Path,
    out_dir: str | Path,
    *,
    detector: Optional[DetectFn] = None,
    renderer=None,
    repo_root: Path | None = None,
    node: str = "node",
    n_azimuths: int = 8,
    n_pullbacks: int = 4,
    width: int = 1024,
    height: int = 768,
    min_views: int = 1,
    merge_gap: float = 0.75,
    min_confidence: float = DEFAULT_MIN_DETECTION_CONFIDENCE,
    depth_scale: int = 4,
    render_timeout_seconds: int = 1200,
) -> dict:
    """PLY-only -> ``labels.bootstrap.json`` + ``task_targets.bootstrap.json``.

    Returns a fail-closed report; ``status == "completed"`` means both sidecars
    were written under ``out_dir``. World frame is assumed z-up (InteriorGS /
    Blueprint convention); up-axis normalization is a caller concern for
    exotic captures.
    """
    splat_path = Path(splat_path)
    out_path = Path(out_dir)
    ensure_dir(out_path)
    report: dict[str, Any] = {
        "schema_version": BOOTSTRAP_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "splat": str(splat_path),
        "steps": {},
        "truth_boundary": (
            "VLM detections lifted through reconstruction depth; approximate boxes "
            "for placement previews, not capture ground truth."
        ),
    }
    if not splat_path.is_file():
        report.update(status="blocked", blockers=["splat_source_missing"])
        return report

    # 1. decode (cached)
    cache_dir = out_path / "cache"
    ensure_dir(cache_dir)
    decoded = cache_dir / "decoded_standard_3dgs.ply"
    if not decoded.is_file():
        status = convert_to_standard_ply(splat_path, decoded, repo_root=repo_root or _repo_root(), node=node)
        if status.get("status") != "completed":
            # A standard PLY input decodes to itself: try reading directly.
            try:
                read_standard_3dgs_ply(splat_path)
                decoded = splat_path
            except Exception:
                report["steps"]["decode"] = status
                report.update(status="blocked", blockers=["splat_decode_failed"])
                return report
    try:
        splat = read_standard_3dgs_ply(decoded)
    except Exception as exc:  # noqa: BLE001
        report["steps"]["decode"] = {"status": "blocked", "error": str(exc)}
        report.update(status="blocked", blockers=["decoded_ply_unreadable"])
        return report
    report["steps"]["decode"] = {"status": "completed", "splat_count": splat.count}

    # 2. floor
    opaque_z = splat.xyz[splat.opacity_sigmoid >= 0.3][:, 2]
    floor_z = estimate_floor_z_from_points(opaque_z)
    report["floor_z"] = round(float(floor_z), 4)

    # 3. cameras
    try:
        cameras = plan_interior_views(
            splat, floor_z=floor_z, n_azimuths=n_azimuths, n_pullbacks=n_pullbacks,
            width=width, height=height,
        )
    except ValueError as exc:
        report.update(status="blocked", blockers=[str(exc)])
        return report
    report["steps"]["cameras"] = {"status": "completed", "count": len(cameras)}

    # 4. render (injectable for tests / alternative renderers)
    render_fn = renderer or (
        lambda sp, cams, out: render_views_with_spark(
            sp, cams, out,
            repo_root=repo_root, node=node, timeout_seconds=render_timeout_seconds,
        )
    )
    render = render_fn(splat_path, cameras, out_path / "views")
    report["steps"]["render"] = {k: v for k, v in render.items() if k != "harness_manifest"}
    if render.get("status") != "completed":
        report.update(status="blocked", blockers=render.get("blockers", ["splat_render_failed"]))
        return report
    rendered = render["rendered"]

    # 5+6. detect + depth per view
    detect = detector or default_view_detector
    views: List[dict] = []
    detect_stats: List[dict] = []
    for cam in cameras:
        cam_id = str(cam["id"])
        png = rendered.get(cam_id)
        if not png:
            detect_stats.append({"view": cam_id, "status": "no_render"})
            continue
        try:
            detections = detect(Path(png), cam)
        except Exception as exc:  # noqa: BLE001 - a failed view degrades coverage, not the run
            detect_stats.append({"view": cam_id, "status": "detector_error", "error": str(exc)[:200]})
            continue
        # Canonicalize labels (desk->table, shelf->cabinet, ...) so cross-view
        # fusion can cluster, and drop low-confidence hedge guesses.
        detections = [
            {**det, "label": canonicalize_detection_label(str(det.get("label", "")))}
            for det in detections
            if float(det.get("confidence", 1.0) or 0.0) >= float(min_confidence)
        ]
        provider = depth_provider_for_camera(splat, cam, depth_scale=depth_scale)
        views.append({"detections": detections, "depth_provider": provider, "camera": cam})
        detect_stats.append({"view": cam_id, "status": "ok", "detections": len(detections)})
    report["steps"]["detect"] = {"status": "completed", "views": detect_stats}
    if not views:
        report.update(status="blocked", blockers=["no_views_with_detections"])
        return report

    # 7. fuse
    index = MultiViewPerceptionSceneSpatialIndex(
        views, min_views=min_views, merge_gap=merge_gap
    )
    objects = index.objects()
    report["steps"]["fuse"] = {
        "status": "completed",
        "fused_objects": len(objects),
        "min_views": min_views,
        "merge_gap": merge_gap,
        "min_confidence": min_confidence,
    }
    if not objects:
        report.update(status="blocked", blockers=["no_fused_objects"])
        return report

    # 8. write sidecars
    labels_path = out_path / "labels.bootstrap.json"
    labels_path.write_text(json.dumps(labels_payload_from_objects(objects), indent=2))
    tasks_path = out_path / "task_targets.bootstrap.json"
    write_json(tasks_path, task_targets_payload_from_objects(objects, floor_z=floor_z))
    report.update(
        status="completed",
        labels_path=str(labels_path),
        task_targets_path=str(tasks_path),
        decoded_ply=str(decoded),
        object_count=len(objects),
        labels=[{"ins_id": str(i), "label": o.label} for i, o in enumerate(objects, start=1)],
    )
    write_json(out_path / "bootstrap_report.json", report)
    return report


__all__ = [
    "BOOTSTRAP_SCHEMA_VERSION",
    "DetectFn",
    "bootstrap_scene_sidecars",
    "detections_from_gemini_boxes",
    "estimate_floor_z_from_points",
    "default_view_detector",
    "gemini_view_detector",
    "openai_view_detector",
    "labels_payload_from_objects",
    "plan_interior_views",
    "render_views_with_spark",
    "task_targets_payload_from_objects",
]
