"""Render a Gaussian-splat capture into per-camera images + an MP4 of the real scene.

This is the local-first, reproducible "the splat actually displays" path that the Isaac
lane uses to reach parity with (and exceed) MuJoCo's box proxies. It:

1. decodes the compressed PlayCanvas PLY (or SPZ) to a full-resolution standard 3DGS PLY via
   the canonical ``splat-transform`` CLI (:mod:`gaussian_splat_decode`),
2. analyzes the splat geometry (up-axis, floor, footprint, free-floor start pose) via
   :mod:`splat_scene_analysis`,
3. frames the eval's 6 named cameras to the real scene and renders them headlessly with
   Spark.js (``tools/splat_render/render_splat.mjs``), and
4. encodes the frames into a slideshow MP4 with ffmpeg.

Truth boundary: the renderer is the reference Spark (three.js) Gaussian renderer, NOT
Isaac RTX. Every result is labeled ``rendered_by: reference_spark_renderer``; this proves
the captured scene displays, not that Isaac itself rendered it (that is the Phase-2 NuRec
GPU proof). All failures are fail-closed blockers, never a fabricated pass.
"""
from __future__ import annotations

import json
import hashlib
import math
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .gaussian_splat_decode import find_splat_transform_cli, read_standard_3dgs_ply
from .decision_evidence_contracts import canonical_digest
from .scene_placement.perception_views import view_ring_for_bounds
from .scene_placement.stance_cameras import to_splat_render_specs
from .splat_scene_analysis import DEFAULT_CAMERA_IDS, analyze_scene, derive_eval_cameras

RENDER_HARNESS_REL = "tools/splat_render/render_splat.mjs"
RENDERED_BY = "reference_spark_renderer"
SCHEMA_VERSION = "splat_scene_render.v1"
_SPLAT_SUFFIXES = {".ply", ".spz", ".splat", ".ksplat"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _normalize_camera_specs(
    camera_specs: Sequence[Mapping[str, Any]] | None,
) -> tuple[list[dict[str, Any]] | None, list[str]]:
    """Validate native Spark camera rows before decoding or rendering.

    The room-survey planner and the renderer intentionally share this small
    scene-neutral contract: ``[{id, spec:{pos,target,fov,up}}]``.  A malformed
    or duplicate camera set fails closed before any expensive source decode.
    """

    if camera_specs is None:
        return None, []
    if not camera_specs:
        return None, ["camera_specs_missing"]
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    errors: list[str] = []
    for row in camera_specs:
        if not isinstance(row, Mapping):
            errors.append("camera_spec_row_invalid")
            continue
        camera_id = str(row.get("id") or "")
        spec = row.get("spec")
        if (
            not camera_id
            or camera_id in seen
            or "/" in camera_id
            or ".." in camera_id
            or not isinstance(spec, Mapping)
        ):
            errors.append("camera_spec_id_or_payload_invalid")
            continue
        seen.add(camera_id)
        try:
            pos = [float(value) for value in spec["pos"]]
            target = [float(value) for value in spec["target"]]
            up = [float(value) for value in spec.get("up", [0.0, 1.0, 0.0])]
            fov = float(spec["fov"])
        except (KeyError, TypeError, ValueError):
            errors.append(f"camera_spec_geometry_invalid:{camera_id}")
            continue
        values = [*pos, *target, *up, fov]
        if (
            len(pos) != 3
            or len(target) != 3
            or len(up) != 3
            or not all(math.isfinite(value) for value in values)
            or not 0.0 < fov < 180.0
            or math.dist(pos, target) <= 1e-9
            or math.sqrt(sum(value * value for value in up)) <= 1e-9
        ):
            errors.append(f"camera_spec_geometry_invalid:{camera_id}")
            continue
        normalized.append(
            {
                "id": camera_id,
                "spec": {"pos": pos, "target": target, "fov": fov, "up": up},
            }
        )
    return (normalized if not errors else None), sorted(set(errors))


def _derive_bounds_camera_specs(
    focus_bounds: Sequence[Sequence[float]],
    *,
    margin: float,
    n_azimuths: int,
    elevations_deg: Sequence[float],
    vfov_deg: float,
    width: int,
    height: int,
    camera_id_prefix: str,
) -> tuple[list[dict[str, Any]] | None, dict[str, Any] | None, list[str]]:
    """Derive a deterministic close-up ring from observed metric bounds.

    This is the task-neutral reconnaissance seam for moving from a completed
    whole-topology survey to target close-ups.  Bounds and every ring parameter
    are retained in the render receipt; callers never need to hand-author a
    scene-specific camera JSON file.
    """

    errors: list[str] = []
    try:
        if len(focus_bounds) != 2:
            raise ValueError
        bbox_min = tuple(float(value) for value in focus_bounds[0])
        bbox_max = tuple(float(value) for value in focus_bounds[1])
    except (TypeError, ValueError, IndexError):
        return None, None, ["focus_bounds_invalid"]
    values = [*bbox_min, *bbox_max]
    if (
        len(bbox_min) != 3
        or len(bbox_max) != 3
        or not all(math.isfinite(value) for value in values)
        or any(bbox_max[index] <= bbox_min[index] for index in range(3))
    ):
        errors.append("focus_bounds_invalid")
    try:
        normalized_margin = float(margin)
        normalized_azimuths = int(n_azimuths)
        normalized_elevations = tuple(float(value) for value in elevations_deg)
        normalized_vfov = float(vfov_deg)
    except (TypeError, ValueError):
        errors.append("bounds_view_ring_parameters_invalid")
        normalized_margin = 0.0
        normalized_azimuths = 0
        normalized_elevations = ()
        normalized_vfov = 0.0
    prefix = str(camera_id_prefix or "")
    if (
        not prefix
        or "/" in prefix
        or ".." in prefix
        or not math.isfinite(normalized_margin)
        or normalized_margin <= 1.0
        or normalized_azimuths < 2
        or not normalized_elevations
        or not all(math.isfinite(value) and -89.0 < value < 89.0 for value in normalized_elevations)
        or not math.isfinite(normalized_vfov)
        or not 1.0 <= normalized_vfov < 179.0
        or isinstance(width, bool)
        or isinstance(height, bool)
        or int(width) <= 0
        or int(height) <= 0
    ):
        errors.append("bounds_view_ring_parameters_invalid")
    if errors:
        return None, None, sorted(set(errors))

    cameras = view_ring_for_bounds(
        bbox_min,  # type: ignore[arg-type]
        bbox_max,  # type: ignore[arg-type]
        margin=normalized_margin,
        n_azimuths=normalized_azimuths,
        elevations_deg=normalized_elevations,
        vfov_deg=normalized_vfov,
        width=int(width),
        height=int(height),
    )
    named = {
        f"{prefix}_e{elevation_index:02d}_a{azimuth_index:02d}": cameras[
            elevation_index * normalized_azimuths + azimuth_index
        ]
        for elevation_index in range(len(normalized_elevations))
        for azimuth_index in range(normalized_azimuths)
    }
    specs = to_splat_render_specs(named)
    plan = {
        "planner": "scene_placement.perception_views.view_ring_for_bounds",
        "focus_bounds": {
            "min": list(bbox_min),
            "max": list(bbox_max),
        },
        "margin": normalized_margin,
        "n_azimuths": normalized_azimuths,
        "elevations_deg": list(normalized_elevations),
        "vfov_deg": normalized_vfov,
        "width": int(width),
        "height": int(height),
        "camera_id_prefix": prefix,
        "camera_count": len(specs),
        "claim_boundary": "reconnaissance_camera_plan_not_source_observation_recovery",
    }
    return specs, plan, []


def _derive_multi_bounds_camera_specs(
    requests: Sequence[Mapping[str, Any]],
    *,
    width: int,
    height: int,
    max_total_cameras: int | None = None,
) -> tuple[list[dict[str, Any]] | None, dict[str, Any] | None, list[str]]:
    """Build one render camera set for several independently bounded entities."""

    if not requests:
        return None, None, ["focus_bounds_requests_missing"]
    if (
        max_total_cameras is not None
        and (
            isinstance(max_total_cameras, bool)
            or not isinstance(max_total_cameras, int)
            or max_total_cameras < 1
        )
    ):
        return None, None, ["focus_bounds_camera_budget_invalid"]
    all_specs: list[dict[str, Any]] = []
    plans: list[dict[str, Any]] = []
    errors: list[str] = []
    target_ids: set[str] = set()
    camera_ids: set[str] = set()
    for index, request in enumerate(requests):
        if not isinstance(request, Mapping):
            errors.append(f"focus_bounds_request_invalid:{index}")
            continue
        target_id = str(request.get("target_id") or "")
        if (
            not target_id
            or target_id in target_ids
            or "/" in target_id
            or ".." in target_id
        ):
            errors.append(f"focus_bounds_request_target_id_invalid:{index}")
            continue
        target_ids.add(target_id)
        bounds = request.get("focus_bounds")
        if not isinstance(bounds, Sequence) or isinstance(bounds, (str, bytes)):
            errors.append(f"focus_bounds_request_invalid:{target_id}")
            continue
        specs, plan, request_errors = _derive_bounds_camera_specs(
            bounds,
            margin=request.get("margin", 3.0),
            n_azimuths=request.get("n_azimuths", 8),
            elevations_deg=request.get("elevations_deg", (10.0, 35.0, 60.0)),
            vfov_deg=request.get("vfov_deg", 48.0),
            width=width,
            height=height,
            camera_id_prefix=str(request.get("camera_id_prefix") or target_id),
        )
        if request_errors:
            errors.extend(f"{code}:{target_id}" for code in request_errors)
            continue
        assert specs is not None and plan is not None
        duplicate_camera_ids = [row["id"] for row in specs if row["id"] in camera_ids]
        if duplicate_camera_ids:
            errors.append(f"focus_bounds_request_camera_ids_duplicate:{target_id}")
            continue
        camera_ids.update(str(row["id"]) for row in specs)
        plan["target_id"] = target_id
        plan["semantic_role"] = str(request.get("semantic_role") or "inspection_target")
        all_specs.extend(specs)
        plans.append(plan)
    if errors:
        return None, None, sorted(set(errors))
    if max_total_cameras is not None and len(all_specs) > max_total_cameras:
        return None, None, ["focus_bounds_camera_budget_exceeded"]
    return (
        all_specs,
        {
            "planner": "multi_entity_bounds_view_ring",
            "target_count": len(plans),
            "camera_count": len(all_specs),
            "max_total_cameras": max_total_cameras,
            "targets": plans,
            "claim_boundary": "reconnaissance_camera_plan_not_source_observation_recovery",
        },
        [],
    )


def _look_at_camera_calibration(camera: Mapping[str, Any], *, width: int, height: int) -> dict:
    spec = camera["spec"]
    vfov_deg = float(spec["fov"])
    focal_pixels = 0.5 * float(height) / math.tan(math.radians(vfov_deg) / 2.0)
    return {
        "id": str(camera["id"]),
        "pose_convention": "world_position_target_up_look_at_z_up",
        "position_world_m": [float(value) for value in spec["pos"]],
        "target_world_m": [float(value) for value in spec["target"]],
        "up_world": [float(value) for value in spec["up"]],
        "intrinsics": {
            "model": "pinhole_centered_square_pixels",
            "fx": focal_pixels,
            "fy": focal_pixels,
            "cx": float(width) / 2.0,
            "cy": float(height) / 2.0,
            "width": int(width),
            "height": int(height),
            "vertical_fov_deg": vfov_deg,
        },
    }


def _renderer_dependency_identity(root: Path) -> dict[str, Any]:
    lock_path = root / "tools/splat_render/package-lock.json"
    versions: dict[str, str] = {}
    if lock_path.is_file():
        try:
            lock = json.loads(lock_path.read_text(encoding="utf-8"))
            packages = lock.get("packages", {})
            for package_name in ("playwright", "playwright-core", "three", "@sparkjsdev/spark"):
                row = packages.get(f"node_modules/{package_name}")
                if isinstance(row, Mapping) and row.get("version"):
                    versions[package_name] = str(row["version"])
        except (OSError, ValueError, TypeError):
            versions = {}
    browser_registry = root / "tools/splat_render/node_modules/playwright-core/browsers.json"
    try:
        node_version = subprocess.run(
            ["node", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        ).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        node_version = "unavailable"
    return {
        "node_version": node_version,
        "package_lock_path": str(lock_path) if lock_path.is_file() else None,
        "package_lock_sha256": _sha256_file(lock_path) if lock_path.is_file() else None,
        "dependency_versions": versions,
        "playwright_browser_registry_sha256": (
            _sha256_file(browser_registry) if browser_registry.is_file() else None
        ),
    }


def _repo_root(repo_root: str | Path | None) -> Path:
    return Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]


def _decimate_to_standard_ply(
    source: Path, dst: Path, decimate: int, *, repo_root: Path, node: str, timeout: int
) -> dict:
    """One splat-transform call: decode + optional decimate -> standard 3DGS PLY."""
    if source.suffix.lower() == ".ply" and decimate <= 0:
        try:
            splat = read_standard_3dgs_ply(source)
        except (OSError, ValueError):
            pass
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dst)
            return {
                "status": "completed",
                "output": str(dst),
                "output_bytes": dst.stat().st_size,
                "decoder": "validated_standard_3dgs_copy",
                "vertex_count": splat.count,
                "decimation_applied": False,
            }
    cli = find_splat_transform_cli(repo_root)
    if cli is None:
        return {"status": "blocked", "blockers": ["splat_transform_cli_unavailable"]}
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [node, str(cli), "-w", "-q", str(source)]
    if decimate and decimate > 0:
        cmd += ["-F", str(int(decimate))]
    cmd += [str(dst)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        return {"status": "blocked", "blockers": ["node_runtime_unavailable"]}
    except subprocess.TimeoutExpired:
        return {"status": "blocked", "blockers": ["splat_transform_timeout"]}
    if proc.returncode != 0 or not dst.is_file():
        return {
            "status": "blocked",
            "blockers": ["splat_transform_decode_failed"],
            "returncode": proc.returncode,
            "stderr_tail": (proc.stderr or "")[-1500:],
        }
    return {"status": "completed", "output": str(dst), "output_bytes": dst.stat().st_size}


def _encode_mp4(
    frame_paths: Sequence[Path], out_path: Path, *, seconds_per_frame: float = 1.6, fps: int = 30
) -> dict:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return {"status": "blocked", "blockers": ["ffmpeg_unavailable"]}
    frames = [p for p in frame_paths if p and Path(p).is_file()]
    if not frames:
        return {"status": "blocked", "blockers": ["no_frames_to_encode"]}
    # build a concat list holding each still for seconds_per_frame
    listing = out_path.with_suffix(".concat.txt")
    lines = []
    for p in frames:
        lines.append(f"file '{Path(p).resolve()}'")
        lines.append(f"duration {seconds_per_frame:.3f}")
    lines.append(f"file '{Path(frames[-1]).resolve()}'")  # last frame needs a repeat
    listing.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2,fps=" + str(fps),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        return {"status": "blocked", "blockers": ["ffmpeg_timeout"]}
    if proc.returncode != 0 or not out_path.is_file():
        return {
            "status": "blocked",
            "blockers": ["ffmpeg_encode_failed"],
            "stderr_tail": (proc.stderr or "")[-1200:],
        }
    return {"status": "completed", "mp4": str(out_path), "bytes": out_path.stat().st_size}


def render_splat_scene(
    source: str | Path,
    out_dir: str | Path,
    *,
    camera_ids: Sequence[str] = DEFAULT_CAMERA_IDS,
    camera_specs: Sequence[Mapping[str, Any]] | None = None,
    focus_bounds: Sequence[Sequence[float]] | None = None,
    focus_bounds_requests: Sequence[Mapping[str, Any]] | None = None,
    max_bounds_cameras: int | None = None,
    view_ring_margin: float = 3.0,
    view_ring_azimuths: int = 8,
    view_ring_elevations_deg: Sequence[float] = (10.0, 35.0, 60.0),
    view_ring_vfov_deg: float = 48.0,
    camera_id_prefix: str = "bounds",
    width: int = 1280,
    height: int = 960,
    decimate: int = 0,
    settle_frames: int = 6,
    settle_ms: int = 100,
    warmup_ms: int = 2000,
    graphics_backend: str = "swiftshader",
    up_axis: int | None = None,
    focus_point: Sequence[float] | None = None,
    overlay_json: str | Path | None = None,
    repo_root: str | Path | None = None,
    node: str = "node",
    decode_timeout: int = 900,
    render_timeout: int = 1800,
    encode_mp4: bool = True,
    require_empty_output: bool = False,
) -> dict:
    """Decode -> analyze -> frame cameras -> headless-render -> MP4. Returns a manifest
    dict with ``status`` in {completed, blocked} and a fail-closed ``blockers`` list."""
    source = Path(source).expanduser()
    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    root = _repo_root(repo_root)
    manifest: dict = {
        "schema_version": SCHEMA_VERSION,
        "status": "blocked",
        "rendered_by": RENDERED_BY,
        "source": str(source),
        "output_dir": str(out_dir),
        "blockers": [],
        "graphics_backend": str(graphics_backend).strip().lower(),
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "appearance_fidelity": {
            "full_resolution_source_is_appearance_truth": True,
            "global_decimation_applied": bool(decimate and decimate > 0),
            "appearance_fidelity_qualified": False,
            "evaluation_input_authorized": False,
            "qualification_required": "appearance_fidelity_qualification.v1",
        },
        "proof_boundary": {
            "captured_scene_displayed": False,
            "interaction_overlay_displayed": False,
            "rendered_by_isaac_rtx": False,
            "physics_or_navigation_proven": False,
        },
    }
    if not source.is_file() or source.suffix.lower() not in _SPLAT_SUFFIXES:
        manifest["blockers"].append("splat_source_missing_or_unsupported")
        return manifest
    if require_empty_output and any(out_dir.iterdir()):
        manifest["blockers"].append("render_output_directory_not_empty")
        return manifest
    manifest["source_digest"] = _sha256_file(source)
    camera_plan_inputs = sum(
        value is not None
        for value in (camera_specs, focus_bounds, focus_bounds_requests, focus_point)
    )
    if camera_plan_inputs > 1:
        manifest["blockers"].append("camera_planning_inputs_are_mutually_exclusive")
        return manifest
    normalized_camera_specs, camera_errors = _normalize_camera_specs(camera_specs)
    if camera_errors:
        manifest["blockers"].extend(camera_errors)
        return manifest
    bounds_camera_specs = None
    bounds_camera_plan = None
    if focus_bounds is not None:
        bounds_camera_specs, bounds_camera_plan, bounds_camera_errors = (
            _derive_bounds_camera_specs(
                focus_bounds,
                margin=view_ring_margin,
                n_azimuths=view_ring_azimuths,
                elevations_deg=view_ring_elevations_deg,
                vfov_deg=view_ring_vfov_deg,
                width=width,
                height=height,
                camera_id_prefix=camera_id_prefix,
            )
        )
        if bounds_camera_errors:
            manifest["blockers"].extend(bounds_camera_errors)
            return manifest
    multi_bounds_camera_specs = None
    multi_bounds_camera_plan = None
    if focus_bounds_requests is not None:
        (
            multi_bounds_camera_specs,
            multi_bounds_camera_plan,
            multi_bounds_camera_errors,
        ) = _derive_multi_bounds_camera_specs(
            focus_bounds_requests,
            width=width,
            height=height,
            max_total_cameras=max_bounds_cameras,
        )
        if multi_bounds_camera_errors:
            manifest["blockers"].extend(multi_bounds_camera_errors)
            return manifest
    normalized_graphics_backend = str(graphics_backend).strip().lower()
    if normalized_graphics_backend not in {"swiftshader", "metal"}:
        manifest["blockers"].append("unsupported_graphics_backend")
        return manifest
    if isinstance(decimate, bool) or not isinstance(decimate, int) or decimate < 0:
        manifest["blockers"].append("invalid_splat_decimation_request")
        return manifest
    if decimate > 0:
        manifest["blockers"].append("unqualified_global_splat_decimation_forbidden")
        return manifest

    # 1) decode/decimate to a standard 3DGS PLY (analysis + render input)
    std_ply = out_dir / "scene_standard.ply"
    dec = _decimate_to_standard_ply(
        source, std_ply, decimate, repo_root=root, node=node, timeout=decode_timeout
    )
    if dec.get("status") != "completed":
        manifest["blockers"].extend(dec.get("blockers", ["splat_decode_failed"]))
        manifest["decode"] = dec
        return manifest
    manifest["decode"] = dec
    if not std_ply.is_file():
        manifest["blockers"].append("standard_ply_missing_after_decode")
        return manifest
    manifest["standard_ply_digest"] = _sha256_file(std_ply)

    # 2) analyze geometry
    try:
        splat = read_standard_3dgs_ply(std_ply)
        geom = analyze_scene(splat, up_axis=up_axis)
    except Exception as exc:  # noqa: BLE001 - surface as a blocker, never a fake pass
        manifest["blockers"].append("scene_analysis_failed")
        manifest["scene_analysis_error"] = repr(exc)
        return manifest
    manifest["scene_geometry"] = geom.to_dict()

    # 3) frame the eval cameras and render headlessly
    if focus_point is not None:
        if len(focus_point) != 3 or not all(math.isfinite(float(value)) for value in focus_point):
            manifest["blockers"].append("invalid_focus_point")
            return manifest
        normalized_focus = [float(value) for value in focus_point]
    else:
        normalized_focus = None
    manifest["appearance_fidelity"].update(
        {
            "source_splat_count": int(splat.count),
            "retained_splat_count": int(splat.count),
            "retained_splat_fraction": 1.0,
            "spherical_harmonics_degree": (
                int(round((1 + int(splat.sh_rest.shape[1] / 3)) ** 0.5)) - 1
                if splat.sh_rest is not None and splat.sh_rest.size
                else 0
            ),
        }
    )
    cameras = (
        normalized_camera_specs
        if normalized_camera_specs is not None
        else bounds_camera_specs
        if bounds_camera_specs is not None
        else multi_bounds_camera_specs
        if multi_bounds_camera_specs is not None
        else derive_eval_cameras(geom, camera_ids, focus_point=normalized_focus)
    )
    manifest["camera_focus_point"] = normalized_focus
    manifest["camera_set_origin"] = (
        "caller_supplied_native_spark_specs"
        if normalized_camera_specs is not None
        else "derived_bounds_view_ring"
        if bounds_camera_specs is not None
        else "derived_multi_entity_bounds_view_ring"
        if multi_bounds_camera_specs is not None
        else "derived_eval_cameras"
    )
    manifest["camera_plan"] = bounds_camera_plan or multi_bounds_camera_plan
    manifest["camera_set_digest"] = canonical_digest({"cameras": cameras})
    cameras_json = out_dir / "cameras.json"
    cameras_json.write_text(json.dumps(cameras), encoding="utf-8")
    manifest["camera_file"] = {
        "path": str(cameras_json),
        "size_bytes": cameras_json.stat().st_size,
        "sha256": _sha256_file(cameras_json),
    }
    manifest["camera_calibration"] = [
        _look_at_camera_calibration(camera, width=width, height=height)
        for camera in cameras
    ]
    harness = root / RENDER_HARNESS_REL
    if not harness.is_file():
        manifest["blockers"].append("render_harness_unavailable")
        return manifest
    entry = root / "tools/splat_render/src/render_entry.mjs"
    manifest["renderer_identity"] = {
        "name": RENDERED_BY,
        "harness_path": str(harness),
        "harness_sha256": _sha256_file(harness),
        "entry_path": str(entry) if entry.is_file() else None,
        "entry_sha256": _sha256_file(entry) if entry.is_file() else None,
        "width": int(width),
        "height": int(height),
        "pixel_ratio": 1,
        "supersampling": 1,
        "antialias": True,
        "alpha": False,
        "background_rgb_hex": "0x0b0b10",
        "output_format": "lossless_png",
        "color_space": "threejs_default_unqualified",
        "warmup_ms": int(warmup_ms),
        "settle_frames": int(settle_frames),
        "settle_ms": int(settle_ms),
        **_renderer_dependency_identity(root),
    }
    frames_dir = out_dir / "frames"
    cmd = [
        node, str(harness),
        "--splat", str(std_ply),
        "--out", str(frames_dir),
        "--cameras", str(cameras_json),
        "--width", str(width), "--height", str(height),
        "--warmup-ms", str(warmup_ms),
        "--settle-frames", str(settle_frames),
        "--settle-ms", str(settle_ms),
        "--graphics-backend", normalized_graphics_backend,
    ]
    if overlay_json:
        overlay_path = Path(overlay_json).expanduser().resolve()
        if not overlay_path.is_file():
            manifest["blockers"].append("interaction_overlay_missing")
            return manifest
        cmd += ["--overlay", str(overlay_path)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=render_timeout)
    except FileNotFoundError:
        manifest["blockers"].append("node_runtime_unavailable")
        return manifest
    except subprocess.TimeoutExpired:
        manifest["blockers"].append("render_timeout")
        return manifest
    render_result = {}
    out = (proc.stdout or "").strip()
    if out:
        try:
            render_result = json.loads(out)
        except Exception:  # noqa: BLE001 - tolerate leading noise, parse trailing object
            try:
                render_result = json.loads(out[out.index("{"):])
            except Exception:  # noqa: BLE001
                render_result = {}
    manifest["render"] = {
        "returncode": proc.returncode,
        "status": render_result.get("status"),
        "blockers": render_result.get("blockers", []),
        "page_errors": render_result.get("page_errors", [])[:5],
        "overlay": render_result.get("overlay"),
        "stderr_tail": (proc.stderr or "")[-1200:] if proc.returncode != 0 else "",
    }
    rendered = render_result.get("cameras", []) or []
    # a frame is "real" (non-blank) if clearly larger than a uniform-color PNG
    blank_threshold = max(30000, int(0.02 * width * height))
    cams_out = []
    nonblank = 0
    for cam in rendered:
        is_real = int(cam.get("bytes", 0)) >= blank_threshold
        nonblank += int(is_real)
        cams_out.append({
            "id": cam.get("id"),
            "path": cam.get("path"),
            "bytes": cam.get("bytes"),
            "nonblank": is_real,
            "digest": (
                _sha256_file(Path(cam["path"]))
                if cam.get("path") and Path(cam["path"]).is_file()
                else None
            ),
        })
    manifest["cameras"] = cams_out
    manifest["nonblank_camera_count"] = nonblank

    if proc.returncode != 0 or render_result.get("status") != "completed" or nonblank == 0:
        manifest["blockers"].append("splat_render_produced_no_scene")
        manifest["blockers"].extend(render_result.get("blockers", []))
        return manifest

    # 4) MP4
    if encode_mp4:
        ordered = [Path(c["path"]) for c in cams_out if c.get("path") and c.get("nonblank")]
        mp4 = _encode_mp4(ordered, out_dir / "scene_render.mp4")
        manifest["mp4"] = mp4
        if mp4.get("status") != "completed":
            # MP4 is a convenience; images already prove the scene. Record but don't fail.
            manifest.setdefault("warnings", []).append("mp4_encode_skipped")

    manifest["status"] = "completed"
    manifest["proof_boundary"]["captured_scene_displayed"] = True
    manifest["proof_boundary"]["interaction_overlay_displayed"] = bool(
        render_result.get("overlay", {}).get("item_count")
        if isinstance(render_result.get("overlay"), dict)
        else False
    )
    manifest["robot_start_pose"] = geom.suggested_start
    manifest["render_manifest_digest"] = canonical_digest(
        manifest, digest_field="render_manifest_digest"
    )
    return manifest


def _pick_splat_source(*candidates) -> Path | None:
    for cand in candidates:
        if not cand:
            continue
        p = Path(cand).expanduser()
        if p.is_file() and p.suffix.lower() in _SPLAT_SUFFIXES:
            return p
    return None


def attach_splat_render_to_eval(
    *,
    job_dir: str | Path,
    ply_asset: str | Path | None,
    spz_asset: str | Path | None,
    camera_ids: Sequence[str] = DEFAULT_CAMERA_IDS,
    generated_at: str | None = None,
    repo_root: str | Path | None = None,
    options: dict | None = None,
) -> dict:
    """Run the local splat render for an Isaac eval job and persist its manifest under
    ``<job_dir>/splat_scene_render/manifest.json``. Returns the manifest. Always
    fail-closed; never raises into the eval."""
    job_dir = Path(job_dir)
    out_dir = job_dir / "splat_scene_render"
    out_dir.mkdir(parents=True, exist_ok=True)
    source = _pick_splat_source(ply_asset, spz_asset)
    if source is None:
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "blocked",
            "rendered_by": RENDERED_BY,
            "blockers": ["splat_source_unavailable_for_render"],
            "proof_boundary": {"captured_scene_displayed": False, "rendered_by_isaac_rtx": False},
        }
    else:
        try:
            manifest = render_splat_scene(
                source, out_dir, camera_ids=camera_ids, repo_root=repo_root, **(options or {})
            )
        except Exception as exc:  # noqa: BLE001 - never break the eval
            manifest = {
                "schema_version": SCHEMA_VERSION,
                "status": "blocked",
                "rendered_by": RENDERED_BY,
                "blockers": ["splat_render_exception"],
                "error": repr(exc),
                "proof_boundary": {"captured_scene_displayed": False, "rendered_by_isaac_rtx": False},
            }
    if generated_at:
        manifest["generated_at"] = generated_at
    try:
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description="Render a Gaussian-splat capture (.ply/.spz) into per-camera images "
        "+ MP4 of the real scene via the reference Spark renderer (local, no GPU)."
    )
    ap.add_argument("--splat", required=True, help="compressed PLY / SPZ / standard PLY")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=960)
    ap.add_argument(
        "--decimate",
        type=int,
        default=0,
        help="must remain 0; global decimation is forbidden for fidelity renders",
    )
    ap.add_argument("--settle-frames", type=int, default=6)
    ap.add_argument("--settle-ms", type=int, default=100)
    ap.add_argument("--warmup-ms", type=int, default=2000)
    ap.add_argument(
        "--graphics-backend",
        default="swiftshader",
        choices=("swiftshader", "metal"),
        help="local Spark graphics backend; Metal uses the Mac GPU, SwiftShader is software",
    )
    ap.add_argument("--up-axis", type=int, default=None, choices=[0, 1, 2])
    ap.add_argument(
        "--focus-point",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="site-coordinate task focus used for wrist and task-focus framing",
    )
    ap.add_argument(
        "--focus-bounds",
        type=float,
        nargs=6,
        metavar=("XMIN", "YMIN", "ZMIN", "XMAX", "YMAX", "ZMAX"),
        help="derive a deterministic close-up camera ring around metric bounds",
    )
    ap.add_argument(
        "--focus-bounds-request-json",
        help="JSON object with a targets[] list of independently bounded entities",
    )
    ap.add_argument("--view-ring-margin", type=float, default=3.0)
    ap.add_argument("--view-ring-azimuths", type=int, default=8)
    ap.add_argument(
        "--view-ring-elevations-deg",
        type=float,
        nargs="+",
        default=(10.0, 35.0, 60.0),
    )
    ap.add_argument("--view-ring-vfov-deg", type=float, default=48.0)
    ap.add_argument("--camera-id-prefix", default="bounds")
    ap.add_argument("--overlay", help="optional JSON box/cylinder preview overlay")
    ap.add_argument(
        "--cameras-json",
        help="optional native Spark camera list [{id,spec:{pos,target,fov,up}}]",
    )
    ap.add_argument("--no-mp4", action="store_true")
    ap.add_argument(
        "--require-empty-output",
        action="store_true",
        help="fail before decode rather than overwrite any retained attempt artifact",
    )
    args = ap.parse_args(argv)
    camera_specs = None
    if args.cameras_json:
        camera_specs = json.loads(Path(args.cameras_json).read_text(encoding="utf-8"))
    focus_bounds_requests = None
    max_bounds_cameras = None
    if args.focus_bounds_request_json:
        request_payload = json.loads(
            Path(args.focus_bounds_request_json).read_text(encoding="utf-8")
        )
        focus_bounds_requests = request_payload.get("targets")
        max_bounds_cameras = request_payload.get("max_total_cameras")
    manifest = render_splat_scene(
        args.splat,
        args.out,
        width=args.width,
        height=args.height,
        decimate=args.decimate,
        settle_frames=args.settle_frames,
        settle_ms=args.settle_ms,
        warmup_ms=args.warmup_ms,
        graphics_backend=args.graphics_backend,
        up_axis=args.up_axis,
        focus_point=args.focus_point,
        focus_bounds=(
            (args.focus_bounds[:3], args.focus_bounds[3:])
            if args.focus_bounds is not None
            else None
        ),
        focus_bounds_requests=focus_bounds_requests,
        max_bounds_cameras=max_bounds_cameras,
        view_ring_margin=args.view_ring_margin,
        view_ring_azimuths=args.view_ring_azimuths,
        view_ring_elevations_deg=args.view_ring_elevations_deg,
        view_ring_vfov_deg=args.view_ring_vfov_deg,
        camera_id_prefix=args.camera_id_prefix,
        camera_specs=camera_specs,
        overlay_json=args.overlay,
        encode_mp4=not args.no_mp4,
        require_empty_output=args.require_empty_output,
    )
    manifest_path = Path(args.out).expanduser() / "splat_render_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0 if manifest.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
