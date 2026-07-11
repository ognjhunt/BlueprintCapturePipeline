"""Isaac review renderer canary: prove the review render lane is visually usable.

This is a SEPARATE contract from the 64x64 fast startup canary in
``isaac_worker_runtime_preflight``.  It renders a deterministic, lit USD scene
at review resolution (480x640 landscape or 640x480 portrait) containing a
visible Unitree G1 plus a floor/target marker, and validates that the rendered
frame is actually reviewable (not blank, not flat, not clipped, not a stale
copy of a prior attempt).  The kitchen scene is intentionally NOT involved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import zlib
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from .claim_contract_keys import PUBLIC_CLAIM_UPGRADE_ALLOWED_KEY
from .common import ensure_dir, utc_now_iso, write_json


ISAAC_REVIEW_RENDERER_CANARY_SCHEMA_VERSION = "isaac_review_renderer_canary.v1"

REVIEW_FRAME_HEIGHT = 480
REVIEW_FRAME_WIDTH = 640

# rows x cols per orientation.
REVIEW_FRAME_ORIENTATIONS: dict[str, tuple[int, int]] = {
    "landscape": (REVIEW_FRAME_HEIGHT, REVIEW_FRAME_WIDTH),
    "portrait": (REVIEW_FRAME_WIDTH, REVIEW_FRAME_HEIGHT),
}

REVIEW_RENDERER_CANARY_CONTRACT = (
    "Only isaac_review_renderer_canary.v1 may establish "
    "isaac_review_renderer_operational=true. It proves the Isaac review render "
    "lane produces a visually usable review-resolution frame with a visible "
    "Unitree G1 and target marker. It does NOT establish kitchen scene "
    "placement, policy execution, or task success, and the 64x64 fast startup "
    "canary cannot substitute for it."
)

FRAME_PNG_FILENAME = "review_renderer_canary_frame.png"
CONTACT_SHEET_PNG_FILENAME = "review_renderer_canary_contact_sheet.png"
RESULT_JSON_FILENAME = "isaac_review_renderer_canary.json"

UNITREE_G1_SEMANTIC_LABEL = "unitree_g1"
TARGET_MARKER_SEMANTIC_LABEL = "target_marker"

BLANK_BLACK_MAX_MEAN_LUMINANCE = 2.0
BLANK_WHITE_MIN_MEAN_LUMINANCE = 250.0
FLAT_FRAME_MAX_LUMINANCE_STD = 1.0
SEVERE_CLIPPING_MIN_FRACTION = 0.5
MIN_G1_PIXEL_FRACTION = 0.005
MIN_TARGET_MARKER_PIXEL_FRACTION = 0.005


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _check(name: str, status: str, **details: Any) -> dict[str, Any]:
    return {"name": name, "status": status, **details}


def _to_uint8_rgb(frame: Any) -> np.ndarray:
    """Best-effort conversion of an RGB(A) frame to a HxWx3 uint8 array."""

    array = np.asarray(frame)
    if array.dtype == np.dtype(object) or array.ndim not in (2, 3) or array.size == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    values = array.astype(np.float64, copy=True)
    values = np.nan_to_num(values, nan=0.0, posinf=255.0, neginf=0.0)
    if np.issubdtype(array.dtype, np.floating) and float(values.max(initial=0.0)) <= 1.0:
        values = values * 255.0
    values = np.clip(values, 0.0, 255.0)
    rgb = values.astype(np.uint8)
    if rgb.ndim == 2:
        rgb = np.stack([rgb, rgb, rgb], axis=-1)
    elif rgb.shape[2] == 1:
        rgb = np.repeat(rgb, 3, axis=2)
    else:
        rgb = rgb[:, :, :3]
    return np.ascontiguousarray(rgb)


def _frame_sha256(rgb: np.ndarray) -> str:
    header = f"{rgb.shape[0]}x{rgb.shape[1]}x{rgb.shape[2]}".encode("ascii")
    return hashlib.sha256(header + rgb.tobytes()).hexdigest()


def validate_review_render_frame(
    frame: Any,
    *,
    expected_orientation: str = "landscape",
    g1_pixel_fraction: float | None = None,
    target_marker_pixel_fraction: float | None = None,
    prior_frame_sha256: str | None = None,
) -> dict[str, Any]:
    """Pure validator for a rendered review frame.

    ``g1_pixel_fraction`` and ``target_marker_pixel_fraction`` are
    segmentation-derived visibility fractions supplied by the renderer
    (fraction of frame pixels labeled Unitree G1 / target marker by the
    Replicator semantic segmentation annotator).
    """

    if expected_orientation not in REVIEW_FRAME_ORIENTATIONS:
        raise ValueError(f"unknown expected_orientation: {expected_orientation!r}")
    expected_rows, expected_cols = REVIEW_FRAME_ORIENTATIONS[expected_orientation]

    blockers: list[str] = []
    array = np.asarray(frame)
    shape = list(array.shape)
    numeric = array.dtype != np.dtype(object)
    dims_ok = (
        numeric
        and array.ndim == 3
        and array.shape[2] in (3, 4)
        and (int(array.shape[0]), int(array.shape[1])) == (expected_rows, expected_cols)
    )
    if not dims_ok:
        blockers.append("review_frame_wrong_dimensions")

    non_finite_count = 0
    if numeric and np.issubdtype(array.dtype, np.floating):
        non_finite_count = int((~np.isfinite(array)).sum())
        if non_finite_count:
            blockers.append("review_frame_non_finite_pixels")

    rgb = _to_uint8_rgb(array)
    frame_sha256 = _frame_sha256(rgb)
    luminance = rgb.astype(np.float64).mean(axis=-1)
    mean_luminance = float(luminance.mean())
    std_luminance = float(luminance.std())
    clip_low_fraction = float((rgb == 0).mean())
    clip_high_fraction = float((rgb == 255).mean())
    clip_total_fraction = clip_low_fraction + clip_high_fraction

    if mean_luminance <= BLANK_BLACK_MAX_MEAN_LUMINANCE:
        blockers.append("review_frame_blank_black")
    elif mean_luminance >= BLANK_WHITE_MIN_MEAN_LUMINANCE:
        blockers.append("review_frame_blank_white")
    elif std_luminance <= FLAT_FRAME_MAX_LUMINANCE_STD:
        blockers.append("review_frame_flat")
    elif clip_total_fraction >= SEVERE_CLIPPING_MIN_FRACTION:
        blockers.append("review_frame_severe_clipping")

    if g1_pixel_fraction is None or float(g1_pixel_fraction) < MIN_G1_PIXEL_FRACTION:
        blockers.append("review_frame_unitree_g1_not_visible")
    if (
        target_marker_pixel_fraction is None
        or float(target_marker_pixel_fraction) < MIN_TARGET_MARKER_PIXEL_FRACTION
    ):
        blockers.append("review_frame_target_marker_missing")

    prior = _string(prior_frame_sha256).lower()
    if prior and frame_sha256 == prior:
        blockers.append("review_frame_checksum_reused_from_prior_attempt")

    return {
        "check": "review_render_frame_validation",
        "status": "passed" if not blockers else "blocked",
        "blockers": blockers,
        "frame_sha256": frame_sha256,
        "orientation": {
            "expected": expected_orientation,
            "expected_rows": expected_rows,
            "expected_cols": expected_cols,
            "actual_shape": shape,
            "matches_expected": dims_ok,
        },
        "luminance": {
            "mean": mean_luminance,
            "std": std_luminance,
            "min": float(luminance.min()),
            "max": float(luminance.max()),
        },
        "clipping": {
            "fraction_saturated_low": clip_low_fraction,
            "fraction_saturated_high": clip_high_fraction,
            "fraction_saturated_total": clip_total_fraction,
        },
        "non_finite_pixel_count": non_finite_count,
        "g1_pixel_fraction": (
            float(g1_pixel_fraction) if g1_pixel_fraction is not None else None
        ),
        "target_marker_pixel_fraction": (
            float(target_marker_pixel_fraction)
            if target_marker_pixel_fraction is not None
            else None
        ),
        "prior_frame_sha256": prior or None,
        "thresholds": {
            "blank_black_max_mean_luminance": BLANK_BLACK_MAX_MEAN_LUMINANCE,
            "blank_white_min_mean_luminance": BLANK_WHITE_MIN_MEAN_LUMINANCE,
            "flat_frame_max_luminance_std": FLAT_FRAME_MAX_LUMINANCE_STD,
            "severe_clipping_min_fraction": SEVERE_CLIPPING_MIN_FRACTION,
            "min_g1_pixel_fraction": MIN_G1_PIXEL_FRACTION,
            "min_target_marker_pixel_fraction": MIN_TARGET_MARKER_PIXEL_FRACTION,
        },
    }


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def _encode_png_rgb(rgb: np.ndarray) -> bytes:
    """Minimal dependency-free PNG encoder (8-bit RGB, no filtering)."""

    height, width = int(rgb.shape[0]), int(rgb.shape[1])
    raw = b"".join(b"\x00" + rgb[row].tobytes() for row in range(height))
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", zlib.compress(raw, 6))
        + _png_chunk(b"IEND", b"")
    )


def _write_png(path: Path, frame: Any) -> str:
    """Write ``frame`` to ``path`` as PNG; returns the encoder used."""

    rgb = _to_uint8_rgb(frame)
    try:
        from PIL import Image  # type: ignore[import-not-found]

        Image.fromarray(rgb, mode="RGB").save(path, format="PNG")
        return "pillow"
    except Exception:
        path.write_bytes(_encode_png_rgb(rgb))
        return "builtin_zlib"


def _contact_sheet(frames: list[Any]) -> np.ndarray:
    tiles = [_to_uint8_rgb(frame) for frame in frames]
    height = max(tile.shape[0] for tile in tiles)
    padded: list[np.ndarray] = []
    for tile in tiles:
        if tile.shape[0] < height:
            pad = np.zeros((height - tile.shape[0], tile.shape[1], 3), dtype=np.uint8)
            tile = np.concatenate([tile, pad], axis=0)
        padded.append(tile)
    return np.concatenate(padded, axis=1)


def run_isaac_review_renderer_canary(
    *,
    output_dir: str | Path,
    launch_session_id: str,
    image_digest: str,
    expected_orientation: str = "landscape",
    renderer_backend: Callable[[], Mapping[str, Any]] | None = None,
    prior_frame_sha256: str | None = None,
) -> dict[str, Any]:
    """Run the review renderer canary and persist its artifacts.

    ``renderer_backend`` is an injectable zero-argument callable returning a
    mapping with keys ``frames`` (list of numpy RGB(A) arrays),
    ``g1_pixel_fraction``, ``target_marker_pixel_fraction``,
    ``rtx_driver_verifier_errors`` (list of strings), ``renderer`` (name), and
    optionally ``close`` (callable invoked after the result JSON is written,
    mirroring the write-before-SimulationApp.close ordering of the fast
    startup canary).  The real Isaac backend only exists on a GPU worker, so
    ``None`` yields a blocked result instead of importing isaacsim here.
    """

    if expected_orientation not in REVIEW_FRAME_ORIENTATIONS:
        raise ValueError(f"unknown expected_orientation: {expected_orientation!r}")
    out_dir = Path(output_dir)
    ensure_dir(out_dir)
    generated_at = utc_now_iso()
    checks: list[dict[str, Any]] = []
    blockers: list[str] = []

    nonce = _string(launch_session_id)
    digest = _string(image_digest)
    if nonce:
        checks.append(_check("launch_nonce_binding", "passed", launch_session_id=nonce))
    else:
        blockers.append("review_canary_launch_nonce_missing")
        checks.append(_check("launch_nonce_binding", "blocked"))
    if digest:
        checks.append(_check("image_digest_binding", "passed", image_digest=digest))
    else:
        blockers.append("review_canary_image_digest_missing")
        checks.append(_check("image_digest_binding", "blocked"))

    backend_result: dict[str, Any] = {}
    backend_available = renderer_backend is not None
    backend_ok = False
    if not backend_available:
        blockers.append("review_renderer_backend_unavailable")
        checks.append(
            _check(
                "review_renderer_backend",
                "blocked",
                reason="review_renderer_backend_unavailable",
                note="real Isaac review renderer backend only runs on a GPU worker",
            )
        )
    else:
        try:
            backend_result = dict(renderer_backend())
            backend_ok = True
            checks.append(
                _check(
                    "review_renderer_backend",
                    "passed",
                    renderer=_string(backend_result.get("renderer")) or "unknown",
                )
            )
        except Exception as exc:
            blockers.append("review_renderer_backend_failed")
            checks.append(
                _check(
                    "review_renderer_backend",
                    "blocked",
                    error_type=type(exc).__name__,
                    error=str(exc)[:2000],
                )
            )

    verifier_errors = [
        _string(error)
        for error in (backend_result.get("rtx_driver_verifier_errors") or [])
        if _string(error)
    ]
    if verifier_errors:
        blockers.append("rtx_driver_verifier_error")
        checks.append(
            _check("rtx_driver_verifier", "blocked", errors=verifier_errors[:20])
        )
    elif backend_ok:
        checks.append(_check("rtx_driver_verifier", "passed", errors=[]))

    frames = list(backend_result.get("frames") or [])
    if backend_ok and not frames:
        blockers.append("review_renderer_no_frames_produced")
        checks.append(_check("review_frame_validation", "blocked_not_run", frame_count=0))

    validation: dict[str, Any] | None = None
    frame_png_path: Path | None = None
    contact_sheet_path: Path | None = None
    png_encoder: str | None = None
    if frames:
        validation = validate_review_render_frame(
            frames[0],
            expected_orientation=expected_orientation,
            g1_pixel_fraction=backend_result.get("g1_pixel_fraction"),
            target_marker_pixel_fraction=backend_result.get("target_marker_pixel_fraction"),
            prior_frame_sha256=prior_frame_sha256,
        )
        blockers.extend(validation["blockers"])
        checks.append(
            _check(
                "review_frame_validation",
                validation["status"],
                frame_sha256=validation["frame_sha256"],
                frame_count=len(frames),
            )
        )
        frame_png_path = out_dir / FRAME_PNG_FILENAME
        png_encoder = _write_png(frame_png_path, frames[0])
        if len(frames) > 1:
            contact_sheet_path = out_dir / CONTACT_SHEET_PNG_FILENAME
            _write_png(contact_sheet_path, _contact_sheet(frames))

    blockers = list(dict.fromkeys(blockers))
    operational = not blockers
    result_json_path = out_dir / RESULT_JSON_FILENAME
    payload: dict[str, Any] = {
        "schema_version": ISAAC_REVIEW_RENDERER_CANARY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "passed" if operational else "blocked",
        "simulator": "isaac",
        "canary": "review_renderer",
        "contract": REVIEW_RENDERER_CANARY_CONTRACT,
        "launch_session_id": nonce or None,
        "image_digest": digest or None,
        "expected_orientation": expected_orientation,
        "checks": checks,
        "blockers": blockers,
        "frame_validation": validation,
        "artifacts": {
            "result_json": str(result_json_path),
            "frame_png": str(frame_png_path) if frame_png_path else None,
            "contact_sheet_png": str(contact_sheet_path) if contact_sheet_path else None,
            "png_encoder": png_encoder,
            "frame_count": len(frames),
            "bound_launch_session_id": nonce or None,
            "bound_image_digest": digest or None,
        },
        "isaac_review_renderer_operational": operational,
        "fast_startup_canary_is_not_review_proof": True,
        "claim_boundary": {
            "isaac_review_renderer_operational_claim_source": (
                "isaac_review_renderer_canary_only"
            ),
            "kitchen_scene_placement_proven": False,
            "policy_execution_proven": False,
            "proves_task_success": False,
            "fast_startup_canary_is_not_review_proof": True,
            PUBLIC_CLAIM_UPGRADE_ALLOWED_KEY: False,
        },
        "secret_values_in_artifact": False,
    }
    write_json(result_json_path, payload)
    close_callable = backend_result.get("close")
    if callable(close_callable):
        try:
            close_callable()
        except Exception:
            # Isaac fastShutdown may kill the process from close(); the JSON
            # artifact was persisted above, so a close failure changes nothing.
            pass
    return payload


def _semantic_label_fraction(seg_output: Any, label: str) -> float | None:
    """Fraction of pixels carrying ``label`` in a Replicator segmentation output."""

    try:
        if isinstance(seg_output, Mapping):
            data = np.asarray(seg_output.get("data"))
            info = seg_output.get("info") or {}
        else:
            data = np.asarray(getattr(seg_output, "data", None))
            info = getattr(seg_output, "info", None) or {}
        id_to_labels = info.get("idToLabels") or {}
        matching_ids = [
            int(seg_id)
            for seg_id, seg_label in id_to_labels.items()
            if label in json.dumps(seg_label).lower()
        ]
        if data.size == 0 or not matching_ids:
            return 0.0
        return float(np.isin(data, matching_ids).mean())
    except Exception:
        return None


def _default_g1_usd_path() -> str:  # pragma: no cover - requires Isaac assets
    override = os.environ.get("BLUEPRINT_G1_USD_PATH")
    if override:
        return override
    from isaacsim.storage.native import get_assets_root_path  # type: ignore

    assets_root = get_assets_root_path() or ""
    return f"{assets_root}/Isaac/Robots/Unitree/G1/g1.usd"


def _isaac_renderer_backend(
    *, expected_orientation: str = "landscape"
) -> dict[str, Any]:  # pragma: no cover - requires an Isaac GPU worker
    """Real Isaac backend: deterministic lit scene with a G1 and target marker."""

    from isaacsim import SimulationApp  # type: ignore[import-not-found]

    simulation_app = SimulationApp({"headless": True, "renderer": "RayTracedLighting"})
    import omni.replicator.core as rep  # type: ignore[import-not-found]

    rows, cols = REVIEW_FRAME_ORIENTATIONS[expected_orientation]
    rep.create.plane(scale=20, semantics=[("class", "floor")])
    rep.create.light(light_type="dome", intensity=1000)
    rep.create.light(light_type="distant", intensity=2500, rotation=(315, 0, 0))
    rep.create.cube(
        position=(1.0, 0.0, 0.1),
        scale=0.2,
        semantics=[("class", TARGET_MARKER_SEMANTIC_LABEL)],
    )
    rep.create.from_usd(
        _default_g1_usd_path(),
        semantics=[("class", UNITREE_G1_SEMANTIC_LABEL)],
    )
    camera = rep.create.camera(position=(3.2, 3.2, 1.8), look_at=(0.0, 0.0, 0.8))
    render_product = rep.create.render_product(camera, (cols, rows))
    rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    seg_annotator = rep.AnnotatorRegistry.get_annotator(
        "semantic_segmentation", init_params={"colorize": False}
    )
    rgb_annotator.attach([render_product])
    seg_annotator.attach([render_product])
    frame = None
    for _ in range(16):
        rep.orchestrator.step()
        data = rgb_annotator.get_data()
        if int(getattr(data, "size", 0) or 0) > 0:
            frame = np.asarray(data)
            break
    seg_output = seg_annotator.get_data()
    return {
        "frames": [frame] if frame is not None else [],
        "g1_pixel_fraction": _semantic_label_fraction(seg_output, UNITREE_G1_SEMANTIC_LABEL),
        "target_marker_pixel_fraction": _semantic_label_fraction(
            seg_output, TARGET_MARKER_SEMANTIC_LABEL
        ),
        "rtx_driver_verifier_errors": [],
        "renderer": "RayTracedLighting",
        "close": simulation_app.close,
    }


def _resolve_renderer_backend(
    *, expected_orientation: str = "landscape"
) -> Callable[[], Mapping[str, Any]] | None:
    """Return the real Isaac backend when isaacsim is importable, else None."""

    try:
        import isaacsim  # type: ignore[import-not-found]  # noqa: F401
    except Exception:
        return None

    def _backend() -> dict[str, Any]:  # pragma: no cover - requires GPU worker
        return _isaac_renderer_backend(expected_orientation=expected_orientation)

    return _backend


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--launch-session-id", default="")
    parser.add_argument("--image-digest", default="")
    parser.add_argument(
        "--orientation",
        choices=sorted(REVIEW_FRAME_ORIENTATIONS),
        default="landscape",
    )
    args = parser.parse_args(argv)
    payload = run_isaac_review_renderer_canary(
        output_dir=args.output_dir,
        launch_session_id=args.launch_session_id,
        image_digest=args.image_digest,
        expected_orientation=args.orientation,
        renderer_backend=_resolve_renderer_backend(expected_orientation=args.orientation),
    )
    print(json.dumps({"status": payload["status"], "blockers": payload["blockers"]}))
    return 0 if payload["status"] == "passed" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
