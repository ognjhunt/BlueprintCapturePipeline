"""One still of a website job's simulation, for the job's thumbnail on the WebApp's job cards and job page.

Owner, 2026-11-23: every job has a thumbnail, a render of its simulation when one exists; the WebApp shows a frame
of the job's company clip until then. This renders the scene preview's source splat (the room deployment companies
open in the viewer) from the viewer's own start view (``website_scene_preview._start_view``), once per delivered
scene preview, with the CPU renderer (``atlas_splat_render``), and tells the WebApp through the signed
``job-thumbnail`` control operation. It reads retained files, calls no provider and buys nothing.

Privacy and rights are hard stops, as for the preview itself: nothing is rendered or delivered for a withdrawn
capture, or unless the WebApp's current task context allows derived scene generation, and the delivery happens under
the preview's lock, which withdrawal deletion takes. A WebApp refusal (a 4xx answer) is final for that preview.

Bounded: the render is at most 480 by 300 (the WebApp's thumbnail size) from at most ``MAX_GAUSSIANS`` of the
source's most opaque Gaussians; a pass renders at most ``RENDERS_PER_TICK`` captures and starts none after
``TICK_BUDGET_SECONDS``. A render that shows next to nothing (``looks_empty``) is recorded and never sent, so the
video frame stays. A failed render or delivery is retried on a later pass, at most ``MAX_ATTEMPTS`` times per preview.
"""
from __future__ import annotations

import base64
import hashlib
import io
import logging
import math
import re
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .website_atlas_preview_store import capture_withdrawn
from .website_scene_preview import (
    DIRECTORY, MANIFEST_FILE, MANIFEST_SCHEMA, ScenePreviewBusy, _current_rights, _identity, _read, _reason,
    _sha256_file, _withdrawn_from_view, capture_roots, preview_lock)

logger = logging.getLogger(__name__)

SCHEMA = "website_job_thumbnail.v1"
ACCEPTANCE_SCHEMA = "website_job_thumbnail_acceptance.v1"
RECEIPT_SCHEMA = "website_job_thumbnail_receipt.v1"
RECEIPT_FILE = "job_thumbnail.json"
IMAGE_FILE = "job_thumbnail.png"
OPERATION = "job-thumbnail"
SOURCE = "simulation_render"
CAMERA = "scene_preview_start_view"
#: The WebApp's thumbnail box; the render keeps the box's own aspect.
WIDTH, HEIGHT = 480, 300
#: A person's field of view standing in the room, wide enough to show it.
VERTICAL_FOV_DEGREES = 55.0
#: Gaussians rendered at most: the most opaque, so a 2 M Gaussian room renders in a fraction of a full pass.
MAX_GAUSSIANS = 600_000
#: The WebApp's limit for a thumbnail's PNG bytes.
MAX_PNG_BYTES = 600_000
#: Next to nothing: under a quarter of the image covered, or darker on average than about 10 of 255.
MIN_COVERAGE = 0.25
MIN_MEAN_LUMINANCE = 0.04
MAX_ATTEMPTS = 3
RENDERS_PER_TICK = 4
TICK_BUDGET_SECONDS = 600
#: Part of the render's inputs digest: a change here renders every thumbnail again.
BUILDER_REVISION = "website_job_thumbnail_builder.v1"
#: The outcomes that end the work for one preview.
FINAL = frozenset({"delivered", "refused", "skipped_empty", "no_start_view"})
_HTTP_CLIENT_REFUSAL = re.compile(r"_http_4(?!29)\d\d:")


def camera_for(start_view: Mapping[str, Any], width: int = WIDTH, height: int = HEIGHT) -> dict[str, Any]:
    """A posed view (``atlas_splat_render.camera_of``) standing at ``position`` and looking at ``target``.

    Camera axes are right-up-back (the camera looks down -Z) in the scene's own +Y-up frame, as the viewer has it.
    """
    import numpy as np
    position = np.asarray(start_view["position"], float)
    forward = np.asarray(start_view["target"], float) - position
    if position.shape != (3,) or forward.shape != (3,) or not np.all(np.isfinite(forward)) or np.linalg.norm(forward) <= 0:
        raise ValueError("website_job_thumbnail_start_view_invalid")
    back = -forward / np.linalg.norm(forward)
    up_world = np.array([0.0, 1.0, 0.0])
    right = np.cross(up_world, back)
    if np.linalg.norm(right) < 1e-6:  # looking straight up or down: any horizontal right will do
        right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(back, right)
    focal = (height / 2.0) / math.tan(math.radians(VERTICAL_FOV_DEGREES) / 2.0)
    return {"position": position, "rotation": np.stack([right, up, back], axis=1),
            "intrinsics": {"width": width, "height": height, "fx": focal, "fy": focal, "cx": width / 2.0, "cy": height / 2.0}}


def _most_opaque(splat: Any, limit: int) -> Any:
    """The splat itself, or its ``limit`` most opaque Gaussians (ties by index), as the preview keeps them."""
    import numpy as np
    from types import SimpleNamespace
    if splat.count <= limit:
        return splat
    keep = np.sort(np.lexsort((np.arange(splat.count), -splat.opacity.astype(np.float64)))[:limit])
    rest = getattr(splat, "sh_rest", None)
    return SimpleNamespace(count=int(limit), xyz=splat.xyz[keep], opacity=splat.opacity[keep], f_dc=splat.f_dc[keep],
                           scales=splat.scales[keep], quats=splat.quats[keep],
                           sh_rest=None if rest is None else rest[keep])


def looks_empty(rgb: Any, alpha: Any) -> bool:
    """Next to nothing landed: too little of the image covered, or too dark on average."""
    import numpy as np
    coverage = float(np.mean(np.asarray(alpha) >= 0.5))
    luminance = float(np.mean(np.asarray(rgb, np.float64) @ np.array([0.2126, 0.7152, 0.0722])))
    return coverage < MIN_COVERAGE or luminance < MIN_MEAN_LUMINANCE


def encode_png(rgb: Any) -> bytes:
    """An RGB PNG of a float image in [0, 1], with no metadata, at most ``MAX_PNG_BYTES`` (a palette if it must)."""
    import numpy as np
    from PIL import Image
    pixels = (np.clip(np.asarray(rgb, np.float64), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    image = Image.fromarray(pixels, mode="RGB")
    for candidate in (image, image.quantize(colors=256).convert("RGB")):
        buffer = io.BytesIO()
        candidate.save(buffer, format="PNG", optimize=True)
        if buffer.tell() <= MAX_PNG_BYTES:
            return buffer.getvalue()
    raise ValueError("website_job_thumbnail_png_over_limit")


def render_png(source: Path, start_view: Mapping[str, Any], *, max_gaussians: int = MAX_GAUSSIANS) -> dict[str, Any]:
    """Render ``source`` (a standard 3DGS PLY) from ``start_view``: the PNG, or ``empty`` when next to nothing shows."""
    from .atlas_splat_render import render
    from .gaussian_splat_decode import read_standard_3dgs_ply
    splat = _most_opaque(read_standard_3dgs_ply(source), max_gaussians)
    rendered = render(splat, camera_for(start_view), WIDTH, HEIGHT)
    if looks_empty(rendered["rgb"], rendered["alpha"]):
        return {"empty": True, "visible_gaussians": rendered["visible_gaussians"]}
    return {"empty": False, "png": encode_png(rendered["rgb"]), "width": WIDTH, "height": HEIGHT,
            "visible_gaussians": rendered["visible_gaussians"], "renderer": rendered["revision"]}


def thumbnail_payload(identity: Mapping[str, str], *, preview_digest: str, png: bytes,
                      width: int, height: int) -> dict[str, Any]:
    """The ``job-thumbnail`` operation's body, exactly as the WebApp reads it."""
    return {"request_id": identity["request_id"], "scene_id": identity["scene_id"], "thumbnail": {
        "schema_version": SCHEMA, "capture_id": identity["capture_id"], "source": SOURCE, "camera": CAMERA,
        "scene_preview_digest": preview_digest, "width": int(width), "height": int(height),
        "png_base64": base64.b64encode(png).decode("ascii"),
        "digest": "sha256:" + hashlib.sha256(png).hexdigest()}}


def _deliver(payload: Mapping[str, Any], *, identity: Mapping[str, str]) -> None:
    from .website_task_context import website_webapp_request
    acceptance = website_webapp_request(capture_id=identity["capture_id"], operation=OPERATION, payload=payload)
    if (acceptance.get("schema_version") != ACCEPTANCE_SCHEMA or acceptance.get("accepted") is not True
            or acceptance.get("digest") != payload["thumbnail"]["digest"]):
        raise ValueError("website_job_thumbnail_acceptance_invalid")


def _delivered_preview(manifest: Mapping[str, Any]) -> str | None:
    """The scene preview's ``preview_digest`` when the WebApp accepted it and its archive stands, else None."""
    built, archive = manifest.get("preview") or {}, manifest.get("archive") or {}
    digest = manifest.get("preview_digest")
    if (manifest.get("schema_version") != MANIFEST_SCHEMA or not isinstance(digest, str)
            or (manifest.get("delivered") or {}).get("preview_digest") != digest
            or archive.get("digest") != built.get("digest") or _withdrawn_from_view(archive)):
        return None
    return digest


def publish_job_thumbnail(capture_root: Path, *, now: Callable[[], float] = time.time,
                          check_rights: Callable[[Mapping[str, str]], None] = _current_rights,
                          render: Callable[..., dict[str, Any]] = render_png,
                          deliver: Callable[..., None] = _deliver) -> dict[str, Any]:
    """One idempotent pass for one capture. Returns ``{"status": ...}``.

    ``current`` (this preview's work is done), ``delivered``, ``skipped_empty``, ``no_start_view``, ``refused``
    (final for a 4xx WebApp answer; a rights doubt before rendering is retried), ``withdrawn``,
    ``no_delivered_preview``, ``gave_up``, ``busy`` or ``failed`` (retried on a later pass).
    """
    root = Path(capture_root)
    identity = _identity(root)
    directory = root.joinpath(*DIRECTORY)
    # Read-only first: nothing here creates a file or a lock for a withdrawn capture or one with no delivered preview.
    if capture_withdrawn(root):
        return {"status": "withdrawn"}
    if _delivered_preview(_read(directory / MANIFEST_FILE)) is None:
        return {"status": "no_delivered_preview"}
    # Withdrawal deletion takes this same lock. It is held from the withdrawal check through rendering, every write
    # and the delivery, as the scene preview's build holds it, so deletion can never finish ahead of a render and have
    # the image or its receipt written again after it.
    try:
        with preview_lock(root):
            return _publish_locked(root, identity, directory, now=now, check_rights=check_rights, render=render,
                                   deliver=deliver)
    except ScenePreviewBusy:
        return {"status": "busy"}


def _publish_locked(root: Path, identity: Mapping[str, str], directory: Path, *, now: Callable[[], float],
                    check_rights: Callable[[Mapping[str, str]], None], render: Callable[..., dict[str, Any]],
                    deliver: Callable[..., None]) -> dict[str, Any]:
    """`publish_job_thumbnail`'s work, with the preview lock held throughout."""
    receipt_path, image_path = directory / RECEIPT_FILE, directory / IMAGE_FILE
    if capture_withdrawn(root):
        return {"status": "withdrawn"}
    manifest = _read(directory / MANIFEST_FILE)
    preview_digest = _delivered_preview(manifest)
    if preview_digest is None:
        return {"status": "no_delivered_preview"}
    receipt = _read(receipt_path)
    same = receipt.get("schema_version") == RECEIPT_SCHEMA and receipt.get("scene_preview_digest") == preview_digest
    if same and receipt.get("outcome") in FINAL:
        return {"status": "current", "outcome": receipt["outcome"]}
    attempts = int(receipt.get("attempts") or 0) if same else 0
    if attempts >= MAX_ATTEMPTS:
        return {"status": "gave_up", "reason": receipt.get("reason")}

    def record(outcome: str, **fields: Any) -> None:
        write_json(receipt_path, {"schema_version": RECEIPT_SCHEMA, **identity, "scene_preview_digest": preview_digest,
                                  "outcome": outcome, "attempts": attempts + (outcome == "failed"),
                                  "at_epoch": int(now()), **fields})

    start_view = (manifest.get("preview") or {}).get("start_view")
    if not isinstance(start_view, Mapping) or "position" not in start_view or "target" not in start_view:
        record("no_start_view")
        return {"status": "no_start_view"}
    source = manifest.get("source") or {}
    inputs = canonical_digest({"builder": BUILDER_REVISION, "source_digest": source.get("digest"),
                               "start_view": dict(start_view), "width": WIDTH, "height": HEIGHT,
                               "fov": VERTICAL_FOV_DEGREES, "max_gaussians": MAX_GAUSSIANS})
    try:
        check_rights(identity)
    except Exception as error:  # noqa: BLE001 - any doubt about rights refuses this pass; nothing is rendered
        return {"status": "refused", "reason": _reason(error)}
    try:
        if not (same and receipt.get("render_inputs_digest") == inputs and image_path.is_file()
                and not image_path.is_symlink() and receipt.get("output_digest") == _sha256_file(image_path)):
            path = root / str(source.get("path") or "")
            if (not source.get("path") or path.is_symlink() or not path.is_file()
                    or not path.resolve().is_relative_to(root.resolve()) or _sha256_file(path) != source.get("digest")):
                raise ValueError("website_job_thumbnail_source_changed")
            made = render(path, start_view)
            if made.get("empty"):
                record("skipped_empty", render_inputs_digest=inputs, visible_gaussians=made.get("visible_gaussians"))
                return {"status": "skipped_empty"}
            image_path.write_bytes(made["png"])
            record("rendered", render_inputs_digest=inputs, output_digest=_sha256_file(image_path),
                   width=made["width"], height=made["height"], visible_gaussians=made.get("visible_gaussians"))
            receipt = _read(receipt_path)
        png = image_path.read_bytes()
        payload = thumbnail_payload(identity, preview_digest=preview_digest, png=png,
                                    width=int(receipt.get("width") or WIDTH), height=int(receipt.get("height") or HEIGHT))
        kept = {key: receipt.get(key) for key in ("render_inputs_digest", "output_digest", "width", "height",
                                                    "visible_gaussians")}
        try:
            deliver(payload, identity=identity)
        except Exception as error:  # noqa: BLE001 - a 4xx answer is final; anything else retries
            reason = _reason(error)
            if _HTTP_CLIENT_REFUSAL.search(str(error)):
                record("refused", reason=reason, **kept)
                return {"status": "refused", "reason": reason}
            record("failed", reason=reason, **kept)
            return {"status": "failed", "reason": reason}
        record("delivered", digest=payload["thumbnail"]["digest"], **kept)
        return {"status": "delivered", "digest": payload["thumbnail"]["digest"]}
    except Exception as error:  # noqa: BLE001 - one capture never stops a pass; retried, at most MAX_ATTEMPTS times
        reason = _reason(error)
        logger.warning("website_job_thumbnail_failed capture=%s type=%s", identity["capture_id"], type(error).__name__)
        record("failed", reason=reason)
        return {"status": "failed", "reason": reason}


def tick(storage_root: Path, *, stop_reason: Callable[[], str | None] | None = None,
         budget_seconds: float = TICK_BUDGET_SECONDS, renders: int = RENDERS_PER_TICK,
         clock: Callable[[], float] = time.monotonic, **options: Any) -> dict[str, int]:
    """One pass over every capture with a scene preview; counts of each outcome.

    It starts no capture after ``budget_seconds``, nor after ``renders`` captures that rendered or delivered, nor
    for a hold, a quiesce or a new release (``stop_reason``); the captures left are counted ``deferred``.
    """
    counts: dict[str, int] = {}
    started, worked = clock(), 0
    captures = capture_roots(storage_root)
    for index, capture in enumerate(captures):
        reason = stop_reason() if stop_reason is not None else None
        if reason or clock() - started >= budget_seconds or worked >= renders:
            counts["deferred"] = len(captures) - index
            counts["stopped_" + (reason or ("time_budget" if worked < renders else "render_budget"))] = 1
            break
        try:
            status = publish_job_thumbnail(capture, **options)["status"]
        except Exception as error:  # noqa: BLE001 - an unreadable capture never stops the others
            logger.warning("website_job_thumbnail_capture_unreadable type=%s", type(error).__name__)
            status = "unreadable"
        # Work is a render or a delivery; a capture refused before rendering, or already current, costs nothing.
        worked += status in {"delivered", "skipped_empty", "failed"}
        counts[status] = counts.get(status, 0) + 1
    return counts
