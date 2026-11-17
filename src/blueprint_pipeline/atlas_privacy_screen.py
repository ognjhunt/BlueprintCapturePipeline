"""Blur faces in every frame before it leaves for Atlas (ADP-040, owner decision D5, 2026-10-22).

The scene planner already excludes the frames it judges to show people. This is
the deterministic second screen, on the exact full-resolution pixels Atlas
receives: YuNet (OpenCV Model Zoo, MIT, vendored beside this module) finds faces
and each one is pixelated and blurred beyond recognition. A missing or changed
model, or an OpenCV without ``FaceDetectorYN``, refuses: frames never go
unscreened.

Owner decision 2026-11-17: the Privacy Policy and Terms versions published that
day no longer promise face blurring. Footage a site sent under such a version is
not blurred; footage sent under any earlier version is still blurred whenever it
is processed. ``face_blur_required`` is the one place that decides, and it
answers "not required" only for a sealed website grant naming a version in
``NO_FACE_BLUR_PRIVACY_VERSIONS``; every other case blurs.
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from PIL import Image

from .local_reconstruction_adapters import _sha256_file

MODEL = Path(__file__).with_name("_atlas_privacy_model") / "face_detection_yunet_2023mar.onnx"
MODEL_SHA256 = "sha256:8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
# Recall over precision: a blurred patch of wall costs little, a missed face does not.
SCORE_THRESHOLD = 0.5
FACE_MARGIN = 0.35
MAXIMUM_DETECTION_SIDE = 4096

#: Privacy Policy versions whose footage is not face-blurred (owner decision 2026-11-17).
#: The website publishes exactly one of these two, named for the UTC day its change
#: deploys; the other will never appear in a sealed grant. A later Privacy Policy
#: version must be added here deliberately when it is written, or its footage is blurred.
NO_FACE_BLUR_PRIVACY_VERSIONS = frozenset({"2026-11-17.1", "2026-11-18.1"})
NO_FACE_BLUR_METHOD = "no_face_blur"
#: The grant the capture's delivery retained beside its pipeline (``website_task_evidence``).
GRANT_FILE = "website_scene_sponsorship.json"
_MAX_GRANT_BYTES = 64 * 1024


def screen_identity() -> dict:
    return {"method": "opencv_yunet_face_blur", "model_sha256": MODEL_SHA256,
            "score_threshold": SCORE_THRESHOLD, "face_margin": FACE_MARGIN}


def no_blur_identity(version: str) -> dict:
    """What every record names when frames go out unblurred, and why."""
    return {"method": NO_FACE_BLUR_METHOD, "privacy_policy_version": version}


def _listed_version(output_root: Path) -> str | None:
    """The grant's Privacy version when it is sealed, this capture's own, and listed; else None."""
    root = Path(output_root).resolve()
    pipeline = next((path for path in (root, *root.parents) if path.name == "pipeline"), None)
    if pipeline is None:
        return None  # not a capture's pipeline: an internal or Blueprint-owned path
    capture = pipeline.parent
    if capture.parent.name != "captures":
        return None
    path = pipeline / GRANT_FILE
    if path.is_symlink() or not path.is_file() or path.stat().st_size > _MAX_GRANT_BYTES:
        return None
    grant = json.loads(path.read_text(encoding="utf-8"))
    from .decision_evidence_contracts import canonical_digest
    if (not isinstance(grant, dict)
            or grant.get("schema_version") != "website_scene_sponsorship.v1"
            or grant.get("sponsor") != "blueprint"
            or grant.get("purpose") != "scene_preparation"
            or not isinstance(grant.get("authority_digest"), str)
            or grant["authority_digest"] != canonical_digest(grant, digest_field="authority_digest")
            or not isinstance(grant.get("request_id"), str) or not grant["request_id"].strip()
            or grant.get("capture_id") != capture.name
            or grant.get("scene_id") != capture.parent.parent.name):
        return None
    version = grant.get("privacy_policy_version")
    return version if isinstance(version, str) and version in NO_FACE_BLUR_PRIVACY_VERSIONS else None


def face_blur_required(output_root: Path) -> tuple[bool, dict]:
    """Whether frames prepared under ``output_root`` must be face-blurred, and the identity to record.

    Not required only when the capture's retained website grant (``<capture>/pipeline/
    website_scene_sponsorship.json``, the nearest ``pipeline`` above the root) is sealed by its
    ``authority_digest``, has the grant's shape, names this capture and scene, and carries a
    ``privacy_policy_version`` in ``NO_FACE_BLUR_PRIVACY_VERSIONS``. No grant, an unreadable or
    malformed one, a failed digest, a missing, unknown, earlier or unlisted later version, a
    capture that is not a website capture, or any error while deciding: required.
    """
    try:
        version = _listed_version(output_root)
    except Exception:  # noqa: BLE001 - any doubt blurs
        version = None
    return (True, screen_identity()) if version is None else (False, no_blur_identity(version))


def is_no_blur_identity(record: Any) -> bool:
    return isinstance(record, Mapping) and record.get("method") == NO_FACE_BLUR_METHOD


def screen_for(output_root: Path, retained: Any = None) -> tuple[bool, dict]:
    """The decision for new frames under ``output_root``, kept consistent with retained work.

    ``retained`` is the privacy record of inputs already prepared there. Retained blurred
    inputs (any record that is not the no-blur identity) keep the job blurring, so a job in
    flight is never invalidated. Retained unblurred inputs never lower the decision: a job
    that now requires blurring still blurs, and the retained-record checks refuse the mismatch.
    """
    required, identity = face_blur_required(output_root)
    if not required and retained is not None and not is_no_blur_identity(retained):
        return True, screen_identity()
    return required, identity


def blur_faces(image: Image.Image) -> tuple[Image.Image, list[list[int]]]:
    """The same image with every detected face unrecognisable, and the boxes blurred."""
    import cv2
    import numpy as np

    if not hasattr(cv2, "FaceDetectorYN") or not MODEL.is_file() or _sha256_file(MODEL) != MODEL_SHA256:
        from .hard_stops import PrivacyRightsStop
        raise PrivacyRightsStop("atlas_privacy_screen_unavailable")
    pixels = np.array(image.convert("RGB"))
    height, width = pixels.shape[:2]
    scale = min(1.0, MAXIMUM_DETECTION_SIDE / max(height, width))
    work = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
    if scale < 1:
        work = cv2.resize(work, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    detector = cv2.FaceDetectorYN.create(str(MODEL), "", (work.shape[1], work.shape[0]), SCORE_THRESHOLD, 0.3, 5000)
    _, faces = detector.detect(work)
    boxes = []
    for face in [] if faces is None else faces:
        x, y, w, h = (float(value) / scale for value in face[:4])
        x0, y0 = max(0, math.floor(x - FACE_MARGIN * w)), max(0, math.floor(y - FACE_MARGIN * h))
        x1 = min(width, math.ceil(x + w + FACE_MARGIN * w))
        y1 = min(height, math.ceil(y + h + FACE_MARGIN * h))
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        coarse = cv2.resize(pixels[y0:y1, x0:x1], (max(1, (x1 - x0) // 12), max(1, (y1 - y0) // 12)),
                            interpolation=cv2.INTER_AREA)
        smooth = cv2.resize(coarse, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
        pixels[y0:y1, x0:x1] = cv2.GaussianBlur(smooth, (0, 0), max(2.0, (x1 - x0) / 10))
        boxes.append([x0, y0, x1, y1])
    return Image.fromarray(pixels), boxes
