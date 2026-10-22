"""Blur faces in every frame before it leaves for Atlas (ADP-040, owner decision D5, 2026-10-22).

The scene planner already excludes the frames it judges to show people. This is
the deterministic second screen, on the exact full-resolution pixels Atlas
receives: YuNet (OpenCV Model Zoo, MIT, vendored beside this module) finds faces
and each one is pixelated and blurred beyond recognition. A missing or changed
model, or an OpenCV without ``FaceDetectorYN``, refuses: frames never go
unscreened.
"""
from __future__ import annotations

import math
from pathlib import Path

from PIL import Image

from .local_reconstruction_adapters import _sha256_file

MODEL = Path(__file__).with_name("_atlas_privacy_model") / "face_detection_yunet_2023mar.onnx"
MODEL_SHA256 = "sha256:8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
# Recall over precision: a blurred patch of wall costs little, a missed face does not.
SCORE_THRESHOLD = 0.5
FACE_MARGIN = 0.35
MAXIMUM_DETECTION_SIDE = 4096


def screen_identity() -> dict:
    return {"method": "opencv_yunet_face_blur", "model_sha256": MODEL_SHA256,
            "score_threshold": SCORE_THRESHOLD, "face_margin": FACE_MARGIN}


def blur_faces(image: Image.Image) -> tuple[Image.Image, list[list[int]]]:
    """The same image with every detected face unrecognisable, and the boxes blurred."""
    import cv2
    import numpy as np

    if not hasattr(cv2, "FaceDetectorYN") or not MODEL.is_file() or _sha256_file(MODEL) != MODEL_SHA256:
        raise ValueError("atlas_privacy_screen_unavailable")
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
