"""Frames sent to Atlas are face-screened on their exact pixels, or not sent (owner decision D5)."""
import numpy as np
import pytest
from PIL import Image

from blueprint_pipeline import atlas_privacy_screen as screen


def _image(width=320, height=240):
    rng = np.random.default_rng(7)
    return Image.fromarray(rng.integers(0, 255, (height, width, 3), dtype=np.uint8))


def test_vendored_model_loads_under_the_installed_opencv_and_leaves_a_faceless_frame_unchanged():
    image = _image()
    screened, boxes = screen.blur_faces(image)
    assert boxes == []
    assert np.array_equal(np.asarray(screened), np.asarray(image))
    assert screen.screen_identity()["model_sha256"] == screen.MODEL_SHA256


def test_every_detected_face_is_obscured_with_margin_and_nothing_else_changes(monkeypatch):
    import cv2

    class Detector:
        def detect(self, work):
            assert work.shape[:2] == (240, 320)
            return 1, np.array([[100, 80, 40, 50] + [0] * 10 + [0.9]], dtype=np.float32)

    monkeypatch.setattr(cv2.FaceDetectorYN, "create", staticmethod(lambda *a, **k: Detector()))
    image = _image()
    screened, boxes = screen.blur_faces(image)
    before, after = np.asarray(image).astype(int), np.asarray(screened).astype(int)
    assert boxes == [[86, 62, 154, 148]]
    x0, y0, x1, y1 = boxes[0]
    outside = np.ones(before.shape[:2], dtype=bool)
    outside[y0:y1, x0:x1] = False
    assert np.array_equal(before[outside], after[outside])
    # Noise texture inside the face box is gone: the region is smooth.
    assert np.abs(np.diff(after[y0:y1, x0:x1], axis=1)).mean() < np.abs(np.diff(before[y0:y1, x0:x1], axis=1)).mean() / 4


def test_a_changed_model_or_missing_detector_refuses_instead_of_sending_unscreened_frames(monkeypatch):
    monkeypatch.setattr(screen, "MODEL_SHA256", "sha256:" + "0" * 64)
    with pytest.raises(ValueError, match="atlas_privacy_screen_unavailable"):
        screen.blur_faces(_image())
    monkeypatch.undo()
    import cv2
    monkeypatch.delattr(cv2, "FaceDetectorYN")
    with pytest.raises(ValueError, match="atlas_privacy_screen_unavailable"):
        screen.blur_faces(_image())
