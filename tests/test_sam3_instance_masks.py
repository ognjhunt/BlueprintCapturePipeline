"""Tests for SAM3 instance mask saving (--save-instance-masks flag).

We cannot directly ``import sam3_detect`` because it requires ``torch`` at
module level.  Instead we re-implement the pure-Python/NumPy helper
``_accumulate_instance_mask`` inline (identical logic) so that the tests
verify the algorithm without GPU-heavy dependencies.

The source of truth is ``scripts/sam3_detect.py::_accumulate_instance_mask``.
If that function's signature or semantics change, these tests must be
updated to match.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# We also verify that the source file contains the function.
_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"


def _has_pil() -> bool:
    try:
        import PIL  # noqa: F401
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Local re-implementation of the helper (same logic as sam3_detect.py)
# ---------------------------------------------------------------------------

def _accumulate_instance_mask(
    mask_np,
    object_id: int,
    frame_name: str,
    instance_masks_dir: Path,
) -> None:
    """Accumulate per-object boolean mask into per-view instance segmentation PNG."""
    if mask_np is None or not mask_np.any():
        return
    try:
        out_path = instance_masks_dir / f"{frame_name}.png"
        if out_path.exists():
            from PIL import Image as _PIL
            existing = np.array(_PIL.open(out_path), dtype=np.uint16)
        else:
            existing = np.zeros(mask_np.shape[:2], dtype=np.uint16)

        existing[mask_np] = min(max(int(object_id), 0), 65535)

        from PIL import Image as _PIL
        _PIL.fromarray(existing.astype(np.uint16), mode="I;16").save(out_path)
    except Exception as exc:
        print(f"    Instance mask write failed for {frame_name} obj {object_id}: {exc}")


# ---------------------------------------------------------------------------
# Source-code guard: ensure the real function exists in sam3_detect.py
# ---------------------------------------------------------------------------

class TestSourceCodeGuard:
    def test_accumulate_instance_mask_exists_in_sam3_detect(self) -> None:
        src = (_SCRIPTS_DIR / "sam3_detect.py").read_text(encoding="utf-8")
        assert "def _accumulate_instance_mask(" in src

    def test_save_instance_masks_cli_arg_exists(self) -> None:
        src = (_SCRIPTS_DIR / "sam3_detect.py").read_text(encoding="utf-8")
        assert "--save-instance-masks" in src

    def test_instance_masks_dir_created_in_video_predictor(self) -> None:
        src = (_SCRIPTS_DIR / "sam3_detect.py").read_text(encoding="utf-8")
        assert "instance_masks" in src
        assert "save_instance_masks" in src

    def test_force_full_video_masks_flag_exists(self) -> None:
        src = (_SCRIPTS_DIR / "sam3_detect.py").read_text(encoding="utf-8")
        assert "--force-full-video-masks" in src


# ---------------------------------------------------------------------------
# TestAccumulateInstanceMask
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _has_pil(), reason="Pillow not available")
class TestAccumulateInstanceMask:
    """Test the _accumulate_instance_mask helper."""

    def test_creates_new_mask_png(self, tmp_path: Path) -> None:
        """First call for a frame should create a new PNG."""
        mask = np.zeros((480, 640), dtype=bool)
        mask[100:200, 100:200] = True

        _accumulate_instance_mask(mask, object_id=1, frame_name="frame_0001", instance_masks_dir=tmp_path)

        out_path = tmp_path / "frame_0001.png"
        assert out_path.is_file()

        from PIL import Image
        img = np.array(Image.open(out_path))
        assert img.shape == (480, 640)
        assert img[150, 150] == 1
        assert img[0, 0] == 0

    def test_compositing_overlap_order(self, tmp_path: Path) -> None:
        """Later objects should overwrite earlier ones at overlapping pixels."""
        # Object 1: covers 100:200, 100:200
        mask1 = np.zeros((480, 640), dtype=bool)
        mask1[100:200, 100:200] = True
        _accumulate_instance_mask(mask1, object_id=1, frame_name="frame_0001", instance_masks_dir=tmp_path)

        # Object 2: overlaps at 150:250, 150:250
        mask2 = np.zeros((480, 640), dtype=bool)
        mask2[150:250, 150:250] = True
        _accumulate_instance_mask(mask2, object_id=2, frame_name="frame_0001", instance_masks_dir=tmp_path)

        from PIL import Image
        img = np.array(Image.open(tmp_path / "frame_0001.png"))

        # Non-overlapping region of obj1
        assert img[110, 110] == 1
        # Overlapping region — obj2 wins (later overwrite)
        assert img[175, 175] == 2
        # Non-overlapping region of obj2
        assert img[220, 220] == 2
        # Background
        assert img[0, 0] == 0

    def test_skips_empty_mask(self, tmp_path: Path) -> None:
        """Empty (all-False) mask should not create a file."""
        mask = np.zeros((480, 640), dtype=bool)
        _accumulate_instance_mask(mask, object_id=1, frame_name="frame_0001", instance_masks_dir=tmp_path)
        assert not (tmp_path / "frame_0001.png").exists()

    def test_skips_none_mask(self, tmp_path: Path) -> None:
        """None mask should not create a file."""
        _accumulate_instance_mask(None, object_id=1, frame_name="frame_0001", instance_masks_dir=tmp_path)
        assert not (tmp_path / "frame_0001.png").exists()

    def test_clamps_object_id_to_65535(self, tmp_path: Path) -> None:
        """Object IDs above 65535 should be clamped (uint16 range)."""
        mask = np.zeros((100, 100), dtype=bool)
        mask[10:20, 10:20] = True
        _accumulate_instance_mask(mask, object_id=999999, frame_name="frame_0001", instance_masks_dir=tmp_path)

        from PIL import Image
        img = np.array(Image.open(tmp_path / "frame_0001.png"))
        assert img[15, 15] == 65535

    def test_multiple_frames_independent(self, tmp_path: Path) -> None:
        """Different frames should have independent mask files."""
        mask1 = np.zeros((100, 100), dtype=bool)
        mask1[10:20, 10:20] = True

        mask2 = np.zeros((100, 100), dtype=bool)
        mask2[50:60, 50:60] = True

        _accumulate_instance_mask(mask1, object_id=1, frame_name="frame_0001", instance_masks_dir=tmp_path)
        _accumulate_instance_mask(mask2, object_id=2, frame_name="frame_0002", instance_masks_dir=tmp_path)

        assert (tmp_path / "frame_0001.png").is_file()
        assert (tmp_path / "frame_0002.png").is_file()

        from PIL import Image
        img1 = np.array(Image.open(tmp_path / "frame_0001.png"))
        img2 = np.array(Image.open(tmp_path / "frame_0002.png"))
        # frame_0001 has object 1, not object 2
        assert img1[15, 15] == 1
        assert img1[55, 55] == 0
        # frame_0002 has object 2, not object 1
        assert img2[55, 55] == 2
        assert img2[15, 15] == 0

    def test_three_objects_composited(self, tmp_path: Path) -> None:
        """Three objects layered on same frame — highest ID wins at overlaps."""
        for oid in (1, 2, 3):
            mask = np.zeros((100, 100), dtype=bool)
            # All three overlap at center
            mask[40:60, 40:60] = True
            _accumulate_instance_mask(mask, object_id=oid, frame_name="frame_0001", instance_masks_dir=tmp_path)

        from PIL import Image
        img = np.array(Image.open(tmp_path / "frame_0001.png"))
        # Object 3 was written last → owns the overlapping pixels
        assert img[50, 50] == 3
