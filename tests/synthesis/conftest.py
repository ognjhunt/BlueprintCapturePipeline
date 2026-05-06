"""Shared fixtures for synthesis tests.

All fixtures are in-memory — no GCS credentials or real capture bundles needed.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest

# Canonical resolutions used across all tests
DEPTH_H, DEPTH_W = 192, 256       # ARKit native depth resolution
IMAGE_H, IMAGE_W = 1440, 1920     # Full-res video frame
EMBED_DIM = 1024


# ---------------------------------------------------------------------------
# Depth fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_depth_raw() -> np.ndarray:
    """uint16 depth array at ARKit native depth resolution (256×192), in millimetres."""
    rng = np.random.default_rng(42)
    arr = rng.integers(500, 5000, size=(DEPTH_H, DEPTH_W), dtype=np.uint16)
    arr[0, 0] = 0  # one invalid pixel (zero = no reading)
    return arr


@pytest.fixture
def synthetic_depth_m(synthetic_depth_raw: np.ndarray) -> np.ndarray:
    """float32 depth in metres at native depth resolution."""
    out = synthetic_depth_raw.astype(np.float32) * 0.001
    out[synthetic_depth_raw == 0] = 0.0
    return out


@pytest.fixture
def synthetic_depth_png_bytes(synthetic_depth_raw: np.ndarray) -> bytes:
    """In-memory 16-bit grayscale PNG bytes of the synthetic depth map."""
    from PIL import Image as PILImage
    img = PILImage.fromarray(synthetic_depth_raw, mode="I;16")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# RGB frame fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_rgb() -> np.ndarray:
    """uint8 RGB array at full video resolution."""
    rng = np.random.default_rng(7)
    return rng.integers(0, 255, size=(IMAGE_H, IMAGE_W, 3), dtype=np.uint8)


@pytest.fixture
def synthetic_rgb_jpg_bytes(synthetic_rgb: np.ndarray) -> bytes:
    """In-memory JPEG bytes of the synthetic RGB frame."""
    from PIL import Image as PILImage
    img = PILImage.fromarray(synthetic_rgb)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Pose / intrinsics
# ---------------------------------------------------------------------------

@pytest.fixture
def identity_T() -> np.ndarray:
    return np.eye(4, dtype=np.float64)


@pytest.fixture
def simple_intrinsics() -> dict:
    return {
        "fx": 1440.0,
        "fy": 1440.0,
        "cx": float(IMAGE_W) / 2.0,
        "cy": float(IMAGE_H) / 2.0,
        "width": IMAGE_W,
        "height": IMAGE_H,
    }


@pytest.fixture
def translated_T() -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[0, 3] = 1.0   # 1 m offset in X
    return T


# ---------------------------------------------------------------------------
# Site index record
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_index_record(simple_intrinsics: dict) -> dict:
    """A minimal site_reference_index.jsonl record."""
    return {
        "reference_id": "ref-0001",
        "frame_id": "000001",
        "capture_id": "cap-abc",
        "scene_id": "scene-abc",
        "site_id": "site-xyz",
        "pass_id": "pass-001",
        "coordinate_frame_session_id": "session-001",
        "T_world_camera": np.eye(4).tolist(),
        "intrinsics": simple_intrinsics,
        "depth_uri": "gs://bucket/depth/000001.png",
        "frame_uri": "gs://bucket/frames/000001.jpg",
        "embedding_uri": "gs://bucket/embeddings/000001.bin",
        "site_frame_transform": None,
        "quality": {
            "tracking_state": "normal",
            "sharpness_score": 120.0,
            "world_mapping_status": "mapped",
        },
    }


# ---------------------------------------------------------------------------
# Fake local storage tree (simulates GCS mirror)
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_storage_root(
    tmp_path: Path,
    synthetic_depth_png_bytes: bytes,
    synthetic_rgb_jpg_bytes: bytes,
    synthetic_index_record: dict,
) -> Path:
    """
    Writes a minimal fake GCS storage tree that synthesize_view() can resolve:

        {tmp_path}/
          bucket/
            depth/000001.png          ← uint16 depth PNG (256×192)
            frames/000001.jpg         ← full-res JPEG
            embeddings/000001.bin     ← 1024-dim float32 binary
            sites/site-xyz/reference_memory/
              site_reference_index.jsonl
    """
    root = tmp_path / "gcs"

    depth_dir = root / "bucket" / "depth"
    frames_dir = root / "bucket" / "frames"
    embed_dir = root / "bucket" / "embeddings"
    index_dir = root / "bucket" / "sites" / "site-xyz" / "reference_memory"

    for d in (depth_dir, frames_dir, embed_dir, index_dir):
        d.mkdir(parents=True, exist_ok=True)

    (depth_dir / "000001.png").write_bytes(synthetic_depth_png_bytes)
    (frames_dir / "000001.jpg").write_bytes(synthetic_rgb_jpg_bytes)

    embed = np.ones(EMBED_DIM, dtype=np.float32)
    embed /= np.linalg.norm(embed)
    (embed_dir / "000001.bin").write_bytes(embed.tobytes())

    # Patch URIs to match this local tree layout
    rec = dict(synthetic_index_record)
    rec["depth_uri"] = "gs://bucket/depth/000001.png"
    rec["frame_uri"] = "gs://bucket/frames/000001.jpg"
    rec["embedding_uri"] = "gs://bucket/embeddings/000001.bin"

    (index_dir / "site_reference_index.jsonl").write_text(
        json.dumps(rec) + "\n", encoding="utf-8"
    )

    return root
