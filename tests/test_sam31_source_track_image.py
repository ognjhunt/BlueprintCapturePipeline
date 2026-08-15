from pathlib import Path

from blueprint_pipeline.sam31_gpu_admission import OFFICIAL_CODE_REVISION


ROOT = Path(__file__).resolve().parents[1]
IMAGE_ROOT = ROOT / "deploy" / "docker" / "sam31_source_tracks"


def test_sam31_source_track_image_is_separate_pinned_and_nonroot() -> None:
    dockerfile = (IMAGE_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert (
        "FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04@sha256:"
        in dockerfile
    )
    assert f"ARG SAM31_CODE_REVISION={OFFICIAL_CODE_REVISION}" in dockerfile
    assert "torch==2.10.0" in dockerfile
    assert "torchvision==0.25.0" in dockerfile
    assert "facebookresearch/sam3.git@${SAM31_CODE_REVISION}" in dockerfile
    assert '".[cloud,validation]"' in dockerfile
    assert '".[cloud,runtime,validation]"' not in dockerfile
    assert "einops==0.8.2" in dockerfile
    assert "numpy==1.26.4" in dockerfile
    assert "opencv-python" not in dockerfile
    assert "psutil==7.2.2" in dockerfile
    assert "pycocotools==2.0.10" in dockerfile
    assert "scipy==1.16.2" in dockerfile
    assert "python -m pip check" in dockerfile
    assert (
        "from sam3.model_builder import build_sam3_multiplex_video_predictor; "
        "assert callable(build_sam3_multiplex_video_predictor)"
    ) in dockerfile
    assert "sam3.1_multiplex.pt" not in dockerfile
    assert "COPY ." not in dockerfile
    assert "USER blueprint:blueprint" in dockerfile
    assert "privacy_runner_service" not in dockerfile
    assert ":latest" not in dockerfile


def test_sam31_source_track_image_readme_preserves_claim_boundary() -> None:
    readme = (IMAGE_ROOT / "README.md").read_text(encoding="utf-8")
    normalized = " ".join(readme.split())
    assert "not baked into the image" in normalized
    assert "verify the exact configured SHA-256 digest" in normalized
    assert "unset" in normalized
    assert "2D source-frame track evidence" in normalized
    assert "does not establish" in normalized
    assert "comparative policy ranking" in normalized
