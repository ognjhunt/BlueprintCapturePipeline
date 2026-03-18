# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    VIDEO_TO_WORLD_REPO_DIR=/opt/video_to_world \
    VIDEO_TO_WORLD_PIPELINE_PRESET=preprocess_plus_alignment \
    VIDEO_TO_WORLD_COMMAND_TEMPLATE="" \
    VIDEO_TO_WORLD_RUNNER_TOKEN="" \
    PORT=8080

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    ffmpeg \
    git \
    git-lfs \
    libgl1 \
    libglib2.0-0 \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/local/bin/python

COPY pyproject.toml ./
COPY src/ src/

RUN python -m pip install ".[cloud,runtime,validation]"

RUN git clone --branch main --single-branch https://github.com/lukasHoel/video_to_world "${VIDEO_TO_WORLD_REPO_DIR}"

RUN python -m pip install "numpy<2" "opencv-python<4.12" "torch>=2" torchvision xformers

RUN git clone https://github.com/ByteDance-Seed/depth-anything-3 "${VIDEO_TO_WORLD_REPO_DIR}/third_party/depth-anything-3" \
    && git -C "${VIDEO_TO_WORLD_REPO_DIR}/third_party/depth-anything-3" checkout 2c21ea849ceec7b469a3e62ea0c0e270afc3281a \
    && git -C "${VIDEO_TO_WORLD_REPO_DIR}/third_party/depth-anything-3" apply "${VIDEO_TO_WORLD_REPO_DIR}/patches/da3-export-trajectory.patch" \
    && python -m pip install -e "${VIDEO_TO_WORLD_REPO_DIR}/third_party/depth-anything-3"

RUN git clone https://github.com/Parskatt/RoMaV2 "${VIDEO_TO_WORLD_REPO_DIR}/third_party/RoMaV2" \
    && git -C "${VIDEO_TO_WORLD_REPO_DIR}/third_party/RoMaV2" apply "${VIDEO_TO_WORLD_REPO_DIR}/patches/romav2-dataclasses.patch" \
    && python -m pip install -e "${VIDEO_TO_WORLD_REPO_DIR}/third_party/RoMaV2[fused-local-corr]"

RUN python -m pip install open3d scipy tyro tqdm tensorboard lpips viser nerfview romatch

ENV PYTHONPATH="/app/src:${PYTHONPATH}"

CMD ["python", "-m", "blueprint_pipeline.video_to_world_runner_service"]
