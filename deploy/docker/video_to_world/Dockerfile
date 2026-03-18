# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    VIDEO_TO_WORLD_REPO_DIR=/opt/video_to_world \
    VIDEO_TO_WORLD_COMMAND_TEMPLATE="cd /opt/video_to_world && python run_reconstruction.py --config.input-video {INPUT_VIDEO} --config.scene-root {SCENE_ROOT} --config.mode fast" \
    VIDEO_TO_WORLD_RUNNER_TOKEN="" \
    PORT=8080

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    ffmpeg \
    git \
    git-lfs \
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

ENV PYTHONPATH="/app/src:${PYTHONPATH}"

CMD ["python", "-m", "blueprint_pipeline.video_to_world_runner_service"]
