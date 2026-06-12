# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    BLUEPRINT_WORKER_IMAGE_FAMILY=mujoco-eval-worker \
    MUJOCO_GL=egl

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    ffmpeg \
    git \
    libegl1 \
    libgl1 \
    libglib2.0-0 \
    libosmesa6 \
    libxext6 \
    libxrender1 \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src/ src/

RUN python -m pip install ".[cloud,validation]" "mujoco>=3.2.0"

ENV PYTHONPATH="/app/src:${PYTHONPATH}"

ENTRYPOINT ["blueprint-run-robot-eval-worker"]
