# syntax=docker/dockerfile:1
FROM python:3.11-slim

ARG MUJOCO_MENAGERIE_REF=4c358ef9d9d7f32ca58b40b490884a0c1726a440

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    BLUEPRINT_WORKER_IMAGE_FAMILY=mujoco-eval-worker \
    BLUEPRINT_MUJOCO_G1_MODEL_ROOT=/opt/blueprint/assets/mujoco_menagerie/unitree_g1 \
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

RUN python -m pip install ".[cloud,validation]" "mujoco>=3.2.0" "trimesh>=4.4.0"

RUN git clone --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git /opt/blueprint/assets/mujoco_menagerie \
    && git -C /opt/blueprint/assets/mujoco_menagerie checkout "${MUJOCO_MENAGERIE_REF}" \
    && git -C /opt/blueprint/assets/mujoco_menagerie sparse-checkout set unitree_g1 \
    && test -f /opt/blueprint/assets/mujoco_menagerie/unitree_g1/g1.xml

ENV PYTHONPATH="/app/src"

ENTRYPOINT ["blueprint-run-robot-eval-worker"]
