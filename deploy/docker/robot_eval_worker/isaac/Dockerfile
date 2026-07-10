# syntax=docker/dockerfile:1
ARG ISAAC_SIM_BASE_IMAGE=nvcr.io/nvidia/isaac-sim:6.0.0@sha256:68735a60b6c15c85e0dd0098570c6d2cc79e928f2d068ce2790aa43284ac165d
FROM ${ISAAC_SIM_BASE_IMAGE}

USER root

ARG APP_UID=10001
ARG APP_GID=10001

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    BLUEPRINT_WORKER_IMAGE_FAMILY=isaac-eval-worker \
    BLUEPRINT_SIMULATOR_FRAMEWORK=isaac_sim \
    BLUEPRINT_ISAAC_PYTHON=/isaac-sim/python.sh \
    BLUEPRINT_ISAAC_UNITREE_G1_USD=/isaac-sim/Isaac/Robots/Unitree/G1/g1.usd \
    NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility,video

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    python3-pip \
    vulkan-tools \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY src/ src/

RUN python3 -m pip install ".[cloud,validation]"

# Build-time content contract. GPU/CUDA/RTX checks run in the paid startup canary, where a real
# NVIDIA device is present.
RUN python3 -m blueprint_pipeline.isaac_worker_image_healthcheck --build-time

RUN groupadd --gid "${APP_GID}" blueprint \
    && useradd --uid "${APP_UID}" --gid "${APP_GID}" --create-home --shell /usr/sbin/nologin blueprint \
    && mkdir -p /workspace/outputs /tmp/blueprint_pipeline \
    && chown -R blueprint:blueprint /workspace /tmp/blueprint_pipeline

ENV PYTHONPATH="/app/src:${PYTHONPATH}" \
    HOME=/home/blueprint

USER blueprint:blueprint
WORKDIR /workspace

ENTRYPOINT ["blueprint-run-robot-eval-worker"]
