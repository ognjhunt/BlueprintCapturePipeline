# syntax=docker/dockerfile:1
ARG ISAAC_SIM_BASE_IMAGE=nvcr.io/nvidia/isaac-sim:6.0.0
FROM ${ISAAC_SIM_BASE_IMAGE}

USER root

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

COPY pyproject.toml ./
COPY src/ src/

RUN python3 -m pip install ".[cloud,validation]"

ENV PYTHONPATH="/app/src:${PYTHONPATH}"

USER 1234:1234

ENTRYPOINT ["blueprint-run-robot-eval-worker"]
