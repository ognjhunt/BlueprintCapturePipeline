# syntax=docker/dockerfile:1
ARG ISAAC_SIM_BASE_IMAGE=nvcr.io/nvidia/isaac-sim:6.0.0@sha256:68735a60b6c15c85e0dd0098570c6d2cc79e928f2d068ce2790aa43284ac165d
FROM ${ISAAC_SIM_BASE_IMAGE}

USER root

ARG APP_UID=10001
ARG APP_GID=10001
ARG BLUEPRINT_SOURCE_COMMIT
ARG BLUEPRINT_SOURCE_DIRTY_PATCH_SHA256

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    BLUEPRINT_WORKER_IMAGE_FAMILY=isaac-eval-worker \
    BLUEPRINT_SIMULATOR_FRAMEWORK=isaac_sim \
    BLUEPRINT_ISAAC_PYTHON=/isaac-sim/python.sh \
    ISAACSIM_ASSET_ROOT=https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0 \
    BLUEPRINT_ISAAC_UNITREE_G1_USD=Isaac/Robots/Unitree/G1/g1.usd \
    BLUEPRINT_SOURCE_COMMIT=${BLUEPRINT_SOURCE_COMMIT} \
    BLUEPRINT_SOURCE_DIRTY_PATCH_SHA256=${BLUEPRINT_SOURCE_DIRTY_PATCH_SHA256} \
    BLUEPRINT_ISAAC_SIM_MAJOR_VERSION=6 \
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

RUN python3 -m pip install ".[cloud,validation]" \
    && rm -rf /root/.cache/pip /tmp/pip-*

# Build-time content contract. GPU/CUDA/RTX checks run in the paid startup canary, where a real
# NVIDIA device is present.
RUN mkdir -p /opt/blueprint \
    && python3 -m blueprint_pipeline.isaac_worker_image_healthcheck \
      --build-time --output /opt/blueprint/image_build_healthcheck.json
RUN PYTHONPATH=/app/src /isaac-sim/python.sh -c \
    "import blueprint_pipeline; print(blueprint_pipeline.__name__)"

RUN groupadd --gid "${APP_GID}" blueprint \
    && useradd --uid "${APP_UID}" --gid "${APP_GID}" --create-home --shell /usr/sbin/nologin blueprint \
    && usermod -aG isaac-sim blueprint \
    && mkdir -p /workspace/outputs /tmp/blueprint_pipeline \
    && mkdir -p /isaac-sim/kit/cache /isaac-sim/kit/data /isaac-sim/kit/logs \
    && chown blueprint:isaac-sim /isaac-sim/kit/cache /isaac-sim/kit/data /isaac-sim/kit/logs \
    && chmod 0775 /isaac-sim/kit/cache /isaac-sim/kit/data /isaac-sim/kit/logs \
    && chown -R blueprint:blueprint /workspace /tmp/blueprint_pipeline
RUN runuser -u blueprint -- test -r /isaac-sim \
    && runuser -u blueprint -- test -w /isaac-sim/kit/cache \
    && runuser -u blueprint -- test -w /isaac-sim/kit/data \
    && runuser -u blueprint -- test -w /isaac-sim/kit/logs \
    && runuser -u blueprint -- env PYTHONPATH=/app/src /isaac-sim/python.sh -c \
    "from blueprint_pipeline.nvidia_warehouse_native_camera_canary import import_simulation_app; assert callable(import_simulation_app())"

ENV PYTHONPATH="/app/src" \
    HOME=/home/blueprint

USER blueprint
WORKDIR /workspace

ENTRYPOINT ["blueprint-run-robot-eval-worker"]
