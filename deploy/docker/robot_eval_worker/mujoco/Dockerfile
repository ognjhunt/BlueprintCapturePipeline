# syntax=docker/dockerfile:1
FROM python:3.12-slim@sha256:423ed6ab25b1921a477529254bfeeabf5855151dc2c3141699a1bfc852199fbf

ARG MUJOCO_MENAGERIE_REF=4c358ef9d9d7f32ca58b40b490884a0c1726a440
ARG APP_UID=10001
ARG APP_GID=10001

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

COPY pyproject.toml README.md LICENSE ./
COPY src/ src/

RUN python -m pip install ".[cloud,validation]" "mujoco>=3.2.0" "trimesh>=4.4.0"

RUN git clone --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git /opt/blueprint/assets/mujoco_menagerie \
    && git -C /opt/blueprint/assets/mujoco_menagerie fetch --depth 1 origin "${MUJOCO_MENAGERIE_REF}" \
    && git -C /opt/blueprint/assets/mujoco_menagerie checkout --detach FETCH_HEAD \
    && test "$(git -C /opt/blueprint/assets/mujoco_menagerie rev-parse HEAD)" = "${MUJOCO_MENAGERIE_REF}" \
    && git -C /opt/blueprint/assets/mujoco_menagerie sparse-checkout set unitree_g1 \
    && test -f /opt/blueprint/assets/mujoco_menagerie/unitree_g1/g1.xml

RUN groupadd --gid "${APP_GID}" blueprint \
    && useradd --uid "${APP_UID}" --gid "${APP_GID}" --create-home --shell /usr/sbin/nologin blueprint \
    && chown -R blueprint:blueprint /app

ENV PYTHONPATH="/app/src"

USER blueprint:blueprint

ENTRYPOINT ["blueprint-run-robot-eval-worker"]
