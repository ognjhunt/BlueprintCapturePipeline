# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04@sha256:ac55d124da4882b497f732d8dfd9a702d5447a5f29d08d56da6f64f0a1eb34bc

ARG APP_UID=10001
ARG APP_GID=10001
ARG SAM31_CODE_REVISION=96914d2425f90a64f45ca977c2b5165418099543

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    HF_HOME=/models/hf \
    BLUEPRINT_SAM31_MODEL_ROOT=/models/sam31

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    ffmpeg \
    git \
    libgl1 \
    libglib2.0-0 \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/local/bin/python

COPY pyproject.toml README.md LICENSE ./
COPY src/ src/

RUN python -m pip install \
    --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.10.0 \
    torchvision==0.25.0

RUN python -m pip install \
    "git+https://github.com/facebookresearch/sam3.git@${SAM31_CODE_REVISION}"

RUN python -m pip install ".[cloud,runtime,validation]"

RUN groupadd --gid "${APP_GID}" blueprint \
    && useradd --uid "${APP_UID}" --gid "${APP_GID}" --create-home --shell /usr/sbin/nologin blueprint \
    && mkdir -p /models/hf /models/sam31 /work \
    && chown -R blueprint:blueprint /app /models/hf /models/sam31 /work

ENV PYTHONPATH="/app/src:${PYTHONPATH}"

USER blueprint:blueprint

CMD ["python", "-m", "blueprint_pipeline.sam31_source_track_canary_worker", "--help"]
