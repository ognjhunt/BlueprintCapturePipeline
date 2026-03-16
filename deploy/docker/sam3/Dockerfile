# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    HF_HOME=/models/hf \
    SAM3_WEIGHTS_PATH=/models/sam3/sam3.pt \
    PRIVACY_RUNNER_KIND=sam3

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
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

RUN python -m pip install \
    --extra-index-url https://download.pytorch.org/whl/cu126 \
    torch==2.7.0 \
    torchvision==0.22.0

RUN python -m pip install \
    "huggingface_hub[cli]" \
    "git+https://github.com/facebookresearch/sam3.git@f6e51f59500a87c576c2df2323ce56b9fd7a12de"

RUN python -m pip install ".[cloud,validation]"

RUN mkdir -p /models/hf /models/sam3

ENV PYTHONPATH="/app/src:${PYTHONPATH}"

CMD ["python", "-m", "blueprint_pipeline.privacy_runner_service"]
