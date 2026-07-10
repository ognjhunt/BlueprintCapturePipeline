# syntax=docker/dockerfile:1
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04@sha256:38e59267704b5d91ef63c7d8f613359c629fab0aead1283d59ca7821029e73bc

ARG DEEPPRIVACY2_SOURCE_REF=f4d8f09d1eb8f758c89cf1795ae41f20533942d3
ARG APP_UID=10001
ARG APP_GID=10001

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    TORCH_HOME=/models/deepprivacy2 \
    DEEPPRIVACY2_MODEL_PATH=/models/deepprivacy2 \
    DEEPPRIVACY2_REPO_DIR=/opt/deepprivacy2 \
    PRIVACY_RUNNER_KIND=deepprivacy2

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    ffmpeg \
    git \
    git-lfs \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/local/bin/python

COPY pyproject.toml README.md LICENSE ./
COPY src/ src/

RUN python -m pip install \
    --upgrade \
    "pip<25" \
    "setuptools<60" \
    wheel

RUN python -m pip install \
    --extra-index-url https://download.pytorch.org/whl/cu117 \
    "torch==1.13.1+cu117" \
    "torchvision==0.14.1+cu117"

RUN git clone --filter=blob:none https://github.com/hukkelas/deep_privacy2.git "${DEEPPRIVACY2_REPO_DIR}" \
    && git -C "${DEEPPRIVACY2_REPO_DIR}" fetch --depth 1 origin "${DEEPPRIVACY2_SOURCE_REF}" \
    && git -C "${DEEPPRIVACY2_REPO_DIR}" checkout --detach FETCH_HEAD \
    && test "$(git -C "${DEEPPRIVACY2_REPO_DIR}" rev-parse HEAD)" = "${DEEPPRIVACY2_SOURCE_REF}"

RUN python -c "from pathlib import Path; setup_py = Path('/opt/deepprivacy2/setup.py'); text = setup_py.read_text(encoding='utf-8'); text = text.replace('\"pillow==8.3.1\"', '\"pillow>=10.0.0\"'); text = text.replace('\"scipy==1.7.1\"', '\"scipy>=1.10.1,<1.11\"'); setup_py.write_text(text, encoding='utf-8')"

RUN python -m pip install ".[cloud,validation]" \
    && python -m pip install "${DEEPPRIVACY2_REPO_DIR}"

RUN groupadd --gid "${APP_GID}" blueprint \
    && useradd --uid "${APP_UID}" --gid "${APP_GID}" --create-home --shell /usr/sbin/nologin blueprint \
    && mkdir -p /models/deepprivacy2 \
    && chown -R blueprint:blueprint /app /models/deepprivacy2

ENV PYTHONPATH="/app/src:${PYTHONPATH}"

USER blueprint:blueprint

CMD ["python", "-m", "blueprint_pipeline.privacy_runner_service"]
