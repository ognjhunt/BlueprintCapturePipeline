FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04@sha256:517da2300c184c9999ec203c2665244bdebd3578d12fcc7065e83667932643d9

ARG APP_UID=10001
ARG APP_GID=10001

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        python3 \
        python3-pip \
        python3-venv \
        unzip \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install \
        msgpack \
        msgpack-numpy \
        numpy \
        pillow \
        pyzmq

WORKDIR /workspace

RUN groupadd --gid "${APP_GID}" blueprint \
    && useradd --uid "${APP_UID}" --gid "${APP_GID}" --create-home --shell /usr/sbin/nologin blueprint \
    && chown -R blueprint:blueprint /workspace

USER blueprint:blueprint
