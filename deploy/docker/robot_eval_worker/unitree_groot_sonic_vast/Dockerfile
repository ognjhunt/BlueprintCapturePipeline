FROM nvidia/cuda:13.3.1-runtime-ubuntu22.04@sha256:6c8e1c94bd76b9a19f5713388da6ef290a734bcc6e96757e8b5b3f32dba9a904

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
