# syntax=docker/dockerfile:1.7
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04@sha256:5645fec64549cc35930eee9d85aafd2b0006c0c3f22632be5a1d85e2604e9749

ARG DEBIAN_FRONTEND=noninteractive
ARG BLUEPRINT_SOURCE_COMMIT
ARG BLUEPRINT_CONTAINER_IMAGE_DIGEST
ARG FFMPEG_VERSION=6.1.1
ARG COLMAP_REVISION=a0d785fba74b2664f31edc4a29026a8b27c00f67
ARG GSPLAT_REVISION=937e29912570c372bed6747a5c9bf85fed877bae
ARG THREEDGRUT_REVISION=0a5832248698ab8456b181d6ea17fe02eda58637

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    BLUEPRINT_WORKER_IMAGE_FAMILY=blueprint-reconstruction-worker \
    BLUEPRINT_SOURCE_COMMIT=${BLUEPRINT_SOURCE_COMMIT} \
    BLUEPRINT_CONTAINER_IMAGE_DIGEST=${BLUEPRINT_CONTAINER_IMAGE_DIGEST} \
    BLUEPRINT_RECONSTRUCTION_MODEL_ROOT=/opt/models/colmap \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential ca-certificates cmake curl git libboost-filesystem-dev \
      libboost-graph-dev libboost-program-options-dev libceres-dev libcgal-dev \
      libeigen3-dev libfreeimage-dev libglew-dev libgoogle-glog-dev libmetis-dev \
      libsqlite3-dev nasm ninja-build pkg-config python3.11 python3.11-dev \
      python3-pip python3-venv yasm \
    && rm -rf /var/lib/apt/lists/*

RUN curl --fail --location --retry 3 \
      "https://ffmpeg.org/releases/ffmpeg-${FFMPEG_VERSION}.tar.xz" \
      --output /tmp/ffmpeg.tar.xz \
    && mkdir /tmp/ffmpeg \
    && tar --extract --xz --file /tmp/ffmpeg.tar.xz --strip-components=1 --directory /tmp/ffmpeg \
    && cd /tmp/ffmpeg \
    && ./configure --disable-debug --disable-doc --disable-ffplay --enable-shared \
    && make -j"$(nproc)" \
    && make install \
    && ldconfig \
    && ffmpeg -version | grep -F "ffmpeg version ${FFMPEG_VERSION}" \
    && rm -rf /tmp/ffmpeg /tmp/ffmpeg.tar.xz

RUN git clone --filter=blob:none https://github.com/colmap/colmap.git /opt/colmap \
    && git -C /opt/colmap fetch --depth 1 origin "${COLMAP_REVISION}" \
    && git -C /opt/colmap checkout --detach FETCH_HEAD \
    && test "$(git -C /opt/colmap rev-parse HEAD)" = "${COLMAP_REVISION}" \
    && printf '%s\n' "${COLMAP_REVISION}" > /opt/colmap/.blueprint-source-revision \
    && cmake -S /opt/colmap -B /opt/colmap/build -GNinja \
      -DCMAKE_BUILD_TYPE=Release -DGUI_ENABLED=OFF -DCUDA_ENABLED=ON \
      -DONNX_ENABLED=ON -DFETCH_ONNX=ON \
      -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89" \
    && cmake --build /opt/colmap/build --parallel \
    && cmake --install /opt/colmap/build \
    && colmap -h >/dev/null

COPY deploy/docker/reconstruction_worker/requirements.lock /opt/blueprint/requirements.lock
RUN python3.11 -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip==24.2 setuptools==74.1.2 wheel==0.44.0 \
    && /opt/venv/bin/pip install --extra-index-url https://download.pytorch.org/whl/cu124 \
      --requirement /opt/blueprint/requirements.lock

RUN git clone --filter=blob:none https://github.com/nerfstudio-project/gsplat.git /opt/gsplat \
    && git -C /opt/gsplat fetch --depth 1 origin "${GSPLAT_REVISION}" \
    && git -C /opt/gsplat checkout --detach FETCH_HEAD \
    && test "$(git -C /opt/gsplat rev-parse HEAD)" = "${GSPLAT_REVISION}" \
    && printf '%s\n' "${GSPLAT_REVISION}" > /opt/gsplat/.blueprint-source-revision \
    && /opt/venv/bin/pip install --no-deps /opt/gsplat

RUN git clone --filter=blob:none --recurse-submodules https://github.com/nv-tlabs/3DGRUT.git /opt/3dgrut \
    && git -C /opt/3dgrut fetch --depth 1 origin "${THREEDGRUT_REVISION}" \
    && git -C /opt/3dgrut checkout --detach FETCH_HEAD \
    && git -C /opt/3dgrut submodule update --init --recursive \
    && test "$(git -C /opt/3dgrut rev-parse HEAD)" = "${THREEDGRUT_REVISION}" \
    && printf '%s\n' "${THREEDGRUT_REVISION}" > /opt/3dgrut/.blueprint-source-revision \
    && /opt/venv/bin/pip install --no-deps --editable /opt/3dgrut \
    && /opt/venv/bin/python -c "import threedgrut"

RUN mkdir -p /opt/models/colmap \
    && curl --fail --location --retry 3 https://github.com/colmap/colmap/releases/download/3.13.0/aliked-n16rot.onnx --output /opt/models/colmap/aliked-n16rot.onnx \
    && curl --fail --location --retry 3 https://github.com/colmap/colmap/releases/download/3.13.0/aliked-lightglue.onnx --output /opt/models/colmap/aliked-lightglue.onnx \
    && curl --fail --location --retry 3 https://github.com/colmap/colmap/releases/download/3.13.0/sift-lightglue.onnx --output /opt/models/colmap/sift-lightglue.onnx \
    && curl --fail --location --retry 3 https://github.com/colmap/colmap/releases/download/3.13.0/bruteforce-matcher.onnx --output /opt/models/colmap/bruteforce-matcher.onnx \
    && printf '%s  %s\n' \
      39c423d0a6f03d39ec89d3d1d61853765c2fb6a8b8381376c703e5758778a547 /opt/models/colmap/aliked-n16rot.onnx \
      b9a5de7204648b18a8cf5dcac819f9d30de1a5961ef03756803c8b86c2dceb8d /opt/models/colmap/aliked-lightglue.onnx \
      e0500228472b43f92b3d36881a09b3310d3b058b56187b246cc7b9ab6429096e /opt/models/colmap/sift-lightglue.onnx \
      3c1282f96d83f5ffc861a873298d08bbe5219f59af59223f5ceab5c41a182a47 /opt/models/colmap/bruteforce-matcher.onnx \
      | sha256sum --check --strict

COPY pyproject.toml README.md LICENSE /opt/blueprint/
COPY src /opt/blueprint/src
RUN /opt/venv/bin/pip install --no-deps -e /opt/blueprint \
    && PATH=/opt/venv/bin:${PATH} /opt/venv/bin/python -m \
      blueprint_pipeline.reconstruction_worker_image_healthcheck \
      --build-time --output /opt/blueprint/reconstruction_worker_build_healthcheck.json

RUN groupadd --gid 10001 blueprint \
    && useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin blueprint \
    && mkdir -p /workspace/inputs /workspace/outputs \
    && chown -R blueprint:blueprint /workspace

ENV PATH=/opt/venv/bin:${PATH}
USER blueprint:blueprint
WORKDIR /workspace
ENTRYPOINT ["python", "-m", "blueprint_pipeline.reconstruction_worker_image_healthcheck"]
