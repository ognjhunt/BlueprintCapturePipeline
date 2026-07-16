ARG CARRIER_BASE_IMAGE=pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime@sha256:b85566342b86d13a67712e9315d40cdc2dad7f8d86df1aff3831f80835edbcca
FROM ${CARRIER_BASE_IMAGE}

USER root
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Keep this runtime-only package set aligned with the final stage of
# Foundation.Dockerfile.  The persistent volume supplies /isaac-sim and /opt;
# the carrier supplies the Ubuntu ABI and pinned NVIDIA runtime libraries those
# copied files were built and already validated against.
ARG TENSORRT_VERSION=10.4.0.26-1+cuda12.6
ARG CUDA_CUDART_VERSION=12.6.77-1
ADD --checksum=sha256:d2a6b11c096396d868758b86dab1823b25e14d70333f1dfa74da5ddaf6a06dba \
    https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb \
    /tmp/cuda-keyring.deb
RUN dpkg -i /tmp/cuda-keyring.deb \
  && rm -f /tmp/cuda-keyring.deb \
  && apt-get update \
  && apt-cache madison libnvinfer10 | awk -v version="${TENSORRT_VERSION}" '$3 == version { found=1 } END { exit !found }' \
  && apt-cache madison cuda-cudart-12-6 | awk -v version="${CUDA_CUDART_VERSION}" '$3 == version { found=1 } END { exit !found }' \
  && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      ca-certificates curl ffmpeg gettext-base gpgv libatomic1 libegl1 libgl1 \
      libglib2.0-0 libglu1-mesa libglx0 libgomp1 libnghttp2-14 libosmesa6 \
      libsm6 libxi6 libxrandr2 libxt6 libyaml-cpp0.8 libzmq5 unzip zlib1g \
      cuda-cudart-12-6=${CUDA_CUDART_VERSION} \
      libnvinfer10=${TENSORRT_VERSION} \
      libnvinfer-plugin10=${TENSORRT_VERSION} \
      libnvonnxparsers10=${TENSORRT_VERSION} \
  && ! dpkg-query -W build-essential clang cmake git git-lfs ninja-build pkg-config >/dev/null 2>&1 \
  && rm -rf /var/lib/apt/lists/*

ENV NVIDIA_DRIVER_CAPABILITIES=all \
    PYTHONPATH=/opt/wbc:/opt/OSCAR \
    LD_LIBRARY_PATH=/opt/wbc/gear_sonic_deploy/thirdparty_runtime/lib:/opt/onnxruntime/lib:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu

WORKDIR /workspace
