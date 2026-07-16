# syntax=docker/dockerfile:1.6
# Small RunPod carrier for a runtime archive extracted from the exact sealed
# GR00T/OSCAR/Isaac release. Runtime source and checkpoints stay on the verified
# network volume; this image supplies only the compatible OS/CUDA link surface.

ARG PYTORCH_CARRIER_BASE=pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime@sha256:b85566342b86d13a67712e9315d40cdc2dad7f8d86df1aff3831f80835edbcca
ARG TENSORRT_VERSION=10.4.0.26-1+cuda12.6
ARG CUDA_CUDART_VERSION=12.6.77-1

FROM ${PYTORCH_CARRIER_BASE}
USER root
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ARG TENSORRT_VERSION CUDA_CUDART_VERSION
ADD --checksum=sha256:d2a6b11c096396d868758b86dab1823b25e14d70333f1dfa74da5ddaf6a06dba \
  https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb \
  /tmp/cuda-keyring.deb
RUN dpkg -i /tmp/cuda-keyring.deb \
  && rm -f /tmp/cuda-keyring.deb \
  && apt-get update \
  && apt-cache madison libnvinfer10 | awk -v version="${TENSORRT_VERSION}" '$3 == version { found=1 } END { exit !found }' \
  && apt-cache madison cuda-cudart-12-6 | awk -v version="${CUDA_CUDART_VERSION}" '$3 == version { found=1 } END { exit !found }' \
  && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      ca-certificates cuda-cudart-12-6=${CUDA_CUDART_VERSION} ffmpeg gettext-base \
      libatomic1 libegl1 libgl1 libglib2.0-0 libglu1-mesa libglx0 libgomp1 \
      libnghttp2-14 \
      libnvinfer10=${TENSORRT_VERSION} libnvinfer-plugin10=${TENSORRT_VERSION} \
      libnvonnxparsers10=${TENSORRT_VERSION} libosmesa6 libsm6 libxi6 libxrandr2 \
      libxt6 libyaml-cpp0.8 libzmq5 zlib1g \
  && rm -rf /var/lib/apt/lists/* \
  && mkdir -p /usr/share/glvnd/egl_vendor.d /etc/vulkan/icd.d /etc/vulkan/implicit_layer.d \
  && printf '%s\n' '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}' \
      > /usr/share/glvnd/egl_vendor.d/10_nvidia.json \
  && printf '%s\n' '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_mesa.so.0"}}' \
      > /usr/share/glvnd/egl_vendor.d/50_mesa.json \
  && printf '%s\n' '{"file_format_version":"1.0.0","ICD":{"library_path":"libGLX_nvidia.so.0","api_version":"1.3.194"}}' \
      > /etc/vulkan/icd.d/nvidia_icd.json
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all \
    VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MUJOCO_GL=osmesa \
    BLUEPRINT_GROOT_OSCAR_REQUIRED_CUDA_VERSION=12.8 \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    PYTHONPATH=/opt/wbc:/opt/OSCAR \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    BLUEPRINT_GROOT_OSCAR_OSCAR_REPO=/opt/OSCAR \
    BLUEPRINT_GROOT_OSCAR_GROOT_VENV_PYTHON=/opt/gr00t-venv/bin/python \
    BLUEPRINT_GROOT_OSCAR_GROOT_ROOT=/opt/gr00t \
    BLUEPRINT_GEAR_SONIC_ROOT=/opt/wbc \
    BLUEPRINT_GEAR_SONIC_ROBOT_MODEL=/opt/wbc/gear_sonic_deploy/g1/g1_29dof_with_hand.xml \
    BLUEPRINT_GEAR_SONIC_EXECUTOR_COMMAND="/opt/oscar-venv/bin/python -m blueprint_pipeline.gear_sonic_official_zmq_executor" \
    BLUEPRINT_ISAAC_PYTHON=/isaac-sim/python.sh \
    BLUEPRINT_ISAAC_UNITREE_G1_USD=/isaac-sim/Isaac/Robots/Unitree/G1/g1.usd \
    LD_LIBRARY_PATH=/opt/wbc/gear_sonic_deploy/thirdparty_runtime/lib:/opt/onnxruntime/lib:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
WORKDIR /workspace
CMD ["/bin/bash"]
