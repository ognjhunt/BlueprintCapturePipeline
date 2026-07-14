# syntax=docker/dockerfile:1.6
# Stable, host-cached Isaac/CUDA/OSCAR/GR00T/WBC foundation.
# No Blueprint release source and no model checkpoints belong in this image.

ARG ISAAC_SIM_BASE_IMAGE=nvcr.io/nvidia/isaac-sim:6.0.0@sha256:68735a60b6c15c85e0dd0098570c6d2cc79e928f2d068ce2790aa43284ac165d
ARG GROOT_SOURCE_URL=https://github.com/NVIDIA/Isaac-GR00T.git
ARG GROOT_SOURCE_REF=e5749287857afd97b78f1147166137de29746392
ARG OSCAR_SOURCE_URL=https://github.com/wuzy2115/oscar-public.git
ARG OSCAR_SOURCE_REF=4dea2f657e221b0ff24c895fcc8ab4d46d5a9adb
ARG WBC_SOURCE_URL=https://github.com/NVlabs/GR00T-WholeBodyControl.git
ARG WBC_SOURCE_REF=6d8e931b9b10a4db2d8e7aba3ad6d5da3529ff3b
ARG TENSORRT_VERSION=10.4.0.26-1+cuda12.6

FROM ${ISAAC_SIM_BASE_IMAGE} AS tensorrt-base
USER root
ADD --checksum=sha256:d2a6b11c096396d868758b86dab1823b25e14d70333f1dfa74da5ddaf6a06dba \
  https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb \
  /tmp/cuda-keyring.deb
RUN dpkg -i /tmp/cuda-keyring.deb \
  && rm -f /tmp/cuda-keyring.deb

FROM ${ISAAC_SIM_BASE_IMAGE} AS robot-env-builder
USER root
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ARG GROOT_SOURCE_URL GROOT_SOURCE_REF OSCAR_SOURCE_URL OSCAR_SOURCE_REF
RUN apt-get update && apt-get install -y --no-install-recommends git python3-pip python3-venv ca-certificates \
  && rm -rf /var/lib/apt/lists/* \
  && python3 -m pip install --break-system-packages --no-cache-dir uv
COPY deploy/docker/robot_eval_worker/groot_oscar_closed_loop/requirements_robot_runtime.txt /tmp/requirements_robot_runtime.txt
COPY src /tmp/blueprint-build-src
RUN git clone --filter=blob:none "${GROOT_SOURCE_URL}" /tmp/gr00t \
  && git -C /tmp/gr00t fetch --depth 1 origin "${GROOT_SOURCE_REF}" \
  && git -C /tmp/gr00t checkout --detach FETCH_HEAD \
  && test "$(git -C /tmp/gr00t rev-parse HEAD)" = "${GROOT_SOURCE_REF}" \
  && git clone --filter=blob:none "${OSCAR_SOURCE_URL}" /tmp/oscar \
  && git -C /tmp/oscar fetch --depth 1 origin "${OSCAR_SOURCE_REF}" \
  && git -C /tmp/oscar checkout --detach FETCH_HEAD \
  && test "$(git -C /tmp/oscar rev-parse HEAD)" = "${OSCAR_SOURCE_REF}" \
  && uv venv /opt/robot-venv --python 3.10 --seed \
  && VIRTUAL_ENV=/opt/robot-venv uv pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.10.0 torchvision==0.25.0 \
  && VIRTUAL_ENV=/opt/robot-venv uv pip install "nvidia-cudnn-cu12>=9.10" -r /tmp/requirements_robot_runtime.txt \
  && VIRTUAL_ENV=/opt/robot-venv uv pip install /tmp/gr00t \
  && PYTHONPATH=/tmp/blueprint-build-src /opt/robot-venv/bin/python -c "from pathlib import Path; from blueprint_pipeline.oscar_wam_gpu_image import filter_requirements_script_text, transformer_engine_shim_script_text; Path('/tmp/filter.py').write_text(filter_requirements_script_text()); Path('/tmp/te_shim.py').write_text(transformer_engine_shim_script_text())" \
  && /opt/robot-venv/bin/python /tmp/filter.py /tmp/oscar/requirements.txt /tmp/oscar_requirements.txt \
  && VIRTUAL_ENV=/opt/robot-venv uv pip install -r /tmp/oscar_requirements.txt \
  && /opt/robot-venv/bin/python /tmp/te_shim.py /tmp/oscar \
  && /opt/robot-venv/bin/python -m pip check \
  && /opt/robot-venv/bin/python -c "from gr00t.policy.gr00t_policy import Gr00tPolicy" \
  && rm -rf /opt/robot-venv/.git /root/.cache /tmp/gr00t/.git /tmp/oscar/.git \
  && mkdir -p /opt/oscar-runtime \
  && cp -a /tmp/oscar/. /opt/oscar-runtime/

FROM tensorrt-base AS wbc-builder
USER root
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ARG WBC_SOURCE_URL WBC_SOURCE_REF TENSORRT_VERSION
ADD --checksum=sha256:a1bc93654f31669fd964ea3011a5e5e9676b9b6f8adcd762606e5140632ea72d \
  https://github.com/casey/just/releases/download/1.43.0/just-1.43.0-x86_64-unknown-linux-musl.tar.gz \
  /tmp/just.tgz
ADD --checksum=sha256:b072f989d6315ac0e22dcb4771b083c5156d974a3496ac3504c77f4062eb248e \
  https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-1.16.3.tgz \
  /tmp/onnxruntime.tgz
RUN tar -xzf /tmp/just.tgz -C /usr/local/bin just \
  && mkdir -p /opt/onnxruntime \
  && tar -xzf /tmp/onnxruntime.tgz -C /opt/onnxruntime --strip-components=1 \
  && rm -f /tmp/just.tgz /tmp/onnxruntime.tgz \
  && apt-get update \
  && apt-cache madison libnvinfer10 | awk -v version="${TENSORRT_VERSION}" '$3 == version { found=1 } END { exit !found }' \
  && apt-get install -y --no-install-recommends \
      build-essential clang cmake git git-lfs ninja-build pkg-config curl ca-certificates sudo cppzmq-dev \
      libnvinfer-headers-dev=${TENSORRT_VERSION} libnvinfer-headers-plugin-dev=${TENSORRT_VERSION} \
      libnvinfer10=${TENSORRT_VERSION} libnvinfer-plugin10=${TENSORRT_VERSION} \
      libnvonnxparsers10=${TENSORRT_VERSION} libnvinfer-dev=${TENSORRT_VERSION} \
      libnvinfer-plugin-dev=${TENSORRT_VERSION} libnvonnxparsers-dev=${TENSORRT_VERSION} \
  && rm -rf /var/lib/apt/lists/* \
  && git clone --filter=blob:none "${WBC_SOURCE_URL}" /tmp/wbc \
  && git -C /tmp/wbc fetch --depth 1 origin "${WBC_SOURCE_REF}" \
  && git -C /tmp/wbc checkout --detach FETCH_HEAD \
  && test "$(git -C /tmp/wbc rev-parse HEAD)" = "${WBC_SOURCE_REF}" \
  && git -C /tmp/wbc lfs pull \
  && cd /tmp/wbc/gear_sonic_deploy \
  && chmod +x scripts/install_deps.sh deploy.sh \
  && scripts/install_deps.sh \
  && test "$(command -v just)" = /usr/local/bin/just \
  && test -f /usr/include/zmq.hpp \
  && test ! -d third_party/cppzmq/.git \
  && sed -i 's/nvinfer nvinfer_plugin nvonnxparser nvparsers/nvinfer nvinfer_plugin nvonnxparser/' cmake/FindTensorRT.cmake \
  && source scripts/setup_env.sh \
  && just build \
  && test -d /opt/onnxruntime \
  && mkdir -p /opt/wbc-runtime/gear_sonic_deploy/target/release \
      /opt/wbc-runtime/gear_sonic_deploy/scripts \
      /opt/wbc-runtime/gear_sonic_deploy/reference \
      /opt/wbc-runtime/gear_sonic/utils \
      /opt/wbc-runtime/gear_sonic/utils/teleop \
      /opt/onnxruntime-runtime/lib \
  && install -m 0755 target/release/g1_deploy_onnx_ref \
      /opt/wbc-runtime/gear_sonic_deploy/target/release/g1_deploy_onnx_ref \
  && cp -a g1 /opt/wbc-runtime/gear_sonic_deploy/g1 \
  && install -m 0755 scripts/setup_env.sh \
      /opt/wbc-runtime/gear_sonic_deploy/scripts/setup_env.sh \
  && cp -a reference/example /opt/wbc-runtime/gear_sonic_deploy/reference/example \
  && mkdir -p /opt/wbc-runtime/gear_sonic_deploy/thirdparty_runtime \
  && cp -a thirdparty/unitree_sdk2/thirdparty/lib/x86_64 /opt/wbc-runtime/gear_sonic_deploy/thirdparty_runtime/lib \
  && install -m 0644 /tmp/wbc/gear_sonic/__init__.py /tmp/wbc/gear_sonic/version.py \
      /opt/wbc-runtime/gear_sonic/ \
  && install -m 0644 /tmp/wbc/gear_sonic/utils/__init__.py \
      /opt/wbc-runtime/gear_sonic/utils/__init__.py \
  && cp -a /tmp/wbc/gear_sonic/utils/teleop/zmq \
      /opt/wbc-runtime/gear_sonic/utils/teleop/zmq \
  && cp -a /opt/onnxruntime/lib/libonnxruntime.so* /opt/onnxruntime-runtime/lib/ \
  && printf '%s\n' "${WBC_SOURCE_REF}" > /opt/wbc-runtime/.blueprint-source-revision \
  && mkdir -p /opt/wbc-runtime/gear_sonic_deploy/policy/release /opt/wbc-runtime/gear_sonic_deploy/planner/target_vel/V2 \
  && test ! -d /opt/wbc-runtime/gear_sonic_deploy/build \
  && test ! -d /opt/wbc-runtime/gear_sonic_deploy/src \
  && test ! -d /opt/wbc-runtime/gear_sonic/tests \
  && test -z "$(find /opt/wbc-runtime -type f \( -name '*.o' -o -name '*.a' -o -name 'CMakeCache.txt' -o -name 'CMakeLists.txt' \) -print -quit)" \
  && test -x /opt/wbc-runtime/gear_sonic_deploy/target/release/g1_deploy_onnx_ref \
  && test -f /opt/wbc-runtime/.blueprint-source-revision

FROM tensorrt-base
USER root
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ARG APP_UID=10001
ARG APP_GID=10001
ARG GROOT_SOURCE_REF OSCAR_SOURCE_REF WBC_SOURCE_REF TENSORRT_VERSION
RUN apt-get update \
  && apt-cache madison libnvinfer10 | awk -v version="${TENSORRT_VERSION}" '$3 == version { found=1 } END { exit !found }' \
  && apt-get install -y --no-install-recommends \
      libosmesa6 ffmpeg ca-certificates gettext-base libzmq5 libyaml-cpp0.8 zlib1g \
      libnvinfer10=${TENSORRT_VERSION} libnvinfer-plugin10=${TENSORRT_VERSION} \
      libnvonnxparsers10=${TENSORRT_VERSION} \
  && installed_build_packages="$(dpkg-query -W -f='${binary:Package}\n' build-essential clang cmake git git-lfs ninja-build pkg-config 2>/dev/null || true)" \
  && if [[ -n "${installed_build_packages}" ]]; then apt-get purge -y ${installed_build_packages}; fi \
  && apt-get autoremove -y \
  && rm -rf /var/lib/apt/lists/* \
  && groupadd --gid "${APP_GID}" blueprint \
  && useradd --uid "${APP_UID}" --gid "${APP_GID}" --create-home --shell /usr/sbin/nologin blueprint \
  && usermod -aG isaac-sim blueprint \
  && mkdir -p /workspace /opt/blueprint /models /isaac-sim/kit/cache /isaac-sim/kit/data /isaac-sim/kit/logs \
  && chown blueprint:blueprint /workspace /opt/blueprint /models \
  && chown blueprint:isaac-sim /isaac-sim/kit/cache /isaac-sim/kit/data /isaac-sim/kit/logs \
  && chmod 0775 /isaac-sim/kit/cache /isaac-sim/kit/data /isaac-sim/kit/logs
COPY --from=robot-env-builder --chown=blueprint:blueprint /opt/robot-venv /opt/robot-venv
COPY --from=robot-env-builder --chown=blueprint:blueprint /opt/oscar-runtime /opt/OSCAR
COPY --from=wbc-builder --chown=blueprint:blueprint /opt/wbc-runtime /opt/wbc
COPY --from=wbc-builder /opt/onnxruntime-runtime /opt/onnxruntime
COPY deploy/docker/robot_eval_worker/groot_oscar_closed_loop/isaac_6_g1_assets.sha256 /opt/blueprint/isaac_6_g1_assets.sha256
COPY deploy/docker/robot_eval_worker/groot_oscar_closed_loop/fetch_pinned_isaac_assets.py /opt/blueprint/fetch_pinned_isaac_assets.py
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 MUJOCO_GL=osmesa \
    BLUEPRINT_GROOT_OSCAR_REQUIRED_CUDA_VERSION=12.6 \
    PYTORCH_ALLOC_CONF=expandable_segments:True PYTHONPATH=/opt/OSCAR \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    BLUEPRINT_GROOT_OSCAR_MODEL_CACHE=/models/blueprint-groot-oscar-v1 \
    BLUEPRINT_GROOT_OSCAR_OSCAR_REPO=/opt/OSCAR \
    BLUEPRINT_GROOT_OSCAR_OSCAR_CHECKPOINT=/models/blueprint-groot-oscar-v1/oscar \
    BLUEPRINT_GROOT_OSCAR_GROOT_VENV_PYTHON=/opt/robot-venv/bin/python \
    BLUEPRINT_GROOT_OSCAR_SONIC_CHECKPOINT=/models/blueprint-groot-oscar-v1/sonic \
    BLUEPRINT_GROOT_OSCAR_GROOT_ROOT=/opt/robot-venv \
    BLUEPRINT_GEAR_SONIC_ROOT=/opt/wbc \
    BLUEPRINT_GEAR_SONIC_ROBOT_MODEL=/opt/wbc/gear_sonic_deploy/g1/g1_29dof_with_hand.xml \
    BLUEPRINT_GEAR_SONIC_EXECUTOR_COMMAND="/opt/robot-venv/bin/python -m blueprint_pipeline.gear_sonic_official_zmq_executor" \
    LD_LIBRARY_PATH=/opt/wbc/gear_sonic_deploy/thirdparty_runtime/lib:/opt/onnxruntime/lib:/usr/lib/x86_64-linux-gnu \
    BLUEPRINT_ISAAC_PYTHON=/isaac-sim/python.sh \
    BLUEPRINT_ISAAC_UNITREE_G1_USD=/isaac-sim/Isaac/Robots/Unitree/G1/g1.usd \
    BLUEPRINT_FOUNDATION_GROOT_SOURCE_REF=${GROOT_SOURCE_REF} \
    BLUEPRINT_FOUNDATION_OSCAR_SOURCE_REF=${OSCAR_SOURCE_REF} \
    BLUEPRINT_FOUNDATION_WBC_SOURCE_REF=${WBC_SOURCE_REF}
RUN ln -s /opt/robot-venv /opt/oscar-venv \
  && ln -s /opt/robot-venv /opt/gr00t-venv \
  && python3 /opt/blueprint/fetch_pinned_isaac_assets.py \
      --manifest /opt/blueprint/isaac_6_g1_assets.sha256 \
      --base-url https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/Robots/Unitree/G1/ \
      --output-dir /isaac-sim/Isaac/Robots/Unitree/G1 \
  && test ! -e /opt/blueprint/ckpts \
  && test ! -d /opt/wbc/.git \
  && test ! -d /opt/OSCAR/.git \
  && test ! -d /opt/wbc/gear_sonic_deploy/build \
  && test ! -d /opt/wbc/gear_sonic_deploy/src \
  && test ! -d /opt/onnxruntime/include \
  && ! dpkg-query -W build-essential clang cmake git git-lfs ninja-build pkg-config >/dev/null 2>&1 \
  && ! ldd /opt/wbc/gear_sonic_deploy/target/release/g1_deploy_onnx_ref | grep -q 'not found' \
  && /opt/robot-venv/bin/python -m pip check
USER blueprint
WORKDIR /workspace
CMD ["bash"]
