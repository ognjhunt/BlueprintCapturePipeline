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
ARG CUDA_CUDART_VERSION=12.6.77-1

FROM ${ISAAC_SIM_BASE_IMAGE} AS tensorrt-base
USER root
COPY deploy/docker/robot_eval_worker/groot_oscar_closed_loop/apt_transport_hardening.conf \
  /etc/apt/apt.conf.d/80blueprint-transport-hardening
RUN find /etc/apt -maxdepth 2 -type f \( -name '*.list' -o -name '*.sources' \) \
      -exec sed -i \
        -e 's#http://archive.ubuntu.com/ubuntu#https://archive.ubuntu.com/ubuntu#g' \
        -e 's#http://security.ubuntu.com/ubuntu#https://security.ubuntu.com/ubuntu#g' \
        '{}' + \
  && ! grep -RqsE 'http://(archive|security)\.ubuntu\.com/ubuntu' /etc/apt
ADD --checksum=sha256:d2a6b11c096396d868758b86dab1823b25e14d70333f1dfa74da5ddaf6a06dba \
  https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb \
  /tmp/cuda-keyring.deb
RUN dpkg -i /tmp/cuda-keyring.deb \
  && rm -f /tmp/cuda-keyring.deb

FROM ${ISAAC_SIM_BASE_IMAGE} AS robot-env-builder
USER root
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ARG GROOT_SOURCE_URL GROOT_SOURCE_REF OSCAR_SOURCE_URL OSCAR_SOURCE_REF
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python PYTHONDONTWRITEBYTECODE=1
COPY deploy/docker/robot_eval_worker/groot_oscar_closed_loop/apt_transport_hardening.conf \
  /etc/apt/apt.conf.d/80blueprint-transport-hardening
RUN find /etc/apt -maxdepth 2 -type f \( -name '*.list' -o -name '*.sources' \) \
      -exec sed -i \
        -e 's#http://archive.ubuntu.com/ubuntu#https://archive.ubuntu.com/ubuntu#g' \
        -e 's#http://security.ubuntu.com/ubuntu#https://security.ubuntu.com/ubuntu#g' \
        '{}' + \
  && ! grep -RqsE 'http://(archive|security)\.ubuntu\.com/ubuntu' /etc/apt
COPY deploy/docker/robot_eval_worker/groot_oscar_closed_loop/requirements_uv_bootstrap.txt /tmp/requirements_uv_bootstrap.txt
RUN apt-get update && apt-get install -y --no-install-recommends git python3-pip python3-venv ca-certificates \
  && rm -rf /var/lib/apt/lists/* \
  && python3 -m pip install --break-system-packages --no-cache-dir --require-hashes -r /tmp/requirements_uv_bootstrap.txt
COPY deploy/docker/robot_eval_worker/groot_oscar_closed_loop/requirements_robot_runtime.txt /tmp/requirements_robot_runtime.txt
COPY deploy/docker/robot_eval_worker/groot_oscar_closed_loop/requirements_oscar_foundation.lock /tmp/requirements_oscar_foundation.lock
COPY deploy/docker/robot_eval_worker/groot_oscar_closed_loop/oscar_cpu_import_probe.py /tmp/oscar_cpu_import_probe.py
COPY src /tmp/blueprint-build-src
RUN git clone --filter=blob:none "${GROOT_SOURCE_URL}" /tmp/gr00t \
  && git -C /tmp/gr00t fetch --depth 1 origin "${GROOT_SOURCE_REF}" \
  && git -C /tmp/gr00t checkout --detach FETCH_HEAD \
  && test "$(git -C /tmp/gr00t rev-parse HEAD)" = "${GROOT_SOURCE_REF}" \
  && git clone --filter=blob:none "${OSCAR_SOURCE_URL}" /tmp/oscar \
  && git -C /tmp/oscar fetch --depth 1 origin "${OSCAR_SOURCE_REF}" \
  && git -C /tmp/oscar checkout --detach FETCH_HEAD \
  && test "$(git -C /tmp/oscar rev-parse HEAD)" = "${OSCAR_SOURCE_REF}" \
  && test -f /tmp/oscar/requirements_minimal.txt \
  && uv venv /opt/oscar-venv --python 3.10 --seed \
  && VIRTUAL_ENV=/opt/oscar-venv uv pip install --require-hashes \
      --index-url https://download.pytorch.org/whl/cu128 \
      --extra-index-url https://pypi.org/simple \
      --index-strategy unsafe-best-match \
      -r /tmp/requirements_oscar_foundation.lock \
  && PYTHONPATH=/tmp/blueprint-build-src /opt/oscar-venv/bin/python -c "from pathlib import Path; from blueprint_pipeline.oscar_wam_gpu_image import transformer_engine_shim_script_text; Path('/tmp/te_shim.py').write_text(transformer_engine_shim_script_text())" \
  && /opt/oscar-venv/bin/python /tmp/te_shim.py /tmp/oscar \
  && decord_wheel=/opt/oscar-venv/lib/python3.10/site-packages/decord-0.6.0.dist-info/WHEEL \
  && test -f "${decord_wheel}" \
  && grep -qx 'Tag: cp36-cp36m-manylinux2010_x86_64' "${decord_wheel}" \
  && sed -i 's/^Tag: cp36-cp36m-manylinux2010_x86_64$/Tag: py3-none-manylinux2010_x86_64/' "${decord_wheel}" \
  && grep -qx 'Tag: py3-none-manylinux2010_x86_64' "${decord_wheel}" \
  && /opt/oscar-venv/bin/python -m pip check \
  && PYTHONPATH=/tmp/oscar /opt/oscar-venv/bin/python /tmp/oscar_cpu_import_probe.py \
  && PYTHONPATH=/tmp/oscar /opt/oscar-venv/bin/python -c "import importlib.metadata; from worldsim._src.configs.agibot_control.config import make_config; assert importlib.metadata.version('pytest') == '9.1.1'; assert make_config() is not None" \
  && find /tmp/oscar -type d -name __pycache__ -prune -exec rm -rf '{}' + \
  && find /tmp/oscar -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete \
  && PYTHONPATH=/tmp/blueprint-build-src /opt/oscar-venv/bin/python -m blueprint_pipeline.oscar_runtime_source_provenance seal \
      --source-root /tmp/oscar \
      --output /tmp/oscar_source_provenance.json \
      --source-url "${OSCAR_SOURCE_URL}" \
      --source-commit "${OSCAR_SOURCE_REF}" \
      --runtime-source-root /opt/OSCAR \
  && uv venv /opt/gr00t-venv --python 3.10 --seed \
  && VIRTUAL_ENV=/opt/gr00t-venv uv sync --project /tmp/gr00t --active --no-dev --frozen --no-install-project \
  && VIRTUAL_ENV=/opt/gr00t-venv uv pip install --no-deps /tmp/gr00t \
  && /opt/gr00t-venv/bin/python -m pip check \
  && /opt/gr00t-venv/bin/python -c "from gr00t.policy.gr00t_policy import Gr00tPolicy" \
  && rm -rf /opt/oscar-venv/.git /opt/gr00t-venv/.git /root/.cache /tmp/gr00t/.git /tmp/oscar/.git \
  && mkdir -p /opt/oscar-runtime \
  && cp -a /tmp/oscar/. /opt/oscar-runtime/ \
  && mkdir -p /opt/gr00t-runtime \
  && cp -a /tmp/gr00t/. /opt/gr00t-runtime/

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
ARG GROOT_SOURCE_REF OSCAR_SOURCE_URL OSCAR_SOURCE_REF WBC_SOURCE_REF TENSORRT_VERSION CUDA_CUDART_VERSION
RUN apt-get update \
  && apt-cache madison libnvinfer10 | awk -v version="${TENSORRT_VERSION}" '$3 == version { found=1 } END { exit !found }' \
  && apt-cache madison cuda-cudart-12-6 | awk -v version="${CUDA_CUDART_VERSION}" '$3 == version { found=1 } END { exit !found }' \
  && apt-get install -y --no-install-recommends \
      libosmesa6 ffmpeg ca-certificates gettext-base libzmq5 libyaml-cpp0.8 zlib1g \
      cuda-cudart-12-6=${CUDA_CUDART_VERSION} \
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
COPY --from=robot-env-builder --chown=blueprint:blueprint /opt/oscar-venv /opt/oscar-venv
COPY --from=robot-env-builder --chown=blueprint:blueprint /opt/gr00t-venv /opt/gr00t-venv
COPY --from=robot-env-builder --chown=blueprint:blueprint /opt/uv-python /opt/uv-python
COPY --from=robot-env-builder --chown=blueprint:blueprint /opt/oscar-runtime /opt/OSCAR
COPY --from=robot-env-builder /tmp/oscar_source_provenance.json /opt/blueprint/oscar_source_provenance.json
COPY --from=robot-env-builder --chown=blueprint:blueprint /opt/gr00t-runtime /opt/gr00t
COPY --from=wbc-builder --chown=blueprint:blueprint /opt/wbc-runtime /opt/wbc
COPY --from=wbc-builder /opt/onnxruntime-runtime /opt/onnxruntime
COPY deploy/docker/robot_eval_worker/groot_oscar_closed_loop/isaac_6_g1_assets.sha256 /opt/blueprint/isaac_6_g1_assets.sha256
COPY deploy/docker/robot_eval_worker/groot_oscar_closed_loop/fetch_pinned_isaac_assets.py /opt/blueprint/fetch_pinned_isaac_assets.py
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 MUJOCO_GL=osmesa \
    BLUEPRINT_GROOT_OSCAR_REQUIRED_CUDA_VERSION=12.8 \
    PYTORCH_ALLOC_CONF=expandable_segments:True PYTHONPATH=/opt/wbc:/opt/OSCAR \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    BLUEPRINT_GROOT_OSCAR_MODEL_CACHE=/models/blueprint-groot-oscar-v1 \
    BLUEPRINT_GROOT_OSCAR_OSCAR_REPO=/opt/OSCAR \
    BLUEPRINT_GROOT_OSCAR_OSCAR_CHECKPOINT=/models/blueprint-groot-oscar-v1/oscar \
    BLUEPRINT_GROOT_OSCAR_GROOT_VENV_PYTHON=/opt/gr00t-venv/bin/python \
    BLUEPRINT_GROOT_OSCAR_SONIC_CHECKPOINT=/models/blueprint-groot-oscar-v1/sonic \
    BLUEPRINT_GROOT_OSCAR_GROOT_ROOT=/opt/gr00t \
    BLUEPRINT_GEAR_SONIC_ROOT=/opt/wbc \
    BLUEPRINT_GEAR_SONIC_ROBOT_MODEL=/opt/wbc/gear_sonic_deploy/g1/g1_29dof_with_hand.xml \
    BLUEPRINT_GEAR_SONIC_EXECUTOR_COMMAND="/opt/oscar-venv/bin/python -m blueprint_pipeline.gear_sonic_official_zmq_executor" \
    LD_LIBRARY_PATH=/opt/wbc/gear_sonic_deploy/thirdparty_runtime/lib:/opt/onnxruntime/lib:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu \
    BLUEPRINT_ISAAC_PYTHON=/isaac-sim/python.sh \
    BLUEPRINT_ISAAC_UNITREE_G1_USD=/isaac-sim/Isaac/Robots/Unitree/G1/g1.usd \
    BLUEPRINT_FOUNDATION_GROOT_SOURCE_REF=${GROOT_SOURCE_REF} \
    BLUEPRINT_FOUNDATION_WBC_SOURCE_REF=${WBC_SOURCE_REF} \
    BLUEPRINT_GEAR_SONIC_SOURCE_REVISION=${WBC_SOURCE_REF}
RUN /opt/oscar-venv/bin/python /opt/blueprint/fetch_pinned_isaac_assets.py \
      --manifest /opt/blueprint/isaac_6_g1_assets.sha256 \
      --base-url https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/Robots/Unitree/G1/ \
      --output-dir /isaac-sim/Isaac/Robots/Unitree/G1 \
  && test -s /opt/blueprint/oscar_source_provenance.json \
  && test ! -e /opt/blueprint/ckpts \
  && test ! -d /opt/wbc/.git \
  && test ! -d /opt/OSCAR/.git \
  && test ! -d /opt/wbc/gear_sonic_deploy/build \
  && test ! -d /opt/wbc/gear_sonic_deploy/src \
  && test ! -d /opt/onnxruntime/include \
  && ! dpkg-query -W build-essential clang cmake git git-lfs ninja-build pkg-config >/dev/null 2>&1 \
  && ldd /opt/wbc/gear_sonic_deploy/target/release/g1_deploy_onnx_ref | tee /tmp/g1_deploy_onnx_ref.ldd \
  && ! grep -q 'not found' /tmp/g1_deploy_onnx_ref.ldd \
  && rm -f /tmp/g1_deploy_onnx_ref.ldd \
  && /opt/oscar-venv/bin/python -m pip check \
  && /opt/gr00t-venv/bin/python -m pip check
USER blueprint
WORKDIR /workspace
CMD ["bash"]
