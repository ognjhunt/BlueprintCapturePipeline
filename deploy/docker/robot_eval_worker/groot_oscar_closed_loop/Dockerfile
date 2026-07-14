# syntax=docker/dockerfile:1.6
#
# Sealed worker image for the GR00T N1.7 + SONIC x OSCAR-2B closed-loop eval
# (oscar_isaac_closed_loop_eval + groot_sonic_policy_endpoint, the "T4" lane).
#
# The final carrier is Isaac Sim 6 itself. OSCAR and GR00T live in isolated
# Python 3.10 environments so their torch pins cannot replace Isaac's runtime.
#
# Build context = repo root (so `src` + pyproject are COPY-able). Checkpoints are
# fetched INSIDE the build (BuildKit secret hf_token), never from the context.
#
# This mirrors deploy/docker/robot_eval_worker/unitree_groot_sonic_wam/Dockerfile
# but for the OSCAR-in-the-loop closed loop, and bakes checkpoints for docker
# pull + go. It proves build/runtime readiness only — not task success.

ARG ISAAC_SIM_BASE_IMAGE=nvcr.io/nvidia/isaac-sim:6.0.0@sha256:68735a60b6c15c85e0dd0098570c6d2cc79e928f2d068ce2790aa43284ac165d

# WBC/GEAR needs nvcc, CUDA development headers, CMake, and other build-only
# packages. Build it in a disposable stage so none of that toolchain ships in
# the paid worker image. The same pinned Isaac carrier preserves compiler ABI.
FROM --platform=linux/amd64 ${ISAAC_SIM_BASE_IMAGE} AS gear_sonic_builder

USER root
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG DEBIAN_FRONTEND=noninteractive
ARG WBC_SOURCE_URL=https://github.com/NVlabs/GR00T-WholeBodyControl.git
ARG WBC_SOURCE_REF=6d8e931b9b10a4db2d8e7aba3ad6d5da3529ff3b
ARG TENSORRT_VERSION=10.4.0.26-1+cuda12.6

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git git-lfs sudo wget \
  && rm -rf /var/lib/apt/lists/*

RUN git clone --filter=blob:none "${WBC_SOURCE_URL}" /opt/wbc \
  && git -C /opt/wbc fetch --depth 1 origin "${WBC_SOURCE_REF}" \
  && git -C /opt/wbc checkout --detach FETCH_HEAD \
  && test "$(git -C /opt/wbc rev-parse HEAD)" = "${WBC_SOURCE_REF}" \
  && printf '%s\n' "${WBC_SOURCE_REF}" > /opt/wbc/.blueprint-source-revision \
  && chmod 0444 /opt/wbc/.blueprint-source-revision \
  && git -C /opt/wbc lfs pull \
  && cd /opt/wbc/gear_sonic_deploy \
  && chmod +x scripts/install_deps.sh deploy.sh \
  && scripts/install_deps.sh \
  && apt-get install -y --no-install-recommends \
       libnvinfer-headers-dev=${TENSORRT_VERSION} \
       libnvinfer-headers-plugin-dev=${TENSORRT_VERSION} \
       libnvinfer10=${TENSORRT_VERSION} \
       libnvinfer-plugin10=${TENSORRT_VERSION} \
       libnvonnxparsers10=${TENSORRT_VERSION} \
       libnvinfer-dev=${TENSORRT_VERSION} \
       libnvinfer-plugin-dev=${TENSORRT_VERSION} \
       libnvonnxparsers-dev=${TENSORRT_VERSION} \
  && sed -i \
       's/nvinfer nvinfer_plugin nvonnxparser nvparsers/nvinfer nvinfer_plugin nvonnxparser/' \
       cmake/FindTensorRT.cmake \
  && source scripts/setup_env.sh \
  && just build \
  && test -x target/release/g1_deploy_onnx_ref \
  && rm -rf /opt/wbc/.git /opt/wbc/gear_sonic_deploy/build \
  && mkdir -p /opt/wbc/gear_sonic_deploy/build

FROM --platform=linux/amd64 ${ISAAC_SIM_BASE_IMAGE} AS runtime

# The pinned Isaac-Sim base ships USER isaac-sim (uid 1234); every layer below
# needs root (apt, /opt installs), and the sealed worker bootstrap runs as root.
USER root

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG DEBIAN_FRONTEND=noninteractive
ARG GROOT_SOURCE_URL=https://github.com/NVIDIA/Isaac-GR00T.git
ARG GROOT_SOURCE_REF=e5749287857afd97b78f1147166137de29746392
ARG WBC_SOURCE_URL=https://github.com/NVlabs/GR00T-WholeBodyControl.git
ARG WBC_SOURCE_REF=6d8e931b9b10a4db2d8e7aba3ad6d5da3529ff3b
ARG GEAR_SONIC_CHECKPOINT_REPO=nvidia/GEAR-SONIC
ARG GEAR_SONIC_CHECKPOINT_REVISION=5e22ddc69abcea2a9aafc40536b14c232d3f9d7f
ARG TENSORRT_VERSION=10.4.0.26-1+cuda12.6
ARG OSCAR_SOURCE_URL=https://github.com/wuzy2115/oscar-public.git
ARG OSCAR_SOURCE_REF=4dea2f657e221b0ff24c895fcc8ab4d46d5a9adb
ARG BLUEPRINT_SOURCE_COMMIT
ARG BLUEPRINT_SOURCE_DIRTY_PATCH_SHA256
ARG SONIC_CHECKPOINT_REPO=LucaFrat/groot-bs16
ARG SONIC_CHECKPOINT_REVISION=86b17337379926a8d8f1ad5c4580c7c33deeb49f
ARG COSMOS_BACKBONE_REPO=nvidia/Cosmos-Reason2-2B
ARG COSMOS_BACKBONE_REVISION=9ce19a195e423419c349abfc86fd07178b230561
ARG OSCAR_CHECKPOINT_REPO=zywu2115/OSCAR-2B
ARG OSCAR_CHECKPOINT_REVISION=c9781ffa7dd8556d862d7d9f338a2ea008a58ca6
ARG PREFETCH_CHECKPOINTS=true
ARG APP_UID=10001
ARG APP_GID=10001

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    MUJOCO_GL=osmesa \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    HF_HOME=/opt/blueprint/hf_home \
    BLUEPRINT_OSCAR_WAM_HF_REVISION=c9781ffa7dd8556d862d7d9f338a2ea008a58ca6 \
    PYTHONPATH=/opt/wbc:/opt/OSCAR \
    BLUEPRINT_GROOT_OSCAR_OSCAR_REPO=/opt/OSCAR \
    BLUEPRINT_GROOT_OSCAR_OSCAR_CHECKPOINT=/opt/blueprint/ckpts/oscar \
    BLUEPRINT_GROOT_OSCAR_GROOT_ROOT=/opt/gr00t \
    BLUEPRINT_GROOT_OSCAR_GROOT_VENV_PYTHON=/opt/gr00t-venv/bin/python \
    BLUEPRINT_GROOT_OSCAR_SONIC_CHECKPOINT=/opt/blueprint/ckpts/sonic \
    BLUEPRINT_UNITREE_GROOT_N17_SONIC_ROOT=/opt/gr00t \
    BLUEPRINT_UNITREE_GROOT_N17_SONIC_WBC_ROOT=/opt/wbc \
    BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_SERVER_URL=tcp://127.0.0.1:5550 \
    BLUEPRINT_WORKER_IMAGE_FAMILY=isaac-eval-worker \
    BLUEPRINT_WORKER_IMAGE_VARIANT=groot-oscar-closed-loop-eval \
    BLUEPRINT_SIMULATOR_FRAMEWORK=isaac_sim \
    BLUEPRINT_ISAAC_SIM_MAJOR_VERSION=6 \
    BLUEPRINT_ISAAC_PYTHON=/isaac-sim/python.sh \
    BLUEPRINT_ISAAC_UNITREE_G1_USD=/isaac-sim/Isaac/Robots/Unitree/G1/g1.usd \
    BLUEPRINT_SOURCE_COMMIT=${BLUEPRINT_SOURCE_COMMIT} \
    BLUEPRINT_SOURCE_DIRTY_PATCH_SHA256=${BLUEPRINT_SOURCE_DIRTY_PATCH_SHA256} \
    BLUEPRINT_GEAR_SONIC_ROOT=/opt/wbc \
    BLUEPRINT_GEAR_SONIC_SOURCE_REVISION=${WBC_SOURCE_REF} \
    GEAR_SONIC_CHECKPOINT_REPO=${GEAR_SONIC_CHECKPOINT_REPO} \
    GEAR_SONIC_CHECKPOINT_REVISION=${GEAR_SONIC_CHECKPOINT_REVISION} \
    TensorRT_ROOT=/usr \
    BLUEPRINT_GEAR_SONIC_ROBOT_MODEL=/opt/wbc/gear_sonic_deploy/g1/g1_29dof_with_hand.xml \
    BLUEPRINT_GEAR_SONIC_EXECUTOR_COMMAND="/opt/oscar-venv/bin/python -m blueprint_pipeline.gear_sonic_official_zmq_executor" \
    BLUEPRINT_GROOT_OSCAR_CLOSED_LOOP_SEALED_IMAGE_CONFIRMED=true \
    SONIC_CHECKPOINT_REPO=${SONIC_CHECKPOINT_REPO} \
    COSMOS_BACKBONE_REPO=${COSMOS_BACKBONE_REPO} \
    COSMOS_BACKBONE_REVISION=${COSMOS_BACKBONE_REVISION} \
    OSCAR_CHECKPOINT_REPO=${OSCAR_CHECKPOINT_REPO} \
    OSCAR_CHECKPOINT_REVISION=${OSCAR_CHECKPOINT_REVISION}

WORKDIR /workspace

# 1. System build/runtime dependencies shared by the isolated environments.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    clang \
    cmake \
    libosmesa6 \
    ffmpeg \
    git \
    git-lfs \
    gettext-base \
    curl \
    ca-certificates \
    ninja-build \
    pkg-config \
    python3-pip \
    python3-venv \
    sudo \
    libyaml-cpp-dev \
    libeigen3-dev \
    libmsgpack-dev \
    libzmq3-dev \
    nlohmann-json3-dev \
    zlib1g-dev \
    libnvinfer10=${TENSORRT_VERSION} \
    libnvinfer-plugin10=${TENSORRT_VERSION} \
    libnvonnxparsers10=${TENSORRT_VERSION} \
  && rm -rf /var/lib/apt/lists/*

# 1b. Runtime identity, created BEFORE the checkpoint layer. OCI layers are
#     copy-on-write: chown-ing /opt/blueprint/ckpts in a later layer would
#     rewrite the ~8.7GB of checkpoint bytes into a duplicate layer, so the
#     user must exist first and ownership must be set inside the layer that
#     produces the checkpoint files.
#     IMGFIX-004: the Isaac 6 kit creates DerivedDataCache under kit/cache and
#     documents/ under kit/data at renderer startup; the base image ships them
#     writable only by uid 1234 (isaac-sim), so the review renderer wedged for
#     USER blueprint (live A40 canary, 2026-07-12). Grant write SURGICALLY:
#     non-recursive chown/chmod of exactly these directories. NEVER recurse
#     into /isaac-sim — registry copy-up would duplicate the multi-GB tree.
RUN groupadd --gid "${APP_GID}" blueprint \
  && useradd --uid "${APP_UID}" --gid "${APP_GID}" --create-home --shell /usr/sbin/nologin blueprint \
  && usermod -aG isaac-sim blueprint \
  && chown -R blueprint:blueprint /workspace \
  && mkdir -p /isaac-sim/kit/cache /isaac-sim/kit/data /isaac-sim/kit/logs \
  && chown blueprint:isaac-sim /isaac-sim/kit/cache /isaac-sim/kit/data /isaac-sim/kit/logs \
  && chmod 0775 /isaac-sim/kit/cache /isaac-sim/kit/data /isaac-sim/kit/logs

# Copy only the compiled GEAR runtime closure from the disposable builder. Do
# not copy nvcc, CUDA headers/static archives, apt state, or WBC git objects.
COPY --from=gear_sonic_builder /opt/wbc/gear_sonic_deploy /opt/wbc/gear_sonic_deploy
COPY --from=gear_sonic_builder /opt/wbc/gear_sonic /opt/wbc/gear_sonic
COPY --from=gear_sonic_builder /opt/wbc/.blueprint-source-revision /opt/wbc/.blueprint-source-revision
COPY --from=gear_sonic_builder /opt/onnxruntime /opt/onnxruntime
COPY --from=gear_sonic_builder /usr/local/bin/just /usr/local/bin/just
COPY --from=gear_sonic_builder /usr/local/cuda-12.4/targets/x86_64-linux/lib/libcudart.so* /usr/local/lib/
RUN ldconfig \
  && test "$(cat /opt/wbc/.blueprint-source-revision)" = "${WBC_SOURCE_REF}" \
  && test -x /opt/wbc/gear_sonic_deploy/target/release/g1_deploy_onnx_ref \
  && ! ldd /opt/wbc/gear_sonic_deploy/target/release/g1_deploy_onnx_ref | grep -F 'not found' \
  && chown -R blueprint:blueprint /opt/wbc/gear_sonic_deploy/build

# 2. Blueprint package in both the OSCAR client environment and Isaac runtime.
COPY pyproject.toml README.md LICENSE /opt/blueprint/blueprint-capture-pipeline/
COPY src /opt/blueprint/blueprint-capture-pipeline/src
COPY deploy/docker/robot_eval_worker/groot_oscar_closed_loop/requirements_groot_oscar_main_env.txt /opt/blueprint/requirements_groot_oscar_main_env.txt
# PEP 668: the Ubuntu 24.04 base marks system python externally managed, and
# the Debian pip cannot self-upgrade (no RECORD); install only uv and let it
# own the /opt/oscar-venv interpreter.
RUN python3 -m pip install --break-system-packages uv \
  && uv venv /opt/oscar-venv --python 3.10 \
  && VIRTUAL_ENV=/opt/oscar-venv uv pip install --index-url https://download.pytorch.org/whl/cu128 \
       torch==2.10.0 torchvision==0.25.0 \
  && VIRTUAL_ENV=/opt/oscar-venv uv pip install "nvidia-cudnn-cu12>=9.10" \
  && VIRTUAL_ENV=/opt/oscar-venv uv pip install -e /opt/blueprint/blueprint-capture-pipeline \
  && VIRTUAL_ENV=/opt/oscar-venv uv pip install -r /opt/blueprint/requirements_groot_oscar_main_env.txt \
  && /opt/oscar-venv/bin/python -c "from gear_sonic.utils.teleop.zmq.zmq_planner_sender import build_command_message, pack_pose_message; assert callable(build_command_message) and callable(pack_pose_message)" \
  && /isaac-sim/python.sh -m pip install --no-deps -e /opt/blueprint/blueprint-capture-pipeline \
  && /isaac-sim/python.sh -m pip install pillow cryptography

# 3. Official pinned OSCAR source and its torch/TE-filtered dependencies.
RUN git clone --filter=blob:none "${OSCAR_SOURCE_URL}" /opt/oscar-public \
  && git -C /opt/oscar-public fetch --depth 1 origin "${OSCAR_SOURCE_REF}" \
  && git -C /opt/oscar-public checkout --detach FETCH_HEAD \
  && test "$(git -C /opt/oscar-public rev-parse HEAD)" = "${OSCAR_SOURCE_REF}" \
  && ln -s /opt/oscar-public /opt/OSCAR \
  && /opt/oscar-venv/bin/python -c "from pathlib import Path; from blueprint_pipeline.oscar_wam_gpu_image import filter_requirements_script_text, transformer_engine_shim_script_text; Path('/tmp/filter.py').write_text(filter_requirements_script_text()); Path('/tmp/te_shim.py').write_text(transformer_engine_shim_script_text())" \
  && /opt/oscar-venv/bin/python /tmp/filter.py /opt/OSCAR/requirements.txt /tmp/oscar_requirements.txt \
  && VIRTUAL_ENV=/opt/oscar-venv uv pip install -r /tmp/oscar_requirements.txt \
  && /opt/oscar-venv/bin/python /tmp/te_shim.py /opt/OSCAR

# 4. GR00T runtime. GEAR-SONIC was compiled in gear_sonic_builder and admitted
#    above as a runtime-only dynamic closure.
RUN git clone --filter=blob:none "${GROOT_SOURCE_URL}" /opt/gr00t \
  && cd /opt/gr00t \
  && git fetch --depth 1 origin "${GROOT_SOURCE_REF}" \
  && git checkout --detach FETCH_HEAD \
  && test "$(git rev-parse HEAD)" = "${GROOT_SOURCE_REF}" \
  && uv venv /opt/gr00t-venv --python 3.10 \
  && ln -sfn /opt/gr00t-venv /opt/gr00t/.venv \
  && VIRTUAL_ENV=/opt/gr00t-venv uv pip install -e /opt/gr00t \
  && /opt/gr00t-venv/bin/python -c "from gr00t.policy.gr00t_policy import Gr00tPolicy; print('GROOT_IMPORT_OK')"

# 5. Bake both checkpoints (docker pull + go, no runtime HF pull). HF token via
#    BuildKit secret; never written into a layer.
RUN --mount=type=secret,id=hf_token,required=false <<'BASH'
set -euo pipefail
if [[ "${PREFETCH_CHECKPOINTS}" == "true" ]]; then
# The system python3.12 has no huggingface_hub (it lives in the isolated
# venvs); the oscar-venv carries it via the blueprint pipeline deps and owns
# the checkpoint prefetch.
/opt/oscar-venv/bin/python - <<'PY'
import os
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download

token = None
secret = Path("/run/secrets/hf_token")
if secret.is_file():
    token = secret.read_text(encoding="utf-8").strip()
token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

sonic = Path("/opt/blueprint/ckpts/sonic")
sonic.mkdir(parents=True, exist_ok=True)
snapshot_download(
    repo_id=os.environ["SONIC_CHECKPOINT_REPO"],
    revision=os.environ["SONIC_CHECKPOINT_REVISION"],
    local_dir=str(sonic),
    allow_patterns=[
        "config.json",
        "model.safetensors.index.json",
        "model-*.safetensors",
        "processor/*",
        "experiment_cfg/*",
        "*.json",
    ],
    # groot-bs16 ships 8 training-intermediate checkpoint-*/ dirs (~52GB); the
    # server only loads the top-level index-referenced shards. Exclude them.
    ignore_patterns=["checkpoint-*/*", "checkpoint-*"],
    token=token,
)

# The SONIC checkpoint config resolves this gated VLM by repository id during
# model construction. Seal its immutable revision into HF_HOME so the worker
# never needs a runtime token or network download.
cosmos_snapshot = Path(snapshot_download(
    repo_id=os.environ["COSMOS_BACKBONE_REPO"],
    revision=os.environ["COSMOS_BACKBONE_REVISION"],
    token=token,
))
cosmos_revision = os.environ["COSMOS_BACKBONE_REVISION"]
if cosmos_snapshot.name != cosmos_revision:
    raise RuntimeError(
        f"Cosmos snapshot mismatch: expected {cosmos_revision}, got {cosmos_snapshot.name}"
    )
# GR00T resolves the nested repo without forwarding a revision. Bind that
# default lookup to the immutable commit before the image enters offline mode.
cosmos_refs = cosmos_snapshot.parent.parent / "refs"
cosmos_refs.mkdir(parents=True, exist_ok=True)
# huggingface_hub reads refs verbatim (without stripping whitespace) before
# using the value as a snapshots/<commit> directory name. A trailing newline
# therefore makes an otherwise valid immutable ref impossible to resolve.
(cosmos_refs / "main").write_text(cosmos_revision, encoding="utf-8")

# The upstream WBC source commit intentionally does not contain deployment
# checkpoints; its documented setup downloads them from nvidia/GEAR-SONIC.
# Seal the exact immutable revision and exact files that deploy.sh opens.
gear_root = Path("/opt/wbc/gear_sonic_deploy")
gear_files = {
    "model_encoder.onnx": gear_root / "policy/release/model_encoder.onnx",
    "model_decoder.onnx": gear_root / "policy/release/model_decoder.onnx",
    "observation_config.yaml": gear_root / "policy/release/observation_config.yaml",
    "planner_sonic.onnx": gear_root / "planner/target_vel/V2/planner_sonic.onnx",
}
for remote_name, destination in gear_files.items():
    cached = hf_hub_download(
        repo_id=os.environ["GEAR_SONIC_CHECKPOINT_REPO"],
        revision=os.environ["GEAR_SONIC_CHECKPOINT_REVISION"],
        filename=remote_name,
        token=token,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(Path(cached).read_bytes())
    if destination.stat().st_size <= 0:
        raise RuntimeError(f"empty GEAR-SONIC deployment asset: {remote_name}")

# GR00T's config names the nested backbone by repository id. Transformers
# 4.57 calls model_info() while patching tokenizer regexes even when every
# file is cached, which fails correctly under HF_HUB_OFFLINE=1. Bind the
# config to a stable local symlink so full processor construction is offline.
cosmos_local = Path("/opt/blueprint/models/cosmos-reason2-2b")
cosmos_local.parent.mkdir(parents=True, exist_ok=True)
cosmos_local.symlink_to(cosmos_snapshot)
sonic_config_path = sonic / "config.json"
sonic_config = __import__("json").loads(sonic_config_path.read_text(encoding="utf-8"))
original_model_name = sonic_config.get("model_name")
if original_model_name != os.environ["COSMOS_BACKBONE_REPO"]:
    raise RuntimeError(
        f"unexpected SONIC nested model: {original_model_name!r}"
    )
sonic_config["blueprint_original_model_name"] = original_model_name
sonic_config["blueprint_model_revision"] = cosmos_revision
sonic_config["model_name"] = str(cosmos_local)
sonic_config_path.write_text(
    __import__("json").dumps(sonic_config, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

oscar = Path("/opt/blueprint/ckpts/oscar")
oscar.mkdir(parents=True, exist_ok=True)
snapshot_download(
    repo_id=os.environ["OSCAR_CHECKPOINT_REPO"],
    revision=os.environ["OSCAR_CHECKPOINT_REVISION"],
    local_dir=str(oscar),
    token=token,
)
print("BLUEPRINT_GROOT_OSCAR_CHECKPOINTS_READY")
PY
fi
# Ownership inside the producing layer: a later recursive chown would
# duplicate every checkpoint byte in registry copy-on-write layer history.
mkdir -p /opt/blueprint/ckpts
chown -R blueprint:blueprint /opt/blueprint/ckpts
chown -R blueprint:blueprint /opt/blueprint/hf_home
chown -R blueprint:blueprint /opt/blueprint/models
chown -R blueprint:blueprint /opt/wbc/gear_sonic_deploy/policy/release
chown -R blueprint:blueprint /opt/wbc/gear_sonic_deploy/planner/target_vel/V2
BASH

# Release workers resolve every model from the sealed cache. A missing nested
# backbone therefore fails locally instead of silently reaching the network.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# 6. Bake the complete official Isaac 6 G1 asset directory. The carrier omits
#    the asset pack; every downloaded byte is size- and digest-pinned here.
COPY deploy/docker/robot_eval_worker/groot_oscar_closed_loop/isaac_6_g1_assets.sha256 /opt/blueprint/isaac_6_g1_assets.sha256
COPY deploy/docker/robot_eval_worker/groot_oscar_closed_loop/fetch_pinned_isaac_assets.py /opt/blueprint/fetch_pinned_isaac_assets.py
RUN /opt/oscar-venv/bin/python /opt/blueprint/fetch_pinned_isaac_assets.py \
      --manifest /opt/blueprint/isaac_6_g1_assets.sha256 \
      --base-url https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/Robots/Unitree/G1/ \
      --output-dir /isaac-sim/Isaac/Robots/Unitree/G1

# 7. Build-time healthcheck (all three environments; no GPU required). It must
#    run AS THE RUNTIME USER: the 20260712 canary (rc=126 on every gate) proved
#    a root-only healthcheck cannot see runtime-user exec/traversal failures
#    (uv interpreter under /root, group-restricted /isaac-sim).
COPY deploy/docker/robot_eval_worker/groot_oscar_closed_loop/groot_oscar_closed_loop_image_healthcheck.py /opt/blueprint/groot_oscar_closed_loop_image_healthcheck.py
RUN runuser -u blueprint -- /opt/oscar-venv/bin/python /opt/blueprint/groot_oscar_closed_loop_image_healthcheck.py --build-time

# Name-only USER: the explicit `USER user:group` form makes the runtime skip
# supplementary-group resolution, which silently dropped the isaac-sim group
# at runtime (live canary attempt 7, 2026-07-12: `id` showed only gid 10001
# while build-time `runuser` had the group). Name-only resolves the primary
# group from /etc/passwd AND supplementary groups from /etc/group.
USER blueprint

CMD ["bash"]
