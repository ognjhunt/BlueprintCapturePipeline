# syntax=docker/dockerfile:1.6
#
# Sealed worker image for the GR00T N1.7 + SONIC x OSCAR-2B closed-loop eval
# (oscar_isaac_closed_loop_eval + groot_sonic_policy_endpoint, the "T4" lane).
#
# Extends the digest-pinned blueprint-oscar-wam base (which already freezes the
# hard archaeology: CUDA 12.8, cuDNN headers on CPATH, torch 2.10.0+cu128,
# transformer-engine, OSCAR source at /opt/oscar-public + its worldsim deps) and
# adds the last mile that used to cost 40-70 min per pod: the GR00T Python-3.10
# venv, both model checkpoints, our package, and the osmesa/mujoco main-env deps.
#
# Build context = repo root (so `src` + pyproject are COPY-able). Checkpoints are
# fetched INSIDE the build (BuildKit secret hf_token), never from the context.
#
# This mirrors deploy/docker/robot_eval_worker/unitree_groot_sonic_wam/Dockerfile
# but for the OSCAR-in-the-loop closed loop, and bakes checkpoints for docker
# pull + go. It proves build/runtime readiness only — not task success.

ARG BASE_IMAGE=docker.io/nijelhunt/blueprint-oscar-wam@sha256:b0f3f675023d4333767d798b565fc049ac5ba788cd7041db5cac7f9784fd49b3
FROM --platform=linux/amd64 ${BASE_IMAGE}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG DEBIAN_FRONTEND=noninteractive
ARG GROOT_SOURCE_URL=https://github.com/NVIDIA/Isaac-GR00T.git
ARG GROOT_SOURCE_REF=e5749287857afd97b78f1147166137de29746392
ARG WBC_SOURCE_URL=https://github.com/NVlabs/GR00T-WholeBodyControl.git
ARG SONIC_CHECKPOINT_REPO=LucaFrat/groot-bs16
ARG OSCAR_CHECKPOINT_REPO=zywu2115/OSCAR-2B
ARG OSCAR_CHECKPOINT_REVISION=c9781ffa7dd8556d862d7d9f338a2ea008a58ca6
ARG PREFETCH_CHECKPOINTS=true

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MUJOCO_GL=osmesa \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    HF_HOME=/opt/blueprint/hf_home \
    BLUEPRINT_OSCAR_WAM_HF_REVISION=c9781ffa7dd8556d862d7d9f338a2ea008a58ca6 \
    PYTHONPATH=/opt/OSCAR \
    BLUEPRINT_GROOT_OSCAR_OSCAR_REPO=/opt/OSCAR \
    BLUEPRINT_GROOT_OSCAR_OSCAR_CHECKPOINT=/opt/blueprint/ckpts/oscar \
    BLUEPRINT_GROOT_OSCAR_GROOT_ROOT=/opt/gr00t \
    BLUEPRINT_GROOT_OSCAR_GROOT_VENV_PYTHON=/opt/gr00t-venv/bin/python \
    BLUEPRINT_GROOT_OSCAR_SONIC_CHECKPOINT=/opt/blueprint/ckpts/sonic \
    BLUEPRINT_UNITREE_GROOT_N17_SONIC_ROOT=/opt/gr00t \
    BLUEPRINT_UNITREE_GROOT_N17_SONIC_WBC_ROOT=/opt/wbc \
    BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_SERVER_URL=tcp://127.0.0.1:5550 \
    BLUEPRINT_WORKER_IMAGE_FAMILY=groot-oscar-closed-loop-eval \
    BLUEPRINT_GROOT_OSCAR_CLOSED_LOOP_SEALED_IMAGE_CONFIRMED=true \
    SONIC_CHECKPOINT_REPO=${SONIC_CHECKPOINT_REPO} \
    OSCAR_CHECKPOINT_REPO=${OSCAR_CHECKPOINT_REPO} \
    OSCAR_CHECKPOINT_REVISION=${OSCAR_CHECKPOINT_REVISION}

WORKDIR /workspace

# 1. System packages on top of the OSCAR base. libosmesa6 is required for
#    MUJOCO_GL=osmesa (EGL is broken on the pytorch base per the ACT/T4 runs).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libosmesa6 \
    ffmpeg \
    git \
    curl \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# 2. Expose the base's OSCAR source at the /opt/OSCAR path the closed-loop CLI
#    expects (--oscar-repo default + PYTHONPATH).
RUN if [[ ! -e /opt/OSCAR ]]; then ln -s /opt/oscar-public /opt/OSCAR; fi

# 3. Our package + closed-loop main-env deps. Installed WITHOUT extras so the
#    base torch 2.10.0+cu128 is never disturbed (matches run_t4.sh's line).
COPY pyproject.toml README.md /opt/blueprint/blueprint-capture-pipeline/
COPY src /opt/blueprint/blueprint-capture-pipeline/src
COPY deploy/docker/robot_eval_worker/groot_oscar_closed_loop/requirements_groot_oscar_main_env.txt /opt/blueprint/requirements_groot_oscar_main_env.txt
RUN python3 -m pip install --upgrade pip \
  && python3 -m pip install -e /opt/blueprint/blueprint-capture-pipeline \
  && python3 -m pip install -r /opt/blueprint/requirements_groot_oscar_main_env.txt

# 4. GR00T runtime + its own Python-3.10 venv (Isaac-GR00T pins python==3.10.*).
#    WBC is cloned for the endpoint lane; harmless for the T4 closed loop.
RUN git clone --filter=blob:none --depth 1 "${WBC_SOURCE_URL}" /opt/wbc
RUN git clone --filter=blob:none "${GROOT_SOURCE_URL}" /opt/gr00t \
  && cd /opt/gr00t \
  && git fetch --depth 1 origin "${GROOT_SOURCE_REF}" \
  && git checkout FETCH_HEAD \
  && uv venv /opt/gr00t-venv --python 3.10 \
  && ln -sfn /opt/gr00t-venv /opt/gr00t/.venv \
  && VIRTUAL_ENV=/opt/gr00t-venv uv pip install -e /opt/gr00t \
  && /opt/gr00t-venv/bin/python -c "from gr00t.policy.gr00t_policy import Gr00tPolicy; print('GROOT_IMPORT_OK')"

# 5. Bake both checkpoints (docker pull + go, no runtime HF pull). HF token via
#    BuildKit secret; never written into a layer.
RUN --mount=type=secret,id=hf_token,required=false <<'BASH'
set -euo pipefail
if [[ "${PREFETCH_CHECKPOINTS}" == "true" ]]; then
python3 - <<'PY'
import os
from pathlib import Path
from huggingface_hub import snapshot_download

token = None
secret = Path("/run/secrets/hf_token")
if secret.is_file():
    token = secret.read_text(encoding="utf-8").strip()
token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

sonic = Path("/opt/blueprint/ckpts/sonic")
sonic.mkdir(parents=True, exist_ok=True)
snapshot_download(
    repo_id=os.environ["SONIC_CHECKPOINT_REPO"],
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
BASH

# 6. Build-time healthcheck (fail-closed on the build host; no GPU required).
COPY deploy/docker/robot_eval_worker/groot_oscar_closed_loop/groot_oscar_closed_loop_image_healthcheck.py /opt/blueprint/groot_oscar_closed_loop_image_healthcheck.py
RUN python3 /opt/blueprint/groot_oscar_closed_loop_image_healthcheck.py --build-time

CMD ["bash"]
