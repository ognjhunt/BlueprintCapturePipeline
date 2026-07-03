# syntax=docker/dockerfile:1.6
ARG BASE_IMAGE=docker.io/nijelhunt/blueprint-oscar-wam@sha256:b0f3f675023d4333767d798b565fc049ac5ba788cd7041db5cac7f9784fd49b3
FROM --platform=linux/amd64 ${BASE_IMAGE}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG GROOT_SOURCE_URL=https://github.com/NVIDIA/Isaac-GR00T.git
ARG GROOT_SOURCE_REF=e5749287857afd97b78f1147166137de29746392
ARG GROOT_CHECKPOINT_REPO=LucaFrat/groot-bs16
ARG PREFETCH_GROOT_CHECKPOINT=true

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    NO_ALBUMENTATIONS_UPDATE=1 \
    HF_HOME=/opt/blueprint/hf_home \
    HUGGINGFACE_HUB_CACHE=/opt/blueprint/hf_hub_cache \
    TRANSFORMERS_CACHE=/opt/blueprint/transformers_cache \
    BLUEPRINT_WORKER_IMAGE_FAMILY=unitree-groot-sonic-wam-carrier \
    BLUEPRINT_RUNPOD_WAM_CARRIER_UNITREE_GROOT_N17_SONIC=true \
    BLUEPRINT_UNITREE_GROOT_N17_SONIC_BOOTSTRAP_MODE=system_python_minimal \
    BLUEPRINT_UNITREE_GROOT_N17_SONIC_SKIP_SYSTEM_PYTHON_DEPS_INSTALL=true \
    BLUEPRINT_UNITREE_GROOT_N17_SONIC_ROOT=/opt/blueprint/groot_runtime/Isaac-GR00T \
    BLUEPRINT_UNITREE_GROOT_N17_SONIC_REMOTE_ROOT=/opt/blueprint/groot_runtime/Isaac-GR00T \
    BLUEPRINT_UNITREE_GROOT_N17_CHECKPOINT=/opt/blueprint/groot_runtime/model_snapshots/LucaFrat__groot-bs16 \
    BLUEPRINT_RUNPOD_UNITREE_GROOT_N17_SONIC_SEALED_IMAGE_CONFIRMED=true

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    ffmpeg \
    git \
    python-is-python3 \
    python3 \
    python3-pip \
    python3-venv \
    unzip \
  && rm -rf /var/lib/apt/lists/*

COPY requirements_unitree_groot_sonic_system_python.txt /opt/blueprint/requirements_unitree_groot_sonic_system_python.txt
COPY unitree_groot_sonic_wam_image_healthcheck.py /opt/blueprint/unitree_groot_sonic_wam_image_healthcheck.py

RUN python3 -m pip install --upgrade pip setuptools wheel \
  && python3 -m pip install --no-cache-dir --ignore-requires-python --no-deps \
      -r /opt/blueprint/requirements_unitree_groot_sonic_system_python.txt

RUN mkdir -p /opt/blueprint/groot_runtime/model_snapshots /opt/blueprint/hf_home \
  && git clone --filter=blob:none --sparse "${GROOT_SOURCE_URL}" "${BLUEPRINT_UNITREE_GROOT_N17_SONIC_ROOT}" \
  && cd "${BLUEPRINT_UNITREE_GROOT_N17_SONIC_ROOT}" \
  && git fetch --depth 1 origin "${GROOT_SOURCE_REF}" \
  && git checkout FETCH_HEAD \
  && git sparse-checkout set gr00t \
  && test -f "${BLUEPRINT_UNITREE_GROOT_N17_SONIC_ROOT}/gr00t/eval/run_gr00t_server.py"

RUN --mount=type=secret,id=hf_token,required=false <<'BASH'
set -euo pipefail
if [[ "${PREFETCH_GROOT_CHECKPOINT}" == "true" ]]; then
python3 - <<'PY'
import os
from pathlib import Path
from huggingface_hub import snapshot_download

repo_id = os.environ.get("GROOT_CHECKPOINT_REPO", "LucaFrat/groot-bs16")
target = Path("/opt/blueprint/groot_runtime/model_snapshots/LucaFrat__groot-bs16")
target.mkdir(parents=True, exist_ok=True)
token = None
secret_path = Path("/run/secrets/hf_token")
if secret_path.is_file():
    token = secret_path.read_text(encoding="utf-8").strip()
snapshot_path = Path(snapshot_download(
    repo_id=repo_id,
    local_dir=str(target),
    allow_patterns=[
        "config.json",
        "model.safetensors.index.json",
        "model-*.safetensors",
        "processor/*",
    ],
    token=token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
))
processor_config = snapshot_path / "processor" / "processor_config.json"
if not processor_config.is_file():
    raise SystemExit("missing processor_config.json after checkpoint prefetch")
print("BLUEPRINT_GROOT_MODEL_SNAPSHOT_READY:" + str(snapshot_path))
PY
fi
BASH

RUN python3 /opt/blueprint/unitree_groot_sonic_wam_image_healthcheck.py --build-time

ENV PYTHONPATH="/opt/blueprint/groot_runtime/Isaac-GR00T"

CMD ["bash"]
