# syntax=docker/dockerfile:1.7
# Diagnostic-only extension of the sealed OpenPI worker. Ctrl-World runs in a
# separate Python environment and shares the one admitted GPU sequentially.

FROM docker.io/nijelhunt/blueprint-openpi-policy-ranking@sha256:badde151b360cd5227ede3e4e037b5e12f36ed59ac09ca89470a22511e72c6c4

ARG CTRL_WORLD_SOURCE_URL=https://github.com/Robert-gyj/Ctrl-World.git
ARG CTRL_WORLD_SOURCE_REVISION=99fb20683fd79dfa6d0c6feb9d49c6c55eecd50d
ARG BLUEPRINT_SOURCE_COMMIT

ENV BLUEPRINT_SOURCE_COMMIT=${BLUEPRINT_SOURCE_COMMIT} \
    BLUEPRINT_CTRL_WORLD_PYTHON=/.ctrl-world-venv/bin/python \
    BLUEPRINT_CTRL_WORLD_MODEL_ROOT=/workspace/ctrl-world-models \
    BLUEPRINT_CTRL_WORLD_SOURCE_ROOT=/opt/ctrl-world \
    BLUEPRINT_CTRL_WORLD_SOURCE_MANIFEST=/opt/ctrl-world-runtime/ctrl_world_source_manifest.json \
    BLUEPRINT_CTRL_WORLD_CHECKPOINT=/workspace/ctrl-world-models/ctrl_world/8cf814693f411962dc866a2ddb5b785afd17a93a/checkpoint-10000.pt \
    BLUEPRINT_CTRL_WORLD_SVD_ROOT=/workspace/ctrl-world-models/stable_video_diffusion/9cf024d5bfa8f56622af86c884f26a52f6676f2e \
    BLUEPRINT_CTRL_WORLD_CLIP_ROOT=/workspace/ctrl-world-models/clip/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268 \
    BLUEPRINT_CTRL_WORLD_STATE_STATS=/opt/ctrl-world/dataset_meta_info/droid/stat.json \
    BLUEPRINT_CTRL_WORLD_TIMEOUT_SECONDS=3600 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false

RUN git clone --filter=blob:none --no-checkout "${CTRL_WORLD_SOURCE_URL}" /opt/ctrl-world \
    && git -C /opt/ctrl-world fetch --depth 1 origin "${CTRL_WORLD_SOURCE_REVISION}" \
    && git -C /opt/ctrl-world checkout --detach FETCH_HEAD \
    && test "$(git -C /opt/ctrl-world rev-parse HEAD)" = "${CTRL_WORLD_SOURCE_REVISION}" \
    && test -z "$(git -C /opt/ctrl-world status --short --untracked-files=no)"

COPY deploy/docker/policy_ranking_openpi_ctrl_world/requirements.lock /opt/ctrl-world-runtime/requirements.lock
COPY deploy/docker/policy_ranking_openpi_ctrl_world/ctrl_world_source_manifest.json /opt/ctrl-world-runtime/ctrl_world_source_manifest.json
COPY pyproject.toml README.md LICENSE /opt/blueprint-current/
COPY src /opt/blueprint-current/src

RUN export UV_HTTP_TIMEOUT=300 UV_HTTP_RETRIES=5 \
    && uv venv --python 3.11.9 /.ctrl-world-venv \
    && uv pip install --python /.ctrl-world-venv/bin/python \
      --index-url https://download.pytorch.org/whl/cu128 \
      torch==2.7.1 torchvision==0.22.1 \
    && uv pip install --python /.ctrl-world-venv/bin/python \
      --requirement /opt/ctrl-world-runtime/requirements.lock \
    && VIRTUAL_ENV=/.venv uv pip install --no-deps -e /opt/blueprint-current \
    && VIRTUAL_ENV=/.ctrl-world-venv uv pip install --no-deps -e /opt/blueprint-current \
    && /.venv/bin/python -m compileall -q /opt/blueprint-current/src/blueprint_pipeline \
    && /.ctrl-world-venv/bin/python -m compileall -q /opt/blueprint-current/src/blueprint_pipeline \
    && /.ctrl-world-venv/bin/python -c "import torch; assert torch.__version__ == '2.7.1+cu128'; import accelerate, decord, diffusers, einops, mediapy, swanlab, transformers, wandb" \
    && /.ctrl-world-venv/bin/python -c "from pathlib import Path; from blueprint_pipeline.ctrl_world_joint_position_reference_runtime import _validate_source_manifest; result=_validate_source_manifest(Path('/opt/ctrl-world'), Path('/opt/ctrl-world-runtime/ctrl_world_source_manifest.json')); assert result['file_count'] == 8"

WORKDIR /workspace
ENTRYPOINT ["/.venv/bin/python", "-m", "blueprint_pipeline.openpi_policy_ranking_gpu_bootstrap", "run"]
