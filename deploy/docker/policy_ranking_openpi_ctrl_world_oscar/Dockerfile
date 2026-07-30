# syntax=docker/dockerfile:1.7
# Diagnostic-only OpenPI + Ctrl-World + OSCAR worker. Each model family runs in
# its own Python environment and the admitted GPU is shared sequentially.

# Use the immutable production monolith as the OSCAR donor. The smaller legacy
# donor omitted packages that are hard-imported by the official OSCAR graph.
FROM docker.io/nijelhunt/blueprint-groot-oscar-eval@sha256:ab8fbccb714242b55811aa5142933001dfba76d56b5cc29dead4d0bdf1346e88 AS oscar_runtime

FROM docker.io/nijelhunt/blueprint-openpi-policy-ranking@sha256:badde151b360cd5227ede3e4e037b5e12f36ed59ac09ca89470a22511e72c6c4

ARG CTRL_WORLD_SOURCE_URL=https://github.com/Robert-gyj/Ctrl-World.git
ARG CTRL_WORLD_SOURCE_REVISION=99fb20683fd79dfa6d0c6feb9d49c6c55eecd50d
ARG OSCAR_SOURCE_URL=https://github.com/wuzy2115/oscar-public.git
ARG OSCAR_SOURCE_REVISION=4dea2f657e221b0ff24c895fcc8ab4d46d5a9adb
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
    BLUEPRINT_OSCAR_RUNTIME_PYTHON=/opt/oscar-venv/bin/python \
    BLUEPRINT_OSCAR_RUNTIME_REPO=/opt/OSCAR \
    BLUEPRINT_OSCAR_RUNTIME_CHECKPOINT=/opt/blueprint/ckpts/oscar \
    BLUEPRINT_OSCAR_RUNTIME_SOURCE_SEAL=/opt/blueprint/oscar_source_provenance.json \
    BLUEPRINT_OSCAR_RUNTIME_ASSET_CACHE=/opt/blueprint/oscar-runtime-assets \
    HF_HOME=/opt/blueprint/oscar-runtime-assets/hf_home \
    HF_HUB_CACHE=/opt/blueprint/oscar-runtime-assets/hf_home/hub \
    HUGGINGFACE_HUB_CACHE=/opt/blueprint/oscar-runtime-assets/hf_home/hub \
    COSMOS_REASON_PATH=/opt/blueprint/oscar-runtime-assets/hf_home/hub/models--nvidia--Cosmos-Reason1-7B/snapshots/3210bec0495fdc7a8d3dbb8d58da5711eab4b423 \
    WAN_VAE_PATH=/opt/blueprint/oscar-runtime-assets/hf_home/hub/models--Wan-AI--Wan2.1-T2V-1.3B/snapshots/37ec512624d61f7aa208f7ea8140a131f93afc9a/Wan2.1_VAE.pth \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    PYTHONPATH=/opt/OSCAR \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    XLA_PYTHON_CLIENT_PREALLOCATE=false

COPY --from=oscar_runtime /opt/uv-python /opt/uv-python
COPY --from=oscar_runtime /opt/oscar-venv /opt/oscar-venv
COPY --from=oscar_runtime /opt/blueprint/ckpts/oscar /opt/blueprint/ckpts/oscar

RUN git clone --filter=blob:none --no-checkout "${CTRL_WORLD_SOURCE_URL}" /opt/ctrl-world \
    && git -C /opt/ctrl-world fetch --depth 1 origin "${CTRL_WORLD_SOURCE_REVISION}" \
    && git -C /opt/ctrl-world checkout --detach FETCH_HEAD \
    && test "$(git -C /opt/ctrl-world rev-parse HEAD)" = "${CTRL_WORLD_SOURCE_REVISION}" \
    && test -z "$(git -C /opt/ctrl-world status --short --untracked-files=no)" \
    && git clone --filter=blob:none --no-checkout "${OSCAR_SOURCE_URL}" /opt/OSCAR \
    && git -C /opt/OSCAR fetch --depth 1 origin "${OSCAR_SOURCE_REVISION}" \
    && git -C /opt/OSCAR checkout --detach FETCH_HEAD \
    && test "$(git -C /opt/OSCAR rev-parse HEAD)" = "${OSCAR_SOURCE_REVISION}" \
    && test -z "$(git -C /opt/OSCAR status --short --untracked-files=no)"

COPY deploy/docker/policy_ranking_openpi_ctrl_world/requirements.lock /opt/ctrl-world-runtime/requirements.lock
COPY deploy/docker/policy_ranking_openpi_ctrl_world/ctrl_world_source_manifest.json /opt/ctrl-world-runtime/ctrl_world_source_manifest.json
COPY deploy/docker/robot_eval_worker/groot_oscar_closed_loop/requirements_embedded_carrier_opencv.lock /opt/oscar-runtime/requirements_embedded_carrier_opencv.lock
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
    && uv pip install --python /opt/oscar-venv/bin/python --no-deps -e /opt/blueprint-current \
    && /opt/oscar-venv/bin/python -c "from pathlib import Path; from blueprint_pipeline.oscar_wam_gpu_image import transformer_engine_shim_script_text; Path('/tmp/oscar_te_shim.py').write_text(transformer_engine_shim_script_text(), encoding='utf-8')" \
    && /opt/oscar-venv/bin/python /tmp/oscar_te_shim.py /opt/OSCAR \
    && find /opt/OSCAR -type d -name __pycache__ -prune -exec rm -rf '{}' + \
    && find /opt/OSCAR -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete \
    && /opt/oscar-venv/bin/python -m blueprint_pipeline.oscar_runtime_source_provenance seal \
      --source-root /opt/OSCAR \
      --output /opt/blueprint/oscar_source_provenance.json \
      --source-url "${OSCAR_SOURCE_URL}" \
      --source-commit "${OSCAR_SOURCE_REVISION}" \
      --runtime-source-root /opt/OSCAR \
    && chmod 0444 /opt/blueprint/oscar_source_provenance.json \
    && rm -rf /opt/OSCAR/.git /tmp/oscar_te_shim.py \
    && /.venv/bin/python -m compileall -q /opt/blueprint-current/src/blueprint_pipeline \
    && /.ctrl-world-venv/bin/python -m compileall -q /opt/blueprint-current/src/blueprint_pipeline \
    && /.ctrl-world-venv/bin/python -c "import torch; assert torch.__version__ == '2.7.1+cu128'; import accelerate, decord, diffusers, einops, mediapy, swanlab, transformers, wandb" \
    && /.ctrl-world-venv/bin/python -c "from pathlib import Path; from blueprint_pipeline.ctrl_world_joint_position_reference_runtime import _validate_source_manifest; result=_validate_source_manifest(Path('/opt/ctrl-world'), Path('/opt/ctrl-world-runtime/ctrl_world_source_manifest.json')); assert result['file_count'] == 8" \
    && if ! /opt/oscar-venv/bin/python -c 'import cv2'; then \
      uv pip install --python /opt/oscar-venv/bin/python \
        --no-cache --no-deps --require-hashes \
        -r /opt/oscar-runtime/requirements_embedded_carrier_opencv.lock; \
      fi \
    && /opt/oscar-venv/bin/python -c "import cv2; assert cv2.__version__ == '4.11.0'" \
    && /opt/oscar-venv/bin/python -c "import torch, cv2; from inference.inference_oscar import build_pipeline; assert callable(build_pipeline)"

RUN env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN \
      HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 \
      /opt/oscar-venv/bin/python - <<'PY'
from pathlib import Path
from blueprint_pipeline.oscar_runtime_asset_contract import prepare_runtime_asset_cache

result = prepare_runtime_asset_cache(
    Path('/opt/blueprint/oscar-runtime-assets'),
    token=False,
)
assert result['status'] == 'passed', result['blockers']
PY

RUN /opt/oscar-venv/bin/python - <<'PY'
from pathlib import Path
from blueprint_pipeline.oscar_runtime_asset_contract import offline_preflight
from blueprint_pipeline.oscar_runtime_source_provenance import verify_source_tree

source = verify_source_tree(
    source_root=Path('/opt/OSCAR'),
    seal_path=Path('/opt/blueprint/oscar_source_provenance.json'),
    artifact_path=Path('/tmp/oscar_runtime_source_provenance.json'),
)
assert source['status'] == 'passed', source['blockers']
assets = offline_preflight(
    Path('/opt/blueprint/oscar-runtime-assets'),
    oscar_checkpoint_root=Path('/opt/blueprint/ckpts/oscar'),
    evidence_output_path=Path('/opt/blueprint/oscar-runtime-assets/oscar_runtime_asset_offline_preflight.json'),
)
assert assets['status'] == 'passed', assets['blockers']
PY

WORKDIR /workspace
ENTRYPOINT ["/.venv/bin/python", "-m", "blueprint_pipeline.openpi_policy_ranking_gpu_bootstrap", "run"]
