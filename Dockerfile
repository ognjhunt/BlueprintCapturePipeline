FROM python:3.11-slim AS production

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
# Prevent any runtime downloads from pip or HuggingFace
ENV PIP_NO_INPUT=1
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    openssh-client \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md requirements.txt /app/
COPY src /app/src
COPY configs /app/configs
COPY scripts /app/scripts

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -e /app && \
    pip install --no-cache-dir -r /app/requirements.txt && \
    pip check

# Clone only the required BlueprintPipeline subdirectories (~2MB vs 13GB)
# Needed for stages D/F/H: asset materialization, interactive validation,
# simready prep, USD assembly
ARG BLUEPRINTPIPELINE_GIT_URL="https://github.com/ognjhunt/BlueprintPipeline.git"
ARG BLUEPRINTPIPELINE_GIT_REF="main"
RUN git clone --depth 1 --branch "$BLUEPRINTPIPELINE_GIT_REF" \
      --filter=blob:none --sparse \
      "$BLUEPRINTPIPELINE_GIT_URL" /opt/BlueprintPipeline && \
    cd /opt/BlueprintPipeline && \
    git sparse-checkout set \
      blueprint_sim \
      interactive-job \
      monitoring \
      simready-job \
      tools \
      usd-assembly-job && \
    rm -rf .git && \
    if [ -f /opt/BlueprintPipeline/interactive-job/requirements.txt ]; then \
      pip install --no-cache-dir -r /opt/BlueprintPipeline/interactive-job/requirements.txt 2>/dev/null || true; \
    fi

# Verify all dependencies are installed and importable at build time
RUN python -c "\
import yaml; import numpy; import trimesh; import pybullet; \
import pybullet_data; \
print('All dependencies verified - no runtime downloads needed')"

ENV BLUEPRINTPIPELINE_ROOT=/opt/BlueprintPipeline
ENV SWAP_POLICY_CONFIG_PATH=/app/configs/swap_policy.yaml
ENV GCS_ROOT=/mnt/gcs
# Ensure scripts/ is importable (nurec_shim.py imports sam3_detect)
ENV PYTHONPATH=/app/scripts

ENTRYPOINT ["python", "-m", "blueprint_pipeline.capture_orchestrator"]
