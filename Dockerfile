FROM python:3.11-slim AS production

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md requirements.txt /app/
COPY src /app/src
COPY configs /app/configs

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -e /app && \
    pip install --no-cache-dir -r /app/requirements.txt && \
    pip check

ARG BLUEPRINTPIPELINE_GIT_URL=""
ARG BLUEPRINTPIPELINE_GIT_REF="main"

RUN if [ -n "$BLUEPRINTPIPELINE_GIT_URL" ]; then \
      if [ -n "$BLUEPRINTPIPELINE_GIT_REF" ]; then \
        git clone --depth 1 --branch "$BLUEPRINTPIPELINE_GIT_REF" "$BLUEPRINTPIPELINE_GIT_URL" /opt/BlueprintPipeline; \
      else \
        git clone --depth 1 "$BLUEPRINTPIPELINE_GIT_URL" /opt/BlueprintPipeline; \
      fi; \
      if [ -f /opt/BlueprintPipeline/requirements.txt ]; then \
        pip install --no-cache-dir -r /opt/BlueprintPipeline/requirements.txt; \
      fi; \
    fi

ENV BLUEPRINTPIPELINE_ROOT=/opt/BlueprintPipeline
ENV SWAP_POLICY_CONFIG_PATH=/app/configs/swap_policy.yaml

ENTRYPOINT ["python", "-m", "blueprint_pipeline.swap_orchestrator"]
