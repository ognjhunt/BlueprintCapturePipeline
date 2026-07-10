FROM python:3.12-slim@sha256:423ed6ab25b1921a477529254bfeeabf5855151dc2c3141699a1bfc852199fbf AS base

ARG UV_VERSION=0.10.7
ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    HF_HOME=/opt/huggingface

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        ffmpeg \
        git \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --no-cache-dir "uv==${UV_VERSION}"


FROM base AS builder

COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src/ src/

RUN uv lock --check \
    && uv sync --frozen --no-dev --no-editable \
        --extra cloud \
        --extra runtime \
        --extra llm \
        --extra validation \
        --extra retrieval

# The model revision is immutable and the final image runs offline. Updating it
# requires an explicit reviewed Dockerfile change and a rebuilt image digest.
ARG DINOV3_MODEL_REVISION=ea8dc2863c51be0a264bab82070e3e8836b02d51
RUN /opt/venv/bin/python -c "from transformers import AutoImageProcessor, AutoModel; revision='${DINOV3_MODEL_REVISION}'; AutoImageProcessor.from_pretrained('facebook/dinov3-vitl16-pretrain-lvd1689m', revision=revision); AutoModel.from_pretrained('facebook/dinov3-vitl16-pretrain-lvd1689m', revision=revision)"


FROM base AS production

ARG APP_UID=10001
ARG APP_GID=10001
RUN groupadd --gid "${APP_GID}" blueprint \
    && useradd --uid "${APP_UID}" --gid "${APP_GID}" --create-home blueprint \
    && mkdir -p /workspace/outputs /tmp/blueprint_pipeline /opt/huggingface \
    && chown -R blueprint:blueprint /workspace /tmp/blueprint_pipeline /opt/huggingface

COPY --from=builder --chown=blueprint:blueprint /opt/venv /opt/venv
COPY --from=builder --chown=blueprint:blueprint /opt/huggingface /opt/huggingface

ENV PATH="/opt/venv/bin:${PATH}" \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PYTHONDONTWRITEBYTECODE=1

USER blueprint:blueprint
WORKDIR /workspace

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import blueprint_pipeline" || exit 1

ENTRYPOINT ["python", "-m", "blueprint_pipeline.capture_orchestrator"]


FROM builder AS development

ARG APP_UID=10001
ARG APP_GID=10001
RUN uv sync --frozen --extra dev \
    && groupadd --gid "${APP_GID}" blueprint \
    && useradd --uid "${APP_UID}" --gid "${APP_GID}" --create-home blueprint \
    && mkdir -p /workspace/outputs /tmp/blueprint_pipeline \
    && chown -R blueprint:blueprint /workspace /tmp/blueprint_pipeline /opt/venv

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1
USER blueprint:blueprint
WORKDIR /app
CMD ["bash"]
