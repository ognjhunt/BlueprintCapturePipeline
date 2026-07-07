# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base

WORKDIR /app

# System deps needed by opencv-headless and trimesh
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1 \
    ffmpeg \
    git curl \
    && rm -rf /var/lib/apt/lists/*

# ---- builder stage ----
FROM base AS builder

COPY pyproject.toml uv.lock* ./
COPY src/ src/

# Install all extras needed for the Cloud Run job
RUN pip install --no-cache-dir -e ".[cloud,runtime,llm,validation,retrieval]"

# Pre-download DINOv3 ViT-L/16 weights so inference doesn't cold-start during jobs
RUN python -c "from transformers import AutoImageProcessor, AutoModel; \
    AutoImageProcessor.from_pretrained('facebook/dinov3-vitl16-pretrain-lvd1689m'); \
    AutoModel.from_pretrained('facebook/dinov3-vitl16-pretrain-lvd1689m')"

# ---- production stage ----
FROM base AS production

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app/src /app/src

# Make src importable
ENV PYTHONPATH="/app/src:${PYTHONPATH}"

ENTRYPOINT ["python", "-m", "blueprint_pipeline.capture_orchestrator"]
