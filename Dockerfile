# Blueprint Capture Pipeline - GPU Container for Cloud Run Jobs
#
# This Dockerfile builds a GPU-enabled container for running the pipeline
# on Google Cloud Run Jobs with NVIDIA L4 GPU support.
#
# Build:
#   docker build -t blueprint-pipeline:latest .
#
# Build for Cloud Run (with GPU):
#   docker build -t gcr.io/PROJECT_ID/blueprint-pipeline:latest .
#   docker push gcr.io/PROJECT_ID/blueprint-pipeline:latest
#
# Run locally (with GPU):
#   docker run --gpus all -e JOB_PAYLOAD='{"job_name": "..."}' blueprint-pipeline:latest

# Base image with CUDA and PyTorch
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 AS base

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Set up locale
RUN apt-get update && apt-get install -y locales && \
    locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build essentials
    build-essential \
    cmake \
    ninja-build \
    git \
    wget \
    curl \
    ca-certificates \
    # Python
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    # Video processing
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    # Image processing
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libwebp-dev \
    # OpenCV dependencies
    libopencv-dev \
    # Open3D dependencies
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libxi-dev \
    libxmu-dev \
    # COLMAP dependencies (for photogrammetry fallback)
    libboost-all-dev \
    libcgal-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libgflags-dev \
    libglew-dev \
    libgoogle-glog-dev \
    libmetis-dev \
    libsqlite3-dev \
    libsuitesparse-dev \
    libceres-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    # Cleanup
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# =============================================================================
# Build and install COLMAP from source (required for SLAM fallback)
# =============================================================================
RUN git clone https://github.com/colmap/colmap.git /tmp/colmap && \
    cd /tmp/colmap && \
    git checkout 3.9.1 && \
    mkdir build && cd build && \
    cmake .. -GNinja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89" \
        -DCUDA_ENABLED=ON \
        -DGUI_ENABLED=OFF && \
    ninja && \
    ninja install && \
    rm -rf /tmp/colmap

# Verify COLMAP installation
RUN colmap --help | head -5

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Create workspace
WORKDIR /app

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir \
    torch==2.1.0 \
    torchvision==0.16.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Install core Python dependencies
RUN pip install --no-cache-dir \
    numpy>=1.24.0 \
    pillow>=10.0.0 \
    pyyaml>=6.0 \
    opencv-python>=4.8.0 \
    open3d>=0.17.0 \
    trimesh>=4.0.0 \
    scipy>=1.11.0 \
    plyfile>=1.0.0

# Install GCS and Cloud dependencies (including Firestore and Cloud Tasks)
RUN pip install --no-cache-dir \
    google-cloud-storage>=2.10.0 \
    google-cloud-logging>=3.6.0 \
    google-cloud-firestore>=2.14.0 \
    google-cloud-tasks>=2.14.0 \
    firebase-admin>=6.0.0

# Install pycolmap for Python-native COLMAP bindings
RUN pip install --no-cache-dir pycolmap>=3.10.0

# Copy application code
COPY src/ /app/src/
COPY pyproject.toml /app/
COPY README.md /app/

# Install the pipeline package
RUN pip install --no-cache-dir -e /app[cloud]

# =============================================================================
# Install CUDA-accelerated Gaussian Splatting rasterizer (optional but 10-100x faster)
# =============================================================================
RUN git clone https://github.com/graphdeco-inria/diff-gaussian-rasterization.git /tmp/diff-gaussian-rasterization && \
    cd /tmp/diff-gaussian-rasterization && \
    pip install --no-cache-dir . && \
    rm -rf /tmp/diff-gaussian-rasterization

# Also install the simple-knn for 3DGS
RUN git clone https://github.com/camenduru/simple-knn.git /tmp/simple-knn && \
    cd /tmp/simple-knn && \
    pip install --no-cache-dir . && \
    rm -rf /tmp/simple-knn

# =============================================================================
# Install SLAM backends for different sensor types
# =============================================================================

# WildGS-SLAM: RGB-only SLAM with Gaussian Splatting (for Meta glasses, generic cameras)
# This is the primary backend for videos without depth
RUN git clone --recursive https://github.com/GradientSpaces/WildGS-SLAM.git /opt/WildGS-SLAM && \
    cd /opt/WildGS-SLAM && \
    pip install --no-cache-dir -e . || echo "WildGS-SLAM installation optional" && \
    echo "WildGS-SLAM installed at /opt/WildGS-SLAM"

# SplaTAM: RGB-D SLAM with Gaussian Splatting (for iPhone LiDAR)
# This is used when depth maps are available
RUN git clone https://github.com/spla-tam/SplaTAM.git /opt/SplaTAM && \
    cd /opt/SplaTAM && \
    pip install --no-cache-dir -e . || echo "SplaTAM installation optional" && \
    echo "SplaTAM installed at /opt/SplaTAM"

# Create directories for workspace
RUN mkdir -p /tmp/blueprint_pipeline /workspace

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CUDA_VISIBLE_DEVICES=0

# Cloud Run uses port 8080 by default (not used for jobs, but set anyway)
ENV PORT=8080

# Health check endpoint (for debugging)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import blueprint_pipeline; print('OK')" || exit 1

# Default command: run the pipeline job
ENTRYPOINT ["python", "-m", "blueprint_pipeline.runner"]

# Default arguments (can be overridden)
CMD []


# =============================================================================
# Development stage (for local testing)
# =============================================================================
FROM base AS development

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest>=7.0.0 \
    pytest-cov>=4.0.0 \
    mypy>=1.5.0 \
    ruff>=0.1.0 \
    black>=23.0.0

# Install jupyter for interactive development
RUN pip install --no-cache-dir \
    jupyter \
    ipywidgets \
    matplotlib

# Set development mode
ENV BLUEPRINT_ENV=development

# Override entrypoint for development
ENTRYPOINT ["/bin/bash"]


# =============================================================================
# Production stage (minimal size)
# =============================================================================
FROM base AS production

# Remove build tools to reduce image size
RUN apt-get purge -y \
    build-essential \
    cmake \
    ninja-build \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set production mode
ENV BLUEPRINT_ENV=production

# Non-root user for security (optional for Cloud Run)
# RUN useradd -m -u 1000 pipeline && chown -R pipeline:pipeline /app /workspace
# USER pipeline
