#!/bin/bash
# Setup script for CUDA-accelerated 3DGS pipeline
# Run this on a machine with CUDA installed (e.g., Cloud Run GPU instance)

set -e

echo "🔧 Setting up CUDA-accelerated 3D Gaussian Splatting..."

# Check CUDA
if [ -z "$CUDA_HOME" ]; then
    if [ -d "/usr/local/cuda" ]; then
        export CUDA_HOME=/usr/local/cuda
    elif [ -d "/opt/cuda" ]; then
        export CUDA_HOME=/opt/cuda
    else
        echo "❌ CUDA_HOME not set and CUDA not found in standard locations"
        echo "   Please install CUDA or set CUDA_HOME environment variable"
        exit 1
    fi
fi

echo "   CUDA_HOME: $CUDA_HOME"

# Verify nvcc
if ! command -v nvcc &> /dev/null; then
    export PATH="$CUDA_HOME/bin:$PATH"
fi

echo "   CUDA version: $(nvcc --version | grep release)"

# Install PyTorch with CUDA support
echo "📦 Installing PyTorch with CUDA support..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install CUDA-accelerated rasterizer (10-100x faster)
echo "📦 Installing diff-gaussian-rasterization (CUDA rasterizer)..."
pip install git+https://github.com/graphdeco-inria/diff-gaussian-rasterization.git

# Install simple-knn for point cloud initialization
echo "📦 Installing simple-knn..."
pip install git+https://github.com/camenduru/simple-knn.git

# Verify installation
echo "🧪 Verifying installation..."
python -c "
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from simple_knn._C import distCUDA2
import torch
print('✅ CUDA rasterizer installed successfully!')
print(f'   PyTorch CUDA: {torch.cuda.is_available()}')
print(f'   GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')
"

echo "✅ CUDA setup complete!"
echo ""
echo "To test the pipeline:"
echo "  python -m blueprint_pipeline.runner --help"
