#!/usr/bin/env bash
# Scaffold-GS setup for NVIDIA RTX 50-series (Blackwell / sm_120) on Ubuntu 24+
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${ENV_NAME:-scaffold_gs}"

source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    conda create -n "$ENV_NAME" python=3.11 pip -y
fi

conda activate "$ENV_NAME"

# PyTorch 2.7+ with CUDA 12.8+ is required for RTX 5080 (sm_120)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

pip install plyfile tqdm einops wandb lpips laspy colorama opencv-python tensorboard jaxtyping
pip install torch-scatter -f https://data.pyg.org/whl/torch-$(python -c "import torch; print(torch.__version__)").html

# CUDA compiler toolchain (match PyTorch cu128; GCC < 14 required by CUDA 12.8)
conda install -y -c nvidia cuda-nvcc=12.8 cuda-cudart-dev
conda install -y -c conda-forge gxx_linux-64=13.2.0 ninja

ln -sf "$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++" "$CONDA_PREFIX/bin/g++"
ln -sf "$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc" "$CONDA_PREFIX/bin/gcc"

export CUDA_HOME="$CONDA_PREFIX"
export TORCH_CUDA_ARCH_LIST="12.0"
export PATH="$CONDA_PREFIX/bin:$PATH"
unset NVCC_PREPEND_FLAGS NVCC_PREPEND_FLAGS_BACKUP CUDAHOSTCC CUDAHOSTCXX

cd "$PROJECT_DIR"
pip install --no-build-isolation submodules/simple-knn submodules/diff-gaussian-rasterization

mkdir -p data

python - <<'PY'
import torch
import simple_knn
import diff_gaussian_rasterization
from scene.gaussian_model import GaussianModel
assert torch.cuda.is_available(), "CUDA not available"
print("Setup OK:", torch.__version__, "|", torch.cuda.get_device_name(0))
PY

echo ""
echo "Done. Activate with: conda activate $ENV_NAME"
echo "Place datasets under: $PROJECT_DIR/data/"
echo "Train a scene with: bash train_method.sh --method baseline -d <dataset/scene> -g 0"
