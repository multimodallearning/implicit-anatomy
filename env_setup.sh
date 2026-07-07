#!/usr/bin/env bash
set -e

ENV_NAME=${1:-ia}

source "$(conda info --base)/etc/profile.d/conda.sh"

ENVS=$(conda env list | awk '{print $1}')

if [[ $ENVS = *"$ENV_NAME"* ]]; then
    echo "[IA INFO] \"$ENV_NAME\" already exists. Using existing environment."
else
    echo "[IA INFO] Creating $ENV_NAME..."
    conda create -n "$ENV_NAME" python=3.7 -y
    echo "[IA INFO] Done!"
fi

conda activate "$ENV_NAME"

echo "[IA INFO] Installing dependencies..."
conda install pytorch=1.9.0 torchvision cudatoolkit=11.1 -c pytorch -c nvidia -y
conda install -c anaconda h5py pyyaml -y
conda install -c conda-forge sharedarray tensorboardx -y
conda install -y -c conda-forge mkl=2024.0
echo "[IA INFO] Done!"

echo "[IA INFO] Installing remaining pip requirements..."
if [ -f requirements-full.txt ]; then
    grep -vEi "^(torch|torchvision|torchaudio|pytorch|cudatoolkit)([=<> ]|$)" requirements-full.txt > requirements-no-torch.txt
    pip install -r requirements-no-torch.txt
fi
echo "[IA INFO] Done!"

echo "[IA INFO] Installing CUDA point operations..."
cd lib/pointops
python setup.py clean --all || true
rm -rf build dist *.egg-info
find . -name "*.so" -delete
python setup.py install
cd ../..
echo "[IA INFO] Done!"

NVCC="$(nvcc --version || true)"
TORCH="$(python -c "import torch; print(torch.__version__)")"

echo "[IA INFO] Testing installation..."
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda, 'CUDA available:', torch.cuda.is_available())"
python -c "import torch, pointops_cuda; print('pointops_cuda ok')"

echo "[IA INFO] Finished the installation!"
echo "[IA INFO] ========== Configurations =========="
echo "$NVCC"
echo "[IA INFO] PyTorch version: $TORCH"
echo "[IA INFO] ===================================="
