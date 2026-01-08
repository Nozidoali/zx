#!/bin/bash
set -e

echo "Creating conda environment 'zx'..."
conda create -n zx python=3.10 -y

echo "Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate zx

echo "Installing PyTorch 2.2.* (CPU)..."
conda install pytorch==2.2.0 torchvision torchaudio cpuonly -c pytorch -y

echo "Installing PyTorch Geometric..."
pip install torch-geometric torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.2.0+cpu.html

echo "Installing PyZX 0.8.*..."
pip install "pyzx>=0.8.0,<0.9.0"

echo "Installing supporting libraries..."
pip install numpy matplotlib pandas pytest

echo "Environment 'zx' created successfully!"
echo "Activate with: conda activate zx"

