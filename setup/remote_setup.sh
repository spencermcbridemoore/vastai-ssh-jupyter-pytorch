#!/bin/bash
# Quick setup script to run on the Vast.ai instance
# Usage: bash <(curl -s <script-url>) OR copy and paste this script

set -e

echo "=========================================="
echo "Setting up Vast.ai instance for PyTorch development"
echo "=========================================="

# Create workspace directory
echo "Creating workspace directory..."
mkdir -p /workspace
cd /workspace

# Check if repository already exists
if [ -d ".git" ]; then
    echo "Repository already exists. Pulling latest changes..."
    git pull
else
    echo "Cloning repository..."
    git clone https://github.com/spencermcbridemoore/vastai-ssh-jupyter-pytorch.git .
fi

# Install system dependencies
echo ""
echo "Installing system dependencies..."
if [ -f "setup/install_deps.sh" ]; then
    bash setup/install_deps.sh
else
    echo "Warning: install_deps.sh not found, skipping system dependencies"
fi

# Install Python dependencies
echo ""
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Verify GPU
echo ""
echo "=========================================="
echo "Verifying GPU setup..."
echo "=========================================="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

echo ""
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Connect via VS Code Remote-SSH"
echo "2. Open folder: /workspace"
echo "3. Open experiments/template_experiment.py"
echo "4. Run cells to test your setup"
echo ""

