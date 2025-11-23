#!/bin/bash
# First-run dependency installation script for Vast.ai instances
# Run this once when setting up a new instance

set -e

echo "=========================================="
echo "Installing dependencies for PyTorch development"
echo "=========================================="

# Update package list
echo "Updating package list..."
apt-get update

# Install system dependencies
echo "Installing system dependencies..."
apt-get install -y \
    git \
    curl \
    wget \
    vim \
    htop \
    tmux \
    build-essential \
    python3-pip \
    python3-dev \
    libssl-dev \
    libffi-dev

# Install Python packages
echo "Installing Python packages..."
pip3 install --upgrade pip setuptools wheel

# Install PyTorch (with CUDA support - adjust version as needed)
echo "Installing PyTorch..."
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install common ML/data science packages
echo "Installing ML/data science packages..."
pip3 install \
    numpy \
    pandas \
    matplotlib \
    seaborn \
    scikit-learn \
    jupyter \
    ipykernel \
    notebook

# Install experiment tracking and utilities
echo "Installing experiment tracking..."
pip3 install \
    wandb \
    tensorboard \
    tensorboardX

# Install AWS CLI for S3 checkpoint syncing (optional)
echo "Installing AWS CLI..."
pip3 install boto3 awscli

# Install other utilities
echo "Installing utilities..."
pip3 install \
    pyyaml \
    tqdm \
    psutil \
    gpustat

# Create workspace directory
echo "Creating workspace directory..."
mkdir -p /workspace/persistent
mkdir -p /workspace/checkpoints
mkdir -p /workspace/data
mkdir -p /workspace/logs

# Set permissions
chmod -R 755 /workspace

echo "=========================================="
echo "Installation complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Clone your repository to /workspace"
echo "2. Install project-specific dependencies: pip install -r requirements.txt"
echo "3. Configure your SSH config for this instance"
echo "4. Connect via VS Code Remote-SSH"

