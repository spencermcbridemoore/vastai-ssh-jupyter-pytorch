# %% [markdown]
# # Experiment: [Name]
# Vast.ai instance: [instance_id]
# Expected cost: $[X]/hour

# %% Setup
import torch
import os
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Check environment
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"CUDA Version: {torch.version.cuda}")

# Auto-install missing packages
def install_if_missing(package, import_name=None):
    """Install package if not available"""
    if import_name is None:
        import_name = package
    try:
        __import__(import_name)
    except ImportError:
        import subprocess
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        __import__(import_name)

# Install common dependencies
install_if_missing("wandb")
install_if_missing("tensorboard", "tensorboard")
install_if_missing("pyyaml", "yaml")

import wandb
import yaml

# Set paths
WORKSPACE = Path('/workspace')
PERSISTENT = WORKSPACE / 'persistent'
PERSISTENT.mkdir(parents=True, exist_ok=True)

# Development mode flag
DEV_MODE = os.getenv('DEV_MODE', 'True').lower() == 'true'
print(f"DEV_MODE: {DEV_MODE}")

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# %% [markdown]
# ## Configuration

# %%
from src.utils.gpu_monitor import get_gpu_info, estimate_batch_size

# Load config
config_path = Path(__file__).parent.parent / 'configs' / 'dev_config.yaml'
if DEV_MODE:
    config_path = Path(__file__).parent.parent / 'configs' / 'dev_config.yaml'
else:
    config_path = Path(__file__).parent.parent / 'configs' / 'prod_config.yaml'

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# GPU info and recommendations
if torch.cuda.is_available():
    gpu_info = get_gpu_info()
    print(f"\nGPU Information:")
    print(f"  Name: {gpu_info['name']}")
    print(f"  Memory: {gpu_info['total_memory_gb']:.2f} GB")
    print(f"  Available: {gpu_info['free_memory_gb']:.2f} GB")
    
    # Estimate batch size
    recommended_batch = estimate_batch_size(
        model_params=config.get('model_params', 1e6),
        sequence_length=config.get('sequence_length', 512)
    )
    print(f"  Recommended batch size: {recommended_batch}")

# %% [markdown]
# ## Data Loading

# %%
# TODO: Implement your data loading here
# Example structure:

def load_data(config, dev_mode=False):
    """
    Load and prepare dataset.
    In dev_mode, use small subset for quick iteration.
    """
    # Example: Load your dataset
    # dataset = YourDataset(config['data_path'])
    
    # if dev_mode:
    #     # Use 10% of data for development
    #     dataset = torch.utils.data.Subset(dataset, range(len(dataset) // 10))
    
    # return dataset
    pass

# Load data
# train_dataset = load_data(config, dev_mode=DEV_MODE)
# val_dataset = load_data(config, dev_mode=False)  # Full validation set

# train_loader = torch.utils.data.DataLoader(
#     train_dataset,
#     batch_size=config.get('batch_size', 32),
#     shuffle=True,
#     num_workers=config.get('num_workers', 4)
# )

print("Data loading cell - implement your dataset here")

# %% [markdown]
# ## Model Definition

# %%
import torch.nn as nn

# TODO: Define your model here
# Example:

class ExampleModel(nn.Module):
    """Example model structure - replace with your architecture"""
    def __init__(self, config):
        super().__init__()
        # Define your layers here
        pass
    
    def forward(self, x):
        # Define forward pass
        pass

# Initialize model
# model = ExampleModel(config['model_config'])
# model = model.to(device)

# Print model info
# total_params = sum(p.numel() for p in model.parameters())
# trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(f"Total parameters: {total_params:,}")
# print(f"Trainable parameters: {trainable_params:,}")

print("Model definition cell - implement your model here")

# %% [markdown]
# ## Training Setup

# %%
from src.trainers.interruptible_trainer import InterruptibleTrainer
from src.utils.checkpoint_manager import CheckpointManager

# Initialize checkpoint manager
checkpoint_dir = PERSISTENT / 'checkpoints' / 'experiment_name'
checkpoint_manager = CheckpointManager(
    local_dir=checkpoint_dir,
    remote_bucket=config.get('s3_bucket', None),
    remote_prefix=config.get('s3_prefix', 'checkpoints')
)

# Initialize trainer
trainer = InterruptibleTrainer(
    model=None,  # Replace with your model
    device=device,
    checkpoint_manager=checkpoint_manager,
    config=config
)

# Initialize optimizer and scheduler
# optimizer = torch.optim.AdamW(
#     model.parameters(),
#     lr=config.get('learning_rate', 1e-4),
#     weight_decay=config.get('weight_decay', 0.01)
# )

# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#     optimizer,
#     T_max=config.get('max_epochs', 100)
# )

# trainer.set_optimizer(optimizer)
# trainer.set_scheduler(scheduler)

print("Training setup cell - configure optimizer, scheduler, and trainer")

# %% [markdown]
# ## Training Loop

# %%
# Initialize Weights & Biases or TensorBoard
if config.get('use_wandb', False):
    wandb.init(
        project=config.get('wandb_project', 'vastai-experiments'),
        config=config,
        resume='allow'
    )

# Resume from checkpoint if available
start_epoch = 0
if checkpoint_manager.has_checkpoint():
    print("Resuming from checkpoint...")
    checkpoint = checkpoint_manager.load_latest()
    start_epoch = checkpoint.get('epoch', 0)
    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Resuming from epoch {start_epoch}")

# Training loop
# Note: This will be handled by the InterruptibleTrainer
# trainer.train(
#     train_loader=train_loader,
#     val_loader=val_loader,
#     start_epoch=start_epoch,
#     max_epochs=config.get('max_epochs', 100),
#     checkpoint_interval=config.get('checkpoint_interval', 1000)
# )

print("Training loop cell - use trainer.train() to start training")

# %% [markdown]
# ## Visualization & Analysis

# %%
# TODO: Add visualization code here
# - Plot training curves
# - Visualize model outputs
# - Generate predictions on test set

print("Visualization cell - add your analysis code here")

# %% [markdown]
# ## Cleanup

# %%
# Final checkpoint upload
if checkpoint_manager.has_checkpoint():
    print("Uploading final checkpoint to persistent storage...")
    checkpoint_manager.upload_latest()

# Close wandb
if config.get('use_wandb', False):
    wandb.finish()

print("Experiment complete!")

