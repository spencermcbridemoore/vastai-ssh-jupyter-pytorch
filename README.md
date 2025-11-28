# PyTorch Remote Development Workflow for Vast.ai GPU Clusters

A production-ready repository structure for remote GPU development on Vast.ai instances using VS Code/Cursor's Remote-SSH features. This setup enables seamless transition from local prototyping to distributed GPU training with cost-optimized workflows.

## Features

- 🔄 **Cell-based Development**: Python files with `# %%` markers for interactive execution (Jupyter-like workflow)
- 🚀 **Remote-First Design**: Optimized for SSH-based remote development on Vast.ai instances
- 💰 **Cost-Optimized**: Easy scaling from cheap instances ($0.20/hr) to powerful GPUs
- 🛡️ **Preemption Handling**: Automatic checkpointing and graceful handling of instance interruptions
- 📊 **Experiment Tracking**: Integrated support for Weights & Biases and TensorBoard
- 🔧 **Development Tools**: GPU monitoring, cost tracking, and batch size estimation

## Repository Structure

```
project/
├── setup/
│   ├── ssh_config.template      # Template for ~/.ssh/config Vast.ai entries
│   ├── install_deps.sh          # First-run dependency installation
│   └── vscode_extensions.txt    # Required VS Code extensions list
├── experiments/
│   ├── template_experiment.py   # Template with cell markers for new experiments
│   └── README.md                # How to run experiments locally vs remote
├── src/
│   ├── trainers/
│   │   └── interruptible_trainer.py  # Trainer that handles preemption
│   ├── utils/
│   │   ├── gpu_monitor.py      # GPU usage and cost tracking utilities
│   │   └── checkpoint_manager.py # S3/persistent storage checkpoint handling
│   └── models/
├── configs/
│   ├── dev_config.yaml          # Small-scale testing configuration  
│   └── prod_config.yaml         # Full training configuration
└── scripts/
    ├── launch_vast.py           # Helper to find and connect to Vast.ai instances
    └── sync_outputs.sh          # Sync results from instance to local/S3
```

## Quick Start

### 1. Initial Setup

#### On Your Local Machine

1. **Install VS Code Extensions**:
   ```bash
   # Install extensions listed in setup/vscode_extensions.txt
   # Or use VS Code's extension marketplace
   ```

2. **Configure SSH**:
   ```bash
   # Copy the SSH config template
   cp setup/ssh_config.template ~/.ssh/config
   
   # Edit ~/.ssh/config with your Vast.ai instance details:
   # - HostName: Instance IP address
   # - Port: SSH port from Vast.ai dashboard
   # - User: Usually "root"
   ```

3. **Clone this repository** (or use it as a template)

#### On Vast.ai Instance (First Connection)

1. **Connect via SSH**:
   ```bash
   ssh vastai-instance-1  # or your configured host name
   ```

2. **Run setup script**:
   ```bash
   # Clone your repository to /workspace
   cd /workspace
   git clone <your-repo-url> .
   
   # Install dependencies
   bash setup/install_deps.sh
   
   # Install project-specific dependencies
   pip install -r requirements.txt
   ```

### 2. Connect via VS Code Remote-SSH

1. **Open VS Code/Cursor**
2. **Press `F1`** → Type "Remote-SSH: Connect to Host"
3. **Select your Vast.ai instance** (e.g., `vastai-instance-1`)
4. **Open the workspace folder** (`/workspace` or your project directory)

### 3. Run Your First Experiment

1. **Open** `experiments/template_experiment.py`
2. **Run cells interactively**:
   - Click "Run Cell" above each `# %%` marker
   - Or use `Shift+Enter` keyboard shortcut
3. **Start with DEV_MODE**:
   ```python
   # In the setup cell, ensure:
   DEV_MODE = True
   ```

### 4. Run the A100 Residual Sweep (optional)

Need to sanity-check every supported model pair on a single A100 40 GB instance? Use the new sweep helper— it reads the inventory in `configs/dev_config.yaml` and reuses the prompts you just configured:

```bash
python scripts/run_residual_sweep.py --config configs/dev_config.yaml --skip-existing
```

Each run streams logs to `/workspace/persistent/analyses/residual_compare/logs` (with a repo-local fallback if `/workspace` isn’t writable) and writes JSON outputs as `residual_compare_<timestamp>_<model>.json`, where `<model>` is the sanitized SFT model identifier (spaces/dashes/slashes → underscores). Add `--names llama-3.1-8b qwen-7b` to target specific entries, `--dry-run` to preview commands, or `--force` if you need to bypass the A100 40 GB guardrail for debugging.

## Manage Vast.ai Instances

Use the helper script to identify which of your Vast.ai instances are currently running (requires `VASTAI_API_KEY` in either your shell environment or `.env` file):

```bash
python scripts/launch_vast.py status          # show running instances
python scripts/launch_vast.py status --all    # include stopped instances
```

Each entry displays the GPU type, hourly rate, and ready-to-copy SSH command.

## Development Workflow

### Interactive Cell-by-Cell Execution

This is the recommended workflow for development:

1. **Open an experiment file** (`.py` file with `# %%` markers)
2. **Run cells individually** to test and debug
3. **Modify code** and re-run cells as needed
4. **All execution happens on the remote GPU instance**

### Full Script Execution

For long-running training jobs:

```bash
# On the remote instance
python experiments/my_experiment.py

# Or with environment variables
DEV_MODE=False python experiments/my_experiment.py
```

### Handling Instance Preemption

Vast.ai instances can be preempted. The `InterruptibleTrainer` handles this automatically:

1. **Automatic checkpointing** every N steps (configurable)
2. **Signal handling** for graceful shutdown on SIGTERM
3. **Automatic resumption** from latest checkpoint on restart

To resume training:
- Simply run the experiment again
- It will automatically detect and load the latest checkpoint

## Configuration

### Development vs Production

- **`configs/dev_config.yaml`**: Small-scale testing
  - Small batch sizes
  - Fewer epochs
  - Limited dataset
  - Lower checkpoint frequency
  
- **`configs/prod_config.yaml`**: Full training
  - Production batch sizes
  - Full epochs
  - Complete dataset
  - Optimized checkpoint frequency

Switch between them by changing the config path in your experiment file.

### Environment Variables

- `DEV_MODE=True`: Enable development mode (small datasets, fewer epochs)
- `CUDA_VISIBLE_DEVICES=0`: Specify which GPU to use

## Cost Optimization

### Instance Selection Strategy

1. **Development/Testing**: Use cheap instances ($0.20-0.50/hr)
   - RTX 3090, RTX 3080, or similar
   - Sufficient for code development and small-scale testing

2. **Production Training**: Use powerful instances ($1-3/hr)
   - A100, RTX 4090, or similar
   - Only when you need maximum performance

### Cost Tracking

Use the GPU monitor utilities to track costs:

```python
from src.utils.gpu_monitor import get_instance_cost, print_gpu_status

# Print current GPU status
print_gpu_status()

# Calculate costs
cost_info = get_instance_cost(hourly_rate=0.50, runtime_hours=10)
print(f"Estimated cost: ${cost_info['total_cost_usd']:.2f}")
```

### Best Practices

1. **Use DEV_MODE** for all development work
2. **Test locally** (CPU) when possible
3. **Use preemptible instances** for cost savings (with checkpointing)
4. **Monitor GPU utilization** to ensure you're getting value
5. **Scale down** when not actively training

## Checkpoint Management

### Local Checkpoints

Checkpoints are saved to `/workspace/persistent/checkpoints/` by default.

### Remote Storage (S3)

Configure S3 in your config file:

```yaml
checkpoint:
  s3_bucket: "your-bucket-name"
  s3_prefix: "checkpoints/prod"
```

The `CheckpointManager` will automatically:
- Upload checkpoints to S3
- Download checkpoints when resuming
- Keep only the last N checkpoints locally

### Manual Checkpoint Operations

```python
from src.utils.checkpoint_manager import CheckpointManager

# Initialize manager
checkpoint_manager = CheckpointManager(
    local_dir="/workspace/persistent/checkpoints",
    remote_bucket="your-bucket",
    remote_prefix="checkpoints"
)

# Upload latest checkpoint
checkpoint_manager.upload_latest()

# Download latest checkpoint
checkpoint_manager.download_latest()

# Load checkpoint
checkpoint = checkpoint_manager.load_latest()
```

## Syncing Outputs

### From Instance to Local

Use the sync script:

```bash
# Sync all outputs
./scripts/sync_outputs.sh vastai-instance-1 ./outputs /workspace/persistent

# Or manually with rsync
rsync -avz vastai-instance-1:/workspace/persistent/checkpoints/ ./checkpoints/
```

### To S3

```bash
# Set S3 bucket
export S3_BUCKET="your-bucket-name"

# Run sync (will also upload to S3 if configured)
./scripts/sync_outputs.sh
```

## Experiment Tracking

### Weights & Biases

Enable in your config:

```yaml
logging:
  use_wandb: true
  wandb_project: "vastai-experiments"
```

Login on the remote instance:

```bash
wandb login
```

### TensorBoard

TensorBoard is automatically configured. Access it via port forwarding:

1. **SSH config** includes: `LocalForward 8080 localhost:6006`
2. **Start TensorBoard** on the instance:
   ```bash
   tensorboard --logdir /workspace/logs/tensorboard --port 6006
   ```
3. **Access locally**: http://localhost:8080

## Troubleshooting

### Connection Issues

- **Check SSH config**: Ensure hostname, port, and key are correct
- **Test connection**: `ssh vastai-instance-1`
- **Check port forwarding**: Verify ports aren't already in use

### GPU Not Available

- **Check CUDA**: `python -c "import torch; print(torch.cuda.is_available())"`
- **Check GPU**: `nvidia-smi`
- **Verify driver**: Ensure CUDA drivers are installed

### Checkpoint Issues

- **Check disk space**: `df -h /workspace`
- **Verify permissions**: `ls -la /workspace/persistent`
- **Test S3 access**: `aws s3 ls s3://your-bucket/`

### Import Errors

- **Install dependencies**: `pip install -r requirements.txt`
- **Check Python path**: Ensure `src/` is in `PYTHONPATH`
- **Verify installation**: `python -c "import torch; import wandb"`

## Advanced Usage

### Custom Model Architectures

Add your models to `src/models/`:

```python
# src/models/my_model.py
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Your model definition
    
    def forward(self, x):
        # Your forward pass
        return x
```

### Custom Data Loaders

Implement in your experiment file or create reusable loaders in `src/data/`.

### Multi-GPU Training

The trainer can be extended for multi-GPU. Use `torch.nn.DataParallel` or `torch.nn.parallel.DistributedDataParallel`.

## Contributing

This is a template repository. Feel free to:
- Add your own models and utilities
- Extend the trainer for your specific needs
- Customize configurations for your use case

## License

MIT License - feel free to use this template for your projects.

## Resources

- [Vast.ai Documentation](https://vast.ai/docs/)
- [VS Code Remote-SSH](https://code.visualstudio.com/docs/remote/ssh)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Weights & Biases](https://wandb.ai/)

## Support

For issues specific to:
- **Vast.ai**: Check [Vast.ai support](https://vast.ai/contact)
- **VS Code Remote-SSH**: [VS Code Remote Development](https://code.visualstudio.com/docs/remote/remote-overview)
- **This template**: Open an issue in the repository

---

**Happy Training! 🚀**

