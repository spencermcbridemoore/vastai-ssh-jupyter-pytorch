# Quick Start Guide

Get up and running with remote PyTorch development on Vast.ai in 5 minutes.

## Step 1: Get a Vast.ai Instance

1. Go to [vast.ai](https://vast.ai)
2. Search for a GPU instance (e.g., RTX 3090, ~$0.50/hr)
3. Note the SSH connection details:
   - IP address
   - SSH port
   - Username (usually "root")

## Step 2: Configure SSH (Local Machine)

1. **Edit your SSH config** (`~/.ssh/config` on Mac/Linux, `C:\Users\YourName\.ssh\config` on Windows):

```bash
Host vastai-instance-1
    HostName [YOUR_IP_ADDRESS]
    Port [YOUR_SSH_PORT]
    User root
    IdentityFile ~/.ssh/id_rsa
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
    ControlMaster auto
    ControlPath ~/.ssh/control-%h-%p-%r
    ControlPersist 10m
    LocalForward 8080 localhost:6006
    LocalForward 8888 localhost:8888
    ServerAliveInterval 60
    ServerAliveCountMax 3
```

2. **Test connection**:
```bash
ssh vastai-instance-1
```

## Step 3: Setup Remote Instance

Once connected via SSH:

```bash
# Create workspace
mkdir -p /workspace
cd /workspace

# Clone your repository (or upload files)
git clone <your-repo-url> .

# Install dependencies
bash setup/install_deps.sh
pip install -r requirements.txt
```

## Step 4: Connect via VS Code

1. **Open VS Code/Cursor**
2. **Install Remote-SSH extension** (if not already installed)
3. **Press `F1`** → Type "Remote-SSH: Connect to Host"
4. **Select** `vastai-instance-1`
5. **Open folder**: `/workspace`

## Step 5: Run Your First Experiment

1. **Open** `experiments/template_experiment.py`
2. **Run the setup cell** (first `# %%` block):
   - Click "Run Cell" above the cell
   - Or press `Shift+Enter`
3. **Verify GPU**:
   - Should see "CUDA available: True"
   - GPU name and memory info

4. **Run remaining cells** one by one to test

## Common Commands

### Check GPU Status
```python
from src.utils.gpu_monitor import print_gpu_status
print_gpu_status()
```

### Estimate Batch Size
```python
from src.utils.gpu_monitor import estimate_batch_size
batch_size = estimate_batch_size(model_params=100e6, sequence_length=512)
print(f"Recommended batch size: {batch_size}")
```

### Run Full Training
```bash
# On remote instance
python experiments/template_experiment.py
```

### Sync Outputs to Local
```bash
# From local machine
./scripts/sync_outputs.sh vastai-instance-1 ./outputs /workspace/persistent
```

## Troubleshooting

### Can't connect via SSH
- Check IP and port in Vast.ai dashboard
- Verify SSH key is set up
- Try: `ssh -v vastai-instance-1` for debug info

### GPU not available
```bash
# On remote instance
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

### Import errors
```bash
# On remote instance
pip install -r requirements.txt
```

### Port forwarding not working
- Check if ports 8080/8888 are already in use locally
- Change ports in SSH config if needed

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check [experiments/README.md](experiments/README.md) for experiment workflow
- Customize `configs/dev_config.yaml` for your needs
- Enable richer residual captures by setting `residual_compare.multi_pass.enabled: true`
  (default prompt fuzzing is now off, so add back variants only when needed) and look for the
  resulting artifacts under `h200_outputs_multi/` (`manifest_multi.csv`, per-pair JSONs, and
  `*.meta.json` summaries)
- Create your own experiment from `template_experiment.py`

## Cost Tips

- **Development**: Use cheap instances ($0.20-0.50/hr)
- **Production**: Scale up only when needed
- **Monitor costs**: Use GPU monitor utilities
- **Use preemptible**: Save money with checkpointing

Happy coding! 🚀

