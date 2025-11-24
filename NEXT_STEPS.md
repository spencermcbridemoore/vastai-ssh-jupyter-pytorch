# Next Steps: Getting Started with Your Vast.ai Setup

Now that your repository is set up, here's your action plan to start using it:

## Step 1: Get a Vast.ai Instance (5 minutes)

1. **Go to [vast.ai](https://vast.ai)**
2. **Create an account** (if you don't have one)
3. **Search for a GPU instance**:
   - Filter: RTX 3090 or RTX 3080
   - Price: $0.20 - $0.50/hr (for development)
   - Click "Rent" on a suitable instance
4. **Note the connection details**:
   - IP Address
   - SSH Port (usually 22 or a custom port)
   - Username (usually "root")
   - SSH Key (you may need to add your public key)

## Step 2: Configure SSH on Your Local Machine (2 minutes)

1. **Open your SSH config file**:
   - Windows: `C:\Users\YourName\.ssh\config`
   - Mac/Linux: `~/.ssh/config`
   - Create the file if it doesn't exist

2. **Add your Vast.ai instance** (use the template from `setup/ssh_config.template`):
   ```bash
   Host vastai-instance-1
       HostName [YOUR_IP_FROM_VAST_AI]
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

3. **Test the connection**:
   ```bash
   ssh vastai-instance-1
   ```
   You should be able to connect without a password (if SSH key is set up correctly)

## Step 3: Set Up the Remote Instance (5 minutes)

Once connected via SSH:

```bash
# Create workspace directory
mkdir -p /workspace
cd /workspace

# Clone your repository
git clone https://github.com/spencermcbridemoore/vastai-ssh-jupyter-pytorch.git .
# Or if you have a different repo URL, use that

# Install system dependencies
bash setup/install_deps.sh

# Install Python dependencies
pip install -r requirements.txt

# Verify GPU is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
nvidia-smi
```

## Step 4: Connect via VS Code Remote-SSH (2 minutes)

1. **Open VS Code/Cursor**
2. **Install Remote-SSH extension** (if not already installed):
   - Press `Ctrl+Shift+X` (or `Cmd+Shift+X` on Mac)
   - Search for "Remote - SSH"
   - Install it

3. **Connect to your instance**:
   - Press `F1` (or `Cmd+Shift+P` on Mac)
   - Type "Remote-SSH: Connect to Host"
   - Select `vastai-instance-1`
   - VS Code will connect and open a new window

4. **Open the workspace**:
   - Click "Open Folder"
   - Navigate to `/workspace` (or wherever you cloned the repo)
   - Click "OK"

## Step 5: Run Your First Experiment (5 minutes)

1. **Open the template experiment**:
   - Navigate to `experiments/template_experiment.py`
   - You should see cells marked with `# %%`

2. **Run the setup cell**:
   - Click "Run Cell" above the first `# %%` cell
   - Or press `Shift+Enter` while the cursor is in the cell
   - Verify you see "CUDA available: True"

3. **Explore the template**:
   - Run cells one by one to understand the structure
   - Modify the template to fit your needs
   - Check out the GPU monitor utilities

## Step 6: Customize for Your Project

1. **Update configurations**:
   - Edit `configs/dev_config.yaml` for your model/data
   - Adjust batch sizes, learning rates, etc.

2. **Create your own experiment**:
   ```bash
   cp experiments/template_experiment.py experiments/my_first_experiment.py
   ```
   - Fill in your data loading code
   - Define your model architecture
   - Configure training parameters

3. **Set up experiment tracking** (optional):
   ```bash
   # On the remote instance
   wandb login
   ```
   - Get your API key from [wandb.ai](https://wandb.ai)
   - Enable wandb in your config file

## Step 7: Start Training!

1. **Test with DEV_MODE**:
   ```python
   # In your experiment file
   DEV_MODE = True
   ```
   - Run a quick test to make sure everything works
   - Verify checkpoints are being saved

2. **Run full training**:
   ```python
   DEV_MODE = False
   ```
   - Or use the production config
   - Monitor GPU usage and costs

## Quick Reference Commands

### On Local Machine:
```bash
# Connect to instance
ssh vastai-instance-1

# Sync outputs from instance
./scripts/sync_outputs.sh vastai-instance-1 ./outputs /workspace/persistent
```

### On Remote Instance:
```bash
# Check GPU status
nvidia-smi

# Monitor GPU usage
watch -n 1 nvidia-smi

# Check disk space
df -h

# View logs
tail -f /workspace/logs/training.log
```

### In VS Code:
- `Shift+Enter`: Run current cell
- `F1` → "Python: Run All Cells": Run entire file
- `Ctrl+` ` (backtick)`: Open terminal

## Troubleshooting

### Can't connect via SSH?
- Verify IP and port in Vast.ai dashboard
- Check if your SSH key is added to the instance
- Try: `ssh -v vastai-instance-1` for debug info

### GPU not showing up?
- Run `nvidia-smi` to check if drivers are installed
- Verify CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
- Check if instance actually has a GPU (some instances don't)

### Import errors?
- Make sure you ran `pip install -r requirements.txt`
- Check Python path: `python -c "import sys; print(sys.path)"`
- Verify you're in the right directory

### Cells not running?
- Make sure Python extension is installed in VS Code
- Check that you're connected via Remote-SSH
- Try restarting VS Code

## Cost Management Tips

1. **Start cheap**: Use $0.20-0.50/hr instances for development
2. **Monitor usage**: Check Vast.ai dashboard regularly
3. **Use preemptible**: Save money, but ensure checkpointing works
4. **Stop when done**: Don't leave instances running unnecessarily
5. **Scale up wisely**: Only use expensive instances for final training

## What's Next After Setup?

- **Implement your model**: Add your architecture to `src/models/`
- **Set up data pipeline**: Create data loaders for your dataset
- **Configure S3**: Set up remote checkpoint storage (optional)
- **Set up monitoring**: Configure Weights & Biases or TensorBoard
- **Optimize costs**: Use GPU monitor to find best instance types

## Need Help?

- Check the main [README.md](README.md) for detailed documentation
- Review [experiments/README.md](experiments/README.md) for experiment workflow
- See [QUICKSTART.md](QUICKSTART.md) for a condensed guide

---

**You're all set! Start with Step 1 and work through each step. Good luck with your training! 🚀**

