# Experiments Directory

This directory contains experiment files using Python cell-based development (similar to Jupyter notebooks but with `.py` files).

## Running Experiments

### Interactive Cell-by-Cell Execution (VS Code/Cursor)

1. **Open an experiment file** (e.g., `template_experiment.py`)
2. **Run individual cells**:
   - Click the "Run Cell" button above each `# %%` cell marker
   - Or use keyboard shortcut: `Shift+Enter` (default)
   - Or right-click and select "Run Cell"

3. **Run all cells**:
   - Use command palette: "Python: Run All Cells"
   - Or use the "Run All" button at the top of the file

### Full Script Execution

Run the entire experiment as a script:

```bash
python experiments/template_experiment.py
```

Or with environment variables:

```bash
DEV_MODE=True python experiments/template_experiment.py
```

## Local vs Remote Execution

### Local Development (CPU/No GPU)

- Good for: Data loading, preprocessing, visualization
- Set `DEV_MODE=True` to use small data subsets
- Code will automatically fall back to CPU if no GPU available

### Remote Execution (Vast.ai GPU Instance)

1. **Connect via VS Code Remote-SSH**:
   - Press `F1` → "Remote-SSH: Connect to Host"
   - Select your Vast.ai instance
   - Open the workspace folder

2. **Run cells interactively** on the remote instance
   - All execution happens on the remote GPU
   - Results are displayed in VS Code

3. **For long-running training**:
   - Use `tmux` or `screen` to keep sessions alive
   - Or run as a script: `nohup python experiment.py > output.log 2>&1 &`

## Creating New Experiments

1. **Copy the template**:
   ```bash
   cp experiments/template_experiment.py experiments/my_experiment.py
   ```

2. **Update the header**:
   - Change experiment name
   - Update instance ID and expected cost

3. **Implement your code**:
   - Fill in data loading cell
   - Define your model
   - Configure training parameters
   - Add visualization code

4. **Run and iterate**:
   - Start with `DEV_MODE=True` for quick testing
   - Switch to production config for full training

## Best Practices

1. **Always use checkpoints**: The `InterruptibleTrainer` handles this automatically
2. **Monitor GPU usage**: Use the GPU monitor utilities in setup cells
3. **Save frequently**: Checkpoints are saved automatically, but save your code often
4. **Use version control**: Commit experiment files to track what you tried
5. **Document results**: Add markdown cells to document findings

## Handling Instance Preemption

If your Vast.ai instance is preempted:

1. **Checkpoints are saved automatically** before termination
2. **Resume training**:
   - Connect to a new instance
   - Run the experiment again
   - It will automatically resume from the latest checkpoint

3. **Check checkpoint location**:
   - Local: `/workspace/persistent/checkpoints/`
   - Remote: S3 bucket (if configured)

## Cost Optimization Tips

1. **Use DEV_MODE** for development and debugging
2. **Start with cheap instances** ($0.20-0.50/hr) for testing
3. **Scale up** to powerful GPUs only for final training
4. **Monitor costs** using the GPU monitor utilities
5. **Use preemptible instances** for cost savings (with checkpointing)

