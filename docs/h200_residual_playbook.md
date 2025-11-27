# H200 Residual Comparison Playbook

This checklist turns the high-level plan into concrete commands so you can drain the
H200 NVL instance, harvest outputs after every model pair, and shut the box down.

## 1. Prep the instance (`prep-instance`)

Run these once per boot:

```bash
ssh <h200-alias>
cd /workspace/vastai-ssh-jupyter-pytorch
git pull
pip install -r requirements.txt
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

If any dependency upgrades pull new model weights, log in to Hugging Face:

```bash
huggingface-cli login
```

## 2. Confirm the target list (`assemble-targets`)

The default sweep in [`configs/dev_config.yaml`](../configs/dev_config.yaml) already
covers every model that comfortably fits inside a 141 GB H200 NVL:

| Entry name | Base ↔ SFT | Notes |
| --- | --- | --- |
| `qwen-0.5b` | Qwen/Qwen2.5-0.5B ↔ …-Instruct | Tiny; good sanity check |
| `qwen-1.5b` | Qwen/Qwen2.5-1.5B ↔ …-Instruct | Still <20 GB |
| `qwen-3b` | Qwen/Qwen2.5-3B ↔ …-Instruct | Fits easily |
| `qwen-7b` | Qwen/Qwen2.5-7B ↔ …-Instruct | Well within H200 headroom |
| `qwen-math-1.5b` | Math variant | Same footprint as generic 1.5B |
| `qwen-math-7b` | Math variant | Similar to 7B generic |
| `llama-3.1-8b` | meta-llama/Llama-3.1-8B ↔ …-Instruct | Needs HF auth |

Feel free to extend the list, but consult [`experiments/residual_hardware.md`](../experiments/residual_hardware.md)
before adding anything ≥14 B parameters.

## 3. Run models + download outputs (`run-models` & `collect-data`)

From your local workstation, use the new helper:

```bash
python scripts/h200_residual_batch.py \
  --ssh-host vast-h200 \
  --ssh-port 22 \
  --remote-dir /workspace/vastai-ssh-jupyter-pytorch \
  --local-output ./h200_outputs \
  --entries qwen-0.5b qwen-1.5b qwen-3b qwen-7b qwen-math-1.5b qwen-math-7b llama-3.1-8b \
  --device cuda:0 \
  --shutdown
```

What it does per entry:

1. Sets `DEV_MODE=False` plus the `RESIDUAL_*` overrides for the pair.
2. Runs `experiments/base_vs_sft_residual.py`, which trims `experiments/prompts/prod_prompts.txt`
   to the shortest tokenized length before generating variants.
3. Captures stdout/stderr into `h200_outputs/<entry>/<timestamp>_<entry>.log`.
4. Parses the emitted `Wrote comparison JSON to:` path, then `scp`s that JSON into the same folder.
5. Appends a row to `h200_outputs/manifest.csv` with the entry, status, and file locations.
6. If the run fails (OOM, download, etc.) it logs the failure and moves to the next entry.

The helper can target the entire sweep by omitting `--entries`, and `--skip-existing`
prevents rerunning pairs whose JSONs are already present locally.

### Optional: Multi-pass residual captures

- **Enable with config only when needed.** Set `residual_compare.multi_pass.enabled: true`
  inside `configs/dev_config.yaml` (or `prod_config.yaml`) to run the four canonical
  embedding/LLM combinations per prompt: BB, BS, SB, SS. The runner now emits
  one prompt record per variant that contains both a `runs` section (single-model
  stats) and a `pairwise` section (the six BB/BS/… diffs). Since prompt fuzzing
  now defaults to `["identity"]`, storage stays roughly flat even though each
  prompt yields more structured data. Re-add `prompt_variants` entries if you
  still want perturbations.
- **Analysis tooling understands the richer file.** Use
  `scripts/residual_report.py ... --record-type runs` to summarize per-run grids,
  or keep the default `pairwise` mode to review BB‑BS style diffs. Existing
  helpers such as `build_residual_grid` continue to work because the `pairwise`
  payload keeps the legacy schema.

## 4. Verify + teardown (`teardown`)

After the script processes every model:

1. Spot-check `h200_outputs/manifest.csv` to confirm each entry shows `status=success`.
2. Inspect a sample JSON/log pair to ensure the prompt lengths are equalized
   (look for `Normalized production prompts …` inside the log).
3. If you did **not** pass `--shutdown`, manually terminate the instance:

```bash
ssh <h200-alias> "sudo shutdown -h now"
# or use vast.ai dashboard → Destroy
```

4. Archive the local `h200_outputs` directory alongside your analysis notebooks.

Following the playbook ensures every prompt run uploads immediately, so the H200 NVL
never sits idle with unretrieved data, and the instance is decommissioned as soon as
the queue is empty.

