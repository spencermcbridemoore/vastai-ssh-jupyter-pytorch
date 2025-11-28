# Multi-Pass Residual Capture Summary

**Date**: 2025-11-27  
**Hardware**: NVIDIA GeForce RTX 4090 (24 GB)  
**Environment**: Conda `vastai` (Python 3.12, PyTorch 2.5.1+cu124)

## Overview

Executed four-pass residual captures (BB, BS, SB, SS) plus six pairwise differences on local RTX 4090. Each pass uses different embedding/LLM combinations while keeping unembedding tied to the active model:

- **BB**: Base embeddings → Base LLM
- **BS**: Base embeddings → SFT LLM  
- **SB**: SFT embeddings → Base LLM
- **SS**: SFT embeddings → SFT LLM

Pairwise diffs computed: BB-BS, BB-SB, BB-SS, BS-SB, BS-SS, SB-SS

## Completed Runs

| Model Pair | Prompts | Tokens/Prompt | Peak VRAM | Status | JSON |
|------------|---------|---------------|-----------|--------|------|
| GPT-2 ↔ GPT-2 | 4 | 9–90 | ~4 GB | ✅ Success | [multi_pass_20251127_032925_gpt2_to_gpt2.json](gpt2_to_gpt2/multi_pass_20251127_032925_gpt2_to_gpt2.json) |
| Qwen 0.5B ↔ Instruct | 4 | 7–76 | ~10 GB | ✅ Success | [multi_pass_20251127_032959_Qwen_Qwen2_5_0_5B_to_Qwen_Qwen2_5_0_5B_Instruct.json](Qwen_Qwen2_5_0_5B_to_Qwen_Qwen2_5_0_5B_Instruct/multi_pass_20251127_032959_Qwen_Qwen2_5_0_5B_to_Qwen_Qwen2_5_0_5B_Instruct.json) |
| Qwen 1.5B ↔ Instruct | 4 | 7–76 | ~17 GB | ✅ Success | [multi_pass_20251127_033047_Qwen_Qwen2_5_1_5B_to_Qwen_Qwen2_5_1_5B_Instruct.json](Qwen_Qwen2_5_1_5B_to_Qwen_Qwen2_5_1_5B_Instruct/multi_pass_20251127_033047_Qwen_Qwen2_5_1_5B_to_Qwen_Qwen2_5_1_5B_Instruct.json) |
| Mistral-NeMo-Minitron-8B ↔ Instruct | 0 | N/A | N/A | ⏸️ Skipped | Exceeds 24 GB budget; requires ≥48 GB for 4 prompts |

## Configuration

- **Prompt fuzzing**: Disabled (only `identity` variant)
- **Multi-pass mode**: `residual_compare.multi_pass.enabled: true`
- **Prompt set**: 4 production prompts from `configs/dev_config.yaml`
- **Output location**: `h200_outputs_multi/<pair>/`
- **Manifest**: [manifest_multi.csv](manifest_multi.csv)

## JSON Structure Verification

Each output file contains **4 records** (one per prompt), and each record has:
- `format_version: 2`
- `runs`: Array of 4 pass objects (BB, BS, SB, SS) with per-layer stats
- `pairwise`: Array of 6 diff objects (BB-BS, …) reusing legacy schema
- `metadata.multi_pass: true`

Verified via grep:
- 4 `"runs"` blocks per file ✅
- 4 `"pairwise"` blocks per file ✅
- 16 run name entries (`"name": "BB|BS|SB|SS"`) across 4 prompts ✅

## Conda Environment Setup

```bash
# Create environment
conda create -y -n vastai python=3.12
conda activate vastai

# Install CUDA PyTorch
pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124 \
  --index-url https://download.pytorch.org/whl/cu124

# Install project dependencies
cd /path/to/vastai-ssh-jupyter-pytorch
pip install -r requirements.txt
pip install transformers>=4.46.0

# Verify CUDA
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Expected: 2.5.1+cu124 True
```

## Analysis Tools

### Inspect per-run stats
```bash
conda activate vastai
python scripts/residual_report.py \
  h200_outputs_multi/gpt2_to_gpt2/*.json \
  --record-type runs \
  --metric norm \
  --top-n 5
```

### Inspect pairwise diffs (default)
```bash
python scripts/residual_report.py \
  h200_outputs_multi/Qwen_Qwen2_5_0_5B_to_Qwen_Qwen2_5_0_5B_Instruct/*.json \
  --record-type pairwise \
  --metric norm_diff \
  --top-n 5
```

## Storage Impact

With fuzzing disabled, each prompt produces exactly **1 multi-pass record** (vs. 3+ records in legacy fuzzing mode). The four passes add minimal overhead since activations are already captured; the main increase is metadata (`runs` array) and pairwise diff blocks. Approximate JSON sizes:

- GPT-2: ~1.4 MB per prompt
- Qwen 0.5B: ~12 MB per prompt
- Qwen 1.5B: ~32 MB per prompt

## Next Steps

- **Larger models**: Run Qwen 3B, 7B, Llama 3.1-8B on H200 NVL (141 GB) with the same multi-pass config
- **Extended analysis**: Use `xarray` or `tensorly` to explore dimension-collapse patterns across the 6 pairwise diffs
- **Minotron-8B**: Requires remote execution with ≥48 GB VRAM; add to H200 sweep if needed

