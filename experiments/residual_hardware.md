# Residual Comparison Hardware Guide

The `experiments/base_vs_sft_residual.py` script keeps both the base and SFT models in GPU memory (bf16, batch size 1) while capturing per-layer residual streams. The table below summarizes practical Vast.ai hardware choices that balance cost and headroom for a single prompt evaluation at different token counts. GPU counts assume tensor-parallel inference when VRAM requirements exceed one device.

| Base ↔ SFT pair | 10 tokens | 50 tokens | 100 tokens | 250 tokens |
| --- | --- | --- | --- | --- |
| Qwen/Qwen2.5-0.5B ↔ Qwen/Qwen2.5-0.5B-Instruct | 1× NVIDIA T4 (16 GB) | 1× T4 (16 GB) | 1× T4 (16 GB) | 1× T4 (16 GB) |
| Qwen/Qwen2.5-1.5B ↔ …-Instruct | 1× RTX 3060 (12 GB) | 1× RTX 3060 (12 GB) | 1× RTX A4000/T4 (16 GB) | 1× RTX 4090 (24 GB) |
| Qwen/Qwen2.5-3B ↔ …-Instruct | 1× RTX 3090/4090 (24 GB) | 1× RTX 3090/4090 (24 GB) | 1× RTX 3090/4090 (24 GB) | 1× RTX 6000 Ada (48 GB) |
| Qwen/Qwen2.5-7B ↔ …-Instruct | 1× A100 40 GB | 1× A100 40 GB | 1× A100 40 GB | 1× RTX 6000 Ada or A100 80 GB |
| Qwen/Qwen2.5-14B ↔ …-Instruct | 1× A100 80 GB | 1× A100 80 GB | 1× A100 80 GB | 1× A100 80 GB |
| Qwen/Qwen2.5-32B ↔ …-Instruct | 2× A100 80 GB (TP=2) | 2× A100 80 GB | 2× A100 80 GB | 2× A100 80 GB |
| Qwen/Qwen2.5-72B ↔ …-Instruct | 4× A100/H100 80 GB | 4× A100/H100 80 GB | 4× A100/H100 80 GB | 6× A100/H100 80 GB |
| Qwen/Qwen2.5-Math-1.5B ↔ …-Instruct | 1× RTX 3060 (12 GB) | 1× RTX 3060 (12 GB) | 1× RTX A4000/T4 (16 GB) | 1× RTX 4090 (24 GB) |
| Qwen/Qwen2.5-Math-7B ↔ …-Instruct | 1× A100 40 GB | 1× A100 40 GB | 1× A100 40 GB | 1× A100 80 GB |
| Qwen/Qwen2.5-Math-72B ↔ …-Instruct | 4× A100/H100 80 GB | 4× A100/H100 80 GB | 4× A100/H100 80 GB | 6× A100/H100 80 GB |
| meta-llama/Llama-3.1-8B ↔ …-Instruct | 1× A100 40 GB | 1× A100 40 GB | 1× RTX 6000 Ada (48 GB) | 1× RTX 6000 Ada (48 GB) |
| meta-llama/Llama-3.1-70B ↔ …-Instruct | 4× H100 80 GB | 4× H100 80 GB | 4× H100 80 GB | 6× H100 80 GB |
| meta-llama/Llama-3.1-405B ↔ …-Instruct | 12× H100 80 GB | 12× H100 80 GB | 12× H100 80 GB | 16× H100 80 GB |
| deepseek-ai/DeepSeek-V3-Base ↔ DeepSeek-V3 | 2× H100 80 GB | 2× H100 80 GB | 3× H100 80 GB | 4× H100 80 GB |
| deepseek-ai/DeepSeek-V3.1-Base ↔ DeepSeek-V3.1 | 3× H100 80 GB | 3× H100 80 GB | 4× H100 80 GB | 5× H100 80 GB |
| nvidia/Mistral-NeMo-12B-Base ↔ …-Instruct | 1× RTX 6000 Ada (48 GB) | 1× RTX 6000 Ada (48 GB) | 1× RTX 6000 Ada (48 GB) | 1× A100 80 GB |
| nvidia/Mistral-NeMo-Minitron-8B-Base ↔ …-Instruct | 1× RTX 4090 (24 GB) | 1× RTX 4090 (24 GB) | 1× RTX 6000 Ada (48 GB) | 1× RTX 6000 Ada (48 GB) |
| nvidia/Nemotron-4-340B-Base ↔ …-Instruct | 6× H100 80 GB | 8× H100 80 GB | 10× H100 80 GB | 12× H100 80 GB |

### Notes

- Prices fluctuate, but as of late 2025, typical Vast.ai spot rates are roughly RTX 4090 24 GB ≈ $0.90/hr, A100 40 GB ≈ $1.60/hr, A100 80 GB ≈ $2.40/hr, H100 80 GB ≈ $4.50/hr. Multiply by GPU count for a ballpark cost.
- Longer prompts increase activation memory by ~5–15 % because the experiment stores hidden states for every layer. Recommendations above already include that overhead; if you collect KV caches for generation or raise batch size, step up one tier.
- To downsize hardware, consider sequentially loading base and SFT models (swapping to CPU between passes), enabling weight-only 4-bit quantization, or disabling residual capture for the earliest layers.
- The JSON output now includes per-model cross-layer diagnostics (`*_cosine_prev`, `*_norm_delta`) in addition to cross-model comparisons, which can help debug depth-specific instabilities during analysis.
- Always verify vRAM consumption with `nvidia-smi` when using custom prompt sets or precision modes.

### Enabling local RTX 4090 runs

You can force `experiments/base_vs_sft_residual.py` to run on your workstation’s RTX 4090 (24 GB) instead of renting a Vast.ai instance. Configure the new `residual_compare.local_run` block inside `configs/dev_config.yaml` or `configs/prod_config.yaml`:

```
residual_compare:
  base_model: "gpt2"
  sft_model: "gpt2"
  ...
  local_run:
    enabled: true
    min_vram_gb: 24
    require_gpu_name_substring: "4090"
    cuda_device_index: 0
    allowed_pairs:
      - base: "gpt2"
        sft: "gpt2"
        device: "cuda"
        dtype: "float16"
        notes: "Lightweight sanity-check pair for desktop runs"
```

How it works:

- The script validates that CUDA is available, the selected GPU name contains `require_gpu_name_substring`, and that total VRAM meets `min_vram_gb`. It errors out immediately if any check fails—there is no silent fallback to remote execution.
- Only model pairs listed under `allowed_pairs` may run locally. Each entry can override `device`, `dtype`, `tokenizer`, `tokenizer_kwargs`, or `model_kwargs`, so you can pin safe precision/settings per pair.
- When a pair is approved, the script logs the GPU name, index, and VRAM before loading checkpoints so you can confirm the right device is in use.

After toggling `enabled: true`, test a lightweight pair first:

```
DEV_MODE=true python experiments/base_vs_sft_residual.py
```

Watch the console for the “Local GPU execution enabled…” line and verify `nvidia-smi` stays below 24 GB before attempting heavier models.


