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


