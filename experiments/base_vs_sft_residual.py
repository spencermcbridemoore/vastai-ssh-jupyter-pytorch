# %% [markdown]
# # Experiment: Base vs SFT Residual Comparison
#
# Compare internal residual streams between a base model and an SFT variant,
# capturing per-layer stats about where their representations diverge.

# %% Setup
import json
import os
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Dict, List

import torch

# Ensure src/ is on the path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from IPython.display import display
except ImportError:
    def display(obj):  # type: ignore
        print(obj)

from src.analysis.residual_compare import ModelSwapSpec, ResidualComparisonRunner
from src.utils.prompt_variants import generate_variants

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"CUDA Version: {torch.version.cuda}")

# %% Configuration
import yaml

WORKSPACE = Path("/workspace")
PERSISTENT = WORKSPACE / "persistent"
PERSISTENT.mkdir(parents=True, exist_ok=True)

DEV_MODE = os.getenv("DEV_MODE", "True").lower() == "true"

config_path = Path(__file__).parent.parent / "configs" / ("dev_config.yaml" if DEV_MODE else "prod_config.yaml")
with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

compare_cfg = config.get("residual_compare", {})
if not compare_cfg:
    raise ValueError("Config missing 'residual_compare' section. Please update configs to run this experiment.")

ANALYSIS_DIR = PERSISTENT / "analyses" / "residual_compare"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

print(f"DEV_MODE: {DEV_MODE}")
print(f"Analysis outputs will be written to: {ANALYSIS_DIR}")

# %% Prompt Loading
def load_prompts(section: Dict[str, object]) -> List[str]:
    prompts: List[str] = []
    prompt_file = section.get("prompt_file")
    if prompt_file:
        prompt_path = Path(prompt_file)
        if not prompt_path.is_absolute():
            prompt_path = Path(__file__).parent.parent / prompt_path
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        with open(prompt_path, "r", encoding="utf-8") as pf:
            for line in pf:
                line = line.strip()
                if line:
                    prompts.append(line)
    prompts.extend(section.get("prompts", []))
    if not prompts:
        raise ValueError("No prompts provided. Add prompts inline or via 'prompt_file'.")
    return prompts


prompts = load_prompts(compare_cfg)
variant_names = compare_cfg.get("prompt_variants", ["identity"])
variant_options = compare_cfg.get("prompt_variant_options", {})

print(f"Loaded {len(prompts)} prompts with variants: {variant_names}")

# %% Runner Initialization
runner = ResidualComparisonRunner(
    base_model_name_or_path=compare_cfg["base_model"],
    sft_model_name_or_path=compare_cfg["sft_model"],
    tokenizer_name_or_path=compare_cfg.get("tokenizer"),
    device=compare_cfg.get("device"),
    dtype=compare_cfg.get("dtype", "auto"),
    top_k=compare_cfg.get("top_k", 20),
    tracked_token_ids=compare_cfg.get("tracked_token_ids"),
    tracked_token_strings=compare_cfg.get("tracked_token_strings"),
    interesting_token_map=compare_cfg.get("interesting_token_map"),
    tokenizer_kwargs=compare_cfg.get("tokenizer_kwargs"),
    model_kwargs=compare_cfg.get("model_kwargs"),
)


def _spec_from_dict(payload: Dict[str, str], default_owner: str) -> ModelSwapSpec:
    return ModelSwapSpec(
        embedding_source=payload.get("embedding_source", default_owner),
        unembedding_source=payload.get("unembedding_source", default_owner),
    )


embedding_variants = compare_cfg.get(
    "embedding_variants",
    [
        {
            "name": "default",
            "base": {"embedding_source": "base", "unembedding_source": "base"},
            "sft": {"embedding_source": "sft", "unembedding_source": "sft"},
        }
    ],
)

# %% Comparison Loop
results = []
for prompt_idx, raw_prompt in enumerate(prompts):
    variant_texts = generate_variants(
        raw_prompt,
        variant_names,
        variant_options=variant_options,
    )
    for variant_name, variant_prompt in variant_texts.items():
        for variant in embedding_variants:
            base_spec = _spec_from_dict(variant.get("base", {}), "base")
            sft_spec = _spec_from_dict(variant.get("sft", {}), "sft")
            metadata = {
                "prompt_idx": prompt_idx,
                "variant_name": variant_name,
                "embedding_variant": variant["name"],
            }
            payload = runner.compare_prompt(
                variant_prompt,
                swap_options={
                    "base": base_spec,
                    "sft": sft_spec,
                },
                metadata=metadata,
            )
            results.append(payload)
            print(
                f"Completed prompt {prompt_idx} variant '{variant_name}' embedding variant '{variant['name']}' "
                f"(len={len(payload['tokens'])})"
            )

print(f"Collected {len(results)} comparison results.")

# %% Persist Results
timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
output_path = ANALYSIS_DIR / f"residual_compare_{timestamp}.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print(f"Wrote comparison JSON to: {output_path}")

# %% Visualization Helpers
try:
    import pandas as pd
except ImportError:
    pd = None
    print("pandas not installed; skipping DataFrame summaries.")


def summarize_cosine(result: Dict[str, object]) -> Dict[str, float]:
    layer_means = {}
    for layer_idx, positions in result["layers"].items():
        cosines = [entry["cosine_sim"] for entry in positions.values()]
        layer_means[int(layer_idx)] = mean(cosines)
    return layer_means


if pd is not None:
    cosine_rows = []
    for item in results:
        row = {
            "prompt": item["metadata"].get("prompt_idx", -1),
            "variant": item["metadata"].get("variant_name", "identity"),
            "embedding_variant": item["metadata"].get("embedding_variant", "default"),
        }
        row.update(summarize_cosine(item))
        cosine_rows.append(row)
    cosine_df = pd.DataFrame(cosine_rows)
    display(cosine_df.head())

# %% Manual Inspection
if results:
    sample = results[0]
    print(f"Sample prompt: {sample['prompt'][:200]}...")
    print("First layer stats for position 0:")
    print(sample["layers"]["0"]["0"])

# %% Cleanup
torch.cuda.empty_cache()
print("Experiment complete!")


