# %% [markdown]
# # Experiment: Base vs SFT Residual Comparison
#
# Compare internal residual streams between a base model and an SFT variant,
# capturing per-layer stats about where their representations diverge.

# %% Setup
import json
import os
import re
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Dict, List

import torch
from transformers import AutoTokenizer

# Ensure repository root (and thereby src/) is on the path
import sys

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

try:
    from IPython.display import display
except ImportError:
    def display(obj):  # type: ignore
        print(obj)

from src.analysis.residual_compare import ModelSwapSpec, MultiPassPlan, ResidualComparisonRunner
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
compare_cfg = dict(compare_cfg)  # copy so we can mutate safely without touching original config dict

ENV_OVERRIDE_MAP = {
    "base_model": "RESIDUAL_BASE_MODEL",
    "sft_model": "RESIDUAL_SFT_MODEL",
    "tokenizer": "RESIDUAL_TOKENIZER",
    "device": "RESIDUAL_DEVICE",
    "dtype": "RESIDUAL_DTYPE",
    "prompt_file": "RESIDUAL_PROMPT_FILE",
    "tokenizer_kwargs": "RESIDUAL_TOKENIZER_KWARGS",
    "model_kwargs": "RESIDUAL_MODEL_KWARGS",
}
JSON_OVERRIDE_KEYS = {"tokenizer_kwargs", "model_kwargs"}

for key, env_var in ENV_OVERRIDE_MAP.items():
    override_value = os.getenv(env_var)
    if not override_value:
        continue
    if key in JSON_OVERRIDE_KEYS:
        try:
            compare_cfg[key] = json.loads(override_value)
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            raise ValueError(f"Failed to parse JSON override for {key}: {override_value}") from exc
    else:
        compare_cfg[key] = override_value

LOCAL_OVERRIDE_KEYS = ("device", "dtype", "tokenizer", "tokenizer_kwargs", "model_kwargs")


def _match_local_pair(local_cfg: Dict[str, object], base_model: str, sft_model: str) -> Dict[str, object]:
    allowed = local_cfg.get("allowed_pairs") or []
    for entry in allowed:
        if entry.get("base") == base_model and entry.get("sft") == sft_model:
            return entry
    raise ValueError(
        f"Local run requested but pair ({base_model}, {sft_model}) is not listed in residual_compare.local_run.allowed_pairs."
    )


def _ensure_local_gpu(local_cfg: Dict[str, object]) -> Dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("Local GPU execution requested, but torch.cuda.is_available() is False.")
    device_index = int(local_cfg.get("cuda_device_index", 0) or 0)
    if device_index >= torch.cuda.device_count():
        raise RuntimeError(
            f"cuda_device_index={device_index} is invalid; this machine exposes {torch.cuda.device_count()} CUDA device(s)."
        )
    torch.cuda.set_device(device_index)
    device_name = torch.cuda.get_device_name(device_index)
    required_name = local_cfg.get("require_gpu_name_substring")
    if required_name and required_name.lower() not in device_name.lower():
        raise RuntimeError(
            f"Local GPU '{device_name}' does not contain required substring '{required_name}'. "
            "Update residual_compare.local_run.require_gpu_name_substring if this is intentional."
        )
    props = torch.cuda.get_device_properties(device_index)
    total_gib = props.total_memory / (1024 ** 3)
    min_vram = float(local_cfg.get("min_vram_gb", 0) or 0)
    if min_vram and total_gib + 1e-6 < min_vram:
        raise RuntimeError(
            f"GPU '{device_name}' exposes {total_gib:.2f} GiB, which is below the configured minimum ({min_vram} GiB)."
        )
    return {"index": device_index, "name": device_name, "total_gib": total_gib}


def _apply_local_overrides(
    base_cfg: Dict[str, object],
    pair_cfg: Dict[str, object],
    device_index: int,
) -> Dict[str, object]:
    updated = dict(base_cfg)
    overrides = {key: pair_cfg[key] for key in LOCAL_OVERRIDE_KEYS if key in pair_cfg}
    if "device" not in overrides:
        overrides["device"] = f"cuda:{device_index}" if device_index else "cuda"
    updated.update(overrides)
    return updated


local_run_cfg = compare_cfg.get("local_run") or {}
if local_run_cfg.get("enabled"):
    pair_cfg = _match_local_pair(local_run_cfg, compare_cfg["base_model"], compare_cfg["sft_model"])
    gpu_info = _ensure_local_gpu(local_run_cfg)
    compare_cfg = _apply_local_overrides(compare_cfg, pair_cfg, gpu_info["index"])
    print(
        "Local GPU execution enabled — "
        f"running pair ({compare_cfg['base_model']}, {compare_cfg['sft_model']}) on "
        f"GPU '{gpu_info['name']}' (index {gpu_info['index']}, {gpu_info['total_gib']:.2f} GiB)."
    )
    if pair_cfg.get("notes"):
        print(f"Pair notes: {pair_cfg['notes']}")

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


def _maybe_truncate_production_prompts(prompts: List[str], section: Dict[str, object]) -> List[str]:
    prompt_file = section.get("prompt_file")
    if not prompt_file:
        return prompts
    prompt_name = Path(prompt_file).name
    if prompt_name != "prod_prompts.txt":
        return prompts

    tokenizer_name = section.get("tokenizer") or section.get("sft_model") or section.get("base_model")
    if not tokenizer_name:
        print(
            "Skipping production prompt truncation: no tokenizer/base model specified to perform tokenization."
        )
        return prompts

    tokenizer_kwargs = section.get("tokenizer_kwargs") or {}
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **tokenizer_kwargs)

    encoded_prompts = []
    lengths = []
    for prompt in prompts:
        encoded = tokenizer(prompt, add_special_tokens=False)
        ids = encoded["input_ids"]
        encoded_prompts.append(ids)
        lengths.append(len(ids))

    if not lengths:
        return prompts

    target_len = min(lengths)
    max_len = max(lengths)
    if target_len == 0 or max_len == target_len:
        return prompts

    truncated_prompts: List[str] = []
    for ids in encoded_prompts:
        truncated_ids = ids[:target_len]
        truncated_text = tokenizer.decode(
            truncated_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        truncated_prompts.append(truncated_text)

    print(
        f"Normalized production prompts to {target_len} tokens "
        f"(previous max length {max_len} tokens)."
    )
    return truncated_prompts


prompts = load_prompts(compare_cfg)
prompts = _maybe_truncate_production_prompts(prompts, compare_cfg)
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
multi_pass_cfg = compare_cfg.get("multi_pass") or {}
multi_pass_plan = None
if multi_pass_cfg.get("enabled"):
    multi_pass_plan = MultiPassPlan.from_payload(multi_pass_cfg)
    if len(embedding_variants) > 1:
        print(
            "multi_pass.enabled=True — ignoring embedding_variants list in favor of configured runs."
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
        base_metadata = {
            "prompt_idx": prompt_idx,
            "variant_name": variant_name,
        }
        if multi_pass_plan:
            metadata = {**base_metadata, "embedding_variant": "multi_pass"}
            payload = runner.compare_prompt_multi(
                variant_prompt,
                pass_specs=multi_pass_plan.runs,
                pair_specs=multi_pass_plan.pairwise,
                metadata=metadata,
            )
            results.append(payload)
            print(
                f"Completed prompt {prompt_idx} variant '{variant_name}' via multi_pass "
                f"(runs={len(payload['runs'])}, pairwise={len(payload['pairwise'])})"
            )
            continue

        for variant in embedding_variants:
            base_spec = _spec_from_dict(variant.get("base", {}), "base")
            sft_spec = _spec_from_dict(variant.get("sft", {}), "sft")
            metadata = {**base_metadata, "embedding_variant": variant["name"]}
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
def _sanitize_for_filename(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9]+", "_", value or "").strip("_")
    return sanitized or "model"

model_label = _sanitize_for_filename(compare_cfg.get("sft_model") or compare_cfg.get("base_model") or "")
output_path = ANALYSIS_DIR / f"residual_compare_{timestamp}_{model_label}.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print(f"Wrote comparison JSON to: {output_path}")

# %% Visualization Helpers
try:
    import pandas as pd
except ImportError:
    pd = None
    print("pandas not installed; skipping DataFrame summaries.")


def summarize_layer_metric(result: Dict[str, object], metric_name: str) -> Dict[int, float]:
    """
    Compute the mean of a metric across positions for each layer.
    Skips entries where the metric is None.
    """
    layer_means: Dict[int, float] = {}
    for layer_idx, positions in result["layers"].items():
        values = [
            entry.get(metric_name)
            for entry in positions.values()
            if entry.get(metric_name) is not None
        ]
        if values:
            layer_means[int(layer_idx)] = mean(values)
    return layer_means


if pd is not None:
    summary_rows = []
    metric_prefixes = {
        "cosine_sim": "cross_cos",
        "base_cosine_prev": "base_prev_cos",
        "sft_cosine_prev": "sft_prev_cos",
        "base_norm_delta": "base_norm_delta",
        "sft_norm_delta": "sft_norm_delta",
    }
    for item in results:
        if "layers" not in item:
            continue
        row = {
            "prompt": item["metadata"].get("prompt_idx", -1),
            "variant": item["metadata"].get("variant_name", "identity"),
            "embedding_variant": item["metadata"].get("embedding_variant", "default"),
        }
        for metric, prefix in metric_prefixes.items():
            summary = summarize_layer_metric(item, metric)
            for layer_idx, value in summary.items():
                row[f"{prefix}_L{layer_idx}"] = value
        summary_rows.append(row)
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        display(summary_df.head())
    elif results:
        print("Multi-pass results detected; layer summary preview skipped.")

# %% Manual Inspection
if results:
    sample = results[0]
    print(f"Sample prompt: {sample['prompt'][:200]}...")
    if "layers" in sample:
        print("First layer stats for position 0:")
        print(sample["layers"]["0"]["0"])
    elif sample.get("pairwise"):
        first_pair = sample["pairwise"][0]
        print(f"Previewing pairwise diff '{first_pair['name']}' layer 0 pos 0:")
        print(first_pair["layers"]["0"]["0"])

# %% Cleanup
torch.cuda.empty_cache()
print("Experiment complete!")


