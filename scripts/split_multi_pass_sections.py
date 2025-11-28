"""Split multi-pass JSONs into activation-only and logit-only variants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

# Keys that should be preserved in each derived JSON
ACTIVATION_TOKEN_KEYS = ("token", "entropy", "norm", "norm_delta", "cosine_prev")
LOGIT_TOKEN_KEYS = ("token", "top_k_logits", "tracked_token_logits")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split a multi_pass_*.json file into activation and logit JSONs."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to the source multi_pass_*.json file.",
    )
    parser.add_argument(
        "--activations-output",
        type=Path,
        help="Optional path for the activations JSON; defaults to <input>_activations.json",
    )
    parser.add_argument(
        "--logits-output",
        type=Path,
        help="Optional path for the logits JSON; defaults to <input>_logits.json",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="Indent level for the emitted JSON files (default: 2).",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any, indent: int) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=indent)
        handle.write("\n")


def derive_default_path(source: Path, suffix: str) -> Path:
    return source.with_name(f"{source.stem}_{suffix}.json")


def filter_token_payload(
    token_payload: Mapping[str, Any],
    allowed_keys: Iterable[str],
) -> Dict[str, Any]:
    return {key: token_payload[key] for key in allowed_keys if key in token_payload}


def filter_layers(
    layers: Mapping[str, Mapping[str, Mapping[str, Any]]],
    allowed_keys: Iterable[str],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    filtered_layers: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for layer_idx, token_dict in layers.items():
        filtered_tokens: Dict[str, Dict[str, Any]] = {}
        for token_idx, token_payload in token_dict.items():
            filtered = filter_token_payload(token_payload, allowed_keys)
            if filtered:
                filtered_tokens[token_idx] = filtered
        if filtered_tokens:
            filtered_layers[layer_idx] = filtered_tokens
    return filtered_layers


def filter_runs(
    runs: Iterable[Mapping[str, Any]], allowed_keys: Iterable[str]
) -> List[Dict[str, Any]]:
    filtered_runs: List[Dict[str, Any]] = []
    for run in runs:
        filtered_run = {key: value for key, value in run.items() if key != "layers"}
        layers = run.get("layers", {})
        if isinstance(layers, Mapping):
            filtered_run["layers"] = filter_layers(layers, allowed_keys)
        else:
            filtered_run["layers"] = {}
        filtered_runs.append(filtered_run)
    return filtered_runs


def build_subset(entries: Iterable[Mapping[str, Any]], allowed_keys: Iterable[str]) -> List[Dict[str, Any]]:
    subset: List[Dict[str, Any]] = []
    for entry in entries:
        filtered_entry = {key: value for key, value in entry.items() if key != "runs"}
        filtered_entry["runs"] = filter_runs(entry.get("runs", []), allowed_keys)
        subset.append(filtered_entry)
    return subset


def main() -> None:
    args = parse_args()
    input_path: Path = args.input_path
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    activations_path = args.activations_output or derive_default_path(
        input_path, "activations"
    )
    logits_path = args.logits_output or derive_default_path(input_path, "logits")

    records = load_json(input_path)
    if isinstance(records, Mapping):
        records_iterable = [records]
    else:
        records_iterable = records

    activations_subset = build_subset(records_iterable, ACTIVATION_TOKEN_KEYS)
    logits_subset = build_subset(records_iterable, LOGIT_TOKEN_KEYS)

    write_json(activations_path, activations_subset, indent=args.indent)
    write_json(logits_path, logits_subset, indent=args.indent)

    prompt_count = len(records_iterable)
    print(
        f"Processed {prompt_count} prompt entries from {input_path}. "
        f"Wrote activations to {activations_path} and logits to {logits_path}."
    )


if __name__ == "__main__":
    main()

