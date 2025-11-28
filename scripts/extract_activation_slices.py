#!/usr/bin/env python
"""
Extract unfuzzed activation slices from residual_compare JSON artifacts.

For each model directory under h200_outputs/, the script selects the latest
residual_compare_*.json file, keeps only the identity/default prompt entries,
strips logit-oriented fields, and emits one JSON per prompt (indices 0..3).
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

TIMESTAMP_RE = re.compile(r"residual_compare_(\d{8})_(\d{6})_.*\.json$")
STRIP_KEYS = ("top_k_increased", "top_k_decreased", "tracked_token_logits")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("h200_outputs"),
        help="Directory that contains per-model residual outputs (default: %(default)s).",
    )
    parser.add_argument(
        "--prompt-count",
        type=int,
        default=4,
        help="Number of prompt indices to export (starting from zero).",
    )
    parser.add_argument(
        "--variant-name",
        default="identity",
        help="Prompt variant name to keep (default: %(default)s).",
    )
    parser.add_argument(
        "--embedding-variant",
        default="default",
        help="Embedding variant to keep (default: %(default)s).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the operations that would be performed without writing files.",
    )
    return parser.parse_args()


def discover_latest_files(root: Path) -> List[Tuple[str, Path]]:
    """Return (model_name, latest_json_path) pairs under the provided root."""
    discovered: List[Tuple[str, Path]] = []
    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir():
            continue
        latest_entry: Tuple[str, Path] | None = None
        for json_path in model_dir.glob("residual_compare_*.json"):
            match = TIMESTAMP_RE.match(json_path.name)
            if not match:
                continue
            timestamp_key = match.group(1) + match.group(2)
            if latest_entry is None or timestamp_key > latest_entry[0]:
                latest_entry = (timestamp_key, json_path)
        if latest_entry:
            discovered.append((model_dir.name, latest_entry[1]))
    return discovered


def load_records(path: Path) -> List[MutableMapping[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"{path} does not contain a JSON list.")
    return payload


def filter_unfuzzed_records(
    records: Iterable[Mapping[str, object]],
    *,
    prompt_limit: int,
    variant_name: str,
    embedding_variant: str,
) -> Dict[int, MutableMapping[str, object]]:
    """Return the first matching record per prompt index."""
    targets = set(range(prompt_limit))
    selected: Dict[int, MutableMapping[str, object]] = {}
    for record in records:
        metadata = record.get("metadata") or {}
        if not isinstance(metadata, Mapping):
            continue
        if metadata.get("variant_name") != variant_name:
            continue
        if metadata.get("embedding_variant") != embedding_variant:
            continue
        idx = metadata.get("prompt_idx")
        if idx not in targets or idx in selected:
            continue
        selected[idx] = copy.deepcopy(record)
        if len(selected) == len(targets):
            break
    return selected


def strip_logit_fields(record: MutableMapping[str, object]) -> None:
    """Remove logit-oriented keys from every position in the layer grid."""
    layers = record.get("layers")
    if not isinstance(layers, Mapping):
        return
    for layer_payload in layers.values():
        if not isinstance(layer_payload, Mapping):
            continue
        for position_payload in layer_payload.values():
            if not isinstance(position_payload, MutableMapping):
                continue
            for key in STRIP_KEYS:
                position_payload.pop(key, None)


def write_prompt_record(base_path: Path, prompt_index: int, record: Mapping[str, object], *, dry_run: bool) -> Path:
    output_name = f"{base_path.stem}_prmpt_{prompt_index + 1}.json"
    output_path = base_path.with_name(output_name)
    if dry_run:
        return output_path
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
    return output_path


def main() -> None:
    args = parse_args()
    root = args.root
    if not root.exists():
        raise SystemExit(f"Root directory not found: {root}")
    summary: List[str] = []
    missing: Dict[str, Sequence[int]] = {}
    for model_name, json_path in discover_latest_files(root):
        records = load_records(json_path)
        selected = filter_unfuzzed_records(
            records,
            prompt_limit=args.prompt_count,
            variant_name=args.variant_name,
            embedding_variant=args.embedding_variant,
        )
        for idx, record in selected.items():
            strip_logit_fields(record)
            write_prompt_record(json_path, idx, record, dry_run=args.dry_run)
        missing_indices = sorted(set(range(args.prompt_count)) - set(selected.keys()))
        if missing_indices:
            missing[model_name] = tuple(idx + 1 for idx in missing_indices)
        summary.append(f"{model_name}: {json_path.name} -> {len(selected)} prompts")

    print("Processed models:")
    for line in summary:
        print(f"- {line}")
    if missing:
        print("\nMissing prompt indices (1-based):")
        for model_name, indices in missing.items():
            joined = ", ".join(str(idx) for idx in indices)
            print(f"- {model_name}: {joined}")


if __name__ == "__main__":
    main()

