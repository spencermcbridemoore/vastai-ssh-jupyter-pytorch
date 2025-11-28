#!/usr/bin/env python
"""
Utility to compare two residual_compare JSON outputs.

For each layer/position/metric, it computes absolute and signed deltas, then aggregates
per-layer statistics (mean, std, max). It also handles tracked token logits and reports
overall summary metrics to make it easy to see whether reruns materially differ.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

NUMERIC_KEYS = {
    "entropy_base",
    "entropy_sft",
    "kl_div",
    "cosine_sim",
    "norm_base",
    "norm_sft",
    "norm_diff",
    "base_cosine_prev",
    "sft_cosine_prev",
    "base_norm_delta",
    "sft_norm_delta",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two residual_compare JSON files.")
    parser.add_argument("a", type=Path, help="First JSON file (baseline).")
    parser.add_argument("b", type=Path, help="Second JSON file (comparison).")
    return parser.parse_args()


def load_records(path: Path) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_matching_metadata(left: List[Dict[str, object]], right: List[Dict[str, object]]) -> None:
    if len(left) != len(right):
        raise ValueError(f"Prompt count mismatch: {len(left)} vs {len(right)}")
    for idx, (lrec, rrec) in enumerate(zip(left, right)):
        if lrec["metadata"] != rrec["metadata"]:
            raise ValueError(f"Metadata mismatch at index {idx}: {lrec['metadata']} vs {rrec['metadata']}")


def collect_layer_positions(record: Dict[str, object]) -> List[Tuple[str, str, Dict[str, object]]]:
    entries: List[Tuple[str, str, Dict[str, object]]] = []
    layers = record["layers"]
    for layer_idx, positions in layers.items():
        for pos_idx, payload in positions.items():
            entries.append((layer_idx, pos_idx, payload))
    return entries


def diff_records(left: List[Dict[str, object]], right: List[Dict[str, object]]) -> Dict[str, Dict[str, List[float]]]:
    ensure_matching_metadata(left, right)
    deltas: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    logits_delta: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for lrec, rrec in zip(left, right):
        l_entries = collect_layer_positions(lrec)
        r_entries = collect_layer_positions(rrec)
        if len(l_entries) != len(r_entries):
            raise ValueError("Layer/position count mismatch between runs.")
        for (_, _, lpayload), (_, _, rpayload) in zip(l_entries, r_entries):
            for key in NUMERIC_KEYS:
                lval = lpayload.get(key)
                rval = rpayload.get(key)
                if lval is None or rval is None:
                    continue
                layer_key = key
                deltas[layer_key]["signed"].append(float(rval) - float(lval))
                deltas[layer_key]["abs"].append(abs(float(rval) - float(lval)))
            l_tracked = lpayload.get("tracked_token_logits") or {}
            r_tracked = rpayload.get("tracked_token_logits") or {}
            for token_id, lpair in l_tracked.items():
                if token_id not in r_tracked:
                    continue
                for head in ("base", "sft"):
                    logits_delta[token_id][head].append(float(r_tracked[token_id][head]) - float(lpair[head]))
    return {"metrics": deltas, "tracked": logits_delta}


def summarize(values: List[float]) -> Dict[str, float]:
    arr = np.array(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()) if arr.size else 0.0,
        "std": float(arr.std(ddof=0)) if arr.size else 0.0,
        "max": float(arr.max()) if arr.size else 0.0,
    }


def main() -> None:
    args = parse_args()
    left = load_records(args.a)
    right = load_records(args.b)
    summary = diff_records(left, right)

    print("# Metric deltas")
    for metric, payload in summary["metrics"].items():
        signed = summarize(payload["signed"])
        absolute = summarize(payload["abs"])
        print(
            f"{metric}: signed_mean={signed['mean']:.6e} signed_std={signed['std']:.6e} "
            f"abs_mean={absolute['mean']:.6e} abs_std={absolute['std']:.6e} abs_max={absolute['max']:.6e}"
        )

    print("\n# Tracked token logit deltas")
    for token_id, pair in summary["tracked"].items():
        for head, values in pair.items():
            stats = summarize(values)
            print(
                f"token {token_id} ({head}): mean={stats['mean']:.6e} std={stats['std']:.6e} max={stats['max']:.6e}"
            )


if __name__ == "__main__":
    main()

