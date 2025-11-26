"""
High-level insight builders and export helpers for residual comparison outputs.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List, Mapping, MutableMapping, Optional, Sequence, Union

try:
    import pandas as pd
except ImportError:  # pragma: no cover - optional
    pd = None  # type: ignore

from .grids import LayerTokenGrid, build_logit_grid, build_residual_grid
from .loader import ResidualResult

PathLike = Union[str, Path]


def top_metric_hotspots(result: ResidualResult, metric: str = "norm_diff", top_n: int = 20) -> List[Mapping[str, object]]:
    """
    Return the largest absolute values for a residual metric across the layer × token grid.
    """
    grid = build_residual_grid(result, metric)
    records = _grid_to_records(grid)
    records.sort(key=lambda row: abs(row["value"]), reverse=True)
    return records[:top_n]


def tracked_token_trajectory(result: ResidualResult, token_id: int) -> List[Mapping[str, object]]:
    """
    Combine residual strength and tracked logits for a token, returning one row per (layer, token_idx).
    """
    norm_grid = build_residual_grid(result, "norm_diff")
    logit_base = build_logit_grid(result, token_id, source="base")
    logit_sft = build_logit_grid(result, token_id, source="sft")
    logit_diff = build_logit_grid(result, token_id, source="diff")

    rows: List[Mapping[str, object]] = []
    for layer_idx, (norm_row, base_row, sft_row, diff_row) in enumerate(
        zip(norm_grid.values, logit_base.values, logit_sft.values, logit_diff.values)
    ):
        layer = norm_grid.layers[layer_idx]
        for token_pos, (norm_value, base_value, sft_value, diff_value) in enumerate(zip(norm_row, base_row, sft_row, diff_row)):
            rows.append(
                {
                    "layer": layer,
                    "token_idx": token_pos,
                    "token": norm_grid.tokens[token_pos],
                    "norm_diff": norm_value,
                    "logit_base": base_value,
                    "logit_sft": sft_value,
                    "logit_diff": diff_value,
                }
            )
    return rows


def relevant_logit_changes(
    result: ResidualResult,
    *,
    min_abs_logit: float = 1.0,
    min_norm_diff: float = 0.0,
    token_ids: Optional[Sequence[int]] = None,
) -> List[Mapping[str, object]]:
    """
    Identify (layer, token) pairs where residual divergence coincides with tracked logit deltas.
    """
    norm_grid = build_residual_grid(result, "norm_diff")
    token_ids = token_ids or _infer_tracked_ids(result)
    findings: List[Mapping[str, object]] = []
    for token_id in token_ids:
        logit_grid = build_logit_grid(result, token_id, source="diff")
        for layer_idx, (norm_row, logit_row) in enumerate(zip(norm_grid.values, logit_grid.values)):
            layer = norm_grid.layers[layer_idx]
            for token_pos, (norm_value, logit_value) in enumerate(zip(norm_row, logit_row)):
                if abs(logit_value) < min_abs_logit:
                    continue
                if abs(norm_value) < min_norm_diff:
                    continue
                findings.append(
                    {
                        "layer": layer,
                        "token_idx": token_pos,
                        "token": norm_grid.tokens[token_pos],
                        "token_id": token_id,
                        "norm_diff": norm_value,
                        "logit_diff": logit_value,
                    }
                )
    findings.sort(key=lambda row: abs(row["logit_diff"]), reverse=True)
    return findings


def export_rows(rows: Sequence[Mapping[str, object]], destination: PathLike, format_hint: Optional[str] = None) -> Path:
    """
    Persist tabular results to CSV/JSON/Parquet.
    """
    destination = Path(destination).expanduser().resolve()
    format_hint = format_hint or destination.suffix.lstrip(".").lower()
    if format_hint not in {"csv", "json", "parquet"}:
        raise ValueError("format_hint must be csv, json, or parquet.")

    destination.parent.mkdir(parents=True, exist_ok=True)

    if format_hint == "csv":
        if not rows:
            destination.touch()
            return destination
        fieldnames = list(rows[0].keys())
        with open(destination, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return destination

    if format_hint == "json":
        with open(destination, "w", encoding="utf-8") as handle:
            json.dump(list(rows), handle, indent=2)
        return destination

    if pd is None:
        raise RuntimeError("pandas is required for Parquet export.")
    frame = pd.DataFrame(rows)
    frame.to_parquet(destination, index=False)
    return destination


def _grid_to_records(grid: LayerTokenGrid) -> List[MutableMapping[str, object]]:
    rows: List[MutableMapping[str, object]] = []
    for layer_idx, row in enumerate(grid.values):
        layer = grid.layers[layer_idx]
        for token_idx, value in enumerate(row):
            rows.append(
                {
                    "layer": layer,
                    "token_idx": token_idx,
                    "token": grid.tokens[token_idx],
                    "value": value,
                    "metric": grid.metric,
                }
            )
    return rows


def _infer_tracked_ids(result: ResidualResult) -> Sequence[int]:
    token_ids = set()
    for layer in result.layers:
        for position in layer.positions:
            token_ids.update(position.tracked_token_logits.keys())
    return sorted(token_ids)

