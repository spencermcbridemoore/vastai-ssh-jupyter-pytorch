"""Summaries for residual metrics suitable for box/whisker plots."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

from .grids import available_residual_metrics, build_residual_grid
from .loader import PathLike, iter_results

BOX_KEYS = ("min", "q1", "median", "q3", "max", "mean", "std")
DEFAULT_METRICS: Sequence[str] = available_residual_metrics()


def summarize_file_metrics(
    path: PathLike,
    *,
    metrics: Optional[Sequence[str]] = None,
) -> Mapping[str, Mapping[str, float]]:
    """
    Reduce each prompt in a residual JSON to a scalar per metric, then emit box summary stats.
    """
    metric_names = list(metrics) if metrics else list(DEFAULT_METRICS)
    if not metric_names:
        raise ValueError("No metrics provided for summary.")

    aggregates: Dict[str, List[float]] = {name: [] for name in metric_names}
    for result in iter_results(path):
        for metric in metric_names:
            grid = build_residual_grid(result, metric)
            scalar = _grid_mean(grid.values)
            aggregates[metric].append(scalar)

    return {metric: _box_summary(values) for metric, values in aggregates.items()}


def save_metric_summary(
    path: PathLike,
    output_path: PathLike,
    *,
    metrics: Optional[Sequence[str]] = None,
) -> Path:
    """
    Convenience helper to persist metric summaries to disk.
    """
    summary = summarize_file_metrics(path, metrics=metrics)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return output


def _grid_mean(values: Sequence[Sequence[float]]) -> float:
    flattened: List[float] = []
    for row in values:
        for value in row:
            if value is None or math.isnan(value):
                continue
            flattened.append(float(value))
    if not flattened:
        return math.nan
    return statistics.fmean(flattened)


def _box_summary(values: Sequence[float]) -> Mapping[str, float]:
    clean = [float(value) for value in values if value is not None and not math.isnan(value)]
    if not clean:
        return {key: math.nan for key in BOX_KEYS}

    clean.sort()
    mins = clean[0]
    maxs = clean[-1]
    mean = statistics.fmean(clean)
    std = statistics.pstdev(clean) if len(clean) > 1 else 0.0
    median = statistics.median(clean)
    try:
        quartiles = statistics.quantiles(clean, n=4, method="inclusive")
        q1, _, q3 = quartiles
    except statistics.StatisticsError:
        q1 = q3 = median

    return {
        "min": mins,
        "q1": q1,
        "median": median,
        "q3": q3,
        "max": maxs,
        "mean": mean,
        "std": std,
    }


__all__ = [
    "BOX_KEYS",
    "DEFAULT_METRICS",
    "save_metric_summary",
    "summarize_file_metrics",
]

