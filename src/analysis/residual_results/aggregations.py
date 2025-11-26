"""
Aggregation helpers that operate on layer × token grids.

The functions here transform dense grids into descriptive statistics grouped by
layer, token, or the full prompt, and expose a registry-based mechanism for
derived metrics such as correlations between two residual metrics.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from .grids import LayerTokenGrid, build_residual_grid
from .loader import ResidualResult

Reducer = Callable[[Sequence[float]], float]
AggregationFn = Callable[[ResidualResult], Mapping[str, object]]


@dataclass(frozen=True)
class LayerStats:
    layer_index: int
    stats: Mapping[str, float]


@dataclass(frozen=True)
class TokenStats:
    token_index: int
    token: str
    stats: Mapping[str, float]


DEFAULT_REDUCERS: Mapping[str, Reducer] = {
    "mean": statistics.fmean,
    "std": lambda values: statistics.pstdev(values) if len(values) > 1 else 0.0,
    "min": min,
    "max": max,
}

AGGREGATION_REGISTRY: Dict[str, AggregationFn] = {}


def register_aggregation(name: str) -> Callable[[AggregationFn], AggregationFn]:
    def decorator(func: AggregationFn) -> AggregationFn:
        AGGREGATION_REGISTRY[name] = func
        return func

    return decorator


def layer_statistics(grid: LayerTokenGrid, reducers: Optional[Mapping[str, Reducer]] = None) -> Tuple[LayerStats, ...]:
    reducers = reducers or DEFAULT_REDUCERS
    summaries: List[LayerStats] = []
    for layer_idx, row in zip(grid.layers, grid.values):
        stats = {name: _apply_reducer(row, reducer) for name, reducer in reducers.items()}
        summaries.append(LayerStats(layer_index=layer_idx, stats=stats))
    return tuple(summaries)


def token_statistics(grid: LayerTokenGrid, reducers: Optional[Mapping[str, Reducer]] = None) -> Tuple[TokenStats, ...]:
    reducers = reducers or DEFAULT_REDUCERS
    columns = list(zip(*grid.values))
    summaries: List[TokenStats] = []
    for token_idx, column in enumerate(columns):
        stats = {name: _apply_reducer(column, reducer) for name, reducer in reducers.items()}
        summaries.append(
            TokenStats(
                token_index=token_idx,
                token=grid.tokens[token_idx],
                stats=stats,
            )
        )
    return tuple(summaries)


def correlate_grids(
    grid_a: LayerTokenGrid,
    grid_b: LayerTokenGrid,
    *,
    drop_na: bool = True,
) -> float:
    """Compute Pearson correlation between two equally shaped grids."""
    if len(grid_a.values) != len(grid_b.values) or len(grid_a.values[0]) != len(grid_b.values[0]):
        raise ValueError("Grid shapes must match for correlation.")

    paired_a: List[float] = []
    paired_b: List[float] = []
    for row_a, row_b in zip(grid_a.values, grid_b.values):
        for val_a, val_b in zip(row_a, row_b):
            if drop_na and (math.isnan(val_a) or math.isnan(val_b)):
                continue
            paired_a.append(val_a)
            paired_b.append(val_b)

    if not paired_a or len(paired_a) != len(paired_b):
        return math.nan

    mean_a = statistics.fmean(paired_a)
    mean_b = statistics.fmean(paired_b)
    numerator = sum((a - mean_a) * (b - mean_b) for a, b in zip(paired_a, paired_b))
    denom_a = math.sqrt(sum((a - mean_a) ** 2 for a in paired_a))
    denom_b = math.sqrt(sum((b - mean_b) ** 2 for b in paired_b))
    if denom_a == 0 or denom_b == 0:
        return math.nan
    return numerator / (denom_a * denom_b)


def run_aggregations(result: ResidualResult, names: Optional[Iterable[str]] = None) -> Mapping[str, object]:
    selected = names or AGGREGATION_REGISTRY.keys()
    output: Dict[str, object] = {}
    for name in selected:
        if name not in AGGREGATION_REGISTRY:
            raise KeyError(f"Aggregation '{name}' is not registered.")
        output[name] = AGGREGATION_REGISTRY[name](result)
    return output


def _apply_reducer(values: Sequence[float], reducer: Reducer) -> float:
    clean = [value for value in values if not math.isnan(value)]
    if not clean:
        return math.nan
    return float(reducer(clean))


@register_aggregation("norm_vs_kl_corr")
def _norm_vs_kl_corr(result: ResidualResult) -> Mapping[str, float]:
    norm_grid = build_residual_grid(result, "norm_diff")
    kl_grid = build_residual_grid(result, "kl_div")
    return {"corr": correlate_grids(norm_grid, kl_grid)}


@register_aggregation("entropy_delta_stats")
def _entropy_delta_stats(result: ResidualResult) -> Mapping[str, object]:
    entropy_grid = build_residual_grid(result, "entropy_delta")
    layers = layer_statistics(entropy_grid)
    tokens = token_statistics(entropy_grid)
    return {
        "layer_stats": [summary.stats for summary in layers],
        "token_stats": [summary.stats for summary in tokens],
    }


@register_aggregation("residual_strength")
def _residual_strength(result: ResidualResult) -> Mapping[str, float]:
    base_grid = build_residual_grid(result, "norm_base")
    sft_grid = build_residual_grid(result, "norm_sft")
    avg_base = statistics.fmean(_flatten(base_grid.values))
    avg_sft = statistics.fmean(_flatten(sft_grid.values))
    return {
        "mean_norm_base": avg_base,
        "mean_norm_sft": avg_sft,
        "mean_norm_delta": avg_base - avg_sft,
    }


def _flatten(values: Tuple[Tuple[float, ...], ...]) -> List[float]:
    flattened: List[float] = []
    for row in values:
        flattened.extend(value for value in row if not math.isnan(value))
    return flattened

