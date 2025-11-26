"""
Utilities to materialize residual/logit metrics on a layer × token grid.

These helpers take `ResidualResult` instances produced by `loader.py` and expose
metric-specific 2D views that can subsequently feed aggregations or
visualizations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency
    np = None  # type: ignore

try:
    import pandas as pd
except ImportError:  # pragma: no cover - optional dependency
    pd = None  # type: ignore

from .loader import ResidualResult, ResidualTokenStats

MetricFunc = Callable[[ResidualTokenStats], float]


@dataclass(frozen=True)
class LayerTokenGrid:
    """Dense layer × token matrix for a specific metric."""

    metric: str
    layers: Tuple[int, ...]
    tokens: Tuple[str, ...]
    values: Tuple[Tuple[float, ...], ...]
    metadata: Mapping[str, object] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", self.metadata or {})

    def as_list(self) -> List[List[float]]:
        return [list(row) for row in self.values]

    def as_numpy(self):
        if np is None:
            raise RuntimeError("numpy is not installed; cannot convert grid to ndarray.")
        return np.array(self.values, dtype=float)

    def as_dataframe(self):
        if pd is None:
            raise RuntimeError("pandas is not installed; cannot create DataFrame.")
        columns = pd.MultiIndex.from_tuples(
            [(idx, token) for idx, token in enumerate(self.tokens)],
            names=["token_idx", "token"],
        )
        return pd.DataFrame(self.values, index=self.layers, columns=columns)

    def filter_tokens(self, predicate: Callable[[int, str], bool]) -> "LayerTokenGrid":
        columns = [i for i, token in enumerate(self.tokens) if predicate(i, token)]
        trimmed_values = tuple(tuple(row[i] for i in columns) for row in self.values)
        trimmed_tokens = tuple(self.tokens[i] for i in columns)
        return LayerTokenGrid(
            metric=self.metric,
            layers=self.layers,
            tokens=trimmed_tokens,
            values=trimmed_values,
            metadata=self.metadata,
        )

    def filter_layers(self, predicate: Callable[[int, int], bool]) -> "LayerTokenGrid":
        rows: List[Tuple[float, ...]] = []
        kept_layers: List[int] = []
        for idx, layer in enumerate(self.layers):
            if predicate(idx, layer):
                rows.append(self.values[idx])
                kept_layers.append(layer)
        return LayerTokenGrid(
            metric=self.metric,
            layers=tuple(kept_layers),
            tokens=self.tokens,
            values=tuple(rows),
            metadata=self.metadata,
        )

    def with_metadata(self, **kwargs: object) -> "LayerTokenGrid":
        meta = dict(self.metadata)
        meta.update(kwargs)
        return replace(self, metadata=meta)


def available_residual_metrics() -> Tuple[str, ...]:
    return tuple(sorted(_RESIDUAL_METRICS.keys()))


def build_residual_grid(result: ResidualResult, metric: str) -> LayerTokenGrid:
    if metric not in _RESIDUAL_METRICS:
        raise KeyError(f"Unknown residual metric '{metric}'. Known metrics: {available_residual_metrics()}")
    evaluator = _RESIDUAL_METRICS[metric]
    rows = tuple(_evaluate_layer(layer.positions, evaluator) for layer in result.layers)
    layer_ids = tuple(layer.layer_index for layer in result.layers)
    return LayerTokenGrid(
        metric=f"residual:{metric}",
        layers=layer_ids,
        tokens=result.tokens,
        values=rows,
        metadata={"prompt": result.prompt},
    )


def build_logit_grid(result: ResidualResult, token_id: int, source: str = "diff") -> LayerTokenGrid:
    """
    Build a grid for tracked logits (base, sft, or difference).

    Args:
        token_id: Token id to extract from tracked logits.
        source: One of {"base", "sft", "diff"}.
    """
    if source not in {"base", "sft", "diff"}:
        raise ValueError("source must be 'base', 'sft', or 'diff'.")

    def evaluator(position: ResidualTokenStats) -> float:
        pair = position.tracked_token_logits.get(token_id)
        if pair is None:
            return math.nan
        if source == "base":
            return pair.base
        if source == "sft":
            return pair.sft
        return pair.diff

    rows = tuple(_evaluate_layer(layer.positions, evaluator) for layer in result.layers)
    layer_ids = tuple(layer.layer_index for layer in result.layers)
    return LayerTokenGrid(
        metric=f"logits:{source}",
        layers=layer_ids,
        tokens=result.tokens,
        values=rows,
        metadata={
            "prompt": result.prompt,
            "token_id": token_id,
        },
    )


def build_topk_delta_grid(
    result: ResidualResult,
    token_id: int,
    *,
    direction: str = "increase",
) -> LayerTokenGrid:
    """
    Represent how strongly a token appears in the per-position top-k shifts.
    Returns the recorded delta when present, otherwise NaN.
    """
    if direction not in {"increase", "decrease"}:
        raise ValueError("direction must be 'increase' or 'decrease'.")

    def evaluator(position: ResidualTokenStats) -> float:
        shifts = position.top_k_increased if direction == "increase" else position.top_k_decreased
        for idx, value in zip(shifts.indices, shifts.values):
            if idx == token_id:
                return float(value)
        return math.nan

    rows = tuple(_evaluate_layer(layer.positions, evaluator) for layer in result.layers)
    layer_ids = tuple(layer.layer_index for layer in result.layers)
    return LayerTokenGrid(
        metric=f"topk:{direction}",
        layers=layer_ids,
        tokens=result.tokens,
        values=rows,
        metadata={"prompt": result.prompt, "token_id": token_id},
    )


def iter_tracked_logit_grids(
    result: ResidualResult,
    *,
    sources: Sequence[str] = ("base", "sft", "diff"),
    token_ids: Optional[Iterable[int]] = None,
) -> Iterator[LayerTokenGrid]:
    """
    Yield grids for each tracked token id and requested source.
    """
    token_ids = tuple(token_ids) if token_ids is not None else _collect_tracked_ids(result)
    for token_id in token_ids:
        for source in sources:
            yield build_logit_grid(result, token_id=token_id, source=source)


def _collect_tracked_ids(result: ResidualResult) -> Tuple[int, ...]:
    token_ids = set()
    for layer in result.layers:
        for position in layer.positions:
            token_ids.update(position.tracked_token_logits.keys())
    return tuple(sorted(token_ids))


def _evaluate_layer(positions: Tuple[ResidualTokenStats, ...], evaluator: MetricFunc) -> Tuple[float, ...]:
    row: List[float] = []
    for stats in positions:
        try:
            value = float(evaluator(stats))
        except (TypeError, ValueError):
            value = math.nan
        row.append(value)
    return tuple(row)


def _metric(name: str, func: MetricFunc) -> Tuple[str, MetricFunc]:
    return (name, func)


_RESIDUAL_METRICS: Dict[str, MetricFunc] = dict(
    [
        _metric("norm_base", lambda pos: pos.norm_base),
        _metric("norm_sft", lambda pos: pos.norm_sft),
        _metric("norm_diff", lambda pos: pos.norm_diff),
        _metric("norm_delta", lambda pos: pos.norm_base - pos.norm_sft),
        _metric("cosine_sim", lambda pos: pos.cosine_sim),
        _metric("entropy_base", lambda pos: pos.entropy_base),
        _metric("entropy_sft", lambda pos: pos.entropy_sft),
        _metric("entropy_delta", lambda pos: pos.entropy_base - pos.entropy_sft),
        _metric("kl_div", lambda pos: pos.kl_div),
        _metric("base_cosine_prev", lambda pos: pos.base_cosine_prev if pos.base_cosine_prev is not None else math.nan),
        _metric("sft_cosine_prev", lambda pos: pos.sft_cosine_prev if pos.sft_cosine_prev is not None else math.nan),
        _metric("base_norm_delta", lambda pos: pos.base_norm_delta if pos.base_norm_delta is not None else math.nan),
        _metric("sft_norm_delta", lambda pos: pos.sft_norm_delta if pos.sft_norm_delta is not None else math.nan),
    ]
)

