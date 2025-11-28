"""
Visualization helpers for residual comparison analytics.

Each helper accepts either pandas DataFrames or any iterable of mappings that
can be coerced into a DataFrame, so notebook workflows can pass lazy iterators
without pre-materializing giant tables.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_metric_distribution(
    data: pd.DataFrame | Iterable[Mapping[str, object]],
    *,
    column: str,
    bins: int = 30,
    ax: plt.Axes | None = None,
    kde: bool = False,
    **hist_kwargs,
) -> plt.Axes:
    """Plot a histogram for a numeric metric column."""

    frame = _ensure_frame(data)
    target_ax = ax or plt.gca()
    sns.histplot(
        frame[column].astype(float),
        bins=bins,
        ax=target_ax,
        kde=kde,
        **hist_kwargs,
    )
    target_ax.set_title(f"Distribution of {column}")
    target_ax.set_xlabel(column)
    target_ax.set_ylabel("Count")
    return target_ax


def plot_metric_scatter(
    data: pd.DataFrame | Iterable[Mapping[str, object]],
    *,
    x: str,
    y: str,
    hue: str | None = None,
    style: str | None = None,
    ax: plt.Axes | None = None,
    **scatter_kwargs,
) -> plt.Axes:
    """Scatter plot comparing two numeric metrics."""

    frame = _ensure_frame(data)
    target_ax = ax or plt.gca()
    sns.scatterplot(
        data=frame,
        x=x,
        y=y,
        hue=hue,
        style=style,
        ax=target_ax,
        **scatter_kwargs,
    )
    target_ax.set_title(f"{y} vs {x}")
    return target_ax


def plot_metric_trend(
    data: pd.DataFrame | Iterable[Mapping[str, object]],
    *,
    x: str,
    y: str,
    hue: str | None = None,
    estimator: str | None = "mean",
    ax: plt.Axes | None = None,
    **line_kwargs,
) -> plt.Axes:
    """Line plot showing how a metric evolves across an ordered axis."""

    frame = _ensure_frame(data)
    target_ax = ax or plt.gca()
    sns.lineplot(
        data=frame,
        x=x,
        y=y,
        hue=hue,
        estimator=estimator,
        ax=target_ax,
        **line_kwargs,
    )
    target_ax.set_title(f"{y} by {x}")
    return target_ax


def plot_correlation_heatmap(
    frame: pd.DataFrame,
    *,
    columns: Sequence[str],
    ax: plt.Axes | None = None,
    cmap: str = "coolwarm",
    annot: bool = True,
) -> plt.Axes:
    """Heatmap for the correlation matrix of a subset of columns."""

    corr = frame.loc[:, columns].corr(numeric_only=True)
    target_ax = ax or plt.gca()
    sns.heatmap(corr, ax=target_ax, cmap=cmap, annot=annot, fmt=".2f")
    target_ax.set_title("Metric Correlation Heatmap")
    return target_ax


def _ensure_frame(data: pd.DataFrame | Iterable[Mapping[str, object]]) -> pd.DataFrame:
    """Coerce arbitrary iterables of mappings into a pandas DataFrame."""

    if isinstance(data, pd.DataFrame):
        return data
    return pd.DataFrame(list(data))

