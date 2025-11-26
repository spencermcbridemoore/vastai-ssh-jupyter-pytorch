"""
Toolkit for loading and analyzing residual comparison experiment outputs.

Modules:
    loader: Typed helpers that read experiment JSON outputs safely.
    grids:  Layer/token aligned data structures and transforms.
    aggregations: Statistical reductions and derived metrics.
    insights: Higher-level summaries and export utilities.
"""

from .loader import (
    ResidualResult,
    ResidualLayerStats,
    ResidualTokenStats,
    ResultFileSummary,
    iter_result_files,
    iter_results,
    load_results,
)

__all__ = [
    "ResidualResult",
    "ResidualLayerStats",
    "ResidualTokenStats",
    "ResultFileSummary",
    "iter_result_files",
    "iter_results",
    "load_results",
]

