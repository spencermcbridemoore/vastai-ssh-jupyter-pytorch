"""
Toolkit for loading and analyzing residual comparison experiment outputs.

Modules:
    loader: Typed helpers that read experiment JSON outputs safely.
    grids:  Layer/token aligned data structures and transforms.
    aggregations: Statistical reductions and derived metrics.
    insights: Higher-level summaries and export utilities.
"""

from .loader import (
    MultiPassResidualRecord,
    ResidualResult,
    ResidualLayerStats,
    ResidualTokenStats,
    ResidualRunLayerStats,
    ResidualRunRecord,
    ResidualRunTokenStats,
    ResultFileSummary,
    iter_result_files,
    iter_results,
    iter_multi_pass_records,
    load_results,
)

__all__ = [
    "MultiPassResidualRecord",
    "ResidualResult",
    "ResidualLayerStats",
    "ResidualTokenStats",
    "ResidualRunLayerStats",
    "ResidualRunRecord",
    "ResidualRunTokenStats",
    "ResultFileSummary",
    "iter_result_files",
    "iter_results",
    "iter_multi_pass_records",
    "load_results",
]

