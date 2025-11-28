"""
Toolkit for loading and analyzing residual comparison experiment outputs.

Modules:
    loader: Typed helpers that read experiment JSON outputs safely.
    grids:  Layer/token aligned data structures and transforms.
    aggregations: Statistical reductions and derived metrics.
    insights: Higher-level summaries and export utilities.
"""

from .latest_runs import (
    ModelRunFile,
    all_latest_jsons,
    iter_model_run_files,
    latest_json_for,
    list_models,
)
from .loader import (
    MultiPassResidualRecord,
    ResidualLayerStats,
    ResidualResult,
    ResidualRunLayerStats,
    ResidualRunRecord,
    ResidualRunTokenStats,
    ResidualTokenStats,
    ResultFileSummary,
    iter_multi_pass_records,
    iter_result_files,
    iter_results,
    load_results,
)
from .loader_streaming import (
    chunked_metric_frames,
    correlate_metric_columns,
    iter_metric_rows,
    stream_multi_pass_records,
    stream_pairwise_results,
    stream_raw_records,
)
from .visualize import (
    plot_correlation_heatmap,
    plot_metric_distribution,
    plot_metric_scatter,
    plot_metric_trend,
)

__all__ = [
    "ModelRunFile",
    "MultiPassResidualRecord",
    "ResidualResult",
    "ResidualLayerStats",
    "ResidualTokenStats",
    "ResidualRunLayerStats",
    "ResidualRunRecord",
    "ResidualRunTokenStats",
    "ResultFileSummary",
    "all_latest_jsons",
    "chunked_metric_frames",
    "correlate_metric_columns",
    "iter_metric_rows",
    "iter_result_files",
    "iter_results",
    "iter_multi_pass_records",
    "iter_model_run_files",
    "latest_json_for",
    "list_models",
    "load_results",
    "stream_multi_pass_records",
    "stream_pairwise_results",
    "stream_raw_records",
    "plot_correlation_heatmap",
    "plot_metric_distribution",
    "plot_metric_scatter",
    "plot_metric_trend",
]

