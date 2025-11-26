"""
CLI utility to derive residual/logit summaries from experiment output files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Sequence

from src.analysis.residual_results import iter_results
from src.analysis.residual_results.aggregations import layer_statistics, token_statistics
from src.analysis.residual_results.grids import (
    LayerTokenGrid,
    available_residual_metrics,
    build_logit_grid,
    build_residual_grid,
    build_topk_delta_grid,
)
from src.analysis.residual_results.insights import export_rows, relevant_logit_changes, top_metric_hotspots


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Residual JSON file(s), directories, or glob patterns.",
    )
    parser.add_argument(
        "--metric",
        action="append",
        default=["norm_diff"],
        choices=available_residual_metrics(),
        help="Residual metrics to summarize (default: norm_diff).",
    )
    parser.add_argument(
        "--axis",
        choices=["layer", "token"],
        default="layer",
        help="Axis for aggregations (layer or token level).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of hotspots/logit findings to print per prompt.",
    )
    parser.add_argument(
        "--token-id",
        dest="token_ids",
        action="append",
        type=int,
        help="Tracked token id to analyze (can be repeated). Defaults to all tracked ids.",
    )
    parser.add_argument(
        "--logit-source",
        choices=["base", "sft", "diff"],
        default="diff",
        help="Which logit values to grid when token ids are provided.",
    )
    parser.add_argument(
        "--export",
        type=Path,
        help="Optional path to persist combined hotspot rows (format inferred from suffix).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    export_rows_buffer: List[dict] = []

    for result in iter_results(*args.inputs):
        print("=" * 80)
        meta_preview = json.dumps(result.metadata)
        print(f"Prompt length: {len(result.tokens)} tokens")
        print(f"Metadata: {meta_preview}")

        for metric in args.metric:
            grid = build_residual_grid(result, metric)
            summaries = _summaries_for_axis(grid, axis=args.axis)
            print(f"\nMetric '{metric}' ({args.axis} stats):")
            _print_summaries(summaries, axis=args.axis, max_rows=args.top_n)

            hotspots = top_metric_hotspots(result, metric=metric, top_n=args.top_n)
            export_rows_buffer.extend({**row, "metric": metric, "prompt_meta": result.metadata} for row in hotspots)
            print(f"\nTop {len(hotspots)} |layer,token| hotspots:")
            _print_rows(hotspots)

        # Optional logit analyses
        findings = relevant_logit_changes(
            result,
            token_ids=args.token_ids,
            min_abs_logit=0.5,
            min_norm_diff=0.0,
        )
        if findings:
            print(f"\nTracked logit changes (top {args.top_n}):")
            _print_rows(findings[: args.top_n])
            export_rows_buffer.extend({**row, "type": "logit_change", "prompt_meta": result.metadata} for row in findings)

        if args.token_ids:
            for token_id in args.token_ids:
                logit_grid = build_logit_grid(result, token_id=token_id, source=args.logit_source)
                topk_grid = build_topk_delta_grid(result, token_id=token_id)
                print(f"\nToken {token_id} logit grid preview ({args.logit_source}):")
                _print_grid_preview(logit_grid, limit=args.top_n)
                print(f"Token {token_id} top-k membership (increase):")
                _print_grid_preview(topk_grid, limit=args.top_n)

    if args.export and export_rows_buffer:
        export_rows(export_rows_buffer, args.export)
        print(f"\nExported {len(export_rows_buffer)} rows to {args.export}")


def _summaries_for_axis(grid: LayerTokenGrid, axis: str) -> Sequence[dict]:
    if axis == "layer":
        summaries = layer_statistics(grid)
        return [{"layer": entry.layer_index, **entry.stats} for entry in summaries]
    summaries = token_statistics(grid)
    return [{"token_idx": entry.token_index, "token": entry.token, **entry.stats} for entry in summaries]


def _print_summaries(rows: Sequence[dict], *, axis: str, max_rows: int) -> None:
    if not rows:
        print("  (no data)")
        return
    header_keys = list(rows[0].keys())
    for row in rows[:max_rows]:
        summary = ", ".join(f"{key}={row[key]:.3f}" if isinstance(row[key], (int, float)) else f"{key}={row[key]}" for key in header_keys)
        print(f"  {summary}")


def _print_rows(rows: Sequence[dict]) -> None:
    if not rows:
        print("  (none)")
        return
    keys = rows[0].keys()
    for row in rows:
        formatted = ", ".join(f"{key}={row[key]}" for key in keys)
        print(f"  {formatted}")


def _print_grid_preview(grid: LayerTokenGrid, limit: int = 5) -> None:
    for layer_idx, row in enumerate(grid.values[:limit]):
        layer = grid.layers[layer_idx]
        subset = ", ".join(f"{grid.tokens[idx]}:{value:.3f}" for idx, value in enumerate(row[:limit]))
        print(f"  L{layer}: {subset}")


if __name__ == "__main__":
    main()

