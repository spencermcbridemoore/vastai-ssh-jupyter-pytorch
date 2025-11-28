"""
Streaming helpers for working with large residual comparison JSON artifacts.

These utilities keep memory usage bounded by operating on iterators, producing
DataFrame chunks only when requested.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator, Mapping, MutableMapping, Sequence

import ijson
import pandas as pd

from . import aggregations
from . import loader as _loader
from .loader import MultiPassResidualRecord, ResidualResult

PathLike = _loader.PathLike


def stream_raw_records(*paths_or_globs: PathLike) -> Iterator[MutableMapping[str, object]]:
    """
    Lazily yield raw JSON entries from each residual comparison file.

    Files are parsed via ``ijson.items`` so only one record resides in memory.
    """

    for file_path in _loader.iter_result_files(*paths_or_globs):
        with open(file_path, "r", encoding="utf-8") as handle:
            for record in ijson.items(handle, "item"):
                if not isinstance(record, MutableMapping):
                    raise ValueError(f"Expected mapping entries in {file_path}")
                yield record


def stream_pairwise_results(*paths_or_globs: PathLike) -> Iterator[ResidualResult]:
    """
    Yield ``ResidualResult`` dataclasses for each pairwise record lazily.
    """

    for record in stream_raw_records(*paths_or_globs):
        if "runs" in record:
            multi = _loader._parse_multi_pass_record(record)  # type: ignore[attr-defined]
            for pair in multi.pairwise:
                yield pair
            continue
        yield _loader._parse_single_result(record)  # type: ignore[attr-defined]


def stream_multi_pass_records(*paths_or_globs: PathLike) -> Iterator[MultiPassResidualRecord]:
    """Yield multi-pass grouped records lazily."""

    for record in stream_raw_records(*paths_or_globs):
        if "runs" in record:
            yield _loader._parse_multi_pass_record(record)  # type: ignore[attr-defined]


def iter_metric_rows(
    *paths_or_globs: PathLike,
    metadata_fields: Sequence[str] | None = None,
    aggregations_to_run: Sequence[str] | None = None,
) -> Iterator[Mapping[str, object]]:
    """
    Transform streaming ``ResidualResult`` entries into flattened metric rows.

    Args:
        metadata_fields: Optional selection of metadata keys to project into the
            output rows. When omitted, the full metadata mapping is stored
            under ``meta``.
        aggregations_to_run: Optional list of aggregation names registered in
            ``aggregations.AGGREGATION_REGISTRY``. Results are flattened into
            prefix-qualified columns (``agg.<name>.<key>``).
    """

    for result in stream_pairwise_results(*paths_or_globs):
        yield _result_to_row(
            result,
            metadata_fields=metadata_fields,
            aggregations_to_run=aggregations_to_run,
        )


def chunked_metric_frames(
    *paths_or_globs: PathLike,
    chunk_size: int = 512,
    metadata_fields: Sequence[str] | None = None,
    aggregations_to_run: Sequence[str] | None = None,
) -> Iterator[pd.DataFrame]:
    """
    Yield pandas DataFrames built from streaming metric rows.

    Keeping chunk sizes modest (~100s) avoids storing millions of rows in
    memory while still enabling vectorized pandas operations.
    """

    buffer: list[Mapping[str, object]] = []
    for row in iter_metric_rows(
        *paths_or_globs,
        metadata_fields=metadata_fields,
        aggregations_to_run=aggregations_to_run,
    ):
        buffer.append(row)
        if len(buffer) >= chunk_size:
            yield pd.DataFrame(buffer)
            buffer.clear()
    if buffer:
        yield pd.DataFrame(buffer)


def correlate_metric_columns(
    frame: pd.DataFrame, columns: Sequence[str] | None = None
) -> pd.DataFrame:
    """
    Convenience wrapper that returns a correlation matrix for selected columns.
    """

    target_cols = list(columns) if columns else frame.select_dtypes("number").columns
    if not target_cols:
        raise ValueError("No numeric columns available for correlation.")
    return frame.loc[:, target_cols].corr(numeric_only=True)


def _result_to_row(
    result: ResidualResult,
    *,
    metadata_fields: Sequence[str] | None,
    aggregations_to_run: Sequence[str] | None,
) -> Mapping[str, object]:
    """Flatten a single result into scalar-friendly fields."""

    row: dict[str, object] = {
        "prompt": result.prompt,
        "num_layers": result.num_layers(),
        "num_tokens": result.num_tokens(),
        "base_embedding": result.base_swap.embedding_source,
        "base_unembedding": result.base_swap.unembedding_source,
        "sft_embedding": result.sft_swap.embedding_source,
        "sft_unembedding": result.sft_swap.unembedding_source,
    }

    metadata = result.metadata or {}
    if metadata_fields:
        for key in metadata_fields:
            row[f"meta.{key}"] = metadata.get(key)
    else:
        row["meta"] = metadata

    if aggregations_to_run:
        agg_payload = aggregations.run_aggregations(result, aggregations_to_run)
        row.update(_flatten_mapping(agg_payload, prefix="agg"))

    return row


def _flatten_mapping(payload: Mapping[str, object], *, prefix: str) -> Mapping[str, object]:
    """Flatten nested mappings/lists into dot-qualified scalars."""

    flat: dict[str, object] = {}

    def _walk(value: object, key_path: tuple[str, ...]) -> None:
        column_name = ".".join((prefix, *key_path))
        if isinstance(value, Mapping):
            for child_key, child_value in value.items():
                _walk(child_value, (*key_path, str(child_key)))
        elif isinstance(value, (list, tuple)):
            # store as JSON string to keep the column scalar
            flat[column_name] = json.dumps(value)
        else:
            flat[column_name] = value

    for key, val in payload.items():
        _walk(val, (str(key),))
    return flat

