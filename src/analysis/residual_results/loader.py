"""
Typed helpers for reading residual comparison experiment outputs.

The experiment at `experiments/base_vs_sft_residual.py` serializes a list of
per-prompt comparison payloads to JSON.  This module converts those records into
dataclasses that are easier to manipulate while preserving the original values.
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

PathLike = Union[str, Path]


@dataclass(frozen=True)
class SwapOptions:
    embedding_source: str
    unembedding_source: str


@dataclass(frozen=True)
class TopKShift:
    indices: Tuple[int, ...]
    values: Tuple[float, ...]

    def as_pairs(self) -> Tuple[Tuple[int, float], ...]:
        """Return (token_id, delta) tuples for convenience."""
        return tuple(zip(self.indices, self.values))


@dataclass(frozen=True)
class TokenLogitPair:
    base: float
    sft: float

    @property
    def diff(self) -> float:
        """Convenience accessor for base minus SFT logits."""
        return self.base - self.sft


@dataclass(frozen=True)
class ResidualTokenStats:
    token: str
    top_k_increased: TopKShift
    top_k_decreased: TopKShift
    entropy_base: float
    entropy_sft: float
    kl_div: float
    cosine_sim: float
    norm_base: float
    norm_sft: float
    norm_diff: float
    base_cosine_prev: Optional[float]
    base_norm_delta: Optional[float]
    sft_cosine_prev: Optional[float]
    sft_norm_delta: Optional[float]
    tracked_token_logits: Mapping[int, TokenLogitPair]

    def logit_for_token(self, token_id: int) -> Optional[TokenLogitPair]:
        """Look up tracked logits for a token id if present."""
        return self.tracked_token_logits.get(token_id)


@dataclass(frozen=True)
class ResidualLayerStats:
    layer_index: int
    positions: Tuple[ResidualTokenStats, ...]

    def position(self, idx: int) -> ResidualTokenStats:
        return self.positions[idx]

    def token_grid(self) -> Tuple[str, ...]:
        """Return the sequence of tokens aligned with this layer."""
        return tuple(pos.token for pos in self.positions)


@dataclass(frozen=True)
class ResidualRunTokenStats:
    token: str
    entropy: float
    norm: float
    norm_delta: Optional[float]
    cosine_prev: Optional[float]
    top_k_logits: TopKShift
    tracked_token_logits: Mapping[int, float]


@dataclass(frozen=True)
class ResidualRunLayerStats:
    layer_index: int
    positions: Tuple[ResidualRunTokenStats, ...]

    def position(self, idx: int) -> ResidualRunTokenStats:
        return self.positions[idx]


@dataclass(frozen=True)
class ResidualRunRecord:
    name: str
    model: str
    embedding_source: str
    unembedding_source: str
    metadata: Mapping[str, Any]
    layers: Tuple[ResidualRunLayerStats, ...]


@dataclass(frozen=True)
class ResidualResult:
    prompt: str
    tokens: Tuple[str, ...]
    metadata: Mapping[str, Any]
    base_swap: SwapOptions
    sft_swap: SwapOptions
    layers: Tuple[ResidualLayerStats, ...]

    def num_layers(self) -> int:
        return len(self.layers)

    def num_tokens(self) -> int:
        return len(self.tokens)


@dataclass(frozen=True)
class MultiPassResidualRecord:
    prompt: str
    tokens: Tuple[str, ...]
    metadata: Mapping[str, Any]
    runs: Tuple[ResidualRunRecord, ...]
    pairwise: Tuple[ResidualResult, ...]


@dataclass(frozen=True)
class ResultFileSummary:
    path: Path
    num_results: int
    total_tokens: int
    avg_tokens: float


def iter_result_files(*paths_or_globs: PathLike) -> Iterator[Path]:
    """
    Yield concrete file paths resolved from explicit files, directories, or glob patterns.
    """
    if not paths_or_globs:
        raise ValueError("At least one path or glob must be provided.")

    for item in paths_or_globs:
        path = Path(item)
        if path.is_dir():
            yield from sorted(path.rglob("residual_compare_*.json"))
            continue
        if path.exists():
            yield path
            continue
        # Treat as glob relative to current working directory.
        for match in Path().glob(str(path)):
            if match.is_file():
                yield match.resolve()


def load_results(path: PathLike) -> List[ResidualResult]:
    """Read a single JSON result file into memory."""
    return list(iter_results(path))


def iter_results(*paths_or_globs: PathLike) -> Iterator[ResidualResult]:
    """Yield ResidualResult entries (pairwise diffs) from each provided file/glob lazily."""
    for record in _iter_raw_records(*paths_or_globs):
        if "runs" in record:
            multi_record = _parse_multi_pass_record(record)
            for pair in multi_record.pairwise:
                yield pair
            continue
        yield _parse_single_result(record)


def iter_multi_pass_records(*paths_or_globs: PathLike) -> Iterator[MultiPassResidualRecord]:
    """Yield multi-pass entries (if present) from the provided files."""
    for record in _iter_raw_records(*paths_or_globs):
        if "runs" in record:
            yield _parse_multi_pass_record(record)


def summarize_file(path: PathLike) -> ResultFileSummary:
    """Return simple stats about a result file (count + token averages)."""
    results = load_results(path)
    token_counts = [len(item.tokens) for item in results]
    avg = statistics.mean(token_counts) if token_counts else math.nan
    return ResultFileSummary(
        path=Path(path),
        num_results=len(results),
        total_tokens=sum(token_counts),
        avg_tokens=avg,
    )


def _iter_raw_records(*paths_or_globs: PathLike) -> Iterator[MutableMapping[str, Any]]:
    for file_path in iter_result_files(*paths_or_globs):
        with open(file_path, "r", encoding="utf-8") as handle:
            raw_payload = json.load(handle)
        if not isinstance(raw_payload, list):
            raise ValueError(f"Expected list at root of {file_path}, found {type(raw_payload).__name__}")
        for record in raw_payload:
            if not isinstance(record, MutableMapping):
                raise ValueError(f"Each record must be a mapping in {file_path}")
            yield record


def _parse_single_result(record: MutableMapping[str, Any]) -> ResidualResult:
    prompt = str(record.get("prompt", ""))
    tokens = tuple(str(token) for token in record.get("tokens", []))
    metadata = record.get("metadata") or {}
    swap_opts = record.get("swap_options") or {}
    base_swap = _parse_swap(swap_opts.get("base") or {})
    sft_swap = _parse_swap(swap_opts.get("sft") or {})

    raw_layers = record.get("layers")
    if not isinstance(raw_layers, Mapping):
        raise ValueError("Record missing 'layers' mapping.")

    layer_entries: List[ResidualLayerStats] = []
    for layer_key in sorted(raw_layers.keys(), key=lambda x: int(x)):
        layer_payload = raw_layers[layer_key]
        if not isinstance(layer_payload, Mapping):
            raise ValueError(f"Layer '{layer_key}' payload must be a mapping.")
        positions = _parse_positions(layer_payload)
        layer_entries.append(
            ResidualLayerStats(
                layer_index=int(layer_key),
                positions=positions,
            )
        )

    return ResidualResult(
        prompt=prompt,
        tokens=tokens,
        metadata=metadata,
        base_swap=base_swap,
        sft_swap=sft_swap,
        layers=tuple(layer_entries),
    )


def _parse_multi_pass_record(record: MutableMapping[str, Any]) -> MultiPassResidualRecord:
    prompt = str(record.get("prompt", ""))
    tokens = tuple(str(token) for token in record.get("tokens", []))
    metadata = record.get("metadata") or {}

    run_payloads = record.get("runs") or []
    runs: List[ResidualRunRecord] = []
    for entry in run_payloads:
        if not isinstance(entry, Mapping):
            raise ValueError("Each multi_pass run entry must be a mapping.")
        runs.append(_parse_run_entry(entry))

    pair_payloads = record.get("pairwise") or []
    pair_results: List[ResidualResult] = []
    for entry in pair_payloads:
        if not isinstance(entry, MutableMapping):
            raise ValueError("Each multi_pass pairwise entry must be a mapping.")
        normalized = _normalize_pair_entry(entry, prompt, tokens, metadata)
        pair_results.append(_parse_single_result(normalized))

    return MultiPassResidualRecord(
        prompt=prompt,
        tokens=tokens,
        metadata=metadata,
        runs=tuple(runs),
        pairwise=tuple(pair_results),
    )


def _parse_swap(payload: Mapping[str, Any]) -> SwapOptions:
    return SwapOptions(
        embedding_source=str(payload.get("embedding_source", "base")),
        unembedding_source=str(payload.get("unembedding_source", "base")),
    )


def _parse_positions(layer_payload: Mapping[str, Any]) -> Tuple[ResidualTokenStats, ...]:
    ordered_positions: List[ResidualTokenStats] = []
    for pos_key in sorted(layer_payload.keys(), key=lambda x: int(x)):
        pos_payload = layer_payload[pos_key]
        if not isinstance(pos_payload, Mapping):
            raise ValueError(f"Layer position '{pos_key}' must be a mapping.")
        ordered_positions.append(_parse_position_stats(pos_payload))
    return tuple(ordered_positions)


def _parse_run_entry(payload: Mapping[str, Any]) -> ResidualRunRecord:
    name = str(payload.get("name", "")).strip()
    if not name:
        raise ValueError("Run entry missing 'name'.")
    model = str(payload.get("model", "")).strip()
    embedding = str(payload.get("embedding_source", "")).strip()
    unembedding = str(payload.get("unembedding_source", "")).strip()
    layers_payload = payload.get("layers")
    if not isinstance(layers_payload, Mapping):
        raise ValueError(f"Run '{name}' layers payload must be a mapping.")
    layers = _parse_run_layers(layers_payload)
    metadata = payload.get("metadata") or {}
    return ResidualRunRecord(
        name=name,
        model=model,
        embedding_source=embedding,
        unembedding_source=unembedding,
        metadata=metadata,
        layers=layers,
    )


def _parse_run_layers(payload: Mapping[str, Any]) -> Tuple[ResidualRunLayerStats, ...]:
    layers: List[ResidualRunLayerStats] = []
    for layer_key in sorted(payload.keys(), key=lambda x: int(x)):
        entries = payload[layer_key]
        if not isinstance(entries, Mapping):
            raise ValueError(f"Run layer '{layer_key}' payload must be a mapping.")
        positions = _parse_run_positions(entries)
        layers.append(
            ResidualRunLayerStats(
                layer_index=int(layer_key),
                positions=positions,
            )
        )
    return tuple(layers)


def _parse_run_positions(payload: Mapping[str, Any]) -> Tuple[ResidualRunTokenStats, ...]:
    ordered: List[ResidualRunTokenStats] = []
    for pos_key in sorted(payload.keys(), key=lambda x: int(x)):
        entry = payload[pos_key]
        if not isinstance(entry, Mapping):
            raise ValueError(f"Run position '{pos_key}' must be a mapping.")
        ordered.append(_parse_run_position(entry))
    return tuple(ordered)


def _parse_run_position(payload: Mapping[str, Any]) -> ResidualRunTokenStats:
    tracked_payload = payload.get("tracked_token_logits") or {}
    tracked: Dict[int, float] = {}
    for token_id, value in tracked_payload.items():
        if value is None:
            continue
        try:
            tracked[int(token_id)] = float(value)
        except (TypeError, ValueError):
            continue
    return ResidualRunTokenStats(
        token=str(payload.get("token", "")),
        entropy=float(payload.get("entropy", 0.0)),
        norm=float(payload.get("norm", 0.0)),
        norm_delta=_optional_float(payload.get("norm_delta")),
        cosine_prev=_optional_float(payload.get("cosine_prev")),
        top_k_logits=_parse_topk(payload.get("top_k_logits")),
        tracked_token_logits=tracked,
    )


def _normalize_pair_entry(
    entry: MutableMapping[str, Any],
    prompt: str,
    tokens: Tuple[str, ...],
    parent_metadata: Mapping[str, Any],
) -> MutableMapping[str, Any]:
    normalized: MutableMapping[str, Any] = dict(entry)
    normalized.setdefault("prompt", prompt)
    normalized.setdefault("tokens", list(tokens))
    existing_meta = normalized.get("metadata") or {}
    merged_meta = {**parent_metadata, **existing_meta}
    normalized["metadata"] = merged_meta
    return normalized


def _parse_position_stats(position: Mapping[str, Any]) -> ResidualTokenStats:
    tracked_logits_payload = position.get("tracked_token_logits") or {}
    tracked_logits = {
        int(token_id): TokenLogitPair(base=float(values["base"]), sft=float(values["sft"]))
        for token_id, values in tracked_logits_payload.items()
        if isinstance(values, Mapping) and "base" in values and "sft" in values
    }

    return ResidualTokenStats(
        token=str(position.get("token", "")),
        top_k_increased=_parse_topk(position.get("top_k_increased")),
        top_k_decreased=_parse_topk(position.get("top_k_decreased")),
        entropy_base=float(position.get("entropy_base", 0.0)),
        entropy_sft=float(position.get("entropy_sft", 0.0)),
        kl_div=float(position.get("kl_div", 0.0)),
        cosine_sim=float(position.get("cosine_sim", 0.0)),
        norm_base=float(position.get("norm_base", 0.0)),
        norm_sft=float(position.get("norm_sft", 0.0)),
        norm_diff=float(position.get("norm_diff", 0.0)),
        base_cosine_prev=_optional_float(position.get("base_cosine_prev")),
        base_norm_delta=_optional_float(position.get("base_norm_delta")),
        sft_cosine_prev=_optional_float(position.get("sft_cosine_prev")),
        sft_norm_delta=_optional_float(position.get("sft_norm_delta")),
        tracked_token_logits=tracked_logits,
    )


def _parse_topk(payload: Any) -> TopKShift:
    if not isinstance(payload, Mapping):
        return TopKShift(indices=tuple(), values=tuple())

    indices = tuple(int(idx) for idx in payload.get("indices", []))
    values = tuple(float(val) for val in payload.get("values", []))
    return TopKShift(indices=indices, values=values)


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

