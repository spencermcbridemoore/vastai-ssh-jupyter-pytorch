"""
Utilities for discovering the latest residual comparison JSON per model.

The helpers here avoid loading files eagerly; they only yield paths and
timestamps so downstream code can decide how/when to stream contents.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import re
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, Mapping, MutableMapping, Sequence

# Root of the repository (two levels above src/analysis/...).
PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_BASE_DIRS: Sequence[Path] = (
    PROJECT_ROOT / "h200_outputs",
    PROJECT_ROOT / "h200_outputs_multi",
)
DEFAULT_PATTERN = "residual_compare_*.json"
TIMESTAMP_RE = re.compile(r"residual_compare_(\d{8})_(\d{6})", re.IGNORECASE)


@dataclass(frozen=True)
class ModelRunFile:
    """Metadata describing a residual comparison artifact."""

    model: str
    path: Path
    timestamp: datetime


def _default_model_resolver(path: Path) -> str:
    """
    Resolve a model identifier for the given residual JSON path.

    By default we use the immediate parent directory (e.g., ``qwen-3b``),
    which works for both single-model and multi-pass directory structures.
    """

    return path.parent.name


def _parse_timestamp(path: Path) -> datetime:
    """Extract the timestamp encoded in the filename."""

    match = TIMESTAMP_RE.search(path.name)
    if not match:
        raise ValueError(f"Could not parse timestamp from {path.name}")
    date_part, time_part = match.groups()
    return datetime.strptime(f"{date_part}{time_part}", "%Y%m%d%H%M%S")


def iter_model_run_files(
    *,
    base_dirs: Iterable[Path | str] = DEFAULT_BASE_DIRS,
    glob_pattern: str = DEFAULT_PATTERN,
    model_resolver: Callable[[Path], str] | None = None,
) -> Iterator[ModelRunFile]:
    """
    Lazily iterate over residual comparison JSON files.

    Args:
        base_dirs: Directories to search (recursively). Paths may be relative
            to the project root or absolute.
        glob_pattern: Glob used to locate result files.
        model_resolver: Optional override to derive model identifiers.

    Yields:
        ``ModelRunFile`` entries sorted by the order the files are discovered.
    """

    resolver = model_resolver or _default_model_resolver
    for base in base_dirs:
        base_path = Path(base)
        if not base_path.is_absolute():
            base_path = PROJECT_ROOT / base_path
        if not base_path.exists():
            continue
        for json_path in base_path.rglob(glob_pattern):
            if not json_path.is_file():
                continue
            yield ModelRunFile(
                model=resolver(json_path),
                path=json_path,
                timestamp=_parse_timestamp(json_path),
            )


def all_latest_jsons(
    *,
    base_dirs: Iterable[Path | str] = DEFAULT_BASE_DIRS,
    glob_pattern: str = DEFAULT_PATTERN,
    model_resolver: Callable[[Path], str] | None = None,
) -> Dict[str, Path]:
    """
    Return the most recent JSON path for every discovered model.

    The function traverses the result files once while keeping only a pointer
    to the latest artifact per model.
    """

    latest: MutableMapping[str, ModelRunFile] = {}
    for entry in iter_model_run_files(
        base_dirs=base_dirs, glob_pattern=glob_pattern, model_resolver=model_resolver
    ):
        current = latest.get(entry.model)
        if current is None or entry.timestamp > current.timestamp:
            latest[entry.model] = entry
    return {model: info.path for model, info in sorted(latest.items())}


def latest_json_for(
    model: str,
    *,
    base_dirs: Iterable[Path | str] = DEFAULT_BASE_DIRS,
    glob_pattern: str = DEFAULT_PATTERN,
    model_resolver: Callable[[Path], str] | None = None,
) -> Path | None:
    """Return the most recent JSON path for a specific model (if available)."""

    latest = None
    for entry in iter_model_run_files(
        base_dirs=base_dirs, glob_pattern=glob_pattern, model_resolver=model_resolver
    ):
        if entry.model != model:
            continue
        if latest is None or entry.timestamp > latest.timestamp:
            latest = entry
    return None if latest is None else latest.path


def list_models(
    *,
    base_dirs: Iterable[Path | str] = DEFAULT_BASE_DIRS,
    glob_pattern: str = DEFAULT_PATTERN,
    model_resolver: Callable[[Path], str] | None = None,
) -> list[str]:
    """List all models that currently have residual comparison outputs."""

    seen: set[str] = set()
    models: list[str] = []
    for entry in iter_model_run_files(
        base_dirs=base_dirs, glob_pattern=glob_pattern, model_resolver=model_resolver
    ):
        if entry.model not in seen:
            seen.add(entry.model)
            models.append(entry.model)
    return models

