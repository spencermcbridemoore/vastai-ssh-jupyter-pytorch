#!/usr/bin/env python
"""
Condense every residual_compare JSON under notebooks/h200_long_outputs into a
single per-model artifact that preserves full activation payloads.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Mapping, MutableMapping, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("notebooks/h200_long_outputs"),
        help="Directory that contains per-model residual outputs (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where condensed files will be written "
        "(default: <root>/condensed).",
    )
    parser.add_argument(
        "--pattern",
        default="residual_compare_*.json",
        help="Glob used to discover residual JSON files (default: %(default)s).",
    )
    parser.add_argument(
        "--variant-name",
        dest="variant_names",
        action="append",
        default=None,
        help="Restrict to records whose metadata.variant_name matches one of the "
        "provided values. Repeat to allow multiple variants. Defaults to all.",
    )
    parser.add_argument(
        "--embedding-variant",
        dest="embedding_variants",
        action="append",
        default=None,
        help="Restrict to records whose metadata.embedding_variant matches one of "
        "the provided values. Repeat to allow multiple variants. Defaults to all.",
    )
    parser.add_argument(
        "--prompt-idx",
        dest="prompt_indices",
        action="append",
        type=int,
        default=None,
        help="Restrict to specific prompt_idx values. Repeat as needed. Defaults to all.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Enumerate the files that would be written without touching disk.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-file statistics while processing.",
    )
    return parser.parse_args()


def discover_model_dirs(root: Path) -> Iterable[Path]:
    for model_dir in sorted(root.iterdir()):
        if model_dir.is_dir():
            yield model_dir


def discover_residual_files(model_dir: Path, pattern: str) -> List[Path]:
    return sorted(model_dir.glob(pattern))


def load_payload(path: Path) -> Sequence[MutableMapping[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Sequence):
        raise ValueError(f"{path} does not contain a JSON list.")
    return payload  # type: ignore[return-value]


def record_matches(
    record: Mapping[str, object],
    *,
    variant_names: Sequence[str] | None,
    embedding_variants: Sequence[str] | None,
    prompt_indices: Sequence[int] | None,
) -> bool:
    metadata = record.get("metadata")
    if not isinstance(metadata, Mapping):
        return False
    if variant_names and metadata.get("variant_name") not in variant_names:
        return False
    if embedding_variants and metadata.get("embedding_variant") not in embedding_variants:
        return False
    if prompt_indices is not None:
        idx = metadata.get("prompt_idx")
        if idx not in prompt_indices:
            return False
    return True


def summarize_file(records: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    prompt_indices = []
    variant_names = []
    embedding_variants = []
    for record in records:
        metadata = record.get("metadata")
        if not isinstance(metadata, Mapping):
            continue
        prompt_indices.append(metadata.get("prompt_idx"))
        variant_names.append(metadata.get("variant_name"))
        embedding_variants.append(metadata.get("embedding_variant"))
    summary = {
        "record_count": len(records),
        "prompt_indices": sorted({idx for idx in prompt_indices if idx is not None}),
        "variant_names": sorted({name for name in variant_names if isinstance(name, str)}),
        "embedding_variants": sorted(
            {name for name in embedding_variants if isinstance(name, str)}
        ),
    }
    return summary


def collect_model_records(
    model_dir: Path,
    *,
    pattern: str,
    variant_names: Sequence[str] | None,
    embedding_variants: Sequence[str] | None,
    prompt_indices: Sequence[int] | None,
    verbose: bool,
) -> tuple[list[MutableMapping[str, object]], list[dict[str, object]]]:
    aggregated: list[MutableMapping[str, object]] = []
    file_summaries: list[dict[str, object]] = []
    for json_path in discover_residual_files(model_dir, pattern):
        payload = load_payload(json_path)
        kept: list[MutableMapping[str, object]] = []
        for idx, record in enumerate(payload):
            if not isinstance(record, MutableMapping):
                continue
            if variant_names or embedding_variants or prompt_indices is not None:
                if not record_matches(
                    record,
                    variant_names=variant_names,
                    embedding_variants=embedding_variants,
                    prompt_indices=prompt_indices,
                ):
                    continue
            record["_source"] = {"file": json_path.name, "record_index": idx}
            aggregated.append(record)
            kept.append(record)
        summary = summarize_file(kept)
        if kept:
            summary = {
                "file": json_path.name,
                **summary,
            }
            file_summaries.append(summary)
        if verbose:
            print(
                f"[{model_dir.name}] {json_path.name}: "
                f"{summary.get('record_count', 0)} records kept"
            )
    return aggregated, file_summaries


def compute_prompt_counts(records: Sequence[Mapping[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        metadata = record.get("metadata")
        if not isinstance(metadata, Mapping):
            key = "unknown"
        else:
            idx = metadata.get("prompt_idx")
            key = str(idx) if idx is not None else "unknown"
        counts[key] = counts.get(key, 0) + 1
    return counts


def write_model_file(
    *,
    model_name: str,
    model_dir: Path,
    records: Sequence[MutableMapping[str, object]],
    file_summaries: Sequence[Mapping[str, object]],
    output_dir: Path,
    dry_run: bool,
) -> Path:
    output_path = output_dir / f"condensed_{model_name}.json"
    payload = {
        "model": model_name,
        "source_root": str(model_dir),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "record_count": len(records),
        "prompt_counts": compute_prompt_counts(records),
        "source_files": list(file_summaries),
        "records": list(records),
    }
    if dry_run:
        return output_path
    output_dir.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return output_path


def main() -> None:
    args = parse_args()
    root = args.root
    if not root.exists():
        raise SystemExit(f"Root directory not found: {root}")
    output_dir = args.output_dir or (root / "condensed")

    total_models = 0
    written = 0
    skipped = 0
    for model_dir in discover_model_dirs(root):
        total_models += 1
        records, file_summaries = collect_model_records(
            model_dir,
            pattern=args.pattern,
            variant_names=args.variant_names,
            embedding_variants=args.embedding_variants,
            prompt_indices=args.prompt_indices,
            verbose=args.verbose,
        )
        if not records:
            skipped += 1
            if args.verbose:
                print(f"[{model_dir.name}] No matching records, skipping.")
            continue
        output_path = write_model_file(
            model_name=model_dir.name,
            model_dir=model_dir,
            records=records,
            file_summaries=file_summaries,
            output_dir=output_dir,
            dry_run=args.dry_run,
        )
        written += 1
        action = "Would write" if args.dry_run else "Wrote"
        print(f"{action} {output_path} ({len(records)} records).")

    print(
        f"Models processed: {total_models}. "
        f"Condensed: {written}. "
        f"Skipped (no matches): {skipped}."
    )


if __name__ == "__main__":
    main()


