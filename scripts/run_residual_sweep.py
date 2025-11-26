#!/usr/bin/env python
"""
Batch runner for experiments/base_vs_sft_residual.py on a single A100 40 GB.

Reads the residual_compare.model_sweep block from the chosen config file and
invokes the experiment once per model pair, logging stdout/stderr per run.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "dev_config.yaml"
RUNNER_PATH = REPO_ROOT / "experiments" / "base_vs_sft_residual.py"


def _resolve_persistent_dir() -> Path:
    """Prefer /workspace/persistent but fall back to repo-local storage."""
    default = Path("/workspace") / "persistent"
    try:
        default.mkdir(parents=True, exist_ok=True)
        return default
    except PermissionError:
        fallback = REPO_ROOT / "persistent"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback
    except OSError:
        fallback = REPO_ROOT / "persistent"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


PERSISTENT_DIR = _resolve_persistent_dir()
ANALYSIS_DIR = PERSISTENT_DIR / "analyses" / "residual_compare"
LOG_DIR = ANALYSIS_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the residual comparison experiment for multiple model pairs.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Config file containing residual_compare.model_sweep (default: %(default)s)",
    )
    parser.add_argument(
        "--names",
        nargs="+",
        help="Subset of sweep entry names to run (default: run all).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of sweep entries to run (after filtering).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip entries that already have a .done marker in the log directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands that would run without executing them.",
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=0,
        help="CUDA device index to target (default: %(default)s).",
    )
    parser.add_argument(
        "--require-gpu-substring",
        default="A100",
        help="GPU name substring required for execution (default: %(default)s).",
    )
    parser.add_argument(
        "--require-vram-gb",
        type=float,
        default=39.0,
        help="Minimum VRAM (GiB) required on the selected GPU (default: %(default)s).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip GPU capability checks.",
    )
    return parser.parse_args()


def ensure_gpu(args: argparse.Namespace) -> Dict[str, Any]:
    if args.force:
        return {"name": "forced", "total_gib": None, "index": args.device_index}
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; cannot run the sweep.")
    if args.device_index >= torch.cuda.device_count():
        raise RuntimeError(
            f"device_index={args.device_index} is invalid for this host "
            f"(only {torch.cuda.device_count()} CUDA device(s) detected)."
        )
    torch.cuda.set_device(args.device_index)
    props = torch.cuda.get_device_properties(args.device_index)
    name = props.name
    total_gib = props.total_memory / (1024**3)
    if args.require_gpu_substring and args.require_gpu_substring.lower() not in name.lower():
        raise RuntimeError(
            f"Selected GPU '{name}' does not contain substring '{args.require_gpu_substring}'. "
            "Use --force to override."
        )
    if total_gib + 1e-6 < args.require_vram_gb:
        raise RuntimeError(
            f"GPU '{name}' exposes {total_gib:.2f} GiB, which is below the required "
            f"{args.require_vram_gb} GiB. Use --force to override."
        )
    return {"name": name, "total_gib": total_gib, "index": args.device_index}


def load_sweep_entries(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    compare_cfg = config.get("residual_compare") or {}
    model_sweep = compare_cfg.get("model_sweep")
    if not model_sweep:
        raise ValueError(
            f"No residual_compare.model_sweep block found in {config_path}. "
            "Add at least one entry before running the sweep."
        )
    ordered_entries: List[Dict[str, Any]] = []
    for raw in model_sweep:
        entry = dict(raw)
        if "name" not in entry:
            raise ValueError(f"model_sweep entry is missing a 'name': {raw}")
        if "base_model" not in entry or "sft_model" not in entry:
            raise ValueError(f"model_sweep entry '{entry['name']}' must include base_model and sft_model.")
        ordered_entries.append(entry)
    return {"entries": ordered_entries, "compare_defaults": compare_cfg}


def filter_entries(
    entries: Iterable[Dict[str, Any]],
    names: Optional[List[str]],
) -> List[Dict[str, Any]]:
    entries_list = list(entries)
    if not names:
        return entries_list
    name_set = set(names)
    filtered = [entry for entry in entries_list if entry["name"] in name_set]
    missing = name_set - {entry["name"] for entry in filtered}
    if missing:
        raise ValueError(f"Unknown sweep entry name(s): {', '.join(sorted(missing))}")
    return filtered


def build_env(entry: Dict[str, Any], args: argparse.Namespace) -> Dict[str, str]:
    env = os.environ.copy()
    env["RESIDUAL_BASE_MODEL"] = entry["base_model"]
    env["RESIDUAL_SFT_MODEL"] = entry["sft_model"]
    env["RESIDUAL_DEVICE"] = f"cuda:{args.device_index}"
    if entry.get("dtype"):
        env["RESIDUAL_DTYPE"] = str(entry["dtype"])
    if entry.get("tokenizer"):
        env["RESIDUAL_TOKENIZER"] = entry["tokenizer"]
    if entry.get("tokenizer_kwargs"):
        env["RESIDUAL_TOKENIZER_KWARGS"] = json.dumps(entry["tokenizer_kwargs"])
    if entry.get("model_kwargs"):
        env["RESIDUAL_MODEL_KWARGS"] = json.dumps(entry["model_kwargs"])
    return env


def run_entry(
    entry: Dict[str, Any],
    env: Dict[str, str],
    idx: int,
    total: int,
    args: argparse.Namespace,
) -> Path:
    log_name = f"{idx:02d}_{entry['name']}.log"
    log_path = LOG_DIR / log_name
    marker_path = LOG_DIR / f"{entry['name']}.done"

    if args.skip_existing and marker_path.exists():
        print(f"[skip] {entry['name']} (marker exists)")
        return marker_path

    cmd = [sys.executable, str(RUNNER_PATH)]
    banner = (
        f"# Sweep entry: {entry['name']}\n"
        f"# Base: {entry['base_model']} | SFT: {entry['sft_model']}\n"
        f"# Estimated VRAM (GiB): {entry.get('estimated_vram_gb', 'unknown')}\n"
        f"# Command: {' '.join(cmd)}\n"
    )

    if args.dry_run:
        print(f"[dry-run] {entry['name']} -> {log_path}")
        with log_path.open("w", encoding="utf-8") as handle:
            handle.write(banner)
        return marker_path

    print(f"[{idx}/{total}] Running {entry['name']}…")
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(banner)
        handle.write("# --- Begin experiment output ---\n")
        handle.flush()
        result = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        handle.write("# --- End experiment output ---\n")
    if result.returncode != 0:
        raise RuntimeError(
            f"Sweep entry '{entry['name']}' failed with exit code {result.returncode}. "
            f"See log: {log_path}"
        )

    payload = {
        "entry": entry,
        "completed_at": datetime.utcnow().isoformat() + "Z",
        "log_path": str(log_path),
        "command": cmd,
    }
    marker_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return marker_path


def main() -> None:
    args = parse_args()
    gpu_info = ensure_gpu(args)
    print(
        f"Using CUDA device {gpu_info['index']} ({gpu_info['name']}, "
        f"{gpu_info['total_gib']:.2f} GiB)"
        if gpu_info["total_gib"] is not None
        else f"Using CUDA device {gpu_info['index']} (GPU checks skipped)"
    )

    sweep_bundle = load_sweep_entries(args.config)
    entries = filter_entries(sweep_bundle["entries"], args.names)
    if args.limit is not None:
        entries = entries[: args.limit]
    if not entries:
        print("No sweep entries to process.")
        return

    print(f"Preparing to run {len(entries)} sweep entr{'y' if len(entries)==1 else 'ies'}")
    completed: List[Path] = []
    for idx, entry in enumerate(entries, start=1):
        env = build_env(entry, args)
        marker = run_entry(entry, env, idx, len(entries), args)
        completed.append(marker)

    print(f"Completed {len(completed)} sweep entr{'y' if len(completed)==1 else 'ies'}.")


if __name__ == "__main__":
    main()

