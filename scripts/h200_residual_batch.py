#!/usr/bin/env python
"""
Utility to drive production residual-comparison runs on a remote H200 NVL host.

The script SSHs into the target instance, launches `experiments/base_vs_sft_residual.py`
with the production prompt file (leveraging the new equal-length trimming), downloads
the emitted JSON artifact after each model pair, and optionally shuts the instance down
when processing completes. It is intentionally conservative: one model runs at a time
and failures do not block subsequent entries.

Example:

    python scripts/h200_residual_batch.py \
        --ssh-host vast-h200 \
        --remote-dir /workspace/vastai-ssh-jupyter-pytorch \
        --local-output ./h200_outputs \
        --entries qwen-0.5b qwen-1.5b qwen-3b \
        --shutdown
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml

JSON_PATH_RE = re.compile(r"Wrote comparison JSON to:\s*(.+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run prod residual comparisons on a remote H200 NVL host, downloading outputs per model.",
    )
    parser.add_argument(
        "--ssh-host",
        required=True,
        help="SSH target in the form user@host (must have key-based auth configured).",
    )
    parser.add_argument(
        "--remote-dir",
        default="/workspace/vastai-ssh-jupyter-pytorch",
        help="Directory on the remote host containing this repository.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/dev_config.yaml"),
        help="Config file that lists residual_compare.model_sweep entries to iterate (default: %(default)s).",
    )
    parser.add_argument(
        "--entries",
        nargs="+",
        help="Subset of model_sweep entry names to run (default: entire sweep block).",
    )
    parser.add_argument(
        "--local-output",
        type=Path,
        default=Path("h200_outputs"),
        help="Local directory where JSON artifacts and logs will be accumulated.",
    )
    parser.add_argument(
        "--identity",
        type=Path,
        help="Optional SSH identity file (passed to ssh/scp via -i).",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device string to export via RESIDUAL_DEVICE (default: %(default)s).",
    )
    parser.add_argument(
        "--extra-env",
        action="append",
        default=[],
        help="Additional KEY=VALUE pairs to export for every run (may be repeated).",
    )
    parser.add_argument(
        "--shutdown",
        action="store_true",
        help="After all runs finish, issue 'sudo shutdown -h now' on the remote host.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip entries that already have a JSON artifact in the local-output/<entry>/ directory.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print full remote command output to stdout as runs progress.",
    )
    return parser.parse_args()


def _load_sweep_entries(config_path: Path, names: Optional[Sequence[str]]) -> List[Dict[str, Any]]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    compare_cfg = config.get("residual_compare") or {}
    sweep = compare_cfg.get("model_sweep")
    if not sweep:
        raise ValueError(
            f"{config_path} is missing residual_compare.model_sweep entries. "
            "Populate it or pass explicit --entries overrides."
        )
    entries: List[Dict[str, Any]] = []
    for row in sweep:
        if "name" not in row or "base_model" not in row or "sft_model" not in row:
            raise ValueError(f"Malformed model_sweep entry: {row}")
        entries.append(dict(row))
    if not names:
        return entries
    name_set = set(names)
    filtered = [entry for entry in entries if entry["name"] in name_set]
    missing = name_set - {entry["name"] for entry in filtered}
    if missing:
        raise ValueError(f"Unknown model_sweep entry name(s): {', '.join(sorted(missing))}")
    return filtered


def _parse_extra_env(pairs: Sequence[str]) -> Dict[str, str]:
    env: Dict[str, str] = {}
    for item in pairs:
        if "=" not in item:
            raise ValueError(f"Invalid --extra-env '{item}'. Expected KEY=VALUE.")
        key, value = item.split("=", 1)
        env[key.strip()] = value.strip()
    return env


def _build_env(entry: Dict[str, Any], device: Optional[str], extra_env: Dict[str, str]) -> Dict[str, str]:
    env: Dict[str, str] = {
        "DEV_MODE": "False",
        "RESIDUAL_BASE_MODEL": entry["base_model"],
        "RESIDUAL_SFT_MODEL": entry["sft_model"],
    }
    if device:
        env["RESIDUAL_DEVICE"] = device
    if entry.get("dtype"):
        env["RESIDUAL_DTYPE"] = str(entry["dtype"])
    if entry.get("tokenizer"):
        env["RESIDUAL_TOKENIZER"] = entry["tokenizer"]
    if entry.get("tokenizer_kwargs"):
        env["RESIDUAL_TOKENIZER_KWARGS"] = json.dumps(entry["tokenizer_kwargs"])
    if entry.get("model_kwargs"):
        env["RESIDUAL_MODEL_KWARGS"] = json.dumps(entry["model_kwargs"])
    env.update(extra_env)
    return env


def _env_exports(env: Dict[str, str]) -> str:
    return " ".join(f"{key}={shlex.quote(str(value))}" for key, value in env.items())


def _ssh_base_cmd(identity: Optional[Path]) -> List[str]:
    base = ["ssh"]
    if identity:
        base.extend(["-i", str(identity)])
    return base


def _scp_base_cmd(identity: Optional[Path]) -> List[str]:
    base = ["scp"]
    if identity:
        base.extend(["-i", str(identity)])
    return base


def _run_remote(ssh_host: str, command: str, identity: Optional[Path]) -> subprocess.CompletedProcess[str]:
    full_cmd = _ssh_base_cmd(identity) + [ssh_host, command]
    return subprocess.run(full_cmd, capture_output=True, text=True)


def _copy_file(ssh_host: str, remote_path: str, local_path: Path, identity: Optional[Path]) -> None:
    normalized = remote_path.replace("\\", "/")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = _scp_base_cmd(identity) + [f"{ssh_host}:{normalized}", str(local_path)]
    subprocess.run(cmd, check=True)


def _extract_json_path(output: str) -> Optional[str]:
    match = JSON_PATH_RE.search(output)
    if match:
        return match.group(1).strip()
    return None


def _write_log(local_dir: Path, entry_name: str, stdout: str, stderr: str) -> Path:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    log_path = local_dir / entry_name / f"{timestamp}_{entry_name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(stdout + ("\n--- stderr ---\n" + stderr if stderr else ""), encoding="utf-8")
    return log_path


def _append_manifest(manifest: Path, row: Dict[str, str]) -> None:
    header_needed = not manifest.exists()
    with manifest.open("a", encoding="utf-8") as handle:
        if header_needed:
            handle.write(",".join(row.keys()) + "\n")
        handle.write(",".join(row.values()) + "\n")


def main() -> None:
    args = parse_args()
    try:
        entries = _load_sweep_entries(args.config, args.entries)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[fatal] {exc}", file=sys.stderr)
        sys.exit(1)

    extra_env = _parse_extra_env(args.extra_env)
    local_root = args.local_output
    local_root.mkdir(parents=True, exist_ok=True)
    manifest_path = local_root / "manifest.csv"

    for entry in entries:
        entry_dir = local_root / entry["name"]
        if args.skip_existing and any(entry_dir.glob("*.json")):
            print(f"[skip] {entry['name']} (local JSON already present)")
            continue

        env_line = _env_exports(_build_env(entry, args.device, extra_env))
        command = (
            f"cd {shlex.quote(str(args.remote_dir))} && "
            f"{env_line} python experiments/base_vs_sft_residual.py"
        )
        print(f"[run] {entry['name']} → {entry['base_model']} vs {entry['sft_model']}")
        result = _run_remote(args.ssh_host, command, args.identity)
        log_path = _write_log(local_root, entry["name"], result.stdout, result.stderr)
        if args.verbose:
            print(result.stdout)
            if result.stderr:
                print(result.stderr, file=sys.stderr)
        if result.returncode != 0:
            print(f"[fail] {entry['name']} (see {log_path})")
            _append_manifest(
                manifest_path,
                {
                    "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
                    "entry": entry["name"],
                    "status": "failed",
                    "json_path": "",
                    "log_path": str(log_path),
                },
            )
            continue

        json_path = _extract_json_path(result.stdout)
        if not json_path:
            print(f"[warn] Could not locate JSON path in output for {entry['name']} (see {log_path})")
            _append_manifest(
                manifest_path,
                {
                    "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
                    "entry": entry["name"],
                    "status": "missing-json",
                    "json_path": "",
                    "log_path": str(log_path),
                },
            )
            continue

        local_json = entry_dir / Path(json_path).name
        try:
            _copy_file(args.ssh_host, json_path, local_json, args.identity)
        except subprocess.CalledProcessError as exc:
            print(f"[warn] Failed to download JSON for {entry['name']}: {exc}")
            _append_manifest(
                manifest_path,
                {
                    "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
                    "entry": entry["name"],
                    "status": "download-failed",
                    "json_path": "",
                    "log_path": str(log_path),
                },
            )
            continue

        print(f"[ok] {entry['name']} → {local_json}")
        _append_manifest(
            manifest_path,
            {
                "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
                "entry": entry["name"],
                "status": "success",
                "json_path": str(local_json),
                "log_path": str(log_path),
            },
        )

    if args.shutdown:
        print("[info] Issuing remote shutdown …")
        shutdown_result = _run_remote(args.ssh_host, "sudo shutdown -h now", args.identity)
        if shutdown_result.returncode != 0:
            print("[warn] Remote shutdown command failed; please verify manually.")
        else:
            print("[info] Shutdown command accepted.")


if __name__ == "__main__":
    main()

