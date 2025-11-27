import json
import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.residual_results.loader import iter_multi_pass_records, iter_results, load_results
from src.analysis.residual_results.grids import build_logit_grid, build_residual_grid, build_run_grid
from src.analysis.residual_results.aggregations import layer_statistics, correlate_grids
from src.analysis.residual_results.insights import export_rows, run_metric_hotspots, top_metric_hotspots
from scripts import residual_report


@pytest.fixture()
def sample_file(tmp_path: Path) -> Path:
    payload = [
        {
            "prompt": "Hello world",
            "tokens": ["Hello", "world"],
            "metadata": {"prompt_idx": 0},
            "swap_options": {
                "base": {"embedding_source": "base", "unembedding_source": "base"},
                "sft": {"embedding_source": "sft", "unembedding_source": "sft"},
            },
            "layers": {
                "0": {
                    "0": _position_stats(token="Hello", tracked={"42": {"base": 0.5, "sft": 0.2}}),
                    "1": _position_stats(token="world", tracked={"42": {"base": -0.1, "sft": -0.3}}),
                },
                "1": {
                    "0": _position_stats(token="Hello", tracked={"42": {"base": 0.1, "sft": -0.4}}),
                    "1": _position_stats(token="world", tracked={"42": {"base": -0.2, "sft": -0.6}}),
                },
            },
        }
    ]
    path = tmp_path / "sample.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.fixture()
def multi_pass_file(tmp_path: Path) -> Path:
    run_layer = {
        "0": _run_position_stats(token="Hello"),
        "1": _run_position_stats(token="world"),
    }
    pair_layers = {
        "0": {
            "0": _position_stats(token="Hello", tracked={"42": {"base": 0.5, "sft": 0.2}}),
            "1": _position_stats(token="world", tracked={"42": {"base": -0.1, "sft": -0.3}}),
        }
    }
    payload = [
        {
            "prompt": "Hello world",
            "tokens": ["Hello", "world"],
            "metadata": {"prompt_idx": 0},
            "runs": [
                {
                    "name": "BB",
                    "model": "base",
                    "embedding_source": "base",
                    "unembedding_source": "base",
                    "layers": {"0": run_layer},
                },
                {
                    "name": "BS",
                    "model": "sft",
                    "embedding_source": "base",
                    "unembedding_source": "sft",
                    "layers": {"0": run_layer},
                },
            ],
            "pairwise": [
                {
                    "name": "BB-BS",
                    "base_run": "BB",
                    "sft_run": "BS",
                    "metadata": {"base_run": "BB", "sft_run": "BS"},
                    "swap_options": {
                        "base": {"embedding_source": "base", "unembedding_source": "base"},
                        "sft": {"embedding_source": "sft", "unembedding_source": "sft"},
                    },
                    "layers": pair_layers,
                }
            ],
        }
    ]
    path = tmp_path / "multi.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _position_stats(*, token: str, tracked):
    return {
        "token": token,
        "top_k_increased": {"indices": [42], "values": [0.5]},
        "top_k_decreased": {"indices": [99], "values": [-0.7]},
        "entropy_base": 1.0,
        "entropy_sft": 1.2,
        "kl_div": 0.1,
        "tracked_token_logits": tracked,
        "cosine_sim": 0.9,
        "norm_base": 2.0,
        "norm_sft": 1.5,
        "norm_diff": 0.5,
        "base_cosine_prev": None,
        "base_norm_delta": None,
        "sft_cosine_prev": None,
        "sft_norm_delta": None,
    }


def _run_position_stats(*, token: str):
    return {
        "token": token,
        "entropy": 1.1,
        "norm": 2.0,
        "norm_delta": 0.1,
        "cosine_prev": 0.9,
        "top_k_logits": {"indices": [42], "values": [1.0]},
        "tracked_token_logits": {"42": 0.25},
    }


def test_loader_parses_file(sample_file: Path):
    results = load_results(sample_file)
    assert len(results) == 1
    result = results[0]
    assert result.tokens == ("Hello", "world")
    assert result.layers[0].layer_index == 0
    assert result.layers[0].positions[0].tracked_token_logits[42].base == 0.5


def test_grid_and_aggregation(sample_file: Path):
    result = load_results(sample_file)[0]
    grid = build_residual_grid(result, "norm_diff")
    stats = layer_statistics(grid)
    assert len(stats) == len(result.layers)
    assert stats[0].stats["mean"] == pytest.approx(0.5)


def test_correlation_between_metrics(sample_file: Path):
    result = load_results(sample_file)[0]
    grid_a = build_residual_grid(result, "norm_base")
    grid_b = build_residual_grid(result, "norm_sft")
    corr = correlate_grids(grid_a, grid_b)
    assert math.isnan(corr) or -1.0 <= corr <= 1.0


def test_top_metric_hotspots(sample_file: Path):
    result = load_results(sample_file)[0]
    hotspots = top_metric_hotspots(result, top_n=2)
    assert len(hotspots) == 2
    assert hotspots[0]["metric"].startswith("residual:")


def test_export_rows_csv(tmp_path: Path):
    destination = tmp_path / "out.csv"
    rows = [{"layer": 0, "token_idx": 0, "value": 1.0}]
    output = export_rows(rows, destination)
    assert output.exists()
    assert destination.read_text(encoding="utf-8").splitlines()[1].startswith("0")


def test_cli_smoke(sample_file: Path, capsys):
    residual_report.main([str(sample_file), "--metric", "norm_diff", "--top-n", "1"])
    captured = capsys.readouterr()
    assert "Prompt length" in captured.out


def test_iter_results_handles_multi_pass(multi_pass_file: Path):
    results = list(iter_results(multi_pass_file))
    assert len(results) == 1
    assert results[0].metadata["base_run"] == "BB"


def test_iter_multi_pass_records_parses_runs(multi_pass_file: Path):
    records = list(iter_multi_pass_records(multi_pass_file))
    assert len(records) == 1
    record = records[0]
    assert len(record.runs) == 2
    assert record.runs[0].layers[0].positions[0].token == "Hello"


def test_run_grid_and_hotspots(multi_pass_file: Path):
    record = next(iter_multi_pass_records(multi_pass_file))
    run = record.runs[0]
    grid = build_run_grid(run, "norm")
    assert grid.metric == "run:norm"
    assert grid.values[0][0] == pytest.approx(2.0)
    hotspots = run_metric_hotspots(run, metric="norm", top_n=1)
    assert hotspots[0]["run_name"] == run.name


def test_cli_runs_mode(multi_pass_file: Path, capsys):
    residual_report.main([str(multi_pass_file), "--record-type", "runs", "--metric", "norm", "--top-n", "1"])
    captured = capsys.readouterr()
    assert "Run BB" in captured.out

