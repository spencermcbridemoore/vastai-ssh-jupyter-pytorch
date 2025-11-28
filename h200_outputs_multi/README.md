# Multi-Pass Residual Outputs

This directory stores the no-fuzzing, four-pass (BB/BS/SB/SS) runs plus all six pairwise differences.

Layout:
- `h200_outputs_multi/<pair>/` — timestamped logs + JSON for each model pair.
- `manifest_multi.csv` — append-only tracker of run metadata (timestamp, prompt_count, prompt_variants, etc.).

Each JSON record uses the new format with top-level `runs` and `pairwise` blocks.
