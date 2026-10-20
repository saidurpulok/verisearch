#!/usr/bin/env bash
set -euo pipefail
# Rerun in a NEW output directory; stages refuse to overwrite frozen data/tuning.
experiment_config="${1:-configs/pilot.yaml}"
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/.cache/matplotlib}"
uv run verisearch data prepare --config "$experiment_config"
uv run verisearch generate --config "$experiment_config" --partition fit
uv run verisearch verifier train --config "$experiment_config"
uv run verisearch verifier evaluate --config "$experiment_config" --split validation
uv run verisearch tune --config "$experiment_config"
uv run verisearch generate --config "$experiment_config" --partition test
uv run verisearch eval --config "$experiment_config"
