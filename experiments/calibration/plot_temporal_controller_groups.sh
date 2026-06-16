#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCHEDULED_DIR="${PROJECT_ROOT}/experiments/calibration/temporal_controller_ablation"
MILP_DIR="${PROJECT_ROOT}/experiments/calibration/temporal_controller_ablation_milp_n200"
ROUND_ROBIN_DIR="${PROJECT_ROOT}/experiments/calibration/temporal_controller_ablation_round_robin_n200"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/experiments/calibration/temporal_controller_ablation_combined/plots}"

cd "${PROJECT_ROOT}"
mkdir -p "${OUT_DIR}"

python3 experiments/calibration/plot_temporal_grid.py \
  --summary "${SCHEDULED_DIR}/summary.csv" \
  --csv-root "${SCHEDULED_DIR}/csv" \
  --dataset-group scheduled \
  --out "${OUT_DIR}"

python3 experiments/calibration/plot_temporal_grid.py \
  --summary "${ROUND_ROBIN_DIR}/summary.csv" \
  --summary "${MILP_DIR}/summary.csv" \
  --csv-root "${ROUND_ROBIN_DIR}/csv" \
  --csv-root "${MILP_DIR}/csv" \
  --dataset-group stationary \
  --out "${OUT_DIR}"

echo "grouped temporal-controller plots: ${OUT_DIR}"
