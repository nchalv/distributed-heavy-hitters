#!/usr/bin/env bash
set -euo pipefail

# Section 8.2.1: calibrate epsilon_M and alpha using the deployed baseline
# controller (upward response + residual-guarded downward probing), with the
# ambiguity guard disabled.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

HH_BENCH="${PROJECT_ROOT}/evaluator/build/hh_bench"
OUT_DIR="${OUT_DIR:-experiments/calibration/margin_service_calibration}"
if [[ "${OUT_DIR}" != /* ]]; then
  OUT_DIR="${PROJECT_ROOT}/${OUT_DIR}"
fi
CSV_DIR="${OUT_DIR}/csv"

M="${M:-100}"
MEM_KIB="${MEM_KIB:-128}"
TOPK="${TOPK:-200}"

if [[ -n "${EPSILON_M_VALUES:-}" ]]; then
  read -r -a EPSILON_M_GRID <<< "${EPSILON_M_VALUES}"
else
  EPSILON_M_GRID=(0.1 0.15 0.2 0.25)
fi
if [[ -n "${ALPHA_GRID_VALUES:-}" ]]; then
  read -r -a ALPHA_VALUES <<< "${ALPHA_GRID_VALUES}"
else
  ALPHA_VALUES=(0.90 0.95 0.98 0.995)
fi

SS_EPS="${SS_EPS:-per-item}"
RES_GUARD_WINDOW="${RES_GUARD_WINDOW:-2}"
CLEAN_OUTPUT="${CLEAN_OUTPUT:-1}"
DRY_RUN="${DRY_RUN:-0}"

declare -A STREAMS=(
  [constrained_certificate_adversary]="${PROJECT_ROOT}/generator/generated/runs/calibration_synthetic_constrained_certificate_adversary/streams"
  [milp_certificate_adversary_n100]="${PROJECT_ROOT}/generator/generated/runs/calibration_synthetic_milp_certificate_adversary_n100/streams"
  [milp_certificate_adversary_n200]="${PROJECT_ROOT}/generator/generated/runs/calibration_synthetic_milp_certificate_adversary_n200/streams"
  [milp_certificate_adversary_n400]="${PROJECT_ROOT}/generator/generated/runs/calibration_synthetic_milp_certificate_adversary_n400/streams"
  [persistent_ambiguity_adversary_n100]="${PROJECT_ROOT}/generator/generated/runs/calibration_synthetic_persistent_ambiguity_adversary_n100/streams"
  [persistent_ambiguity_adversary_n200]="${PROJECT_ROOT}/generator/generated/runs/calibration_synthetic_persistent_ambiguity_adversary_n200/streams"
  [persistent_ambiguity_adversary_n400]="${PROJECT_ROOT}/generator/generated/runs/calibration_synthetic_persistent_ambiguity_adversary_n400/streams"
)

declare -A N_PARAMS=(
  [constrained_certificate_adversary]=200
  [milp_certificate_adversary_n100]=100
  [milp_certificate_adversary_n200]=200
  [milp_certificate_adversary_n400]=400
  [persistent_ambiguity_adversary_n100]=100
  [persistent_ambiguity_adversary_n200]=200
  [persistent_ambiguity_adversary_n400]=400
)

DATASETS=(${DATASETS:-milp_certificate_adversary_n100 milp_certificate_adversary_n200 milp_certificate_adversary_n400})

cd "${PROJECT_ROOT}"

if [[ "${DRY_RUN}" != "1" ]]; then
  cmake -S evaluator -B evaluator/build -DCMAKE_BUILD_TYPE=Release
  cmake --build evaluator/build --target hh_bench
fi

if [[ "${DRY_RUN}" != "1" ]]; then
  if [[ "${CLEAN_OUTPUT}" == "1" ]]; then
    rm -rf \
      "${CSV_DIR}" \
      "${OUT_DIR}/plots" \
      "${OUT_DIR}/summary.csv" \
      "${OUT_DIR}/configuration.txt"
  fi
  mkdir -p "${CSV_DIR}"
  {
    printf 'experiment=margin_service_calibration\n'
    printf 'datasets=%s\n' "${DATASETS[*]}"
    printf 'epsilon_m_values=%s\n' "${EPSILON_M_GRID[*]}"
    printf 'alpha_values=%s\n' "${ALPHA_VALUES[*]}"
    printf 'm=%s\n' "${M}"
    printf 'memory_kib=%s\n' "${MEM_KIB}"
    printf 'topk=%s\n' "${TOPK}"
    printf 'residual_guard_window=%s\n' "${RES_GUARD_WINDOW}"
    printf 'downward_probing=on\n'
    printf 'probe_residual_guard=on\n'
    printf 'ambiguity_adjustment=off\n'
    printf 'ss_error_mode=%s\n' "${SS_EPS}"
  } > "${OUT_DIR}/configuration.txt"
fi

for dataset in "${DATASETS[@]}"; do
  if [[ -z "${STREAMS[$dataset]:-}" ]]; then
    echo "unknown dataset: ${dataset}" >&2
    echo "known datasets: ${!STREAMS[*]}" >&2
    exit 1
  fi

  n_param="${N_PARAMS[$dataset]}"
  stream="${STREAMS[$dataset]}"
  if [[ ! -d "${stream}" ]]; then
    echo "missing generated stream directory: ${stream}" >&2
    echo "generate it first with the matching generator/config/runs/calibration_synthetic_${dataset}.json" >&2
    exit 1
  fi

  for epsilon_m in "${EPSILON_M_GRID[@]}"; do
    rm="$(awk -v epsilon_m="${epsilon_m}" -v n="${n_param}" 'BEGIN { printf "%.10g", epsilon_m / n }')"
    for alpha in "${ALPHA_VALUES[@]}"; do
      method="oracle,ss[policy=difficulty alpha-req=${alpha} r-m=${rm} res-guard-window=${RES_GUARD_WINDOW} downward-probing=on probe-residual-guard=on amb-adjust=off diff-mode=predictive ss-eps=${SS_EPS}]"
      csv_out="${CSV_DIR}/${dataset}_rm${rm}_alpha${alpha}.csv"

      echo "running ${dataset}: n=${n_param}, epsilon_m=${epsilon_m}, r_m=${rm}, alpha=${alpha}, downward_probing=on, residual_guard=on, ambiguity=off"
      command=(
        "${HH_BENCH}"
        "${stream}"
        "${M}"
        "${n_param}"
        "${MEM_KIB}"
        "${method}"
        --topk "${TOPK}"
        --csv-out "${csv_out}"
      )
      if [[ "${DRY_RUN}" == "1" ]]; then
        printf '  '
        printf '%q ' "${command[@]}"
        printf '\n'
      else
        "${command[@]}"
      fi
    done
  done
done

if [[ "${DRY_RUN}" == "1" ]]; then
  exit 0
fi

SUMMARY_ARGS=()
for dataset in "${DATASETS[@]}"; do
  SUMMARY_ARGS+=(--dataset "${dataset}")
done

python3 experiments/calibration/summarize_rm_alpha.py --root "${CSV_DIR}" "${SUMMARY_ARGS[@]}" > "${OUT_DIR}/summary.csv"
echo "summary: ${OUT_DIR}/summary.csv"
python3 experiments/calibration/plot_rm_alpha_grid.py --summary "${OUT_DIR}/summary.csv" --out "${OUT_DIR}/plots"
python3 experiments/calibration/plot_rm_alpha_topology.py \
  --summary "${OUT_DIR}/summary.csv" \
  --out "${OUT_DIR}/plots"
