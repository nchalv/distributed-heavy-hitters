#!/usr/bin/env bash
set -euo pipefail

# Section 8.2.3: ambiguity-guard ablation after fixing the margin profile and
# guarded margin-comfort temporal controller.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HH_BENCH="${PROJECT_ROOT}/evaluator/build/hh_bench"

OUT_DIR="${OUT_DIR:-experiments/calibration/ambiguity_guard_ablation}"
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
  EPSILON_M_GRID=(${EPSILON_M:-0.10})
fi
ALPHA_REQ="${ALPHA_REQ:-0.95}"
AMBIGUITY_BETA="${AMBIGUITY_BETA:-0.50}"
SS_EPS="${SS_EPS:-per-item}"

RUN_MODES=(${RUN_MODES:-baseline ambiguity})
CLEAN_OUTPUT="${CLEAN_OUTPUT:-1}"
DRY_RUN="${DRY_RUN:-0}"

declare -A STREAMS=(
  [milp_certificate_adversary_n100]="${PROJECT_ROOT}/generator/generated/runs/calibration_synthetic_milp_certificate_adversary_n100/streams"
  [milp_certificate_adversary_n200]="${PROJECT_ROOT}/generator/generated/runs/calibration_synthetic_milp_certificate_adversary_n200/streams"
  [milp_certificate_adversary_n400]="${PROJECT_ROOT}/generator/generated/runs/calibration_synthetic_milp_certificate_adversary_n400/streams"
  [persistent_ambiguity_adversary_n100]="${PROJECT_ROOT}/generator/generated/runs/calibration_synthetic_persistent_ambiguity_adversary_n100/streams"
  [persistent_ambiguity_adversary_n200]="${PROJECT_ROOT}/generator/generated/runs/calibration_synthetic_persistent_ambiguity_adversary_n200/streams"
  [persistent_ambiguity_adversary_n400]="${PROJECT_ROOT}/generator/generated/runs/calibration_synthetic_persistent_ambiguity_adversary_n400/streams"
  [round_robin_n200]="${PROJECT_ROOT}/generator/generated/runs/calibration_synthetic_round_robin_n200/streams"
)

declare -A N_PARAMS=(
  [milp_certificate_adversary_n100]=100
  [milp_certificate_adversary_n200]=200
  [milp_certificate_adversary_n400]=400
  [persistent_ambiguity_adversary_n100]=100
  [persistent_ambiguity_adversary_n200]=200
  [persistent_ambiguity_adversary_n400]=400
  [round_robin_n200]=200
)

DATASETS=(${DATASETS:-persistent_ambiguity_adversary_n100 persistent_ambiguity_adversary_n200 persistent_ambiguity_adversary_n400 milp_certificate_adversary_n100 milp_certificate_adversary_n200 milp_certificate_adversary_n400})

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
    printf 'experiment=ambiguity_guard_ablation\n'
    printf 'datasets=%s\n' "${DATASETS[*]}"
    printf 'm=%s\n' "${M}"
    printf 'epsilon_m_values=%s\n' "${EPSILON_M_GRID[*]}"
    printf 'alpha=%s\n' "${ALPHA_REQ}"
    printf 'ambiguity_beta=%s\n' "${AMBIGUITY_BETA}"
    printf 'temporal_controller=guarded_margin_comfort\n'
    printf 'modes=%s\n' "${RUN_MODES[*]}"
    printf 'memory_kib=%s\n' "${MEM_KIB}"
    printf 'topk=%s\n' "${TOPK}"
    printf 'ss_error_mode=%s\n' "${SS_EPS}"
  } > "${OUT_DIR}/configuration.txt"
fi

for dataset in "${DATASETS[@]}"; do
  if [[ -z "${STREAMS[$dataset]:-}" ]]; then
    echo "unknown dataset: ${dataset}" >&2
    echo "known datasets: ${!STREAMS[*]}" >&2
    exit 1
  fi

  stream="${STREAMS[$dataset]}"
  n_param="${N_PARAMS[$dataset]}"
  if [[ ! -d "${stream}" ]]; then
    echo "missing generated stream directory: ${stream}" >&2
    exit 1
  fi

  for epsilon_m in "${EPSILON_M_GRID[@]}"; do
    r_m="$(awk -v epsilon_m="${epsilon_m}" -v n="${n_param}" 'BEGIN { printf "%.10g", epsilon_m / n }')"
    epsilon_tag="$(printf '%s' "${epsilon_m}" | tr -c '[:alnum:]' '_')"
    for mode in "${RUN_MODES[@]}"; do
      case "${mode}" in
        baseline) amb_adjust=off ;;
        ambiguity) amb_adjust=on ;;
        *)
          echo "unknown mode: ${mode}; expected baseline or ambiguity" >&2
          exit 1
          ;;
      esac

      method="oracle,ss[policy=difficulty alpha-req=${ALPHA_REQ} r-m=${r_m} res-guard-window=2 res-guard-decay=${AMBIGUITY_BETA} downward-probing=on probe-residual-guard=on probe-strategy=comfort probe-pressure-gate=0.5 amb-adjust=${amb_adjust} diff-mode=predictive ss-eps=${SS_EPS}]"
      csv_out="${CSV_DIR}/${dataset}_eps${epsilon_tag}_${mode}.csv"

      echo "running ambiguity ablation: dataset=${dataset}, n=${n_param}, epsilon_m=${epsilon_m}, mode=${mode}"
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

python3 experiments/calibration/summarize_ambiguity_grid.py \
  --root "${CSV_DIR}" \
  > "${OUT_DIR}/summary.csv"

python3 experiments/calibration/plot_ambiguity_grid.py \
  --summary "${OUT_DIR}/summary.csv" \
  --out "${OUT_DIR}/plots"

echo "summary: ${OUT_DIR}/summary.csv"
echo "plots: ${OUT_DIR}/plots"
