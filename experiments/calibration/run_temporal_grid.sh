#!/usr/bin/env bash
set -euo pipefail

# Section 8.2.2: temporal-controller ablation using the quality-oriented
# margin profile selected in Section 8.2.1.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

HH_BENCH="${PROJECT_ROOT}/evaluator/build/hh_bench"
OUT_DIR="${OUT_DIR:-experiments/calibration/temporal_controller_ablation}"
if [[ "${OUT_DIR}" != /* ]]; then
  OUT_DIR="${PROJECT_ROOT}/${OUT_DIR}"
fi
CSV_DIR="${OUT_DIR}/csv"

M="${M:-100}"
MEM_KIB="${MEM_KIB:-128}"
TOPK="${TOPK:-200}"

EPSILON_M="${EPSILON_M:-0.10}"
ALPHA_REQ="${ALPHA_REQ:-0.90}"

SS_EPS="${SS_EPS:-per-item}"
RUN_MODES=(${RUN_MODES:-probing residual_guarded_probing comfort_guided_probing pressure_gated_comfort_probing})
CLEAN_OUTPUT="${CLEAN_OUTPUT:-1}"
DRY_RUN="${DRY_RUN:-0}"

declare -A STREAMS=(
  [round_robin_n200]="${PROJECT_ROOT}/generator/generated/runs/calibration_synthetic_round_robin_n200/streams"
  [milp_certificate_adversary_n200]="${PROJECT_ROOT}/generator/generated/runs/calibration_synthetic_milp_certificate_adversary_n200/streams"
  [temporal_guard_step_schedule_n200]="${PROJECT_ROOT}/generator/generated/runs/temporal_guard_step_schedule_n200/streams"
  [temporal_guard_ramp_schedule_n200]="${PROJECT_ROOT}/generator/generated/runs/temporal_guard_ramp_schedule_n200/streams"
  [temporal_guard_burst_schedule_n200]="${PROJECT_ROOT}/generator/generated/runs/temporal_guard_burst_schedule_n200/streams"
  [temporal_guard_oscillation_schedule_n200]="${PROJECT_ROOT}/generator/generated/runs/temporal_guard_oscillation_schedule_n200/streams"
  [temporal_pressure_step_locality_n200]="${PROJECT_ROOT}/generator/generated/runs/temporal_pressure_step_locality_n200/streams"
  [temporal_pressure_ramp_locality_n200]="${PROJECT_ROOT}/generator/generated/runs/temporal_pressure_ramp_locality_n200/streams"
  [temporal_pressure_burst_locality_n200]="${PROJECT_ROOT}/generator/generated/runs/temporal_pressure_burst_locality_n200/streams"
)

declare -A N_PARAMS=(
  [round_robin_n200]=200
  [milp_certificate_adversary_n200]=200
  [temporal_guard_step_schedule_n200]=200
  [temporal_guard_ramp_schedule_n200]=200
  [temporal_guard_burst_schedule_n200]=200
  [temporal_guard_oscillation_schedule_n200]=200
  [temporal_pressure_step_locality_n200]=200
  [temporal_pressure_ramp_locality_n200]=200
  [temporal_pressure_burst_locality_n200]=200
)

DATASETS=(${DATASETS:-temporal_guard_step_schedule_n200 temporal_guard_ramp_schedule_n200 temporal_guard_burst_schedule_n200 temporal_guard_oscillation_schedule_n200})

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
    printf 'experiment=temporal_controller_ablation\n'
    printf 'datasets=%s\n' "${DATASETS[*]}"
    printf 'm=%s\n' "${M}"
    printf 'n=200\n'
    printf 'epsilon_m=%s\n' "${EPSILON_M}"
    printf 'alpha=%s\n' "${ALPHA_REQ}"
    printf 'memory_kib=%s\n' "${MEM_KIB}"
    printf 'topk=%s\n' "${TOPK}"
    printf 'modes=probing residual_guarded_probing comfort_guided_probing pressure_gated_comfort_probing\n'
    printf 'executed_modes=%s\n' "${RUN_MODES[*]}"
    printf 'sufficient_observations=2\n'
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
    exit 1
  fi

  r_m="$(awk -v epsilon_m="${EPSILON_M}" -v n="${n_param}" 'BEGIN { printf "%.10g", epsilon_m / n }')"
  for mode in "${RUN_MODES[@]}"; do
    case "${mode}" in
      probing)
        guard_window=2
        guard_decay=0
        symmetric_relaxation=off
        downward_probing=on
        probe_residual_guard=off
        probe_strategy=bracket
        probe_pressure_gate=0.5
        amb_adjust=off
        diff_mode=predictive
        ;;
      residual_guarded_probing)
        guard_window=2
        guard_decay=0
        symmetric_relaxation=off
        downward_probing=on
        probe_residual_guard=on
        probe_strategy=bracket
        probe_pressure_gate=0.5
        amb_adjust=off
        diff_mode=predictive
        ;;
      comfort_guided_probing)
        guard_window=2
        guard_decay=0
        symmetric_relaxation=off
        downward_probing=on
        probe_residual_guard=on
        probe_strategy=comfort
        probe_pressure_gate=0.5
        amb_adjust=off
        diff_mode=predictive
        ;;
      pressure_gated_comfort_probing)
        guard_window=2
        guard_decay=0
        symmetric_relaxation=off
        downward_probing=on
        probe_residual_guard=on
        probe_strategy=pressure
        probe_pressure_gate=0.5
        amb_adjust=off
        diff_mode=predictive
        ;;
      *)
        echo "unknown temporal mode: ${mode}" >&2
        echo "known modes: probing residual_guarded_probing comfort_guided_probing pressure_gated_comfort_probing" >&2
        exit 1
        ;;
    esac
    method="oracle,ss[policy=difficulty alpha-req=${ALPHA_REQ} r-m=${r_m} res-guard-window=${guard_window} res-guard-decay=${guard_decay} symmetric-relaxation=${symmetric_relaxation} downward-probing=${downward_probing} probe-residual-guard=${probe_residual_guard} probe-strategy=${probe_strategy} probe-pressure-gate=${probe_pressure_gate} amb-adjust=${amb_adjust} diff-mode=${diff_mode} ss-eps=${SS_EPS}]"
    csv_out="${CSV_DIR}/${dataset}_${mode}.csv"

    echo "running temporal ablation ${dataset}: n=${n_param}, epsilon_m=${EPSILON_M}, alpha=${ALPHA_REQ}, mode=${mode}, probe_strategy=${probe_strategy}, pressure_gate=${probe_pressure_gate}, residual_guard=${probe_residual_guard}"
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

if [[ "${DRY_RUN}" == "1" ]]; then
  exit 0
fi

python3 experiments/calibration/summarize_temporal_grid.py --root "${CSV_DIR}" > "${OUT_DIR}/summary.csv"
echo "summary: ${OUT_DIR}/summary.csv"
python3 experiments/calibration/plot_temporal_grid.py \
  --summary "${OUT_DIR}/summary.csv" \
  --csv-root "${CSV_DIR}" \
  --out "${OUT_DIR}/plots"
