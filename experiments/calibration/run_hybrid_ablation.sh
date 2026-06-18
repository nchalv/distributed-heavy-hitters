#!/usr/bin/env bash
set -euo pipefail

# Hybrid exact-approximate ablation after fixing the baseline controller
# configuration. The hybrid promoted set is Top_n union K_+ union K_?.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HH_BENCH="${PROJECT_ROOT}/evaluator/build/hh_bench"

OUT_DIR="${OUT_DIR:-experiments/calibration/hybrid_ablation}"
if [[ "${OUT_DIR}" != /* ]]; then
  OUT_DIR="${PROJECT_ROOT}/${OUT_DIR}"
fi
CSV_DIR="${OUT_DIR}/csv"

M="${M:-100}"
MEM_KIB="${MEM_KIB:-128}"
TOPK="${TOPK:-200}"

EPSILON_M="${EPSILON_M:-0.15}"
ALPHA_REQ="${ALPHA_REQ:-0.95}"
AMBIGUITY_BETA="${AMBIGUITY_BETA:-0.50}"
SS_EPS="${SS_EPS:-per-item}"

HEAD_POLICIES=(${HEAD_POLICIES:-candidate})
CLEAN_OUTPUT="${CLEAN_OUTPUT:-1}"
DRY_RUN="${DRY_RUN:-0}"

declare -A STREAMS=(
  [milp_certificate_adversary_n100]="${PROJECT_ROOT}/generator/generated/runs/calibration_synthetic_milp_certificate_adversary_n100/streams"
  [milp_certificate_adversary_n200]="${PROJECT_ROOT}/generator/generated/runs/calibration_synthetic_milp_certificate_adversary_n200/streams"
  [milp_certificate_adversary_n400]="${PROJECT_ROOT}/generator/generated/runs/calibration_synthetic_milp_certificate_adversary_n400/streams"
  [persistent_ambiguity_adversary_n100]="${PROJECT_ROOT}/generator/generated/runs/calibration_synthetic_persistent_ambiguity_adversary_n100/streams"
  [persistent_ambiguity_adversary_n200]="${PROJECT_ROOT}/generator/generated/runs/calibration_synthetic_persistent_ambiguity_adversary_n200/streams"
  [persistent_ambiguity_adversary_n400]="${PROJECT_ROOT}/generator/generated/runs/calibration_synthetic_persistent_ambiguity_adversary_n400/streams"
)

declare -A N_PARAMS=(
  [milp_certificate_adversary_n100]=100
  [milp_certificate_adversary_n200]=200
  [milp_certificate_adversary_n400]=400
  [persistent_ambiguity_adversary_n100]=100
  [persistent_ambiguity_adversary_n200]=200
  [persistent_ambiguity_adversary_n400]=400
)

DATASETS=(${DATASETS:-milp_certificate_adversary_n100 milp_certificate_adversary_n200 milp_certificate_adversary_n400 persistent_ambiguity_adversary_n100 persistent_ambiguity_adversary_n200 persistent_ambiguity_adversary_n400})

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
    printf 'experiment=hybrid_ablation\n'
    printf 'datasets=%s\n' "${DATASETS[*]}"
    printf 'head_policies=%s\n' "${HEAD_POLICIES[*]}"
    printf 'm=%s\n' "${M}"
    printf 'epsilon_m=%s\n' "${EPSILON_M}"
    printf 'alpha=%s\n' "${ALPHA_REQ}"
    printf 'ambiguity_beta=%s\n' "${AMBIGUITY_BETA}"
    printf 'temporal_controller=guarded_margin_comfort\n'
    printf 'tail_policy=difficulty\n'
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

  r_m="$(awk -v epsilon_m="${EPSILON_M}" -v n="${n_param}" 'BEGIN { printf "%.10g", epsilon_m / n }')"
  ss_method="ss[policy=difficulty alpha-req=${ALPHA_REQ} r-m=${r_m} res-guard-window=2 res-guard-decay=${AMBIGUITY_BETA} downward-probing=on probe-residual-guard=on probe-strategy=comfort probe-pressure-gate=0.5 amb-adjust=on diff-mode=predictive ss-eps=${SS_EPS}]"

  for head_policy in "${HEAD_POLICIES[@]}"; do
    case "${head_policy}" in
      candidate) ;;
      *)
        echo "unknown hybrid head policy: ${head_policy}; expected candidate" >&2
        exit 1
        ;;
    esac

    hybrid_method="hybrid[hyb-head=${head_policy} hyb-tail=difficulty alpha-req=${ALPHA_REQ} r-m=${r_m} res-guard-window=2 res-guard-decay=${AMBIGUITY_BETA} downward-probing=on probe-residual-guard=on probe-strategy=comfort probe-pressure-gate=0.5 amb-adjust=on diff-mode=predictive ss-eps=${SS_EPS}]"
    csv_out="${CSV_DIR}/${dataset}_${head_policy}.csv"

    echo "running hybrid ablation: dataset=${dataset}, n=${n_param}, head_policy=${head_policy}"
    command=(
      "${HH_BENCH}"
      "${stream}"
      "${M}"
      "${n_param}"
      "${MEM_KIB}"
      "oracle,${ss_method},${hybrid_method}"
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

python3 experiments/calibration/summarize_hybrid_ablation.py \
  --root "${CSV_DIR}" \
  > "${OUT_DIR}/summary.csv"

python3 experiments/calibration/plot_hybrid_ablation.py \
  --summary "${OUT_DIR}/summary.csv" \
  --out "${OUT_DIR}/plots"

echo "summary: ${OUT_DIR}/summary.csv"
echo "plots: ${OUT_DIR}/plots"
