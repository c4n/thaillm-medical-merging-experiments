#!/bin/bash -l
# Judge strict official-protocol IQ outputs from a networked LANTA login node.
set -euo pipefail
umask 0002

: "${OPENROUTER_API_KEY:?Export OPENROUTER_API_KEY before judging}"

WORK_ROOT="${WORK_ROOT:-/project/lt200394-thllmV/can/fix_tooluse_x}"
RESULT_ROOT="${OFFICIAL_RESULT_ROOT:-${WORK_ROOT}/results/official_replication_7377263}"
SIF="${SIF:-${WORK_ROOT}/apptainer/thaillm-merge.sif}"
CONCURRENCY="${CONCURRENCY:-4}"
ALIASES=("$@")
if [ "${#ALIASES[@]}" -eq 0 ]; then
  ALIASES=(merged medapp)
fi

resolve_run_name() {
  case "$1" in
    merged) RUN_NAME="medapp_tool_linear_t0p3__official_r3" ;;
    medapp) RUN_NAME="baseline_medapp__official_r3" ;;
    tooluse) RUN_NAME="baseline_tooluse__official_r3" ;;
    sft_iq) RUN_NAME="baseline_sft_iq__official_r3" ;;
    typhoon) RUN_NAME="baseline_typhoon__official_r3" ;;
    *) RUN_NAME="$1" ;;
  esac
}

cd "${WORK_ROOT}"
module load Apptainer/1.1.6
test -f "${SIF}" || {
  echo "FATAL: missing ${SIF}" >&2
  exit 1
}
export APPTAINERENV_OPENROUTER_API_KEY="${OPENROUTER_API_KEY}"

for alias in "${ALIASES[@]}"; do
  resolve_run_name "${alias}"
  search_root="${RESULT_ROOT}/${RUN_NAME}/vf/evals/med_iq--${RUN_NAME}"
  input=$(
    find "${search_root}" -type f -name results.jsonl -printf '%T@ %p\n' 2>/dev/null |
      sort -nr | head -n 1 | cut -d' ' -f2-
  )
  if [ -z "${input}" ]; then
    echo "FATAL: no official med-IQ results.jsonl found for ${RUN_NAME}" >&2
    exit 1
  fi

  raw_uuid=$(basename "$(dirname "${input}")")
  output="${RESULT_ROOT}/${RUN_NAME}/iq_judged_official-${raw_uuid}.jsonl"
  echo "== judging official IQ: ${RUN_NAME}"
  echo "   input:  ${input}"
  echo "   output: ${output}"
  apptainer exec \
    --env PYTHONPATH="${WORK_ROOT}/environments/official_med_iq" \
    -B "${WORK_ROOT}:${WORK_ROOT}" -B /project:/project \
    "${SIF}" /opt/eval-env/bin/python \
    "${WORK_ROOT}/scripts/judge_med_iq_official_results.py" \
    --input "${input}" \
    --output "${output}" \
    --expected 600 \
    --concurrency "${CONCURRENCY}"
done
