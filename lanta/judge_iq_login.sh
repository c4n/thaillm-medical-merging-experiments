#!/bin/bash -l
# Run post-hoc med-IQ judging from a LANTA login node with internet access.
set -euo pipefail
umask 0002

: "${OPENROUTER_API_KEY:?Export OPENROUTER_API_KEY before judging}"

WORK_ROOT="${WORK_ROOT:-/project/lt200394-thllmV/can/fix_tooluse_x}"
SIF="${SIF:-${WORK_ROOT}/apptainer/thaillm-merge.sif}"
CONCURRENCY="${CONCURRENCY:-4}"
TAGS=("$@")
if [ "${#TAGS[@]}" -eq 0 ]; then
  TAGS=(0p1 0p2 0p3 0p4 0p5)
fi

cd "${WORK_ROOT}"
module load Apptainer/1.1.6
test -f "${SIF}" || { echo "FATAL: missing ${SIF}" >&2; exit 1; }
export APPTAINERENV_OPENROUTER_API_KEY="${OPENROUTER_API_KEY}"

for tag in "${TAGS[@]}"; do
  if [[ "${tag}" == 0p* ]]; then
    run_name="task_a${tag}"
  else
    run_name="${tag}"
  fi
  search_root="${WORK_ROOT}/results/${run_name}/vf/evals/med_iq--${run_name}"
  input=$(
    find "${search_root}" -type f -name results.jsonl -printf '%T@ %p\n' 2>/dev/null |
      sort -nr | head -n 1 | cut -d' ' -f2-
  )
  if [ -z "${input}" ]; then
    echo "FATAL: no med-IQ results.jsonl found for ${run_name}" >&2
    exit 1
  fi

  output="${WORK_ROOT}/results/${run_name}/iq_judged.jsonl"
  echo "== judging ${run_name}"
  echo "   input:  ${input}"
  echo "   output: ${output}"
  apptainer exec \
    --env PYTHONPATH="${WORK_ROOT}/environments/med_iq" \
    -B "${WORK_ROOT}:${WORK_ROOT}" -B /project:/project \
    "${SIF}" /opt/eval-env/bin/python \
    "${WORK_ROOT}/scripts/judge_med_iq_results.py" \
    --input "${input}" \
    --output "${output}" \
    --concurrency "${CONCURRENCY}"
done
