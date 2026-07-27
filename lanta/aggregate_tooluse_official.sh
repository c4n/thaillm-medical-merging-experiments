#!/bin/bash -l
# Aggregate official-protocol ToolUse outputs inside the Apptainer eval env.
set -euo pipefail
umask 0002

WORK_ROOT="${WORK_ROOT:-/project/lt200394-thllmV/can/fix_tooluse_x}"
RESULT_ROOT="${OFFICIAL_RESULT_ROOT:-${WORK_ROOT}/results/official_replication_7377263}"
SIF="${SIF:-${WORK_ROOT}/apptainer/thaillm-merge.sif}"
ALIASES=("$@")
if [ "${#ALIASES[@]}" -eq 0 ]; then
  ALIASES=(merged medapp)
fi

RUNS=()
for alias in "${ALIASES[@]}"; do
  case "${alias}" in
    merged) RUNS+=("medapp_tool_linear_t0p3__official_r3") ;;
    medapp) RUNS+=("baseline_medapp__official_r3") ;;
    tooluse) RUNS+=("baseline_tooluse__official_r3") ;;
    sft_iq) RUNS+=("baseline_sft_iq__official_r3") ;;
    typhoon) RUNS+=("baseline_typhoon__official_r3") ;;
    *) RUNS+=("${alias}") ;;
  esac
done

cd "${WORK_ROOT}"
module load Apptainer/1.1.6
test -f "${SIF}" || {
  echo "FATAL: missing ${SIF}" >&2
  exit 1
}
mkdir -p "${RESULT_ROOT}"

apptainer exec \
  -B "${WORK_ROOT}:${WORK_ROOT}" -B /project:/project \
  "${SIF}" /opt/eval-env/bin/python \
  "${WORK_ROOT}/scripts/aggregate_official_tooluse.py" \
  --results-root "${RESULT_ROOT}" \
  --runs "${RUNS[@]}" \
  --k 3 \
  --expected-samples 5122 \
  --output-json "${RESULT_ROOT}/tooluse_official_comparison.json" |
  tee "${RESULT_ROOT}/tooluse_official_comparison.txt"
