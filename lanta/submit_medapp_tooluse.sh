#!/bin/bash -l
# Submit MedApp/ToolUse interpolation merges or their evaluations.
set -euo pipefail
umask 0002

MODE="${1:?Usage: $0 merge-or-tooluse-or-iq}"
WORK_ROOT="${WORK_ROOT:-/project/lt200394-thllmV/can/fix_tooluse_x}"
LANTA_ACCOUNT="${LANTA_ACCOUNT:-lt200394}"
MODEL_ROOT="${MODEL_ROOT:-${WORK_ROOT}/models/merged}"
TOOL_WEIGHTS=(0p1 0p2 0p3)

cd "${WORK_ROOT}"
case "${MODE}" in
  merge)
    sbatch -A "${LANTA_ACCOUNT}" lanta/merge_medapp_tooluse_array.sbatch
    ;;
  tooluse|iq)
    for tag in "${TOOL_WEIGHTS[@]}"; do
      model="${MODEL_ROOT}/medapp_tool_linear_t${tag}"
      run="medapp_tool_linear_t${tag}"
      test -s "${model}/config.json" || {
        echo "FATAL: missing merged model: ${model}" >&2
        exit 1
      }
      if [ "${MODE}" = tooluse ]; then
        exports="EVAL_STAGE=tooluse,TOOLUSE_N=5122,ROLLOUTS=1,TOOLUSE_MAX_TOKENS=1024"
        time_limit="02:00:00"
      else
        exports="EVAL_STAGE=iq,IQ_N=200,ROLLOUTS=1,MED_IQ_JUDGE_MODE=deferred"
        time_limit="02:00:00"
      fi
      echo "== submitting ${MODE}: ${run}"
      sbatch -A "${LANTA_ACCOUNT}" --array=0-0 --time="${time_limit}" \
        --export="ALL,MODEL_OVERRIDE=${model},RUN_NAME_OVERRIDE=${run},${exports}" \
        lanta/eval_array.sbatch
    done
    ;;
  *)
    echo "Usage: $0 {merge|tooluse|iq}" >&2
    exit 2
    ;;
esac
