#!/bin/bash -l
# Submit the isolated published-protocol replica.
#
# Usage:
#   bash lanta/submit_official_eval.sh iq [merged|medapp ...]
#   bash lanta/submit_official_eval.sh tooluse [merged|medapp ...]
#
# With no model aliases, both the selected 70/30 merge and MedApp control run.
set -euo pipefail
umask 0002

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 {iq|tooluse} [merged|medapp|tooluse|sft_iq|typhoon ...]" >&2
  exit 2
fi
STAGE="$1"
shift
case "${STAGE}" in
  iq)
    TIME_LIMIT="${TIME_LIMIT:-03:00:00}"
    ;;
  tooluse)
    TIME_LIMIT="${TIME_LIMIT:-12:00:00}"
    ;;
  *)
    echo "Usage: $0 {iq|tooluse} [merged|medapp|tooluse|sft_iq|typhoon ...]" >&2
    exit 2
    ;;
esac

WORK_ROOT="${WORK_ROOT:-/project/lt200394-thllmV/can/fix_tooluse_x}"
LANTA_ACCOUNT="${LANTA_ACCOUNT:-lt200394}"
CACHE_HUB="${WORK_ROOT}/.cache/huggingface/hub"
RESULT_ROOT="${OFFICIAL_RESULT_ROOT:-${WORK_ROOT}/results/official_replication_7377263}"
GENERATION_CONFIG_OVERRIDE="${GENERATION_CONFIG_OVERRIDE:-}"
RESUME_EVAL="${RESUME_EVAL:-0}"
ALIASES=("$@")
if [ "${#ALIASES[@]}" -eq 0 ]; then
  ALIASES=(merged medapp)
fi

snapshot_path() {
  local repo_slug="$1"
  local root="${CACHE_HUB}/models--${repo_slug//\//--}"
  local revision
  test -f "${root}/refs/main" || {
    echo "FATAL: ${repo_slug} is not cached" >&2
    return 1
  }
  revision=$(cat "${root}/refs/main")
  # Keep the logical /project path: Apptainer binds /project, while realpath
  # resolves it to LANTA's host-only /lustrefs/disk/project path.
  printf '%s\n' "${root}/snapshots/${revision}"
}

MEDAPP_MODEL=$(snapshot_path "ThaiLLM/ThaiLLM-8B-MedApp")
BASE_MODEL=""
IQ_MODEL=""
TOOLUSE_MODEL="${WORK_ROOT}/models/merged/tooluse-full"

resolve_alias() {
  local alias="$1"
  case "${alias}" in
    merged)
      MODEL="${WORK_ROOT}/models/merged/medapp_tool_linear_t0p3"
      RUN_NAME="medapp_tool_linear_t0p3__official_r3"
      GENERATION_CONFIG="${WORK_ROOT}/configs/eval/medapp_published"
      ;;
    medapp)
      MODEL="${MEDAPP_MODEL}"
      RUN_NAME="baseline_medapp__official_r3"
      GENERATION_CONFIG="${WORK_ROOT}/configs/eval/medapp_published"
      ;;
    tooluse)
      MODEL="${TOOLUSE_MODEL}"
      RUN_NAME="baseline_tooluse__official_r3"
      if [ -z "${BASE_MODEL}" ]; then
        BASE_MODEL=$(snapshot_path "typhoon-ai/typhoon-s-thaillm-8b-instruct-research-preview")
      fi
      GENERATION_CONFIG="${BASE_MODEL}"
      ;;
    sft_iq)
      if [ -z "${IQ_MODEL}" ]; then
        IQ_MODEL=$(snapshot_path "ThaiLLM/ThaiLLM-8B-SFT-IQ")
      fi
      MODEL="${IQ_MODEL}"
      RUN_NAME="baseline_sft_iq__official_r3"
      GENERATION_CONFIG="${IQ_MODEL}"
      ;;
    typhoon)
      if [ -z "${BASE_MODEL}" ]; then
        BASE_MODEL=$(snapshot_path "typhoon-ai/typhoon-s-thaillm-8b-instruct-research-preview")
      fi
      MODEL="${BASE_MODEL}"
      RUN_NAME="baseline_typhoon__official_r3"
      GENERATION_CONFIG="${BASE_MODEL}"
      ;;
    *)
      echo "FATAL: unknown model alias '${alias}'" >&2
      echo "Choose merged, medapp, tooluse, sft_iq, or typhoon." >&2
      return 2
      ;;
  esac
  if [ -n "${GENERATION_CONFIG_OVERRIDE}" ]; then
    GENERATION_CONFIG="${GENERATION_CONFIG_OVERRIDE}"
  fi
  test -s "${MODEL}/config.json" || {
    echo "FATAL: model is missing or incomplete: ${MODEL}" >&2
    return 1
  }
  test -s "${GENERATION_CONFIG}/generation_config.json" || {
    echo "FATAL: generation config is missing: ${GENERATION_CONFIG}/generation_config.json" >&2
    return 1
  }
}

cd "${WORK_ROOT}"
mkdir -p logs "${RESULT_ROOT}"
for alias in "${ALIASES[@]}"; do
  resolve_alias "${alias}"
  echo "== submitting official ${STAGE}: ${alias}"
  echo "   model: ${MODEL}"
  echo "   gen:   ${GENERATION_CONFIG}/generation_config.json"
  echo "   run:   ${RUN_NAME}"
  echo "   root:  ${RESULT_ROOT}"
  sbatch \
    -A "${LANTA_ACCOUNT}" \
    --time="${TIME_LIMIT}" \
    --export="WORK_ROOT=${WORK_ROOT},MODEL_OVERRIDE=${MODEL},TOKENIZER_OVERRIDE=${MODEL},GENERATION_CONFIG_OVERRIDE=${GENERATION_CONFIG},RUN_NAME_OVERRIDE=${RUN_NAME},EVAL_STAGE=${STAGE},OFFICIAL_RESULT_ROOT=${RESULT_ROOT},RESUME_EVAL=${RESUME_EVAL}" \
    lanta/eval_official.sbatch
done
