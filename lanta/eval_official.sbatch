#!/bin/bash -l
# Isolated replica of the published ThaiLLM MedApp IQ/ToolUse protocol.
#
# This intentionally differs from eval_array.sbatch:
#   - exactly three rollouts per example
#   - strict med-IQ JSON scoring
#   - no max-token or temperature override
#   - a separate result root and unique run names
#   - ToolUse requests 5192 examples, which Verifiers clamps to the 5122-row
#     test split (15366 generated rollouts)
#
# Submit through lanta/submit_official_eval.sh.
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=16
#SBATCH -t 12:00:00
#SBATCH -A lt200394
#SBATCH -J thaillm-official
#SBATCH -o logs/%x-%j.out

set -euo pipefail
umask 0002

UPSTREAM_COMMIT="73772633663dfe02eff558a85eacbac9f617d329"
WORK_ROOT="${WORK_ROOT:-${SLURM_SUBMIT_DIR}}"
CACHE_ROOT="${CACHE_ROOT:-${WORK_ROOT}/.cache/huggingface}"
RESULT_ROOT="${OFFICIAL_RESULT_ROOT:-${WORK_ROOT}/results/official_replication_7377263}"
SIF="${SIF:-${WORK_ROOT}/apptainer/thaillm-merge.sif}"
MODEL="${MODEL_OVERRIDE:?Set MODEL_OVERRIDE to the full model directory}"
TOKENIZER="${TOKENIZER_OVERRIDE:-${MODEL}}"
RUN_NAME="${RUN_NAME_OVERRIDE:?Set RUN_NAME_OVERRIDE to a unique official run name}"
EVAL_STAGE="${EVAL_STAGE:?Set EVAL_STAGE to iq or tooluse}"
GENERATION_CONFIG="${GENERATION_CONFIG_OVERRIDE:-${WORK_ROOT}/configs/eval/medapp_published}"
RESUME_EVAL="${RESUME_EVAL:-0}"

ROLLOUTS=3
IQ_REQUESTED_N=200
IQ_ACTUAL_N=200
TOOLUSE_REQUESTED_N=5192
TOOLUSE_ACTUAL_N=5122
EVAL_PYTHONPATH="${WORK_ROOT}/environments/official_med_iq:${WORK_ROOT}/environments/official_med_tooluse"

case "${EVAL_STAGE}" in
  iq)
    REQUESTED_N="${IQ_REQUESTED_N}"
    EXPECTED_ROWS="$((IQ_ACTUAL_N * ROLLOUTS))"
    ENVIRONMENT_ID="med_iq"
    ;;
  tooluse)
    REQUESTED_N="${TOOLUSE_REQUESTED_N}"
    EXPECTED_ROWS="$((TOOLUSE_ACTUAL_N * ROLLOUTS))"
    ENVIRONMENT_ID="med_app_env"
    ;;
  *)
    echo "FATAL: EVAL_STAGE must be iq or tooluse" >&2
    exit 2
    ;;
esac

VF_RESUME_ARGS=()
if [ "${RESUME_EVAL}" = 1 ]; then
  VF_RESUME_ARGS=(--resume)
elif [ "${RESUME_EVAL}" != 0 ]; then
  echo "FATAL: RESUME_EVAL must be 0 or 1" >&2
  exit 2
fi

JOB_PORT_SEED="${SLURM_JOB_ID:-$$}"
PORT=$((20000 + JOB_PORT_SEED % 20000))
RUN_ROOT="${RESULT_ROOT}/${RUN_NAME}"

mkdir -p "${CACHE_ROOT}" "${RUN_ROOT}" "${WORK_ROOT}/logs"
export HF_HOME="${CACHE_ROOT}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
module load Apptainer/1.1.6
test -f "${SIF}" || {
  echo "FATAL: missing ${SIF}; run apptainer/build_lanta.sh first" >&2
  exit 1
}
test -s "${MODEL}/config.json" || {
  echo "FATAL: model is missing config.json: ${MODEL}" >&2
  exit 1
}
test -s "${GENERATION_CONFIG}/generation_config.json" || {
  echo "FATAL: generation config is missing: ${GENERATION_CONFIG}/generation_config.json" >&2
  exit 1
}
cd "${WORK_ROOT}"

echo "== official upstream commit: ${UPSTREAM_COMMIT}"
echo "== model: ${MODEL}"
echo "== tokenizer: ${TOKENIZER}"
echo "== generation config: ${GENERATION_CONFIG}/generation_config.json"
echo "== run: ${RUN_NAME}; stage: ${EVAL_STAGE}"
echo "== requested examples: ${REQUESTED_N}; rollouts/example: ${ROLLOUTS}"
echo "== expected saved rows: ${EXPECTED_ROWS}"
echo "== max tokens: unset (published protocol)"
echo "== temperature CLI override: unset (generation config applies)"
echo "== result root: ${RESULT_ROOT}"
echo "== resume evaluation: ${RESUME_EVAL}"
echo "== vLLM port: ${PORT}"

apptainer exec \
  --env PYTHONPATH="${EVAL_PYTHONPATH}" \
  -B "${WORK_ROOT}:${WORK_ROOT}" -B /project:/project \
  "${SIF}" /opt/eval-env/bin/python -c \
  'import med_iq, med_app_env
paths = (med_iq.__file__, med_app_env.__file__)
print("Official evaluator modules:", paths)
assert all("/official_med_" in path for path in paths), paths'

apptainer exec \
  --env HF_HOME="${CACHE_ROOT}" \
  --env HF_HUB_OFFLINE=1 \
  --env HF_DATASETS_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  -B "${WORK_ROOT}:${WORK_ROOT}" -B /project:/project \
  "${SIF}" /opt/eval-env/bin/python \
  "${WORK_ROOT}/scripts/validate_official_dataset_sizes.py" \
  --output "${RUN_ROOT}/protocol_manifest-${EVAL_STAGE}-${SLURM_JOB_ID:-local}.json" \
  --model "${MODEL}" \
  --tokenizer "${TOKENIZER}" \
  --generation-config "${GENERATION_CONFIG}/generation_config.json" \
  --stage "${EVAL_STAGE}" \
  --run-name "${RUN_NAME}" \
  --requested-examples "${REQUESTED_N}" \
  --actual-examples "$((EXPECTED_ROWS / ROLLOUTS))" \
  --rollouts "${ROLLOUTS}" \
  --image "${SIF}"

VLLM_LOG="${RUN_ROOT}/vllm-${EVAL_STAGE}-${SLURM_JOB_ID:-local}.log"
apptainer exec --nv \
  --env HF_HOME="${CACHE_ROOT}" \
  --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  -B "${WORK_ROOT}:${WORK_ROOT}" -B /project:/project \
  "${SIF}" vllm serve "${MODEL}" \
  --tokenizer "${TOKENIZER}" \
  --generation-config "${GENERATION_CONFIG}" \
  --served-model-name "${RUN_NAME}" \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.90 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --port "${PORT}" \
  >"${VLLM_LOG}" 2>&1 &
SERVER_PID=$!
trap 'kill "${SERVER_PID}" 2>/dev/null || true' EXIT

apptainer exec -B "${WORK_ROOT}:${WORK_ROOT}" "${SIF}" \
  python3 - "${PORT}" "${RUN_NAME}" <<'PY'
import json
import sys
import time
import urllib.request

health_url = "http://127.0.0.1:{}/health".format(sys.argv[1])
for _ in range(180):
    try:
        with urllib.request.urlopen(health_url, timeout=2) as response:
            if response.status == 200:
                break
    except Exception:
        time.sleep(2)
else:
    raise SystemExit("vLLM did not become healthy within six minutes")

models_url = "http://127.0.0.1:{}/v1/models".format(sys.argv[1])
with urllib.request.urlopen(models_url, timeout=5) as response:
    model_ids = {item["id"] for item in json.load(response).get("data", [])}
expected = sys.argv[2]
if expected not in model_ids:
    raise SystemExit(
        "Wrong vLLM server on port {}: expected {!r}, found {!r}".format(
            sys.argv[1],
            expected,
            sorted(model_ids),
        )
    )
PY

OPENAI_BASE_URL="http://127.0.0.1:${PORT}/v1"
TRANSCRIPT="${RUN_ROOT}/${EVAL_STAGE}-${SLURM_JOB_ID:-local}.txt"
COMMON_ARGS=(
  --model "${RUN_NAME}"
  --api-base-url "${OPENAI_BASE_URL}"
  --api-key-var OPENAI_API_KEY
  -n "${REQUESTED_N}"
  -r "${ROLLOUTS}"
  "${VF_RESUME_ARGS[@]}"
  -o "${RUN_ROOT}/vf"
  --save-results
)

if [ "${EVAL_STAGE}" = iq ]; then
  apptainer exec \
    --env HF_HOME="${CACHE_ROOT}" \
    --env HF_HUB_OFFLINE=1 \
    --env HF_DATASETS_OFFLINE=1 \
    --env TRANSFORMERS_OFFLINE=1 \
    --env OPENAI_BASE_URL="${OPENAI_BASE_URL}" \
    --env OPENAI_API_KEY=local \
    --env MED_IQ_JUDGE_MODE=deferred \
    --env PYTHONPATH="${EVAL_PYTHONPATH}" \
    -B "${WORK_ROOT}:${WORK_ROOT}" -B /project:/project \
    "${SIF}" /opt/eval-env/bin/vf-eval "${ENVIRONMENT_ID}" \
    "${COMMON_ARGS[@]}" -C judge_response 2>&1 | tee "${TRANSCRIPT}"
else
  apptainer exec \
    --env HF_HOME="${CACHE_ROOT}" \
    --env HF_HUB_OFFLINE=1 \
    --env HF_DATASETS_OFFLINE=1 \
    --env TRANSFORMERS_OFFLINE=1 \
    --env OPENAI_BASE_URL="${OPENAI_BASE_URL}" \
    --env OPENAI_API_KEY=local \
    --env PYTHONPATH="${EVAL_PYTHONPATH}" \
    -B "${WORK_ROOT}:${WORK_ROOT}" -B /project:/project \
    "${SIF}" /opt/eval-env/bin/vf-eval "${ENVIRONMENT_ID}" \
    "${COMMON_ARGS[@]}" 2>&1 | tee "${TRANSCRIPT}"
fi

apptainer exec \
  -B "${WORK_ROOT}:${WORK_ROOT}" -B /project:/project \
  "${SIF}" /opt/eval-env/bin/python \
  "${WORK_ROOT}/scripts/validate_eval_results.py" \
  --root "${RUN_ROOT}/vf/evals/${ENVIRONMENT_ID}--${RUN_NAME}" \
  --expected "${EXPECTED_ROWS}" \
  --max-errors "${EXPECTED_ROWS}"

echo "COMPLETED official ${EVAL_STAGE}: ${RUN_NAME}"
