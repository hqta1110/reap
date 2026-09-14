#!/usr/bin/env bash
# Machine-local paths. Sourced relative to this script, so a clone anywhere works.
_P=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); _P=${_P%/scheduler}
. "$_P/reap.env.example"; [ -f "$_P/reap.env" ] && . "$_P/reap.env"
# One REAP cell = one pruned checkpoint, one batch size, one or more tasks.
# Usage: ARM=reap25_math256 BS=64 TASKS="gsm8k ..." launch_reap.sh <mkey> <gpus>
#
# Drop-in for sweep2/launch_cell.sh (same argv + PRINT_MODEL_DIR contract) so
# sweep2/worker.sh and supervisor.sh drive this sweep unmodified.
#
# REAP needs NO plugin and NO env var: with VLLM_PLUGINS set you are measuring
# REAP+SERE and reporting it as REAP. The environment is scrubbed, not trusted.
set -u
# start.sh guards the sweep with flock on fd 9, and a bash redirection is not
# close-on-exec: every descendant inherits the lock. An orphaned engine or
# prune would then hold it after run.sh died, and cron could never restart
# the sweep. Drop the inherited fd -- only run.sh should hold the lock.
# Braces, not `exec 9>&- 2>/dev/null`: that form redirects THIS SCRIPT'S stderr
# to /dev/null for the rest of its life, so a cell that dies before the engine
# starts writes a ZERO-BYTE log and the failure is unattributable. Only the
# fd-close's own "Bad file descriptor" needs suppressing.
{ exec 9>&-; } 2>/dev/null || true
MKEY="$1"; GPUS="$2"; shift 2
# worker.sh exports ARM/TASKS/BS but not the campaign, so re-read campaign.env
# here: DS is what picks the calibration corpus out of the artifacts tree, and
# guessing it from the arm name is exactly the silent mislabel to avoid.
CAMPAIGN="${CAMPAIGN:?set CAMPAIGN=$REAP_STATE/<corpus>}"
. "$CAMPAIGN/campaign.env"
ARM="${ARM:-reap25_$SUFFIX}"
TASKS="${TASKS:-gsm8k}"
REPO=$EVAL_REPO
OUT=$CAMPAIGN/results
ART=$REAP_ROOT/artifacts
# OFFLINE_DISABLE_AR_FUSION=1 guards the flashinfer allreduce-workspace race
# between two engines on disjoint GPU pairs -- this sweep's exact shape. vLLM
# 0.18.1 stacks only; NOT gemma4 (0.29), whose numbers were collected without it.
EXTRA_ENV=(OFFLINE_DISABLE_AR_FUSION=1)

case "$ARM" in
  reap25*) RATIO=0.25 ;;   # bare `reap25` is the codealpaca arm
  reap50*) RATIO=0.50 ;;
  *) echo "unknown arm $ARM (expected reap25_*/reap50_*)" >&2; exit 2 ;;
esac

case "$MKEY" in
  qwen3-30b)     V=$VENV_QWEN
                 NAME='Qwen/Qwen3-30B-A3B'        ; MDIR=Qwen3-30B-A3B ;;
  qwen36-35b)    V=$VENV_QWEN
                 NAME='Qwen/Qwen3.6-35B-A3B'      ; MDIR=Qwen3.6-35B-A3B ;;
  glm-4.7-flash) V=$VENV_GLM
                 NAME='zai-org/GLM-4.7-Flash'     ; MDIR=GLM-4.7-Flash ;;
  gemma4)        V=$VENV_GEMMA4
                 NAME='google/gemma-4-26B-A4B-it' ; MDIR=gemma-4-26B-A4B-it
                 EXTRA_ENV=() ;;
  *) echo "unknown model key $MKEY" >&2; exit 2 ;;
esac

# --out-model-name files both ratios under the BASE model name, so they land in
# one tree instead of one directory per checkpoint path. worker.sh needs that
# directory to verify the cell it just ran; mirrors _safe_name() in
# tools/run_offline_pipeline.py.
if [ -n "${PRINT_MODEL_DIR:-}" ]; then
  printf '%s\n' "$NAME" | sed -e 's#[/ ]#_#g' \
                              -e 's#[^A-Za-z0-9._+-][^A-Za-z0-9._+-]*#_#g' \
                              -e 's#^[._]*##' -e 's#[._]*$##'
  exit 0
fi

MODEL="$ART/$MDIR/$DS/pruned_models/reap-renorm_true-seed_42-$RATIO"
[ -f "$MODEL/config.json" ] || { echo "no pruned checkpoint at $MODEL" >&2; exit 2; }

CLEAN_LD=$(printf '%s' "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v '^$' | grep -v gib | paste -sd:)

cd "$REPO" || exit 2
# -u VLLM_PLUGINS / -u PYTHONPATH: an ambient SERE plugin inherited from the
# shell would silently turn every one of these cells into REAP+SERE.
exec env -u NCCL_TUNER_CONFIG_PATH -u SERE_ALLOW_TORCH_REROUTE \
         -u VLLM_PLUGINS -u PYTHONPATH -u SERE_SIMILARITY_PT \
  CUDA_VISIBLE_DEVICES="$GPUS" \
  NCCL_NET=Socket NCCL_IB_DISABLE=1 LD_LIBRARY_PATH="$CLEAN_LD" \
  VLLM_DISABLE_COMPILE_CACHE=1 \
  EXPERT_SKIP_MODE=off \
  "${EXTRA_ENV[@]}" \
  "$V" tools/run_offline_pipeline.py \
    --method "$ARM" \
    --model "$MODEL" \
    --out-model-name "$NAME" \
    --out-root "$OUT" \
    --task $TASKS \
    --batch-sizes "${BS:-64}" \
    --tensor-parallel-size 2 \
    --max-model-len auto \
    --gpu-memory-utilization 0.92 \
    --score-jobs 3 \
    "$@"
