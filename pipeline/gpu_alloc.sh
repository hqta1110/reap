#!/usr/bin/env bash
# Machine-local paths. Sourced relative to this script, so a clone anywhere works.
_P=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); _P=${_P%/scheduler}
. "$_P/reap.env.example"; [ -f "$_P/reap.env" ] && . "$_P/reap.env"
# Run a command on the first GPU pair that is both UNLOCKED and IDLE, waiting
# until one becomes so. Exports GPUS (e.g. "2,3") to the command.
#
#   gpu_alloc.sh ./prune_model.sh qwen3-30b qwen330 Qwen3-30B-A3B
#
# Why this exists: prunes used to be pinned to pair 0,1 and started
# unconditionally. Two things went wrong with that. Pair 2,3 sat idle through
# every first prune of a model, and -- worse -- when a CO-TENANT held the cards,
# every prune charged in, OOMed in nine seconds, and run.sh wrote off the whole
# model. Cron then repeated that every ten minutes. Waiting is the correct
# response to a busy GPU; failing is not.
#
# "Idle" is worker.sh's definition, deliberately: no compute apps on either GPU
# and under 4 GB resident. Anything looser races a co-tenant mid-teardown.
#
# The pair lock is the SAME file worker.sh takes, so a prune and an eval cell can
# never double-book a pair regardless of which got there first.
set -u
POLL="${GPU_POLL:-60}"
HEARTBEAT="${GPU_HEARTBEAT:-900}"   # log "still waiting" at most this often
PAIRS=(${GPU_PAIRS:-0,1 2,3})
[ $# -gt 0 ] || { echo "usage: gpu_alloc.sh <cmd> [args...]" >&2; exit 2; }

# Reserved exits for "this pair is unavailable". Chosen high so a real command's
# status (prune.py exits 1/2) can never be mistaken for one.
BUSY_LOCK=75; BUSY_GPU=76

pair_idle() {   # pair_idle <gpus>
  local apps used
  apps=$(nvidia-smi -i "$1" --query-compute-apps=pid --format=csv,noheader 2>/dev/null)
  [ -n "$apps" ] && return 1
  used=$(nvidia-smi -i "$1" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
         | awk '{s+=$1} END{print s+0}')
  [ "${used:-999999}" -lt 4000 ]
}

waited=0; announced=0
while :; do
  for pair in "${PAIRS[@]}"; do
    (
      # flock is released when this subshell exits, i.e. when the command ends.
      exec 8>"$REAP_STATE/locks/pair_g${pair//,/}.lock"
      flock -n 8 || exit $BUSY_LOCK
      pair_idle "$pair" || exit $BUSY_GPU
      echo "--- $(date -Is) gpu_alloc: acquired pair $pair for: $*"
      GPUS="$pair" CUDA_VISIBLE_DEVICES="$pair" exec "$@"
    )
    rc=$?
    case $rc in
      $BUSY_LOCK|$BUSY_GPU) ;;        # try the next pair
      *) exit $rc ;;                  # the command ran; its status is ours
    esac
  done
  if [ $((waited - announced)) -ge "$HEARTBEAT" ] || [ "$waited" = 0 ]; then
    echo "--- $(date -Is) gpu_alloc: no free pair (waited ${waited}s), holders:"
    nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader 2>/dev/null \
      | sed 's/^/      /'
    announced=$waited
  fi
  sleep "$POLL"; waited=$((waited + POLL))
done
