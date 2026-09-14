#!/usr/bin/env bash
# Machine-local paths. Sourced relative to this script, so a clone anywhere works.
_P=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); _P=${_P%/scheduler}
. "$_P/reap.env.example"; [ -f "$_P/reap.env" ] && . "$_P/reap.env"
# Liveness watchdog for the REAP sweep. Supervisor.sh already restarts a worker
# that DIED; what nothing covers is a worker that is alive holding a cell that
# stopped making progress -- a wedged NCCL collective, a stuck engine, a scorer
# blocked on a dead subprocess. Those hold their GPU pair forever.
#
# Signal: the cell's own log stops growing while its pair still has compute
# processes on it. vLLM writes progress continuously, so a log that has not moved
# in STALL_MIN minutes with the GPUs busy is wedged, not merely slow.
#
# Action: kill the compute processes on that pair. The launcher exits nonzero,
# verify_reap.sh fails the cell, worker.sh moves on, run.sh requeues it once.
# It never kills a healthy cell and it never touches the queue.
set -u
SWEEP_LOGS=$REAP_STATE/logs
STALL_MIN="${STALL_MIN:-45}"
POLL="${POLL:-300}"
PRUNE_MAX_MIN="${PRUNE_MAX_MIN:-45}"
PAIRS=(${GPU_PAIRS:-0,1 2,3})
echo "$(date -Is) watchdog up (cell stall=${STALL_MIN}m, prune ceiling=${PRUNE_MAX_MIN}m, poll=${POLL}s)"

while :; do
  for pair in "${PAIRS[@]}"; do
    WL="$SWEEP_LOGS/worker_${pair//,/}.log"
    [ -f "$WL" ] || continue
    # The cell currently in hand: last START with no DONE after it.
    last=$(grep -nE '(START|DONE) ' "$WL" | tail -1)
    case "$last" in *" START "*) ;; *) continue ;; esac
    # [w01] <ts> START <mkey> <arm> B=<bs> [<tasks>] on <gpus>
    read -r _ _ _ MKEY ARM BSF _ <<<"$last"
    BS="${BSF#B=}"
    TASKS=$(sed 's/.*\[\(.*\)\] on .*/\1/' <<<"$last")
    TTAG=$(echo "$TASKS" | tr -d " " | sed "s/gpqa_diamond/gpqa/;s/mmlu_pro/mmlu/")
    CL="$SWEEP_LOGS/${MKEY}_${ARM}_b${BS}_${TTAG}.log"
    [ -f "$CL" ] || continue

    pids=$(nvidia-smi -i "$pair" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')
    [ -n "$pids" ] || continue          # pair idle: nothing to be wedged
    age=$(( ( $(date +%s) - $(stat -c %Y "$CL") ) / 60 ))
    [ "$age" -lt "$STALL_MIN" ] && continue

    echo "$(date -Is) STALL pair=$pair $MKEY $ARM B=$BS [$TASKS] -- log idle ${age}m, killing $pids"
    for p in $pids; do kill -TERM "$p" 2>/dev/null; done
    sleep 30
    for p in $pids; do kill -KILL "$p" 2>/dev/null; done
    # Let the pair actually release before the next probe, or the same kill fires
    # again against a corpse while the worker is still tearing down.
    sleep 120
  done
  # A wedged PRUNE is invisible to the check above: it owns no cell log, and its
  # own log is block-buffered (a healthy 20-minute prune can write nothing at all
  # until it exits), so neither log mtime nor silence says anything. What is known
  # is the ceiling: the slowest measured prune here is ~20 min including the
  # checkpoint write. Past PRUNE_MAX_MIN it is wedged, and killing it lets run.sh
  # fail the model and move on rather than holding two GPUs until morning.
  while read -r pid etime; do
    [ -n "${pid:-}" ] || continue
    mins=$(awk -F: '{ if (NF==3) print $1*60+$2; else if (NF==2) print $1; else print 0 }' <<<"${etime%%.*}")
    # etime is [[DD-]HH:]MM:SS -- a day-prefixed one is long past the ceiling.
    case "$etime" in *-*) mins=$((PRUNE_MAX_MIN+1)) ;; esac
    if [ "${mins:-0}" -gt "$PRUNE_MAX_MIN" ]; then
      echo "$(date -Is) STALL prune pid=$pid running ${mins}m (> ${PRUNE_MAX_MIN}m) -- killing"
      kill -TERM "$pid" 2>/dev/null; sleep 30; kill -KILL "$pid" 2>/dev/null
    fi
  done < <(pgrep -f 'reap/src/reap/prune\.py' | xargs -r ps -o pid=,etime= -p 2>/dev/null)

  sleep "$POLL"
done
