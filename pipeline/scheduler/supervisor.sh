#!/usr/bin/env bash
# Machine-local paths. Sourced relative to this script, so a clone anywhere works.
_P=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); _P=${_P%/scheduler}
. "$_P/reap.env.example"; [ -f "$_P/reap.env" ] && . "$_P/reap.env"
# Keep one worker alive per GPU pair for as long as the queue has work.
#
# The workers dispatch dynamically on their own: each polls its OWN pair every
# 15s and pops the next job the moment that pair goes idle. What a worker cannot
# do is notice its own death -- OOM, a stray signal, a VM preemption that takes
# the tmux server -- leaving its pair idle with jobs queued. That is this loop.
#
# It also covers the "pair not joined yet" case, which is the same condition: a
# pair with queued work and no worker gets one. That is why the old cutover.sh /
# join01.sh waiters are gone -- they were this rule with extra steps.
#
# It only ever STARTS processes. It never kills an eval, and a worker that exits 3
# ("pair held by another worker") is left alone: that is the pair lock doing its
# job, e.g. a speed benchmark already owning that pair.
#
# QUEUE is the live queue file and is PASSED DOWN to every worker it starts --
# a supervisor on one queue that spawns workers reading another silently runs the
# wrong sweep.
set -u
cd $_P/scheduler
QUEUE="${QUEUE:-jobs2.txt}"
# Passed down so a supervisor started for the mix sweep cannot spawn workers that
# silently run the per-task launcher instead.
LAUNCH="${LAUNCH:-./launch_cell.sh}"
VERIFY="${VERIFY:-./verify_cell.sh}"
RESULTS="${RESULTS:-results}"
SWEEP="${SWEEP:-sweep2}"
POLL=60
PAIRS=(${GPU_PAIRS:-0,1 2,3})

# grep -c prints 0 AND exits 1 on no match, so `|| echo 0` yields TWO lines and
# every integer test downstream dies with "integer expression expected" -- which
# reads as "queue empty" and retires the sweep early. Let grep's own count stand.
queue_left() { local n; n=$(grep -cvE '^[[:space:]]*(#|$)' "$QUEUE" 2>/dev/null; true); echo "${n:-0}"; }

echo "$(date -Is) supervisor up (queue=$QUEUE, poll ${POLL}s, $(queue_left) jobs)"
while :; do
  n=$(queue_left)
  if [ "$n" -eq 0 ]; then
    # Workers finish the cell in hand after the queue drains; wait them out.
    if ! pgrep -f '[w]orker\.sh [0-9],[0-9]' >/dev/null 2>&1; then
      echo "$(date -Is) queue drained and no workers running -- supervisor exiting"; break
    fi
  else
    for pair in "${PAIRS[@]}"; do
      # Anchored so it matches neither this supervisor nor a grep of itself.
      if ! pgrep -f "[w]orker\.sh ${pair}\$" >/dev/null 2>&1; then
        win="w${pair//,/}"
        echo "$(date -Is) START worker $pair ($n jobs queued)"
        tmux kill-window -t "moe:$win" 2>/dev/null
        tmux new-window -t moe -n "$win" -d \
          "bash -c 'QUEUE=$QUEUE LAUNCH=$LAUNCH VERIFY=$VERIFY RESULTS=$RESULTS SWEEP=$SWEEP CAMPAIGN=${CAMPAIGN:-} $_P/scheduler/worker.sh $pair 2>&1 | tee -a $REAP_STATE/logs/worker_${pair//,/}.log; exec bash'"
        sleep 10
      fi
    done
  fi
  sleep "$POLL"
done
