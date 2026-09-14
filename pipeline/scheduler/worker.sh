#!/usr/bin/env bash
# Machine-local paths. Sourced relative to this script, so a clone anywhere works.
_P=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); _P=${_P%/scheduler}
. "$_P/reap.env.example"; [ -f "$_P/reap.env" ] && . "$_P/reap.env"
# One worker per GPU pair. Pops the next job the moment its OWN pair is idle, so
# a 2h B=8 cell on one pair never blocks a 20min B=64 cell on the other.
# Usage: worker.sh <gpus>    e.g. worker.sh 0,1
set -u
cd $_P/scheduler
GPUS="$1"
TAG="w${GPUS//,/}"
QUEUE="${QUEUE:-jobs.txt}"
# The mix sweep reuses this worker with a different launcher/verifier, so both
# are indirected rather than forked. Defaults keep the per-task sweep unchanged.
LAUNCH="${LAUNCH:-./launch_cell.sh}"
VERIFY="${VERIFY:-./verify_cell.sh}"
SWEEP="${SWEEP:-sweep2}"
RESULTS="${RESULTS:-results}"
QLOCK=.jobs.lock
PUSHLOCK=.push.lock

# Per-pair lock held for the worker's whole life. Without it two workers each see
# the pair as idle while the other's engine is still starting, both dispatch onto
# it, and the second dies on "Free memory on device less than desired". Same lock
# file sere_speed_bench.sh and gemma4_worker.sh take, so those cannot collide
# with this sweep either.
PAIRLOCK="$REAP_STATE/locks/pair_g${GPUS//,/}.lock"
mkdir -p "$REAP_STATE/locks" 2>/dev/null || true
exec 8>"$PAIRLOCK"
if ! flock -n 8; then
  echo "[$TAG] pair $GPUS is held by another worker -- refusing to double-book"; exit 3
fi

push_rebased() {
  git config merge.keepours.driver true
  git fetch -q origin main 2>/dev/null || true
  if ! git rebase -q origin/main 2>/dev/null; then
    git rebase --abort 2>/dev/null || true
    echo "REBASE CONFLICT -- commit is local, not pushed"; return 1
  fi
  git push -q origin HEAD:main 2>&1 | tail -2
}

gpus_idle() {
  local apps used
  apps=$(nvidia-smi -i "$GPUS" --query-compute-apps=pid --format=csv,noheader 2>/dev/null)
  [ -n "$apps" ] && return 1
  used=$(nvidia-smi -i "$GPUS" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
         | awk '{s+=$1} END{print s+0}')
  [ "$used" -lt 4000 ]
}

# grep -vxF exits 1 when it prints NOTHING, which is exactly what happens when the
# job being removed is the last line in the queue. With `&& mv` that meant the final
# job was never consumed: both workers re-popped it forever and neither ever saw an
# empty queue to exit on. Let the rewrite stand on its own.
pop_job() { flock 9; local j; j=$(grep -vE '^[[:space:]]*(#|$)' "$QUEUE" | head -n1)
            [ -z "$j" ] && return 1
            { grep -vxF "$j" "$QUEUE" || true; } > "$QUEUE.tmp"
            mv "$QUEUE.tmp" "$QUEUE"
            printf '%s' "$j"; }

while :; do
  until gpus_idle; do sleep 15; done
  job=$(exec 9>"$QLOCK"; pop_job) || { echo "[$TAG] $(date -Is) queue empty, exiting"; break; }
  read -r MKEY ARM BS TASKS <<<"$job"
  TASKS="${TASKS:-gpqa_diamond mmlu_pro}"; TTAG=$(echo "$TASKS" | tr -d " " | sed "s/gpqa_diamond/gpqa/;s/mmlu_pro/mmlu/")
  LOG="logs/${MKEY}_${ARM}_b${BS}_${TTAG}.log"
  echo "[$TAG] $(date -Is) START $MKEY $ARM B=$BS [$TASKS] on $GPUS"
  ARM="$ARM" BS="$BS" TASKS="$TASKS" "$LAUNCH" "$MKEY" "$GPUS" > "$LOG" 2>&1
  rc=$?
  MDIR=$(PRINT_MODEL_DIR=1 "$LAUNCH" "$MKEY" "$GPUS" 2>/dev/null)
  # Take the dir THIS cell wrote, which the pipeline prints as
  #   [gen] batch_size=64 -> <dir>
  # rather than the newest matching glob. With both pairs running cells for the
  # same model+arm+batch, "newest" is the OTHER pair's dir: a cell that scored
  # perfectly gets verified against a sibling's empty directory and reported
  # FAIL, and the sibling's dir gets pushed twice. Glob only as a fallback, for
  # a launcher that printed nothing.
  DIR=$(sed -n 's/^\[gen\] batch_size=[0-9]* -> //p' "$LOG" 2>/dev/null | tail -1)
  [ -d "${DIR:-}" ] || \
    DIR=$(ls -d "$RESULTS"/"${MDIR:-__nomodel__}"/"$ARM"/*_b$(printf '%04d' "$BS") 2>/dev/null | tail -1)
  # Fold the plugin evidence into the meta BEFORE verify, so it is part of what
  # gets mirrored and pushed. Logs do not travel; results do.
  [ -n "${DIR:-}" ] && ./attach_evidence.py "$LOG" "$DIR" $TASKS 2>&1 | sed "s/^/[$TAG]   /"
  V=$("$VERIFY" "$ARM" "$LOG" "${DIR:-/nonexistent}" "$TASKS" 2>&1)
  echo "$V" | sed "s/^/[$TAG]   /"
  if echo "$V" | grep -q 'VERDICT=PASS'; then
    # Push only verified cells: a half-written or silently-not-SERE cell on the
    # remote is worse than no cell, because it looks like a result.
    ( flock -w 600 7 || exit 0
      cd "${RESULTS_REPO:?set RESULTS_REPO to the results git checkout}" \
        && python3 mirror.py >/dev/null 2>&1 \
        && git add -A \
        && git diff --cached --quiet \
        || { git commit -q -m "$SWEEP: $MKEY $ARM B=$BS ($TASKS)" \
             && push_rebased; }
      echo "[$TAG]   pushed $MKEY $ARM B=$BS" ) 7>"$PUSHLOCK"
    echo "$(date -Is) PASS $MKEY $ARM B=$BS rc=$rc $(echo "$V" | grep -o 'value=[0-9.]*' | tr '\n' ' ')" >> DONE.txt
  else
    echo "$(date -Is) FAIL $MKEY $ARM B=$BS rc=$rc $(echo "$V" | grep VERDICT)" >> DONE.txt
  fi
  echo "[$TAG] $(date -Is) DONE  $MKEY $ARM B=$BS rc=$rc"
  sleep 20   # let the engine release memory before the next idle probe
done
