#!/usr/bin/env bash
# Machine-local paths. Sourced relative to this script, so a clone anywhere works.
_P=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); _P=${_P%/scheduler}
. "$_P/reap.env.example"; [ -f "$_P/reap.env" ] && . "$_P/reap.env"
# REAP end-to-end, calibrated on whatever corpus $CAMPAIGN/campaign.env names.
# Four models IN ORDER, both ratios, all nine accuracy tasks, TP=2, B=64.
#
# Shape, and why it is serial per model rather than a flat queue:
#   4 GPUs = one prune job (2 GPUs) OR two eval lanes (2x2). They cannot overlap,
#   so each model runs prune(0.25) -> prune(0.50) -> both arms evaluated in
#   parallel -> checkpoint weights deleted. ~305 GB never coexists on a 449 GB disk.
#
# Observations are ratio-independent: 0.25 builds the cache, 0.50 reuses it, so
# the second prune is minutes. Calibration itself is ~3-5 min (256 samples at
# L256) -- wall time is model load and checkpoint write, not calibration.
#
# RESTARTABLE. Every step is guarded by what is on disk, not by how far a previous
# invocation got: a model whose cells all have a summary.json is skipped entirely,
# a ratio whose checkpoint weights are present is not re-pruned, and the
# observation cache survives the weight deletion so a re-prune is minutes. Killing
# this script at any point and re-running start.sh resumes.
#
# Scheduling, verification and push are sweep2's worker.sh/supervisor.sh, driven
# through their LAUNCH/VERIFY/QUEUE/RESULTS indirection. Nothing is forked.
#
# CORPUS-AGNOSTIC. $CAMPAIGN is a directory holding campaign.env (DATASET/DS/
# SUFFIX/SWEEP) plus that campaign's queue, logs and results. Everything a
# corpus changes lives in that one file, so a third corpus is a new campaign.env
# and nothing else -- no forked copy of this script to drift out of sync.
set -u
CAMPAIGN="${CAMPAIGN:?set CAMPAIGN=$REAP_STATE/<corpus>}"
. "$CAMPAIGN/campaign.env"
cd "$CAMPAIGN"
Q=$CAMPAIGN/jobs.txt
LOGS=$CAMPAIGN/logs
ART=$REAP_ROOT/artifacts
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# Split 4+4 rather than one 8-task cell: two lanes, so an odd number of long
# jobs leaves one lane idle. Each extra job costs a ~2 min engine start and buys
# a packing unit. livebench_mixed stays its own cell -- it is the long one (581
# rows at a 12288-token cap) and a crash there must not cost the short tasks.
TASK_CELLS=("gsm8k math_hard math500 aime" "lcb humaneval_plus mmlu_pro_400 gpqa_diamond")
BS=64
# eval key : prune key : artifacts dir
MODELS=("qwen3-30b:qwen330:Qwen3-30B-A3B"
        "glm-4.7-flash:glm47:GLM-4.7-Flash"
        "qwen36-35b:qwen36:Qwen3.6-35B-A3B"
        "gemma4:gemma4:gemma-4-26B-A4B-it")

export QUEUE="$Q" LAUNCH="$HERE/launch_reap.sh" VERIFY="$HERE/verify_reap.sh" \
       RESULTS="$CAMPAIGN/results" SWEEP CAMPAIGN DATASET DS SUFFIX
mkdir -p "$LOGS" "$RESULTS"

tmux has-session -t moe 2>/dev/null || tmux new-session -d -s moe
# One watchdog for the whole sweep, not one per model. Restarted here if the
# window is gone, so a resumed run is never left without liveness cover.
WD=wd-$(basename "$CAMPAIGN")
tmux list-windows -t moe -F '#W' 2>/dev/null | grep -qx "$WD" || \
  tmux new-window -t moe -n "$WD" -d \
    "bash -c '$HERE/watchdog.sh 2>&1 | tee -a $LOGS/watchdog.log; exec bash'"

# The full job list for one model. livebench_mixed is split off the other eight:
# it is the long cell (581 rows at a 12288-token cap) and a crash there must not
# cost the eight short tasks.
full_jobs() {
  for ARM in "reap25_$SUFFIX" "reap50_$SUFFIX"; do
    for t in "${TASK_CELLS[@]}"; do echo "$1 $ARM $BS $t"; done
    echo "$1 $ARM $BS livebench_mixed"
  done
}

# Ground truth for "is this cell done": the summary.json files that exist.
# DONE.txt records the arm but not which tasks the cell was running, so it cannot
# answer this after a restart.
missing_jobs() {
  while read -r m a b t; do
    [ -n "${m:-}" ] || continue
    md=$(PRINT_MODEL_DIR=1 "$LAUNCH" "$m" 0,1)
    miss=""
    for task in $t; do
      ls "$RESULTS/$md/$a"/*_b$(printf '%04d' "$b")/"$task"/summary.json >/dev/null 2>&1 \
        || miss="$miss $task"
    done
    [ -n "$miss" ] && echo "$m $a $b$miss"
  done
}

drain() {   # run the supervisor until the queue is empty and no worker is left
  # grep -c prints 0 AND exits 1 on no match, so `|| echo 0` emits TWO lines and
  # the test dies with "integer expression expected" -- which reads as "queue
  # empty" and marches on to the next model. Let grep's own count stand.
  while [ "$(grep -cvE '^[[:space:]]*(#|$)' "$Q" 2>/dev/null; true)" -gt 0 ] \
        || pgrep -f '[w]orker\.sh [0-9],[0-9]' >/dev/null 2>&1; do
    # Foreground, so a supervisor that dies is restarted by this loop instead of
    # silently letting run.sh march on to the next model.
    bash $_P/scheduler/supervisor.sh 2>&1 | tee -a "$LOGS/supervisor.log"
    sleep 30
  done
}

# Prune: the allocator picks the pair, waiting for one to come free rather than
# charging at a busy card. prune_model.sh does the work.
prune_model() { "$HERE/gpu_alloc.sh" "$HERE/prune_model.sh" "$@"; }

# Start the NEXT model's prune now, on whichever pair the allocator can get.
# Called BEFORE this model's own prune, so both pairs are working from the first
# second of the sweep instead of 2,3 idling through every head-of-model prune.
# The allocator takes the same pair lock worker.sh does, so this can never
# double-book a pair against an eval cell.
prefetch() {   # prefetch <index of next model>
  local nxt="${MODELS[$1]:-}"
  [ -n "$nxt" ] || return 0
  local mk pk md; IFS=: read -r mk pk md <<<"$nxt"
  prune_model "$mk" "$pk" "$md" >> "$LOGS/prefetch.log" 2>&1 &
  PREFETCH_PID=$!
  echo "--- $(date -Is) prefetch prune queued for $mk (pid $PREFETCH_PID)"
}

PREFETCH_PID=""
idx=-1
for spec in "${MODELS[@]}"; do
  idx=$((idx+1))
  IFS=: read -r MKEY PKEY MDIR <<<"$spec"
  echo "=== $(date -Is) MODEL $MKEY =========================================="

  full_jobs "$MKEY" > "$Q.all"
  if ! missing_jobs < "$Q.all" | grep -q .; then
    echo "--- all cells already have a summary.json -- skipping $MKEY"
    continue
  fi

  # A prefetch for THIS model may still be running on the other pair.
  [ -n "$PREFETCH_PID" ] && { wait "$PREFETCH_PID" 2>/dev/null; PREFETCH_PID=""; }
  # Queue the next model's prune FIRST: it grabs the second pair and runs
  # concurrently with this model's, instead of waiting for this model's cells.
  prefetch $((idx+1))
  prune_model "$MKEY" "$PKEY" "$MDIR" \
    || { echo "--- skipping $MKEY, its prune failed"; continue; }

  missing_jobs < "$Q.all" > "$Q"
  echo "--- $(date -Is) queued $(wc -l < "$Q") cell(s) for $MKEY"
  drain

  # Retry while the missing set is still SHRINKING. Concurrent vLLM engine
  # starts fail transiently -- a cell that dies in 5s with no output reruns fine
  # 20s later -- so one requeue is too few. A pass that clears nothing means the
  # cells are genuinely broken, and repeating it would just spin.
  prev=$(( $(wc -l < "$Q.all") + 1 ))
  for _ in 1 2 3; do
    missing_jobs < "$Q.all" > "$Q.retry"
    n=$(wc -l < "$Q.retry")
    [ "$n" -eq 0 ] && break
    if [ "$n" -ge "$prev" ]; then
      echo "--- $(date -Is) retry pass cleared nothing -- $n cell(s) look genuinely broken:"
      sed 's/^/  STILL MISSING: /' "$Q.retry"
      break
    fi
    prev="$n"
    echo "--- $(date -Is) requeueing $n incomplete cell(s):"
    sed 's/^/      /' "$Q.retry"
    cp "$Q.retry" "$Q"; drain
  done

  # ...but only if this model is actually finished. Freeing the weights under a
  # still-missing cell makes cron's retry pay a full re-prune for nothing.
  if missing_jobs < "$Q.all" | grep -q .; then
    echo "--- $(date -Is) $MKEY still has missing cells -- KEEPING its checkpoints"
    continue
  fi

  # Free the disk before the next model. The observation cache (1-22 MB) and the
  # config stay -- re-pruning from a kept cache is minutes, and the cache is the
  # only artifact here that is expensive to recompute.
  for R in 0.25 0.50; do
    D="$ART/$MDIR/$DS/pruned_models/reap-renorm_true-seed_42-$R"
    ls "$D"/*.safetensors >/dev/null 2>&1 || continue
    echo "--- freeing $(du -sh "$D" | cut -f1) from $D"
    rm -f "$D"/*.safetensors "$D"/*.bin
  done
  df -h "$REAP_ROOT" | tail -1
done
left=$(for spec in "${MODELS[@]}"; do full_jobs "${spec%%:*}"; done | missing_jobs)
: > "$Q"
if [ -z "$left" ]; then
  echo "=== $(date -Is) SWEEP COMPLETE"
  touch "$CAMPAIGN/.sweep_done"   # stops cron from respawning this
else
  echo "=== $(date -Is) PASS ENDED with cells still missing -- cron will retry:"
  sed 's/^/  /' <<<"$left"
  exit 1
fi
tmux kill-window -t "moe:$WD" 2>/dev/null
