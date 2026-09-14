#!/usr/bin/env bash
# Machine-local paths. Sourced relative to this script, so a clone anywhere works.
_P=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); _P=${_P%/scheduler}
. "$_P/reap.env.example"; [ -f "$_P/reap.env" ] && . "$_P/reap.env"
# Idempotent entry point for one REAP campaign: start.sh <campaign-dir>. Safe to run any number of
# times, from cron or by hand: if the sweep is already running it does nothing.
#
# This is the recovery layer. run.sh resumes from what is on disk, worker.sh and
# supervisor.sh cover a dead worker, watchdog.sh covers a wedged one -- but
# nothing covered run.sh itself dying, or the host rebooting and taking the tmux
# server with it. Cron calls this every 10 minutes; the lock makes that a no-op
# while the sweep is healthy and a restart when it is not.
set -u
CAMPAIGN="${1:-${CAMPAIGN:?pass a campaign dir, e.g. $REAP_STATE/fineweb}}"
export CAMPAIGN
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LOCK=$CAMPAIGN/.run.lock
LOGS=$CAMPAIGN/logs
DONE=$CAMPAIGN/.sweep_done
WIN=$(basename "$CAMPAIGN")   # one window per campaign; two can run back to back
export PATH=/usr/local/bin:/usr/bin:/bin:$PATH   # cron's PATH does not carry tmux
mkdir -p "$LOGS"

# Cron keeps calling this forever; once the sweep has finished, stop respawning a
# run.sh that would only re-scan and exit. Delete the sentinel to force a rerun.
[ -f "$DONE" ] && exit 0

# flock, not a pidfile or a pgrep: the lock dies with the process that holds it,
# so a killed run.sh leaves nothing stale to clean up. (A `pgrep -f run.sh` here
# would also match this script's own tmux command line -- the self-matching wait
# loop that once cost four idle hours.)
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "$(date -Is) sweep already running (lock held) -- nothing to do"; exit 0
fi
flock -u 9    # hand the lock to run.sh inside tmux rather than holding it here

tmux has-session -t moe 2>/dev/null || tmux new-session -d -s moe
tmux kill-window -t "moe:$WIN" 2>/dev/null
tmux new-window -t moe -n "$WIN" -d \
  "bash -c 'flock -n 9 || exit 0; CAMPAIGN=$CAMPAIGN $HERE/run.sh 2>&1 | tee -a $LOGS/run.log; exec 9>&-; exec bash' 9>$LOCK"
echo "$(date -Is) $WIN started in tmux moe:$WIN  (attach: tmux attach -t moe)"
