#!/usr/bin/env bash
# Machine-local paths. Sourced relative to this script, so a clone anywhere works.
_P=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); _P=${_P%/scheduler}
. "$_P/reap.env.example"; [ -f "$_P/reap.env" ] && . "$_P/reap.env"
# Prune one model at both ratios on the pair in $GPUS (set by gpu_alloc.sh).
# Split out of run.sh so the allocator can invoke it directly, instead of the
# `declare -f prune_model` trick the old prefetch used to smuggle a function
# through bash -c.
set -u
CAMPAIGN="${CAMPAIGN:?set CAMPAIGN=$REAP_STATE/<corpus>}"
. "$CAMPAIGN/campaign.env"
MK="$1"; PK="$2"; MD="$3"
ART=$REAP_ROOT/artifacts
LOGS=$CAMPAIGN/logs
# Two models' checkpoints now coexist (this one plus the prefetched next), so
# check the disk rather than assume the old one-model-at-a-time peak.
MIN_FREE_GB="${MIN_FREE_GB:-150}"
# Only the ratios asked for: filling a single missing cell should not pay for
# a second checkpoint nobody needs.
RATIOS="${RATIOS:-0.25 0.50}"

for R in $RATIOS; do
  D="$ART/$MD/$DS/pruned_models/reap-renorm_true-seed_42-$R"
  if ls "$D"/*.safetensors >/dev/null 2>&1; then
    echo "--- checkpoint already present for $MK $R -- skipping prune"; continue
  fi
  free=$(df -BG --output=avail "$REAP_ROOT" | tail -1 | tr -dc '0-9')
  if [ "${free:-0}" -lt "$MIN_FREE_GB" ]; then
    echo "--- $(date -Is) only ${free}G free (< ${MIN_FREE_GB}G) -- refusing to prune $MK $R"
    exit 1
  fi
  PL="$LOGS/prune_${MK}_${R}.log"
  echo "--- $(date -Is) prune $MK ratio=$R on GPU ${GPUS:-?} -> $PL"
  DATASET="$DATASET" $REAP_ROOT/prune_reap_any.sh "$PK" "$R" "${CALIB_BS:-8}" "${CALIB_NB:-32}" > "$PL" 2>&1 && { tail -3 "$PL"; continue; }
  # Surface the reason in run.log. A bare "PRUNE FAILED" plus a log that the next
  # cron pass truncates is how four models' worth of OOM went unnoticed.
  echo "PRUNE FAILED $MK $R (see $PL):"
  tail -20 "$PL" | sed 's/^/    /'
  exit 1
done
