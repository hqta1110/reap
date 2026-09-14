#!/usr/bin/env bash
# Copy the committed observation caches into $REAP_ROOT/artifacts, where
# prune.py looks for them. Run once on a new machine, before any campaign.
#
# Why this is worth 73 MB of repo: the observation cache IS the calibration.
# With it, pruning a model is a checkpoint write (minutes). Without it, every
# ratio pays a full forward pass over the calibration set on 2 GPUs first.
# Observations are ratio-independent -- one cache serves 0.25 and 0.50 both.
#
# Safe to re-run: never overwrites an existing cache, since a local one may have
# been built at a budget these were not.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); P=$HERE/../pipeline
. "$P/reap.env.example"; [ -f "$P/reap.env" ] && . "$P/reap.env"
n=0; skip=0
while IFS= read -r src; do
  rel=${src#$HERE/observations/}
  dst="$REAP_ROOT/artifacts/$rel"
  if [ -f "$dst" ]; then echo "  keep (exists)  $rel"; skip=$((skip+1)); continue; fi
  mkdir -p "$(dirname "$dst")" && cp "$src" "$dst" && { echo "  installed      $rel"; n=$((n+1)); }
done < <(find "$HERE/observations" -name '*.pt')
mkdir -p "$(dirname "$REAP_FINEWEB_PARQUET")"
[ -f "$REAP_FINEWEB_PARQUET" ] || cp "$HERE/data/fineweb_edu_calibration.parquet" "$REAP_FINEWEB_PARQUET"
echo "installed $n cache(s), kept $skip; fineweb parquet -> $REAP_FINEWEB_PARQUET"
