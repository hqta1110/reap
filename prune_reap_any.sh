#!/usr/bin/env bash
# Calibrate + prune any of the four benchmark models with REAP, then apply the
# three post-prune fixups from REAP_RUNBOOK.md §3. The runbooks have referenced
# this script for a while; it only existed as the GLM-only glm_acc_scratch/prune_reap.sh.
#
# Usage:  [GPUS=0,1] [DATASET=<hf-id>] ./prune_reap_any.sh <mkey> <ratio> <BS> <NB>
#   mkey   qwen330 | glm47 | qwen36 | gemma4
#   BS*NB  = calibration samples (256 is the house number)
#
# Observations are ratio-independent: run 0.25 first, 0.50 reuses the cache.
# The observation filename IS the cache key and nothing validates it, so every
# knob is encoded into the name (runbook §6).
set -uo pipefail
# start.sh guards the sweep with flock on fd 9, and a bash redirection is not
# close-on-exec: every descendant inherits the lock. An orphaned engine or
# prune would then hold it after run.sh died, and cron could never restart
# the sweep. Drop the inherited fd -- only run.sh should hold the lock.
# Braces, not `exec 9>&- 2>/dev/null`: that form redirects THIS SCRIPT'S stderr
# to /dev/null for the rest of its life, so every prune traceback -- OOM, a bad
# dataset name, a missing snapshot -- vanished and the caller saw a bare `rc=1`.
# Only the fd-close's own "Bad file descriptor" needs suppressing.
{ exec 9>&-; } 2>/dev/null || true
MKEY="$1"; RATIO="$2"; BS="${3:-8}"; NB="${4:-32}"
DATASET="${DATASET:-allenai/tulu-3-sft-personas-math}"
DSDIR="${DATASET##*/}"
HUB=/home/PC/.cache/huggingface/hub
# REAP_ROOT, not a hardcoded path: this script was pinned to one machine's clone
# (/home/PC/reap), so on any other checkout the prune wrote its checkpoints into
# the OTHER tree while launch_reap.sh looked under $REAP_ROOT/artifacts -- every
# cell then failed with "no pruned checkpoint at ...". reap.env is the only
# per-machine file; honour it here too.
_RP=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$_RP/pipeline/reap.env.example"
[ -f "$_RP/pipeline/reap.env" ] && . "$_RP/pipeline/reap.env"
cd "$REAP_ROOT" || exit 2
PY=/home/PC/reap/.venv-reap/bin/python

case "$MKEY" in
  qwen330) MODEL='Qwen/Qwen3-30B-A3B'        ; MDIR=Qwen3-30B-A3B      ; BASE_EXPERTS=128
           SNAP=$HUB/models--Qwen--Qwen3-30B-A3B/snapshots/* ;;
  glm47)   MODEL='zai-org/GLM-4.7-Flash'     ; MDIR=GLM-4.7-Flash      ; BASE_EXPERTS=64
           SNAP=$HUB/models--zai-org--GLM-4.7-Flash/snapshots/* ;;
  qwen36)  MODEL='Qwen/Qwen3.6-35B-A3B'      ; MDIR=Qwen3.6-35B-A3B    ; BASE_EXPERTS=256
           SNAP=$HUB/models--Qwen--Qwen3.6-35B-A3B/snapshots/* ;;
  gemma4)  MODEL='google/gemma-4-26B-A4B-it' ; MDIR=gemma-4-26B-A4B-it ; BASE_EXPERTS=128
           SNAP=$HUB/models--google--gemma-4-26B-A4B-it/snapshots/* ;;
  *) echo "unknown model key $MKEY" >&2; exit 2 ;;
esac
# One snapshot per model here, but glob-then-head so a second revision fails loud
# rather than silently pruning the stale one (runbook §3 of the calibration doc).
set -- $SNAP
[ $# -eq 1 ] || { echo "expected exactly 1 snapshot for $MKEY, found $#: $*" >&2; exit 2; }
SNAP="$1"

export CUDA_VISIBLE_DEVICES="${GPUS:-0,1}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_NET=Socket NCCL_IB_DISABLE=1
unset NCCL_TUNER_CONFIG_PATH 2>/dev/null || true
export LD_LIBRARY_PATH="$(printf '%s' "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v gib | paste -sd':' -)"

OBS="observations_bs${BS}x${NB}_L256_cosine-seed_42.pt"
echo "=== [$(date -Is)] REAP $MKEY ratio=$RATIO (${BS}x${NB}=$((BS*NB)) @ L256) dataset=$DATASET gpus=$CUDA_VISIBLE_DEVICES"

# An aborted earlier run leaves partial.pkl next to the cache. It is NOT a usable
# cache and its presence means the real one may be from a different run.
PART="artifacts/$MDIR/$DSDIR/all/partial.pkl"
[ -f "$PART" ] && { echo "removing stale $PART"; rm -f "$PART"; }

# --model_max_length 256 is the knob that keeps calibration at minutes instead of
# hours: without it the packing length is the model default (up to 1e30 on gemma4).
"$PY" src/reap/prune.py \
  --model-name "$MODEL" \
  --dataset-name "$DATASET" \
  --prune-method reap --compression-ratio "$RATIO" \
  --profile false --do-eval false --distance_measure cosine --seed 42 \
  --model_max_length 256 --truncate true --smoke_test false \
  --batch_size "$BS" --batches_per_category "$NB" \
  --record_pruning_metrics_only true \
  --output_file_name "$OBS"
rc=$?; [ $rc -ne 0 ] && { echo "=== prune rc=$rc ==="; exit $rc; }

RSTR=$(printf "%.2f" "$RATIO")
D=$(find "artifacts/$MDIR/$DSDIR" -type d -path "*pruned_models*seed_42-$RSTR" 2>/dev/null | head -1)
[ -n "$D" ] || { echo "PRUNED DIR NOT FOUND under artifacts/$MDIR/$DSDIR" >&2; exit 2; }
D=$(readlink -f "$D")
echo "=== pruned dir: $D"

# --- fixup 2: a stale shard index outvotes the files on disk -------------------
# vLLM trusts model.safetensors.index.json over what is actually there and reports
# "Cannot find any model weights" when a previous prune left one behind.
if [ -f "$D/model.safetensors" ] && [ -f "$D/model.safetensors.index.json" ]; then
  mv -f "$D/model.safetensors.index.json" "$D/model.safetensors.index.json.stale"
  echo "  moved aside stale shard index"
fi

# --- fixup 1: restore auxiliary files from the base snapshot -------------------
# Everything except config.json, the weights and the shard index. Qwen3.6 died in
# vLLM for want of video_preprocessor_config.json, surfacing as an unrelated
# "Transformers does not recognize qwen3_5_moe".
# OVERWRITE, do not merely fill gaps. prune.py re-serialises the tokenizer with
# the PRUNING env's transformers (5.16), which writes `extra_special_tokens` as a
# list; the older transformers in every eval venv reads it as a dict and dies with
#   AttributeError: 'list' object has no attribute 'keys'
# before a single token is generated. The base snapshot's copy is the one both
# versions can read, so the restore has to replace what pruning just wrote.
n=0
for f in "$SNAP"/*; do
  b=$(basename "$f"); [ -f "$f" ] || continue
  case "$b" in config.json|*.safetensors|*.bin|*.pth|model.safetensors.index.json|*.md|.git*) continue ;; esac
  cp -f "$f" "$D/" && n=$((n+1))
done
echo "  restored $n auxiliary file(s) from the base snapshot (overwriting)"

# --- fixup 3: mirror the expert-count key -------------------------------------
# transformers 5.16 writes Qwen3Moe's count as num_local_experts; the older
# transformers in the eval venvs reads num_experts and, when absent, silently
# falls back to the class default of 128. Wrong expert count, garbage scores,
# no warning. Mirror whatever the base config named onto the pruned value.
"$PY" - "$D" "$SNAP" <<'PYEOF'
import json, sys
KEYS = ("n_routed_experts", "num_experts", "num_local_experts")
d, snap = sys.argv[1], sys.argv[2]
pc = json.load(open(f"{d}/config.json"))
bc = json.load(open(f"{snap}/config.json"))

def scopes(c):
    yield c
    for nest in ("text_config", "language_config"):
        if isinstance(c.get(nest), dict):
            yield c[nest]

changed = []
for ps, bs_ in zip(scopes(pc), scopes(bc)):
    have = {k: ps[k] for k in KEYS if k in ps}
    if not have:
        continue
    val = min(have.values())          # the pruned count is the smaller one if both linger
    # Only keys the BASE config actually used -- do not invent keys for this arch.
    for k in KEYS:
        if k in bs_ and ps.get(k) != val:
            ps[k] = val; changed.append(f"{k}={val}")
if changed:
    json.dump(pc, open(f"{d}/config.json", "w"), indent=2)
print("  mirrored:", ", ".join(changed) if changed else "nothing to mirror")
PYEOF

# --- the gate: empty output here is a FAILURE, not a pass ---------------------
EXPECT=$("$PY" -c "print(round($BASE_EXPERTS*(1-$RATIO)))")
echo "=== GATE: expert count (expect $EXPECT = $BASE_EXPERTS x (1-$RATIO))"
GOT=$(grep -oE '"(n_routed_experts|num_experts|num_local_experts)":[[:space:]]*[0-9]+' "$D/config.json" \
      | grep -oE '[0-9]+$' | sort -u | paste -sd,)
echo "  found: ${GOT:-<none>}"
[ "$GOT" = "$EXPECT" ] || { echo "GATE FAILED: expert count is '${GOT:-<none>}', expected $EXPECT" >&2; exit 3; }
echo "PRUNE_DONE dir=$D"
