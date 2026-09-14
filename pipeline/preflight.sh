#!/usr/bin/env bash
# Check everything a campaign needs BEFORE starting one. Exits nonzero if any
# hard dependency is missing, so a bad setup fails here in seconds rather than
# 40 minutes into a prune.
#
# This repo carries the REAP half (drivers, caches, results). The EVAL half --
# harness, per-model venvs, base weights -- is external and is the bulk of the
# setup on a fresh machine.
set -u
P=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$P/reap.env.example"; [ -f "$P/reap.env" ] && . "$P/reap.env"
ok=0; bad=0
chk() { # chk <label> <test-result> <fix>
  if [ "$2" = 0 ]; then printf '  \033[32mOK\033[0m   %s\n' "$1"; ok=$((ok+1))
  else printf '  \033[31mMISS\033[0m %s\n       -> %s\n' "$1" "$3"; bad=$((bad+1)); fi
}
echo "== paths (edit pipeline/reap.env to change)"
for v in REAP_ROOT REAP_PY EVAL_REPO REAP_STATE HF_HUB; do printf '  %-12s %s\n' "$v" "${!v}"; done

echo "== pruning side (this repo)"
[ -f "$REAP_ROOT/src/reap/prune.py" ]; chk "REAP source at \$REAP_ROOT" $? "clone this repo to \$REAP_ROOT"
[ -x "$REAP_PY" ]; chk "prune venv \$REAP_PY" $? "uv venv \$REAP_ROOT/.venv-reap && uv pip install -e \$REAP_ROOT"
if [ -x "$REAP_PY" ]; then
  tf=$("$REAP_PY" -c 'import transformers;print(transformers.__version__)' 2>/dev/null)
  "$REAP_PY" - <<'PY' >/dev/null 2>&1
import transformers,sys
sys.exit(0 if tuple(int(x) for x in transformers.__version__.split(".")[:2]) >= (5,16) else 1)
PY
  chk "transformers >= 5.16 (got ${tf:-none}) -- Qwen3-MoE/Gemma-4 need it" $? "uv pip install -U 'transformers>=5.16'"
fi
n=$(find "$P/../calibration/observations" -name '*.pt' 2>/dev/null | wc -l)
[ "$n" -gt 0 ]; chk "observation caches in repo ($n)" $? "git lfs pull / re-clone"
ls "$REAP_ROOT"/artifacts/*/*/all/*.pt >/dev/null 2>&1
chk "caches installed into \$REAP_ROOT/artifacts" $? "./calibration/install_caches.sh"

echo "== eval side (external)"
[ -f "$EVAL_REPO/tools/run_offline_pipeline.py" ]
chk "eval harness at \$EVAL_REPO" $? "git clone -b feat/offline-batch-pinned-eval https://github.com/hqta1110/moe-eval-unified.git \$EVAL_REPO"
[ -n "${RESULTS_REPO:-}" ] && [ -d "${RESULTS_REPO:-/nonexistent}/.git" ]
chk "RESULTS_REPO (worker pushes landed cells here)" $? "git clone https://github.com/hqta1110/moe-eval-results.git; export RESULTS_REPO=..."
for pair in "VENV_QWEN qwen3-30b/qwen3.6" "VENV_GLM glm-4.7-flash" "VENV_GEMMA4 gemma-4"; do
  set -- $pair; v=${!1}
  [ -x "$v" ]; chk "$1 ($2)" $? "create the venv; gemma-4 needs vLLM 0.29, the others 0.18.1"
done
ls -d "$HF_HUB"/models--*/snapshots/* >/dev/null 2>&1
chk "base model snapshots in \$HF_HUB (~232 GB for all four)" $? "huggingface-cli download <model>"

echo "== hardware"
g=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
[ "$g" -ge 2 ]; chk "GPUs visible ($g) -- 2 per TP=2 lane" $? "need at least one pair"
free=$(df -BG --output=avail "$REAP_ROOT" 2>/dev/null | tail -1 | tr -dc '0-9')
[ "${free:-0}" -ge 150 ]; chk "free disk ${free:-?}G (>=150G for checkpoints)" $? "free space under \$REAP_ROOT"

echo
echo "$ok ok, $bad missing"
[ "$bad" -eq 0 ] && echo "ready: ./pipeline/start.sh \$REAP_STATE/<corpus>" || echo "fix the above first"
exit $((bad > 0))
