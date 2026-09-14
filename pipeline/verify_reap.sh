#!/usr/bin/env bash
# Machine-local paths. Sourced relative to this script, so a clone anywhere works.
_P=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); _P=${_P%/scheduler}
. "$_P/reap.env.example"; [ -f "$_P/reap.env" ] && . "$_P/reap.env"
# Gate one finished REAP cell. Prints VERDICT=PASS or VERDICT=FAIL <reason>.
# Usage: verify_reap.sh <arm> <logfile> <result_dir> [tasks]
#
# Same contract as sweep2/verify_cell.sh. Two differences:
#   - the arm check is inverted: a REAP cell that loaded the SERE plugin is
#     REAP+SERE mislabelled as REAP, which is the one failure a score cannot show.
#   - expected row counts come from the task yaml (n_expected) instead of a
#     hardcoded list, so a new task does not read as a failed cell.
set -u
ARM="$1"; LOG="$2"; DIR="$3"; TASKS="${4:-gsm8k}"
CFG=$EVAL_REPO/configs/tasks
fail() { echo "VERDICT=FAIL $*"; exit 1; }

[ -f "$LOG" ] || fail "no log $LOG"
# grep -c prints "0" AND exits 1 on no match; `|| echo 0` would emit TWO lines and
# every integer test below would die with "integer expression expected" -- which
# reads as a FAILED cell. Let grep's own 0 stand.
EN=$(grep -c 'Enabled SERE' "$LOG" 2>/dev/null; true); EN=${EN:-0}
[ "$EN" -eq 0 ] || fail "REAP arm has $EN 'Enabled SERE' lines -- this is REAP+SERE, not REAP"

# The expert count is what silently regresses: a config key the eval venv cannot
# read falls back to the class default and the checkpoint loads UNPRUNED.
EXPECT_E=$(grep -ho 'num_experts=[0-9]*\|n_routed_experts=[0-9]*' "$LOG" | head -1)
[ -n "$EXPECT_E" ] && echo "  engine reported $EXPECT_E"

for t in $TASKS; do
  S="$DIR/$t/summary.json"
  [ -f "$S" ] || fail "missing $t/summary.json"
  Y="$CFG/$t.yaml"
  [ -f "$Y" ] || fail "no task config $Y"
  exp=$(grep -m1 -oE 'n_expected:[[:space:]]*[0-9]+' "$Y" | grep -oE '[0-9]+')
  [ -n "$exp" ] || fail "$Y has no n_expected"
  read -r n val <<<"$(python3 -c "
import json;d=json.load(open('$S'));print(d.get('n'),d.get('value'))" 2>/dev/null)"
  [ "$n" = "$exp" ] || fail "$t scored n=$n, expected $exp (partial coverage)"
  [ "$val" != "None" ] && [ -n "$val" ] || fail "$t has no score"
  # Cap-hit is REPORTED, never gated: above ~20% the score measures termination,
  # not accuracy (REAP_RUNBOOK §5), and that is a finding about the arm rather
  # than a broken cell. Refusing to push it would hide the dominant effect.
  CAP=$(python3 - "$DIR/$t" <<'PYEOF'
import json, glob, sys
tot = hit = 0
for f in glob.glob(f"{sys.argv[1]}/*.meta.json"):
    try: d = json.load(open(f))
    except Exception: continue
    h = d.get("hit_length_cap")
    if isinstance(h, dict): hit += h.get("n", 0); tot += h.get("total", 0)
    # scalar form is a COUNT of capped generations, not a percent -- the row
    # count in the same file is its denominator.
    elif isinstance(h, (int, float)): hit += h; tot += d.get("n", 0)
print(f"{100.0*hit/tot:.1f}" if tot else "")
PYEOF
)
  echo "  $t: n=$n value=$val${CAP:+ cap=${CAP}%}"
done
echo "VERDICT=PASS"
