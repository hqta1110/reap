#!/usr/bin/env bash
# Machine-local paths. Sourced relative to this script, so a clone anywhere works.
_P=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); _P=${_P%/scheduler}
. "$_P/reap.env.example"; [ -f "$_P/reap.env" ] && . "$_P/reap.env"
# Self-check for gpu_alloc.sh: pair selection, and that a command's own exit
# status is never confused with the reserved "pair busy" codes. Fakes nvidia-smi
# on PATH; BUSY_GPUS says which GPU indices are occupied.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
cat > "$T/nvidia-smi" <<'STUB'
#!/usr/bin/env bash
for a in "$@"; do case "$a" in -i) next=1;; *) [ "${next:-}" = 1 ] && { ids="$a"; next=0; };; esac; done
busy=",${BUSY_GPUS:-},"
hit=0; for g in ${ids//,/ }; do case "$busy" in *",$g,"*) hit=1;; esac; done
case "$*" in
  *compute-apps*) [ "$hit" = 1 ] && echo 4242 ;;
  *memory.used*)  for g in ${ids//,/ }; do case "$busy" in *",$g,"*) echo 75000;; *) echo 0;; esac; done ;;
esac
STUB
chmod +x "$T/nvidia-smi"; export PATH="$T:$PATH"
# Pair locks live at a fixed path; point them somewhere disposable for the test.
export GPU_POLL=1
pass=0; fail=0
check() { if [ "$2" = "$3" ]; then pass=$((pass+1)); else fail=$((fail+1)); echo "FAIL: $1 -- got '$2' want '$3'"; fi; }

got=$(BUSY_GPUS= "$HERE/gpu_alloc.sh" bash -c 'echo $GPUS' | tail -1)
check "both pairs idle -> takes one" "$got" "0,1"

got=$(BUSY_GPUS=0,1 "$HERE/gpu_alloc.sh" bash -c 'echo $GPUS' | tail -1)
check "pair 0,1 occupied -> falls through to 2,3" "$got" "2,3"

BUSY_GPUS= "$HERE/gpu_alloc.sh" bash -c 'exit 3' >/dev/null 2>&1
check "command exit status propagates" "$?" "3"

BUSY_GPUS= "$HERE/gpu_alloc.sh" bash -c 'exit 1' >/dev/null 2>&1
check "a failing prune is not read as 'pair busy'" "$?" "1"

timeout 5 env BUSY_GPUS=0,1,2,3 "$HERE/gpu_alloc.sh" bash -c 'echo RAN' >/dev/null 2>&1
check "all GPUs occupied -> waits, never runs" "$?" "124"

echo "gpu_alloc: $pass passed, $fail failed"; [ "$fail" -eq 0 ]
