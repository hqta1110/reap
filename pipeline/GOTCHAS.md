# Things that cost real time here

Each of these was a silent failure — the run looked fine and the numbers were
wrong or absent. They are in the code as comments too; this is the index.

## 1. `exec 9>&- 2>/dev/null` destroys the script's stderr

That form redirects stderr **for the life of the script**, not just the fd close.
Every prune traceback (OOM, bad dataset, missing snapshot) vanished and the caller
saw a bare `rc=1`; cells that died before the engine started wrote **zero-byte
logs**. Use `{ exec 9>&-; } 2>/dev/null`. Cost: four models' worth of OOM went
unnoticed through repeated cron passes, and an unrelated bug was misdiagnosed
because its error was invisible.

## 2. The observation-cache filename IS the cache key

`observations_bs{BS}x{NB}_L{LEN}_{measure}-seed_{SEED}.pt`. **Nothing validates
it.** A cache built at a different budget loads silently. Every knob must be
encoded in the name — and when completing an existing run, match the budget it
used. codealpaca here is `4x32=128` samples; tulu-math and fineweb are `8x32=256`.

## 3. Calibration corpus is not recorded in any result metadata

It is recoverable only from the checkpoint path or the arm name. Hence arms carry
the corpus (`reap25_fw256`, `reap25_math256`) and bare `reap25` means codealpaca.
A result whose corpus cannot be determined prints as `unrecorded`.

## 4. `VLLM_PLUGINS` must be unset for REAP arms

With it set you are measuring REAP+SERE and reporting it as REAP. `verify_reap.sh`
fails any cell whose log contains `Enabled SERE`. The environment is scrubbed at
launch, not trusted.

## 5. `hit_length_cap` is a COUNT, not a percent

Its denominator is `n` in the same file. Above ~20% cap-hit a score measures
**termination, not ability** — a pruned model that never stops generating scores 0
on everything. Always read a score next to its cap rate. Several fineweb rows here
are truncation-limited and cannot carry a corpus comparison.

## 6. `grep -c` prints 0 AND exits 1

`$(grep -c ... || echo 0)` emits **two lines**, and every integer test on it dies
with "integer expression expected" — which reads as *queue empty* and marches on.
Let grep's own count stand.

## 7. `tmux new-window` does not inherit the caller's environment

It gets the **tmux server's**. Every variable a worker needs must be named
explicitly in supervisor's `bash -c` line. A new variable added to `run.sh`'s
export list silently does not reach the launcher.

## 8. `pgrep -f X` matches its own wrapper

A `pgrep -f run.sh` wait loop matches the tmux command line containing that string
and never exits. Cost: four idle hours once, and one killed shell here. Anchor the
pattern (`[w]orker\.sh 0,1$`) or filter out `$$`.

## 9. `device_map="auto"` is sequential model parallel

More GPUs do not make pruning faster. Two is the right number; the rest of the box
should be running eval cells.

## 10. Newest-glob directory resolution races concurrent pairs

Picking the newest matching result dir gets the *other* pair's directory when both
lanes run the same model+arm+batch: a cell that scored perfectly verifies against a
sibling's empty dir and reports FAIL. Take the dir the pipeline printed.
