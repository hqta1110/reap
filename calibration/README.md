# Calibration caches

`prune.py` writes an **observation cache** during calibration and reuses it for
every compression ratio. It is the expensive artifact: with it, pruning is a
checkpoint write; without it, each model pays a full calibration pass on 2 GPUs.

    ./calibration/install_caches.sh     # copy into $REAP_ROOT/artifacts

## What is here

| model | codealpaca | tulu-personas-math | fineweb-edu |
|---|---|---|---|
| Qwen3-30B-A3B | ✅ 4×32 | ✅ 8×32 | ✅ 8×32 |
| GLM-4.7-Flash | ✅ 32×8 | ✅ 8×32 | ✅ 8×32 |
| Qwen3.6-35B-A3B | — | ✅ 8×32 | ✅ 8×32 |
| gemma-4-26B-A4B-it | — | ✅ 8×32 | **— must calibrate** |

So of the fineweb work still outstanding, **qwen3.6 prunes in minutes** and
**gemma-4 needs a real calibration pass** (~30-45 min including checkpoint writes).
The three missing cells are simply campaigns that were never run, not losses.

`data/fineweb_edu_calibration.parquet` is a 1200-row local slice, so fineweb
calibration needs no network.

## The filename is the cache key — nothing validates it

    observations_bs{BS}x{NB}_L{LEN}_{measure}-seed_{SEED}.pt

A cache built at a different budget **loads silently** and you get a checkpoint
that is not the one you think. Hence every knob is in the name, and only the cache
each corpus's existing checkpoints were actually built from is committed here.

Note the budgets are not uniform, because the runs were not: codealpaca used
128 samples (Qwen 4×32, GLM 32×8), tulu-math and fineweb used 256 (8×32). That is
a real confound in any code-vs-other comparison — see STATUS.md.
