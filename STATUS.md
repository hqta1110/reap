# REAP campaign status — 2026-09-14

## Accuracy matrix

4 models × 2 ratios × 9 tasks = 72 cells per calibration corpus. Protocol: 0-shot,
greedy, `max_tokens` 4096, offline batch-pinned at B=64, TP=2, seed 42.

| calibration corpus | cells | state |
|---|---|---|
| codealpaca (`evol-codealpaca-v1`, 4×32=128 samples) | **72/72** | complete |
| tulu-3-sft-personas-math (8×32=256) | **72/72** | complete |
| fineweb-edu (8×32=256) | **36/72** | **stopped mid-run** |

Regenerate any time:

    python3 tools/reap_coverage.py     # what exists
    python3 tools/reap_report.py       # what it says

## Where fineweb stopped

Done: **qwen3-30b** and **glm-4.7-flash**, both ratios, all nine tasks.
Not started: **qwen3.6-35b** and **gemma-4**.

The sweep was stopped cleanly — cron removed, tmux windows killed, one in-flight
qwen3.6 cell terminated. Nothing landed was lost; everything that completed is
pushed. `run.sh` resumes from disk, so restarting re-queues exactly what is missing.

To continue:

    ./pipeline/start.sh $REAP_STATE/fineweb

Prune cost remaining, on the machine that ran this:

| model | fineweb observation cache | cost |
|---|---|---|
| qwen3.6-35b | present (`bs8x32_L256_cosine-seed_42`) | minutes — checkpoint write only |
| gemma-4 | **absent** | full calibration + 2 checkpoint writes, ~30–45 min |

Estimated ~2.5 h of GPU time for the remaining 36 cells on 4×H100.

## Read the numbers with the cap rate

`hit_length_cap` above ~20% means the score measures **termination, not ability**.
Several fineweb rows are truncation-limited and must not be used for a corpus
comparison. Measured on the fineweb reap25 rows:

| | glm-4.7 | qwen3-30b |
|---|---|---|
| gsm8k | 62.77 · cap 7.1% ✅ | 83.32 · cap 9.4% ✅ |
| math500 | 12.60 · cap 72.2% ✗ | 46.20 · cap 46.6% ✗ |
| humaneval | 1.22 · cap **92.7%** ✗ | 12.20 · cap 70.1% ✗ |
| mmlu400 | 30.25 · cap 45.8% ✗ | 47.25 · cap 20.8% ~ |

qwen3.6 and gemma-4 ran much lower cap rates on the other two corpora, so they are
the rows that will actually carry the three-way corpus comparison. Worth finishing
before drawing a conclusion.

## Known confound

codealpaca checkpoints were calibrated at **128 samples**, tulu-math and fineweb at
**256**. So code-vs-{math,fineweb} carries a calibration-*budget* difference on top
of the corpus difference. Both are far below REAP's 8192-sample default, so the
effect is probably small — but it is not zero and the accuracy report does not yet
mention it.

## Speed

`reports/REAP_SPEED_REPORT.md` is complete: 4 models × {vanilla, reap25, reap50} ×
7 regimes, measured 2026-09-13. reap50 gives **+15% to +34% output tok/s** at c=64,
growing with concurrency, flat in the prefill-dominated control — consistent with a
memory-traffic effect, since REAP leaves `num_experts_per_tok` at top-8 and
per-token FLOPs unchanged.

Those speed checkpoints were calibrated on **fineweb-edu**. Fine for timing (corpus
decides which experts drop, not how many), but pair speed with the fineweb accuracy
rows, not the math or code ones.
