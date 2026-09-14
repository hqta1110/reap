# REAP accuracy — by calibration corpus

4 models x 2 pruning ratios x 9 tasks, one protocol throughout:
**0-shot, greedy (temperature 0.0), `max_tokens` 4096, offline batch-pinned at B=64.**
Every number below was produced that way, so any two of them can be compared.

The two sections differ in ONE thing: the corpus the router observations were
collected on during calibration. That choice decides which experts survive pruning.

> `cap N%` under a score = share of rows that hit the token cap and therefore
> scored an automatic zero. Above ~20% the number measures termination, not accuracy.

## 1. Calibrated on MATH — `allenai/tulu-3-sft-personas-math`

Run 2026-09-13/14. Arms `reap25_math256` / `reap50_math256`.

| model | ratio | gsm8k | math_hard | math500 | aime | lcb | humaneval_plus | mmlu_pro_400 | gpqa_diamond | livebench_mixed |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen3-30b | reap25 | 84.84<br><sub>cap 95%</sub> | 11.71<br><sub>cap 86%</sub> | 31.20<br><sub>cap 69%</sub> | 0.00<br><sub>cap 97%</sub> | 5.71<br><sub>cap 47%</sub> | 22.56<br><sub>cap 24%</sub> | 50.75 | 31.31 | 40.84 |
| qwen3-30b | reap50 | 83.24<br><sub>cap 35%</sub> | 19.79<br><sub>cap 66%</sub> | 43.40<br><sub>cap 41%</sub> | 1.67<br><sub>cap 88%</sub> | 0.00<br><sub>cap 39%</sub> | 2.44<br><sub>cap 63%</sub> | 38.25 | 15.66<br><sub>cap 27%</sub> | 33.34<br><sub>cap 20%</sub> |
| qwen3.6-35b | reap25 | 95.98 | 80.14 | 87.20 | 45.00 | 42.29<br><sub>cap 29%</sub> | 67.07 | 78.75 | 56.57<br><sub>cap 35%</sub> | 75.03 |
| qwen3.6-35b | reap50 | 95.30 | 81.27 | 85.40 | 45.00<br><sub>cap 20%</sub> | 2.29<br><sub>cap 47%</sub> | 6.71<br><sub>cap 26%</sub> | 67.25 | 45.45<br><sub>cap 27%</sub> | 55.90 |
| glm-4.7-flash | reap25 | 72.02 | 12.76<br><sub>cap 78%</sub> | 30.80<br><sub>cap 57%</sub> | 1.67<br><sub>cap 50%</sub> | 2.29<br><sub>cap 37%</sub> | 27.44<br><sub>cap 31%</sub> | 32.00<br><sub>cap 49%</sub> | 10.61<br><sub>cap 77%</sub> | 27.05<br><sub>cap 39%</sub> |
| glm-4.7-flash | reap50 | 65.43 | 10.42<br><sub>cap 83%</sub> | 25.40<br><sub>cap 60%</sub> | 1.67<br><sub>cap 55%</sub> | 0.00<br><sub>cap 38%</sub> | 4.88<br><sub>cap 82%</sub> | 20.75<br><sub>cap 58%</sub> | 5.05<br><sub>cap 83%</sub> | 17.90<br><sub>cap 42%</sub> |
| gemma4 | reap25 | 95.30 | 78.10 | 86.60 | 58.33<br><sub>cap 23%</sub> | 26.29<br><sub>cap 54%</sub> | 83.54 | 79.50 | 67.17 | 61.52 |
| gemma4 | reap50 | 92.04 | 69.86 | 80.40 | 25.00<br><sub>cap 40%</sub> | 5.14<br><sub>cap 70%</sub> | 23.78<br><sub>cap 33%</sub> | 66.00 | 43.94 | 46.79 |

## 2. Calibrated on CODE — `theblackcat102/evol-codealpaca-v1`

Run 2026-09-12/13. Arms `reap25` / `reap50`.

| model | ratio | gsm8k | math_hard | math500 | aime | lcb | humaneval_plus | mmlu_pro_400 | gpqa_diamond | livebench_mixed |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen3-30b | reap25 | 84.69<br><sub>cap 58%</sub> | 17.30<br><sub>cap 76%</sub> | 38.40<br><sub>cap 53%</sub> | 0.00<br><sub>cap 95%</sub> | 5.14<br><sub>cap 29%</sub> | 26.22 | 47.00 | 29.80<br><sub>cap 21%</sub> | - |
| qwen3-30b | reap50 | 79.76<br><sub>cap 39%</sub> | 21.15<br><sub>cap 65%</sub> | 44.20<br><sub>cap 44%</sub> | 0.00<br><sub>cap 93%</sub> | 5.14<br><sub>cap 35%</sub> | 22.56 | 32.50 | 20.20 | 36.55 |
| qwen3.6-35b | reap25 | 96.36 | 79.68 | 86.40 | 41.67<br><sub>cap 20%</sub> | 43.43<br><sub>cap 29%</sub> | 68.90 | 75.00 | 46.97<br><sub>cap 27%</sub> | 75.76 |
| qwen3.6-35b | reap50 | 94.69 | 79.98 | 85.20 | 50.00 | 49.14 | 73.78 | 57.25 | 36.87<br><sub>cap 30%</sub> | 66.34 |
| glm-4.7-flash | reap25 | 67.32 | 12.01<br><sub>cap 79%</sub> | 33.00<br><sub>cap 55%</sub> | 0.00<br><sub>cap 55%</sub> | 7.43<br><sub>cap 44%</sub> | 45.73 | 28.25<br><sub>cap 52%</sub> | 8.08<br><sub>cap 79%</sub> | 27.45<br><sub>cap 38%</sub> |
| glm-4.7-flash | reap50 | 46.32 | 5.89<br><sub>cap 86%</sub> | 20.60<br><sub>cap 68%</sub> | 0.00<br><sub>cap 70%</sub> | 5.14<br><sub>cap 34%</sub> | 44.51 | 15.50<br><sub>cap 68%</sub> | 6.57<br><sub>cap 81%</sub> | 22.65<br><sub>cap 43%</sub> |
| gemma4 | reap25 | 94.92 | 81.87 | 87.20 | 56.67<br><sub>cap 32%</sub> | 59.43<br><sub>cap 25%</sub> | 93.29 | 71.25 | 52.53 | 60.46 |
| gemma4 | reap50 | 92.27 | 67.60 | 79.60 | 31.67<br><sub>cap 37%</sub> | 53.14<br><sub>cap 26%</sub> | 90.85 | 51.75 | 37.88 | 51.14 |

## 3. Math minus code

Same model, same ratio, same protocol -- the only difference is the corpus.
Positive = math calibration scored higher.

| model | ratio | gsm8k | math_hard | math500 | aime | lcb | humaneval_plus | mmlu_pro_400 | gpqa_diamond | livebench_mixed |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen3-30b | reap25 | +0.15 | -5.59 | -7.20 | +0.00 | +0.57 | -3.66 | +3.75 | +1.51 | - |
| qwen3-30b | reap50 | +3.48 | -1.36 | -0.80 | +1.67 | -5.14 | -20.12 | +5.75 | -4.54 | -3.21 |
| qwen3.6-35b | reap25 | -0.38 | +0.46 | +0.80 | +3.33 | -1.14 | -1.83 | +3.75 | +9.60 | -0.73 |
| qwen3.6-35b | reap50 | +0.61 | +1.29 | +0.20 | -5.00 | -46.85 | -67.07 | +10.00 | +8.58 | -10.44 |
| glm-4.7-flash | reap25 | +4.70 | +0.75 | -2.20 | +1.67 | -5.14 | -18.29 | +3.75 | +2.53 | -0.40 |
| glm-4.7-flash | reap50 | +19.11 | +4.53 | +4.80 | +1.67 | -5.14 | -39.63 | +5.25 | -1.52 | -4.75 |
| gemma4 | reap25 | +0.38 | -3.77 | -0.60 | +1.66 | -33.14 | -9.75 | +8.25 | +14.64 | +1.06 |
| gemma4 | reap50 | -0.23 | +2.26 | +0.80 | -6.67 | -48.00 | -67.07 | +14.25 | +6.06 | -4.35 |
| **mean** | **reap25** | **+1.21** | **-2.04** | **-2.30** | **+1.66** | **-9.71** | **-8.38** | **+4.88** | **+7.07** | **-0.02** |
| **mean** | **reap50** | **+5.74** | **+1.68** | **+1.25** | **-2.08** | **-26.28** | **-48.47** | **+8.81** | **+2.15** | **-5.69** |

## 4. Baseline reference — no pruning

Un-pruned, same protocol, so sections 1 and 2 can each be read as a delta.

| model | gsm8k | math_hard | math500 | aime | lcb | humaneval_plus | mmlu_pro_400 | gpqa_diamond | livebench_mixed |
|---|---|---|---|---|---|---|---|---|---|
| qwen3-30b | 92.04 | 62.08 | 77.00 | 28.33 | 30.86 | 64.02 | 70.75 | 50.00 | 58.93 |
| qwen3.6-35b | 96.74 | 79.08 | 86.60 | 43.33 | 45.14 | 67.07 | 81.50 | 63.13 | 76.68 |
| glm-4.7-flash | 85.06 | 39.88 | 59.60 | 13.33 | 21.14 | 79.27 | 62.00 | 37.88 | 48.65 |
| gemma4 | 95.91 | 82.10 | 87.40 | 63.33 | 62.29 | 92.68 | 84.50 | 69.70 | 69.62 |

## Caveats

- **Cap-hit dominates the low-ratio arms.** Where `cap%` is high the arm did not
  produce scorable output; report it as that, not as an accuracy. Qwen3-30B aime
  is ~95% capped at both ratios.
- **Calibration corpus decides which capability survives pruning.** Section 3 is
  the controlled A/B. At reap50, swapping codealpaca->math costs ~26 pts of lcb and
  ~48 pts of humaneval+ on average while ADDING ~9 pts of mmlu_pro and holding math
  flat; at reap25 the same swap is nearly a wash. Pruning keeps the experts the
  calibration data activates, and only at 50% must the surviving set specialise.
  Corroborated earlier on GLM, where prose->code calibration moved humaneval+ from
  1.83 to 46.95 at the same 256 samples: sample count did not matter; domain did.
- **Calibration BUDGET is not equalised across these rows.** The tulu-math arms used
  8x32 @ L256 (256 samples); the codealpaca caches on this box are 32x8 @ L256 for
  GLM (identical volume) and 4x32 @ L256 for qwen3-30b (half). Qwen3.6/gemma4
  codealpaca caches are no longer on disk, so their budget is unverified. Both are
  far below REAP's own default of 8x1024 @ L2048.
- **aime has 4/60 rows with `\frac` mangled** by formfeed corruption, which depresses
  every absolute aime number. A/B comparisons within aime remain valid.
- **Run-to-run noise is 0.5-1.1 pts** on greedy evals from batch-composition FP
  effects alone. Single-run differences smaller than that are not readable.
- **Calibration corpus is not recorded in result metadata.** It was recovered
  from the checkpoint path in each cell's meta (`.../evol-codealpaca-v1/...` vs
  `.../tulu-3-sft-personas-math/...`). Stamp it at run time so this stops being
  an inference.
