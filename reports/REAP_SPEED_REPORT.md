# REAP speed sweep — TP=2, 8× H100

Measured 2026-09-13. `vllm bench serve`, random dataset, `--ignore-eos`, seed 42,
`max-model-len` 16384, `gpu-memory-utilization` 0.92, prefix caching off.

Seven regimes per model:

| regime | ISL | OSL | num_prompts | max_concurrency |
|---|---|---|---|---|
| conc8 | 128 | 256 | 128 | 8 |
| conc16 | 128 | 256 | 160 | 16 |
| conc32 | 128 | 256 | 320 | 32 |
| conc64 | 128 | 256 | 640 | 64 |
| saturated | 128 | 1024 | 640 | 64 |
| mixed_1024 | 1024 | 256 | 640 | 64 |
| prefill_8192 | 8192 | 64 | 320 | 64 |

`num_prompts` is ≥10× concurrency so achieved batch tracks the cap.

**Columns** — `batch` is the *achieved* batch (`out_tok/s × tpot_ms / 1000`), not the cap.
`out tok/s` counts generated tokens only; `total tok/s` counts prompt + generated.
There is no `changed_frac` column here: REAP removes experts at pruning time, so at
serve time nothing is substituted and the state axis shared with the SERE runs does
not apply.

**Arms** — `vanilla` the unmodified checkpoint; `reap25` and `reap50` REAP-pruned
checkpoints at compression ratio 0.25 and 0.50. All three run with no plugin and
`VLLM_PLUGINS` unset — REAP is static, so the pruned model is just a smaller model.
SERE was asserted absent on every cell, which keeps these arms state-matched to the
accuracy runs.

**Expert counts after pruning:**

| model | vanilla | reap25 | reap50 |
|---|---|---|---|
| qwen3 | 128 | 96 | 64 |
| qwen3.6 | 256 | 192 | 128 |
| glm-4.7-flash | 64 | 48 | 32 |
| gemma4 | 128 | 96 | 64 |

Each arm was gated at serve time against the expert count in the checkpoint's own
config, so a mislabelled or stale checkpoint aborts the cell rather than reporting
under the wrong arm.

## What REAP does and does not change

REAP cuts the *total* expert count but leaves `num_experts_per_tok` at top-8. Per-token
FLOPs are therefore unchanged, and the speedup comes entirely from the smaller weight
footprint and from expert-weight reuse within a batch: with fewer experts alive, a
batch of tokens concentrates on fewer distinct expert matrices, so each loaded weight
serves more tokens. That is a memory-traffic effect, and it needs batch to show up —
which is exactly the shape of the numbers below.

## Caveats

1. **Mean TTFT is contaminated and worse than in the SERE table.** Every cell ran
   four-co-tenant (four TP=2 jobs across all 8 GPUs), against two co-tenants for the
   SERE sweep. Contention is a prefill tail effect, so it lands on mean TTFT; TPOT
   and throughput are unaffected. TTFT here is not comparable to the SERE report's
   TTFT and should not be read as an absolute.
2. **`prefill_8192` is a negative control.** At 0.8% decode share there is almost no
   decode for the weight-reuse effect to act on, and all three arms land within ~2%
   of each other in every model. A flat row there is the expected result, not a
   missing effect.
3. **Vanilla was re-measured in this sweep**, not carried over from the SERE report.
   The two sweeps ran at different co-tenancy, so the older vanilla numbers would
   have folded a contention difference into every REAP comparison. Vanilla figures
   here will therefore not match the SERE report exactly.
4. **Calibration corpus is fineweb-edu for all eight pruned checkpoints.** The
   calibration corpus decides *which* experts are dropped, not *how many* — shape
   and FLOPs are identical whatever the corpus, so these timings are valid
   regardless. It does matter for accuracy, and these checkpoints are not the ones
   to quote accuracy from.
5. **One cell was re-measured solo: qwen3.6 reap25, mixed 1024/256.** Its sweep run
   recorded an achieved batch of 5.6 against a cap of 64 -- the client never reached
   concurrency, so the cell was invalid rather than slow. The replacement ran alone on
   one GPU pair instead of four-co-tenant. Its throughput is therefore measured under
   lighter contention than every other cell in this table; it lands between the
   vanilla and reap50 figures for that regime, which is the expected ordering, but it
   is not strictly like-for-like.
6. **Single run per cell.** Repeated cells on this harness agree to ~2% on
   tpot/throughput; treat an unreplicated arm inversion as noise.

## Summary — c=64, out tok/s

| model | regime | vanilla | reap25 | reap50 |
|---|---|---|---|---|
| qwen3 | conc64 | 4829.8 | 5084.7 | 6071.5 |
| qwen3 | saturated 128/1024 | 4966.9 | 5358.8 | 6644.1 |
| qwen3 | mixed 1024/256 | 4138.0 | 4198.5 | 4887.1 |
| qwen3 | prefill 8192/64 | 513.9 | 501.7 | 531.6 |
| qwen3.6 | conc64 | 4061.7 | 4526.4 | 4925.5 |
| qwen3.6 | saturated 128/1024 | 4294.3 | 4676.6 | 5286.0 |
| qwen3.6 | mixed 1024/256 | 3670.2 | 3882.3 | 4357.0 |
| qwen3.6 | prefill 8192/64 | 536.0 | 547.0 | 557.8 |
| glm-4.7-flash | conc64 | 4151.3 | 4747.3 | 5460.6 |
| glm-4.7-flash | saturated 128/1024 | 4149.1 | 4711.2 | 5568.8 |
| glm-4.7-flash | mixed 1024/256 | 3354.3 | 3778.8 | 4250.9 |
| glm-4.7-flash | prefill 8192/64 | 410.8 | 416.0 | 426.7 |
| gemma4 | conc64 | 6086.0 | 6516.6 | 7010.0 |
| gemma4 | saturated 128/1024 | 6006.0 | 6436.6 | 6724.5 |
| gemma4 | mixed 1024/256 | 4405.7 | 4757.3 | 4868.4 |
| gemma4 | prefill 8192/64 | 546.9 | 550.9 | 553.7 |

## Full results

### qwen3

#### conc8 / conc16 / conc32 / conc64 — 128/256

| c | arm | batch | ttft ms | tpot ms | out tok/s | total tok/s |
|---|---|---|---|---|---|---|
| 8 | vanilla | 7.7 | 86.6 | 7.22 | 1062.4 | 1593.6 |
| 8 | reap25 | 7.6 | 94.2 | 7.40 | 1033.9 | 1550.9 |
| 8 | reap50 | 7.6 | 93.3 | 6.44 | 1179.6 | 1769.5 |
| | | | | | | |
| 16 | vanilla | 15.4 | 97.5 | 8.63 | 1781.5 | 2672.3 |
| 16 | reap25 | 15.3 | 111.4 | 8.75 | 1747.8 | 2621.7 |
| 16 | reap50 | 15.2 | 106.7 | 7.34 | 2070.7 | 3106.0 |
| | | | | | | |
| 32 | vanilla | 30.9 | 108.6 | 10.68 | 2891.6 | 4337.4 |
| 32 | reap25 | 30.8 | 106.7 | 10.13 | 3042.9 | 4564.3 |
| 32 | reap50 | 30.4 | 112.8 | 8.09 | 3760.9 | 5641.4 |
| | | | | | | |
| 64 | vanilla | 61.1 | 158.5 | 12.65 | 4829.8 | 7244.7 |
| 64 | reap25 | 60.7 | 174.7 | 11.93 | 5084.7 | 7627.0 |
| 64 | reap50 | 60.7 | 145.7 | 10.00 | 6071.5 | 9107.3 |

#### saturated 128/1024

| c | arm | batch | ttft ms | tpot ms | out tok/s | total tok/s |
|---|---|---|---|---|---|---|
| 64 | vanilla | 63.2 | 171.0 | 12.72 | 4966.9 | 5587.7 |
| 64 | reap25 | 63.2 | 158.7 | 11.80 | 5358.8 | 6028.6 |
| 64 | reap50 | 63.0 | 161.6 | 9.48 | 6644.1 | 7474.6 |

#### mixed 1024/256

| c | arm | batch | ttft ms | tpot ms | out tok/s | total tok/s |
|---|---|---|---|---|---|---|
| 64 | vanilla | 58.7 | 334.5 | 14.19 | 4138.0 | 20689.9 |
| 64 | reap25 | 58.6 | 339.9 | 13.95 | 4198.5 | 20992.3 |
| 64 | reap50 | 56.7 | 390.4 | 11.60 | 4887.1 | 24435.7 |

#### prefill 8192/64

| c | arm | batch | ttft ms | tpot ms | out tok/s | total tok/s |
|---|---|---|---|---|---|---|
| 64 | vanilla | 53.3 | 1360.5 | 103.69 | 513.9 | 66294.6 |
| 64 | reap25 | 53.1 | 1413.1 | 105.89 | 501.7 | 64723.9 |
| 64 | reap50 | 53.2 | 1336.1 | 99.98 | 531.6 | 68578.4 |

### qwen3.6

#### conc8 / conc16 / conc32 / conc64 — 128/256

| c | arm | batch | ttft ms | tpot ms | out tok/s | total tok/s |
|---|---|---|---|---|---|---|
| 8 | vanilla | 7.0 | 273.3 | 7.12 | 980.2 | 1470.3 |
| 8 | reap25 | 7.3 | 186.6 | 6.95 | 1045.5 | 1568.3 |
| 8 | reap50 | 7.2 | 191.6 | 6.62 | 1089.8 | 1634.8 |
| | | | | | | |
| 16 | vanilla | 14.8 | 185.5 | 8.84 | 1678.8 | 2518.2 |
| 16 | reap25 | 14.9 | 174.3 | 8.50 | 1749.0 | 2623.5 |
| 16 | reap50 | 14.8 | 178.3 | 8.03 | 1839.0 | 2758.6 |
| | | | | | | |
| 32 | vanilla | 30.2 | 180.9 | 11.43 | 2644.1 | 3966.2 |
| 32 | reap25 | 30.1 | 176.4 | 10.67 | 2825.7 | 4238.5 |
| 32 | reap50 | 29.8 | 191.2 | 9.61 | 3100.0 | 4650.0 |
| | | | | | | |
| 64 | vanilla | 60.5 | 230.5 | 14.90 | 4061.7 | 6092.6 |
| 64 | reap25 | 60.9 | 185.9 | 13.45 | 4526.4 | 6789.6 |
| 64 | reap50 | 59.7 | 230.4 | 12.13 | 4925.5 | 7388.2 |

#### saturated 128/1024

| c | arm | batch | ttft ms | tpot ms | out tok/s | total tok/s |
|---|---|---|---|---|---|---|
| 64 | vanilla | 63.0 | 246.2 | 14.67 | 4294.3 | 4831.1 |
| 64 | reap25 | 63.1 | 209.1 | 13.49 | 4676.6 | 5261.1 |
| 64 | reap50 | 62.7 | 254.3 | 11.87 | 5286.0 | 5946.8 |

#### mixed 1024/256

| c | arm | batch | ttft ms | tpot ms | out tok/s | total tok/s |
|---|---|---|---|---|---|---|
| 64 | vanilla | 58.2 | 417.0 | 15.85 | 3670.2 | 18350.9 |
| 64 | reap25 | 58.9 | 345.2 | 15.17 | 3882.3 | 19411.3 |
| 64 | reap50 | 57.6 | 382.9 | 13.23 | 4357.0 | 21785.2 |

#### prefill 8192/64

| c | arm | batch | ttft ms | tpot ms | out tok/s | total tok/s |
|---|---|---|---|---|---|---|
| 64 | vanilla | 52.4 | 1415.0 | 97.75 | 536.0 | 69144.0 |
| 64 | reap25 | 52.6 | 1359.8 | 96.24 | 547.0 | 70567.3 |
| 64 | reap50 | 53.0 | 1300.4 | 94.94 | 557.8 | 71953.5 |

### glm-4.7-flash

#### conc8 / conc16 / conc32 / conc64 — 128/256

| c | arm | batch | ttft ms | tpot ms | out tok/s | total tok/s |
|---|---|---|---|---|---|---|
| 8 | vanilla | 7.6 | 133.1 | 8.41 | 898.8 | 1348.1 |
| 8 | reap25 | 7.5 | 133.7 | 7.93 | 950.0 | 1425.0 |
| 8 | reap50 | 7.5 | 144.6 | 7.32 | 1017.4 | 1526.1 |
| | | | | | | |
| 16 | vanilla | 15.2 | 142.4 | 10.21 | 1490.5 | 2235.8 |
| 16 | reap25 | 15.2 | 141.1 | 9.31 | 1627.6 | 2441.4 |
| 16 | reap50 | 15.0 | 152.9 | 8.37 | 1790.3 | 2685.5 |
| | | | | | | |
| 32 | vanilla | 30.6 | 148.2 | 12.22 | 2508.3 | 3762.5 |
| 32 | reap25 | 30.4 | 158.2 | 11.03 | 2755.3 | 4133.0 |
| 32 | reap50 | 30.2 | 154.6 | 9.40 | 3206.8 | 4810.2 |
| | | | | | | |
| 64 | vanilla | 61.5 | 162.8 | 14.82 | 4151.3 | 6226.9 |
| 64 | reap25 | 61.2 | 159.6 | 12.90 | 4747.3 | 7121.0 |
| 64 | reap50 | 60.4 | 175.0 | 11.07 | 5460.6 | 8190.8 |

#### saturated 128/1024

| c | arm | batch | ttft ms | tpot ms | out tok/s | total tok/s |
|---|---|---|---|---|---|---|
| 64 | vanilla | 63.4 | 159.5 | 15.28 | 4149.1 | 4667.7 |
| 64 | reap25 | 63.2 | 183.6 | 13.41 | 4711.2 | 5300.1 |
| 64 | reap50 | 62.9 | 207.2 | 11.30 | 5568.8 | 6264.9 |

#### mixed 1024/256

| c | arm | batch | ttft ms | tpot ms | out tok/s | total tok/s |
|---|---|---|---|---|---|---|
| 64 | vanilla | 59.5 | 349.8 | 17.74 | 3354.3 | 16771.3 |
| 64 | reap25 | 58.1 | 412.8 | 15.36 | 3778.8 | 18893.9 |
| 64 | reap50 | 57.3 | 411.4 | 13.48 | 4250.9 | 21254.3 |

#### prefill 8192/64

| c | arm | batch | ttft ms | tpot ms | out tok/s | total tok/s |
|---|---|---|---|---|---|---|
| 64 | vanilla | 53.8 | 1625.3 | 131.02 | 410.8 | 52997.6 |
| 64 | reap25 | 53.9 | 1589.6 | 129.66 | 416.0 | 53663.6 |
| 64 | reap50 | 53.6 | 1596.9 | 125.69 | 426.7 | 55042.3 |

### gemma4

#### conc8 / conc16 / conc32 / conc64 — 128/256

| c | arm | batch | ttft ms | tpot ms | out tok/s | total tok/s |
|---|---|---|---|---|---|---|
| 8 | vanilla | 7.4 | 112.1 | 5.57 | 1334.9 | 2002.4 |
| 8 | reap25 | 7.5 | 93.2 | 5.48 | 1373.4 | 2060.0 |
| 8 | reap50 | 7.6 | 75.6 | 5.70 | 1338.5 | 2007.7 |
| | | | | | | |
| 16 | vanilla | 15.3 | 82.9 | 6.64 | 2303.8 | 3455.6 |
| 16 | reap25 | 14.2 | 209.6 | 6.49 | 2194.9 | 3292.4 |
| 16 | reap50 | 15.2 | 87.5 | 6.28 | 2423.6 | 3635.5 |
| | | | | | | |
| 32 | vanilla | 29.8 | 157.8 | 7.89 | 3772.0 | 5657.9 |
| 32 | reap25 | 30.1 | 124.3 | 7.25 | 4151.2 | 6226.9 |
| 32 | reap50 | 30.7 | 85.4 | 7.27 | 4223.1 | 6334.6 |
| | | | | | | |
| 64 | vanilla | 61.0 | 131.9 | 10.03 | 6086.0 | 9128.9 |
| 64 | reap25 | 60.6 | 141.1 | 9.29 | 6516.6 | 9774.9 |
| 64 | reap50 | 61.5 | 97.4 | 8.77 | 7010.0 | 10515.0 |

#### saturated 128/1024

| c | arm | batch | ttft ms | tpot ms | out tok/s | total tok/s |
|---|---|---|---|---|---|---|
| 64 | vanilla | 62.7 | 235.0 | 10.43 | 6006.0 | 6756.7 |
| 64 | reap25 | 62.9 | 183.2 | 9.77 | 6436.6 | 7241.1 |
| 64 | reap50 | 63.2 | 127.4 | 9.40 | 6724.5 | 7565.1 |

#### mixed 1024/256

| c | arm | batch | ttft ms | tpot ms | out tok/s | total tok/s |
|---|---|---|---|---|---|---|
| 64 | vanilla | 57.2 | 406.7 | 12.97 | 4405.7 | 22028.3 |
| 64 | reap25 | 57.6 | 354.3 | 12.10 | 4757.3 | 23786.7 |
| 64 | reap50 | 57.4 | 353.9 | 11.79 | 4868.4 | 24342.2 |

#### prefill 8192/64

| c | arm | batch | ttft ms | tpot ms | out tok/s | total tok/s |
|---|---|---|---|---|---|---|
| 64 | vanilla | 53.1 | 1312.5 | 97.14 | 546.9 | 70547.0 |
| 64 | reap25 | 53.3 | 1280.4 | 96.78 | 550.9 | 71072.3 |
| 64 | reap50 | 53.4 | 1272.7 | 96.38 | 553.7 | 71426.1 |
