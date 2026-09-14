# Handoff — continuing the REAP campaigns on another machine

Everything needed to finish the outstanding work. Read this first; `STATUS.md`
has the numbers, `pipeline/README.md` the design, `pipeline/GOTCHAS.md` the
failure modes.

---

## 1. What is done, what is left

4 models × 2 ratios × 9 tasks = **72 cells per calibration corpus**.
Protocol for every cell: 0-shot, greedy, `max_tokens` 4096, offline batch-pinned
B=64, TP=2, seed 42.

| corpus | cells | state |
|---|---|---|
| codealpaca | 72/72 | done |
| tulu-3-sft-personas-math | 72/72 | done |
| **fineweb-edu** | **36/72** | **your job** |

Remaining: **qwen3.6-35b** and **gemma-4**, both ratios, all nine tasks — 36 cells.
qwen3-30b and glm-4.7-flash are already complete on fineweb.

The sweep was stopped cleanly. Nothing partial is on disk that would confuse a
restart: `run.sh` decides what to do from the `summary.json` files that exist, so
it will queue exactly the 36 missing cells and nothing else.

---

## 2. Setup

```bash
git clone https://github.com/hqta1110/reap.git && cd reap
cp pipeline/reap.env.example pipeline/reap.env     # edit the paths for this box
./calibration/install_caches.sh
./pipeline/preflight.sh
```

`preflight.sh` is the honest gate. It checks thirteen things and prints a fix for
each. **Do not skip it** — a missing venv or an absent base model surfaces forty
minutes into a prune otherwise. Expect:

```
13 ok, 0 missing
ready: ./pipeline/start.sh $REAP_STATE/<corpus>
```

### What this repo does NOT contain

The REAP half is here. The eval half is external and is most of the work on a
bare machine:

| dependency | where / note |
|---|---|
| eval harness | `hqta1110/moe-eval-unified`, branch **`feat/offline-batch-pinned-eval`** (not main) |
| results repo | `hqta1110/moe-eval-results` — set `RESULTS_REPO`, the worker pushes landed cells there |
| venv: qwen3-30b, qwen3.6 | vLLM **0.18.1** |
| venv: glm-4.7-flash | vLLM **0.18.1** |
| venv: gemma-4 | vLLM **0.29** — deliberately different, do not "fix" this |
| prune venv | transformers **>= 5.16** (Qwen3-MoE + Gemma-4 support) |
| base weights | **~232 GB** of HF snapshots for the four models |

That last row is the one that bites: if this box has no warm HF cache and no
internet, stop and fix that before anything else.

---

## 3. Run it

```bash
export CAMPAIGN=$REAP_STATE/fineweb
mkdir -p "$CAMPAIGN" && cp pipeline/campaigns/fineweb.env "$CAMPAIGN/campaign.env"
./pipeline/start.sh "$CAMPAIGN"
```

Unattended (survives a dead driver and a reboot):

```cron
*/10 * * * * /path/to/reap/pipeline/start.sh /path/to/state/fineweb >> /path/to/state/fineweb/logs/start.log 2>&1
@reboot sleep 150 && /path/to/reap/pipeline/start.sh /path/to/state/fineweb >> /path/to/state/fineweb/logs/start.log 2>&1
```

`start.sh` is idempotent — a no-op while healthy, a restart when not. Everything
runs in tmux session `moe`.

### Cost

| step | cost | why |
|---|---|---|
| qwen3.6 prune, both ratios | **minutes** | observation cache is committed; this is only a checkpoint write |
| gemma-4 prune, both ratios | **~30–45 min** | no fineweb cache exists anywhere — real calibration |
| 36 eval cells | **~2.5 h** on 4×H100 | measured: mean 16.8 min/cell, two lanes |

Observations are ratio-independent: 0.25 builds the cache, 0.50 reuses it, so the
second prune of each model is always short.

---

## 4. Watching it

```bash
tmux attach -t moe                                  # windows: fineweb, wd-*, w01, w23
tail -f $CAMPAIGN/logs/run.log                      # campaign driver
python3 tools/reap_coverage.py                      # what has landed
REAP_RESULT_ROOTS=$CAMPAIGN/results python3 tools/reap_coverage.py   # live tree
```

Healthy looks like both GPU pairs busy and cells closing with `rc=0` / `VERDICT=PASS`:

```
--- gpu_alloc: acquired pair 0,1 for: prune_model.sh qwen36-35b ...
=== GATE: expert count (expect 192 = 256 x (1-0.25))
PRUNE_DONE dir=.../Qwen3.6-35B-A3B/fineweb-edu/pruned_models/reap-...-0.25
[w01] START qwen36-35b reap25_fw256 B=64 [gsm8k math_hard math500 aime] on 0,1
[w01]   VERDICT=PASS
```

### If it looks stuck

Check for a **co-tenant** before debugging the code — another user holding the
GPUs is normal and the scheduler correctly waits rather than failing:

```bash
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
grep 'gpu_alloc: no free pair' $CAMPAIGN/logs/run.log | tail -3
```

Recovery is layered and automatic: supervisor respawns a dead worker, watchdog
kills a wedged cell at a 45-minute log stall and an over-ceiling prune, cron
restarts a dead driver, `@reboot` survives a power cycle. You should not need to
intervene; if you do, `pipeline/GOTCHAS.md` lists the ten failure modes that have
actually happened here.

---

## 5. Reading the results

**Always read a score next to its cap-hit rate.** `hit_length_cap` above ~20%
means the number measures *termination*, not ability — a pruned model that never
stops generating scores ~0 on everything, and that is a property of the arm, not
a broken cell. `verify_reap.sh` reports it per task and deliberately does not gate
on it.

Several landed fineweb rows are already truncation-limited (glm-4.7 reap25
humaneval: 1.22 at **92.7%** cap). Of the two models you are about to run,
qwen3.6 and gemma-4 had much lower cap rates on the other corpora — so **they are
the rows that will actually carry the three-way corpus comparison.** That is the
main reason finishing this matters.

### Known confound, not yet resolved

codealpaca checkpoints were calibrated at **128 samples**, tulu-math and fineweb
at **256**. So code-vs-{math,fineweb} carries a calibration-*budget* difference on
top of the corpus difference. Both are far below REAP's 8192-sample default so the
effect is probably small, but it is not zero and `reports/REAP_ACCURACY_REPORT.md`
does not mention it. Re-running codealpaca at 8×32 would remove it.

---

## 6. When the 36 cells land

```bash
python3 tools/reap_coverage.py     # expect 72/72 on all three corpora
python3 tools/reap_report.py       # regenerates the accuracy report
```

Then update `reports/REAP_ACCURACY_REPORT.md` with the fineweb section (it is
currently math-vs-code only) and add the budget caveat above.

Speed is already complete — `reports/REAP_SPEED_REPORT.md`, 4 models × 3 arms × 7
regimes. Note those checkpoints were calibrated on **fineweb-edu**, so pair the
speed table with the fineweb accuracy rows rather than the math or code ones.
