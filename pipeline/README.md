# REAP calibrate → prune → evaluate pipeline

End-to-end REAP campaigns: calibrate on a corpus, prune to a ratio, evaluate all
nine accuracy tasks, verify, publish. Four models × two ratios × nine tasks = 72
cells per corpus, unattended and restartable.

## Why it is shaped this way

Four GPUs = **one prune (2 GPUs) or two eval lanes (2×2)**, never both on the same
pair. A campaign therefore interleaves rather than batches: while one model's cells
drain on one pair, the *next* model prunes on the other. `gpu_alloc.sh` is what
makes that safe — it hands a command the first pair that is both unlocked and
idle, waiting when neither is, and takes the same pair lock the eval workers take.

Nothing here assumes it is the only tenant of the box. A neighbour holding all
four cards is normal and the correct response is to wait, not to fail.

## Layout

    pipeline/
      reap.env.example    machine-local paths -- THE ONLY per-machine file
      campaigns/*.env     one per calibration corpus (dataset, budget, arm suffix)
      gpu_alloc.sh        allocate a free GPU pair, wait if none
      prune_model.sh      prune one model at the requested ratios on $GPUS
      run.sh              the campaign driver: prune -> queue -> drain -> free disk
      launch_reap.sh      one cell = one checkpoint, one batch size, N tasks
      verify_reap.sh      gate a finished cell (row counts, score, REAP purity)
      watchdog.sh         kill wedged cells and over-ceiling prunes
      start.sh            idempotent entry point (cron/tmux safe)
      test_gpu_alloc.sh   self-check for the allocator
      scheduler/          worker.sh + supervisor.sh (queue, verify, push)
    tools/                reap_coverage.py (what exists), reap_report.py (what it says)
    reports/              accuracy + speed reports
    results/              per-corpus cell summaries (JSON only, no generations)

## Set up on a new machine

    cp pipeline/reap.env.example pipeline/reap.env   # edit the paths
    ./pipeline/test_gpu_alloc.sh                     # allocator self-check, no GPU needed

`reap.env` is gitignored. If you ever need to edit a `.sh` to move a path, add a
knob to `reap.env.example` instead — that is the invariant that keeps this portable.

## Run a campaign

    export CAMPAIGN=$REAP_STATE/fineweb
    mkdir -p "$CAMPAIGN" && cp pipeline/campaigns/fineweb.env "$CAMPAIGN/campaign.env"
    ./pipeline/start.sh "$CAMPAIGN"

`start.sh` is idempotent: a no-op while the sweep is healthy, a restart when it is
not. For unattended running, put it in cron:

    */10 * * * * /path/to/reap/pipeline/start.sh /path/to/state/fineweb >> .../start.log 2>&1
    @reboot sleep 150 && /path/to/reap/pipeline/start.sh /path/to/state/fineweb >> .../start.log 2>&1

## Recovery layers

Everything resumes from **what is on disk**, never from how far a previous run got.

| failure | covered by |
|---|---|
| a worker dies | `supervisor.sh` respawns it |
| a worker wedges (log stops, GPUs busy) | `watchdog.sh` kills it at 45 min stall |
| a prune hangs | `watchdog.sh` 45 min ceiling |
| `run.sh` dies | cron `*/10` re-runs `start.sh` |
| host reboots | cron `@reboot` |
| a cell fails transiently | `run.sh` requeues while the missing set still shrinks |
| GPUs occupied | `gpu_alloc.sh` waits and logs the holders |

A model whose cells all have a `summary.json` is skipped entirely; a ratio whose
checkpoint weights exist is not re-pruned; the observation cache survives the
weight deletion, so a re-prune is minutes.

## Adding a corpus

Write `pipeline/campaigns/<name>.env`. Nothing else changes:

    DATASET=HuggingFaceFW/fineweb-edu   # HF id passed to prune.py
    DS=fineweb-edu                      # artifacts/<model>/<DS>/ subdirectory
    SUFFIX=fw256                        # arms: reap25_$SUFFIX / reap50_$SUFFIX
    SWEEP=reap_fineweb                  # publishing key
    CALIB_BS=8
    CALIB_NB=32                         # 8x32 = 256 calibration samples @ L256
