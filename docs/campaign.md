# Running the experiment campaign

`scripts/campaign.py` keeps a prioritized list of federated runs moving on
mscluster, where jobs hit wall-time limits, nodes die, and some GPUs are
unusable. The campaign file lists runs in priority order; the driver is
stateless between ticks apart from `logs/campaign/<name>/state.json`.

```bash
python3 scripts/campaign.py tick   configs/campaign/deadline_2026_10.yaml
python3 scripts/campaign.py status configs/campaign/deadline_2026_10.yaml
```

Run `tick` every few minutes from cron on the login node:

```bash
(crontab -l 2>/dev/null; echo '*/5 * * * * cd $HOME/fl-brats && bash -lc "python3 scripts/campaign.py tick configs/campaign/deadline_2026_10.yaml" >> logs/campaign/cron.log 2>&1') | crontab -
```

## What a tick does

1. Cancels jobs of ended attempts that are still queued (a failed `scancel`
   would otherwise hold per-user job slots).
2. For each active run: cancels it when a client died while the others wait,
   when the server died, when all clients ended before `final.pt` exists, or
   when no round finished for `stall_minutes`. A server finalizing after its
   clients ended gets `finalize_minutes`.
3. Excludes nodes whose job-start GPU check failed (`FL_BRATS_GPU_UNAVAILABLE`).
4. Marks runs done when `state.json` says `completed` or `early_stopped`, and
   resubmits the rest from their latest round checkpoint, highest priority
   first, within each partition's `max_active`. A run stuck for
   `max_attempts_without_progress` attempts is marked `stuck`.
5. Submits the post-training job for finished runs (periodic full metrics
   every 10th round across the whole resume chain, then the test report of the
   selected checkpoint).

Events are appended to `logs/campaign/<name>/events.log`.

## Faithful resumes

A resume continues from the latest round checkpoint in a new run directory.
The server restores the model, experiment state and strategy state (FedAMM's
global prototype library); each client restores its optimizer moments and
compression residuals from `client_state/`. A resumed run's round checkpoints
are split across its directories; the post-training job accepts the chain.

## Where results are

| What | Where (newest run directory) |
|---|---|
| Per-round held-out validation (selection) | `metrics/validation_rounds.jsonl` |
| Per-round timings, client fits | `metrics/rounds.jsonl`, `metrics/clients.jsonl` |
| Best/final checkpoint, full metrics and robustness | `validation_evaluation/{best,final}/` |
| Every 10th round: full metrics, robustness | `periodic_evaluation/` |
| Test cohort, selected checkpoint | `final_evaluation/` |
| Run summary | `summary.json`, `state.json` |

Runs finished before the held-out evaluation refactor (FedAvg baselines, k3,
k5) are revalidated with `slurm/_shared/revalidate_federated_runs.sbatch`
into `revalidation/`.

## Performance notes

Preprocessed volumes are cached on each node's local SSD
(`data.cache_location: node`); the first round on a node builds the cache.
DataLoader workers are spawned (forked workers abort under Flower's gRPC
threads) and reseeded per worker. HD95 runs in a process pool
(`evaluation.hd95_workers`). See `docs/cluster_nodes.md` for partition limits
and broken nodes.
