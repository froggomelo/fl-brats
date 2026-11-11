# Running the experiment campaign

`scripts/monitoring/campaign.py` keeps a prioritized list of federated runs moving on
mscluster, where jobs hit wall-time limits, nodes die, and some GPUs are
unusable. The campaign file lists runs in priority order; the driver is
stateless between ticks apart from `logs/monitoring/campaign/<name>/state.json`.

```bash
python3 scripts/monitoring/campaign.py tick   configs/campaign/deadline_2026_10.yaml
python3 scripts/monitoring/campaign.py status configs/campaign/deadline_2026_10.yaml
```

Run `tick` every few minutes from cron on the login node:

```bash
(crontab -l 2>/dev/null; echo '*/5 * * * * cd $HOME/fl-brats && bash -lc "python3 scripts/monitoring/campaign.py tick configs/campaign/deadline_2026_10.yaml" >> logs/monitoring/campaign/cron.log 2>&1') | crontab -
```

## Where files go

`outputs/` and `logs/` follow the Experiments.md layout, mirroring `configs/`
and `scripts/`: one folder per experiment family (with `client_sweeps/`,
`local_epochs_sweeps/`, `mini_batch_sweeps/`, `modality_sweep/` under
`fedavg_train_configs_uneven/` and `sam1/`, `sam2/` under
`model_agnostic_uneven/`), plus `data_preprocessing/`, `environment/` and
`monitoring/` (campaign state, revalidation). A run's folder is
`<family>/<variant>_seed_<seed>_<timestamp>` in both. Smoke tests go under
`smoke/<family>/`, and runs from before this layout under `<family>/legacy/`.

## What a tick does

1. Cancels jobs of ended attempts that are still queued (a failed `scancel`
   would otherwise hold per-user job slots).
2. For each active run: cancels it when a client died while the others wait,
   when the server died, when all clients ended before `final.pt` exists, or
   when no round finished for `stall_minutes`. A server finalizing after its
   clients ended gets `finalize_minutes`.
3. Excludes nodes whose job-start GPU check failed (`FL_BRATS_GPU_UNAVAILABLE`).
4. Marks runs done when `state.json` says `completed` or `early_stopped` (a
   centralized run when `summary.json` exists), and resubmits the rest from
   their latest round checkpoint (a centralized run from its `last.pt`). A run
   stuck for `max_attempts_without_progress` attempts is marked `stuck`.
5. Sets aside finished attempts made on a partition the run may no longer use
   (`archived_attempts`): the run is never resumed from, evaluated from, or
   counted done by them, and restarts on its allowed partition. Every final
   result must come from one GPU model.
6. Submits runs in strict priority order: it stops at the first pending run
   that does not fit, so a smaller later run never takes the slots the next
   one is waiting for. A run fits when all its jobs (server + clients, or one
   centralized job) can hold a slot at once: training runs stay within the
   partition's `max_jobs`, counted from the jobs they still hold in the queue,
   and all our jobs within `job_limit` (the per-user limit; held and
   dependency-waiting jobs do not count). `max_active` caps concurrent runs.
7. Submits the post-training job for finished runs (periodic full metrics
   every 10th round across the whole resume chain, then the test report of the
   selected checkpoint), within `job_limit`.

Events are appended to `logs/monitoring/campaign/<name>/events.log`.

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

The August runs trained on bigbatch before the held-out evaluation refactor
(FedAvg baselines, k3, k5, centralized) are retrained on batch in this
campaign. Their revalidations (`revalidation/`) remain only as a fallback.

Centralized runs write `metrics.jsonl` (one line per epoch), `checkpoints/
{last,best}.pt` and `summary.json` (test metrics of the best epoch).

## Performance notes

Preprocessed volumes are cached on each node's local SSD
(`data.cache_location: node`); the first round on a node builds the cache.
DataLoader workers are spawned (forked workers abort under Flower's gRPC
threads) and reseeded per worker. HD95 runs on the GPU when evaluation does,
and in a process pool (`evaluation.hd95_workers`) on CPU. Post-training
evaluation of a missing-modality run (15 checkpoints, 16 modality passes each)
takes about 4 h on an RTX 3060. Clients re-list the run directory while waiting
for the server's address, because NFS can cache a failed lookup for minutes.
Jobs set `TORCH_DISABLE_NATIVE_JIT=1` (no C compiler for Triton in the image).
See [`logs/environment/cluster_nodes.md`](../../environment/cluster_nodes.md) for partition limits and broken nodes, and
[`docs/models.md`](../../../docs/models.md) for the SAM runs.
