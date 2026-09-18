# Training sweep relaunch record — 2026-08-31

This record is based on the launcher output from the resume attempt gated by
`afterany:48356`. It records **submission status**, not final training outcome.
Confirm completed/failed execution with `sacct` and the run logs before
relaunching anything listed as successfully submitted.

## Successfully submitted as complete job triplets

| Experiment | Run ID | Server | Clients | Cleanup | Initial dependency |
|---|---|---:|---:|---:|---:|
| `k8` | `k8_seed_42_20260831_081446_611357_1` | 48357 | 48358 | 48359 | 48356 |
| `e5` | `e5_seed_42_20260831_081447_445754` | 48360 | 48361 | 48362 | 48359 |
| `e10` | `e10_seed_42_20260831_081447_982161` | 48363 | 48364 | 48365 | 48362 |
| `b4` | `b4_seed_42_20260831_081448_320317` | 48366 | 48367 | 48368 | 48365 |

These submissions formed a serial chain. Successful submission does not prove
that training completed successfully.

## Failed or incomplete submissions

### `b8`

- Run ID: `b8_seed_42_20260831_081448_597481`
- The server `sbatch` call succeeded, but the client-array submission failed
  with `QOSMaxSubmitJobPerUserLimit`.
- No cleanup job was created and the launcher emitted no job map.
- This may have left an orphan `flb-b8` server job depending on `48368`.
- Before relaunching, find and cancel that orphan if it still exists:

  ```bash
  squeue -u "$USER" -n flb-b8 -o '%.18i %.20j %.2t %.30R'
  # After verifying the exact job ID:
  scancel ORPHAN_B8_SERVER_JOB_ID
  ```

### Modality `alpha_0_001`

- Run ID attempted: `alpha_0_001_seed_42_20260831_081449_276143`
- The server submission itself failed with `QOSMaxSubmitJobPerUserLimit`.
- No server, client array, or cleanup job was submitted.
- Because parsing the failed `b8` output cleared the shell's `dependency`
  variable, this attempt also had no dependency flag.

## Execution-status check

Use accounting, rather than submission output, to establish which accepted
runs actually succeeded:

```bash
sacct -j 48357,48358,48359,48360,48361,48362,48363,48364,48365,48366,48367,48368 \
  --format=JobID,JobName%20,State,ExitCode,Elapsed -X
```

## Future relaunch

After job `48368` has finished and any orphan `flb-b8` server has been removed,
submit only the two incomplete experiments. Submit them separately so the
second command can use the actual `b8` cleanup job ID:

```bash
python3 scripts/launch_experiment.py \
  configs/fedavg_train_configs_uneven/mini_batch_sweeps/b8.yaml \
  --mode slurm --partition bigbatch --afterany 48368
```

Copy the cleanup ID printed by that command, then run:

```bash
python3 scripts/launch_experiment.py \
  configs/fedavg_train_configs_uneven/modality_sweep/alpha_0_001.yaml \
  --mode slurm --partition bigbatch --afterany B8_CLEANUP_JOB_ID
```

If accounting shows that one of `k8`, `e5`, `e10`, or `b4` failed during
execution, relaunch that configuration too; do not infer execution success
from the presence of its three job IDs.
