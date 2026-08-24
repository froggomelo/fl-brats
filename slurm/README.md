# SLURM templates

Submit with `sbatch slurm/environment/environment_check.sbatch` and inspect the configured log path. Use `srun` for interactive allocations and `scancel JOBID` to stop work. Client placeholders demonstrate array jobs only; they do not start processes. Set cluster-specific partition and resource flags at submission time, run Apptainer with clean environments, avoid login-node computation, and ensure future server/client processes shut down cleanly to prevent ghost processes.

Scheduler output mirrors the job layout under `logs/slurm/`: `centralized/`,
`data/`, `environment/`, `federated/baseline/`, `federated/sam/`, and
`missing_modality/`. Standard output and error use the same stem with `.out`
and `.err` suffixes. Keep application-level artifacts inside each run under
`outputs/`; `cache/matplotlib/` holds Matplotlib's generated cache.

## Layout

- `centralized/`: centralized UNet baseline jobs and seed-sweep submitters.
- `federated/baseline/`: FedAvg/FedProx-style baseline server, client, and submit jobs.
- `federated/sam/`: SAM1/SAM2 federated jobs.
- `missing_modality/`: missing-modality baseline and federated submitters.
- `data/`: BraTS validation, preprocessing, partitioning, and missing-modality partition preparation.
- `environment/`: cluster setup and environment checks.
- `templates/`: placeholders for future job templates.

Submit concurrent centralized seed runs with
`bash slurm/centralized/centralized_sweep_submit.sh`. Set
`FL_BRATS_PARTITION=stampede|bigbatch` or edit the script's `partition` default;
the worker automatically selects the matching Apptainer image. GPU training
excludes `mscluster25` by default because CUDA is unavailable there; override
the exclusion list with `FL_BRATS_EXCLUDE_NODES`.

`missing_modality/missing_modality_baseline.sbatch` creates a deterministic split and missing-modality partition, validates and reports it, then runs the baseline training workflow. Configure it through exported variables, notably `MM_MODE=complete|arbitrary|mixed`, `MM_T1CE_PROBABILITY`, and `MM_CLIENT_POLICY`.

## Even and uneven FedAvg baselines

Prepare both full-modality baseline partitions from the fixed dataset split:

```bash
bash slurm/data_preprocessing/prepare_fedavg_baseline_partitions_submit.sh
```

Partition preparation runs on `stampede` by default. Override it only when
needed with `FL_BRATS_DATA_PARTITION`. The even and uneven training YAML files
submit their server and client jobs to `bigbatch`.

Set `FL_BRATS_SPLIT_PATH` if the split is not at
`data/splits/brats2021.json`. Set `FL_BRATS_OVERWRITE_PARTITIONS=1` only when
the existing even and uneven partition manifests should be replaced. Wait for
both preparation jobs to complete successfully, then submit each three-seed
experiment independently:

```bash
bash slurm/fedavg_uneven/federated_baseline_submit.sh
bash slurm/fedavg_even/federated_baseline_submit.sh
```

Both launchers accept the generic launcher's `--dry-run`, `--mode`, and
`--seed` options. With no options, each launcher submits the three seeds from
its experiment YAML to SLURM.
