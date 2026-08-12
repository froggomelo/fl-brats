# SLURM templates

Submit with `sbatch slurm/environment/environment_check.sbatch` and inspect the configured log path. Use `srun` for interactive allocations and `scancel JOBID` to stop work. Client placeholders demonstrate array jobs only; they do not start processes. Set cluster-specific partition and resource flags at submission time, run Apptainer with clean environments, avoid login-node computation, and ensure future server/client processes shut down cleanly to prevent ghost processes.

## Layout

- `centralized/`: centralized UNet baseline jobs and seed-sweep submitters.
- `federated/baseline/`: FedAvg/FedProx-style baseline server, client, and submit jobs.
- `federated/sam/`: SAM1/SAM2 federated jobs.
- `missing_modality/`: missing-modality baseline and federated submitters.
- `data/`: BraTS validation, preprocessing, partitioning, and missing-modality partition preparation.
- `environment/`: cluster setup and environment checks.
- `templates/`: placeholders for future job templates.

`missing_modality/missing_modality_baseline.sbatch` creates a deterministic split and missing-modality partition, validates and reports it, then runs the baseline training workflow. Configure it through exported variables, notably `MM_MODE=complete|arbitrary|mixed`, `MM_T1CE_PROBABILITY`, and `MM_CLIENT_POLICY`.
