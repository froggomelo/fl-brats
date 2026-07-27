# SLURM templates

Submit with `sbatch slurm/environment_check.sbatch` and inspect the configured log path. Use `srun` for interactive allocations and `scancel JOBID` to stop work. Client placeholders demonstrate array jobs only; they do not start processes. Set cluster-specific partition and resource flags at submission time, run Apptainer with clean environments, avoid login-node computation, and ensure future server/client processes shut down cleanly to prevent ghost processes.

`missing_modality_baseline.sbatch` creates a deterministic split and missing-modality partition, validates and reports it, then runs the baseline training workflow. Configure it through exported variables, notably `MM_MODE=complete|arbitrary|mixed`, `MM_T1CE_PROBABILITY`, and `MM_CLIENT_POLICY`.
