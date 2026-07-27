# SLURM preprocessing

Submit `sbatch slurm/brats_validate.sbatch`, `sbatch slurm/brats_preprocess.sbatch`, and `sbatch slurm/brats_partition.sbatch`. Set `FL_BRATS_DATA_ROOT` and `FL_BRATS_IMAGE`; resource settings are environment-overridable. Validation/preprocessing use CPU by default. Do not run full-dataset validation on the login node.

For a single ordered, resumable job, use:

```bash
export FL_BRATS_DATA_ROOT="/datasets/$USER/fedbrats"
export FL_BRATS_IMAGE="$PWD/apptainer/fl-brats.sif"
sbatch --export=ALL,FL_BRATS_DATA_ROOT,FL_BRATS_IMAGE slurm/brats_pipeline.sbatch
```

The pipeline binds raw data read-only and binds processed/partition roots
writable. It stops at strict validation failures, writes an index, runs both
manifest-only and materialized preprocessing, creates the fixed split and
partition, then verifies every manifest path. Re-submit the same job after a
preemption: materialized subjects with matching completion metadata are skipped.
