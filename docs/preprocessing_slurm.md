# SLURM preprocessing

Submit `sbatch slurm/data/brats_validate.sbatch`, `sbatch slurm/data/brats_preprocess.sbatch`, and `sbatch slurm/data/brats_partition.sbatch`. Set `FL_BRATS_DATA_ROOT`; resource settings are environment-overridable. SLURM jobs choose `apptainer/fl-brats-stampede.sif` or `apptainer/fl-brats-batch.sif` from the listed partition unless `APPTAINER_IMAGE` is explicitly set. Validation/preprocessing use CPU by default. Do not run full-dataset validation on the login node.

For a single ordered, resumable job, use:

```bash
export FL_BRATS_DATA_ROOT="/datasets/$USER/fedbrats"
export FL_BRATS_RAW_ROOT="/datasets/$USER/brats/raw/brats2021/PKG - RSNA-ASNR-MICCAI-BraTS-2021/RSNA-ASNR-MICCAI-BraTS-2021/BraTS2021_TrainingSet"
bash apptainer/build.sh
sbatch --export=ALL,FL_BRATS_DATA_ROOT,FL_BRATS_RAW_ROOT slurm/data/brats_pipeline.sbatch
```

The raw root deliberately points at the labeled training set; the package's
validation set does not include the required `*_seg.nii.gz` files. The pipeline binds raw data read-only and binds processed/partition roots
writable. It stops at strict validation failures, writes an index, runs both
manifest-only and materialized preprocessing, creates the fixed split and
partition, then verifies every manifest path. Re-submit the same job after a
preemption: materialized subjects with matching completion metadata are skipped.
