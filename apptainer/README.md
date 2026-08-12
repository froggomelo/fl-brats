# Apptainer

Apptainer is the primary no-sudo cluster runtime. Set user-owned `FL_BRATS_*` paths, then run `./apptainer/build.sh` to create both partition images:

- `apptainer/fl-brats-batch.sif` uses the normal locked environment.
- `apptainer/fl-brats-stampede.sif` replaces PyTorch with the CUDA 11.8 wheel pair for older stampede GPUs.

SLURM jobs source `apptainer/select_image.sh` and choose the image from `SLURM_JOB_PARTITION`. Use `FL_BRATS_IMAGE_STAMPEDE` or `FL_BRATS_IMAGE_BATCH` to override those defaults. A direct `APPTAINER_IMAGE` value is still treated as an explicit one-off override. Set `FL_BRATS_USE_GPU=1` for local `apptainer/run.sh` GPU checks.
