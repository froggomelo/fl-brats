# Apptainer

Apptainer is the primary no-sudo cluster runtime. Set user-owned `FL_BRATS_*` paths and `APPTAINER_IMAGE`, then use `./apptainer/run.sh info`. `build.sh` tries a configured remote builder or an unprivileged build; build on another Linux host and transfer the SIF if needed. Set `FL_BRATS_USE_GPU=1` to add `--nv`; all normal runs use a clean environment and bind data read-only. Use writable cache and output paths, including under SLURM.
