# Apptainer

This folder holds the container definitions and helper scripts for running the code on the cluster. Build both images with `./apptainer/build.sh` or `make apptainer-build`.

| File | Purpose |
|---|---|
| `fl-brats.def` | Definition of `fl-brats-batch.sif`, the locked environment used on `batch` and `bigbatch`. |
| `fl-brats-stampede.def` | Definition of `fl-brats-stampede.sif`, with CUDA 11.8 PyTorch wheels for the older stampede GPUs. |
| `build.sh` | Builds both images without sudo, with an optional remote builder. |
| `select_image.sh` | Maps a Slurm partition to its image for the standalone job scripts. |
| `run.sh` | Runs one `fedbrats` command inside the image with the data and output roots mounted. |
| `shell.sh` | Opens an interactive shell in the image with the same mounts. |
| `test.sh` | Runs the environment checks inside the image. |

Image files end in `.sif` and are not tracked by git. [docs/setup.md](../docs/setup.md#build-the-apptainer-images) explains the build and how a job picks its image. [docs/cluster.md](../docs/cluster.md) covers running jobs.
