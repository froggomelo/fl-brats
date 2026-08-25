# Apptainer

Apptainer is the primary no-sudo cluster runtime. Set user-owned `FL_BRATS_*` paths, then run `./apptainer/build.sh` to create both partition images:

- `apptainer/fl-brats-batch.sif` uses the normal locked environment.
- `apptainer/fl-brats-stampede.sif` replaces PyTorch with the CUDA 11.8 wheel pair for older stampede GPUs.

SLURM jobs source `apptainer/select_image.sh` and choose the image deterministically
from `SLURM_JOB_PARTITION`: Stampede uses `fl-brats-stampede.sif`, while Batch
and Bigbatch use `fl-brats-batch.sif`. Image selection never reads `.env` or
inherited `APPTAINER_IMAGE`/`FL_BRATS_IMAGE*` variables. Set
`FL_BRATS_USE_GPU=1` for local `apptainer/run.sh` GPU checks.

The image build sets `UV_HTTP_TIMEOUT=300` by default so large PyTorch/CUDA wheels have enough time to download and extract on slower cluster links. Override `UV_HTTP_TIMEOUT` before running `./apptainer/build.sh` if your builder needs a different value.
