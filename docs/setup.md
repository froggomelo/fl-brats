# Setup

This guide takes a fresh checkout to a working environment on a workstation and on the Wits mscluster. Read it before running any experiment. [cluster.md](cluster.md) covers job submission once setup is complete.

## Prerequisites

The code runs on Linux with Python 3.11. `pyproject.toml` accepts only 3.11, and `fedbrats check-environment` reports any other version as an error. Install these tools first.

| Tool | Needed for | Notes |
|---|---|---|
| git | Version metadata in environment reports | Any recent version. |
| [uv](https://docs.astral.sh/uv/) | Python, dependencies, and the `.venv` | The bootstrap scripts install it into `~/.local/bin` when it is missing. |
| Python 3.11 | Running the code | uv installs a managed 3.11 when the system has none. |
| Apptainer | Cluster jobs and container checks | Not needed for local development. Builds run without sudo. |
| Slurm client tools | Submitting jobs | Present on the mscluster login node only. |

A graphics processing unit, or GPU, is optional for local work. Training falls back to the CPU when `device: auto` finds no CUDA device. CUDA is NVIDIA's GPU computing platform.

## Install the Python environment

Run every command from the repository root. Install the locked dependencies together with the development tools.

```bash
uv sync --group dev
```

uv creates `.venv/` and installs the exact versions recorded in `uv.lock`. Prefix commands with `uv run` to use that environment without activating it.

`scripts/environment/bootstrap.sh` wraps the same step for a workstation. It requires a system `python3.11`, installs uv when absent, runs `uv sync --group dev`, installs the pre-commit hooks, and stops if `.env` is missing.

```bash
./scripts/environment/bootstrap.sh
```

## Configure paths in `.env`

`.env` at the repository root holds the paths and links for one machine. The copy in the repository records the author's cluster account. Edit every value before the first run on a new account.

| Variable | Meaning |
|---|---|
| `PROJECT_DIR` | Base folder for relative paths in `scripts/environment/install_aspera.sh` and `scripts/data_preprocessing/brats_download.sbatch`. Both fall back to the repository root when it is unset or missing. |
| `FL_BRATS_DATA_ROOT` | Dataset root. It holds `raw/`, `processed/`, and `partitions/`. The downloader writes to `raw/brats2021` below it. The Slurm wrapper refuses to start without it. |
| `FL_BRATS_RAW_ROOT` | Raw BraTS 2021 folder. Defaults to `$FL_BRATS_DATA_ROOT/raw/brats2021`. |
| `DATASET_INDEX_PATH` | Dataset index CSV written by preprocessing. Default for `--index-path` in `train-centralized`, `visualize-segmentation`, and the data scripts. |
| `FL_BRATS_SPLIT_PATH` | Canonical train, validation, and test split manifest shared by every partition. It replaces `data.split_path` in every experiment YAML. The comment beside it records the expected split hash. |
| `FASPEX_URL` | TCIA Faspex package link used by the BraTS downloader. TCIA is The Cancer Imaging Archive. Treat the link as private. |
| `FL_BRATS_OUTPUT_ROOT` | Root for run outputs. The launcher writes each run to `<root>/<family>/<run_id>/`. |
| `FL_BRATS_LOG_ROOT` | Root for logs. The launcher writes each run's Slurm logs to `<root>/<family>/<run_id>/`. |
| `FL_BRATS_CACHE_ROOT` | Cache root, mounted as `/cache` by `apptainer/run.sh`. |
| `FL_BRATS_CHECKPOINT_ROOT` | Checkpoint root created by `fedbrats create-dirs`. Federated runs keep their checkpoints inside the run folder. |
| `TOOLCHAIN_DIR` | Install folder for the Aspera download client. Defaults to `.toolchain` under `PROJECT_DIR`. |

`fedbrats` does not read `.env` on its own. Export it into the shell before running commands locally.

```bash
set -a; source .env; set +a
```

Slurm jobs need no such step. `slurm/_shared/run_with_configured_environment.sh` and most submit scripts source `.env` from the submission folder. [data.md](data.md) describes the dataset layout behind these paths.

## Check the configuration

Three commands confirm that the environment and configuration agree. All accept Hydra overrides such as `environment=debug`.

```bash
uv run fedbrats info
uv run fedbrats validate-config
uv run fedbrats check-environment
```

| Command | What it does |
|---|---|
| `fedbrats info` | Prints the platform, Python, PyTorch, CUDA, GPU, library versions, git branch and commit, and execution mode. It reads no secrets. |
| `fedbrats validate-config` | Composes `configs/config.yaml` with Hydra and validates it against the Pydantic schema. |
| `fedbrats check-environment` | Fails on a Python other than 3.11, a missing core library, or an unwritable output, log, cache, or checkpoint root. It warns when uv, git, Apptainer, or `sbatch` is absent. In `environment=apptainer` it also fails when the configured image is missing. |
| `fedbrats show-config` | Prints the fully resolved configuration without creating anything. |

The `environment` config group selects the execution mode. `local` is the default.

| Group file | Mode | Use |
|---|---|---|
| `configs/environment/local.yaml` | `local` | Workstation runs in `.venv`. |
| `configs/environment/debug.yaml` | `debug` | Small CPU-only checks. |
| `configs/environment/cluster.yaml` | `cluster` | Slurm runs with Apptainer and GPUs. |
| `configs/environment/apptainer.yaml` | `apptainer` | Runs inside `apptainer/fl-brats-batch.sif`. |

[configuration.md](configuration.md) explains the rest of the Hydra tree and the experiment YAML files.

## Create the working folders

Create the configured roots and every log folder the Slurm scripts write to.

```bash
make create-dirs
```

The target runs `make log-dirs` first. That target collects every `#SBATCH --output` and `--error` path under `slurm/` and `scripts/` and creates the parent folders. Slurm will not start a job whose output folder is missing. `fedbrats create-dirs` then creates the data, output, log, cache, and checkpoint roots. It refuses paths under `/usr`, `/opt`, `/root`, `/etc`, and `/var`.

Rerun `make create-dirs` after adding a script with a new log path.

## Set up the cluster account

Log in to the mscluster login node and clone the repository into your home folder. Do not run training or installs on the login node. Submit the setup job instead.

```bash
sbatch slurm/environment/setup_environment.sbatch
```

The job runs `scripts/environment/bootstrap_cluster.sh` on a compute node. That script installs uv when absent, installs a uv-managed Python 3.11 when the node has none, runs `uv sync --group dev --python 3.11`, and checks the core imports. The job then runs `validate-config` and `check-environment` and writes `.setup/environment_ready` on success. Set `FL_BRATS_PYTHON_MODULE` to load an environment module first. `make cluster-bootstrap` runs the same script on the current node.

Edit `.env` for your account, then run `make create-dirs`. Submit the environment check to confirm that a compute node can run the image.

```bash
sbatch slurm/environment/environment_check.sbatch
```

The check writes to `logs/environment/check_<jobid>.out`.

## Build the Apptainer images

Apptainer runs the code on the cluster from a single image file. Build both images with one command. Run it inside a Slurm job or on another Linux host with Apptainer, then place the `.sif` files in `apptainer/`.

```bash
make apptainer-build
```

`apptainer/build.sh` builds two images from their definition files.

| Image | Definition | Contents |
|---|---|---|
| `apptainer/fl-brats-batch.sif` | `apptainer/fl-brats.def` | Ubuntu 24.04, uv-managed Python 3.11, and `uv sync --frozen --no-dev` from `uv.lock`. The SAM packages are included. |
| `apptainer/fl-brats-stampede.sif` | `apptainer/fl-brats-stampede.def` | The same base with the CUDA 11.8 wheels `torch==2.3.1+cu118` and `torchvision==0.18.1+cu118` for the older stampede GPUs. |

The build never uses sudo. It keeps temporary files in `.apptainer-tmp/` and its cache in `.apptainer-cache/` unless `APPTAINER_TMPDIR` or `APPTAINER_CACHEDIR` is set. Set `APPTAINER_REMOTE` to fall back to a remote builder when a local build fails. `UV_HTTP_TIMEOUT` defaults to 300 seconds inside the build. Raise it on a slow link.

Each image copies `src/` and `configs/` at build time. Rebuild after changing dependencies. Experiment configs set `slurm.bind_source: true`, which mounts the checked-out `src/` over the copy in the image. Source changes therefore need no rebuild for those runs.

A job picks its image in one of two ways.

- Runs submitted through `scripts/launch_experiment.py` use `slurm.image` from the experiment YAML, or `--image` when given. The shared base config sets `apptainer/fl-brats-batch.sif`.
- Standalone `sbatch` scripts, `apptainer/run.sh`, and `apptainer/shell.sh` source `apptainer/select_image.sh`. It maps the `stampede` partition to `fl-brats-stampede.sif` and every other partition to `fl-brats-batch.sif`. It ignores `.env` and any inherited `APPTAINER_IMAGE`.

The SAM configs under `configs/model_agnostic_uneven/` name `apptainer/fl-brats-batch.sam.sif`. `build.sh` does not produce that file. Copy the batch image to that name or point `slurm.image` at the batch image.

`apptainer/run.sh` runs a `fedbrats` command inside the image with the `FL_BRATS_*` roots mounted at `/data`, `/outputs`, `/logs`, `/cache`, and `/checkpoints`. `apptainer/shell.sh` opens an interactive shell with the same mounts. Set `FL_BRATS_USE_GPU=1` to pass `--nv` to `run.sh`.

```bash
./apptainer/run.sh info
```

## Run the tests

Run the test suite and the linters before changing code.

```bash
uv run pytest
make lint
make typecheck
```

`make cluster-env-check` runs `scripts/environment/validate_environment.sh`. It repeats `check-environment` and `validate-config`, imports the core libraries, and runs `tests/test_config.py` and `tests/test_environment.py`.

## Makefile targets

| Target | Command | Purpose |
|---|---|---|
| `install-dev` | `uv sync --group dev` | Locked install with development tools. |
| `install`, `sync` | `uv sync` | Locked install without development tools. |
| `lock` | `uv lock` | Refresh `uv.lock` after editing dependencies. |
| `lint` | `uv run ruff check .` | Lint. |
| `format`, `format-check` | `uv run ruff format .` | Format, or check formatting only. |
| `typecheck` | `uv run mypy src` | Static type check. |
| `test`, `coverage` | `uv run pytest` | Tests, with a coverage report for `coverage`. |
| `precommit` | `uv run pre-commit run --all-files` | All pre-commit hooks. |
| `info`, `show-config`, `validate-config`, `check-environment` | `uv run fedbrats ...` | The checks described above. |
| `create-dirs` | `make log-dirs` then `fedbrats create-dirs` | Roots and Slurm log folders. |
| `log-dirs` | `grep` and `mkdir -p` | Slurm log folders only. |
| `environment-report` | `fedbrats write-environment-report --path outputs/environment.json` | Saves the environment report as JSON. |
| `init-run` | `fedbrats init-run` | Creates a run folder with the resolved config and environment report, without training. |
| `apptainer-build` | `./apptainer/build.sh` | Builds both images. |
| `apptainer-test` | `./apptainer/test.sh` | Runs `info`, `validate-config`, and `check-environment` inside the image. Its final pytest step fails at present. The image holds no `tests/` folder and `run.sh` prefixes every command with `fedbrats`. |
| `cluster-bootstrap` | `./scripts/environment/bootstrap_cluster.sh` | Cluster install on the current node. |
| `cluster-env-check` | `./scripts/environment/validate_environment.sh` | Environment check with the config and environment tests. |
| `clean` | `rm -rf` | Removes tool caches, coverage output, and build folders. |
