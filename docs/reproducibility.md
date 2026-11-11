# Reproducibility

This page lists what pins a run's software, randomness, data, and starting point, and what each run records about itself. Read it before producing results for a report or rerunning someone else's experiment.

## Software

`uv.lock` fixes every Python dependency to an exact version and hash. `uv sync --group dev` installs from it, and the batch image installs from it with `uv sync --frozen --no-dev`. Run `uv lock` only when deliberately changing a dependency, and commit the new lock file with that change.

Python is fixed at 3.11. `.python-version` names it, `pyproject.toml` accepts only `>=3.11,<3.12`, and `fedbrats check-environment` fails on any other version. The SAM packages come from official GitHub archives pinned by commit in `pyproject.toml`.

The two Apptainer images differ in how tightly they are pinned.

| Image | Pinning |
|---|---|
| `apptainer/fl-brats-batch.sif` | Ubuntu 24.04 base, Python 3.11, and the exact `uv.lock` versions. Matches the local `.venv`. |
| `apptainer/fl-brats-stampede.sif` | Pins only `torch==2.3.1+cu118` and `torchvision==0.18.1+cu118`. Other packages resolve to their latest versions at build time. |

Image files are not tracked by git. Keep the exact `.sif` file used for a set of results. A rebuild pulls the current `ubuntu:24.04` tag and uv installer, which can change system libraries.

Experiment configs set `slurm.bind_source: true`. Jobs then run the checked-out `src/` instead of the copy baked into the image. The image supplies the dependencies and the checkout supplies the code. Record the git commit of the checkout for every set of runs.

## Seeds

Each experiment YAML lists its repetitions under `experiment.seeds`. The full experiments use 42, 43, and 44. The launcher submits one run per seed and writes the seed into the run's resolved config. `configs/config.yaml` sets `seed: 42` and `deterministic: true` for commands that use the Hydra tree.

`seed_everything` in `src/fl_brats/utils/reproducibility.py` applies a seed in one call. It seeds Python's `random`, NumPy, PyTorch, and every CUDA device. With `deterministic` set it enables `torch.use_deterministic_algorithms` in warning mode, sets cuDNN, NVIDIA's library of neural network kernels, to deterministic mode, and turns off cuDNN benchmarking. It returns a worker seed function that gives data loader worker `i` the seed `seed + i`. `train-centralized`, `train-federated`, and the training-configuration sweep call it.

The distributed server and client processes do not call `seed_everything`. Each client seeds its data loader generator from the run seed and loads the same initial weights file.

Several seeds stay fixed across all runs and never follow the run seed.

| Setting | Value | Controls |
|---|---|---|
| `seed` in `configs/split/default.yaml` | 42 | Subject shuffle for the train, validation, and test split. |
| `partition_seed`, else the top-level `seed`, read by `fedbrats prepare-experiment-partition` | 42 | Client partitioning. Each partition is built once and shared by every run seed. |
| `missing_modalities.seed` | 42 | Missing-modality assignment. |
| `evaluation.modality_seed` | 42 | Natural modality masks for held-out evaluation. |

Two limits apply. Warning mode lets a non-deterministic GPU kernel run with a warning instead of an error. Runs with the same seed are therefore close but not guaranteed to match bit for bit on a GPU. `seed_everything` also sets `PYTHONHASHSEED`, which affects only child processes. Export `PYTHONHASHSEED` before starting Python to fix hashing in the main interpreter.

## Initial weights

Every seed has its own initial weights file. All methods compared at that seed start from the same file. Create a file with `fedbrats create-initial-weights`.

```bash
uv run fedbrats create-initial-weights --model-config configs/model/unet_baseline.yaml --seed 42
```

The command seeds PyTorch, builds the model, and writes `<output-dir>/<model name>_seed_<seed>.pt`. The default output folder is `artifacts/initial_weights`. A JSON file beside it records the seed, the SHA-256 hash of the weights, and a model report. Repeat for seeds 43 and 44, and for each model the configs reference.

| Model config | Output folder | Used by |
|---|---|---|
| `configs/model/unet_baseline.yaml` | `artifacts/initial_weights` | Full-modality U-Net runs and the centralised baseline. |
| `configs/model/fedamm.yaml` | `artifacts/initial_weights` | Missing-modality runs. |
| `configs/model/segresnet.yaml` | `artifacts/initial_weights/segresnet` | SegResNet backbone comparison. |
| `configs/model/segresnet_8.yaml` | `artifacts/initial_weights/segresnet_8` | SegResNet backbone comparison. |

Pass `--output-dir` for the SegResNet folders. On the cluster, submit `slurm/_shared/create_initial_weights.sbatch` with `MODEL_CONFIG`, `INITIAL_WEIGHTS_DIR`, and `INITIAL_WEIGHTS_SEED` exported.

Experiment YAML files map each seed to its file under `initial_weights_by_seed`. The launcher picks the matching entry. The server hashes the loaded file and stores it as `initial_weight_hash` in the run's `summary.json`. Compare that value with the JSON beside the weights to confirm the starting point. A resumed run stores the hash of its resume checkpoint there instead. `artifacts/initial_weights/` is not tracked by git. Keep the files with the results they produced. [models.md](models.md) describes each model.

## Fixed split and partitions

The canonical split, the dataset index it was built from, and every client partition used in the report are published under `data/`. Using them guarantees the same subjects in every cohort and client, with the same modality masks. [data/README.md](../data/README.md) lists the files and how they were checked.

Every experiment reads one split manifest, set by `FL_BRATS_SPLIT_PATH` in `.env`. `fedbrats create-split` shuffles the complete, valid subjects with a fixed seed and assigns 70 % to training, 15 % to validation, and 15 % to testing. The manifest records the split seed, the fractions, the dataset index hash, a creation timestamp, and `split_manifest_hash`.

The hash covers the timestamp. Running `create-split` again yields the same subjects under a new hash. Always reuse the canonical manifest instead of regenerating it. `.env` notes the expected hash beside `FL_BRATS_SPLIT_PATH`. Check a manifest before use.

```bash
uv run fedbrats inspect-split --manifest-path "$FL_BRATS_SPLIT_PATH"
```

Partition manifests store the hash of the split they came from. Training and manifest validation reject a partition built from a different split. [data.md](data.md) covers the split and partition pipeline.

## What each run records

| File | Written by | Contents |
|---|---|---|
| `logs/<family>/<run_id>/resolved-launch-config.yaml` | Launcher | The config after the seed, paths, initial weights, and `--set` overrides are applied. |
| `outputs/<family>/<run_id>/resolved_config.yaml` | Federated server | The runtime config the server used. |
| `outputs/<family>/<run_id>/evaluation_metadata.json` | Federated server | Hashes of the split file, the validation and test subject lists, and the preprocessing config, plus the modality and metric policies. |
| `outputs/<family>/<run_id>/summary.json` | Federated server or `train-centralized` | Seed, initial weight hash, rounds or epochs, and final metrics. The centralised summary also records the dataset index, split, and partition hashes. |

A resumed run must match the source run's evaluation metadata. The server refuses the resume otherwise.

Training runs do not record git metadata or hardware details. Capture them separately with an environment report, which records the platform, library versions, GPU, Slurm job, git branch, commit, and dirty flag.

```bash
uv run fedbrats write-environment-report --path outputs/environment.json
```

`fedbrats init-run` writes the same report together with the resolved Hydra config into a fresh run folder, without training.
