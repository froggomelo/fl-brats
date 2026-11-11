# Configuration

This guide explains how fl-brats is configured, how experiment files inherit from each other, and which value wins when the same key is set in several places. Read it before editing or adding an experiment YAML file.

## Two kinds of configuration

The repository uses two configuration systems. They live side by side under `configs/` and serve different commands.

| System | Entry point | Read by | Purpose |
|---|---|---|---|
| Hydra project config | `configs/config.yaml` | `fedbrats show-config`, `validate-config`, `info`, `create-dirs`, `check-environment`, `init-run`, and `train-federated` without `--config` | Infrastructure settings such as paths, device, execution environment, tracking, and logging |
| Experiment YAML | `configs/<family>/<variant>.yaml` | `scripts/launch_experiment.py`, `fedbrats train-federated --config`, `prepare-experiment-partition`, `resolve-experiment` | One self-contained description of a training run, its data partition, model, federated strategy, compression, evaluation, and Slurm job shape |

Every experiment in [Experiments.md](../Experiments.md) is an experiment YAML file. The Hydra project config matters mainly for environment checks and interactive inspection.

## The Hydra project config

[Hydra](https://hydra.cc) composes `configs/config.yaml` from a defaults list of config groups. Each group is a folder under `configs/`, and each option is one YAML file in that folder.

```yaml
defaults:
  - environment: local
  - tracking: local
  - logging: default
  - data: brats2021
  - model: unet_baseline
  - federated: fedavg
  - quantization: default
  - missing_modalities: default
  # ... further groups, then _self_
```

Three groups describe the infrastructure.

| Group | Options | Contents |
|---|---|---|
| `environment` | `local`, `debug`, `apptainer`, `cluster` | Execution mode, container runtime, scheduler, node rank, server address and port, graphics processing unit (GPU) use, and Apptainer image path |
| `tracking` | `local`, `mlflow` | Tracking backend and where runs are recorded |
| `logging` | `default`, `debug` | Log level, console and file output, and the log file name |

The top-level keys in `config.yaml` set the project name, seed, determinism, device, precision, worker count, and the data, output, log, cache, and checkpoint roots. The roots read `FL_BRATS_DATA_ROOT`, `FL_BRATS_OUTPUT_ROOT`, `FL_BRATS_LOG_ROOT`, `FL_BRATS_CACHE_ROOT`, and `FL_BRATS_CHECKPOINT_ROOT` through Hydra's `oc.env` interpolation and fall back to built-in defaults when a variable is unset. `_self_` comes last in the defaults list. Values written in `config.yaml` itself therefore override the group files.

Select a different group option or change a single key with Hydra overrides on the command line.

```bash
uv run fedbrats show-config environment=debug tracking=mlflow logging=debug
uv run fedbrats show-config seed=43 device=cpu
```

## Inspecting and validating the project config

Load `.env` first so that the path variables match the machine. [Setup](setup.md) describes the file.

```bash
set -a; source .env; set +a
uv run fedbrats show-config environment=cluster
uv run fedbrats validate-config environment=cluster
```

`show-config` prints the fully resolved configuration without creating any files. `validate-config` resolves the same configuration and checks it against the Pydantic schema in `src/fl_brats/config/schema.py`. It prints `Configuration is valid.` on success and lists each failing key otherwise.

Among other checks, the schema rejects an unknown environment mode, cluster mode without `scheduler: slurm`, Apptainer mode without an image path, a server port outside 1024 to 65535, a negative worker count, a precision other than `float32`, `float16`, or `bfloat16`, and a seed outside 0 to 2^32 − 1. The `missing_modalities` and `quantization` groups are validated strictly. Unknown keys fail, quantisation must be enabled together with a `fixed` or `mixed` mode, and the sparsification fraction must lie in (0, 1].

The Pydantic schema does not run on experiment YAML files. Inspect those with `resolve-experiment` and a launcher dry run, as described below.

## Experiment YAML files

An experiment YAML file describes one experiment variant. It lives at `configs/<family>/<variant>.yaml`, where the family folder matches a section of [Experiments.md](../Experiments.md). The launcher reads these keys directly.

| Key | Meaning |
|---|---|
| `experiment.family` | Folder name used under `outputs/` and `logs/`, for example `fedavg_uneven` or `model_agnostic_uneven/sam1` |
| `experiment.variant` | Prefix of each run ID. Defaults to the file name |
| `experiment.seeds` | Seeds to launch, one run per seed |
| `initial_weights`, `initial_weights_by_seed` | Shared starting weights. The per-seed map selects the file for each seed |
| `data.partition_root` | Client partition folder written by `prepare-experiment-partition` |
| `slurm` | Partition, wall time, CPUs, memory, node exclusions, Apptainer image, and GPU check for each job |

The `model`, `optimizer`, `training`, `federated`, `quantization`, `missing_modalities`, `evaluation`, and `communication_measurement` blocks configure the run itself. An experiment file carries its own copy of each block. It does not compose the Hydra groups.

## Inheritance with `extends`

An experiment file may name one parent or a list of parents with `extends`. Paths are relative to the file that contains them.

```yaml
# configs/fedamm_uneven/dirichlet_alpha_0_1_k5.yaml
extends: [dirichlet_alpha_0_1.yaml, ../_experiment_base/clients_lorenz/k5.yaml]
experiment: {variant: dirichlet_alpha_0_1_k5, runs: 1, seeds: [42]}
data: {partition_root: data/partitions/missing_dirichlet_k5/alpha_0_1}
```

The loader resolves parents recursively and merges them in list order, then merges the child on top. Nested mappings merge key by key. Scalars and lists replace the inherited value outright. A cycle in the `extends` chain is an error.

Shared defaults live in `configs/_experiment_base/`.

| File or folder | Role |
|---|---|
| `federated_uneven.yaml` | Root of every federated experiment. U-Net, four clients with the 50/25/15/10 % split, federated averaging (FedAvg) for 150 rounds with a 100-round minimum and 50-round early-stopping patience, held-out evaluation policy, and the default `batch` Slurm job |
| `missing_uneven.yaml` | Adds the modality-specific encoder model and a per-modality dropout probability of q = 0.1 for the missing-modality families |
| `missing_dirichlet/alpha_*.yaml` | Layered onto a method config to switch to the Dirichlet modality partition of the FedAMM method at one concentration |
| `missing_per_client/flair_largest.yaml` | Layered onto a method config so that every subject of a client lacks the same scan |
| `clients_lorenz/k5.yaml`, `k8.yaml` | Five or eight clients with the four-client split's shape |
| `clients_lorenz/k5_bit_budgets.yaml`, `k8_bit_budgets.yaml` | Per-client bit-width budgets for compressed runs with five or eight clients |

Print the fully inherited file to check what a run will receive.

```bash
uv run fedbrats resolve-experiment --config configs/fedamm_uneven/dirichlet_alpha_0_1_k5.yaml
```

## Override precedence

A federated run assembles its configuration in five layers. Each later layer overrides the earlier ones.

| Order | Layer | Where it is set |
|---|---|---|
| 1 | Inherited YAML | Parent files named by `extends`, resolved recursively |
| 2 | Experiment YAML | The file passed to the launcher |
| 3 | Launcher options | `--seed`, `--partition`, `--image`, `--exclude`, `--resume-checkpoint`, and `--set KEY=VALUE` on `scripts/launch_experiment.py` |
| 4 | `FL_BRATS_*` environment variables | The job's environment, usually from `.env` |
| 5 | Command-line interface (CLI) `key=value` overrides | Extra arguments after `fedbrats train-federated --config FILE --run-dir DIR` |

The launcher writes layers 1 to 3 into `logs/<family>/<run-id>/resolved-launch-config.yaml`. The training command reads that file and applies layers 4 and 5 when the job starts. The recognised environment variables are listed below.

| Variable | Key it sets |
|---|---|
| `FL_BRATS_OUTPUT_ROOT` | `output_root` |
| `FL_BRATS_LOG_ROOT` | `log_root` |
| `FL_BRATS_CACHE_ROOT` | `cache_root` |
| `FL_BRATS_CHECKPOINT_ROOT` | `checkpoint_root` |
| `FL_BRATS_PARTITION_ROOT` | `data.partition_root` |
| `FL_BRATS_SPLIT_PATH` | `data.split_path` |
| `FL_BRATS_INITIAL_WEIGHTS` | `initial_weights` |
| `FL_BRATS_DEVICE` | `device` |

CLI overrides use OmegaConf dot-list syntax and merge the same way as `extends`.

```bash
uv run fedbrats train-federated --config configs/fedavg_uneven/baseline.yaml \
  --run-dir outputs/scratch/fedavg_check training.max_rounds=2 federated.num_rounds=2
```

Centralised runs follow layers 1 to 3 only. The launcher translates their YAML into `fedbrats train-centralized` flags, and the index and split paths come from `DATASET_INDEX_PATH` and `FL_BRATS_SPLIT_PATH` when the YAML leaves them unset.

## Building-block groups

The remaining Hydra groups hold reusable defaults for the composed project config. Several files are also read directly by commands, and experiment YAML files copy their values. The `model` files, for example, feed `fedbrats create-initial-weights --model-config`.

| Group | Options | Contents |
|---|---|---|
| `data` | `brats2021` | Dataset name, raw, processed, and partition roots, modality list, and expected subject count |
| `preprocessing` | `brats2021_v1` | Orientation, resampling, cropping, padding, intensity normalisation, and verification used by `preprocess-data` |
| `split` | `default` | Seed and 70/15/15 % train, validation, and test fractions |
| `partition` | `fixed_quantity_skew` | Client fractions 50/25/15/10 % and the partition manifest path |
| `model` | `unet_baseline`, `unet_missing_modality_baseline`, `fedamm`, `segresnet`, `segresnet_8`, `sam1`, `sam2` | Architecture definitions, also passed to `create-initial-weights` and `verify-pretrained-model` |
| `loss` | `dice_ce` | Dice plus cross-entropy weights and smoothing |
| `optimizer` | `adam` | Adam learning rate, weight decay, betas, and epsilon |
| `scheduler` | `none` | Learning-rate scheduler, disabled |
| `training` | `centralized`, `federated_client` | Epochs, batch size, workers, and mixed precision, plus early stopping for centralised training |
| `federated` | `fedavg`, `fedprox` | Strategy, server port, rounds, client counts, and sampling fractions |
| `quantization` | `default` | Communication quantisation, sparsification, and error feedback, all disabled |
| `missing_modalities` | `default` | Modality availability policy, disabled |
| `campaign` | `deadline_2026_10`, `smoke_2026_09_24` | Priority-ordered run lists for the campaign driver. See [logs/monitoring/campaign/README.md](../logs/monitoring/campaign/README.md) |

[Models](models.md), [Missing modalities](missing_modalities.md), and [Communication efficiency](communication_efficiency.md) explain the model, missing-modality, and compression settings in detail.

## Adding an experiment

Create a new file in the family folder and inherit from the closest existing variant. Override only what differs and give the variant a new name.

```yaml
# configs/fedavg_uneven/lr_3e_4.yaml
extends: baseline.yaml
experiment: {variant: lr_3e_4, seeds: [42]}
optimizer: {lr: 0.0003}
```

Give every new client partition its own `data.partition_root`. Two partitions must never share a folder. Prepare the partition, then check the resolved run with a dry run before submitting. [Experiments](experiments.md) describes both steps.
