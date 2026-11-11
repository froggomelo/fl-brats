# Communication-Efficient Federated Learning for Brain Tumour Segmentation with Missing Modalities

Federated brain tumour segmentation on BraTS 2021 under conditions closer to clinical practice. Clients hold uneven amounts of data, some scans lack MRI modalities, and every byte sent between the server and the clients is counted. FedAvg trains a 3D U-Net across physically separate cluster nodes, with one Flower server and one process per client, and is compared against a centrally trained U-Net.

This repository holds the code, configurations, partitioning logic, and Slurm scripts behind every result in the accompanying research report.

## Quick Start

The project needs Python 3.11 and [uv](https://docs.astral.sh/uv/). Install the locked dependencies, check the configuration, and run the tests.

```bash
uv sync --group dev
uv run fedbrats info
uv run fedbrats validate-config
make create-dirs
make test
```

The `.env` file at the repository root sets the dataset, output, log, and cache locations. Edit it to match your machine before preparing data. [Setup](docs/setup.md) explains each variable and the cluster setup with Apptainer.

## Reproducing a Result

Every result follows the same five steps. Each guide linked below gives the full commands.

1. Download BraTS 2021 and point `FL_BRATS_RAW_ROOT` at it. See [Data](docs/data.md).
2. Point `DATASET_INDEX_PATH` and `FL_BRATS_SPLIT_PATH` at the published index and split under `data/`. The client partitions of every reported experiment are already in `data/partitions/`.
3. Create the per-seed initial weights once with `fedbrats create-initial-weights`. Methods that share a network start from the same weights for a given seed.
4. Launch the experiment. The command below submits one Slurm job set per seed. Add `--mode local` to run on one machine or `--dry-run` to print the resolved plan.

   ```bash
   python scripts/launch_experiment.py configs/fedavg_uneven/baseline.yaml
   ```

5. Read the run's `summary.json` for validation results and `final_evaluation/` for the test report of the selected checkpoint. See [Evaluation](docs/evaluation.md).

[Experiments](docs/experiments.md) lists every family, its configurations, and what a finished run directory contains.

## Repository Layout

| Path | Contents |
|---|---|
| `src/fl_brats/` | The Python package and the `fedbrats` command line |
| `configs/` | Hydra project configuration, reusable building blocks, and one folder of experiment YAML files per family |
| `scripts/` | Experiment launchers, data preparation, environment setup, and cluster monitoring |
| `slurm/` | Slurm jobs for federated runs, data preparation, and environment checks |
| `apptainer/` | Container definitions and helpers for building and running the images |
| `tests/` | Pytest suite |
| `diagrams/` | Scripts that draw the report figures from exported results |
| `data/` | The published dataset index, canonical split, and client partitions. The scans themselves are not included |
| `outputs/`, `logs/`, `checkpoints/`, `cache/` | Run results, logs, and caches, ignored by git except for hand-written records under `logs/` |
| `Experiments.md` | Record of every experiment, its settings, and its results |

## Research Questions

Each research question maps to a set of experiment families. Every family has a folder of YAML files under `configs/`, a launch script under `scripts/`, and a record of its runs and results in [Experiments.md](Experiments.md).

| Research question | Experiment families |
|---|---|
| RQ1. How do communication cost and segmentation performance change as the federation grows, and how far can local epochs and mini-batch size offset these changes? | `centralized`, `fedavg_uneven`, `fedavg_even`, `fedavg_train_configs_uneven` |
| RQ2. How do existing missing-modality methods compare when clients hold uneven amounts of data and different modality combinations? | `fedavg_missing_modalities_uneven` (zero-filled U-Net and FedAvg-MM), `fedamm_uneven` |
| RQ3. How well do communication-efficiency strategies preserve segmentation performance, alone and combined with missing-modality handling? | `fedmpq_uneven`, `selective_transmission_uneven`, `comm_eff_uneven`, `fedavg_mm_comm_eff_uneven`, `fedamm_comm_eff_uneven`, `fedamm_comm_eff_ablation_uneven` |

The `model_agnostic_uneven` family swaps the U-Net for SegResNet, SAM 1, and SAM 2. These runs are exploratory and do not appear in the report.

## Documentation

| Guide | Read it to |
|---|---|
| [Setup](docs/setup.md) | install the environment locally and on the cluster |
| [Architecture](docs/architecture.md) | understand how the package is organised and how a run executes |
| [Configuration](docs/configuration.md) | read, write, and override experiment YAML files |
| [Data](docs/data.md) | prepare BraTS 2021, the split, and the client partitions |
| [Experiments](docs/experiments.md) | launch experiments and find their outputs |
| [Evaluation](docs/evaluation.md) | understand the metrics, checkpoint selection, and test reporting |
| [Models](docs/models.md) | select or add a segmentation model |
| [Missing modalities](docs/missing_modalities.md) | configure modality dropout, FedAvg-MM, and FedAMM |
| [Communication efficiency](docs/communication_efficiency.md) | configure quantisation, selective transmission, and quantised broadcast |
| [Cluster](docs/cluster.md) | run distributed jobs on Slurm |
| [Reproducibility](docs/reproducibility.md) | see what fixes a run's environment, data, and seeds |

Dated operational notes, such as the campaign driver that kept runs going on the cluster and the record of unusable GPU nodes, live beside the logs they describe under `logs/`.

## Licence

MIT. See [LICENSE](LICENSE).
