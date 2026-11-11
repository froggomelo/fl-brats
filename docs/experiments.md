# Experiments

This guide explains how to launch an experiment, where each run writes its results, and which configuration folder belongs to which experiment. Read it once the data is prepared and the environment works, before submitting the first training job.

## Terms

The method and parameter names below recur throughout this guide and [Experiments.md](../Experiments.md).

| Term | Meaning |
|---|---|
| FedAvg | Federated averaging, the baseline aggregation rule |
| FedAMM | Missing-modality method with modality-specific encoders, prototype alignment, and modality-weighted aggregation |
| FedAvg-MM | FedAvg with the FedAMM encoders and without prototypes or modality weighting |
| FedMPQ | Mixed-precision quantisation of client uploads |
| K, E, B | Number of clients, local epochs per round, and mini-batch size |
| q | Per-modality probability that a scan is missing |
| p | Fraction of update values a client sends under top-p sparsification |

## Before the first run

Every experiment needs a preprocessed dataset with a train, validation, and test split, a client partition for its configuration, and initial weights for each seed. [Data](data.md) covers the dataset and split. The steps below cover the rest.

Activate the project environment and load `.env` so that `python`, `fedbrats`, and the dataset paths resolve correctly.

```bash
source .venv/bin/activate
set -a; source .env; set +a
```

### Prepare the client partition

Each experiment YAML names a partition folder in `data.partition_root`. The repository already holds the partition of every reported experiment under `data/partitions/`, so skip this step unless the config is new. Create a new partition from the split with `prepare-experiment-partition`.

```bash
fedbrats prepare-experiment-partition --config configs/fedavg_uneven/baseline.yaml
fedbrats prepare-experiment-partition --config configs/fedavg_even/baseline.yaml \
  --split-path data/splits/brats2021.json
```

The split defaults to `data.split_path` in the YAML, then to `data/splits/brats2021.json`. The command refuses to write into a non-empty folder unless `--overwrite` is given. Several configurations share one partition folder. Every method at missing-modality dropout q = 0.1 reads `data/partitions/missing_uneven/alpha_0_1`, for example. Prepare each folder once. The centralised configurations read `data/partitions/fedavg_uneven/manifest.json`. Prepare the `fedavg_uneven` partition before any centralised run.

`scripts/data_preprocessing/prepare_experiment_partition.sh CONFIG` wraps the same command.

### Create initial weights

Every method starts each seed from a stored, hashed set of weights. This keeps the starting point identical across methods. Create the weights once per seed and model.

```bash
for seed in 42 43 44; do
  fedbrats create-initial-weights --model-config configs/model/unet_baseline.yaml \
    --output-dir artifacts/initial_weights --seed "$seed"
  fedbrats create-initial-weights --model-config configs/model/fedamm.yaml \
    --output-dir artifacts/initial_weights --seed "$seed"
done
```

Each call writes `<model-name>_seed_<seed>.pt` and a JSON file with the weight hash. The experiment YAML files expect these locations.

| Model config | Output directory | Used by |
|---|---|---|
| `configs/model/unet_baseline.yaml` | `artifacts/initial_weights` | Centralised, FedAvg, communication-efficiency, and zero-filled U-Net runs |
| `configs/model/fedamm.yaml` | `artifacts/initial_weights` | FedAMM and FedAvg-MM runs |
| `configs/model/segresnet.yaml` | `artifacts/initial_weights/segresnet` | SegResNet runs |
| `configs/model/segresnet_8.yaml` | `artifacts/initial_weights/segresnet_8` | SegResNet-8 runs |

The Segment Anything Model (SAM) runs also need pretrained checkpoints under `artifacts/pretrained/`. [Models](models.md) describes them.

### Build the container image

Cluster jobs run inside the Apptainer image named in each YAML's `slurm.image`, by default `apptainer/fl-brats-batch.sif`. [Setup](setup.md) and [Cluster](cluster.md) explain how to build it and which partitions to use.

## Launching a run

`scripts/launch_experiment.py` turns one experiment YAML into one run per seed. Preview what it would do with a dry run first. A dry run prints the run IDs, commands, and resolved configuration and creates nothing.

```bash
python scripts/launch_experiment.py configs/fedavg_uneven/baseline.yaml --dry-run
```

Submit to Slurm, the default mode.

```bash
python scripts/launch_experiment.py configs/fedavg_uneven/baseline.yaml
```

Run on the current machine instead.

```bash
python scripts/launch_experiment.py configs/fedavg_uneven/baseline.yaml --mode local
```

The launcher reads `experiment.seeds` and starts one run per seed. `--seed N` launches only that seed. Local mode runs the seeds one after another and stops at the first failure.

In Slurm mode a federated run becomes one server job, one client job array with a task per client, and a cleanup job. The cleanup job cancels a server that is still running long after its clients ended. A centralised run becomes a single job. The launcher chains the seeds of one invocation with `afterany` dependencies. They therefore run in sequence.

| Option | Effect |
|---|---|
| `--mode {slurm,local}` | Submit to Slurm (default) or run in the current shell |
| `--dry-run` | Print the planned runs and create nothing |
| `--seed N` | Launch only seed N |
| `--partition NAME` | Override `slurm.partition` |
| `--image PATH` | Override `slurm.image` |
| `--exclude NODES` | Add comma-separated nodes to `slurm.exclude`. Repeatable |
| `--set KEY=VALUE` | Override any dotted key with a YAML value, for example `--set training.max_rounds=10`. Repeatable |
| `--resume-checkpoint PATH` | Continue a federated run from a round checkpoint up to the configured total rounds |
| `--afterany JOBID` | Start only after that Slurm job ends, whatever its exit status |
| `--no-cleanup` | Skip the cleanup job, for callers that watch the run themselves |

[Configuration](configuration.md) explains how these options combine with the YAML, the `FL_BRATS_*` variables, and command-line overrides.

### Quick check without the dataset

`scripts/smoke_heldout_federation.py` runs a real two-client, two-round Flower federation on tiny synthetic volumes. Use it to confirm that the environment trains and evaluates end to end.

```bash
python scripts/smoke_heldout_federation.py --output outputs/smoke/heldout_check
```

## Family launch scripts

Each experiment family with a launch script has it at `scripts/<family>/launch.sh`. The scripts call `launch_experiment.py` and differ only in how they choose the configuration.

| Script | Configuration |
|---|---|
| `scripts/centralized/launch.sh`, `scripts/fedavg_even/launch.sh`, `scripts/fedavg_uneven/launch.sh`, `scripts/fedavg_missing_modalities_uneven/launch.sh`, `scripts/comm_eff_uneven/launch.sh`, `scripts/fedamm_comm_eff_uneven/launch.sh`, `scripts/model_agnostic_uneven/sam1/launch.sh`, `scripts/model_agnostic_uneven/sam2/launch.sh` | Fixed. All arguments go to the launcher |
| `scripts/fedamm_uneven/launch.sh`, `scripts/fedmpq_uneven/launch.sh`, `scripts/selective_transmission_uneven/launch.sh` | Optional first argument. Without one, or when the first argument is a launcher option such as `--dry-run`, the script runs its default config |
| `scripts/fedavg_train_configs_uneven/launch.sh` | Required first argument |

Pass launcher options after the configuration.

```bash
scripts/fedavg_uneven/launch.sh --dry-run
scripts/fedmpq_uneven/launch.sh configs/fedmpq_uneven/heterogeneous_2_4_8bit.yaml --mode local
scripts/fedavg_train_configs_uneven/launch.sh configs/fedavg_train_configs_uneven/client_sweeps/k5.yaml
```

For any other configuration, call `launch_experiment.py` directly.

## Experiment matrix

Each row is one folder under `configs/`. Brackets in the launch script column give the script's fixed configuration, or its default where the script accepts another.

| Family | What it tests | Configs | Launch script |
|---|---|---|---|
| `centralized` | Pooled training on one graphics processing unit (GPU) as the reference for the federated results, for 100 epochs, to convergence, for a fixed 200 epochs, and with the SegResNet backbones | `configs/centralized/` | `scripts/centralized/launch.sh` (`baseline.yaml`) |
| `fedavg_even` | FedAvg with four equal clients | `configs/fedavg_even/` | `scripts/fedavg_even/launch.sh` (`baseline.yaml`) |
| `fedavg_uneven` | FedAvg with the 50/25/15/10 % quantity-skewed split, the main baseline | `configs/fedavg_uneven/` | `scripts/fedavg_uneven/launch.sh` (`baseline.yaml`) |
| `fedavg_train_configs_uneven` | FedAvg sweeps over K, E, and B, one modality-dropout setting, and E = 5 at K = 5 and K = 8 | `configs/fedavg_train_configs_uneven/{client_sweeps,local_epochs_sweeps,mini_batch_sweeps,modality_sweep}/` | `scripts/fedavg_train_configs_uneven/launch.sh` (config required) |
| `fedavg_missing_modalities_uneven` | FedAvg-MM and the zero-filled U-Net with per-modality dropout, Dirichlet modality partitions, and one missing scan per client | `configs/fedavg_missing_modalities_uneven/` | `scripts/fedavg_missing_modalities_uneven/launch.sh` (`alpha_0_1.yaml`) |
| `fedamm_uneven` | FedAMM on the same missing-modality partitions, plus short diagnostic variants | `configs/fedamm_uneven/` | `scripts/fedamm_uneven/launch.sh` (default `alpha_0_1.yaml`) |
| `fedmpq_uneven` | Upload quantisation at a uniform 4 bits and at 2/4/8 bits per client | `configs/fedmpq_uneven/` | `scripts/fedmpq_uneven/launch.sh` (default `homogeneous_4bit.yaml`) |
| `selective_transmission_uneven` | Top-p sparsification of client updates at p = 0.25, 0.1, and 0.01 | `configs/selective_transmission_uneven/` | `scripts/selective_transmission_uneven/launch.sh` (default `p_0_1.yaml`) |
| `comm_eff_uneven` | Top-p sparsification combined with 2/4/8-bit quantisation, a quantised broadcast, and K = 5 and K = 8 | `configs/comm_eff_uneven/` | `scripts/comm_eff_uneven/launch.sh` (`best_p_0_1.yaml`) |
| `fedavg_mm_comm_eff_uneven` | FedAvg-MM with the combined compression | `configs/fedavg_mm_comm_eff_uneven/` | None |
| `fedamm_comm_eff_uneven` | FedAMM with the combined compression | `configs/fedamm_comm_eff_uneven/` | `scripts/fedamm_comm_eff_uneven/launch.sh` (`best_p_0_1.yaml`) |
| `fedamm_comm_eff_ablation_uneven` | FedAMM with quantisation alone and with sparsification alone. `bigbatch/` holds interim copies for the RTX 3090 partition | `configs/fedamm_comm_eff_ablation_uneven/` | None |
| `model_agnostic_uneven` | FedAvg with SAM 1, SAM 2, SegResNet, and SegResNet-8 in place of the U-Net | `configs/model_agnostic_uneven/{sam1,sam2,segresnet,segresnet_8}/` | `scripts/model_agnostic_uneven/sam1/launch.sh`, `scripts/model_agnostic_uneven/sam2/launch.sh` |
| `smoke` | One-round and four-round end-to-end checks of the missing-modality, FedMPQ, and selective-transmission code paths | `configs/smoke/` | `scripts/smoke/launch_communication_smokes.sh` |
| `cache_check_uneven` | FedAvg for 50 rounds with and without the persistent dataset cache | `configs/cache_check_uneven/` | None |

Both launch scripts run `best_p_0_1.yaml` with p = 0.1, the setting in the report. `best.yaml` keeps p = 0.25, the highest validation Dice of the tuning runs. Each sent value carries a 4-byte index, and p = 0.25 uploads more than quantisation alone. Launch `best.yaml` directly to repeat that variant.

```bash
python scripts/launch_experiment.py configs/comm_eff_uneven/best.yaml
```

`scripts/smoke/launch_communication_smokes.sh` submits `configs/smoke/missing_modalities.yaml`, `fedmpq.yaml`, and `selective_transmission.yaml` in sequence. It reads the `PARTITION` and `IMAGE` environment variables and creates the FedAMM seed 42 weights when they are missing.

## Where a run writes

Every run gets a run ID of the form `<variant>_seed_<seed>_<YYYYMMDD_HHMMSS_ffffff>`, for example `baseline_seed_43_20260926_000643_789028`. Results and logs go into matching folders.

```text
outputs/<family>/<run-id>/   results, checkpoints, metrics
logs/<family>/<run-id>/      resolved config, Slurm output, job IDs
```

The family comes from `experiment.family` and may contain a slash. Sweep runs land in `outputs/fedavg_train_configs_uneven/client_sweeps/`, SAM runs in `outputs/model_agnostic_uneven/sam1/`, and smoke runs in `outputs/smoke/<name>/`. `FL_BRATS_OUTPUT_ROOT` and `FL_BRATS_LOG_ROOT` move both trees elsewhere.

The log folder holds these files.

| File | Content |
|---|---|
| `resolved-launch-config.yaml` | The exact configuration the run received |
| `server-<job>.out`, `server-<job>.err` | Flower server output |
| `client-<job>-<task>.out`, `client-<job>-<task>.err` | Output of each client |
| `cleanup-<job>.out`, `cleanup-<job>.err` | Output of the cleanup job |
| `slurm-job-ids.json` | Server, client, and cleanup job IDs of a federated run |
| `slurm-<job>.out`, `slurm-<job>.err`, `slurm-job-id.txt` | Output and job ID of a centralised run |
| `launcher.out`, `launcher.err` | Output of a `--mode local` run |

## After training

Training saves a checkpoint every round and evaluates the best and final checkpoints on the held-out validation set. Two reports come afterwards. One gives full metrics for every tenth round. The other gives test-cohort metrics for the checkpoint selected on validation.

```bash
fedbrats evaluate-periodic-checkpoints --run-dir outputs/fedavg_uneven/<run-id> --every 10 --test-best
```

List every directory of a resumed run with repeated `--run-dir` options, oldest first. The campaign driver submits this step automatically. [Evaluation](evaluation.md) describes the metrics and the selection rule.

## What a finished run directory contains

A federated run directory holds the files below once training and the post-training evaluation have finished. ET, TC, and WT are the enhancing tumour, tumour core, and whole tumour regions. HD95 is the 95th percentile Hausdorff distance.

| Path | Content |
|---|---|
| `summary.json` | Status, completed rounds, best round, best and final validation Dice, the final round's ET, TC, WT, and mean Dice, convergence rounds, K, E, B, learning rate, seed, strategy, model, initial weight hash, communication totals, Dice against cumulative traffic, training wall-clock seconds, and checkpoint paths |
| `state.json` | Resumable experiment state, including the latest and best checkpoints and cumulative upload and download bytes |
| `resolved_config.yaml` | Configuration as seen by the training code |
| `metrics/validation_rounds.jsonl` | Held-out validation loss and Dice per round, the basis for checkpoint selection |
| `metrics/rounds.jsonl` | Per-round timings, client drift, and loss components |
| `metrics/clients.jsonl` | One record per client and round with training loss, fit time, examples, update norm, and payload bytes |
| `metrics/evaluation_rounds.jsonl` | Per-round evaluation written by runs from before the held-out evaluation refactor. Current runs write `validation_rounds.jsonl` instead |
| `communication/rounds.jsonl` | Upload and download bytes per round |
| `communication/summary.json` | Total and cumulative traffic, Dice against traffic, and training and evaluation time |
| `validation_evaluation/best/`, `validation_evaluation/final/` | Full validation metrics of the best and final checkpoints, including HD95, sensitivity, and the forced modality combinations |
| `periodic_evaluation/` | Full metrics for every tenth round checkpoint |
| `final_evaluation/` | Test-cohort metrics of the selected checkpoint |
| `checkpoints/` | `round_NNNN.pt` for every round, plus `best.pt` and `final.pt` |
| `client_state/` | Optimiser moments and compression residuals of each client for the latest rounds, used on resume |

The `validation_evaluation` and `final_evaluation` folders each hold `metrics.json`, `metadata.json`, and `modality_combinations.jsonl`. `periodic_evaluation/` holds `rounds.jsonl` with one line per evaluated round and a `summary.json`. Missing-modality runs also write `metrics/modality_combinations.jsonl`.

Byte counts are measured directly on the serialised payloads. Timings are not split into network and coordination parts. The `network_transmission_seconds` and `communication_and_coordination_seconds` fields stay null in current runs. Only an external collector on the interconnect could fill the network field. Any communication time derived from the measured round components is a proxy. It mixes serialisation, remote procedure calls, scheduler waiting, and aggregation.

A centralised run directory is simpler. It holds `metrics.jsonl` with one line per epoch, `checkpoints/last.pt` and `checkpoints/best.pt`, and `summary.json` with the test metrics of the best epoch.

## Results and the campaign record

[Experiments.md](../Experiments.md) at the repository root is the record of each experiment, its settings, and its results. Check it before repeating a run. [logs/monitoring/campaign/README.md](../logs/monitoring/campaign/README.md) documents the campaign driver, `scripts/monitoring/campaign.py`, which kept the runs going on the cluster through wall-time limits, node failures, and resumes.
