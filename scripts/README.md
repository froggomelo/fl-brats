# Scripts

This folder holds the command-line entry points for setting up the environment, preparing data, launching experiments, and watching the cluster. Read it to find the right script. [docs/experiments.md](../docs/experiments.md) explains how to run experiments end to end.

## Top-level files

| File | Purpose |
|---|---|
| `launch_experiment.py` | Generic launcher. Turns one experiment YAML into one run per seed, on Slurm or locally |
| `smoke_heldout_federation.py` | Two-client, two-round Flower federation on synthetic volumes. Checks the environment without the dataset |

## Folders

| Folder | Contents |
|---|---|
| `environment/` | `bootstrap.sh` and `bootstrap_cluster.sh` create the Python 3.11 environment with uv. `run_local.sh` and `run_apptainer.sh` run `fedbrats` on the host or in the container. `validate_environment.sh` runs the environment and configuration checks. `install_aspera.sh` installs the IBM Aspera client used for the download |
| `data_preprocessing/` | BraTS 2021 download (`download_brats2021.py`, `brats_download.sbatch`), indexing, validation, and preprocessing (`index_brats.sh`, `validate_brats.sh`, `preprocess_brats.sh`), split and partition creation (`create_dataset_split.sh`, `create_fixed_partition.sh`, `prepare_experiment_partition.sh`), and `inspect_data_pipeline.sh` |
| `monitoring/` | `campaign.py`, the campaign driver that keeps prioritised runs going on the cluster. `sync_mscluster_logs.sh` mirrors cluster logs and outputs to this machine. `watch_squeue.sh` refreshes the Slurm queue view |
| `centralized/` | `launch.sh` for the centralised baseline |
| `fedavg_even/`, `fedavg_uneven/` | `launch.sh` for the federated averaging (FedAvg) baselines on the even and uneven splits |
| `fedavg_train_configs_uneven/` | `launch.sh CONFIG` for the client, local-epoch, mini-batch, and modality sweeps |
| `fedavg_missing_modalities_uneven/` | `launch.sh` for FedAvg with missing modalities |
| `fedamm_uneven/` | `launch.sh [CONFIG]` for FedAMM, the missing-modality method with prototype alignment |
| `fedmpq_uneven/` | `launch.sh [CONFIG]` for the mixed-precision quantisation (FedMPQ) runs |
| `selective_transmission_uneven/` | `launch.sh [CONFIG]` for top-p sparsification |
| `comm_eff_uneven/` | `launch.sh` for the combined communication-efficient setting |
| `fedamm_comm_eff_uneven/` | `launch.sh` for FedAMM with the combined compression |
| `model_agnostic_uneven/` | `sam1/launch.sh` and `sam2/launch.sh` for the Segment Anything Model (SAM) runs |
| `smoke/` | `launch_communication_smokes.sh` submits the missing-modality, FedMPQ, and selective-transmission smoke runs in sequence |

Each `launch.sh` calls `launch_experiment.py` and forwards launcher options such as `--dry-run`, `--mode local`, and `--seed`. Scripts marked `[CONFIG]` take an optional configuration path as their first argument. The families without a script run through `launch_experiment.py` directly. [docs/experiments.md](../docs/experiments.md) lists every family, its configurations, and its default.

The Slurm batch files that these scripts submit live in `slurm/`, not here. [slurm/README.md](../slurm/README.md) describes them.
