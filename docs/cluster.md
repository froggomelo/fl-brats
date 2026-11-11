# Running on the cluster

This guide covers running experiments on the Wits mscluster through Slurm, the cluster's job scheduler. Read it after [setup.md](setup.md), once the environment, `.env`, and Apptainer images are in place.

## Before the first job

Work from the repository checkout on the login node. The login node only edits files and submits jobs. Every install, check, and training step runs inside a Slurm job.

Run these once per account and again after pulling changes that add scripts.

```bash
make create-dirs
sbatch slurm/environment/environment_check.sbatch
```

`make create-dirs` creates every log folder that an `#SBATCH --output` line names. Slurm does not start a job whose output folder is missing.

## Partitions and images

Jobs use three partitions. The hardware notes come from the experiment configs and [`logs/environment/cluster_nodes.md`](../logs/environment/cluster_nodes.md).

| Partition | GPU per node | Use | Image picked by `select_image.sh` |
|---|---|---|---|
| `batch` | RTX 3060 | All final training and post-training evaluation. The shared experiment base requests 6 CPUs and 29000M memory per job. | `apptainer/fl-brats-batch.sif` |
| `bigbatch` | RTX 3090 | Interim runs while `batch` was down, and older baselines. | `apptainer/fl-brats-batch.sif` |
| `stampede` | GTX 1060 | CPU work such as data preparation and initial weights. The batch image's PyTorch has no kernels for this GPU. | `apptainer/fl-brats-stampede.sif` |

The cluster rejects generic resource, or GRES, requests. Never pass `--gres` or set `slurm.gres`. The GPU comes with the node. Per-user job limits for each partition are recorded in `configs/campaign/deadline_2026_10.yaml`.

Runs launched from an experiment YAML use the image in `slurm.image`. The shared base `configs/_experiment_base/federated_uneven.yaml` sets `apptainer/fl-brats-batch.sif`. Changing the partition with `--partition` does not change the image. Pass `--image` as well whenever the new partition needs a different image. Standalone `sbatch` scripts call `apptainer/select_image.sh`, which picks the image from the job's partition as shown in the table.

## How `slurm/` is organised

`slurm/` holds the job scripts in three folders. Training is never submitted from here directly. `scripts/launch_experiment.py` submits the jobs in `slurm/_shared/` for every experiment family.

| Path | Contents |
|---|---|
| `slurm/_shared/` | Server, client, evaluation, and wrapper scripts used by `scripts/launch_experiment.py`, plus the initial weights job. |
| `slurm/environment/` | `setup_environment.sbatch` and `environment_check.sbatch`. |
| `slurm/data_preprocessing/` | BraTS validation, preprocessing, partitioning, and experiment partition preparation. |

The scripts in `_shared/` are listed below.

| Script | Role |
|---|---|
| `federated_server.sbatch` | Writes `server_address.txt` into the run folder, then starts `fedbrats start-federated-server`. |
| `federated_client.sbatch` | One array task per client. Waits for the server address and port, then starts `fedbrats start-federated-client`. |
| `run_with_configured_environment.sh` | Sources `.env`, checks the data roots, runs the GPU check, and wraps the command in `apptainer exec --nv`. |
| `run_experiment.sbatch` | Minimal Apptainer wrapper for single commands such as evaluation aggregation. |
| `evaluate_federated_checkpoint.sbatch` | Optional per-client checkpoint diagnostics. |
| `create_initial_weights.sbatch` | Creates the initial weights of one model for one seed. |
| `revalidate_federated_runs.sbatch` | Array job that re-scores finished runs on the held-out split. [`logs/monitoring/revalidation/implementation_record.md`](../logs/monitoring/revalidation/implementation_record.md) records why it exists. |

Launch any family through `scripts/<family>/launch.sh` or by passing its config to the launcher.

```bash
python3 scripts/launch_experiment.py configs/fedamm_uneven/alpha_0_1.yaml
```

## Submit an experiment

`scripts/launch_experiment.py` turns one experiment YAML into Slurm jobs. It submits one run per seed listed under `experiment.seeds`.

```bash
python3 scripts/launch_experiment.py configs/fedavg_uneven/baseline.yaml
```

Preview the commands and resolved config without submitting anything.

```bash
python3 scripts/launch_experiment.py configs/fedavg_uneven/baseline.yaml --dry-run
```

| Flag | Effect |
|---|---|
| `--mode local` | Runs in the current shell instead of submitting. The default is `slurm`. |
| `--seed N` | Runs one seed instead of the YAML list. |
| `--partition NAME` | Overrides `slurm.partition`. |
| `--image PATH` | Overrides `slurm.image`. |
| `--resume-checkpoint PATH` | Continues a federated run from a round checkpoint. |
| `--afterany JOBID` | Starts only after that job ends, whatever its exit status. |
| `--set KEY=VALUE` | Overrides a dotted config key with a YAML value, for example `slurm.exclusive=true`. Repeatable. |
| `--exclude NODES` | Adds comma-separated nodes to `slurm.exclude`. Repeatable. |
| `--no-cleanup` | Skips the job that cancels a stuck server. The campaign driver uses it. |

Resources come from the `slurm:` block of the YAML. The keys are `partition`, `time`, `cpus_per_task`, `memory`, `exclude`, `require_gpu`, `use_apptainer`, `image`, and `bind_source`. `require_gpu` and `bind_source` reach the jobs as `FL_BRATS_REQUIRE_GPU` and `FL_BRATS_BIND_SOURCE`. The wrapper scripts `scripts/<family>/launch.sh` call the launcher with a fixed config and pass any extra flags through. [experiments.md](experiments.md) lists the configs for each family.

## Standalone job scripts

Data preparation, environment checks, and initial weights call `sbatch` directly. They take their settings from exported variables instead of a YAML file.

| Variable | Read by | Effect |
|---|---|---|
| `FL_BRATS_DATA_PARTITION` | `prepare_fedavg_baseline_partitions_submit.sh`, `prepare_training_sweep_partitions_submit.sh` | Slurm partition for building client partitions. Defaults to `stampede`. |
| `FL_BRATS_OVERWRITE_PARTITIONS` | The same two scripts | Set to `1` to replace existing partition manifests. |

`brats_validate.sbatch`, `brats_preprocess.sbatch`, and `brats_partition.sbatch` put shell variables in their `#SBATCH` lines. Slurm does not expand those. Pass the partition and resources on the command line for these three.

```bash
sbatch --partition=stampede --cpus-per-task=4 --mem=32G slurm/data_preprocessing/brats_preprocess.sbatch
```

Prepare the even and uneven full-modality partitions from the canonical split before the first baseline run.

```bash
bash slurm/data_preprocessing/prepare_fedavg_baseline_partitions_submit.sh
```

The script sources `.env` and reads the split from `FL_BRATS_SPLIT_PATH`. Wait for both jobs to succeed before launching `configs/fedavg_uneven/baseline.yaml` or `configs/fedavg_even/baseline.yaml`. [data.md](data.md) describes the full data pipeline.

## Distributed server and client jobs

A federated run with K clients becomes three Slurm submissions.

1. The server job runs `slurm/_shared/federated_server.sbatch`. It writes `<host>:<port>` to `server_address.txt` in the run folder. The port is `federated.server_port`, 8080 in the shared base.
2. The client job is an array of K tasks running `slurm/_shared/federated_client.sbatch`. Each task waits up to 20 minutes for the address file and up to 30 minutes for the server port. Task `i` joins as `client_i`.
3. A cleanup job starts once every client has ended. It cancels the server only if the server is still running after `slurm.finalize_grace_minutes`, 180 minutes by default. A healthy server finishes its best and final evaluation within that window.

Server and clients share one resource request. Each client therefore holds its own node and GPU. A centralised run is a single job. The launcher writes the job IDs to `slurm-job-ids.json` in the run's log folder.

Every job goes through `slurm/_shared/run_with_configured_environment.sh`. With `require_gpu: true` it reads the GPU compute capability through `nvidia-smi`. A node with no working GPU or a capability below 7.0 makes the job exit with code 75 and print `FL_BRATS_GPU_UNAVAILABLE` to its `.err` file. Resubmit the run with that node excluded.

```bash
python3 scripts/launch_experiment.py configs/fedavg_uneven/baseline.yaml --exclude mscluster129
```

[`logs/environment/cluster_nodes.md`](../logs/environment/cluster_nodes.md) lists the nodes found broken, slow, or unable to start jobs. The campaign file excludes them from every submission.

## Resume an interrupted run

Federated runs save `checkpoints/round_NNNN.pt` after every round. Pass one to the launcher to continue the run.

```bash
python3 scripts/launch_experiment.py \
  configs/fedavg_train_configs_uneven/local_epochs_sweeps/e5.yaml \
  --seed 42 \
  --resume-checkpoint outputs/fedavg_train_configs_uneven/<run>/checkpoints/round_0135.pt
```

`training.max_rounds` stays the global target. A round 135 checkpoint with a target of 150 runs 15 more rounds. Resume creates a new continuation run folder and never modifies the source run. It restores the model, the completed round, early-stopping progress, and cumulative payload counters. The metric histories up to the checkpoint round are copied across, and `best.pt` is rebuilt from the copied validation rows. New checkpoints keep the global numbering. The first one in this example is `round_0136.pt`.

The server refuses to resume when the validation subjects, test subjects, preprocessing config, modality policy, or metric policy differ from the source run.

## Evaluation cadence

The server scores the global model on the held-out validation set after every round. That pass uses natural modality masks and reports Dice and loss. It drives `best.pt` and early stopping. At the end of training the server evaluates `best.pt` and `final.pt` with the full metrics under `validation_evaluation/best/` and `validation_evaluation/final/`. Missing-modality runs also get all 15 modality combinations there.

The 95th percentile Hausdorff distance, known as HD95, is one of three reporting-only measures. HD95, sensitivity, and the modality robustness pass run after training on every tenth round checkpoint. None of them affects model selection.

```bash
fedbrats evaluate-periodic-checkpoints --run-dir RUN --run-dir CONTINUATION \
  --every 10 --device cuda --test-best
```

List every folder of a resumed run, oldest first. Results go to `periodic_evaluation/` in the newest folder. `--test-best` then reports test metrics for the validation-selected checkpoint in `final_evaluation/`. [evaluation.md](evaluation.md) defines the metrics.

## Logs and outputs

Launcher runs share one run ID, `<variant>_seed_<seed>_<timestamp>`, between the output and log folders.

| Path | Contents |
|---|---|
| `outputs/<family>/<run_id>/` | Checkpoints, metrics, communication records, `resolved_config.yaml`, `state.json`, and the evaluation folders. |
| `logs/<family>/<run_id>/resolved-launch-config.yaml` | The config the jobs read, after seed, paths, and overrides are applied. |
| `logs/<family>/<run_id>/server-<jobid>.out`, `.err` | Server output. |
| `logs/<family>/<run_id>/client-<jobid>-<task>.out`, `.err` | One pair per client. |
| `logs/<family>/<run_id>/cleanup-<jobid>.out` | Cleanup job output. |
| `logs/<family>/<run_id>/slurm-<jobid>.out`, `.err` | Centralised job output. |
| `logs/<family>/<run_id>/launcher.out`, `.err` | Output of a `--mode local` run. |

`FL_BRATS_OUTPUT_ROOT` and `FL_BRATS_LOG_ROOT` move both trees. Standalone `sbatch` scripts write to the fixed paths in their `#SBATCH` lines, such as `logs/environment/` and `logs/data_preprocessing/`. Some older scripts write under `logs/slurm/`.

## Monitor and cancel jobs

Use the standard Slurm tools to watch and stop jobs.

```bash
squeue -u "$USER"
scancel JOBID
```

`scripts/monitoring/watch_squeue.sh` refreshes `squeue` every few seconds and passes its arguments through. `scripts/monitoring/sync_mscluster_logs.sh` runs on a workstation and mirrors the cluster's `logs/` and `outputs/` into the local checkout.

For many runs, use the campaign driver `scripts/monitoring/campaign.py`. It submits runs in priority order, resubmits failed runs from their latest checkpoint, excludes failing nodes, and queues post-training evaluation. [`logs/monitoring/campaign/README.md`](../logs/monitoring/campaign/README.md) explains how to run it and records its decisions.
