# Slurm job scripts

This folder holds the Slurm job scripts. Training runs are submitted through `scripts/launch_experiment.py`, which reads an experiment YAML and uses the scripts in `_shared/`. Read [docs/cluster.md](../docs/cluster.md) before submitting anything.

| Path | Contents |
|---|---|
| `_shared/` | Server, client, evaluation, and environment wrapper scripts used by every launcher run, plus the initial weights job. |
| `environment/` | One-off account setup and the environment check job. |
| `data_preprocessing/` | BraTS validation, preprocessing, partitioning, and experiment partition preparation. |

Run `make create-dirs` after adding a script. It creates the folder named in every `#SBATCH --output` and `--error` line.
