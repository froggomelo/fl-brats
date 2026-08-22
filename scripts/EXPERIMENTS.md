# Experiment launchers

Every experiment is defined by YAML under `configs/`. YAML values are
authoritative; CLI flags only select launch mode or override one repetition seed.

```bash
python scripts/launch_experiment.py configs/fedavg_uneven/baseline.yaml
python scripts/launch_experiment.py configs/fedavg_uneven/baseline.yaml --mode local
python scripts/launch_experiment.py configs/fedavg_uneven/baseline.yaml --dry-run
```

Each repetition gets `outputs/<family>/<run-id>/` and
`logs/<family>/<run-id>/`. SLURM output, error, job ID, and the resolved launch
configuration live together in that log directory.
