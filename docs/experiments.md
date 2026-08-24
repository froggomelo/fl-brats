# Experiment matrix and launching

The experiment definitions mirror `Experiments.md` under `configs/`. Shared
defaults live in `configs/_experiment_base/`; child YAML files use relative
`extends` and override only their experiment-specific values. YAML is authoritative.
Supported precedence is: inherited YAML, selected experiment YAML, optional
`FL_BRATS_*` path/device overrides, then explicit CLI `key=value` overrides.

Use the matching script under `scripts/<experiment>/`, or call the launcher:

```bash
python scripts/launch_experiment.py configs/fedavg_uneven/baseline.yaml --dry-run
python scripts/launch_experiment.py configs/fedavg_uneven/baseline.yaml
```

The default mode submits one SLURM job per configured seed. `--mode local` runs
the same resolved YAML locally. Each run gets an isolated output and log directory.

Prepare experiment-specific partitions before training:

```bash
fedbrats prepare-experiment-partition \
  --config configs/fedavg_even/baseline.yaml \
  --split-path data/splits/brats2021.json
```

Create deterministic initial weights once for each seed and applicable model:

```bash
for seed in 42 43 44; do
  fedbrats create-initial-weights --model-config configs/model/unet_baseline.yaml \
    --output-dir artifacts/initial_weights --seed "$seed"
  fedbrats create-initial-weights --model-config configs/model/fedamm.yaml \
    --output-dir artifacts/initial_weights --seed "$seed"
done
```

The `comm_eff_uneven/best.yaml` and `fedamm_comm_eff_uneven/best.yaml` choices are
marked provisional. Update their selected values after the prerequisite tuning
runs establish the actual best sparsification fraction, bit-width policy, and alpha.
