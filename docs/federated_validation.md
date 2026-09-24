# Held-out federated evaluation

The split JSON is the source of truth: `train_subjects` are optimized on clients,
`validation_subjects` select models, and `test_subjects` report final performance.
All three lists must be unique and pairwise disjoint. Partition creation and client
startup enforce these boundaries. Clients receive training subjects only.

The server reads a dedicated research validation cohort from the dataset index.
It does not pool client training records. This requires the server job to have
read access to validation volumes and enough memory for one model and inference.
The natural loader uses deterministic evaluation transforms, full volumes, and
batch size 1 by default. Existing preprocessing defaults are preserved.

For each successful aggregation, the strategy saves `checkpoints/round_NNNN.pt`,
then runs one global natural validation. Flower's initial round-zero evaluation
is skipped. `best.pt` is a byte-for-byte copy of the round checkpoint with the
highest natural held-out mean Dice; ties keep the earlier round. Early stopping
uses the same score, existing minimum rounds, and patience measured in rounds.
The successful local/distributed completion path writes `final.pt`, `summary.json`,
and a completed (or early-stopped) `state.json`.

FedAvg, FedProx, FedAMM, sparse transport and quantized transport share this
checkpoint lifecycle. Modality-weighted aggregation and prototype logic remain.
The classic Flower API remains in use; installed Flower 1.32.1 was exercised.
Its server signal handlers require local orchestration to run the server on the
main thread, with clients in background threads.

## Configuration

```yaml
data:
  split_path: /path/to/frozen-split.json
  # Optional when the recorded index moved; contents must match the split hash.
  index_path: /path/to/dataset_index.csv
evaluation:
  natural_every_rounds: 1
  expensive_metrics_every_rounds: 10  # 0 disables periodic HD95/sensitivity
  robustness_every_rounds: 10        # 0 disables periodic robustness
  evaluate_best_checkpoint: true
  evaluate_final_checkpoint: true
  client_diagnostics: false
  batch_size: 1
  modality_seed: 42
```

Existing `missing_modalities.evaluation` keys remain accepted. Top-level
`evaluation` keys take precedence. `fraction_evaluate`/`min_evaluate_clients`
are ignored unless `client_diagnostics` is explicitly enabled; old nonzero
fractions emit a warning. Optional client results are training-partition
**diagnostics**, and cannot select checkpoints. The older
`evaluate-federated-checkpoint` CLI remains available with that explicit label.
The launcher no longer submits those jobs by default.

Natural loss and Dice normally run every round. HD95 and sensitivity are added
in the same inference pass at the configured interval. If missing modalities
are enabled, robustness evaluates all 15 nonempty masks at its interval, using
the same held-out subjects and the full source images. It never selects models.
After training, best and final checkpoints receive full metrics on validation,
plus all 15 combinations when enabled, under `validation_evaluation/{best,final}`.
These post-training passes are separate from the once-per-round selection pass.

Natural missingness uses `missing_modalities` with a fixed evaluation seed of 42,
independent of training seed, and client ID `global_evaluation`. Masks are generated
once per evaluator, recorded in metadata and reused. Optional
`evaluation.natural_missingness` overrides the missingness policy for the evaluation
cohort. Use identical split, preprocessing, evaluation policy and modality seed for
all comparable algorithms. Full images are preprocessed identically before masking;
robustness never attempts to restore modalities removed from client manifests.

## Artifacts and timing

`metrics/validation_rounds.jsonl` contains the round, checkpoint, loss, region Dice,
mean Dice, subjects, elapsed evaluation time, and flags for expensive/robustness
metrics. Region fields use `dice_wt`, `hd95_wt`, `sensitivity_wt`, etc.
`evaluation_metadata.json` records subject IDs and hashes, split file hash, index
hash, preprocessing, loss, natural masks, threshold (0.5), metric empty-mask policy
and cadence. HD95 is in preprocessed voxel units, preserving the existing metric.

`metrics/rounds.jsonl` records measured aggregation/serialization, checkpoint writes,
natural validation, robustness, expensive metric computation, total round wall time,
and the slowest reported local training time. All client training times are in
`metrics/clients.jsonl`. Expensive metric time is a subset of natural validation
wall time; do not add it twice. Network-only time is unavailable, not estimated.
Round wall time spans fit configuration through checkpoint selection; optional
client diagnostics and post-training evaluation are outside that interval.

Historical `evaluation_rounds.jsonl` records client training-partition evaluation.
It is not held-out validation. Historical wall-clock values include distributed
client evaluation overhead and cannot be compared directly to the corrected
pipeline without disclosing that difference.

Resume training requires a new run directory. Historical training-partition best
scores and patience history are reset. Corrected runs recover best selection from
held-out rows through the resumed round and reject changed evaluation policies.
Existing checkpoint tensors and historical run directories are never rewritten.

## Revalidate retained checkpoints without training

Use a Python environment with this repository on `PYTHONPATH` (or an editable
installation). The installed executable is `fedbrats`; `python -m fl_brats` is
identical. In a container, bind dataset/output paths and expose the updated source.

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export RUN=/absolute/path/to/completed-run
export SPLIT=/absolute/path/to/original-frozen-split.json
export INDEX=/absolute/path/to/dataset_index.csv
python -m fl_brats revalidate-federated-run \
  --run-dir "$RUN" --split "$SPLIT" --index-path "$INDEX" --dry-run
python -m fl_brats revalidate-federated-run \
  --run-dir "$RUN" --split "$SPLIT" --index-path "$INDEX" --device cuda
```

The dry run discovers and numerically orders checkpoints, validates split disjointness,
and reports the cohort identity/index availability without loading images or writing
outputs. Actual evaluation restores each checkpoint's model configuration and uses
the run's saved preprocessing/loss configuration. Dice/loss are the default;
`--expensive-metrics` explicitly adds HD95/sensitivity to the entire search.

Outputs are confined to `revalidation/`: atomic `validation_rounds.jsonl`,
`metadata.json`, `state.json`, `summary.json`, and `best_revalidated.pt`.
An interrupted run continues with the first unevaluated checkpoint; completed
checkpoint hashes and evaluation identity must still match. A file lock prevents
concurrent writers. Historical `best.pt`, metrics, summary, and state are preserved.
An interruption before a checkpoint row commits may repeat that one checkpoint.
The summary identifies historical metrics as training-partition evaluation.

Sequential batch, covering even/uneven seeds 42, 43 and 44:

```bash
python -m fl_brats revalidate-federated-runs \
  --run-dir "$EVEN42" --run-dir "$EVEN43" --run-dir "$EVEN44" \
  --run-dir "$UNEVEN42" --run-dir "$UNEVEN43" --run-dir "$UNEVEN44" \
  --split "$SPLIT" --index-path "$INDEX" --device cuda
# Alternatively: --run-glob '/absolute/root/fedavg_*/*' (quote the glob).
```

For Slurm, put those six absolute run directories in a text file, one per line
(no header). The existing environment wrapper sources `.env` and binds the data
root. Set `APPTAINER_IMAGE` if needed; `USE_APPTAINER=false` uses the host environment.
Submit only after reviewing the dry-run output:

```bash
printf '%s\n' "$EVEN42" "$EVEN43" "$EVEN44" "$UNEVEN42" "$UNEVEN43" "$UNEVEN44" > "$PWD/fedavg-runs.txt"
sbatch --array=0-5%1 --partition=stampede --cpus-per-task=8 --mem=64G \
  --time=24:00:00 --output='revalidation-%A-%a.log' \
  slurm/_shared/revalidate_federated_runs.sbatch "$PWD/fedavg-runs.txt" "$SPLIT" "$INDEX"
```

Keep `fedavg-runs.txt` on a shared filesystem accessible to compute nodes. Choose the site's GPU partition/resource flags; `stampede` here follows
this repository's existing configuration. `%1` serializes runs; raise it to match
the number of available GPUs. The wall limit is a starting request, not a measured
runtime guarantee. Rerun the same command after interruption to resume.

## Final test reporting (no checkpoint search)

```bash
python -m fl_brats evaluate-final-run --run-dir "$RUN" \
  --checkpoint revalidation/best_revalidated.pt --split "$SPLIT" \
  --index-path "$INDEX" --device cuda --robustness
# Explicit predetermined final checkpoint; works even if final.pt is absent:
python -m fl_brats evaluate-final-run --run-dir "$RUN" --final-round 150 \
  --split "$SPLIT" --index-path "$INDEX" --device cuda --robustness
```

This command exclusively uses `test_subjects` and records full Dice, HD95 and
sensitivity. `--robustness` requests all 15 combinations; otherwise the saved
missing-modality configuration determines whether they run. Results go to
`final_evaluation/` or `final_evaluation_round_0150/`. Metadata explicitly records
`predetermined_final` and round 150 for `--final-round 150`; there is no implicit
last-round reconstruction. Existing final-evaluation directories are refused;
use a new `--output-dir` for another report. Test results never modify any alias,
validation history, or selection state.

For a Slurm final evaluation using the existing environment wrapper:

```bash
sbatch --partition=stampede --cpus-per-task=8 --mem=64G --time=04:00:00 \
  --export=ALL,PYTHONPATH="$PWD/src" \
  slurm/_shared/run_experiment.sbatch false unused \
  bash slurm/_shared/run_with_configured_environment.sh true apptainer/fl-brats.sif \
  python -m fl_brats evaluate-final-run --run-dir "$RUN" \
  --checkpoint revalidation/best_revalidated.pt --split "$SPLIT" \
  --index-path "$INDEX" --device cuda --robustness
```

## Compute planning

No production-volume inference benchmark is available in this workspace. Let V
be measured seconds for one complete held-out validation checkpoint, including
loading, preprocessing and Dice; let E be seconds for one test pass with all
metrics. Estimate one 150-checkpoint run as `150 V`, all six as `900 V`, natural
final test as `E`, and natural plus 15-combination test as approximately `16 E`.
Checkpoint I/O and hashes add overhead. HD95 runtime depends strongly on masks.

Illustrative assumptions (not measurements): V=1–5 minutes gives 2.5–12.5 GPU-hours
per run and 15–75 GPU-hours for six. E=2–10 minutes gives 2–10 minutes for natural
test, or 32–160 minutes with all combinations. Reserve one GPU per concurrent job,
CPU workers for NIfTI preprocessing, and begin with 64 GB host RAM; GPU memory must
fit the saved architecture's full-volume inference. Memory usage is independent
of checkpoint count. Additional storage is roughly one checkpoint per run plus
small JSON artifacts. Calibrate these estimates using the first completed rows;
CPU-only synthetic smoke timings must not be extrapolated to BraTS.
