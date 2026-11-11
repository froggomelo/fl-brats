# Evaluation

This guide explains the segmentation metrics, how a federated run selects its checkpoint on held-out validation subjects, and the commands that add periodic metrics, revalidate old runs, and report test results. Read it before interpreting a run directory or reporting a number.

## Metrics

Every model family shares one evaluator, `segmentation_metrics` in `src/fl_brats/metrics/segmentation.py`. Runs of the Segment Anything Model (SAM) reassemble their slice predictions into a volume first. The evaluator thresholds the sigmoid of each logit at 0.5 and scores three tumour regions per subject.

### Regions

BraTS, the Brain Tumour Segmentation challenge dataset, labels each voxel 0 for background, 1 for necrotic tumour core, 2 for peritumoural oedema, and 4 for enhancing tumour. `brats_to_regions` in `src/fl_brats/data/labels.py` turns the labels into three overlapping binary channels. Any other label value raises an error.

| Region | Labels | Channel |
|---|---|---|
| Whole tumour (WT) | 1, 2, 4 | 0 |
| Tumour core (TC) | 1, 4 | 1 |
| Enhancing tumour (ET) | 4 | 2 |

### Scores

| Metric | Definition | Range |
|---|---|---|
| Dice | Twice the overlap divided by the sum of predicted and true voxels | 0 to 1, higher is better |
| HD95 | 95th-percentile Hausdorff distance, the larger of the two directed 95th-percentile distances between the predicted and true region surfaces | Voxels of the preprocessed volume, lower is better |
| Sensitivity | True voxels that the prediction covers, divided by all true voxels | 0 to 1, higher is better |

Each score is computed per subject and region, then averaged over subjects. `mean_dice`, `mean_hd95`, and `mean_sensitivity` are the arithmetic means over the three regions.

Empty masks follow a fixed policy that keeps every value finite.

| Case | Dice | Sensitivity | HD95 |
|---|---|---|---|
| Truth and prediction both empty | 1 | 1 | 0 |
| Only one of them empty | 0 | 0 | Diagonal of the image grid |

HD95 uses MONAI's `compute_hausdorff_distance` on the CPU. On a GPU an exact equivalent of the same computation runs instead. HD95 dominates evaluation time. `evaluation.hd95_workers` moves it into a pool of CPU processes while the GPU continues.

Federated validation and test files name the fields `dice_wt`, `hd95_wt`, `sensitivity_wt`, and so on for each region.

## Data roles

The split JSON fixes three disjoint subject lists. `train_subjects` are divided among the clients, `validation_subjects` select checkpoints, and `test_subjects` are used only for final reporting. Split loading rejects duplicates and any overlap. Client start-up rejects a manifest that contains a non-training subject. [Data](data.md) describes how the split is created.

The server reads the validation cohort directly from the dataset index. It never pools client data. It finds the split through `data.split_path`, or `split_manifest.json` in the partition root, and the index through `data.index_path` or the path the split records. The index must match the hash stored in the split.

## Selection during training

After each aggregated round the server saves `checkpoints/round_NNNN.pt` and evaluates the new global model on the validation cohort. The pass scores loss and Dice. Every validation subject keeps its natural modality mask, which is all four modalities for full-modality runs. [Missing modalities](missing_modalities.md) explains the masks of the other runs.

`checkpoints/best.pt` is a copy of the round checkpoint with the highest validation `mean_dice`. A later round must beat it strictly. Ties therefore keep the earlier round. Early stopping uses the same score. After `federated.minimum_rounds`, training stops once `federated.early_stopping_patience` rounds pass without a new best. At the end the last round is copied to `checkpoints/final.pt`. Flower's round-zero evaluation is skipped.

The `evaluation` section sets the cadence. `configs/_experiment_base/federated_uneven.yaml` holds the shipped values.

| Key | Shipped | Effect |
|---|---|---|
| `natural_every_rounds` | `1` | Rounds between validation passes. Only evaluated rounds can become `best.pt`. |
| `expensive_metrics_every_rounds` | `0` | Rounds between passes that add HD95 and sensitivity. `0` disables them. |
| `robustness_every_rounds` | `0` | Rounds between passes over all 15 modality combinations. `0` disables them. |
| `hd95_workers` | `5` | CPU processes for HD95 |
| `evaluate_best_checkpoint`, `evaluate_final_checkpoint` | `true` | Full metrics for `best.pt` and `final.pt` after training |
| `client_diagnostics` | `false` | Lets clients also evaluate their own training data |
| `modality_seed` | `42` | Seed of the held-out modality masks |
| `batch_size` | `1` | Validation batch size |

The code defaults for the two intervals are 10. The shipped configs set them to 0. Computing HD95 and the combination pass during training stalled the whole federation for hours every tenth round. Those metrics now run after training on the saved checkpoints. Selection only ever uses natural validation Dice.

Keys under `missing_modalities.evaluation` are also read. Top-level `evaluation` keys take precedence.

Client diagnostics score the training partitions. They can never select a checkpoint. `fedbrats evaluate-federated-checkpoint` and `fedbrats aggregate-federated-checkpoint-evaluation` produce the same diagnostics for a saved checkpoint.

## What a finished run contains

| Path | Content |
|---|---|
| `metrics/validation_rounds.jsonl` | One row per validated round with loss, region and mean scores, timing, and the checkpoint path |
| `metrics/modality_combinations.jsonl` | Combination results from any in-training robustness pass |
| `evaluation_metadata.json` | Validation subject IDs and their hash, split and index hashes, preprocessing and loss configuration, natural masks, threshold, empty-mask policy, and cadence |
| `validation_evaluation/best/`, `validation_evaluation/final/` | Full metrics for `best.pt` and `final.pt`, with all 15 combinations for missing-modality runs |
| `metrics/rounds.jsonl` | Per-round timing of aggregation, checkpoint writing, validation, and the slowest client |
| `summary.json` | Best round and its validation Dice, final validation Dice, rounds to reach 90 % and 95 % of the final Dice, and the communication summary |

Each directory under `validation_evaluation/` holds `metadata.json`, `modality_combinations.jsonl`, and `metrics.json`. `metrics.json` is written last, and its presence marks a complete evaluation.

A resumed run needs a new run directory. It copies the earlier rows up to the resumed round, recovers the best round from them, and refuses to continue if the validation subjects, preprocessing, modality policy, natural masks, or metric policy changed.

## Periodic evaluation

`fedbrats evaluate-periodic-checkpoints` computes full metrics, including HD95, sensitivity, and the 15-combination pass for missing-modality runs, on every Nth round checkpoint of a finished run. It uses the same evaluator and the run's saved `resolved_config.yaml`.

```bash
fedbrats evaluate-periodic-checkpoints --run-dir outputs/<run> --every 10 --device cuda --dry-run
fedbrats evaluate-periodic-checkpoints --run-dir outputs/<run> --every 10 --device cuda --test-best
```

| Option | Effect |
|---|---|
| `--run-dir` | Run directory. Repeat it oldest first for a resumed run to search the whole chain. |
| `--every` | Evaluate rounds divisible by this number, default 10 |
| `--split`, `--index-path` | Override the split and index recorded in the run |
| `--hd95-workers` | Override `evaluation.hd95_workers` |
| `--test-best` | Afterwards, run the test report for the checkpoint selected during training |
| `--dry-run` | Print the rounds to evaluate without loading images |

Results go to `periodic_evaluation/` in the newest run directory, as `rounds.jsonl`, `modality_combinations.jsonl`, `metadata.json`, `state.json`, and `summary.json`. The job is resumable. A rerun skips evaluated rounds, checks that their checkpoint hashes are unchanged, and refuses to continue if the evaluation identity changed. A lock file stops two jobs writing the same output.

`--test-best` writes `final_evaluation/` in the newest run directory from the `best_checkpoint` recorded in `state.json`. It skips the step if a complete report already exists, and it requires the run to have finished.

On the cluster, wrap the command in the environment script. `scripts/monitoring/campaign.py` submits it this way for every finished run.

```bash
sbatch --partition=batch --cpus-per-task=6 --mem=29000M --time=24:00:00 \
  --export=ALL,FL_BRATS_REQUIRE_GPU=1,FL_BRATS_BIND_SOURCE=1 \
  slurm/_shared/run_with_configured_environment.sh true apptainer/fl-brats-batch.sif \
  fedbrats evaluate-periodic-checkpoints --run-dir outputs/<run> --every 10 --device cuda --test-best
```

SAM runs need `apptainer/fl-brats-batch.sam.sif` in place of the batch image.

## Final test reporting

`fedbrats evaluate-final-run` scores one checkpoint on the test cohort with Dice, HD95, and sensitivity. It never searches checkpoints and never changes selection state. Give exactly one of `--checkpoint` or `--final-round`.

```bash
fedbrats evaluate-final-run --run-dir outputs/<run> --split <split.json> \
  --checkpoint checkpoints/best.pt --device cuda
fedbrats evaluate-final-run --run-dir outputs/<run> --split <split.json> \
  --final-round 150 --device cuda --robustness
```

| Option | Effect |
|---|---|
| `--split` | Required. The split whose `test_subjects` are scored. |
| `--checkpoint` | A selected checkpoint, absolute or relative to the run directory |
| `--final-round` | A predetermined round, read from `checkpoints/round_NNNN.pt` |
| `--index-path` | Override the dataset index |
| `--device` | Device for inference, for example `cuda` |
| `--robustness`, `--no-robustness` | Force the 15-combination pass on or off. By default the run's missing-modality configuration decides. |
| `--output-dir` | Write elsewhere |

Results go to `final_evaluation/`, or `final_evaluation_round_NNNN/` with `--final-round`, as `metadata.json`, `modality_combinations.jsonl`, and `metrics.json`. The metadata records the checkpoint hash and its role, `selected_best` or `predetermined_final`. A directory that already holds results is refused. Use `--output-dir` for a second report.

Centralised runs select the best epoch on the validation cohort during training and test that checkpoint at the end. The `test_*` fields of their `summary.json` hold the result. `fedbrats evaluate-checkpoint --checkpoint <path> --index-path <index.csv> --subjects <id1,id2>` scores any checkpoint on a chosen list of subjects on the CPU.

## Revalidating retained checkpoints

Runs from before the held-out evaluator existed chose `best.pt` by evaluating clients on their own training data. `fedbrats revalidate-federated-run` repeats the checkpoint search on the validation cohort without retraining and without changing the original files.

```bash
fedbrats revalidate-federated-run --run-dir outputs/<run> --split <split.json> \
  --index-path <index.csv> --dry-run
fedbrats revalidate-federated-run --run-dir outputs/<run> --split <split.json> \
  --index-path <index.csv> --device cuda
```

The dry run lists the checkpoints in numeric order, checks the split, and reports whether the index is available. It loads no images and writes nothing. The full run restores each checkpoint's model and evaluates it with the run's saved preprocessing and loss configuration. It scores Dice and loss only. `--expensive-metrics` adds HD95 and sensitivity to every checkpoint.

Outputs stay inside `revalidation/`, as `validation_rounds.jsonl`, `metadata.json`, `state.json`, `summary.json`, and `best_revalidated.pt`. An interrupted run resumes at the first unevaluated checkpoint after checking the hashes of the evaluated ones. The summary reports whether the best round changed. Report test results for the new choice with `--checkpoint revalidation/best_revalidated.pt`.

`fedbrats revalidate-federated-runs` processes several runs in sequence. Give `--run-dir` several times or a quoted `--run-glob`.

```bash
fedbrats revalidate-federated-runs --run-glob '<runs root>/*' \
  --split <split.json> --index-path <index.csv> --device cuda
```

On Slurm, list one absolute run directory per line in a file on a shared filesystem and submit an array with one task per run. `slurm/_shared/revalidate_federated_runs.sbatch` takes the list, the split, and an optional index. It runs inside `apptainer/fl-brats-batch.sif` unless `APPTAINER_IMAGE` names another image, or on the host with `USE_APPTAINER=false`. Review the dry run before submitting.

```bash
sbatch --array=0-5%1 --partition=batch --cpus-per-task=6 --mem=29000M --time=24:00:00 \
  slurm/_shared/revalidate_federated_runs.sbatch "$PWD/runs.txt" <split.json> <index.csv>
```

`--array=0-5` matches a list of six runs. `%1` runs one task at a time. Raise it to the number of free GPUs. Resubmit the same command after an interruption to resume.

The dated record of how the held-out evaluator and revalidation were implemented and verified is in [logs/monitoring/revalidation/implementation_record.md](../logs/monitoring/revalidation/implementation_record.md).
