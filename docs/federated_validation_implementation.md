# Implementation and verification record

Implemented the held-out evaluator, shared FedAvg/FedAMM checkpoint lifecycle,
resumable revalidation CLI, separate test CLI, batch/Slurm entry points, and timing.
See [usage and submission commands](federated_validation.md).

## Verification performed

- 58 focused tests passed in `apptainer/fl-brats-batch.sif` using the workspace
  source. Coverage includes split boundaries, absent default client validation
  loader, one global evaluation, exact checkpoint tensors, natural-only selection,
  patience, FedAMM, sparse and quantized transports with resumed round offsets,
  resumed checkpoint search, historical preservation, separate test reporting,
  corrected training resume, distributed Slurm bootstrap, and successful finalization.
- The full suite, before adding the final six regression cases, produced 104
  passes and four failures. The same four failures reproduced with the original
  HEAD versions of `data/brats.py` and `data/partitioning.py`: manifest verification,
  index relative paths, raw-root override, and client relative paths. Three are
  affected by the container test raw-root environment; the raw-root override test
  also expects a path without the fixture's `CollectionA` directory. These existing
  failures were not changed to make this patch pass.
- Ruff passed on the new modules/tests and core edited strategy/orchestration files;
  `git diff --check` and shell syntax verification passed.
- A real two-client, two-round Flower 1.32.1 CPU smoke experiment passed with tiny
  synthetic NIfTI volumes. Best/final authoritative validation also completed.
- Actual CLI revalidation of both smoke checkpoints succeeded. Actual final-test
  CLI evaluation used only `synthetic_5` and produced all 15 combinations with Dice,
  HD95 and sensitivity. No production revalidation or test evaluation was launched.

Smoke artifacts: `/tmp/flbrats-heldout-smoke-yctq9nuv/run`.
Logs from this session: `/tmp/flbrats-smoke2.log`,
`/tmp/flbrats-revalidation-smoke.log`, `/tmp/flbrats-final-smoke.log`,
`/tmp/flbrats-verified-tests.log`, `/tmp/flbrats-full-tests.log`, and
`/tmp/flbrats-baseline-path-tests.log`.

Representative smoke log:

```text
train subjects: synthetic_0, synthetic_1, synthetic_2, synthetic_3
validation subjects: synthetic_4
test subjects: synthetic_5
client_0: training only, 2 subjects; disjoint from validation/test
client_1: training only, 2 subjects; disjoint from validation/test
Global validation round=1 subjects=1.0 mean_dice=0.038489
Selected best.pt round=1 held-out mean_dice=0.038489
configure_evaluate: no clients selected, skipping evaluation
Global validation round=2 subjects=1.0 mean_dice=0.038489
configure_evaluate: no clients selected, skipping evaluation
SMOKE PASSED
```

## Historical dry-run limitation

Attempted a dry run against the locally available historical run
`outputs/federated/unet_baseline_20260806_134425`, with
`--split data/splits/brats2021.json`. It stopped because that split file does not
exist here. The actual frozen split and the six completed baseline run locations
were requested; a substitute split was not fabricated. This historical dry-run
verification remains outstanding. No historical artifacts were created or changed.
The dry-run implementation was separately verified with a valid synthetic split.

## Architecture decisions

Server-side validation is supported by Flower's existing centralized hook. A shared
strategy override handles selection immediately after that hook, so distributed
client results cannot participate. Round-zero evaluation is skipped. No dedicated
evaluation node was needed. Classic APIs remain compatible, though Flower emits
deprecation warnings. Local startup now places the server on the main thread
because this Flower version registers process signal handlers there.

Checkpoint files are not rewritten to insert later evaluation metrics; immutable
round tensors are linked by round/path to the validation JSONL. Resuming corrected
training recovers selection from those committed rows. Historical selection scores
are reset. Revalidation checks identity and checkpoint hashes before reusing rows.
An adjacent pre-existing transport issue was fixed: fit configuration now carries
the global round number so quantized updates validate correctly after resume.

## Exact files changed by this task

Modified (existing user changes in overlapping files were retained):

```text
README.md
configs/_experiment_base/federated_uneven.yaml
scripts/launch_experiment.py
src/fl_brats/cli.py
src/fl_brats/data/brats.py
src/fl_brats/data/partitioning.py
src/fl_brats/federated/fedamm.py
src/fl_brats/federated/model_client.py
src/fl_brats/federated/orchestration.py
src/fl_brats/federated/research.py
src/fl_brats/federated/server.py
src/fl_brats/federated/strategy.py
src/fl_brats/training/engine.py
tests/test_research_strategy.py
```

Created:

```text
docs/federated_validation.md
docs/federated_validation_implementation.md
scripts/smoke_heldout_federation.py
slurm/_shared/revalidate_federated_runs.sbatch
src/fl_brats/data/cohorts.py
src/fl_brats/federated/evaluation.py
src/fl_brats/federated/finalization.py
src/fl_brats/federated/revalidation.py
tests/test_heldout_federation.py
```

Other initially dirty files, including existing segmentation metrics, training
modality-evaluation changes, missingness config, and the client-evaluation Slurm
script, were preserved and are not new changes from this task.
