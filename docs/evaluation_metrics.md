# Evaluation metrics

Every model family uses the shared subject-macro segmentation evaluator after its
architecture-specific prediction or reconstruction step. It reports Dice, HD95,
and sensitivity separately for whole tumour (WT), tumour core (TC), and enhancing
tumour (ET), plus the arithmetic mean of each metric.

HD95 is reported in voxel units. Empty truth and prediction masks receive Dice 1,
sensitivity 1, and HD95 0. If only one mask is empty, Dice and sensitivity are 0
and HD95 is the image-grid diagonal. This keeps persisted metrics finite while
making the empty-mask policy reproducible.

Federated runs additionally persist per-client scores and population variance,
rounds reaching 90% and 95% of the final validation Dice, application payload
bytes per round and cumulatively, and Dice-versus-cumulative-communication points.
Sweeps write a cross-run communication Pareto frontier. Missing-modality runs
evaluate the natural assigned distribution and all 15 non-empty modality
combinations, including the all-four-modality condition.

Flower does not expose network-only transmission duration or transport-header
bytes. Consequently `network_transmission_seconds` is explicitly null and the
summary marks it unavailable; round wall time and application payload volume are
kept as separate measurements and are not described as network transmission time.
