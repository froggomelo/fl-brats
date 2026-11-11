# Missing MRI modalities

This guide explains how the repository simulates clients that lack some magnetic resonance imaging (MRI) scans, the three methods compared under that condition, and the configuration that selects each method and partition. Read it before running the `fedavg_missing_modalities_uneven` or `fedamm_uneven` families, or any communication-efficient family built on them.

## The problem

Every subject in BraTS, the Brain Tumour Segmentation challenge dataset, has four MRI modalities, namely T1, contrast-enhanced T1 (T1ce), T2, and FLAIR. Hospitals do not always acquire all four. A federated client may therefore hold subjects with only some of the scans, and different clients may hold different combinations. Four modalities give 15 non-empty combinations.

Missingness is explicit metadata. It is never inferred from image intensities. A missingness policy assigns each training subject its available modalities when the client partition is created, and the client manifest records the result. Each subject entry in `client_N.json` stores the following fields.

| Field | Meaning |
|---|---|
| `available_modalities`, `missing_modalities` | Modality names in canonical order |
| `modality_mask` | Boolean list in the order `[t1, t1ce, t2, flair]` |
| `modality_combination_id` | Integer from 1 to 15, with bit weights t1 = 1, t1ce = 2, t2 = 4, flair = 8 |
| `paths` | File paths, with `null` for every missing scan |
| `missingness_realisation_id` | Fingerprint of the policy configuration |

A client cannot read a missing scan. Its path is `null` in the manifest. The data loader rebuilds the four-channel image with zeros in the missing channels and passes the mask to the model as `modality_mask`. The code lives in `src/fl_brats/data/modalities.py`, `src/fl_brats/data/partitioning.py`, and `src/fl_brats/data/brats.py`.

## Methods

Three methods are compared on the same partitions, policies, and held-out evaluation.

| Method | Config files | `model.name` | `federated.strategy` |
|---|---|---|---|
| Zero-filled U-Net | `configs/fedavg_missing_modalities_uneven/unet_zero_fill_*.yaml` | `unet` | `fedavg` |
| FedAvg-MM | `configs/fedavg_missing_modalities_uneven/` files without the `unet_zero_fill_` prefix | `fedamm` | `fedavg` |
| FedAMM | `configs/fedamm_uneven/` | `fedamm` | `fedamm` |

### Zero-filled U-Net

The zero-filled U-Net is the full-modality 3D U-Net of `configs/_experiment_base/federated_uneven.yaml`, trained with Federated Averaging (FedAvg), which averages client models weighted by their training sample counts. It starts from the same `artifacts/initial_weights/unet_seed_*.pt` files as the full-modality baseline. Absent scans arrive as zero channels during training. Validation and test multiply the image by the natural or counterfactual mask. The configs set `concat_mask: false`. The network sees only the four image channels.

`configs/model/unet_missing_modality_baseline.yaml` is the same U-Net with `concat_mask: true`. That option appends four constant presence channels, one per modality, and raises the input to eight channels. No experiment family uses it. Select it in the composed Hydra configuration with `model=unet_missing_modality_baseline`.

### FedAvg-MM

FedAvg-MM is the FedAMM network trained with plain FedAvg and every FedAMM training component switched off. It isolates the architecture from the prototype losses and the modality-weighted aggregation. `configs/_experiment_base/missing_uneven.yaml` defines it.

```yaml
model:
  name: fedamm
  channels: [8, 16, 32, 64]
  prototype_loss_weight: 0.0
  prototype_distillation_enabled: false
  global_prototype_alignment_enabled: false
  emit_prototypes: false
  modality_weighted_aggregation: false
federated: {strategy: fedavg}
```

FedAvg-MM and FedAMM share the initial weights `artifacts/initial_weights/fedamm_seed_*.pt`. A FedAvg-MM run and a FedAMM run with the same seed and partition differ only in the training components.

### FedAMM

FedAMM is a federated method for arbitrary missing modalities that combines a modality-specific network, prototype losses, and modality-weighted aggregation. The model is `FedAMMUNet` in `src/fl_brats/models/fedamm.py`. The server strategy is `FedAMMStrategy` in `src/fl_brats/federated/fedamm.py`.

**Network.** Each modality has its own 3D encoder that reads one channel. An encoder is a stack of double convolution blocks with instance normalisation and PReLU, separated by max pooling, with widths set by `model.channels`. The encoder of a modality that no sample in the batch has is skipped. At every level the features of the available modalities are fused by a masked mean over the modalities each sample has. A shared decoder upsamples the fused bottleneck with the fused skip features and outputs three region logits.

**Loss.** The training loss is the segmentation loss plus a prototype term.

```text
loss = DiceCE(logits, target) + prototype_loss_weight * mean(distillation terms, alignment term)
```

A semantic prototype is the mean feature vector of the voxels inside one tumour region. Prototypes are pooled from the bottleneck features for each of the three regions, with the label resized to the bottleneck by nearest-neighbour interpolation. The prototype term has two parts.

| Term | What it does |
|---|---|
| Prototype distillation | For each modality a sample has, the mean squared error between that modality's prototype and the fused prototype. The fused prototype is the teacher and receives no gradient. |
| Global prototype alignment | For each sample, the mean squared distance from its fused prototype to the nearest server centre for its modality combination, averaged over the batch. It is absent until the client has received a prototype library. Round 1 therefore has no alignment term. |

**Prototype exchange.** After local training each client averages its fused prototypes per modality combination over the last local epoch. It returns them in the fit metrics as `fedamm_prototypes_json`. The server groups the prototypes by combination and runs k-means with `fedamm.clusters_per_combination` centres. The default of one centre is the mean. The server sends the library to every client in the next round's fit configuration as `fedamm_global_prototypes_json`. Round checkpoints store the library, and a resumed run restores it.

**Aggregation.** Each client reports how many of its training subjects have each modality, counted from its manifest masks, as `fedamm_modality_count_<modality>`. The server averages every tensor under `modality_encoders.<modality>.` with those counts as weights. A client without a modality therefore contributes nothing to that encoder. The decoder and every other tensor are averaged by training sample count, as in FedAvg. With `modality_weighted_aggregation: false` every tensor is averaged by sample count.

**Aggregation under compression.** When the uplink is sparse or quantised, the strategy updates the prototype library and then passes aggregation to the shared research strategy. That strategy decodes each client update and applies the sample-weighted mean to every tensor. Modality-weighted aggregation is not applied in this case. [Communication efficiency](communication_efficiency.md) describes the compressed uplinks.

**Payload accounting.** Prototypes and the library travel in Flower's metrics and configuration dictionaries. They are not part of `upload_payload_bytes` or `download_payload_bytes`, which count only the model arrays.

### FedAMM configuration keys

| Key | Default | Effect |
|---|---|---|
| `federated.strategy` | `fedavg` | `fedamm` selects `FedAMMStrategy` |
| `model.channels` | `[8, 16, 32, 64]` | Encoder and decoder widths, one per level |
| `model.prototype_loss_weight` | `1.0` | Weight of the prototype term. Zero or less disables it. |
| `model.prototype_distillation_enabled` | `true` | Adds the distillation terms |
| `model.global_prototype_alignment_enabled` | `true` | Adds the alignment term once a library exists |
| `model.emit_prototypes` | `true` | Clients return prototypes to the server |
| `model.modality_weighted_aggregation` | `true` | Weights each encoder by modality counts |
| `fedamm.clusters_per_combination` | `1` | k-means centres per combination in the server library |
| `model.prototype_layer`, `model.prototype_temperature` | `-1`, `1.0` | Read by the model but have no effect in the current code |

`configs/fedamm_uneven/alpha_0_1.yaml` shows a complete FedAMM experiment.

```yaml
extends: ../_experiment_base/missing_uneven.yaml
experiment: {family: fedamm_uneven, variant: alpha_0_1, runs: 3, seeds: [42, 43, 44]}
federated: {strategy: fedamm}
model:
  prototype_loss_weight: 1.0
  prototype_distillation_enabled: true
  global_prototype_alignment_enabled: true
  emit_prototypes: true
  modality_weighted_aggregation: true
fedamm: {clusters_per_combination: 1}
```

## Partition variants

The `missing_modalities` section selects the policy. Its defaults are in `configs/missing_modalities/default.yaml`.

| `missing_modalities.mode` | Assignment | Config suffix | Partition root |
|---|---|---|---|
| `arbitrary` | Each modality dropped independently per subject | `alpha_0_001`, `alpha_0_1`, `alpha_0_2` | `data/partitions/missing_uneven/alpha_*` |
| `dirichlet_combination` | Per-client Dirichlet over the 15 combinations | `dirichlet_alpha_0_001`, `dirichlet_alpha_0_1`, `dirichlet_alpha_1` | `data/partitions/missing_dirichlet/alpha_*` |
| `dirichlet_combination`, 5 or 8 clients | As above with Lorenz client sizes | `dirichlet_alpha_0_1_k5`, `dirichlet_alpha_0_1_k8` | `data/partitions/missing_dirichlet_k{5,8}/alpha_0_1` |
| `complete` | One fixed scan set per client | `per_client_flair_largest` | `data/partitions/missing_per_client/flair_largest` |

Every variant uses the subjects and client sizes of the uneven split. The four clients hold 437, 218, 131, and 89 training subjects. Not every family has every suffix. List the folder to see which exist. Files named `*smoke*`, `*batch_10_rounds*`, or `*diagnostic*` are short checks, not reported runs.

Set `missing_modalities.enabled: true` for any mode other than `none`. `seed` fixes the assignment and defaults to 42. `require_at_least_one: true` guarantees that no subject loses all four scans.

### Two meanings of alpha

The `alpha_*` and `dirichlet_alpha_*` files both carry an alpha, but the two numbers measure different things.

In the `arbitrary` mode alpha is a dropout probability. Every modality of every subject is dropped independently with probability alpha, the same on every client. At alpha 0.1 about 66 % of subjects keep all four modalities and about 95 % keep at least three. Every client has the same expected mix.

In the `dirichlet_combination` mode alpha is the concentration of a Dirichlet distribution. Small alpha makes each client dominated by one or two combinations. Availability then differs strongly between clients, which is the setting that FedAMM's prototype alignment and modality-weighted aggregation target. The FedAMM paper uses alpha in {0.001, 0.1, 1}, and small alpha is the hardest setting.

### Independent per-modality dropout

`configs/_experiment_base/missing_uneven.yaml` sets the policy. The `alpha_0_001` and `alpha_0_2` files override the probability and the partition root.

```yaml
missing_modalities:
  enabled: true
  mode: arbitrary
  seed: 42
  require_at_least_one: true
  arbitrary:
    per_modality_missing_probability: {t1: 0.1, t1ce: 0.1, t2: 0.1, flair: 0.1}
    combination_probabilities: {}
```

Each subject's draw comes from a random stream seeded by SHA-256 of the policy version, seed, split, client, and subject. The same configuration always gives the same masks. `combination_probabilities` replaces the independent draws with explicit probabilities per combination, keyed as `t1+t2+flair` and summing to 1.

### FedAMM-style Dirichlet over combinations

`configs/_experiment_base/missing_dirichlet/alpha_{0_001,0_1,1}.yaml` are overlays. Layer one onto a method config with `extends: [<method config>, <overlay>]`.

```yaml
data: {partition_root: data/partitions/missing_dirichlet/alpha_0_1}
missing_modalities:
  mode: dirichlet_combination
  dirichlet_combination: {alpha: 0.1}
  arbitrary:
    per_modality_missing_probability: {t1: 0.0, t1ce: 0.0, t2: 0.0, flair: 0.0}
```

The assignment ports `generate_mask_distribution_only_maskid` from `utils/preprocessing/generate_dir_imb_mr.py` in the FedAMM authors' code at [13sky/FedAMM](https://github.com/13sky/FedAMM). Each client draws its own proportions from Dirichlet(alpha × ones(15)) over the 15 combinations. It receives one subject of every combination and draws the rest from its proportions. The list is shuffled and dealt to the client's subjects in sorted order. One draw seeded with `missing_modalities.seed` produces every client's proportions.

The clients here are larger than the authors' 55-subject clients. The one-of-each guarantee is therefore a smaller share, and the Dirichlet draw dominates. With seed 42, pooled over the training clients, all four modalities cover 13.7 %, 6.7 %, and 1.1 % of subjects at alpha 0.001, 0.1, and 1. A single modality covers 2.2 %, 25.6 %, and 31.5 %. At alpha 0.001 each client is 85 % to 97 % one or two combinations.

The assignment depends on every client's subject list. Client start-up therefore re-derives the whole partition from the seed and compares it with the manifest. A mismatch stops the run with an instruction to recreate the partition.

### Per-client missing modality

`configs/_experiment_base/missing_per_client/flair_largest.yaml` uses the `complete` mode. Every subject of a client lacks the same scan, and each client lacks a different one.

```yaml
missing_modalities:
  mode: complete
  complete:
    client_available_modalities:
      client_0: [t1, t1ce, t2]      # no FLAIR (437 subjects)
      client_1: [t1, t1ce, flair]   # no T2 (218)
      client_2: [t1, t2, flair]     # no T1ce (131)
      client_3: [t1ce, t2, flair]   # no T1 (89)
```

Every scan exists somewhere in the federation, but no client sees all four. The largest client lacks FLAIR, the scan that matters most for the whole-tumour Dice score, which measures the overlap between predicted and true tumour. A client missing from `client_available_modalities` receives a random set of between `min_available_modalities` and `max_available_modalities` scans.

## Held-out masks

Validation and test subjects belong to no client. Each held-out subject receives one natural mask from the evaluation policy, seeded with `evaluation.modality_seed` (default 42) and the client ID `global_evaluation`. The run seed does not affect it.

| Mode | Natural held-out mask |
|---|---|
| `arbitrary` | Drawn per subject from the same policy |
| `dirichlet_combination` | Drawn per subject from the pooled training distribution of `data.partition_root` |
| `complete` | Set by `missing_modalities.evaluation.natural_missingness` |

The `complete` mode alone would give every held-out subject one random combination. The `flair_largest` overlay instead sets `natural_missingness` to an `arbitrary` policy with explicit `combination_probabilities`. Each held-out subject misses one scan, drawn with the pooled training shares 437/875, 218/875, 131/875, and 89/875.

Missing-modality runs also evaluate all 15 combinations as counterfactual masks on the full held-out images. [Evaluation](evaluation.md) covers when that pass runs.

## Building and inspecting partitions

Build the partition that an experiment config names, then check its modality mix.

```bash
fedbrats prepare-experiment-partition \
  --config configs/fedamm_uneven/dirichlet_alpha_0_1.yaml \
  --split-path <split_manifest.json>
fedbrats inspect-modalities --partition-root data/partitions/missing_dirichlet/alpha_0_1
```

`--split-path` defaults to `data.split_path` from the config. A non-empty destination is refused unless `--overwrite` is given. `inspect-modalities` prints, per client, the subject count, the count and share of each combination, the all-four and single-modality shares, the presence of each modality, and the policy fingerprint.

Apptainer images bake in `src/` at build time. An image built before `dirichlet_combination` and `partition_remainder` existed ignores both and silently writes a full-modality partition. Run the checked-out source on the cluster with `FL_BRATS_BIND_SOURCE=1`.

```bash
FL_BRATS_BIND_SOURCE=1 slurm/_shared/run_with_configured_environment.sh true \
  apptainer/fl-brats-batch.sif \
  fedbrats prepare-experiment-partition --config <config> --split-path <split_manifest.json>
```

[Data](data.md) covers the split and the full-modality partitions.

## Training-loss components

Every client fit reports the parts of `train_loss`, and `metrics/clients.jsonl` stores them beside it. Each is a sample-weighted mean over the last local epoch. Logging copies values that are already computed. Gradients and training are unchanged.

| Field | Meaning |
|---|---|
| `seg_loss` | The segmentation loss alone. It equals `train_loss` for every method without a prototype term or the FedProx proximal penalty. |
| `distill_loss` | Batch mean of the per-modality distillation terms, before the prototype weight |
| `distill_loss_t1`, `distill_loss_t1ce`, `distill_loss_t2`, `distill_loss_flair` | Each modality's term, averaged over the samples that have it |
| `align_loss` | The global alignment term before the weight. Null until a library arrives. |
| `aux_loss_weighted` | What was added to the loss, `prototype_loss_weight` times the mean of the terms |
| `aux_fraction` | `aux_loss_weighted / train_loss` |

The FedAMM-only fields are null for the other methods. `metrics/rounds.jsonl` stores the unweighted mean of each across the round's clients as `mean_<name>`, for example `mean_seg_loss` and `mean_aux_fraction`.
