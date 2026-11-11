# Models

This guide describes every segmentation network the model factory builds, how a config selects one, and how to add a new one. Read it before changing the network of an experiment or comparing architectures.

## Selecting a model

`create_model` in `src/fl_brats/models/factory.py` builds a network from the `model` section of a config. The `name` key picks the network, and the other keys are passed to it.

| `model.name` | Network | Model config | Parameters |
|---|---|---|---|
| `unet` | MONAI 3D U-Net | `configs/model/unet_baseline.yaml` | 4,810,074 |
| `unet` with `concat_mask: true` | 3D U-Net with modality presence channels | `configs/model/unet_missing_modality_baseline.yaml` | 4,813,530 |
| `fedamm` | Modality-specific encoders with a shared decoder | `configs/model/fedamm.yaml` | 1,012,393 |
| `segresnet` | MONAI SegResNet with 16 initial filters | `configs/model/segresnet.yaml` | 4,702,227 |
| `segresnet` | MONAI SegResNet with 8 initial filters | `configs/model/segresnet_8.yaml` | 1,176,843 |
| `sam1` | Segment Anything ViT-B with an adapter | `configs/model/sam1.yaml` | 91.4 M total, 4.07 M trainable |
| `sam2` | Segment Anything 2 Hiera-Tiny with an adapter | `configs/model/sam2.yaml` | 31.4 M total, 4.22 M trainable |

Every network takes the four magnetic resonance imaging (MRI) channels in the order `[t1, t1ce, t2, flair]` and returns three logits per voxel for whole tumour (WT), tumour core (TC), and enhancing tumour (ET). The regions overlap. Training uses a sigmoid per channel and a loss that adds Dice, an overlap measure, to binary cross-entropy. It never applies a softmax. [Evaluation](evaluation.md) explains how the regions come from the labels of BraTS, the Brain Tumour Segmentation challenge dataset.

An experiment YAML sets the model under `model:`. The section merges over the one inherited through `extends`. A file therefore lists only what changes. Keys a network does not read are ignored. The SegResNet experiments inherit the U-Net's `channels` and `strides`, for example, and SegResNet ignores both.

```yaml
extends: ../../_experiment_base/federated_uneven.yaml
model:
  name: segresnet
  init_filters: 16
  blocks_down: [1, 2, 2, 4]
  blocks_up: [1, 1, 1]
  norm: [GROUP, {num_groups: 8}]
  dropout_prob: null
```

The composed Hydra configuration selects a file from `configs/model/` with an override such as `model=segresnet`. Centralised training reads a model YAML through `fedbrats train-centralized --model-config`. A centralised experiment YAML names that file with `model.config`, as in `configs/centralized/segresnet_converged.yaml`.

Each seed starts from a fixed initial-weights file. Methods that share a network share these files. Their runs then start from the same weights.

```bash
fedbrats create-initial-weights --model-config configs/model/segresnet.yaml \
  --output-dir artifacts/initial_weights/segresnet --seed 42
```

The command writes `<name>_seed_<seed>.pt` and a JSON file with the seed, the SHA-256 hash of the weights, and a parameter report. An experiment YAML points at the files with `initial_weights` and `initial_weights_by_seed`.

## 3D U-Net baseline

The baseline is MONAI's `UNet` with five levels of 16, 32, 64, 128, and 256 channels, stride-2 downsampling, two residual units per level, instance normalisation, and PReLU activations. `configs/model/unet_baseline.yaml` holds every key.

| Key | Default |
|---|---|
| `channels` | `[16, 32, 64, 128, 256]` |
| `strides` | `[2, 2, 2, 2]` |
| `kernel_size`, `up_kernel_size` | `3`, `3` |
| `num_res_units` | `2` |
| `dropout` | `0.0` |
| `normalisation` | `INSTANCE` |
| `activation` | `PRELU` |
| `concat_mask` | `false` |

The network sits inside `MaskConcatUNet` in `src/fl_brats/models/unet.py`. The wrapper accepts the per-sample modality mask from the data loader. With `concat_mask: false` it ignores the mask. Missing scans then reach the network as zero channels.

## Missing-modality U-Net

`configs/model/unet_missing_modality_baseline.yaml` is the U-Net with `concat_mask: true`. The wrapper expands the four-value mask into four constant channels and appends them to the image, and the first convolution takes eight channels. No experiment family uses this variant. The zero-filled U-Net of the missing-modality experiments keeps `concat_mask: false`. [Missing modalities](missing_modalities.md) describes it.

## FedAMM

FedAMM is a federated method for arbitrary missing modalities. Its network gives each modality its own 3D encoder, fuses the features of the available modalities by a masked mean, and decodes with one shared decoder. The model returns the logits together with the fused and per-modality features that the prototype losses use. `configs/model/fedamm.yaml` lists its keys. [Missing modalities](missing_modalities.md) covers the network, the losses, and the server aggregation in full.

## SegResNet

SegResNet is the residual encoder-decoder that Myronenko published in 2018 and that won BraTS 2018. The `model_agnostic_uneven` family uses it to test whether the federated results depend on the U-Net. It is MONAI's `SegResNet` with four encoder stages of 1, 2, 2, and 4 residual blocks, three decoder stages of one block each, and group normalisation with 8 groups.

Two sizes exist. `configs/model/segresnet.yaml` uses 16 initial filters and roughly matches the U-Net's size. `configs/model/segresnet_8.yaml` uses 8 initial filters, a quarter of the parameters, and a quarter of the traffic per round.

| Experiment | Config |
|---|---|
| Federated, 16 filters | `configs/model_agnostic_uneven/segresnet/baseline.yaml` |
| Federated, 8 filters | `configs/model_agnostic_uneven/segresnet_8/baseline.yaml` |
| Centralised, 16 filters | `configs/centralized/segresnet_converged.yaml` |
| Centralised, 8 filters | `configs/centralized/segresnet_8_converged.yaml` |

`create_segresnet` in `src/fl_brats/models/segresnet.py` places the network in the same `MaskConcatUNet` wrapper as the U-Net. Training, Federated Averaging (FedAvg), checkpoints, and evaluation treat it exactly like the U-Net. Only the network changes. The initial weights live in `artifacts/initial_weights/segresnet/` and `artifacts/initial_weights/segresnet_8/`.

## SAM 1 and SAM 2

The Segment Anything Model (SAM) and its successor SAM 2 are pretrained 2D image segmentation models from Meta. The `model_agnostic_uneven/sam1` and `model_agnostic_uneven/sam2` experiments fine-tune them under the same federated protocol as the U-Net, with four uneven clients, one local epoch, one subject per step, full modalities, and the same validation selection and test metrics. The code is `SAMAdapter` in `src/fl_brats/models/sam_adapter.py`, with helpers in `src/fl_brats/sam/`.

| | SAM 1 | SAM 2 |
|---|---|---|
| Package | `segment_anything`, commit `dca509f` | `sam2`, installed as `sam-2`, commit `2b90b9f` |
| `model.variant` | `vit_b` | `sam2_hiera_tiny` |
| Checkpoint | `artifacts/pretrained/sam_vit_b_01ec64.pth` | `artifacts/pretrained/sam2_hiera_tiny.pt` |
| Transmitted per round | 16.3 MB | 16.9 MB |

`pyproject.toml` pins both packages to commit archives of Meta's GitHub repositories. The PyPI project named `sam2` is an unrelated fork. The SAM experiments run in `apptainer/fl-brats-batch.sam.sif`, the batch image plus the two packages, which their configs select with `slurm.image`.

### Pretrained checkpoints

The checkpoints are not in git. Download them into `artifacts/pretrained/` from the official repositories. Every time a SAM model is built, `verify_pretrained` in `src/fl_brats/sam/validation.py` hashes the file and stops if the SHA-256 differs from `model.checkpoint_sha256`. Nothing is downloaded automatically.

```yaml
model:
  name: sam1
  variant: vit_b
  checkpoint: artifacts/pretrained/sam_vit_b_01ec64.pth
  checkpoint_sha256: ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912
```

The SAM 2 hash is `65b50056e05bcb13694174f51bb6da89c894b57b75ccdf0ba6352c597c5d1125`. Check a checkpoint before submitting jobs with `fedbrats verify-pretrained-model`. That command reads the key `checkpoint_path`, not `checkpoint`. The files in `configs/model/sam*.yaml` leave `checkpoint_path` and `checkpoint_sha256` null. Copy one, fill in both, and pass the copy.

```bash
fedbrats verify-pretrained-model --model-config <sam1_with_checkpoint.yaml>
```

`fedbrats create-initial-weights` also builds the full model and needs the same filled-in file. It writes only the trainable weights to `artifacts/initial_weights/sam1_seed_<seed>.pt` or `sam2_seed_<seed>.pt`.

### Adaptation

The adapter turns the 3D, four-channel BraTS problem into the 2D, three-channel task SAM expects.

| Stage | What happens |
|---|---|
| Input projection | A learned 1×1 convolution maps the four MRI channels of a slice to three. It starts as the fixed FLAIR, T1ce, T2 mapping of `rgb_mapping`. `multimodal_adapter: fixed_rgb` uses that mapping without learning and drops T1. |
| Intensity | The z-scored values are clipped to `clip` and scaled to [0, 1]. SAM 1 then resizes the longest side and pads with its own pixel statistics. SAM 2 resizes to a square with ImageNet statistics. |
| Resolution | `input_size` is 512 instead of the native 1024. SAM 1's absolute positional embeddings are interpolated bicubically. The libraries interpolate the relative positions and Hiera's embeddings themselves. Masks come back at low resolution and are resized to the slice. |
| Prompts | No clicks or boxes. One learned prompt token per region goes to the mask decoder, which returns one mask per token. Labels never reach the model. External prompts raise an error. |
| Trainable scope | The projection, the region tokens, the prompt encoder's small embeddings, and the pretrained mask decoder. The image encoder is frozen and is never transmitted. |

`fine_tuning_mode` sets what trains. `adapter_decoder`, the default, trains the components listed under trainable scope. `prompt_decoder` currently selects the same parameters. `decoder_only` trains only parameters whose names contain `decoder`. `full_model` would train the encoder and is rejected. `federated_scope: trainable_only` sends only the trainable weights. `federated_scope: full_model` sends the whole state.

### Training and evaluation on slices

SAM sees whole axial slices, not 3D patches. The SAM configs set `data.train_crop: false`, which skips the 3D crop and keeps the left-right flip. Each step takes one subject and samples `train_slices_per_subject` axial slices from it, default 8. A share of `train_tumour_slice_fraction`, default 0.5, comes from slices containing tumour when the subject has any. The rest come from slices containing brain.

The loss is the same Dice plus cross-entropy as the U-Net, computed per subject over its sampled slices stacked into a small volume. Per-slice Dice would reward empty predictions on every slice without tumour. Training then collapses to predicting nothing.

At validation and test every axial slice of the volume is predicted in chunks of `slice_batch_size`. The slices are reassembled into a volume, and the same code as for the U-Net scores it with Dice, the 95th-percentile Hausdorff distance (HD95), and sensitivity.

Checkpoints store only the trainable weights. Loading one rebuilds the frozen encoder from the verified pretrained file, as implemented in `src/fl_brats/models/state.py`.

### SAM configuration keys

| Key | Default | Meaning |
|---|---|---|
| `variant` | `vit_b` or `sam2_hiera_tiny` | Pretrained architecture. SAM 2 supports only `sam2_hiera_tiny`. |
| `checkpoint`, `checkpoint_path` | none | Path to the pretrained file. The model reads either. |
| `checkpoint_sha256` | none | Expected hash. The comparison is skipped when it is unset. |
| `input_size` | `512` | Side length of the encoder input |
| `multimodal_adapter` | `learned_4_to_3` | Or `fixed_rgb` |
| `rgb_mapping` | `[flair, t1ce, t2]` | Initial or fixed channel mapping |
| `clip` | `[-5.0, 5.0]` | Intensity clip before scaling |
| `fine_tuning_mode` | `adapter_decoder` | Trainable scope |
| `federated_scope` | `trainable_only` | Transmitted state |
| `train_slices_per_subject` | `8` | Slices per training step |
| `train_tumour_slice_fraction` | `0.5` | Share of tumour slices |
| `slice_batch_size` | `16` | Slices per forward pass at inference |
| `video_memory_propagation` | `false` | SAM 2 only. `true` is rejected. |
| `backend` | `pretrained` | `tiny` builds a small random encoder for tests without a checkpoint |

`configs/sam/` and the `prompt_mode`, `oracle_prompt`, `spatial_mode`, `plane`, `source_url`, `license`, and `automatic_download` keys describe the protocol but do not change the model. `fedbrats train-federated-sam1`, `train-federated-sam2`, `evaluate-federated-sam1`, and `evaluate-federated-sam2` only check that the config names the right model. Train SAM like any other experiment.

```bash
python scripts/launch_experiment.py configs/model_agnostic_uneven/sam2/baseline.yaml
```

The cluster wrapper `slurm/_shared/run_with_configured_environment.sh` sets `TORCH_DISABLE_NATIVE_JIT=1`. Without it torch 2.13 sends an operation in the encoder backward pass to a Triton kernel that needs a C compiler the image lacks.

### Measured cost

These figures were measured on 2026-09-24 on one NVIDIA RTX 3060 with 12 GB, per client.

| | Training step, 8 slices | Peak memory | Inference, 150-slice volume |
|---|---|---|---|
| SAM 1, learned projection | 0.49 s | 7.2 GB | 3.7 s |
| SAM 1, fixed mapping | 0.22 s | 1.6 GB | 3.7 s |
| SAM 2, learned projection | 0.16 s | 2.8 GB | 1.2 s |
| SAM 2, fixed mapping | 0.08 s | 0.9 GB | 1.2 s |

The learned projection sits before the frozen encoder. Its gradient passes back through the encoder, which doubles the training step. Four SAM 1 clients do not fit on one 12 GB GPU at 8 slices each. Real runs give every client its own GPU.

## Adding a model

A new network plugs into the factory and then works with every federated strategy, compression method, and evaluation command.

1. Write `src/fl_brats/models/<name>.py` with a function `create_<name>(config)` that returns a `torch.nn.Module`. The module takes a `[B, 4, H, W, D]` image and returns `[B, 3, H, W, D]` logits for WT, TC, and ET.
2. Add a branch for the new name in `create_model` in `src/fl_brats/models/factory.py` and extend its error message.
3. Write `configs/model/<name>.yaml` with `name` and every key the network reads.
4. Create the per-seed initial weights with `fedbrats create-initial-weights --model-config configs/model/<name>.yaml`.
5. Write an experiment YAML that extends a base, sets `model.name` and its keys, and points `initial_weights_by_seed` at the new files.

The shortest path for a plain 3D network is the SegResNet pattern. Wrap it in `MaskConcatUNet` and it inherits mask handling and the optional presence channels. A network with different needs can declare them through optional attributes and methods.

| Attribute or method | Effect |
|---|---|
| `supports_modality_mask = True` | The engine calls `model(image, modality_mask)` |
| `federated_state_dict()`, `load_federated_state_dict()` | Restrict the state sent each round, as SAM does |
| `checkpoint_state_dict()`, `load_checkpoint_state_dict()` | Restrict what checkpoints store |
| `model_kind = "sam"` | The engine samples 2D slices for training and predicts slice by slice at evaluation |

Only floating-point tensors take part in uplink quantisation and selective transmission. The quantised broadcast requires every transmitted tensor to be floating point. [Communication efficiency](communication_efficiency.md) describes all three.
