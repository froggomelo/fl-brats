# SAM 1 and SAM 2 as federated segmentation models

The model-agnostic experiments (`configs/model_agnostic_uneven/sam{1,2}/`)
replace the 3D U-Net with a pretrained Segment Anything model under the same
federated protocol: four uneven clients, E = 1, B = 1, full modalities, the
same held-out validation selection and the same test metrics.

## Models

| | SAM 1 | SAM 2 |
|---|---|---|
| Package | `segment_anything`, commit `dca509f` | `sam2` (PyPI name `sam-2`), commit `2b90b9f` |
| Checkpoint | `sam_vit_b_01ec64.pth` (ViT-B) | `sam2_hiera_tiny.pt` (Hiera-Tiny) |
| SHA-256 | `ec2df627…c8912` | `65b50056…d1125` |
| Parameters, total / trainable | 91.4M / 4.07M | 31.4M / 4.22M |

Both packages come from Meta's GitHub repositories, pinned by commit archive
in `pyproject.toml`; the PyPI project named `sam2` is an unrelated fork. The
checkpoints live in `artifacts/pretrained/` (not in git) and are verified by
SHA-256 whenever a model is built.

## Adaptation

* **Input.** Each axial slice's four z-scored MRI channels go through a 1x1
  projection to three channels, initialised to the fixed FLAIR/T1ce/T2 mapping
  and then learned. Values are clipped to [-5, 5] and scaled to [0, 1], then
  prepared as each model expects: SAM 1 resizes the longest side and pads, with
  its pixel mean and standard deviation; SAM 2 resizes to a square with
  ImageNet statistics.
* **Resolution.** 512 instead of SAM's native 1024. Positional embeddings are
  interpolated (SAM 1's absolute embeddings bicubically; relative positions and
  Hiera's embeddings by the libraries themselves). Masks come out at 128 x 128
  and are resized back to the slice, finer than the ~240-pixel slices need.
* **Prompts.** No clicks or boxes: one learned prompt token per region (WT, TC,
  ET) is fed to the mask decoder, which returns one mask per token. Labels never
  reach the model at inference.
* **Trainable and federated.** The projection, the region tokens, the prompt
  encoder's small embeddings and the pretrained mask decoder. The image
  encoder is frozen and never transmitted, so each round sends ~16 MB, close to
  the U-Net's 19 MB.
* **Training.** B = 1 subject per step. Each step uses 8 whole axial slices of
  the subject, half drawn from slices containing tumour when it has any, the
  rest from slices with brain (`train_slices_per_subject`,
  `train_tumour_slice_fraction`). There is no 3D patch crop
  (`data.train_crop: false`); the left-right flip still applies.
* **Loss.** The same Dice-CE as the U-Net, computed per subject over its 8
  slices stacked into a small volume. Per-slice Dice rewards predicting nothing
  on every slice and region without tumour: in a check, SAM 2 reached 0.46
  validation Dice after 437 steps and then collapsed to empty predictions while
  its training loss kept falling.
* **Evaluation.** Every axial slice of a validation or test volume is
  predicted, the volume is reassembled, and the same Dice, HD95 and sensitivity
  code as for the U-Net scores it.
* **Checkpoints.** Only the trainable weights are stored; the frozen part is
  rebuilt from the verified pretrained file (`fl_brats.models.state`).
* **Initial weights.** Per seed, like the U-Net:
  `artifacts/initial_weights/sam{1,2}_seed_{42,43,44}.pt`, made with
  `fedbrats create-initial-weights` from the experiment config's model section.

## Cost on one RTX 3060 (per client, measured 2026-09-24)

| | Training step, 8 slices | Peak memory | Inference, 150-slice volume |
|---|---|---|---|
| SAM 1, learned projection | 0.49 s | 7.2 GB | 3.7 s |
| SAM 1, fixed mapping | 0.22 s | 1.6 GB | 3.7 s |
| SAM 2, learned projection | 0.16 s | 2.8 GB | 1.2 s |
| SAM 2, fixed mapping | 0.08 s | 0.9 GB | 1.2 s |

The learned projection sits before the frozen encoder, so its gradient passes
back through the encoder; that doubles the training step. A round is roughly
4.7 min of client training (client 0) plus 11.6 min of validation for SAM 1,
and 1.7 plus 3.7 min for SAM 2. SAM 1 runs longer than batch's 24 h limit and resumes from
its latest round.

## Learning check (2026-09-25)

Client 0's 437 training subjects, the production data pipeline and loss,
Adam at 1e-4, validation Dice on the first 24 validation subjects:

| | Start | Epoch 1 | Epoch 2 | Epoch 3 | Seconds per epoch |
|---|---|---|---|---|---|
| SAM 2 | 0.055 | 0.543 | 0.680 | 0.761 | ~100 |
| SAM 1 | 0.032 | 0.512 | 0.566 | | ~280 |

## Runtime notes

* The SAM image is `apptainer/fl-brats-batch.sam.sif`: the batch image plus the
  two SAM packages, with the same Python, torch, MONAI and Flower versions.
* Jobs set `TORCH_DISABLE_NATIVE_JIT=1`: torch 2.13 would otherwise send an op
  in the encoder backward to a Triton kernel that needs a C compiler.
* The single-job local federation cannot hold four SAM 1 clients on one 12 GB
  GPU at 8 slices each; real runs give every client its own GPU.
