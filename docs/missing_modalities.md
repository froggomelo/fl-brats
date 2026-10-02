# Missing MRI modalities

Missingness is explicit metadata, never inferred from image intensities. `complete` assigns one deterministic availability set to every subject of a client; `arbitrary` assigns each subject independently from a SHA-256-derived random stream. Assignments are written into client manifests with available/missing lists, a canonical `[t1, t1ce, t2, flair]` boolean mask, and a policy fingerprint.

The legacy masked U-Net baseline retains four MRI image channels, zero-fills unavailable channels, and can concatenate four spatial presence channels (`model=unet_missing_modality_baseline`). It is a modality-aware baseline, not a reproduction of FedAMM.

For the direct FedAMM control, use `configs/experiment/federated_missing_modality_fedavg.yaml`. It uses the FedAMM-compatible modality-specific encoders and shared decoder with the same explicit per-subject masks, but runs `strategy: fedavg`, disables prototype distillation/global prototype alignment/prototype emission, and disables modality-weighted encoder aggregation. Compare that run directly against `configs/experiment/federated_fedamm.yaml` using the same saved partition manifests and the same `initial_weights` checkpoint.

Example federation: client 0 can retain all modalities, client 1 `[t1, t2, flair]`, client 2 `[t2, flair]`, and client 3 can use arbitrary subject-level masks. Build it with `python -m fl_brats create-partition --split-path ... --output-dir ... missing_modalities.enabled=true missing_modalities.mode=complete`; inspect it with `python -m fl_brats inspect-modalities --partition-root ...`.

Use `missing_modalities.mode=arbitrary` and overrides such as `missing_modalities.arbitrary.per_modality_missing_probability.t1ce=0.4` for subject-level missingness. `evaluate_modality_combinations` applies all fifteen non-empty counterfactual masks without changing the stored manifest.

## Two meanings of alpha

The campaign uses two partition schemes, and their alphas are different quantities.

**Independent per-modality dropout** (`mode: arbitrary`, `data/partitions/missing_uneven/alpha_*`, `configs/_experiment_base/missing_uneven.yaml`). Every modality is dropped independently with probability alpha, the same on every client (at least one modality is kept). At alpha 0.1 about 66 % of subjects have all four modalities and 95 % have at least three, and every client has the same expected mix. Kept as the control.

**FedAMM-style Dirichlet over combinations** (`mode: dirichlet_combination`, `data/partitions/missing_dirichlet/alpha_{0_001,0_1,1}`, `configs/_experiment_base/missing_dirichlet/`). A port of `generate_mask_distribution_only_maskid` from the FedAMM authors' code ([13sky/FedAMM](https://github.com/13sky/FedAMM), `utils/preprocessing/generate_dir_imb_mr.py`). Each client draws its own proportions from Dirichlet(alpha * ones(15)) over the 15 non-empty combinations, holds one subject of every combination, and draws the rest from its proportions; the list is shuffled and dealt to the client's subjects in sorted order. All clients' proportions come from one draw seeded with `missing_modalities.seed` (42). Here alpha is the Dirichlet concentration: small alpha makes each client dominated by one or two combinations, so availability differs strongly between clients, which is what FedAMM's prototype alignment and modality-weighted aggregation correct. The paper uses alpha in {0.001, 0.1, 1}; small alpha is the hardest setting.

The client sizes stay those of the uneven split (437, 218, 131 and 89 subjects, the same subjects as `missing_uneven`). With the authors' 55-subject clients the 15 guaranteed subjects are 27 % of each client, which gives about 7 % full-modality and 26 % single-modality samples at every alpha. Our clients are larger, so the Dirichlet draw dominates: pooled over the training clients, all four modalities 13.7 %, 6.7 % and 1.1 % and a single modality 2.2 %, 25.6 % and 31.5 % at alpha 0.001, 0.1 and 1 (seed 42). At alpha 0.001 each client is 85-97 % one or two combinations, and most combinations appear in a client only through the one-of-each guarantee.

Training masks are written into the client manifests like the other modes. Because the assignment depends on every client's subject list, the training-time check (`validate_client_missingness`) re-derives the whole partition from the seed and compares. Validation and test subjects draw their natural combination per subject, seeded, from the pooled training distribution of the partition (`data.partition_root`); the 15-combination counterfactual evaluation is unchanged. Build a partition with `fedbrats prepare-experiment-partition --config configs/fedamm_uneven/dirichlet_alpha_0_1.yaml --split-path <split_manifest.json>` and inspect it with `fedbrats inspect-modalities --partition-root ...`, which reports per-client combination shares, the full- and single-modality shares and per-modality presence.

Modality-weighted aggregation needs no change: each client reports, every round, how many of its training subjects have each modality, counted from its manifest masks (`fedamm_modality_count_*`), and the server weights each modality encoder by those counts.

## Training-loss components

Every client fit reports the parts of `train_loss` (`fl_brats.training.engine.LossComponents`), and `metrics/clients.jsonl` stores them next to `train_loss`. Like `train_loss`, they are sample-weighted means over the last local epoch. Logging only copies values that are already computed, so gradients and training are unchanged; a 3-round FedAMM smoke run, seed 42, gives the same `train_loss` and validation Dice before and after the change.

- `seg_loss`: the segmentation loss alone (Dice + BCE). It is the only component for FedAvg-MM, the zero-fill U-Net and the other U-Net and SAM runs, where it equals `train_loss` (without FedProx).
- `distill_loss`: FedAMM's distillation term before the prototype weight, the batch mean of the per-modality MSE terms between unimodal and fused prototypes.
- `distill_loss_t1`, `distill_loss_t1ce`, `distill_loss_t2`, `distill_loss_flair`: each modality's term, averaged over the samples that have it.
- `align_loss`: the global prototype alignment term before the weight. It is null until the client has received a global prototype library, so it is null in round 1.
- `aux_loss_weighted`: what was added to the loss, `prototype_loss_weight` times the mean of all the distillation and alignment terms.
- `aux_fraction`: `aux_loss_weighted / train_loss`.

The FedAMM-only fields are null for the other methods. `metrics/rounds.jsonl` stores the unweighted mean of each across the clients of the round as `mean_<name>`, for example `mean_seg_loss` and `mean_aux_fraction`.
