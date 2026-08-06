# Missing MRI modalities

Missingness is explicit metadata, never inferred from image intensities. `complete` assigns one deterministic availability set to every subject of a client; `arbitrary` assigns each subject independently from a SHA-256-derived random stream. Assignments are written into client manifests with available/missing lists, a canonical `[t1, t1ce, t2, flair]` boolean mask, and a policy fingerprint.

The legacy masked U-Net baseline retains four MRI image channels, zero-fills unavailable channels, and can concatenate four spatial presence channels (`model=unet_missing_modality_baseline`). It is a modality-aware baseline, not a reproduction of FedAMM.

For the direct FedAMM control, use `configs/experiment/federated_missing_modality_fedavg.yaml`. It uses the FedAMM-compatible modality-specific encoders and shared decoder with the same explicit per-subject masks, but runs `strategy: fedavg`, disables prototype distillation/global prototype alignment/prototype emission, and disables modality-weighted encoder aggregation. Compare that run directly against `configs/experiment/federated_fedamm.yaml` using the same saved partition manifests and the same `initial_weights` checkpoint.

Example federation: client 0 can retain all modalities, client 1 `[t1, t2, flair]`, client 2 `[t2, flair]`, and client 3 can use arbitrary subject-level masks. Build it with `python -m fl_brats create-partition --split-path ... --output-dir ... missing_modalities.enabled=true missing_modalities.mode=complete`; inspect it with `python -m fl_brats inspect-modalities --partition-root ...`.

Use `missing_modalities.mode=arbitrary` and overrides such as `missing_modalities.arbitrary.per_modality_missing_probability.t1ce=0.4` for subject-level missingness. `evaluate_modality_combinations` applies all fifteen non-empty counterfactual masks without changing the stored manifest.
