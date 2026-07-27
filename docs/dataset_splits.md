# Dataset splits

Only complete, valid subjects participate. A seeded, subject-level 70/15/15 split is written once to `train_subjects.json`, `validation_subjects.json`, `test_subjects.json`, and `split_manifest.json`. The manifest records requested/realised fractions and the source index hash. Training must not regenerate it.
