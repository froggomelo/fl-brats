# data

This folder holds the published split, dataset index, and client partitions behind every reported result. Read it to know what is tracked here and what stays outside the repository.

The BraTS 2021 volumes are not redistributed. Download them as described in [docs/data.md](../docs/data.md) and point `FL_BRATS_RAW_ROOT` at them. Every file below refers to scans by subject ID and by a path relative to that folder, never by image content.

```text
data/
├── index/brats2021_v1/               dataset index the split was built from
│   ├── dataset_index.csv             one row per subject, SHA-256 7b723da8...
│   ├── dataset_summary.json
│   └── invalid_subjects.json
├── splits/brats2021.json             canonical 70/15/15 split, split_manifest_hash 8c1e4804...
├── partitions/                       client partitions, one folder per data.partition_root
│   ├── fedavg_uneven/                client_0.json ... client_3.json and manifest.json
│   ├── fedavg_even/
│   ├── fedavg_train_configs_uneven/  k3, k5, k8 and the k3_lorenz, k5_lorenz, k8_lorenz sweeps
│   ├── missing_uneven/               per-modality dropout q = 0.001, 0.1, 0.2
│   ├── missing_dirichlet/            Dirichlet combinations alpha = 0.001, 0.1, 1
│   ├── missing_dirichlet_k5/         alpha = 0.1 at K = 5
│   ├── missing_dirichlet_k8/         alpha = 0.1 at K = 8
│   └── missing_per_client/           flair_largest
├── raw/                              placeholder, ignored by git
└── processed/                        placeholder, ignored by git
```

These are the exact files the reported runs read, copied from the cluster. Every partition derives from the split above and covers its 875 training subjects. The validation and test subjects match the cohorts recorded by every synced run, and the client sizes and modality mixes match `diagrams/data/partitions.json` and the runs' client logs.

Training refuses a dataset index whose SHA-256 differs from the one frozen in the split. Use the published index rather than building a new one.
