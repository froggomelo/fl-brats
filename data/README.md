# Dataset layout

The default cluster dataset root is `/datasets/$USER/brats` (set it with
`FL_BRATS_DATA_ROOT`). It is user-owned and intentionally excluded from Git.

```text
/datasets/$USER/brats/
├── raw/
│   └── brats2021/
├── processed/
│   └── brats2021_v1/
└── partitions/
    └── fixed_quantity_skew/
        ├── base/
        │   ├── client_0.json
        │   ├── client_1.json
        │   ├── client_2.json
        │   ├── client_3.json
        │   └── manifest.json
        ├── modality_control/
        ├── alpha_0p001/
        ├── alpha_0p1/
        └── alpha_1p0/
```

`base` will contain the canonical subject assignment. The modality-control and
alpha variants must retain that same subject allocation, changing only modality
availability. JSON partition manifests are future data infrastructure and are
not created by this repository skeleton.
