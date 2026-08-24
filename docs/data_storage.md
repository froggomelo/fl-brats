# Storage policy

Use `/datasets/$USER/brats` as `FL_BRATS_DATA_ROOT`. Raw files and headers are
never modified. Derived data, reports, and all manifests belong only in the
processed or partition roots.

## Wits mscluster BraTS 2021 path

The extracted training set on mscluster is:

```text
/datasets/$USER/brats/raw/brats2021/PKG - RSNA-ASNR-MICCAI-BraTS-2021/RSNA-ASNR-MICCAI-BraTS-2021/BraTS2021_TrainingSet
```

Configure it explicitly before indexing, preprocessing, verification, or
training:

```bash
export FL_BRATS_DATA_ROOT="/datasets/$USER/brats"
export FL_BRATS_RAW_ROOT="$FL_BRATS_DATA_ROOT/raw/brats2021/PKG - RSNA-ASNR-MICCAI-BraTS-2021/RSNA-ASNR-MICCAI-BraTS-2021/BraTS2021_TrainingSet"
```

Index and client-manifest paths such as
`new-not-previously-in-TCIA/BraTS2021_01637/...` are relative to
`FL_BRATS_RAW_ROOT`. Do not resolve them directly below
`$FL_BRATS_DATA_ROOT/raw/brats2021`; that omits the package and training-set
directories shown above.

Verify an experiment partition with the raw root exported in the same shell:

```bash
FL_BRATS_RAW_ROOT="/datasets/$USER/brats/raw/brats2021/PKG - RSNA-ASNR-MICCAI-BraTS-2021/RSNA-ASNR-MICCAI-BraTS-2021/BraTS2021_TrainingSet" \
  .venv/bin/fedbrats verify-manifests \
  --partition-root data/partitions/fedavg_uneven
```
