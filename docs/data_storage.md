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

The path `$FL_BRATS_DATA_ROOT/raw/brats2021/new-not-previously-in-TCIA` does
not exist in the mscluster extraction. An index or partition manifest that
contains that path is stale and must be rebuilt from the configured
`FL_BRATS_RAW_ROOT`; setting the environment variable alone does not rewrite
absolute paths already stored in a manifest.
