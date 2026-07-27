# BraTS 2021 data pipeline

Raw BraTS data are immutable and remain below `$FL_BRATS_DATA_ROOT/raw/brats2021`. Discovery is recursive, so source-collection directories are supported. The pipeline validates NIfTI files, writes an index, then creates deterministic split and client manifests. It does not train a model, copy raw data per client, or create missing-modality conditions.

Set `FL_BRATS_DATA_ROOT=/datasets/$USER/fedbrats`; the project-level `data_root`
configuration resolves this environment value. Use strict validation before
indexing. `preprocessing.materialize=false` performs manifest-only processing;
set it to `true` only for reusable offline output. Explicit command options
override trailing Hydra-style `key=value` arguments.

## Known limitations

Size and modification-time source detection is weaker than full SHA-256
hashing. Offline preprocessing deliberately includes no augmentation, missing-
modality partitions, model checks, or training checks. Runtime and storage for
the full dataset depend on selected transforms and output format.
