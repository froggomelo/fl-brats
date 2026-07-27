# BraTS preprocessing

Manifest-only indexing is the default. Materialization is optional and writes atomically under `processed/brats2021_v1/subjects`. It can canonically orient to RAS, optionally resample (nearest neighbour for segmentation), crop one shared foreground box, and apply per-subject/per-modality foreground z-score normalization. NIfTI is the default output; tensors are optional. Metadata records hashes and transform details, enabling safe resume.

Resume requires a completion marker, expected outputs, matching configuration,
and matching source metadata. Enable `hashing.source_files=true` for SHA-256
comparison; otherwise size and nanosecond mtime are used. Padding supports an
explicit target shape or divisibility constraint and records exact widths.
