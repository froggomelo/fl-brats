# Fixed quantity skew

Only training IDs are partitioned. The default client fractions are 0.50, 0.25, 0.15, and 0.10; floor rounding is used and the final client receives the remainder. Client JSON files contain IDs and metadata, never copied images. `verify-manifests` checks hashes, coverage, duplicates, and leakage.

Each client subject includes its dataset-index row and relative modality paths.
Use `verify-manifests --check-nifti` to additionally open every referenced
NIfTI file.
