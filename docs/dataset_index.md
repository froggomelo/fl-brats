# Dataset index

`dataset_index.csv` is concise and `dataset_index.jsonl` carries the same rows. `dataset_summary.json` describes geometry, labels, sizes, and index hash; `invalid_subjects.json` preserves all validation failures. Paths are relative to the configured raw root by default, allowing relocation.

The summary reports duplicate IDs, ambiguous files, zero-byte and unreadable
files by role with paths/reasons, and valid/invalid byte totals. Review the
invalid-subject report before making a split.
