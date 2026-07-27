"""Validation for CSV dataset indexes."""

import csv
from pathlib import Path


def validate_index(index_path: Path, allow_incomplete: bool = False) -> list[str]:
    """Return validation errors; incomplete records require explicit opt-in."""
    with index_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    ids = [row["subject_id"] for row in rows]
    errors = [f"duplicate subject ID: {value}" for value in sorted({x for x in ids if ids.count(x) > 1})]
    if not allow_incomplete:
        errors.extend(f"incomplete subject: {row['subject_id']}" for row in rows if row.get("complete", "False") != "True")
    return errors
