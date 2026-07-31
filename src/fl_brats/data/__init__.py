"""BraTS discovery, validation, preprocessing manifests, and data loading."""

from .labels import brats_to_regions

__all__ = [
    "BraTSDataset",
    "SubjectRecord",
    "brats_to_regions",
    "build_dataloader",
    "discover_records",
    "preprocess_index",
    "validate_subject",
]


def __getattr__(name: str):
    if name in {"BraTSDataset", "build_dataloader"}:
        from .brats import BraTSDataset, build_dataloader

        return {"BraTSDataset": BraTSDataset, "build_dataloader": build_dataloader}[name]
    if name in {"SubjectRecord", "discover_records"}:
        from .discovery import SubjectRecord, discover_records

        return {"SubjectRecord": SubjectRecord, "discover_records": discover_records}[name]
    if name == "preprocess_index":
        from .preprocessing import preprocess_index

        return preprocess_index
    if name == "validate_subject":
        from .validation import validate_subject

        return validate_subject
    raise AttributeError(name)
