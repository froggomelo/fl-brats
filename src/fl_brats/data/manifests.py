"""Deep integrity and path verification for split and client manifests."""

import csv
import json
import os
from pathlib import Path

from .hashing import sha256_file, sha256_payload
from .nifti import inspect_nifti
from .modalities import MODALITIES, combination_to_mask, canonical_modality_order


def _hashed(payload: dict[str, object], key: str) -> bool:
    copy = dict(payload)
    claimed = copy.pop(key, None)
    return claimed == sha256_payload(copy)


def verify_manifests(root: Path, check_nifti: bool = False) -> list[str]:
    errors = []
    split_path = root / "split_manifest.json"
    partition_path = root / "manifest.json"
    if not split_path.exists() or not partition_path.exists():
        return ["missing split or partition manifest"]
    split = json.loads(split_path.read_text())
    partition = json.loads(partition_path.read_text())
    if not _hashed(split, "split_manifest_hash"):
        errors.append("split hash mismatch")
    if not _hashed(partition, "partition_manifest_hash"):
        errors.append("partition hash mismatch")
    index_path = Path(partition.get("dataset_index_path", split.get("dataset_index_path", "")))
    if not index_path.is_file():
        return errors + [f"dataset index missing: {index_path}"]
    if sha256_file(index_path) != split.get("dataset_index_hash") or sha256_file(
        index_path
    ) != partition.get("dataset_index_hash"):
        errors.append("dataset index hash mismatch")
    rows = list(csv.DictReader(index_path.open(encoding="utf-8")))
    by_id = {row["subject_id"]: (i, row) for i, row in enumerate(rows)}
    raw_root = (
        Path(os.environ["FL_BRATS_RAW_ROOT"])
        if os.environ.get("FL_BRATS_RAW_ROOT")
        else (
            Path(
                json.loads((index_path.parent / "dataset_summary.json").read_text()).get(
                    "raw_root", index_path.parent
                )
            )
        )
        if (index_path.parent / "dataset_summary.json").exists()
        else index_path.parent
    )
    train, valid, test = map(
        set, (split["train_subjects"], split["validation_subjects"], split["test_subjects"])
    )
    assigned = []
    if train & valid or train & test or valid & test:
        errors.append("split overlap")
    for listed in partition["clients"]:
        path = root / f"client_{listed['client_id']}.json"
        if not path.exists():
            errors.append(f"missing client manifest: {path.name}")
            continue
        client = json.loads(path.read_text())
        if not _hashed(client, "client_manifest_hash"):
            errors.append(f"client {listed['client_id']}: hash mismatch")
        if client.get("source_split_hash") != split.get("split_manifest_hash"):
            errors.append(f"client {listed['client_id']}: wrong split hash")
        if client.get("dataset_index_hash") != split.get("dataset_index_hash"):
            errors.append(f"client {listed['client_id']}: wrong index hash")
        if client.get("subject_count") != len(client.get("subject_ids", [])):
            errors.append(f"client {listed['client_id']}: count mismatch")
        for ref in client.get("subjects", []):
            subject = ref.get("subject_id")
            assigned.append(subject)
            if subject not in by_id:
                errors.append(f"client {listed['client_id']}: unknown subject {subject}")
                continue
            row_number, row = by_id[subject]
            if ref.get("index_row") != row_number:
                errors.append(f"client {listed['client_id']}: incorrect index row for {subject}")
            available = ref.get("available_modalities", list(MODALITIES))
            missing = ref.get("missing_modalities", [m for m in MODALITIES if m not in available])
            try:
                ordered = canonical_modality_order(available)
                if list(combination_to_mask(ordered).tolist()) != ref.get("modality_mask", combination_to_mask(ordered).tolist()):
                    errors.append(f"client {listed['client_id']}: inconsistent modality mask for {subject}")
                if set(missing) != set(MODALITIES) - set(ordered):
                    errors.append(f"client {listed['client_id']}: inconsistent missing modalities for {subject}")
            except ValueError as error:
                errors.append(f"client {listed['client_id']}: invalid modalities for {subject}: {error}")
                ordered = ()
            for role, path_value in ref.get("paths", {}).items():
                if role in MODALITIES and role not in ordered:
                    if path_value is not None:
                        errors.append(f"client {listed['client_id']}: unavailable {role} has a path for {subject}")
                    continue
                expected = row["segmentation_path" if role == "seg" else f"{role}_path"]
                if path_value != expected:
                    errors.append(
                        f"client {listed['client_id']}: incorrect {role} path for {subject}"
                    )
                    continue
                file = raw_root / path_value
                if not file.is_file():
                    errors.append(f"client {listed['client_id']}: missing path {file}")
                    continue
                if not file.name.startswith(f"{subject}_") or not file.name.endswith(
                    f"_{'seg' if role == 'seg' else role}.nii.gz"
                ):
                    errors.append(f"client {listed['client_id']}: path role mismatch {file}")
                if check_nifti:
                    try:
                        inspect_nifti(file)
                    except Exception as error:
                        errors.append(
                            f"client {listed['client_id']}: unreadable path {file}: {error}"
                        )
        masks = {tuple(x.get("modality_mask", [True] * 4)) for x in client.get("subjects", [])}
        # Complete policies are auditable through their fingerprint/config; detect obvious inconsistency too.
        if client.get("missingness_mode") == "complete" and len(masks) > 1:
            errors.append(f"client {listed['client_id']}: complete missingness varies by subject")
    if len(assigned) != len(set(assigned)):
        errors.append("duplicate client subject")
    if set(assigned) != train:
        errors.append("partition does not cover exactly the training split")
    if set(assigned) & (valid | test):
        errors.append("validation/test leakage")
    return errors
