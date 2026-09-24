#!/usr/bin/env python3
"""Real two-client, two-round Flower smoke run with tiny synthetic NIfTI data."""

import argparse
import csv
import json
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from fl_brats.data.hashing import sha256_file, sha256_payload
from fl_brats.data.partitioning import create_partition
from fl_brats.federated.orchestration import run_local_federation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.output or Path(tempfile.mkdtemp(prefix="flbrats-heldout-smoke-"))
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(42)
    rows = []
    for number in range(6):
        subject = f"synthetic_{number}"
        row = {"subject_id": subject}
        for modality in ("t1", "t1ce", "t2", "flair", "seg"):
            path = root / f"{subject}_{modality}.nii.gz"
            if path.exists():
                raise ValueError(f"refusing to overwrite {path}")
            data = np.random.default_rng(number).normal(size=(8, 8, 8)).astype(np.float32)
            if modality == "seg":
                data = np.zeros((8, 8, 8), dtype=np.uint8)
                data[2:4, 2:4, 2:4] = 4
            nib.save(nib.Nifti1Image(data, np.eye(4)), path)
            row[f"{modality}_path"] = str(path.resolve())
        row["segmentation_path"] = row["seg_path"]
        rows.append(row)
    index = root / "index.csv"
    with index.open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    split = {
        "train_subjects": [r["subject_id"] for r in rows[:4]],
        "validation_subjects": [rows[4]["subject_id"]],
        "test_subjects": [rows[5]["subject_id"]],
        "dataset_index_path": str(index.resolve()),
        "dataset_index_hash": sha256_file(index),
    }
    split["split_manifest_hash"] = sha256_payload(split)
    split_path = root / "split.json"
    split_path.write_text(json.dumps(split))
    create_partition(split_path, root / "partitions", [0.5, 0.5])
    print(json.dumps(split, indent=2), flush=True)
    values = {
        "device": "cpu",
        "seed": 42,
        "num_clients": 2,
        "num_rounds": 2,
        "server_port": 0,
        "num_workers": 0,
        "local_batch_size": 1,
        "local_epochs": 1,
        "partition_root": str(root / "partitions"),
        "model": {"name": "unet", "channels": [2, 4], "strides": [2], "num_res_units": 0},
        "data": {"split_path": str(split_path), "patch_size": [8, 8, 8], "divisible_by": 2},
        "loss": {"name": "dice_ce"},
    }
    summary = run_local_federation(values, root / "run")
    rounds = [
        json.loads(line)
        for line in (root / "run/metrics/validation_rounds.jsonl").read_text().splitlines()
    ]
    assert [row["round"] for row in rounds] == [1, 2]
    assert all(row["subjects"] == 1 for row in rounds)
    assert not (root / "run/metrics/client_diagnostics.jsonl").exists()
    assert summary["status"] == "completed"
    print(f"SMOKE PASSED: {root}", flush=True)


if __name__ == "__main__":
    main()
