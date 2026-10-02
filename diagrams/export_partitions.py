"""Snapshot the saved client partitions for the diagram scripts.

Run on the cluster, where data/partitions/ holds the client manifests:

    python3 diagrams/export_partitions.py diagrams/data/partitions.json

Each client manifest (data/partitions/<name>/client_<k>.json) lists its
subjects with their available modalities. The snapshot keeps only the subject
count and the count of each modality combination per client.
"""

import json
import sys
from collections import Counter
from pathlib import Path

PARTITIONS = {
    "fedavg_even": "Even, K = 4",
    "fedavg_uneven": "Uneven, K = 4",
    "fedavg_train_configs_uneven/k3": "Uneven, K = 3",
    "fedavg_train_configs_uneven/k5": "Uneven, K = 5",
    "fedavg_train_configs_uneven/k8": "Uneven, K = 8",
    "missing_uneven/alpha_0_001": "q = 0.001",
    "missing_uneven/alpha_0_1": "q = 0.1",
    "missing_uneven/alpha_0_2": "q = 0.2",
    "missing_dirichlet/alpha_0_001": "α = 0.001",
    "missing_dirichlet/alpha_0_1": "α = 0.1",
    "missing_dirichlet/alpha_1": "α = 1",
}


def client_index(path: Path) -> int:
    return int(path.stem.split("_")[1])


def main() -> None:
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("diagrams/data/partitions.json")
    snapshot = {}
    for name, label in PARTITIONS.items():
        root = Path("data/partitions") / name
        clients = []
        for path in sorted(root.glob("client_*.json"), key=client_index):
            client = json.loads(path.read_text())
            combos = Counter("+".join(s["available_modalities"]) for s in client["subjects"])
            clients.append(
                {
                    "client": client.get("client_name", path.stem),
                    "n": client["subject_count"],
                    "combinations": dict(sorted(combos.items())),
                }
            )
        if not clients:
            raise SystemExit(f"no client manifests under {root}")
        snapshot[name] = {"label": label, "clients": clients}
    text = json.dumps(snapshot, indent=1) + "\n"
    if str(output) == "-":
        sys.stdout.write(text)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text)


if __name__ == "__main__":
    main()
