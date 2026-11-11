# Diagrams

Scripts that draw the data pipeline and the client partitions in the results page's style.

```bash
python3 diagrams/make_all.py
```

Each figure is written to `diagrams/out/` as PNG (220 dpi, for slides and email) and PDF (vector, for the thesis).

| Script | Output | Shows |
|---|---|---|
| `fig_pipeline.py` | `pipeline` | Split, preprocessing, partitioning, the federated round, and evaluation |
| `fig_client_sizes.py` | `client_sizes` | Training subjects per client for the even, uneven, and K = 3, 5, 8 partitions |
| `fig_modality_counts.py` | `modality_counts` | Modalities kept per subject under the current q partitions and the FedAMM α procedure |
| `fig_combinations.py` | `modality_combinations` | Each client's share of the 15 modality combinations under both schemes |
| `fig_system.py` | `system` | The experimental system at one-column report width |
| `fig_partitions.py` | `partition_sizes`, `partition_modalities` | Report figures from the real partitions: client sizes (one column) and modalities kept per client (two columns) |
| `fig_fedamm.py` | `fedamm` | The modality-specific network and what FedAMM adds to it |
| `fig_compression.py` | `compression` | How a client update is compressed for upload |
| `fig_results.py` | `tradeoff`, `robustness`, `qualitative` | Report Results figures: Dice against traffic, Dice per modality combination, and example segmentations |

`fig_results.py` reads `data/results.json`, the reported batch (RTX 3060) runs taken from the results page's embedded data on 8 November 2026; the interim bigbatch copies of the ablation are left out. Re-export it if runs are re-evaluated.

The partition figures read `data/partitions.json`, a snapshot of the client manifests. Refresh it on the cluster after any partition changes:

```bash
python3 diagrams/export_partitions.py diagrams/data/partitions.json
```

The FedAMM α panels are simulated with `paper_style_partition` in `common.py`, which follows `generate_mask_distribution_only_maskid` in the FedAMM release (github.com/13sky/FedAMM) on this study's client sizes, seed 1024. The real partitions (`missing_dirichlet/alpha_{0_001,0_1,1}`, built with seed 42) are now in `partitions.json`; the α panels can switch to them.
