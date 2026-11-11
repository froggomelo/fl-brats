# Data pipeline

This guide takes BraTS 2021, the Brain Tumour Segmentation challenge dataset, from download to the client manifests that training reads. Read it before the first experiment on a new account, and again whenever a run needs a partition that does not exist yet.

The pipeline runs in a fixed order. Download the data, validate it, index it, preprocess it, freeze one train/validation/test split, and partition the training subjects across clients. Each stage writes hashed JSON or CSV outputs that the next stage checks. Run the `fedbrats` commands from the repository root with the project environment active, or prefix them with `uv run`. [setup.md](setup.md) covers the environment.

## Storage layout

All data live outside the repository under one data root, set by `FL_BRATS_DATA_ROOT`. On the Wits mscluster use `/datasets/$USER/brats`. The repository `.env`, the downloader, and every data script default to that path. Home directories on mscluster are capped at 50 GB and the BraTS package is about 142 GB.

```text
$FL_BRATS_DATA_ROOT/
├── raw/
│   └── brats2021/                  downloaded package, never modified
├── processed/
│   └── brats2021_v1/               index, reports, and optional materialised subjects
└── partitions/
    └── fixed_quantity_skew/        canonical split and the base four-client partition
```

Raw files and their headers are read-only for the whole pipeline. Every derived file goes below `processed/` or `partitions/`. Experiment partitions are the exception. They are written inside the repository under `data/partitions/<name>`. [data/README.md](../data/README.md) describes that folder.

## Replicate with the published files

The repository ships the dataset index, the canonical split, and all 17 client partitions used in the report, under `data/`. A replication needs only the raw scans. Download BraTS 2021 as described below, then point `.env` at the published files and skip the index, split, and partition steps.

```bash
export FL_BRATS_RAW_ROOT="/path/to/RSNA-ASNR-MICCAI-BraTS-2021/BraTS2021_TrainingSet"
export DATASET_INDEX_PATH="$PWD/data/index/brats2021_v1/dataset_index.csv"
export FL_BRATS_SPLIT_PATH="$PWD/data/splits/brats2021.json"
```

`FL_BRATS_RAW_ROOT` must be the folder that holds the source collection folders, such as `UCSF-PDGM/`. Every path in the index and the client manifests is relative to it. The rest of this guide describes how those files were produced, and how to build partitions for new configs.

## Environment variables

Set these variables in `.env` before running anything. The Slurm scripts source `.env` themselves. Source it by hand in an interactive shell.

```bash
set -a
source .env
set +a
```

| Variable | Meaning |
|---|---|
| `FL_BRATS_DATA_ROOT` | Data root above `raw/`, `processed/`, and `partitions/` |
| `FL_BRATS_RAW_ROOT` | Folder that holds the labelled subject directories |
| `DATASET_INDEX_PATH` | Dataset index. `$FL_BRATS_DATA_ROOT/processed/brats2021_v1/dataset_index.csv` on mscluster, or the published `data/index/brats2021_v1/dataset_index.csv` |
| `FL_BRATS_SPLIT_PATH` | Canonical split. `$FL_BRATS_DATA_ROOT/partitions/fixed_quantity_skew/split_manifest.json` on mscluster, or the published `data/splits/brats2021.json` |
| `FASPEX_URL` | Package link for the downloader |
| `TOOLCHAIN_DIR` | Install location of the Aspera client used by the downloader |
| `FL_BRATS_CACHE_ROOT` | Shared cache root for the dataset cache and Matplotlib |

`FL_BRATS_RAW_ROOT` must point at the labelled training set inside the package. The package's own validation set has no `*_seg.nii.gz` labels and cannot be used.

```bash
export FL_BRATS_RAW_ROOT="$FL_BRATS_DATA_ROOT/raw/brats2021/PKG - RSNA-ASNR-MICCAI-BraTS-2021/RSNA-ASNR-MICCAI-BraTS-2021/BraTS2021_TrainingSet"
```

Source `.env` before running `fedbrats`. The command does not load it itself. Without it the Hydra config falls back to `/datasets/$USER/brats` for the data root and `<data root>/raw/brats2021` for the raw root.

## Download BraTS 2021

The dataset is distributed by TCIA, The Cancer Imaging Archive, as a single Aspera Faspex package of about 142 GB. Put the package link in `FASPEX_URL` in `.env`, then submit the download job from the repository root.

```bash
make log-dirs
sbatch scripts/data_preprocessing/brats_download.sbatch
```

The job installs the Aspera client with `scripts/environment/install_aspera.sh`, transfers the package to `$FL_BRATS_DATA_ROOT/raw/brats2021`, verifies MD5 checksums of the training NIfTI files against the package's checksum list, and logs the on-disk size every 30 minutes. NIfTI, the Neuroimaging Informatics Technology Initiative format, is the volume format BraTS ships in. A positional argument replaces the destination. Any further arguments go to `scripts/data_preprocessing/download_brats2021.py`.

```bash
sbatch scripts/data_preprocessing/brats_download.sbatch /datasets/$USER/somewhere-else
sbatch scripts/data_preprocessing/brats_download.sbatch --target-rate-kbps 500000
```

A failed or preempted transfer resumes from where it stopped. Resubmit the identical command. Compute nodes need outbound UDP on port 33001. If the transfer stalls at 0 bytes, run the Python downloader on the cluster's sideload host instead, as the comment block at the top of `brats_download.sbatch` describes. Check a finished download without transferring anything with `--validate-only`.

```bash
python3 scripts/data_preprocessing/download_brats2021.py --dest "$FL_BRATS_DATA_ROOT/raw/brats2021" --validate-only
```

## Validate the NIfTI files

Validation opens every modality and segmentation with nibabel. It checks for missing or ambiguous files, duplicate subject IDs, 3-D shape, finite affine and spacing, matching geometry and orientation across the five files of a subject, and integer segmentation labels in the BraTS set 0, 1, 2, and 4. Unexpected labels are errors by default. The report keeps every failure of every subject.

```bash
bash scripts/data_preprocessing/validate_brats.sh
```

The script runs the following command and writes the per-subject report to `processed/brats2021_v1/validation_report.json`.

```bash
fedbrats validate-data --root "$FL_BRATS_RAW_ROOT" --strict \
  --report-path "$FL_BRATS_DATA_ROOT/processed/brats2021_v1/validation_report.json"
```

`--strict` exits non-zero on any error. `--report-only` writes the report and exits zero. For a fast count of complete and incomplete subject directories without opening any file, use `discover-data`.

```bash
fedbrats discover-data --root "$FL_BRATS_RAW_ROOT"
```

Do not validate the full dataset on the login node. Submit it as a job, as described in [Run the pipeline as Slurm jobs](#run-the-pipeline-as-slurm-jobs).

## Build the dataset index

The index lists every discovered subject with its file paths, geometry, labels, size, and validation status. Discovery is recursive. Nested collection folders inside the package are found.

```bash
bash scripts/data_preprocessing/index_brats.sh
```

This calls `fedbrats index-data --root "$FL_BRATS_RAW_ROOT" --output-dir "$FL_BRATS_DATA_ROOT/processed/brats2021_v1"` and writes four files.

| File | Contents |
|---|---|
| `dataset_index.csv` | One row per subject. Split and partition manifests hash this file. |
| `dataset_index.jsonl` | The same rows as JSON lines |
| `dataset_summary.json` | Counts, geometry and label distributions, byte totals, raw root, and the index SHA-256 |
| `invalid_subjects.json` | Full validation results of every invalid subject |

Paths in the index are relative to `FL_BRATS_RAW_ROOT`. A path such as `new-not-previously-in-TCIA/BraTS2021_01637/...` resolves only against that root, not against `$FL_BRATS_DATA_ROOT/raw/brats2021`. Relative paths let the data move without rebuilding the index. Review `invalid_subjects.json` before creating a split. Print the summary with `inspect-data`.

```bash
fedbrats inspect-data --summary-path "$FL_BRATS_DATA_ROOT/processed/brats2021_v1/dataset_summary.json"
```

## Preprocess

Training does not read preprocessed volumes. Clients load the raw NIfTI files named in their manifests and transform them on the fly with MONAI, the PyTorch medical imaging library. [Cache preprocessed volumes on the node](#cache-preprocessed-volumes-on-the-node) describes the optional cache. Offline preprocessing is a reusable product for inspection and other tools. Skip it if neither is needed.

`configs/preprocessing/brats2021_v1.yaml` controls this stage. Its default `materialize: true` writes one folder per valid subject under `processed/brats2021_v1/subjects/<subject_id>/`. Each subject is reoriented to RAS, the canonical right-anterior-superior axis order, cropped to one foreground box shared by its four modalities, and z-score normalised per modality over non-zero voxels. Resampling and padding are available in the same file and are off by default. Output is NIfTI unless `output_format` is set to `tensor`.

```bash
bash scripts/data_preprocessing/preprocess_brats.sh
```

The script runs `preprocess-data` with `--resume`.

```bash
fedbrats preprocess-data \
  --index-path "$FL_BRATS_DATA_ROOT/processed/brats2021_v1/dataset_index.csv" \
  --raw-root "$FL_BRATS_RAW_ROOT" \
  --output-dir "$FL_BRATS_DATA_ROOT/processed/brats2021_v1" --resume
```

For a manifest-only run that writes `preprocessing_config.yaml` and `preprocessing_summary.json` but no subject volumes, add a Hydra override. Explicit command options take precedence over trailing `key=value` overrides.

```bash
fedbrats preprocess-data --index-path "$DATASET_INDEX_PATH" --raw-root "$FL_BRATS_RAW_ROOT" \
  --output-dir "$FL_BRATS_DATA_ROOT/processed/brats2021_v1" preprocessing.materialize=false
```

Each subject is written to a temporary folder and renamed into place on success. Its `metadata.json` records source file state, original and final geometry, crop and padding widths, normalisation statistics, and the configuration hash. `complete.json` marks a finished subject. With `--resume`, a subject is skipped only when both markers and all outputs exist and the configuration hash and source state match. Source state is file size and modification time by default. Set `preprocessing.hashing.source_files=true` to compare SHA-256 hashes instead, at the cost of reading every file. `--overwrite` rebuilds existing subjects. `--subject-id` and `--limit` restrict a run for testing.

## Create the fixed split

One subject-level split serves every experiment. It is created once and never regenerated by training. Only subjects that are complete and valid in the index take part.

```bash
bash scripts/data_preprocessing/create_dataset_split.sh
```

The script runs the following command with the defaults of `configs/split/default.yaml`, seed 42 and fractions 0.70, 0.15, and 0.15.

```bash
fedbrats create-split \
  --index-path "$FL_BRATS_DATA_ROOT/processed/brats2021_v1/dataset_index.csv" \
  --manifest-path "$FL_BRATS_DATA_ROOT/partitions/fixed_quantity_skew/split_manifest.json"
```

It writes `train_subjects.json`, `validation_subjects.json`, `test_subjects.json`, and `split_manifest.json` beside each other. The manifest records the seed, requested and realised fractions and counts, the subject lists, the path and SHA-256 of the dataset index, and its own hash in `split_manifest_hash`. The comment above `FL_BRATS_SPLIT_PATH` in `.env` gives the expected hash prefix of the canonical split. Every partition and every held-out evaluation refers back to this file.

```bash
fedbrats inspect-split --manifest-path "$FL_BRATS_SPLIT_PATH"
```

## Partition the clients

A partition assigns the training subjects of the split to K clients, where K is the number of clients. Validation and test subjects are never assigned. Each client gets a `client_<n>.json` manifest that lists its subject IDs, index rows, relative modality paths, modality mask, and hashes. No images are copied. `manifest.json` summarises the partition, records requested and realised client fractions, and carries `partition_manifest_hash`.

The training subjects are shuffled with the partition seed and cut into consecutive blocks of the requested fractions. Each block size is floored. By default the final client takes the rounding remainder. With `data.partition_remainder: largest` every client is floored and the largest client takes the remainder instead. The same split, seed, and fractions always yield the same subjects per client.

### Base partition

The base partition is fixed quantity skew. Four clients hold 50, 25, 15, and 10 percent of the training subjects, as set in `configs/partition/fixed_quantity_skew.yaml`.

```bash
bash scripts/data_preprocessing/create_fixed_partition.sh
```

This calls `create-partition`, which writes the client manifests beside the split and copies the split into the output folder when it lives elsewhere. `--fractions` takes a JSON list and `--seed` sets the shuffle seed. Trailing `missing_modalities.*` overrides bake modality masks into the manifests.

```bash
fedbrats create-partition \
  --split-path "$FL_BRATS_SPLIT_PATH" \
  --output-dir "$FL_BRATS_DATA_ROOT/partitions/fixed_quantity_skew" \
  --fractions '[0.5,0.25,0.15,0.1]' --seed 42
```

### Experiment partitions

Every experiment config names its own partition in `data.partition_root`, a path under `data/partitions/` in the repository. The partitions of every reported experiment are already there. Create one only for a new config. `prepare-experiment-partition` reads `data.client_proportions`, `data.partition_remainder`, the seed, and the `missing_modalities` block from the config, including any `extends` chain.

```bash
fedbrats prepare-experiment-partition --config configs/fedavg_uneven/baseline.yaml \
  --split-path "$FL_BRATS_SPLIT_PATH"
```

`scripts/data_preprocessing/prepare_experiment_partition.sh` wraps the same command and takes the config as its first argument. Without `--split-path`, the command falls back to `data.split_path` from the config, which is `data/splits/brats2021.json` in `configs/_experiment_base/federated_uneven.yaml`. Always pass the canonical split. The command refuses a non-empty destination unless `--overwrite` is given.

The sizes below come from `diagrams/data/partitions.json` and the config comments. All K = 4 partitions split the same 875 training subjects.

| Partition root under `data/partitions/` | Config | Clients | Subjects per client |
|---|---|---|---|
| `fedavg_uneven` | `configs/fedavg_uneven/baseline.yaml` | 4 | 437, 218, 131, 89 |
| `fedavg_even` | `configs/fedavg_even/baseline.yaml` | 4 | 218, 218, 218, 221 |
| `fedavg_train_configs_uneven/k3` | `configs/fedavg_train_configs_uneven/client_sweeps/k3.yaml` | 3 | 500, 249, 126 |
| `fedavg_train_configs_uneven/k5` | `configs/fedavg_train_configs_uneven/client_sweeps/k5.yaml` | 5 | 451, 225, 112, 56, 31 |
| `fedavg_train_configs_uneven/k8` | `configs/fedavg_train_configs_uneven/client_sweeps/k8.yaml` | 8 | 439, 219, 109, 54, 27, 13, 6, 8 |
| `fedavg_train_configs_uneven/k5_lorenz` | `configs/fedavg_train_configs_uneven/client_sweeps/k5_lorenz.yaml` | 5 | 351, 218, 140, 96, 70 |
| `fedavg_train_configs_uneven/k8_lorenz` | `configs/fedavg_train_configs_uneven/client_sweeps/k8_lorenz.yaml` | 8 | 223, 218, 109, 109, 65, 65, 43, 43 |

### Client-count sweeps

Two sweeps vary K. The halving sweep in `client_sweeps/k3.yaml`, `k5.yaml`, and `k8.yaml` halves each successive client's share. At K = 8 its smallest clients hold only a handful of subjects. The Lorenz sweep in `client_sweeps/k3_lorenz.yaml`, `k5_lorenz.yaml`, and `k8_lorenz.yaml` keeps the shape of the K = 4 split. With clients sorted largest first, the K = 4 shares 50/25/15/10 give the cumulative points (0, 0), (0.25, 0.50), (0.50, 0.75), (0.75, 0.90), and (1, 1) in fraction of clients against fraction of subjects. Client i of K, counted from zero, gets F((i + 1) / K) − F(i / K), where F interpolates those points linearly. The Lorenz configs set `partition_remainder: largest`. For K = 3 this gives shares of 58.3, 26.7, and 15.0 percent.

The overlays `configs/_experiment_base/clients_lorenz/k5.yaml` and `k8.yaml` apply the Lorenz client counts to any method config. Layer one onto the method config and set `data.partition_root` after it.

### Missing-modality partitions

Missing-modality partitions keep the subject allocation of the matching quantity-skew split and change only which modalities each subject has. The masks are fixed when the partition is created. Training checks them against the config and refuses a partition built under a different `missing_modalities` policy. Create each variant from a config that carries the intended policy.

| Partition root under `data/partitions/` | Policy | Example config |
|---|---|---|
| `missing_uneven/alpha_0_001`, `alpha_0_1`, `alpha_0_2` | Each modality dropped independently with probability q of 0.001, 0.1, or 0.2 | `configs/fedavg_missing_modalities_uneven/alpha_0_2.yaml` |
| `missing_dirichlet/alpha_0_001`, `alpha_0_1`, `alpha_1` | Dirichlet draw over the 15 non-empty modality combinations per client, with concentration α | `configs/fedavg_missing_modalities_uneven/dirichlet_alpha_0_1.yaml` |
| `missing_dirichlet_k5/alpha_0_1`, `missing_dirichlet_k8/alpha_0_1` | Dirichlet α = 0.1 on the Lorenz K = 5 and K = 8 sizes | `configs/fedavg_missing_modalities_uneven/dirichlet_alpha_0_1_k5.yaml` |
| `missing_per_client/flair_largest` | Each client lacks one scan for every subject, and the largest client lacks FLAIR | `configs/fedavg_missing_modalities_uneven/per_client_flair_largest.yaml` |

The policy overlays live in `configs/_experiment_base/missing_uneven.yaml`, `configs/_experiment_base/missing_dirichlet/`, and `configs/_experiment_base/missing_per_client/`. [missing_modalities.md](missing_modalities.md) explains the policies, the two meanings of alpha, and how held-out subjects are masked.

### Verify and inspect a partition

`verify-manifests` checks the split, partition, and client hashes, the index hash, coverage of exactly the training split, duplicates, validation and test leakage, modality mask consistency, and that every referenced file exists. `--check-nifti` also opens every file. The command uses the `split_manifest.json` copy beside `manifest.json` when one exists. `create-partition` and `prepare-experiment-partition` both leave one there. Some published partitions have no copy, so the command falls back to `--split-path`, which defaults to `FL_BRATS_SPLIT_PATH`. The manifests store the absolute index path on mscluster. `--index-path`, which defaults to `DATASET_INDEX_PATH`, replaces it. The index hash is still checked.

```bash
export FL_BRATS_SPLIT_PATH="$PWD/data/splits/brats2021.json"
export DATASET_INDEX_PATH="$PWD/data/index/brats2021_v1/dataset_index.csv"
fedbrats verify-manifests --partition-root data/partitions/missing_dirichlet/alpha_0_1 --check-nifti
```

Relative scan paths resolve against `FL_BRATS_RAW_ROOT`. Export it in the same shell. Print a partition summary or its per-client modality counts with the inspect commands.

```bash
fedbrats inspect-partition --manifest-path data/partitions/fedavg_uneven/manifest.json
fedbrats inspect-modalities --partition-root data/partitions/missing_dirichlet/alpha_0_1
```

`scripts/data_preprocessing/inspect_data_pipeline.sh` prints the dataset summary, the canonical split, and the base partition in one run.

## Cache preprocessed volumes on the node

Loading and transforming a compressed NIfTI volume costs CPU time on every read. A persistent cache stores the output of the deterministic transforms once per subject. Random crops and flips stay live. Three keys in the `data` block of an experiment config control it.

| Key | Values | Effect |
|---|---|---|
| `data.cache_mode` | `dataset`, `persistent` | `dataset` reads raw files every time. `persistent` caches transformed volumes on disk. |
| `data.cache_location` | `shared`, `node` | `shared` writes to `$FL_BRATS_CACHE_ROOT/persistent`. `node` writes to `/var/tmp/$USER/fl-brats-cache` on the node's local SSD. |
| `data.cache_dir` | path | Overrides both locations |

`configs/_experiment_base/federated_uneven.yaml` sets `cache_mode: persistent` and `cache_location: node`. Every experiment that extends it caches on the local disk of each client node. An uncompressed cache entry is several times the size of its source file. Network storage cannot sustain that traffic from many nodes at once. Set `FL_BRATS_NODE_CACHE_ROOT` to move the node cache elsewhere.

Cache folders are named by a hash of orientation, spacing, patch size, and the divisibility constraint. Entries inside are keyed on the source file paths. A subject with the same available modalities therefore shares one entry across clients and experiments. The first round on a fresh node builds the cache and later rounds read it. Entries are written to a temporary file and renamed. A killed job never leaves a truncated entry. Delete a cache folder to force a rebuild.

## Run the pipeline as Slurm jobs

On the cluster, run every full-dataset stage as a job inside the Apptainer image. Build the images first as described in [cluster.md](cluster.md). The scripts pick `apptainer/fl-brats-stampede.sif` on the stampede partition and `apptainer/fl-brats-batch.sif` elsewhere. Run `make log-dirs` once per clone. Slurm does not create a job's output folder, and a job whose output folder is missing fails without a log.

### One resumable job

`slurm/data_preprocessing/brats_pipeline.sbatch` runs every stage in order on stampede with 10 CPUs and a 72-hour limit. It runs strict validation, indexing, a manifest-only preprocessing pass, materialised preprocessing with `--resume`, the fixed split, the base partition, and `verify-manifests --check-nifti`. Raw data are bound read-only. The processed and partition folders are bound writable. It stops at the first failure.

```bash
sbatch --export=ALL slurm/data_preprocessing/brats_pipeline.sbatch
```

After a preemption, resubmit the same command. Materialised subjects with matching completion metadata are skipped. Inside the container the data root is mounted at `/datasets/$USER/brats`. The manifests record the absolute index path. Keep `FL_BRATS_DATA_ROOT` at that path for them to resolve outside the container.

### Single stages

| Script | Stage |
|---|---|
| `slurm/data_preprocessing/brats_validate.sbatch` | Strict validation |
| `slurm/data_preprocessing/brats_preprocess.sbatch` | Materialised preprocessing with `--resume` |
| `slurm/data_preprocessing/brats_partition.sbatch` | Split, base partition, and verification |

Their `#SBATCH` lines hold shell defaults such as `${FL_BRATS_CPU_PARTITION:-cpu}`, which sbatch does not expand. A command-line `--partition` replaces the literal partition name.

```bash
sbatch --partition=stampede slurm/data_preprocessing/brats_partition.sbatch
```

`brats_validate.sbatch` and `brats_preprocess.sbatch` also put these defaults in their numeric `--cpus-per-task` and `--mem` lines. Slurm rejects such values while it reads the script. Use `brats_pipeline.sbatch` for validation and preprocessing until those lines hold plain values.

### Experiment partitions

`slurm/data_preprocessing/prepare_experiment_partition.sbatch` runs `prepare-experiment-partition` for one config inside the container. It has no partition directive. Name one at submission.

```bash
sbatch --partition=stampede slurm/data_preprocessing/prepare_experiment_partition.sbatch \
  configs/fedavg_uneven/baseline.yaml --split-path "$FL_BRATS_SPLIT_PATH"
```

Two helpers submit one such job per config. Both read `FL_BRATS_DATA_PARTITION` for the Slurm partition, with stampede as the default, and add `--overwrite` when `FL_BRATS_OVERWRITE_PARTITIONS=1`.

```bash
bash slurm/data_preprocessing/prepare_fedavg_baseline_partitions_submit.sh
bash slurm/data_preprocessing/prepare_training_sweep_partitions_submit.sh
```

The first creates `fedavg_uneven` and `fedavg_even`. The second creates the halving client-count sweeps and the partitions of the local-epoch and modality sweeps. Wait for every partition job to finish before submitting training.
