# fl-brats

Download, verify and validate the **RSNA-ASNR-MICCAI BraTS 2021** dataset
(1,251 labelled training subjects, NIfTI) on a headless Linux server.

## What the official source actually gives you

The TCIA page — <https://www.cancerimagingarchive.net/analysis-result/rsna-asnr-miccai-brats-2021/>
— publishes the challenge data as **one IBM Aspera Faspex package (~142 GB)**.
Facts verified against the live server:

| | |
| --- | --- |
| Transport | Aspera `fasp` — not HTTP. `curl`/`wget` cannot fetch it. |
| Partial download | **Not possible.** Per-path selection returns HTTP 500 (`undefined method 'starts_with?'`), a TCIA-side Faspex bug. You must transfer all 142 GB. |
| Auth | Public package link with an embedded passcode. No account needed. |
| Archives | None. The package is a plain file tree — nothing to extract. |
| Checksums | Yes: `RSNA-ASNR-MICCAI-BraTS-2021.sums`, MD5 for all 407,245 files. |

Layout inside the package:

```
RSNA-ASNR-MICCAI-BraTS-2021.sums          MD5 manifest, 47 MB
RSNA-ASNR-MICCAI-BraTS-2021/
├── BraTS2021_TrainingSet/                ← the 1,251 labelled subjects (NIfTI)
│   ├── ACRIN-FMISO-Brain/            (20 files)
│   ├── CPTAC-GBM/                   (165)
│   ├── IvyGAP/                      (150)
│   ├── TCGA-GBM/                    (510)
│   ├── TCGA-LGG/                    (325)
│   ├── UCSF-PDGM/                  (1315)
│   ├── UPENN-GBM/                  (2015)
│   └── new-not-previously-in-TCIA/ (1755)
│        └── BraTS2021_01374/
│             ├── BraTS2021_01374_t1.nii.gz
│             ├── BraTS2021_01374_t1ce.nii.gz
│             ├── BraTS2021_01374_t2.nii.gz
│             ├── BraTS2021_01374_flair.nii.gz
│             └── BraTS2021_01374_seg.nii.gz
├── BraTS2021_TrainingSet_dcm/            source DICOM (the bulk of the 142 GB)
├── BraTS2021_ValidationSet/              219 subjects, no seg labels
└── BraTS2021_ValidationSet_dcm/
```

Counts above are *file* counts from the manifest; subjects are one fifth of
each. The manifest holds exactly 1,251 × 5 = **6,255** training NIfTI files
(`t1`, `t1ce`, `t2`, `flair`, `seg`), which is what the validator checks.

## Setup

```bash
git clone <this repo> && cd fl-brats
cp .env.example .env          # then paste your Faspex link into FASPEX_URL

scripts/install_aspera.sh     # optional pre-install; the SLURM job also runs this
```

`install_aspera.sh` needs **no root**: it gets Ruby from environment modules, an
existing conda, or a private micromamba it bootstraps itself, and `ascp` is just
an unzipped binary. Everything lands in `TOOLCHAIN_DIR` (default `.toolchain/`)
and deletes as one directory. `scripts/brats_download.sbatch` invokes the
installer automatically before starting the download; repeated runs reuse the
existing installation.

The Python scripts need **no pip packages** — standard library only, Python 3.9+.
See [scripts/requirements.txt](scripts/requirements.txt).

Create the destination once (the only step needing sudo):

```bash
sudo mkdir -p /data/brats2021
sudo chown "$USER" /data/brats2021
```

## Usage

```bash
# put the toolchain on PATH for this shell
source scripts/_common.sh && activate_toolchain

scripts/download_brats2021.py --dest /data/brats2021
```

That is the whole run. It is **idempotent** — re-run the identical command after
any interruption and it resumes.

| Behaviour | How |
| --- | --- |
| Resume | `ascp` resume policy `sparse_csum`; complete files are skipped, partial files continue from their offset |
| Retries | 8 attempts, exponential backoff 15s → 600s with jitter |
| Progress | live `ascp` progress bar on the console; checkpointed to the log every 30s |
| Logging | everything to `/data/brats2021/download.log` |
| Disk gate | refuses to start below 152 GiB free (`--min-free-gb`) |
| Integrity | MD5 of every training file against the package `.sums` manifest |
| Re-verify cost | cached by (size, mtime) in `.verify-state.json`, so a second run re-hashes nothing |
| Secrets | the passcode never enters `argv` or the log — see below |

Useful flags:

```bash
--validate-only             # don't download; verify + validate what's on disk
--skip-verify               # skip MD5 (structure check only)
--verify-scope all          # MD5 all 407k files, not just the 6,255 training NIfTIs
--target-rate-kbps 500000   # cap at ~500 Mbps
--max-retries 20
```

### Credential handling

The Faspex link embeds a package passcode in its base64 `context=` parameter.
Two leaks are avoided deliberately:

- **Process list.** The URL is written to a mode-0600 temp file and handed to
  `ascli` as `--url=@file:<path>`, so it never appears in `argv` (which any
  local user can read via `/proc/<pid>/cmdline`). The file is created with
  `O_EXCL` at 0600 *before* any bytes are written, and removed when the run ends.
- **Log file.** Every line written to `download.log` passes through a redactor
  that replaces `context=…` and `passcode…` values with `<redacted>`.

Prefer `--url-file` or `BRATS_FASPEX_URL_FILE` over `BRATS_FASPEX_URL`: an
environment variable stays readable through `/proc/<pid>/environ` for the life
of the process tree.

```bash
umask 077; printf '%s' 'https://faspex.../package?context=...' > ~/.brats-url
scripts/download_brats2021.py --dest /data/brats2021 --url-file ~/.brats-url
```

## Validation

A separate read-only validator, safe to run at any time — including while a
download is still in flight:

```bash
scripts/validate_brats2021.py --dest /data/brats2021
scripts/validate_brats2021.py --dest /data/brats2021 --checksums
scripts/validate_brats2021.py --dest /data/brats2021 --json /tmp/report.json
```

It reports subjects found, complete subjects, incomplete subjects with their
specific missing files, per-collection breakdown, and total size on disk.
Exit status is 0 only when all 1,251 subjects are present with all five files.

## Running it safely under tmux

The transfer takes hours; an SSH drop must not kill it.

```bash
sudo apt-get install -y tmux          # if needed

tmux new -s brats                     # create a detachable session
source scripts/_common.sh && activate_toolchain
scripts/download_brats2021.py --dest /data/brats2021
#   detach with:  Ctrl-b  then  d
```

```bash
tmux ls                               # list sessions
tmux attach -t brats                  # reattach later, from any SSH session
tail -f /data/brats2021/download.log  # or just watch the log
```

`screen` equivalent: `screen -S brats`, detach with `Ctrl-a` then `d`,
reattach with `screen -r brats`.

Even without tmux the download is safe to lose — re-running the command resumes
from where it stopped.

## Configuration

`.env` (gitignored; see `.env.example`):

| Variable | Purpose |
| --- | --- |
| `PROJECT_DIR` | Project root. Blank = repo root. |
| `DATA_DIR` | Destination for the legacy Bash script. Relative paths resolve against `PROJECT_DIR`. |
| `FASPEX_URL` | TCIA Faspex public package link. |
| `ASPERA_TARGET_RATE_KBPS` | Bandwidth cap. Blank = unthrottled. |
| `ASPERA_RESUME_POLICY` | `none`, `attrs`, `sparse_csum`, `full_csum`. |
| `RUBY_INSTALL_METHOD` | `auto`, `system`, `module`, `conda`, `micromamba`, `apt` (sudo), `rvm`. |

`scripts/download_dataset.sh` is the earlier thin Bash wrapper around `ascli`.
It still works, but `download_brats2021.py` supersedes it — the Bash version has
no retries, no checksum verification, no validation, and passes the URL on the
command line where it is visible in `ps`.

## On a cluster

- Point `TOOLCHAIN_DIR` and `--dest` at scratch/project space if `$HOME` is quota'd.
- If your site provides Ruby ≥ 3.1, `module load ruby` and set
  `RUBY_INSTALL_METHOD=system` to skip the micromamba bootstrap.
- Submit as a batch job, or use tmux as above.
