#!/bin/bash
# Download the BraTS 2021 package (~142 GB) on the Wits mscluster via SLURM.
#
#   sbatch scripts/data_preprocessing/brats_download.sbatch
#   sbatch scripts/data_preprocessing/brats_download.sbatch /datasets/$USER/somewhere-else
#
# Submit from the repo root (the job uses $SLURM_SUBMIT_DIR to find the repo;
# override with FL_BRATS_REPO=/path/to/fl-brats).
#
# Before the first submission, ensure the repository's .env contains
# FL_BRATS_DATA_ROOT and FASPEX_URL.
# The job runs install_aspera.sh before downloading. The first run may download
# ~250 MB and take a couple of minutes; later runs reuse the installed toolchain.
#
# Cluster guidelines this script is built around (mscluster, 2024-02-08):
#   §2.2.5  no heavy IO on the login node   -> this runs under sbatch
#   §2.2.6  no ghost processes              -> trap kills the transfer tree
#   §2.2.7  use the lowest adequate tier    -> stampede, no GPU requested
#   §7.1    /home-mscluster is capped 50 GB -> destination must be /datasets
#   §7.4    /datasets/<user> holds 50 GB+   -> the default destination
#   §4      report progress periodically    -> size checkpoint every 30 min
#
# ALTERNATIVE, and what §8 actually prescribes for datasets this size: the
# dedicated sideload host, which has its own NIC and does not consume a node.
#   ssh <user>@10.100.14.250
#   cd /datasets/<user>/brats/raw && <repo>/scripts/data_preprocessing/download_brats2021.py --dest ./brats2021
# Use that if this job cannot reach the Aspera server (see NETWORK below).
#
# NETWORK: fasp is UDP/33001 outbound (plus TCP/33001 for the session). If the
# worker nodes have no outbound UDP route, the transfer will hang at 0 bytes --
# that is not a bug in this script; fall back to the sideload host above.

#SBATCH --job-name=brats-dl
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=72:00:00
#SBATCH --output=logs/data_preprocessing_preprocessing/download_%j.out
#SBATCH --error=logs/data_preprocessing_preprocessing/download_%j.err
#SBATCH --open-mode=append
#SBATCH --requeue

set -uo pipefail

MONITOR_INTERVAL="${MONITOR_INTERVAL:-1800}"   # seconds between size checkpoints

stamp() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }

REPO="${FL_BRATS_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"

# Destination: the first argument, else FL_BRATS_DATA_ROOT/raw/brats2021.
# Anything left over is forwarded verbatim to
# download_brats2021.py, e.g.
#   sbatch scripts/data_preprocessing/brats_download.sbatch --target-rate-kbps 500000
DEST=""
if [[ $# -gt 0 && "$1" != -* ]]; then DEST="$1"; shift; fi

# --- preflight -------------------------------------------------------------

[[ -f "$REPO/scripts/data_preprocessing/download_brats2021.py" ]] || {
  stamp "ERROR: $REPO is not the fl-brats repo. Submit from the repo root, or"
  stamp "       set FL_BRATS_REPO=/path/to/fl-brats."
  exit 2
}

# Toolchain activation, mirroring install_aspera.sh. FASPEX_URL is not read
# here -- the Python script picks it up from .env on its own.
if [[ -f "$REPO/.env" ]]; then
  set -a; . "$REPO/.env"; set +a
fi

# Relative paths in .env resolve against PROJECT_DIR, as they do for the installer.
BASE="${PROJECT_DIR:-$REPO}"
[[ -d "$BASE" ]] || BASE="$REPO"
: "${FL_BRATS_DATA_ROOT:=/datasets/$USER/brats}"
: "${DEST:=$FL_BRATS_DATA_ROOT/raw/brats2021}"
[[ "$DEST" = /* ]] || DEST="$BASE/$DEST"

# /home-mscluster is capped at 50 GB (§7.1); the package is ~142 GB.
case "$DEST" in
  /home-mscluster/*|"$HOME"/*)
    stamp "ERROR: $DEST is inside your home directory, which is capped at 50 GB."
    stamp "       The package is ~142 GB -- use /datasets/$USER/... (§7.1, §7.4)."
    exit 2 ;;
esac

mkdir -p "$DEST" || { stamp "ERROR: cannot create $DEST"; exit 2; }

stamp "running install_aspera.sh"
"$REPO/scripts/environment/install_aspera.sh" || {
  stamp "ERROR: Aspera toolchain installation failed"
  exit 2
}

TOOLCHAIN_DIR="${TOOLCHAIN_DIR:-.toolchain}"
[[ "$TOOLCHAIN_DIR" = /* ]] || TOOLCHAIN_DIR="$BASE/$TOOLCHAIN_DIR"   # same base as the installer
export GEM_HOME="$TOOLCHAIN_DIR/gems" GEM_PATH="$TOOLCHAIN_DIR/gems"
[[ -x "$TOOLCHAIN_DIR/rubyenv/bin/ruby" ]] && PATH="$TOOLCHAIN_DIR/rubyenv/bin:$PATH"
[[ -d "$GEM_HOME/bin" ]] && PATH="$GEM_HOME/bin:$PATH"
export PATH

command -v ascli >/dev/null 2>&1 || {
  stamp "ERROR: ascli not found after install_aspera.sh completed"
  exit 2
}

stamp "job          : ${SLURM_JOB_ID:-none} on $(hostname) [${SLURM_JOB_PARTITION:-none}]"
stamp "repo         : $REPO"
stamp "destination  : $DEST"
stamp "free space   : $(df -h "$DEST" | awk 'NR==2 {print $4}') (package is ~142 GB)"
stamp "ascli        : $(command -v ascli) ($(ascli --version 2>/dev/null))"
stamp "detail log   : $DEST/download.log"

# --- run -------------------------------------------------------------------

# ascli prints no progress bar when stdout is not a tty, so the transfer's own
# output is quiet. Keep it in its own file and checkpoint the on-disk size into
# the SLURM log instead -- that is the progress signal (§4).
RAW="$DEST/transfer-${SLURM_JOB_ID:-manual}.out"

# setsid puts python + ascli + ascp in their own process group. That way a
# signal aimed at this script does not race them, and cleanup can take the whole
# group down in one call -- ascp is a grandchild and outlives its parent
# otherwise, which is exactly the ghost process §2.2.6 warns about.
setsid python3 "$REPO/scripts/data_preprocessing/download_brats2021.py" \
  --dest "$DEST" \
  --verify-workers "${SLURM_CPUS_PER_TASK:-4}" \
  "$@" \
  >"$RAW" 2>&1 &
DL_PID=$!

pgid_of() { ps -o pgid= -p "$1" 2>/dev/null | tr -d ' '; }
sleep 1
DL_PGID="$(pgid_of "$DL_PID")"
# Never group-kill our own group: that would take this script with it.
[[ -n "$DL_PGID" && "$DL_PGID" != "$(pgid_of $$)" ]] || DL_PGID=""
# Only worth reporting if it is still running -- an already-finished transfer
# has no process group to find, and that is not a problem.
if [[ -z "$DL_PGID" ]] && kill -0 "$DL_PID" 2>/dev/null; then
  stamp "note: transfer is not in its own process group; falling back to tree kill"
fi

monitor() {
  while sleep "$MONITOR_INTERVAL"; do
    stamp "progress: $(du -sh "$DEST" 2>/dev/null | cut -f1) on disk"
  done
}
monitor & MON_PID=$!

# §2.2.6: leave nothing behind. python -> ascli -> ascp is three levels deep, so
# walk the tree rather than pkill'ing by name -- a name match would also hit a
# co-scheduled job of yours on the same node.
kill_tree() {
  local pid=$1 sig=$2 child
  for child in $(pgrep -P "$pid" 2>/dev/null); do kill_tree "$child" "$sig"; done
  kill "-$sig" "$pid" 2>/dev/null
}

cleanup() {
  # Signal first, report second, and never let a write failure interrupt the
  # kill: if stdout has already gone away, an untrapped SIGPIPE mid-cleanup is
  # precisely how the transfer gets orphaned.
  trap '' PIPE
  # Tree, not just the subshell: its `sleep` child would otherwise outlive the
  # job holding the output file open.
  kill_tree "$MON_PID" TERM
  # Group first: this reaps ascli/ascp even when python is already gone.
  if [[ -n "$DL_PGID" ]] && kill -0 -- "-$DL_PGID" 2>/dev/null; then
    kill -TERM -- "-$DL_PGID" 2>/dev/null
    stamp "terminating transfer process group $DL_PGID"
    sleep 10
    kill -KILL -- "-$DL_PGID" 2>/dev/null
  elif kill -0 "$DL_PID" 2>/dev/null; then
    kill_tree "$DL_PID" TERM
    stamp "terminating transfer (pid $DL_PID) and its descendants"
    sleep 10
    kill_tree "$DL_PID" KILL
  fi
}
trap cleanup EXIT
# PIPE and HUP matter as much as TERM here: if whatever is reading our stdout
# goes away, an untrapped SIGPIPE kills this script *without* running the EXIT
# trap, and the transfer is orphaned.
trap 'stamp "caught signal -- transfer is resumable, requeue or resubmit"; exit 143' TERM INT HUP PIPE

wait "$DL_PID"; RC=$?
kill_tree "$MON_PID" TERM   # tree: a plain kill here would orphan its `sleep`

stamp "final size   : $(du -sh "$DEST" 2>/dev/null | cut -f1)"
tail -n 30 "$RAW"

if [[ $RC -eq 0 ]]; then
  stamp "PASS -- download verified and validated. Full log: $DEST/download.log"
else
  stamp "FAIL (exit $RC) -- the partial download is intact and resumable."
  stamp "Resubmit the identical command to continue from where it stopped."
fi
exit $RC
