#!/usr/bin/env bash
# Installs the IBM Aspera CLI (ascli) plus the ascp transfer binary.
#
# This is the command-line equivalent of the "IBM-Aspera-Connect plugin" the
# TCIA download button asks for. Connect is a browser plugin; ascp is the same
# fasp transfer engine without the browser.
#
# NO SUDO REQUIRED. ascp itself is just an unzipped binary; the only real
# dependency is Ruby >= 3.1, which we get without root (module / conda /
# micromamba). Everything lands in TOOLCHAIN_DIR and can be deleted wholesale.
#
# Usage: scripts/setup/install_aspera.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

log()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mwarn:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31mError:\033[0m %s\n' "$*" >&2; exit 1; }

have_sudo() { command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; }

# .env is optional here: installing Ruby needs TOOLCHAIN_DIR and
# RUBY_INSTALL_METHOD at most, never FASPEX_URL.
if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env"
  set +a
fi

# A PROJECT_DIR that does not exist on this machine (e.g. a .env shared between
# your laptop and the cluster) is a warning, not a failure.
if [[ -n "${PROJECT_DIR:-}" && ! -d "$PROJECT_DIR" ]]; then
  warn "PROJECT_DIR=$PROJECT_DIR does not exist here; using $REPO_ROOT"
  PROJECT_DIR=""
fi
PROJECT_DIR="$(cd "${PROJECT_DIR:-$REPO_ROOT}" && pwd)"
resolve_dir() { [[ "$1" = /* ]] && printf '%s' "$1" || printf '%s' "$PROJECT_DIR/$1"; }

TOOLCHAIN_DIR="$(resolve_dir "${TOOLCHAIN_DIR:-.toolchain}")"
RUBY_INSTALL_METHOD="${RUBY_INSTALL_METHOD:-auto}"

# Keep gems inside the toolchain dir so nothing lands in a quota'd ~/.gem and
# the whole install stays relocatable/deletable as one directory.
export GEM_HOME="$TOOLCHAIN_DIR/gems"
export GEM_PATH="$GEM_HOME"

MAMBA_ROOT="$TOOLCHAIN_DIR/micromamba"
RUBY_ENV="$TOOLCHAIN_DIR/rubyenv"

# Put whatever has been installed so far on PATH. brats_download.sbatch carries
# its own copy of this -- keep the two in step if you change it.
activate_toolchain() {
  [[ -x "$RUBY_ENV/bin/ruby" ]] && PATH="$RUBY_ENV/bin:$PATH"
  if [[ -s "$HOME/.rvm/scripts/rvm" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/.rvm/scripts/rvm" >/dev/null 2>&1 || true
  fi
  [[ -d "$GEM_HOME/bin" ]] && PATH="$GEM_HOME/bin:$PATH"
  export PATH
}

MIN_RUBY=3.1

version_ge() { [[ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -n1)" == "$2" ]]; }

ruby_ok() {
  command -v ruby >/dev/null 2>&1 || return 1
  local v
  v="$(ruby -e 'print RUBY_VERSION' 2>/dev/null)" || return 1
  version_ge "$v" "$MIN_RUBY"
}

# Each try_* returns 0 on success. They are deliberately quiet on failure so
# `auto` can walk down the list without scaring anyone with stack traces.

try_system() {
  activate_toolchain
  ruby_ok
}

try_module() {
  # HPC environment modules. `module` is usually a shell function, so it only
  # exists after sourcing the init script -- check both.
  if ! type module >/dev/null 2>&1; then
    for init in /etc/profile.d/modules.sh /etc/profile.d/lmod.sh \
                /usr/share/lmod/lmod/init/bash; do
      # shellcheck disable=SC1090
      [[ -r "$init" ]] && source "$init" && break
    done
  fi
  type module >/dev/null 2>&1 || return 1
  log "Trying environment modules"
  set +e
  module load ruby >/dev/null 2>&1 || module load Ruby >/dev/null 2>&1
  set -e
  ruby_ok
}

try_conda() {
  local conda_bin=""
  for c in micromamba mamba conda; do
    command -v "$c" >/dev/null 2>&1 && { conda_bin="$c"; break; }
  done
  [[ -n "$conda_bin" ]] || return 1
  log "Installing Ruby via existing $conda_bin into $RUBY_ENV"
  "$conda_bin" create -y -p "$RUBY_ENV" -c conda-forge "ruby>=$MIN_RUBY" >/dev/null || return 1
  activate_toolchain
  ruby_ok
}

try_micromamba() {
  # Universal no-sudo fallback: micromamba is a single static binary, and
  # conda-forge ships prebuilt Ruby -- no compiler, no modules, no root.
  local arch
  case "$(uname -m)" in
    x86_64)          arch=linux-64 ;;
    aarch64|arm64)   arch=linux-aarch64 ;;
    ppc64le)         arch=linux-ppc64le ;;
    *) warn "unsupported arch $(uname -m) for micromamba"; return 1 ;;
  esac

  log "Bootstrapping micromamba into $MAMBA_ROOT"
  mkdir -p "$MAMBA_ROOT"
  local mm="$MAMBA_ROOT/micromamba"

  # Preferred: the raw binary, no extraction at all.
  if ! curl -fLs \
      "https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-$arch" \
      -o "$mm"; then
    # Fallback: the .tar.bz2 from the anaconda.org API. Extract with Python
    # rather than tar -- bzip2 is genuinely absent on many minimal images,
    # and Python's stdlib handles bz2 without any external decompressor.
    warn "raw binary unavailable, falling back to the tarball"
    curl -Ls "https://micro.mamba.pm/api/micromamba/$arch/latest" -o "$MAMBA_ROOT/mm.tar.bz2" || return 1
    python3 - "$MAMBA_ROOT/mm.tar.bz2" "$mm" <<'PY' || return 1
import sys, tarfile, shutil
src, dest = sys.argv[1], sys.argv[2]
with tarfile.open(src, "r:bz2") as t:
    member = next(m for m in t.getmembers() if m.name.endswith("bin/micromamba"))
    with t.extractfile(member) as fh, open(dest, "wb") as out:
        shutil.copyfileobj(fh, out)
PY
    rm -f "$MAMBA_ROOT/mm.tar.bz2"
  fi

  chmod +x "$mm" 2>/dev/null || true
  [[ -s "$mm" ]] && "$mm" --version >/dev/null 2>&1 || {
    warn "micromamba binary is not runnable"; return 1; }

  log "Installing Ruby from conda-forge into $RUBY_ENV"
  MAMBA_ROOT_PREFIX="$MAMBA_ROOT" "$mm" create -y -p "$RUBY_ENV" \
    -c conda-forge "ruby>=$MIN_RUBY" || return 1
  activate_toolchain
  ruby_ok
}

try_apt() {
  have_sudo || [[ -t 0 ]] || return 1
  log "Installing Ruby via apt (requires sudo)"
  sudo apt-get update -y && \
  sudo apt-get install -y ruby-full build-essential libssl-dev zlib1g-dev unzip curl || return 1
  activate_toolchain
  ruby_ok
}

try_rvm() {
  # Opt-in only (RUBY_INSTALL_METHOD=rvm), never reached by `auto`: it compiles
  # Ruby from source, needs a toolchain a locked-down cluster often lacks, and
  # rewrites your shell rc files. micromamba does the same job in seconds.
  log "Installing Ruby via RVM (compiles from source, this takes a while)"
  local gpgbin
  gpgbin="$(command -v gpg2 || command -v gpg)" || return 1
  "$gpgbin" --keyserver hkp://keyserver.ubuntu.com --recv-keys \
    409B6B1796C275462A1703113804BB82D39DC0E3 \
    7D2BAF1CF37B13E2069D6956105BD0E739499BDB || return 1
  # --ignore-dotfiles keeps it out of .bashrc/.profile/.zshrc.
  curl -sSL https://get.rvm.io | bash -s stable --ignore-dotfiles || return 1
  # RVM's scripts are not `set -u`/`set -e` clean, and it refuses to run with a
  # custom GEM_HOME, so drop both for the duration.
  local saved_gem_home="$GEM_HOME"
  set +eu
  unset GEM_HOME GEM_PATH
  # shellcheck disable=SC1091
  source "$HOME/.rvm/scripts/rvm"
  rvm install 3.3.4 && rvm use 3.3.4 --default
  local rc=$?
  set -eu
  export GEM_HOME="$saved_gem_home" GEM_PATH="$saved_gem_home"
  [[ $rc -eq 0 ]] || return 1
  ruby_ok
}

# --- 1. Ruby ---------------------------------------------------------------
mkdir -p "$TOOLCHAIN_DIR"

if try_system; then
  log "Ruby $(ruby -e 'print RUBY_VERSION') already available"
else
  case "$RUBY_INSTALL_METHOD" in
    system)     die "No usable Ruby >= $MIN_RUBY on PATH and RUBY_INSTALL_METHOD=system" ;;
    module)     try_module     || die "module load ruby did not yield Ruby >= $MIN_RUBY" ;;
    conda)      try_conda      || die "conda/mamba install failed" ;;
    micromamba) try_micromamba || die "micromamba bootstrap failed" ;;
    apt)        try_apt        || die "apt install failed" ;;
    rvm)        try_rvm        || die "RVM install failed" ;;
    auto)
      # RVM is deliberately absent here -- it is opt-in only, see try_rvm.
      ok=0
      for m in try_module try_conda try_micromamba try_apt; do
        # apt is pointless without sudo; skip it rather than prompt on a cluster.
        [[ "$m" == try_apt ]] && ! have_sudo && continue
        if "$m"; then ok=1; break; fi
        warn "${m#try_} did not work, trying next"
      done
      [[ "$ok" == 1 ]] || die "Could not obtain Ruby >= $MIN_RUBY by any method.
Set RUBY_INSTALL_METHOD in .env explicitly, or load a Ruby module yourself."
      ;;
    *) die "Unknown RUBY_INSTALL_METHOD '$RUBY_INSTALL_METHOD'" ;;
  esac
fi

activate_toolchain
ruby_ok || die "Ruby >= $MIN_RUBY still not on PATH"
log "Using ruby $(ruby -e 'print RUBY_VERSION') at $(command -v ruby)"

# --- 2. aspera-cli gem -----------------------------------------------------
mkdir -p "$GEM_HOME"
if command -v ascli >/dev/null 2>&1; then
  log "ascli already installed"
else
  log "Installing the aspera-cli gem into $GEM_HOME"
  # --install-dir/--bindir explicitly: some Ruby builds (notably conda-forge's)
  # patch their gem paths so GEM_HOME alone is not honoured, and the binstub
  # silently lands somewhere that is not on PATH.
  gem install aspera-cli --no-document \
    --install-dir "$GEM_HOME" --bindir "$GEM_HOME/bin"
  activate_toolchain
  command -v ascli >/dev/null 2>&1 || die "ascli not on PATH after gem install"
fi

# --- 3. ascp transfer binary ----------------------------------------------
# The gem is only the API client; the actual fasp transfer needs ascp. Newer
# aspera-cli calls this "transferd", older releases "ascp". Neither needs root.
if ascli config ascp info >/dev/null 2>&1; then
  log "Aspera transfer binary already installed"
else
  log "Installing the Aspera transfer SDK (ascp)"
  ascli config transferd install 2>/dev/null ||
    ascli config ascp install ||
    die "Could not install the transfer SDK. Try manually:
  curl -Lso $TOOLCHAIN_DIR/sdk.zip https://ibm.biz/aspera_transfer_sdk
  ascli config ascp install --sdk-url=file://$TOOLCHAIN_DIR/sdk.zip"
fi

log "Done."
echo
echo "  ruby  : $(command -v ruby)"
echo "  ascli : $(command -v ascli)  ($(ascli --version 2>/dev/null))"
echo
echo "Next: sbatch scripts/data/brats_download.sbatch      (on the cluster)"
echo "  or: scripts/data/download_brats2021.py --dest DIR  (with $GEM_HOME/bin on PATH)"
