#!/usr/bin/env python3
"""Download and verify the RSNA-ASNR-MICCAI BraTS 2021 dataset from TCIA.

The dataset is published as a single IBM Aspera Faspex package. TCIA's Faspex
server rejects partial (per-path) selections with a 500, so the whole ~142 GB
package is transferred; the 1,251 labelled NIfTI training subjects are the
BraTS2021_TrainingSet/ subtree of it.

The transfer itself is done by `ascli` (the aspera-cli gem), which speaks fasp
and handles resume. This script wraps it with the operational bits ascli does
not provide: credential hygiene, retry/backoff, disk gating, MD5 verification
against the package's own .sums manifest, and dataset validation.

Run scripts/install_aspera.sh first to get ascli + ascp. See README.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

# --- package facts, verified against the live Faspex listing ----------------

PACKAGE_NAME = "RSNA-ASNR-MICCAI-BraTS-2021"
SUMS_NAME = f"{PACKAGE_NAME}.sums"
TRAINING_SUBDIR = "BraTS2021_TrainingSet"

# Every training subject must have exactly these five NIfTI files.
MODALITIES = ("t1", "t1ce", "t2", "flair", "seg")
EXPECTED_SUBJECTS = 1251

PACKAGE_SIZE_GB = 142
# fasp writes into the final path directly, so no double-space is needed, but
# leave headroom for the filesystem and the .sums/state files.
DEFAULT_HEADROOM_GB = 10

SUBJECT_RE = re.compile(r"^BraTS2021_\d{5}$")

log = logging.getLogger("brats")


# --- credential handling ----------------------------------------------------

# The Faspex package URL embeds a base64 `context` blob containing the package
# passcode. Anything derived from it is a credential: it must not reach argv
# (world-readable via /proc/*/cmdline) nor the log file.
_SECRET_RE = re.compile(r"(context=|passcode\"?\s*[:=]\s*\"?)([A-Za-z0-9+/=_-]{8,})")


def redact(text: str) -> str:
    """Blank out Faspex passcodes/context blobs in anything bound for a log."""
    return _SECRET_RE.sub(r"\1<redacted>", text)


class RedactingFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return redact(super().format(record))


def _validated(url: str, source: str) -> str:
    """Reject blank or obviously wrong values before they reach ascli."""
    url = url.strip()
    if not url:
        raise SystemExit(f"Faspex URL from {source} is empty.")
    if not url.startswith("http"):
        raise SystemExit(
            f"Faspex URL from {source} does not look like a URL: {url[:40]!r}"
        )
    if "context=" not in url:
        log.warning("URL from %s has no 'context=' parameter; ascli may reject it.",
                    source)
    return url


def load_faspex_url(args: argparse.Namespace, project_dir: Path) -> str:
    """Resolve the Faspex URL from the first source that supplies one.

    Order: --url-file, $BRATS_FASPEX_URL_FILE, $BRATS_FASPEX_URL, .env.
    A file is preferred: an env var is readable through /proc/<pid>/environ for
    the whole lifetime of the process tree.
    """
    if args.url_file:
        return _validated(Path(args.url_file).expanduser().read_text(), args.url_file)

    env_file = os.environ.get("BRATS_FASPEX_URL_FILE")
    if env_file:
        return _validated(Path(env_file).expanduser().read_text(),
                          "$BRATS_FASPEX_URL_FILE")

    env_url = os.environ.get("BRATS_FASPEX_URL")
    if env_url:
        return _validated(env_url, "$BRATS_FASPEX_URL")

    dotenv = project_dir / ".env"
    if dotenv.is_file():
        for line in dotenv.read_text().splitlines():
            line = line.strip()
            if line.startswith("FASPEX_URL="):
                return _validated(line.split("=", 1)[1].strip().strip("\"'"),
                                  str(dotenv))

    raise SystemExit(
        "No Faspex URL found. Provide one of:\n"
        "  --url-file PATH            (file containing the URL, mode 0600)\n"
        "  BRATS_FASPEX_URL_FILE=PATH\n"
        "  BRATS_FASPEX_URL=...\n"
        "  FASPEX_URL=... in .env\n"
        "Get the link from the 'Download (142gb)' button at\n"
        "https://www.cancerimagingarchive.net/analysis-result/rsna-asnr-miccai-brats-2021/"
    )


# --- logging ----------------------------------------------------------------


def setup_logging(logfile: Path, verbose: bool) -> None:
    logfile.parent.mkdir(parents=True, exist_ok=True)
    log.setLevel(logging.DEBUG)

    fh = logging.FileHandler(logfile, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(RedactingFormatter("%(asctime)s %(levelname)-7s %(message)s"))
    log.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.DEBUG if verbose else logging.INFO)
    sh.setFormatter(RedactingFormatter("%(message)s"))
    log.addHandler(sh)


def human(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PiB"


# --- preflight --------------------------------------------------------------


def check_disk_space(dest: Path, required_gb: float) -> None:
    probe = dest
    while not probe.exists():
        probe = probe.parent
    usage = shutil.disk_usage(probe)
    free_gb = usage.free / 1024**3
    log.info(
        "Disk: %s free of %s on %s",
        human(usage.free),
        human(usage.total),
        probe,
    )
    if free_gb < required_gb:
        raise SystemExit(
            f"Not enough free space: need ~{required_gb:.0f} GiB, have "
            f"{free_gb:.0f} GiB on {probe}. Point --dest at a bigger filesystem "
            f"or free space, then re-run."
        )


def require_ascli() -> str:
    exe = shutil.which("ascli")
    if not exe:
        raise SystemExit(
            "ascli not found on PATH. Run scripts/install_aspera.sh, or activate "
            "an existing install (see README.md)."
        )
    return exe


# --- transfer ---------------------------------------------------------------


def run_ascli_receive(
    ascli: str,
    url: str,
    dest: Path,
    resume_policy: str,
    target_rate_kbps: str | None,
) -> int:
    """One transfer attempt. Returns the ascli exit code.

    The URL goes to ascli through a 0600 file referenced as `@file:`, never as
    an argv value, so the passcode never appears in `ps` output.
    """
    ts: dict[str, object] = {"resume_policy": resume_policy}
    if target_rate_kbps:
        ts["target_rate_kbps"] = int(target_rate_kbps)

    with tempfile.TemporaryDirectory(prefix="brats-cred-") as tmp:
        url_path = Path(tmp) / "faspex_url"
        # Create 0600 before writing so the value is never briefly world-readable.
        fd = os.open(url_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as fh:
            fh.write(url)

        cmd = [
            ascli,
            "faspex5",
            "packages",
            "receive",
            f"--url=@file:{url_path}",
            f"--to-folder={dest}",
            f"--ts=@json:{json.dumps(ts)}",
        ]
        log.debug("exec: %s", " ".join(cmd))

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        last_progress = 0.0
        for raw in proc.stdout:
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            # ascli redraws a progress bar; echo it live but only checkpoint it
            # to the log every 30s so download.log stays readable.
            is_progress = ("\r" in raw) or ("%" in line and "ETA" in line)
            if is_progress:
                print(line, end="\r", flush=True)
                now = time.monotonic()
                if now - last_progress > 30:
                    last_progress = now
                    log.debug("progress: %s", line.strip())
            else:
                log.info("%s", line)
        return proc.wait()


def download_with_retries(
    ascli: str,
    url: str,
    dest: Path,
    max_retries: int,
    resume_policy: str,
    target_rate_kbps: str | None,
) -> None:
    """Retry the transfer with exponential backoff.

    Each retry re-invokes ascli, and fasp's resume policy means already-complete
    files are skipped and partial files continue from their existing offset.
    """
    delay = 15.0
    for attempt in range(1, max_retries + 1):
        log.info("Transfer attempt %d/%d", attempt, max_retries)
        started = time.time()
        rc = run_ascli_receive(ascli, url, dest, resume_policy, target_rate_kbps)
        elapsed = time.time() - started
        if rc == 0:
            log.info("Transfer finished in %.1f min", elapsed / 60)
            return
        log.warning("Attempt %d failed (exit %d) after %.1f min", attempt, rc, elapsed / 60)
        if attempt == max_retries:
            raise SystemExit(
                f"Transfer still failing after {max_retries} attempts. The partial "
                f"download in {dest} is intact -- re-run this script to resume."
            )
        # Exponential backoff, capped, with jitter to avoid lockstep retries.
        jitter = (os.getpid() % 7) + 1
        sleep_for = min(delay, 600) + jitter
        log.info("Backing off %.0fs before retry", sleep_for)
        time.sleep(sleep_for)
        delay *= 2


# --- checksum verification --------------------------------------------------


def find_package_root(dest: Path) -> Path:
    """Locate the directory holding <PACKAGE_NAME>.sums.

    ascli may nest the payload under a 'PKG - <name>' folder depending on its
    package-folder setting, so find it rather than assuming a fixed depth.
    """
    direct = dest / SUMS_NAME
    if direct.is_file():
        return dest
    for candidate in sorted(dest.glob(f"*/{SUMS_NAME}")):
        return candidate.parent
    for candidate in sorted(dest.rglob(SUMS_NAME)):
        return candidate.parent
    raise SystemExit(
        f"{SUMS_NAME} not found under {dest}. The download is incomplete -- "
        f"re-run to resume."
    )


def parse_sums(sums_path: Path, scope: str) -> dict[str, str]:
    """Parse the .sums manifest into {relative_path: md5}.

    Format is `<md5> <path>` with a single space; paths are relative to the
    directory containing the .sums file.
    """
    wanted_prefix = f"{PACKAGE_NAME}/{TRAINING_SUBDIR}/" if scope == "training" else ""
    entries: dict[str, str] = {}
    with sums_path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            digest, sep, relpath = line.partition(" ")
            if not sep:
                continue
            relpath = relpath.lstrip("*").strip()
            if wanted_prefix and not relpath.startswith(wanted_prefix):
                continue
            entries[relpath] = digest.lower()
    return entries


def md5_of(path: Path, chunk: int = 4 * 1024 * 1024) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


@dataclass
class VerifyResult:
    ok: list[str] = field(default_factory=list)
    bad: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    bytes_ok: int = 0


def verify_checksums(
    package_root: Path,
    entries: dict[str, str],
    state_path: Path,
    workers: int,
) -> VerifyResult:
    """MD5 every file in `entries`, skipping ones unchanged since last verified.

    The cache keys on (size, mtime_ns) so a re-run after a resumed download only
    re-hashes files that actually changed.
    """
    state: dict[str, list] = {}
    if state_path.is_file():
        try:
            state = json.loads(state_path.read_text())
        except (json.JSONDecodeError, OSError):
            log.warning("Verification cache unreadable, re-hashing everything")

    result = VerifyResult()
    new_state: dict[str, list] = {}
    total = len(entries)
    done = 0
    t0 = time.time()

    def check(item: tuple[str, str]) -> tuple[str, str, int]:
        relpath, expected = item
        path = package_root / relpath
        try:
            st = path.stat()
        except FileNotFoundError:
            return relpath, "missing", 0
        cached = state.get(relpath)
        if cached and cached[0] == st.st_size and cached[1] == st.st_mtime_ns:
            return relpath, "ok-cached", st.st_size
        return (
            relpath,
            "ok" if md5_of(path) == expected else "bad",
            st.st_size,
        )

    log.info("Verifying %d files against %s (%d workers)", total, SUMS_NAME, workers)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for relpath, status, size in pool.map(check, entries.items()):
            done += 1
            if status in ("ok", "ok-cached"):
                result.ok.append(relpath)
                result.bytes_ok += size
                st = (package_root / relpath).stat()
                new_state[relpath] = [st.st_size, st.st_mtime_ns]
            elif status == "bad":
                result.bad.append(relpath)
                log.error("CHECKSUM MISMATCH: %s", relpath)
            else:
                result.missing.append(relpath)
            if done % 2000 == 0 or done == total:
                rate = done / max(time.time() - t0, 1e-6)
                print(
                    f"  verified {done}/{total} ({100*done/total:.1f}%) "
                    f"{rate:.0f} files/s",
                    end="\r",
                    flush=True,
                )
    print()
    state_path.write_text(json.dumps(new_state))
    log.info(
        "Checksums: %d ok, %d mismatched, %d missing",
        len(result.ok),
        len(result.bad),
        len(result.missing),
    )
    return result


# --- dataset validation -----------------------------------------------------


@dataclass
class Subject:
    subject_id: str
    collection: str
    path: Path
    present: set[str] = field(default_factory=set)

    @property
    def missing(self) -> list[str]:
        return [m for m in MODALITIES if m not in self.present]

    @property
    def complete(self) -> bool:
        return not self.missing


def scan_subjects(training_root: Path) -> list[Subject]:
    """Walk BraTS2021_TrainingSet/<collection>/<subject>/ and record modalities."""
    subjects: list[Subject] = []
    if not training_root.is_dir():
        return subjects
    for collection_dir in sorted(p for p in training_root.iterdir() if p.is_dir()):
        for subject_dir in sorted(p for p in collection_dir.iterdir() if p.is_dir()):
            if not SUBJECT_RE.match(subject_dir.name):
                continue
            subj = Subject(subject_dir.name, collection_dir.name, subject_dir)
            for f in subject_dir.iterdir():
                name = f.name
                if not name.endswith(".nii.gz") or f.stat().st_size == 0:
                    continue
                suffix = name[: -len(".nii.gz")].rsplit("_", 1)[-1].lower()
                if suffix in MODALITIES:
                    subj.present.add(suffix)
            subjects.append(subj)
    return subjects


def directory_size(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.stat(os.path.join(root, name)).st_size
            except OSError:
                pass
    return total


def report(subjects: list[Subject], training_root: Path, max_listed: int) -> bool:
    complete = [s for s in subjects if s.complete]
    incomplete = [s for s in subjects if not s.complete]
    total_bytes = directory_size(training_root)

    log.info("")
    log.info("=" * 68)
    log.info("BraTS 2021 training set validation")
    log.info("=" * 68)
    log.info("Location            : %s", training_root)
    log.info("Subjects found      : %d (expected %d)", len(subjects), EXPECTED_SUBJECTS)
    log.info("Complete subjects   : %d", len(complete))
    log.info("Incomplete subjects : %d", len(incomplete))
    log.info("Total size on disk  : %s", human(total_bytes))

    if incomplete:
        log.info("")
        log.info("Incomplete subjects and missing files:")
        for subj in incomplete[:max_listed]:
            log.info(
                "  %-18s [%s] missing: %s",
                subj.subject_id,
                subj.collection,
                ", ".join(f"{subj.subject_id}_{m}.nii.gz" for m in subj.missing),
            )
        if len(incomplete) > max_listed:
            log.info("  ... and %d more (see %s)", len(incomplete) - max_listed, "download.log")

    by_collection: dict[str, int] = {}
    for subj in subjects:
        by_collection[subj.collection] = by_collection.get(subj.collection, 0) + 1
    log.info("")
    log.info("Subjects per source collection:")
    for name, count in sorted(by_collection.items()):
        log.info("  %-28s %4d", name, count)
    log.info("=" * 68)

    return len(subjects) == EXPECTED_SUBJECTS and not incomplete


# --- main -------------------------------------------------------------------


def main() -> int:
    project_dir = Path(__file__).resolve().parent.parent

    ap = argparse.ArgumentParser(
        description="Download and verify the BraTS 2021 dataset from TCIA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--dest", type=Path, default=Path("/data/brats2021"),
                    help="Destination directory")
    ap.add_argument("--url-file", help="File containing the Faspex package URL (mode 0600)")
    ap.add_argument("--max-retries", type=int, default=8)
    ap.add_argument("--resume-policy", default="sparse_csum",
                    choices=["none", "attrs", "sparse_csum", "full_csum"])
    ap.add_argument("--target-rate-kbps", default=os.environ.get("ASPERA_TARGET_RATE_KBPS", ""),
                    help="Bandwidth cap in kbps; blank for unthrottled")
    ap.add_argument("--verify-scope", default="training", choices=["training", "all"],
                    help="Which files to MD5: the training NIfTI set, or all 407k files")
    ap.add_argument("--verify-workers", type=int, default=min(8, (os.cpu_count() or 4)))
    ap.add_argument("--min-free-gb", type=float,
                    default=PACKAGE_SIZE_GB + DEFAULT_HEADROOM_GB)
    ap.add_argument("--skip-verify", action="store_true", help="Skip MD5 verification")
    ap.add_argument("--validate-only", action="store_true",
                    help="Do not download; verify and validate what is already on disk")
    ap.add_argument("--max-listed", type=int, default=50,
                    help="Cap on incomplete subjects printed to the console")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    dest: Path = args.dest.expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)
    setup_logging(dest / "download.log", args.verbose)

    log.info("")
    log.info("BraTS 2021 download -- destination %s", dest)

    signal.signal(signal.SIGINT, lambda *_: sys.exit(
        "\nInterrupted. Re-run the same command to resume."))

    if not args.validate_only:
        ascli = require_ascli()
        url = load_faspex_url(args, project_dir)
        check_disk_space(dest, args.min_free_gb)
        log.info("Package is ~%d GB; transfer resumes automatically if interrupted.",
                 PACKAGE_SIZE_GB)
        download_with_retries(
            ascli, url, dest, args.max_retries, args.resume_policy,
            args.target_rate_kbps or None,
        )

    package_root = find_package_root(dest)
    training_root = package_root / PACKAGE_NAME / TRAINING_SUBDIR
    log.info("Package root: %s", package_root)

    checksums_ok = True
    if not args.skip_verify:
        entries = parse_sums(package_root / SUMS_NAME, args.verify_scope)
        if not entries:
            log.warning("No checksum entries matched scope %r", args.verify_scope)
        else:
            res = verify_checksums(
                package_root, entries, dest / ".verify-state.json", args.verify_workers
            )
            checksums_ok = not res.bad and not res.missing
            if res.missing:
                log.warning("%d files listed in %s are absent -- re-run to resume.",
                            len(res.missing), SUMS_NAME)
            if res.bad:
                log.error("%d files failed MD5. Delete them and re-run to re-fetch.",
                          len(res.bad))

    subjects = scan_subjects(training_root)
    structure_ok = report(subjects, training_root, args.max_listed)

    if checksums_ok and structure_ok:
        log.info("VALIDATION PASSED -- %d complete subjects.", len(subjects))
        return 0
    log.error("VALIDATION FAILED -- re-run this script to resume/repair.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
