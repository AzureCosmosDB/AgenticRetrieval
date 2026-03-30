"""
PMC Commercial Open Access Bulk Downloader
==========================================
Downloads all baseline and incremental XML packages from the PMC OA Commercial
subset (CC0, CC BY, CC BY-SA, CC BY-ND licensed articles).

Source directory: https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_comm/xml/

Usage
-----
    python download.py [--outdir DIR] [--workers N] [--type xml|txt] [--no-incr] [--no-baseline]

Options
-------
    --outdir      Where to save downloaded files (default: ./downloads)
    --workers     Parallel download workers (default: 4)
    --type        File type to download: xml or txt (default: xml)
    --no-incr     Skip incremental update packages
    --no-baseline Skip baseline packages
    --dry-run     Print files that would be downloaded without downloading

Requirements
------------
    Python 3.12+
    pip install requests tqdm
"""

import argparse
import logging
import re
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

if sys.version_info < (3, 12):
    raise RuntimeError("Python 3.12 or later is required")

import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL = "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_comm/"
CHUNK_SIZE = 1024 * 1024  # 1 MB read chunks
RETRY_ATTEMPTS = 5
RETRY_BACKOFF = 10  # seconds between retries (doubles each attempt)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def list_remote_files(url: str, pattern: re.Pattern) -> list[str]:
    """Return all hrefs from an NCBI FTP index page that match *pattern*."""
    log.info("Fetching directory listing: %s", url)
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return [
        m.group(0)
        for m in pattern.finditer(resp.text)
    ]


def file_is_complete(local_path: Path, expected_size: Optional[int]) -> bool:
    """Return True if the local file exists and matches the expected byte size."""
    if not local_path.exists():
        return False
    if expected_size is not None and local_path.stat().st_size != expected_size:
        return False
    return True


def remote_file_size(url: str) -> Optional[int]:
    """Return Content-Length for *url* via a HEAD request, or None if unavailable."""
    try:
        resp = requests.head(url, timeout=30, allow_redirects=True)
        resp.raise_for_status()
        length = resp.headers.get("Content-Length")
        return int(length) if length else None
    except Exception:
        return None


def download_file(url: str, dest: Path, desc: str = "") -> Path:
    """
    Download *url* to *dest*, resuming if a partial file exists.
    Shows a tqdm progress bar.  Retries on transient errors.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    expected_size = remote_file_size(url)

    # Skip if already fully downloaded
    if file_is_complete(dest, expected_size):
        log.info("SKIP (already complete): %s", dest.name)
        return dest

    attempt = 0
    backoff = RETRY_BACKOFF

    while attempt < RETRY_ATTEMPTS:
        attempt += 1
        partial = dest.with_suffix(dest.suffix + ".part")

        # Resume from partial download if present
        resume_pos = partial.stat().st_size if partial.exists() else 0
        headers = {"Range": f"bytes={resume_pos}-"} if resume_pos > 0 else {}

        try:
            with requests.get(url, headers=headers, stream=True, timeout=120) as resp:
                # 416 → server can't satisfy range, restart from 0
                if resp.status_code == 416:
                    resume_pos = 0
                    partial.unlink(missing_ok=True)
                    resp = requests.get(url, stream=True, timeout=120)
                    resp.raise_for_status()
                else:
                    resp.raise_for_status()

                total = expected_size or int(resp.headers.get("Content-Length", 0))
                mode = "ab" if resume_pos > 0 else "wb"

                with (
                    open(partial, mode) as fh,
                    tqdm(
                        total=total,
                        initial=resume_pos,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        desc=desc or dest.name,
                        leave=False,
                        ncols=100,
                    ) as bar,
                ):
                    for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                        fh.write(chunk)
                        bar.update(len(chunk))

            # Rename to final name once complete
            partial.replace(dest)
            log.info("DONE: %s", dest.name)
            return dest

        except (requests.RequestException, OSError) as exc:
            log.warning(
                "Attempt %d/%d failed for %s: %s",
                attempt, RETRY_ATTEMPTS, dest.name, exc,
            )
            if attempt < RETRY_ATTEMPTS:
                log.info("Retrying in %d s…", backoff)
                time.sleep(backoff)
                backoff = min(backoff * 2, 300)
            else:
                log.error("Giving up on %s after %d attempts", dest.name, RETRY_ATTEMPTS)
                raise


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_packages(file_type: str, include_baseline: bool, include_incr: bool) -> list[str]:
    """
    Scrape the NCBI FTP index and return a sorted list of .tar.gz filenames
    (baselines first by PMCID range, then incrementals by date).
    """
    index_url = urljoin(BASE_URL, f"{file_type}/")

    # Match any .tar.gz file (not filelist metadata)
    pattern = re.compile(
        rf'oa_comm_{file_type}\.[^"]+\.tar\.gz'
    )
    found = list_remote_files(index_url, pattern)

    # Deduplicate (HTML may list links multiple times)
    found = sorted(set(found))

    baseline_files = [f for f in found if ".baseline." in f]
    incr_files = [f for f in found if ".incr." in f]

    result: list[str] = []
    if include_baseline:
        # Sort baselines by PMC range embedded in name
        result.extend(sorted(baseline_files))
        log.info("Found %d baseline packages", len(baseline_files))
    if include_incr:
        result.extend(sorted(incr_files))
        log.info("Found %d incremental packages", len(incr_files))

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Bulk-download PMC OA Commercial subset packages.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--outdir",
        default="downloads",
        help="Directory to save downloaded files.",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel download threads.",
    )
    p.add_argument(
        "--type",
        choices=["xml", "txt"],
        default="xml",
        dest="file_type",
        help="File content type: xml or txt.",
    )
    p.add_argument(
        "--no-baseline",
        action="store_true",
        help="Skip baseline packages.",
    )
    p.add_argument(
        "--no-incr",
        action="store_true",
        help="Skip incremental update packages.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be downloaded; do not download.",
    )
    p.add_argument(
        "--uncompress",
        action="store_true",
        help="Extract downloaded .tar.gz files after download.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    log.info("Output directory: %s", outdir)

    include_baseline = not args.no_baseline
    include_incr = not args.no_incr

    if not include_baseline and not include_incr:
        log.error("Both --no-baseline and --no-incr specified. Nothing to do.")
        sys.exit(1)

    packages = discover_packages(args.file_type, include_baseline, include_incr)

    if not packages:
        log.warning("No packages found. Check the directory URL or your network connection.")
        sys.exit(0)

    index_url = urljoin(BASE_URL, f"{args.file_type}/")
    download_urls = [(urljoin(index_url, fname), outdir / fname) for fname in packages]

    if args.dry_run:
        print(f"\nDry run — {len(download_urls)} package(s) would be downloaded:\n")
        total_estimated = 0
        for url, dest in download_urls:
            size = remote_file_size(url)
            size_str = f"{size / 1e9:.1f} GB" if size else "unknown"
            status = "EXISTS" if dest.exists() else "MISSING"
            print(f"  [{status}]  {dest.name}  ({size_str})")
            if size:
                total_estimated += size
        if total_estimated:
            print(f"\nEstimated total: {total_estimated / 1e12:.2f} TB")
        return

    log.info("Starting download of %d package(s) with %d worker(s)…", len(download_urls), args.workers)

    failed: list[str] = []

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(download_file, url, dest, dest.name): dest.name
            for url, dest in download_urls
        }
        with tqdm(total=len(futures), desc="Overall progress", unit="pkg", ncols=100) as overall:
            for future in as_completed(futures):
                name = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    log.error("FAILED: %s — %s", name, exc)
                    failed.append(name)
                finally:
                    overall.update(1)

    if failed:
        log.error("\n%d package(s) failed to download:", len(failed))
        for f in failed:
            log.error("  %s", f)
        sys.exit(1)
    else:
        log.info("All packages downloaded successfully to %s", outdir)

    if args.uncompress:
        to_extract = [
            dest for _url, dest in download_urls
            if dest.name not in failed and dest.exists()
        ]
        log.info("Extracting %d .tar.gz file(s) with %d worker(s)…", len(to_extract), args.workers)

        def _extract(archive: Path) -> str:
            extract_dir = archive.parent / archive.name.replace(".tar.gz", "")
            with tarfile.open(archive, "r:gz") as tar:
                tar.extractall(path=extract_dir, filter="data")
            return archive.name

        extract_failed: list[str] = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(_extract, dest): dest.name
                for dest in to_extract
            }
            with tqdm(total=len(futures), desc="Extracting", unit="pkg", ncols=100) as bar:
                for future in as_completed(futures):
                    name = futures[future]
                    try:
                        future.result()
                        log.info("EXTRACTED: %s", name)
                    except (tarfile.TarError, OSError) as exc:
                        log.error("EXTRACT FAILED: %s — %s", name, exc)
                        extract_failed.append(name)
                    finally:
                        bar.update(1)

        if extract_failed:
            log.error("%d package(s) failed to extract:", len(extract_failed))
            for f in extract_failed:
                log.error("  %s", f)
            sys.exit(1)
        else:
            log.info("All packages extracted successfully.")


if __name__ == "__main__":
    main()
