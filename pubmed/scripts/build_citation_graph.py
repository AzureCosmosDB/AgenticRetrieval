"""
Build a citation graph from PMC Open Access XML articles (JATS format).

Produces two mappings:
  - **cited_by**: pmcid → list of pmcids that cite it  (reverse index)
  - **cites**:    pmcid → list of pmcids it references  (forward index)

Results are saved to a JSON file and can optionally be patched into existing
Cosmos DB documents (adds ``cited_by``, ``cited_by_count``, ``cites``,
``cites_count`` fields).

Usage examples::

    # Build graph and save to JSON
    python build_citation_graph.py --input downloads/temp --output citation_maps.json

    # Build + patch Cosmos DB (requires config.yaml with Cosmos settings)
    python build_citation_graph.py --input downloads/temp --output citation_maps.json \\
        --patch-cosmos --config config.pubmed.yaml --cosmos-container articles
"""

import argparse
import asyncio
import csv
import glob
import gzip
import json
import logging
import os
import sys
import urllib.request
from collections import defaultdict
from multiprocessing import Pool as MPool
from pathlib import Path
from typing import Optional
from xml.etree import ElementTree as ET

_REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from utils.xml_helpers import text as _text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PMID ↔ PMCID mapping (from PMC-ids.csv)
# ---------------------------------------------------------------------------

PMC_IDS_URL = "https://ftp.ncbi.nlm.nih.gov/pub/pmc/PMC-ids.csv.gz"


def ensure_pmc_ids_csv(data_dir: str) -> str:
    """Download PMC-ids.csv.gz and decompress if not already present. Return path to .csv."""
    csv_path = os.path.join(data_dir, "PMC-ids.csv")
    gz_path = csv_path + ".gz"

    if os.path.isfile(csv_path):
        log.info(f"PMC-ids.csv already exists at {csv_path}")
        return csv_path

    if not os.path.isfile(gz_path):
        os.makedirs(os.path.dirname(gz_path), exist_ok=True)
        log.info(f"Downloading {PMC_IDS_URL} ...")
        urllib.request.urlretrieve(PMC_IDS_URL, gz_path)
        log.info(f"Downloaded to {gz_path}")

    log.info(f"Decompressing {gz_path} ...")
    with gzip.open(gz_path, "rb") as f_in, open(csv_path, "wb") as f_out:
        while chunk := f_in.read(1 << 20):
            f_out.write(chunk)
    log.info(f"Decompressed to {csv_path}")
    return csv_path


def load_pmid_to_pmcid(csv_path: str) -> dict[str, str]:
    """Load PMC-ids.csv and return a pmid → pmcid dict."""
    pmid_to_pmcid: dict[str, str] = {}
    log.info(f"Loading PMID→PMCID mappings from {csv_path} ...")
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, quotechar='"')
        for row in reader:
            pmcid = (row.get("PMCID") or "").strip()
            pmid = (row.get("PMID") or "").strip()
            if pmcid and pmid:
                pmid_to_pmcid[pmid] = pmcid
    log.info(f"  Loaded {len(pmid_to_pmcid)} PMID→PMCID mappings")
    return pmid_to_pmcid


# ---------------------------------------------------------------------------
# Citation extraction (per XML file)
# ---------------------------------------------------------------------------

# Module-level lookup table, set before multiprocessing pool starts so forked
# workers inherit it via copy-on-write (no pickling overhead).
_pmid_to_pmcid: dict[str, str] = {}


def _extract_cited_pmcids(xml_path: str) -> tuple[str, list[str]]:
    """Quick parse: return (this_pmcid, [pmcids referenced by this article])."""
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError:
        return ("", [])
    root = tree.getroot()
    pmcid = root.findtext(".//article-id[@pub-id-type='pmc']", "") or Path(xml_path).stem
    back = root.find("back")
    if back is None:
        return (pmcid, [])
    cited = []
    for ref in back.findall(".//ref"):
        cite = ref.find("element-citation")
        if cite is None:
            cite = ref.find("mixed-citation")
        if cite is None:
            continue
        ref_pmcid = _text(cite, ".//pub-id[@pub-id-type='pmc']")
        if ref_pmcid:
            if not ref_pmcid.startswith("PMC"):
                ref_pmcid = "PMC" + ref_pmcid
            cited.append(ref_pmcid)
            continue
        ref_pmid = _text(cite, ".//pub-id[@pub-id-type='pmid']")
        if ref_pmid:
            resolved = _pmid_to_pmcid.get(ref_pmid)
            if resolved:
                cited.append(resolved)
    return (pmcid, cited)


def _init_worker_pmid_to_pmcid(mapping: dict[str, str]) -> None:
    """Initializer for worker processes to set the PMID→PMCID mapping."""
    global _pmid_to_pmcid
    _pmid_to_pmcid = mapping


def build_citation_maps(
    xml_files: list[str],
    pmid_to_pmcid: dict[str, str],
    workers: int = 16,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Build cited_by and cites mappings from a list of XML files."""
    cited_by: dict[str, list[str]] = defaultdict(list)
    cites: dict[str, list[str]] = {}

    log.info(f"Building citation graph from {len(xml_files)} files ({workers} workers)...")
    with MPool(
        processes=workers,
        initializer=_init_worker_pmid_to_pmcid,
        initargs=(pmid_to_pmcid,),
    ) as pool:
        for i, (src_pmcid, ref_pmcids) in enumerate(
            pool.imap_unordered(_extract_cited_pmcids, xml_files, chunksize=256)
        ):
            if src_pmcid and ref_pmcids:
                cites[src_pmcid] = sorted(set(ref_pmcids))
            for ref_pmcid in ref_pmcids:
                cited_by[ref_pmcid].append(src_pmcid)
            if (i + 1) % 100_000 == 0:
                log.info(f"  citation scan: {i+1}/{len(xml_files)}")

    for k in cited_by:
        cited_by[k] = sorted(set(cited_by[k]))
    log.info(f"Citation graph complete: {len(cited_by)} cited-by, {len(cites)} cites")
    return dict(cited_by), cites


# ---------------------------------------------------------------------------
# Cosmos DB patching
# ---------------------------------------------------------------------------

async def patch_cosmos_documents(
    cited_by_map: dict[str, list[str]],
    cites_map: dict[str, list[str]],
    config_path: str,
    db_name: Optional[str] = None,
    container_name: str = "articles",
) -> None:
    """Patch existing Cosmos DB documents with citation graph data."""
    import cosmos_db_upload as cdb

    cdb.load_config(Path(config_path))
    db_name = db_name or cdb.DATABASE_NAME or "pubmed"

    use_rbac = cdb.CONFIG.get("cosmos", {}).get("use_rbac_auth", False)
    credential = None
    if use_rbac:
        from azure.identity.aio import DefaultAzureCredential as AsyncDefaultAzureCredential
        credential = AsyncDefaultAzureCredential()
    cosmos_client = cdb.get_cosmos_client(use_rbac_auth=use_rbac, credential=credential)
    database = cosmos_client.get_database_client(db_name)
    container = database.get_container_client(container_name)

    all_pmcids = set(cited_by_map.keys()) | set(cites_map.keys())
    log.info(f"Patching {len(all_pmcids)} documents with citation data...")

    max_concurrent_patches = 20
    semaphore = asyncio.Semaphore(max_concurrent_patches)
    counter_lock = asyncio.Lock()
    success = 0
    failed = 0

    async def _patch_one(pmcid: str) -> None:
        nonlocal success, failed
        async with semaphore:
            patch_ops = [
                {"op": "set", "path": "/cited_by", "value": cited_by_map.get(pmcid, [])},
                {"op": "set", "path": "/cited_by_count", "value": len(cited_by_map.get(pmcid, []))},
                {"op": "set", "path": "/cites", "value": cites_map.get(pmcid, [])},
                {"op": "set", "path": "/cites_count", "value": len(cites_map.get(pmcid, []))},
            ]
            try:
                await container.patch_item(
                    item=pmcid, partition_key=pmcid, patch_operations=patch_ops,
                )
                async with counter_lock:
                    success += 1
                    total = success + failed
                    if total % 10_000 == 0:
                        log.info(f"  patched {success} ok, {failed} failed so far")
            except Exception as e:
                log.debug(f"Patch failed for {pmcid}: {e}")
                async with counter_lock:
                    failed += 1
                    total = success + failed
                    if total % 10_000 == 0:
                        log.info(f"  patched {success} ok, {failed} failed so far")

    tasks = [asyncio.create_task(_patch_one(pmcid)) for pmcid in all_pmcids]
    if tasks:
        await asyncio.gather(*tasks)

    log.info(f"Patching done: {success} succeeded, {failed} failed")

    await cosmos_client.close()
    if credential is not None:
        await credential.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build PMC citation graph and optionally patch Cosmos DB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", required=True,
                        help="Root directory containing extracted XML files")
    parser.add_argument("--output", default=None,
                        help="Path for citation_maps.json (default: <input>/citation_maps.json)")
    parser.add_argument("--workers", type=int, default=16,
                        help="Parallel workers for citation scanning (default: 16)")
    parser.add_argument("--patch-cosmos", action="store_true",
                        help="Patch existing Cosmos DB documents with cited_by / cites")
    parser.add_argument("--config", default=None,
                        help="Config YAML (required when --patch-cosmos)")
    parser.add_argument("--cosmos-db", default=None,
                        help="Cosmos DB database name override")
    parser.add_argument("--cosmos-container", default="articles",
                        help="Cosmos DB container name (default: articles)")
    return parser.parse_args()


async def main_async():
    args = parse_args()

    output_path = args.output or os.path.join(args.input, "citation_maps.json")

    # Discover XML files
    xml_files = sorted(glob.glob(
        os.path.join(args.input, "**", "*.xml"), recursive=True,
    ))
    log.info(f"Found {len(xml_files)} XML files under '{args.input}'")

    if not xml_files:
        log.error("No XML files found. Check --input path.")
        sys.exit(1)

    # Download PMID→PMCID mapping
    pmc_csv = ensure_pmc_ids_csv(args.input)
    pmid_to_pmcid = load_pmid_to_pmcid(pmc_csv)

    # Build citation maps
    cited_by_map, cites_map = build_citation_maps(
        xml_files, pmid_to_pmcid, workers=args.workers,
    )

    # Save
    log.info(f"Saving citation maps to {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"cited_by": cited_by_map, "cites": cites_map}, f)
    log.info(f"Saved: {len(cited_by_map)} cited-by, {len(cites_map)} cites entries")

    # Optionally patch Cosmos DB
    if args.patch_cosmos:
        if not args.config:
            log.error("--config is required when --patch-cosmos is set")
            sys.exit(1)
        await patch_cosmos_documents(
            cited_by_map, cites_map,
            config_path=args.config,
            db_name=args.cosmos_db,
            container_name=args.cosmos_container,
        )


if __name__ == "__main__":
    asyncio.run(main_async())
