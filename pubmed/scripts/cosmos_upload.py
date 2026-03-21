"""
Convert PMC Open Access XML articles (JATS format) to JSON for Azure Cosmos DB.

Supports:
  - Attribute/field search  (structured metadata fields)
  - Full-text search        (Cosmos DB integrated vectorization / Azure AI Search)
  - Vector search           (OpenAI text-embedding-3-small → DiskANN index)

Usage examples:
  # Dry-run: parse + print JSON, no embedding or upload
  python cosmos_upload.py --dry-run --limit 3

  # Write to a JSONL file (no Cosmos upload)
  python cosmos_upload.py --output articles.jsonl

  # Full pipeline: embed + upsert to Cosmos DB
  python cosmos_upload.py \
      --cosmos-endpoint https://<account>.documents.azure.com:443/ \
      --cosmos-key <key> \
      --openai-key <key>
"""

import os
import sys
import csv
import json
import glob
import gzip
import time
import logging
import argparse
import urllib.request
from pathlib import Path
from xml.etree import ElementTree as ET
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

XLINK_NS = "http://www.w3.org/1999/xlink"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMS = 1536
# Cosmos DiskANN vector index requires dimensions to be declared at container creation.


# ---------------------------------------------------------------------------
# XML parsing helpers
# ---------------------------------------------------------------------------

def _text(el, path: str, default: str = "") -> str:
    """Find a child element by path and return all inner text concatenated."""
    if el is None:
        return default
    node = el.find(path)
    if node is None:
        return default
    return "".join(node.itertext()).strip()


def _all_text(el, path: str) -> list[str]:
    """Return text of every element matching path."""
    if el is None:
        return []
    return ["".join(n.itertext()).strip() for n in el.findall(path) if "".join(n.itertext()).strip()]


def _attr(el, path: str, attr: str, ns_map: dict | None = None, default: str = "") -> str:
    """Find element by path and return one of its attributes."""
    if el is None:
        return default
    node = el.find(path)
    if node is None:
        return default
    if ns_map:
        # Build Clark notation key, e.g. {http://www.w3.org/1999/xlink}href
        for prefix, uri in ns_map.items():
            attr = attr.replace(f"{prefix}:", f"{{{uri}}}")
    return node.get(attr, default)


def _parse_date(el) -> Optional[str]:
    """Convert a JATS <date> or <pub-date> element to an ISO date string."""
    if el is None:
        return None
    day   = _text(el, "day")   or "01"
    month = _text(el, "month") or "01"
    year  = _text(el, "year")
    if year:
        return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
    return None


# ---------------------------------------------------------------------------
# Body / full-text extraction
# ---------------------------------------------------------------------------

def _extract_body(body_el) -> tuple[str, list[dict]]:
    """
    Walk <body> sections and return:
      - full_text: all paragraph text joined as one string
      - sections:  list of {"title": ..., "text": ...}
    """
    if body_el is None:
        return "", []

    sections = []
    all_parts = []

    for sec in body_el.findall(".//sec"):
        title_el = sec.find("title")
        title = "".join(title_el.itertext()).strip() if title_el is not None else ""

        para_texts = []
        # Only direct <p> children of this sec (not nested secs) to avoid duplication
        for p in sec.findall("p"):
            t = "".join(p.itertext()).strip()
            if t:
                para_texts.append(t)

        if para_texts:
            sec_text = " ".join(para_texts)
            sections.append({"title": title, "text": sec_text})
            all_parts.append(sec_text)

    return " ".join(all_parts), sections


# ---------------------------------------------------------------------------
# Reference extraction
# ---------------------------------------------------------------------------

def _parse_references(back_el) -> list[dict]:
    """Extract structured references from <back><ref-list><ref>."""
    if back_el is None:
        return []

    refs = []
    for ref in back_el.findall(".//ref"):
        ref_id = ref.get("id", "")

        # JATS uses <element-citation> or <mixed-citation>
        cite = ref.find("element-citation")
        if cite is None:
            cite = ref.find("mixed-citation")
        if cite is None:
            # Fallback: grab whatever text is in the <ref>
            raw = "".join(ref.itertext()).strip()
            if raw:
                refs.append({"ref_id": ref_id, "raw": raw})
            continue

        pub_type = cite.get("publication-type", "")

        # Authors
        authors = []
        for name_el in cite.findall(".//person-group/name"):
            surname    = _text(name_el, "surname")
            given      = _text(name_el, "given-names")
            if surname:
                authors.append(f"{surname} {given}".strip())
        has_etal = cite.find(".//person-group/etal") is not None

        # Identifiers
        doi  = _text(cite, ".//pub-id[@pub-id-type='doi']")
        pmid = _text(cite, ".//pub-id[@pub-id-type='pmid']")
        pmcid = _text(cite, ".//pub-id[@pub-id-type='pmc']")

        # Core fields
        article_title = _text(cite, "article-title")
        source        = _text(cite, "source")        # journal / book title
        year          = _text(cite, "year")
        volume        = _text(cite, "volume")
        fpage         = _text(cite, "fpage")
        lpage         = _text(cite, "lpage")

        entry = {
            "ref_id":         ref_id,
            "publication_type": pub_type,
            "article_title":  article_title,
            "source":         source,
            "year":           year,
            "volume":         volume,
            "fpage":          fpage,
            "lpage":          lpage,
            "authors":        authors,
            "has_etal":       has_etal,
            "doi":            doi,
            "pmid":           pmid,
            "pmcid":          pmcid,
        }
        refs.append(entry)

    return refs


# ---------------------------------------------------------------------------
# Main XML → dict converter
# ---------------------------------------------------------------------------

def parse_article(xml_path: str) -> Optional[dict]:
    """Parse a JATS XML file and return a Cosmos-DB-ready dict, or None on failure."""
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError as e:
        log.error(f"XML parse error {xml_path}: {e}")
        return None

    root = tree.getroot()
    front = root.find("front")
    if front is None:
        log.warning(f"No <front> element in {xml_path}")
        return None

    jmeta = front.find("journal-meta")
    ameta = front.find("article-meta")
    if ameta is None:
        log.warning(f"No <article-meta> element in {xml_path}")
        return None
    body  = root.find("body")
    back  = root.find("back")

    # ------------------------------------------------------------------
    # Identifiers
    # ------------------------------------------------------------------
    pmcid = _text(ameta, ".//article-id[@pub-id-type='pmc']")
    pmid  = _text(ameta, ".//article-id[@pub-id-type='pmid']")
    doi   = _text(ameta, ".//article-id[@pub-id-type='doi']")
    if not pmcid:
        pmcid = Path(xml_path).stem

    # ------------------------------------------------------------------
    # Journal metadata
    # ------------------------------------------------------------------
    journal = {}
    if jmeta is not None:
        journal = {
            "title":        _text(jmeta, ".//journal-title"),
            "nlm_ta":       _text(jmeta, ".//journal-id[@journal-id-type='nlm-ta']"),
            "iso_abbrev":   _text(jmeta, ".//journal-id[@journal-id-type='iso-abbrev']"),
            "publisher_id": _text(jmeta, ".//journal-id[@journal-id-type='publisher-id']"),
            "pmc_abbrev":   _text(jmeta, ".//journal-id[@journal-id-type='pmc']"),
            "issn_print":   _text(jmeta, ".//issn[@pub-type='ppub']"),
            "issn_epub":    _text(jmeta, ".//issn[@pub-type='epub']"),
            "publisher":    _text(jmeta, ".//publisher-name"),
            "publisher_loc":_text(jmeta, ".//publisher-loc"),
        }

    # ------------------------------------------------------------------
    # Title
    # ------------------------------------------------------------------
    title = _text(ameta, ".//title-group/article-title")
    running_head = _text(ameta, ".//title-group/alt-title[@alt-title-type='running-head']")

    # ------------------------------------------------------------------
    # Authors & affiliations
    # ------------------------------------------------------------------
    # Build affiliation lookup from <aff id="...">
    affiliations: dict[str, dict] = {}
    for aff in ameta.findall(".//aff"):
        aff_id = aff.get("id", "")
        affiliations[aff_id] = {
            "institution": _text(aff, "institution"),
            "addr_line":   _text(aff, "addr-line"),
            "country":     _text(aff, "country"),
        }

    authors = []
    for contrib in ameta.findall(".//contrib[@contrib-type='author']"):
        name_el = contrib.find("name")
        if name_el is None:
            continue
        aff_refs = [xr.get("rid", "") for xr in contrib.findall("xref[@ref-type='aff']")]
        author = {
            "surname":      _text(name_el, "surname"),
            "given_names":  _text(name_el, "given-names"),
            "corresponding": contrib.get("corresp") == "yes",
            "equal_contrib": contrib.get("equal-contrib") == "yes",
            "deceased":      contrib.get("deceased") == "yes",
            "affiliations": [affiliations[r] for r in aff_refs if r in affiliations],
        }
        email_el = contrib.find("email")
        if email_el is not None:
            author["email"] = "".join(email_el.itertext()).strip()
        authors.append(author)

    # Flat surname list makes attribute-equality queries trivial:
    #   SELECT * FROM c WHERE ARRAY_CONTAINS(c.author_surnames, "Smith")
    author_surnames = [a["surname"] for a in authors]

    # ------------------------------------------------------------------
    # Publication dates
    # ------------------------------------------------------------------
    pub_date_print   = _parse_date(ameta.find(".//pub-date[@pub-type='ppub']"))
    pub_date_epub    = _parse_date(ameta.find(".//pub-date[@pub-type='epub']"))
    pub_date_release = _parse_date(ameta.find(".//pub-date[@pub-type='pmc-release']"))

    # Best available publication year (used as partition key)
    pub_year = "unknown"
    for pd in ameta.findall(".//pub-date"):
        yr = _text(pd, "year")
        if yr:
            pub_year = int(yr)
            break

    date_received = _parse_date(ameta.find(".//history/date[@date-type='received']"))
    date_accepted = _parse_date(ameta.find(".//history/date[@date-type='accepted']"))

    # ------------------------------------------------------------------
    # Categories & keywords
    # ------------------------------------------------------------------
    heading   = _text(ameta, ".//subj-group[@subj-group-type='heading']/subject")
    subjects  = _all_text(ameta, ".//subj-group[@subj-group-type='Discipline']/subject")
    organisms = _all_text(ameta, ".//subj-group[@subj-group-type='System Taxonomy']/subject")
    keywords  = _all_text(ameta, ".//kwd-group/kwd")

    # ------------------------------------------------------------------
    # Abstract
    # ------------------------------------------------------------------
    abstract_el = ameta.find("abstract")
    abstract = "".join(abstract_el.itertext()).strip() if abstract_el is not None else ""

    # Table-of-contents teaser (shorter blurb used in some journals)
    toc_abstract_el = ameta.find("abstract[@abstract-type='toc']")
    toc_abstract = "".join(toc_abstract_el.itertext()).strip() if toc_abstract_el is not None else ""

    # ------------------------------------------------------------------
    # Volume / issue / pages
    # ------------------------------------------------------------------
    volume     = _text(ameta, "volume")
    issue      = _text(ameta, "issue")
    fpage      = _text(ameta, "fpage")
    lpage      = _text(ameta, "lpage")
    elocation  = _text(ameta, "elocation-id")

    # ------------------------------------------------------------------
    # License
    # ------------------------------------------------------------------
    license_el  = ameta.find(".//license")
    license_url = ""
    if license_el is not None:
        license_url = license_el.get(f"{{{XLINK_NS}}}href", "")

    # Normalise license into a short tag for easy filtering
    license_tag = ""
    if "creativecommons.org/licenses/by/" in license_url:
        license_tag = "CC-BY"
    elif "creativecommons.org/licenses/by-nc/" in license_url:
        license_tag = "CC-BY-NC"
    elif "creativecommons.org/publicdomain/zero" in license_url:
        license_tag = "CC0"

    # ------------------------------------------------------------------
    # Body text & sections
    # ------------------------------------------------------------------
    full_text, sections = _extract_body(body)

    # ------------------------------------------------------------------
    # References (full structured data)
    # ------------------------------------------------------------------
    references = _parse_references(back)
    ref_count  = len(references)

    # ------------------------------------------------------------------
    # Figure / table counts (useful filters)
    # ------------------------------------------------------------------
    fig_count   = len(body.findall(".//fig"))   if body is not None else 0
    table_count = len(body.findall(".//table-wrap")) if body is not None else 0

    # ------------------------------------------------------------------
    # Supplementary material
    # ------------------------------------------------------------------
    supp_labels = _all_text(body if body is not None else front,
                            ".//supplementary-material/label")

    # ------------------------------------------------------------------
    # Build document
    # ------------------------------------------------------------------
    doc = {
        # --- Cosmos DB required ---
        "id":              pmcid,           # unique document ID
        # --- Primary identifiers ---
        "pmcid":           pmcid,
        "pmid":            pmid,
        "doi":             doi,
        "article_type":    root.get("article-type", ""),
        "open_access":     True,            # all oa_comm articles are OA

        # --- Journal ---
        "journal":         journal,
        "journal_title":   journal.get("title", ""),    # top-level copy for easy querying

        # --- Content ---
        "title":           title,
        "running_head":    running_head,
        "abstract":        abstract,
        "toc_abstract":    toc_abstract,
        "full_text":       full_text,       # all body paragraphs joined
        "sections":        sections,        # [{"title": ..., "text": ...}]

        # --- Authors ---
        "authors":         authors,
        "author_surnames": author_surnames, # flat list for ARRAY_CONTAINS queries

        # --- Dates ---
        "pub_date_print":   pub_date_print,
        "pub_date_epub":    pub_date_epub,
        "pub_date_release": pub_date_release,
        "pub_year":         pub_year,       # partition key
        "date_received":    date_received,
        "date_accepted":    date_accepted,

        # --- Classification ---
        "heading":         heading,
        "subjects":        subjects,
        "organisms":       organisms,
        "keywords":        keywords,

        # --- Issue info ---
        "volume":          volume,
        "issue":           issue,
        "fpage":           fpage,
        "lpage":           lpage,
        "elocation_id":    elocation,

        # --- License ---
        "license_url":     license_url,
        "license_tag":     license_tag,     # "CC-BY" | "CC-BY-NC" | "CC0" | ""

        # --- References ---
        "references":     references,       # [{ref_id, article_title, source, year, authors, doi, pmid, ...}]
        "ref_count":       ref_count,

        # --- Stats ---
        "fig_count":       fig_count,
        "table_count":     table_count,
        "supp_labels":     supp_labels,

        # --- Vector (populated later) ---
        "embedding":        None,
        "embedding_source": None,
        "embedding_model":  None,
    }

    return doc


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
# Citation graph ("cited_by" / "who cites me?")
# ---------------------------------------------------------------------------

# Module-level lookup table, set from main() before multiprocessing starts.
_pmid_to_pmcid: dict[str, str] = {}

def _extract_cited_pmcids(xml_path: str) -> tuple[str, list[str]]:
    """Quick parse: return (this_pmcid, [pmcids referenced by this article]).

    Resolves references that only have a PMID by looking up _pmid_to_pmcid.
    """
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
        # Try PMCID directly first
        ref_pmcid = _text(cite, ".//pub-id[@pub-id-type='pmc']")
        if ref_pmcid:
            if not ref_pmcid.startswith("PMC"):
                ref_pmcid = "PMC" + ref_pmcid
            cited.append(ref_pmcid)
            continue
        # Fall back: resolve PMID → PMCID
        ref_pmid = _text(cite, ".//pub-id[@pub-id-type='pmid']")
        if ref_pmid:
            resolved = _pmid_to_pmcid.get(ref_pmid)
            if resolved:
                cited.append(resolved)
    return (pmcid, cited)


def build_citation_maps(
    xml_files: list[str],
    pmid_to_pmcid: dict[str, str],
    workers: int = 16,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Build two mappings:
      - cited_by:  pmcid → list of pmcids that cite it
      - cites:     pmcid → list of pmcids it references

    On Linux (fork), the pmid_to_pmcid dict is shared with workers via
    copy-on-write by setting the module-level global before pool creation.
    No pickling overhead.
    """
    from collections import defaultdict
    from multiprocessing import Pool as MPool

    # Set the module-level lookup so forked workers inherit it (COW, no pickle).
    global _pmid_to_pmcid
    _pmid_to_pmcid = pmid_to_pmcid

    cited_by: dict[str, list[str]] = defaultdict(list)
    cites: dict[str, list[str]] = {}

    log.info(f"Building citation graph from {len(xml_files)} files ({workers} workers)...")
    with MPool(processes=workers) as pool:
        for i, (src_pmcid, ref_pmcids) in enumerate(
            pool.imap_unordered(_extract_cited_pmcids, xml_files, chunksize=256)
        ):
            if src_pmcid and ref_pmcids:
                cites[src_pmcid] = sorted(set(ref_pmcids))
            for ref_pmcid in ref_pmcids:
                cited_by[ref_pmcid].append(src_pmcid)
            if (i + 1) % 100_000 == 0:
                log.info(f"  citation scan: {i+1}/{len(xml_files)}")

    # Sort each list for deterministic output
    for k in cited_by:
        cited_by[k] = sorted(set(cited_by[k]))
    log.info(f"Citation graph complete: {len(cited_by)} cited-by, {len(cites)} cites")
    return dict(cited_by), cites


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

PREFIX_LENGTH = int(8192 * 1.5)

def embed_document(doc: dict, client, model: str = EMBEDDING_MODEL) -> dict:
    """
    Generate a text embedding from title + abstract + full body text + journal title + TOC abstract.
    The text is truncated to PREFIX_LENGTH characters to stay within
    the model's 8,192-token limit.
    """
    text = f"Journal Title: {doc['journal_title']}\n\n{doc['title']}\n\nTOC Abstract: {doc['toc_abstract']}\n\n{doc['abstract']}\n\n{doc['full_text']}"
    text = text[:PREFIX_LENGTH]

    try:
        response = client.embeddings.create(input=text, model=model)
        doc["embedding"]        = response.data[0].embedding
        doc["embedding_source"] = "title+abstract+body+journal_title+toc_abstract"
        doc["embedding_model"]  = model
    except Exception as e:
        log.warning(f"Embedding failed for {doc['id']}: {e}")

    return doc


# ---------------------------------------------------------------------------
# Cosmos DB helpers
# ---------------------------------------------------------------------------

def create_cosmos_container(endpoint: str, key: str, db_name: str, container_name: str):
    """
    Create a Cosmos DB container with:
      - DiskANN vector index on /embedding (float32, cosine, 1536 dims)
      - Full-text indexes on /title, /abstract, /full_text
      - Range index on all other paths
      - Partition key: /pub_year
    """
    from azure.cosmos import CosmosClient, PartitionKey, exceptions

    client = CosmosClient(endpoint, credential=key)
    db = client.create_database_if_not_exists(db_name)

    indexing_policy = {
        "indexingMode": "consistent",
        "automatic": True,
        "includedPaths": [{"path": "/*"}],
        "excludedPaths": [
            {"path": "/full_text/?"},     # too large for range index
            {"path": "/embedding/*"},     # handled by vector index
            {"path": "/sections/*"},
        ],
        "fullTextIndexes": [
            {"path": "/title"},
            {"path": "/abstract"},
            {"path": "/full_text"},
        ],
        "vectorIndexes": [
            {"path": "/embedding", "type": "diskANN"},
        ],
    }

    vector_embedding_policy = {
        "vectorEmbeddings": [
            {
                "path":             "/embedding",
                "dataType":         "float32",
                "distanceFunction": "cosine",
                "dimensions":       EMBEDDING_DIMS,
            }
        ]
    }

    full_text_policy = {
        "defaultLanguage": "en-US",
        "fullTextPaths": [
            {"path": "/title",     "language": "en-US"},
            {"path": "/abstract",  "language": "en-US"},
            {"path": "/full_text", "language": "en-US"},
        ],
    }

    try:
        container = db.create_container_if_not_exists(
            id=container_name,
            partition_key=PartitionKey(path="/pub_year"),
            indexing_policy=indexing_policy,
            vector_embedding_policy=vector_embedding_policy,
            # full_text_policy requires SDK >= 4.7.0 and preview features enabled
            # Uncomment if your SDK supports it:
            # full_text_policy=full_text_policy,
        )
        log.info(f"Container '{container_name}' ready with vector + full-text indexes")
        return container
    except Exception as e:
        log.warning(f"Advanced container creation failed ({e}). Falling back to basic.")
        return db.create_container_if_not_exists(
            id=container_name,
            partition_key=PartitionKey(path="/pub_year"),
        )


def upsert_batch(docs: list[dict], container) -> tuple[int, int]:
    """Upsert a list of docs. Returns (success_count, fail_count)."""
    from azure.cosmos import exceptions

    ok = fail = 0
    for doc in docs:
        try:
            container.upsert_item(doc)
            ok += 1
        except exceptions.CosmosHttpResponseError as e:
            log.error(f"Upsert failed for {doc.get('id')}: {e}")
            fail += 1
    return ok, fail


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert PMC JATS XML → Cosmos DB JSON with embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input",    default="downloads/temp",
                        help="Root directory containing extracted XML files (searched recursively)")
    parser.add_argument("--output",   help="Write JSONL to this file (one doc per line)")
    parser.add_argument("--cosmos-endpoint", metavar="URL",  help="Cosmos DB endpoint")
    parser.add_argument("--cosmos-key",      metavar="KEY",  help="Cosmos DB primary key")
    parser.add_argument("--cosmos-db",       default="pubmed",   help="Database name (default: pubmed)")
    parser.add_argument("--cosmos-container",default="articles", help="Container name (default: articles)")
    parser.add_argument("--openai-key",      metavar="KEY",
                        help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--no-embed",  action="store_true", help="Skip embedding generation")
    parser.add_argument("--limit",     type=int,  help="Process only first N files (testing)")
    parser.add_argument("--dry-run",   action="store_true",
                        help="Parse & pretty-print first 3 docs only; no embed or upload")
    parser.add_argument("--citation-workers", type=int, default=16,
                        help="Parallel workers for building the citation graph (default: 16)")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Discover XML files
    # ------------------------------------------------------------------
    xml_files = sorted(glob.glob(
        os.path.join(args.input, "**", "*.xml"), recursive=True
    ))
    if args.limit:
        xml_files = xml_files[: args.limit]
    log.info(f"Found {len(xml_files)} XML files under '{args.input}'")

    # ------------------------------------------------------------------
    # Build citation graph (cited_by map) from ALL xml files
    # ------------------------------------------------------------------
    citation_cache = os.path.join(args.input, "citation_maps.json")
    if os.path.isfile(citation_cache):
        log.info(f"Loading cached citation maps from {citation_cache}")
        with open(citation_cache, "r", encoding="utf-8") as f:
            cached = json.load(f)
        cited_by_map = cached["cited_by"]
        cites_map = cached["cites"]
        log.info(f"Loaded {len(cited_by_map)} cited-by, {len(cites_map)} cites entries")
    else:
        pmc_csv = ensure_pmc_ids_csv(args.input)
        pmid_to_pmcid = load_pmid_to_pmcid(pmc_csv)
        all_xml_for_citations = sorted(glob.glob(
            os.path.join(args.input, "**", "*.xml"), recursive=True
        ))
        cited_by_map, cites_map = build_citation_maps(
            all_xml_for_citations, pmid_to_pmcid, workers=args.citation_workers
        )
        log.info(f"Saving citation maps to {citation_cache}")
        with open(citation_cache, "w", encoding="utf-8") as f:
            json.dump({"cited_by": cited_by_map, "cites": cites_map}, f)

    # ------------------------------------------------------------------
    # Initialise OpenAI client
    # ------------------------------------------------------------------
    oai_client = None
    if not args.no_embed and not args.dry_run:
        api_key = args.openai_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            log.error(
                "OpenAI API key required for embeddings. "
                "Pass --openai-key or set OPENAI_API_KEY, or use --no-embed to skip."
            )
            sys.exit(1)
        from openai import OpenAI
        oai_client = OpenAI(api_key=api_key)
        log.info("OpenAI client initialised")

    # ------------------------------------------------------------------
    # Initialise Cosmos container
    # ------------------------------------------------------------------
    container = None
    if args.cosmos_endpoint and args.cosmos_key and not args.dry_run:
        container = create_cosmos_container(
            args.cosmos_endpoint, args.cosmos_key,
            args.cosmos_db, args.cosmos_container,
        )

    # ------------------------------------------------------------------
    # Process files
    # ------------------------------------------------------------------
    out_file = open(args.output, "w", encoding="utf-8") if args.output else None
    success = failed = 0
    upsert_queue: list[dict] = []
    UPSERT_BATCH = 50

    for i, xml_path in enumerate(xml_files):
        log.info(f"[{i+1}/{len(xml_files)}] {Path(xml_path).name}")

        doc = parse_article(xml_path)
        if doc is None:
            failed += 1
            continue

        # --- Attach citation lists ---
        doc["cited_by"] = cited_by_map.get(doc["pmcid"], [])
        doc["cited_by_count"] = len(doc["cited_by"])
        doc["cites"] = cites_map.get(doc["pmcid"], [])
        doc["cites_count"] = len(doc["cites"])

        # --- Dry run: pretty-print first doc and exit ---
        if args.dry_run:
            print(json.dumps(
                {k: v for k, v in doc.items()
                 if k not in ("full_text", "sections", "embedding")},
                indent=2, ensure_ascii=False
            ))
            if i >= 2:   # show 3 docs then stop
                break
            continue

        # --- Embed ---
        if oai_client:
            doc = embed_document(doc, oai_client)
            # Respect ~3 000 RPM limit
            if (i + 1) % 50 == 0:
                time.sleep(1)

        # --- Write JSONL (one doc per line, compact JSON) ---
        if out_file:
            out_file.write(json.dumps(doc, ensure_ascii=False) + "\n")

        # --- Cosmos upsert (batched) ---
        if container:
            upsert_queue.append(doc)
            if len(upsert_queue) >= UPSERT_BATCH:
                ok, fail = upsert_batch(upsert_queue, container)
                success += ok
                failed  += fail
                upsert_queue.clear()
                log.info(f"  Upserted batch — total ok={success} fail={failed}")
        else:
            success += 1

    # Flush remaining
    if container and upsert_queue:
        ok, fail = upsert_batch(upsert_queue, container)
        success += ok
        failed  += fail

    if out_file:
        out_file.close()

    log.info(f"Finished. success={success}  failed={failed}")


if __name__ == "__main__":
    main()
 