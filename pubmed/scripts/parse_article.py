"""
JATS XML → dict converter for PMC Open Access articles.

Parses a single JATS/NLM XML file and returns a flat dictionary ready for
Cosmos DB ingestion (embeddings are *not* generated here).

Usage as a library::

    from pubmed.scripts.parse_article import parse_article
    doc = parse_article("path/to/article.xml")

Or standalone for quick inspection::

    python -m pubmed.scripts.parse_article path/to/article.xml
"""

import json
import logging
import sys
from pathlib import Path
from typing import Optional

# Ensure repo root is on sys.path so ``utils`` is importable regardless of
# how this module is loaded (direct execution, dynamic import, etc.).
_REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from xml.etree import ElementTree as ET

from utils.xml_helpers import (
    text as _text,
    all_text as _all_text,
    parse_date as _parse_date,
    XLINK_NS,
)

log = logging.getLogger(__name__)


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

        cite = ref.find("element-citation")
        if cite is None:
            cite = ref.find("mixed-citation")
        if cite is None:
            raw = "".join(ref.itertext()).strip()
            if raw:
                refs.append({"ref_id": ref_id, "raw": raw})
            continue

        pub_type = cite.get("publication-type", "")

        authors = []
        for name_el in cite.findall(".//person-group/name"):
            surname = _text(name_el, "surname")
            given = _text(name_el, "given-names")
            if surname:
                authors.append(f"{surname} {given}".strip())
        has_etal = cite.find(".//person-group/etal") is not None

        doi = _text(cite, ".//pub-id[@pub-id-type='doi']")
        pmid = _text(cite, ".//pub-id[@pub-id-type='pmid']")
        pmcid = _text(cite, ".//pub-id[@pub-id-type='pmc']")

        article_title = _text(cite, "article-title")
        source = _text(cite, "source")
        year = _text(cite, "year")
        volume = _text(cite, "volume")
        fpage = _text(cite, "fpage")
        lpage = _text(cite, "lpage")

        entry = {
            "ref_id": ref_id,
            "publication_type": pub_type,
            "article_title": article_title,
            "source": source,
            "year": year,
            "volume": volume,
            "fpage": fpage,
            "lpage": lpage,
            "authors": authors,
            "has_etal": has_etal,
            "doi": doi,
            "pmid": pmid,
            "pmcid": pmcid,
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
    body = root.find("body")
    back = root.find("back")

    # Identifiers
    pmcid = _text(ameta, ".//article-id[@pub-id-type='pmc']")
    pmid = _text(ameta, ".//article-id[@pub-id-type='pmid']")
    doi = _text(ameta, ".//article-id[@pub-id-type='doi']")
    if not pmcid:
        pmcid = Path(xml_path).stem

    # Journal metadata
    journal = {}
    if jmeta is not None:
        journal = {
            "title": _text(jmeta, ".//journal-title"),
            "nlm_ta": _text(jmeta, ".//journal-id[@journal-id-type='nlm-ta']"),
            "iso_abbrev": _text(jmeta, ".//journal-id[@journal-id-type='iso-abbrev']"),
            "publisher_id": _text(jmeta, ".//journal-id[@journal-id-type='publisher-id']"),
            "pmc_abbrev": _text(jmeta, ".//journal-id[@journal-id-type='pmc']"),
            "issn_print": _text(jmeta, ".//issn[@pub-type='ppub']"),
            "issn_epub": _text(jmeta, ".//issn[@pub-type='epub']"),
            "publisher": _text(jmeta, ".//publisher-name"),
            "publisher_loc": _text(jmeta, ".//publisher-loc"),
        }

    # Title
    title = _text(ameta, ".//title-group/article-title")
    running_head = _text(ameta, ".//title-group/alt-title[@alt-title-type='running-head']")

    # Authors & affiliations
    affiliations: dict[str, dict] = {}
    for aff in ameta.findall(".//aff"):
        aff_id = aff.get("id", "")
        affiliations[aff_id] = {
            "institution": _text(aff, "institution"),
            "addr_line": _text(aff, "addr-line"),
            "country": _text(aff, "country"),
        }

    authors = []
    for contrib in ameta.findall(".//contrib[@contrib-type='author']"):
        name_el = contrib.find("name")
        if name_el is None:
            continue
        aff_refs = [xr.get("rid", "") for xr in contrib.findall("xref[@ref-type='aff']")]
        author = {
            "surname": _text(name_el, "surname"),
            "given_names": _text(name_el, "given-names"),
            "corresponding": contrib.get("corresp") == "yes",
            "equal_contrib": contrib.get("equal-contrib") == "yes",
            "deceased": contrib.get("deceased") == "yes",
            "affiliations": [affiliations[r] for r in aff_refs if r in affiliations],
        }
        email_el = contrib.find("email")
        if email_el is not None:
            author["email"] = "".join(email_el.itertext()).strip()
        authors.append(author)

    author_surnames = [a["surname"] for a in authors]

    # Publication dates
    pub_date_print = _parse_date(ameta.find(".//pub-date[@pub-type='ppub']"))
    pub_date_epub = _parse_date(ameta.find(".//pub-date[@pub-type='epub']"))
    pub_date_release = _parse_date(ameta.find(".//pub-date[@pub-type='pmc-release']"))

    pub_year: int | None = None
    for pd in ameta.findall(".//pub-date"):
        yr = _text(pd, "year")
        if yr:
            pub_year = int(yr)
            break

    date_received = _parse_date(ameta.find(".//history/date[@date-type='received']"))
    date_accepted = _parse_date(ameta.find(".//history/date[@date-type='accepted']"))

    # Categories & keywords
    heading = _text(ameta, ".//subj-group[@subj-group-type='heading']/subject")
    subjects = _all_text(ameta, ".//subj-group[@subj-group-type='Discipline']/subject")
    organisms = _all_text(ameta, ".//subj-group[@subj-group-type='System Taxonomy']/subject")
    keywords = _all_text(ameta, ".//kwd-group/kwd")

    # Abstract
    abstract_el = ameta.find("abstract")
    abstract = "".join(abstract_el.itertext()).strip() if abstract_el is not None else ""

    toc_abstract_el = ameta.find("abstract[@abstract-type='toc']")
    toc_abstract = "".join(toc_abstract_el.itertext()).strip() if toc_abstract_el is not None else ""

    # Volume / issue / pages
    volume = _text(ameta, "volume")
    issue = _text(ameta, "issue")
    fpage = _text(ameta, "fpage")
    lpage = _text(ameta, "lpage")
    elocation = _text(ameta, "elocation-id")

    # License
    license_el = ameta.find(".//license")
    license_url = ""
    if license_el is not None:
        license_url = license_el.get(f"{{{XLINK_NS}}}href", "")

    license_tag = ""
    if "creativecommons.org/licenses/by/" in license_url:
        license_tag = "CC-BY"
    elif "creativecommons.org/licenses/by-nc/" in license_url:
        license_tag = "CC-BY-NC"
    elif "creativecommons.org/publicdomain/zero" in license_url:
        license_tag = "CC0"

    # Body text & sections
    full_text, sections = _extract_body(body)

    # References
    references = _parse_references(back)
    ref_count = len(references)

    # Figure / table counts
    fig_count = len(body.findall(".//fig")) if body is not None else 0
    table_count = len(body.findall(".//table-wrap")) if body is not None else 0

    # Supplementary material
    supp_labels = _all_text(
        body if body is not None else front,
        ".//supplementary-material/label",
    )

    doc = {
        "id": pmcid,
        "pmcid": pmcid,
        "pmid": pmid,
        "doi": doi,
        "article_type": root.get("article-type", ""),
        "open_access": True,
        "journal": journal,
        "journal_title": journal.get("title", ""),
        "title": title,
        "running_head": running_head,
        "abstract": abstract,
        "toc_abstract": toc_abstract,
        "full_text": full_text,
        "sections": sections,
        "authors": authors,
        "author_surnames": author_surnames,
        "pub_date_print": pub_date_print,
        "pub_date_epub": pub_date_epub,
        "pub_date_release": pub_date_release,
        "pub_year": pub_year,
        "date_received": date_received,
        "date_accepted": date_accepted,
        "heading": heading,
        "subjects": subjects,
        "organisms": organisms,
        "keywords": keywords,
        "volume": volume,
        "issue": issue,
        "fpage": fpage,
        "lpage": lpage,
        "elocation_id": elocation,
        "license_url": license_url,
        "license_tag": license_tag,
        "references": references,
        "ref_count": ref_count,
        "fig_count": fig_count,
        "table_count": table_count,
        "supp_labels": supp_labels,
    }

    return doc


# ---------------------------------------------------------------------------
# CLI: quick-inspect one or more XML files
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Parse JATS XML and print JSON")
    parser.add_argument("files", nargs="+", help="XML file paths")
    parser.add_argument("--compact", action="store_true", help="One-line JSON output")
    args = parser.parse_args()

    for path in args.files:
        doc = parse_article(path)
        if doc is None:
            print(f"SKIP (parse failed): {path}", file=sys.stderr)
            continue
        indent = None if args.compact else 2
        print(json.dumps(
            {k: v for k, v in doc.items() if k not in ("full_text", "sections")},
            indent=indent, ensure_ascii=False,
        ))
