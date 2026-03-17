import argparse
import csv
import glob
import json
import os
import pickle
import xml.etree.ElementTree as ET
from collections import defaultdict
from tqdm import tqdm


def load_pmcid_pmid_maps(csv_path):
    """Load PMC-ids.csv into two dicts: pmcid->pmid and pmid->pmcid."""
    pmcid_to_pmid = {}
    pmid_to_pmcid = {}
    print(f"Loading {csv_path} ...")
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, quotechar='"')
        for row in reader:
            pmcid = (row.get("PMCID") or "").strip()
            pmid = (row.get("PMID") or "").strip()
            if pmcid and pmid:
                pmcid_to_pmid[pmcid] = pmid
                pmid_to_pmcid[pmid] = pmcid
    print(f"  Loaded {len(pmcid_to_pmid)} PMCID<->PMID mappings")
    return pmcid_to_pmid, pmid_to_pmcid


# def load_refs(tsv_path):
#     """Load C04_ReferenceList_Papers.tsv into two dicts: pmid->{cited pmids} and pmid->{citing pmids}."""
#     cited_by = defaultdict(list)   # RefPMID -> [PMIDs that cite it]
#     cites = defaultdict(list)      # PMID -> [RefPMIDs it cites]
#     print(f"Loading {tsv_path} ...")
#     with open(tsv_path, newline="", encoding="utf-8") as f:
#         reader = csv.DictReader(f, delimiter="\t")
#         for row in reader:
#             pmid = (row.get("PMID") or "").strip()
#             ref_pmid = (row.get("RefPMID") or "").strip()
#             if pmid and ref_pmid:
#                 cited_by[ref_pmid].append(pmid)
#                 cites[pmid].append(ref_pmid)
#     print(f"  Loaded {sum(len(v) for v in cites.values())} citation links")
#     return cited_by, cites

def load_refs(tsv_path, relevant_pmids):
    """Load only citation rows where at least one PMID is in relevant_pmids. Uses pickle cache."""
    cache_path = tsv_path + f".cache.{len(relevant_pmids)}.pkl"
    if os.path.exists(cache_path):
        print(f"Loading cached refs from {cache_path} ...")
        with open(cache_path, "rb") as f:
            cited_by, cites = pickle.load(f)
        print(f"  Loaded from cache ({sum(len(v) for v in cites.values())} citation links)")
        return cited_by, cites

    cited_by = defaultdict(list)
    cites = defaultdict(list)
    print(f"Loading {tsv_path} into memory ...")
    with open(tsv_path, "rb") as f:
        data = f.read()
    print(f"  Read {len(data) / (1024**3):.1f} GB, processing lines ...")
    lines = data.decode("utf-8").split("\n")
    del data
    count = 0
    for line in tqdm(lines[1:]):  # skip header
        tab = line.find("\t")
        if tab == -1:
            continue
        pmid = line[:tab]
        ref_pmid = line[tab + 1:].rstrip("\r")
        if pmid in relevant_pmids or ref_pmid in relevant_pmids:
            cited_by[ref_pmid].append(pmid)
            cites[pmid].append(ref_pmid)
            count += 1
    del lines
    print(f"  Loaded {count} citation links, saving cache ...")
    with open(cache_path, "wb") as f:
        pickle.dump((dict(cited_by), dict(cites)), f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Cache saved to {cache_path}")
    return cited_by, cites


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_xml", help="Path to folder containing XML files")
    parser.add_argument("--pmid_pmcid_csv", help="Path to CSV file mapping PMIDs to PMCIDs and back")
    parser.add_argument("--refs_tsv", help="Path to TSV file containing paper references")
    parser.add_argument("--limit_low", type=int, default=0, help="Lower index of the papers to process")
    parser.add_argument("--limit_high", type=int, default=1000, help="Upper index of the papers to process")
    args = parser.parse_args()
    PATH_XML = args.path_xml

    xml_files = glob.glob(os.path.join(PATH_XML, "PMC*xxxxxx", "*.xml"))
    ALL_PAPERS = {os.path.splitext(os.path.basename(p))[0] for p in xml_files}

    pmcid_to_pmid, pmid_to_pmcid = load_pmcid_pmid_maps(args.pmid_pmcid_csv)
    # Only load citation rows involving papers we have XML for
    relevant_pmids = {pmcid_to_pmid[p] for p in ALL_PAPERS if p in pmcid_to_pmid}
    cited_by, cites = load_refs(args.refs_tsv, relevant_pmids)


    def filter_pmids_to_available_pmcids(pmids):
        result = []
        for pmid in pmids:
            pmcid = pmid_to_pmcid.get(str(pmid))
            if pmcid and pmcid in ALL_PAPERS:
                result.append(pmcid)
        return result

    def find_citing_and_cited_papers(pmcid):
        pmid = pmcid_to_pmid.get(pmcid)
        if not pmid:
            return [], []
        citing_pmids = cited_by.get(pmid, [])
        cited_pmids = cites.get(pmid, [])
        return filter_pmids_to_available_pmcids(citing_pmids), filter_pmids_to_available_pmcids(cited_pmids)

    def extract_text(element):
        """Recursively extract all text content from an XML element."""
        if element is None:
            return ""
        return " ".join(element.itertext()).strip()


    def extract_citations(root):
        """Extract all element-citation entries as a list of {authors, name}."""
        citations = []
        for cite_el in root.findall(".//element-citation"):
            authors = []
            for name_el in cite_el.findall(".//person-group/name"):
                surname = name_el.findtext("surname", "")
                given = name_el.findtext("given-names", "")
                authors.append(f"{surname} {given}".strip())
            title = extract_text(cite_el.find("article-title"))
            if not title:
                title = extract_text(cite_el.find("source"))
            citations.append({"authors": authors, "name": title})
        return citations


    def extract_fields(filepath):
        try:
            tree = ET.parse(filepath)
            root = tree.getroot()
        except ET.ParseError:
            return None

        title_el = root.find(".//article-title")
        abstract_el = root.find(".//abstract")
        body_el = root.find(".//body")
        journal_title_el = root.find(".//journal-title")
        ack_el = root.find(".//ack")

        pmcid = os.path.basename(filepath).replace(".xml", "")
        citing, cited = find_citing_and_cited_papers(pmcid)

        return {
            "file_name": os.path.basename(filepath),
            "journal_title": extract_text(journal_title_el),
            "title": extract_text(title_el),
            "abstract": extract_text(abstract_el),
            "body": extract_text(body_el),
            "acknowledgement": extract_text(ack_el),
            "citations": extract_citations(root),
            "citing": citing,
            "cited": cited,
        }



    print(f"Found {len(xml_files)} XML files")

    results = []
    for filepath in tqdm(xml_files[args.limit_low:args.limit_high]):
        record = extract_fields(filepath)
        if record is not None:
            results.append(record)

    output_path = os.path.join(PATH_XML, "extracted.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Extracted {len(results)} records to {output_path}")

if __name__ == "__main__":
    main()