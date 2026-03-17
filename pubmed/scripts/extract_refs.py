"""Add citing/cited paper lists to an extracted JSON using PKG reference data."""

import argparse
import csv
import json
import os
import pickle
from collections import defaultdict
from tqdm import tqdm
import glob


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


def load_refs(tsv_path, relevant_pmids):
    """Load only citation rows where at least one PMID is in relevant_pmids. Uses pickle cache."""
    cache_path = tsv_path + f".cache.{len(relevant_pmids)}.pkl"
    print(f"Looking for cache at {cache_path} ...")
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
    parser.add_argument("--path_xml", required=True, help="Path to folder containing XML files")
    parser.add_argument("--input_json", required=True, help="Path to extracted JSON from extract_metadata.py")
    parser.add_argument("--output", default=None, help="Output JSON path (default: <input>_with_refs.json)")
    parser.add_argument("--pmid_pmcid_csv", required=True, help="Path to PMC-ids.csv")
    parser.add_argument("--refs_tsv", required=True, help="Path to C04_ReferenceList_Papers.tsv")
    args = parser.parse_args()

    print(f"Loading {args.input_json} ...")
    with open(args.input_json) as f:
        records = json.load(f)
    print(f"  Loaded {len(records)} records")

    xml_files = glob.glob(os.path.join(args.path_xml, "PMC*xxxxxx", "*.xml"))
    all_pmcids = {os.path.splitext(os.path.basename(p))[0] for p in xml_files}

    pmcid_to_pmid, pmid_to_pmcid = load_pmcid_pmid_maps(args.pmid_pmcid_csv)
    relevant_pmids = {pmcid_to_pmid[p] for p in all_pmcids if p in pmcid_to_pmid}
    cited_by, cites = load_refs(args.refs_tsv, relevant_pmids)

    def filter_pmids_to_available_pmcids(pmids):
        result = []
        for pmid in pmids:
            pmcid = pmid_to_pmcid.get(str(pmid))
            if pmcid and pmcid in all_pmcids:
                result.append(pmcid)
        return result

    print("Adding citing/cited papers ...")
    for rec in tqdm(records):
        pmcid = rec["pmcid"]
        pmid = pmcid_to_pmid.get(pmcid)
        if pmid:
            rec["citing"] = filter_pmids_to_available_pmcids(cited_by.get(pmid, []))
            rec["cited"] = filter_pmids_to_available_pmcids(cites.get(pmid, []))
        else:
            rec["citing"] = []
            rec["cited"] = []

    output_path = args.output or args.input_json.replace(".json", "_with_refs.json")
    with open(output_path, "w") as f:
        json.dump(records, f, indent=2)

    print(f"Saved {len(records)} records to {output_path}")


if __name__ == "__main__":
    main()
