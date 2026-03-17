import argparse
import glob
import json
import os
import xml.etree.ElementTree as ET
from tqdm import tqdm
import duckdb

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_xml", help="Path to folder containing XML files")
    parser.add_argument("--pmid_pmcid_csv", help="Path to CSV file mapping PMIDs to PMCIDs and back")
    parser.add_argument("--refs_tsv", help="Path to TSV file containing paper references")
    parser.add_argument("--limit_low", type=int, default=0, help="Lower index of the papers to process")
    parser.add_argument("--limit_high", type=int, default=1000, help="Upper index of the papers to process")
    parser.add_argument("--path_duckdb", default=None, help="Path to DuckDB database file (optional)")
    args = parser.parse_args()
    PATH_XML = args.path_xml
    PMID_PMCID_CSV = args.pmid_pmcid_csv
    REFS_TSV = args.refs_tsv

    xml_files = glob.glob(os.path.join(PATH_XML, "PMC*xxxxxx", "*.xml"))
    ALL_PAPERS = {os.path.splitext(os.path.basename(p))[0] for p in xml_files}

    if args.path_duckdb:
        con = duckdb.connect(args.path_duckdb)
    else:
        # Pre-load tables into memory with indexes for faster querying
        con = duckdb.connect()
    print("Loading PMC-ids table...")
    con.sql(f"""
        CREATE TABLE pmc_ids AS
        SELECT PMID, PMCID FROM read_csv('{PMID_PMCID_CSV}',
            header=true, all_varchar=true, quote='"')
    """)
    con.sql("CREATE INDEX idx_pmc_pmcid ON pmc_ids(PMCID)")
    con.sql("CREATE INDEX idx_pmc_pmid ON pmc_ids(PMID)")
    print("Loading references table...")
    con.sql(f"""
        CREATE TABLE refs AS
        SELECT PMID, RefPMID FROM read_csv('{REFS_TSV}',
            delim='\t', header=true)
    """)
    con.sql("CREATE INDEX idx_refs_refpmid ON refs(RefPMID)")
    con.sql("CREATE INDEX idx_refs_pmid ON refs(PMID)")
    print("Tables loaded.")



    def filter_pmids_to_available_pmcids(refs):
        refs = [r[0] for r in refs]
        if not refs:
            return []
        matches = con.execute(
            "SELECT PMCID FROM pmc_ids WHERE PMID IN (SELECT UNNEST($1::VARCHAR[]))",
            [[str(r) for r in refs]]
        ).fetchall()
        ref_filtered = [r[0] for r in matches if r[0] in ALL_PAPERS]
        return ref_filtered

    def find_citing_and_cited_papers(pmcid):
        row = con.execute("SELECT PMID FROM pmc_ids WHERE PMCID = $1 LIMIT 1", [pmcid]).fetchone()
        if row is None or row[0] is None:
            return [], []
        pmid = row[0]

        cite = con.execute("SELECT PMID FROM refs WHERE RefPMID = $1", [pmid]).fetchall()

        cited = con.execute("SELECT RefPMID FROM refs WHERE PMID = $1", [pmid]).fetchall()

        return filter_pmids_to_available_pmcids(cite), filter_pmids_to_available_pmcids(cited)

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