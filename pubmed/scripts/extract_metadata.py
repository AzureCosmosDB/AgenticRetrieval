"""Extract metadata (title, abstract, body, etc.) from PubMed XML files into JSON."""

import argparse
import glob
import json
import os
import xml.etree.ElementTree as ET
from tqdm import tqdm


def extract_text(element):
    """Recursively extract all text content from an XML element."""
    if element is None:
        return ""
    return " ".join(element.itertext()).strip()

def extract_fields(filepath):
    try:
        tree = ET.parse(filepath)
        root = tree.getroot()
    except ET.ParseError:
        return None

    with open(filepath) as f:
        xml_content = f.read()

    return {
        "pmcid": os.path.basename(filepath).replace(".xml", ""),
        "journal_title": extract_text(root.find(".//journal-title")),
        "title": extract_text(root.find(".//article-title")),
        "abstract": extract_text(root.find(".//abstract")),
        "body": extract_text(root.find(".//body")),
        "acknowledgement": extract_text(root.find(".//ack")),
        "raw_xml": xml_content,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_xml", required=True, help="Path to folder containing XML files")
    parser.add_argument("--output", default=None, help="Output JSON path (default: <path_xml>/extracted.json)")
    parser.add_argument("--limit_low", type=int, default=0)
    parser.add_argument("--limit_high", type=int, default=1000)
    args = parser.parse_args()

    xml_files = sorted(glob.glob(os.path.join(args.path_xml, "PMC*xxxxxx", "*.xml")))
    print(f"Found {len(xml_files)} XML files")

    subset = xml_files[args.limit_low:args.limit_high]
    results = []
    for filepath in tqdm(subset):
        record = extract_fields(filepath)
        if record is not None:
            results.append(record)

    output_path = args.output or os.path.join(args.path_xml, "extracted.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Extracted {len(results)} records to {output_path}")


if __name__ == "__main__":
    main()
