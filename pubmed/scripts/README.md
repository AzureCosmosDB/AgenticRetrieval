# Preparation

Download and unzip all files from https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_comm/xml/ . It requires a lot of space! Let `A` be the folder containing all unzipped folders that have name `PMC*xxxxxx`.

Download file `PMC-ids.csv.gz` from `https://ftp.ncbi.nlm.nih.gov/pub/pmc/` and unzip it. Let `B` be the path to the unzipped `PMC-ids.csv` file.

Download file `C04_ReferenceList_Papers.tsv.gz` from `https://doi.org/10.6084/m9.figshare.26893861` and unzip it. Let `C` be the path to the unzipped `C04_ReferenceList_Papers.tsv` file.

Install the required Python dependency `duckdb` (if not already installed):
```bash
pip install duckdb
```

Run `python extract_fields.py --path_xml <A> --pmid_pmcid_csv <B> --refs_tsv <C>`.

The script will save extracted fields from the first 1000 publications in a json file `extracted.json` in folder `A`. Running it for all publications will take a long time!