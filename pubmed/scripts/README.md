# Preparation

Download and unzip all files from https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_comm/xml/ . It requires a lot of space! Let `A` be the folder containing all unzipped folders that have name `PMC*xxxxxx`.

Download file `PMC-ids.csv.gz` from `https://ftp.ncbi.nlm.nih.gov/pub/pmc/` and unzip it. Let `B` be the path to the unzipped `PMC-ids.csv` file.

Download file `C04_ReferenceList_Papers.tsv.gz` from `https://doi.org/10.6084/m9.figshare.26893861` and unzip it. Let `C` be the path to the unzipped `C04_ReferenceList_Papers.tsv` file.

Install the required Python dependency `duckdb` (if not already installed):
```bash
pip install duckdb
```

Run `python extract_metadata.py --path_xml <A>`.

The script will save extracted fields from the first 1000 publications in a json file `extracted.json` in folder `A`. Running it for all publications will take a long time! If you want to run it for all publications, pass the parameter `--limit_high 4994196` (this is equal to the toal number of XML files).

Run `python extract_refs.py --path_xml <A> --pmid_pmcid_csv <B> --refs_tsv <C> --input_json <A>/extracted.json`.

This script adds citing and cited papers to the previously saved json file. When running this script for the first time, it will preprocess file `C04_ReferenceList_Papers.tsv`, which might take a longer time. The subsequent runs will be faster because it will read the proprocessed dataset.