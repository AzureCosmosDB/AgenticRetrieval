# PubMed / PMC Open Access — Ingestion Pipeline

End-to-end workflow for downloading PMC Open Access articles, uploading them to
Azure Cosmos DB with vector embeddings, and optionally building a citation
graph.

## Data source & licensing

This pipeline uses the **PMC Open Access Commercial subset** — a collection of
full-text biomedical articles from [PubMed Central](https://www.ncbi.nlm.nih.gov/pmc/)
that are available for commercial reuse under permissive licenses.

| Resource | Link |
|---|---|
| PMC OA Subset overview | <https://www.ncbi.nlm.nih.gov/pmc/tools/openftlist/> |
| Commercial-use file list (oa_comm) | <https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_comm/> |
| PMC OA license terms | <https://www.ncbi.nlm.nih.gov/pmc/tools/textmining/> |
| NLM Terms of Service | <https://www.nlm.nih.gov/databases/download/terms_and_conditions.html> |

Articles in the commercial subset are distributed under one of these Creative
Commons licenses:

- **CC0** — Public Domain
- **CC BY** — Attribution
- **CC BY-SA** — Attribution-ShareAlike
- **CC BY-ND** — Attribution-NoDerivatives

Each article's specific license is recorded in the `license_tag` and
`license_url` fields of the parsed document. Always check and comply with the
individual article's license when redistributing content.

## Prerequisites

```bash
pip install -r requirements.txt   # from repo root
```

You also need:
- An **Azure Cosmos DB** account (NoSQL API) with vector search enabled
- An **Azure OpenAI** embedding deployment (e.g. `text-embedding-3-small`)
- Credentials configured in `pubmed/scripts/config.pubmed.yaml`

## 1. Download XML articles

Use `pubmed/scripts/download.py` to bulk-download PMC Open Access Commercial
subset packages:

```bash
cd pubmed/scripts

# Dry-run — see what would be downloaded
python download.py --dry-run --type xml

# Download baseline + incremental packages, extract .tar.gz
python download.py --outdir ../downloads --workers 4 --type xml --uncompress
```

This downloads `.tar.gz` archives from the NCBI FTP server and (with
`--uncompress`) extracts them into `pubmed/downloads/`. Each archive contains
thousands of JATS XML files.

## 2. Upload to Cosmos DB

The generic `cosmos_db_upload.py` (at repo root) handles embedding generation
and Cosmos DB upload. The PubMed config uses a **custom document parser** that
converts JATS XML → JSON on the fly:

```bash
# From repo root
python cosmos_db_upload.py --config pubmed/scripts/config.pubmed.yaml
```

### How it works

The `config.pubmed.yaml` source entry includes:

```yaml
document_parser: "pubmed/scripts/parse_article.py:parse_article"
file_glob: "**/*.xml"
documents_root: "pubmed/downloads/temp"
```

- **`document_parser`** — a `file:function` reference to the JATS XML parser.
  Each XML file is passed through `parse_article()` which extracts structured
  metadata (title, abstract, authors, references, full text, etc.) into a flat
  dict ready for Cosmos DB.
- **`file_glob`** — limits file discovery to XML files.
- **`documents_root`** — the directory containing extracted XML files.

The rest of the pipeline (embedding generation, container creation, batch
upload) is handled by `cosmos_db_upload.py` exactly as for JSON sources.

### Quick inspection (no Azure needed)

To test XML parsing without Cosmos DB or embeddings:

```bash
python pubmed/scripts/parse_article.py path/to/article.xml
```

## 3. Build citation graph (optional)

After the primary corpus is ingested, you can build a citation graph and patch
the Cosmos DB documents with `cited_by` / `cites` fields:

```bash
# Build citation_maps.json from XML files
python pubmed/scripts/build_citation_graph.py \
    --input pubmed/downloads/temp \
    --output pubmed/downloads/citation_maps.json \
    --workers 16

# Patch Cosmos DB documents with citation data
python pubmed/scripts/build_citation_graph.py \
    --input pubmed/downloads/temp \
    --output pubmed/downloads/citation_maps.json \
    --patch-cosmos \
    --config pubmed/scripts/config.pubmed.yaml \
    --cosmos-container articles
```

The citation graph script:
1. Downloads the NCBI PMC-ids.csv mapping (PMID ↔ PMCID)
2. Scans all XML files to extract references
3. Resolves PMID-only references to PMCIDs
4. Produces two mappings:
   - **cited_by**: article → list of articles that cite it
   - **cites**: article → list of articles it references
5. Saves to `citation_maps.json`
6. With `--patch-cosmos`, sends patch operations to update each document

## Configuration reference

See `pubmed/scripts/config.pubmed.yaml` for all settings. Key fields:

| Field | Description |
|---|---|
| `cosmos.uri` | Cosmos DB endpoint URL |
| `cosmos.database_name` | Database name (default: `pubmed`) |
| `cosmos.use_rbac_auth` | Use Entra ID RBAC instead of keys |
| `embedding.embed_endpoint` | Azure OpenAI endpoint |
| `embedding.embed_model` | Embedding model deployment name |
| `sources[].document_parser` | Custom parser (`file.py:function`) |
| `sources[].file_glob` | Glob pattern for file discovery |
| `sources[].documents_root` | Input directory or JSONL path |
