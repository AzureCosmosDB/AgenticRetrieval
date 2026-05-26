# How to Use Agentic Retrieval

This guide expands the root README workflow for the sample-data pipeline. It assumes you are using the root `config.yaml.example` template and data under `data/`. For PubMed / PMC Open Access data, use the PubMed-specific README and config files under `pubmed/`.

## What You Need

### Local Runtime

Use Python 3.10 or later. The checked-in `requirements.txt` was generated with Python 3.13, but the project targets Python 3.10+.

Install dependencies from the repository root:

```bash
pip install -r requirements.txt
```

The dependency set includes:

- Azure SDK packages for Cosmos DB, identity, and management operations.
- OpenAI and HTTP clients for LLM and embedding requests.
- YAML, dotenv, tqdm, and numerical packages used by configuration, progress, and retrieval logic.

You can also use the helper scripts from the root folder:

```bash
source ./run.sh
```

```powershell
./run.ps1
```

### Azure Resources

The root workflow expects:

- An Azure Cosmos DB account using the NoSQL API.
- A Cosmos DB database and containers, or permissions/settings that let the upload script create them.
- An Azure OpenAI or compatible embedding endpoint.
- An Azure OpenAI or compatible chat/completions endpoint for answer generation.

If you use Cosmos DB vector or full-text search, the account and container policies must support those features. The sample config includes vector and full-text policy JSON for the sample sources.

## Configure the Sample Pipeline

### 1. Copy the Example Config

The root config template is for the sample data layout under `data/`.

```bash
cp config.yaml.example config.yaml
```

Do not put real secrets in `config.yaml.example`; fill in the copied `config.yaml` instead.

### 2. Fill LLM Settings

In `config.yaml`, configure `llm`:

```yaml
llm:
  llm_endpoint: "https://<your-resource>.openai.azure.com/"
  api_version: "2024-05-01-preview"
  llm_model: "<deployment-name>"
  use_rbac_auth: true
  token_scope: ""
  llm_api_key: ""
```

Use either identity-based auth or key auth:

- For RBAC, keep `use_rbac_auth: true` and make sure your signed-in identity has access.
- For key auth, set `use_rbac_auth: false` and provide `llm_api_key`.

### 3. Fill Embedding Settings

Configure `embedding`:

```yaml
embedding:
  embed_endpoint: "https://<your-resource>.openai.azure.com/"
  embed_model: "<embedding-deployment-name>"
  embed_dimensions: 1024
  use_rbac_auth: true
  embed_api_key: ""
```

`embed_dimensions` must match the vectors your embedding model returns and the Cosmos DB vector policy dimensions. If you change dimensions, update `cosmos.vector_embedding_policy_json` as well.

### 4. Fill Cosmos DB Settings

Configure the Cosmos DB account and database:

```yaml
cosmos:
  uri: "https://<account>.documents.azure.com:443/"
  database_name: "divdet"
  use_rbac_auth: true
  key: ""
```

For RBAC, your identity needs data-plane access to read and write items. If the script will create missing containers or enable account features, you also need management-plane permissions and these fields:

```yaml
cosmos:
  cosmos_account_name: "<account-name>"
  cosmos_resource_group: "<resource-group>"
  azure_subscription_id: "<subscription-id>"
```

If you already created the database and containers manually, the management fields are less important, but `cosmos.uri`, `database_name`, and authentication still need to be correct.

### 5. Configure Sources

Each entry in `cosmos.sources` maps one local input to one Cosmos DB container.

A source needs at least:

```yaml
sources:
  - id: "source_1"
    container_name: "container_1"
    partition_key_path: "/pk"
    embedding_field: "e"
    documents_root: "data/solar-system.jsonl"
    embedding_text_fields:
      - title
      - summary
      - content
    retrieval:
      vector_k: 10
      fulltext_k: 10
      fulltext_fields:
        - title
```

Keep the embedding field and vector paths consistent. If `embedding_field` is `e`, vector paths should point at `/e` and the excluded path should include `/e/*`.

### 6. Configure Paths

Set the question and output paths:

```yaml
paths:
  questions_path: "data/questions-answers.json"
  output_root: "out"
```

The retrieval command can override both values, but setting them in YAML makes repeated runs easier.

## Upload Documents

Run from the repository root:

```bash
python cosmos_db_upload.py --config config.yaml
```

The upload script does the following:

1. Loads `config.yaml`.
2. Connects to Cosmos DB.
3. Creates the database and containers if needed and if management settings allow it.
4. Reads documents from each source's `documents_root`.
5. Builds embedding input from `embedding_text_fields`.
6. Calls the embedding endpoint in batches.
7. Writes each document with its embedding vector to the configured container.

Upload concurrency is controlled by:

```yaml
cosmos:
  embedding_batch_size: 100
  concurrent_batches: 4
```

Use smaller values if your embedding endpoint is rate limited. Use larger values only after confirming your endpoint and Cosmos DB throughput can handle the load.

## Run Retrieval and Answer Generation

Run from the repository root:

```bash
python agentic_retriever.py --config config.yaml --questions-path data/questions-answers.json
```

For a quick smoke test:

```bash
python agentic_retriever.py --config config.yaml --questions-path data/questions-answers.json --max-questions 1
```

The questions file should be a JSON array with `question_id` and `question_text`. If you want evaluation output, include `answer` as the ground-truth answer.

Example:

```json
[
  {
    "question_id": "1",
    "question_text": "Which planet is known for its rings?",
    "answer": "Saturn"
  }
]
```

## Tune Retrieval

The main retrieval knobs are in `config.yaml` under `retrieval`, `pipeline`, and each source's `retrieval` block.

Source-level settings:

- `retrieval.vector_k`: vector results per source.
- `retrieval.fulltext_k`: full-text results per source.
- `retrieval.fulltext_fields`: fields searched lexically.

Global retrieval settings:

- `retrieval.k_diverse`: number of chunks kept after diversity selection. `0` disables diversity selection.
- `retrieval.eta`: Gram matrix regularization for diversity selection.
- `retrieval.rescale_power`: query-similarity rescaling for diversity selection.

Pipeline settings:

- `pipeline.max_sub_questions`: maximum sub-questions per round.
- `pipeline.subq_fanout_cap`: cap on generated follow-up searches.
- `pipeline.subq_max_concurrency`: concurrent sub-question work.
- `pipeline.rounds`: maximum refinement rounds.
- `pipeline.dataset_description`: optional description added to prompts.

## Runtime Overrides

You can override many YAML settings from the command line:

```bash
python agentic_retriever.py \
  --config config.yaml \
  --questions-path data/questions-answers.json \
  --max-questions 1 \
  --k-diverse 20 \
  --rounds 2 \
  --timing
```

Common overrides:

- `--k-diverse`
- `--k-ranker`
- `--eta`
- `--rescale-power`
- `--max-sub-questions`
- `--subq-fanout-cap`
- `--subq-max-concurrency`
- `--rounds`
- `--max-questions`
- `--max-workers`
- `--questions-path`
- `--output-root`
- `--timing`
- `--cosmos-az-login`
- `--azure-az-login`

## Read Outputs

Outputs are written under `paths.output_root` or the value passed to `--output-root`.

Look for:

- `questions_with_answers.json`: final grouped answers.
- `intermediate/`: per-question traces, retrieved context, and intermediate reasoning artifacts.
- timing logs when timing is enabled.

The exact output folder can include retrieval settings, so separate parameter runs do not overwrite each other.

## Generate a Timing Summary

Run:

```bash
python timing_summary.py
```

This script reruns a short timed benchmark and writes comparison files under the output root. Use it after you have a working config and question file.

For one-off profiling, add `--timing` directly to retrieval:

```bash
python agentic_retriever.py --config config.yaml --questions-path data/questions-answers.json --max-questions 1 --timing
```

## Troubleshooting

### `cosmos.sources` Is Missing or Empty

Both upload and retrieval expect `cosmos.sources` to be a non-empty list. Add at least one source with `container_name`, `partition_key_path`, `documents_root`, `embedding_field`, retrieval settings, and policies.

### A Source Is Skipped During Upload

The upload script skips sources that do not have required upload fields. Check:

- `container_name`
- `partition_key_path`
- `documents_root`
- `embedding_text_fields`

### Missing Container

If you want the script to create containers, configure:

- `cosmos.azure_subscription_id`
- `cosmos.cosmos_resource_group`
- `cosmos.cosmos_account_name`

Your identity also needs permission to manage Cosmos DB resources. Otherwise, create the database and containers manually before upload.

### Embedding Dimension Errors

Make sure these agree:

- `embedding.embed_dimensions`
- `cosmos.vector_embedding_policy_json.vectorEmbeddings[].dimensions`
- the actual dimension returned by the embedding deployment
- any existing container vector policy

Vector policies are container-level settings. If an existing container was created with different dimensions or a different vector path, create a new compatible container.

### Vector Path Mismatch

If retrieval returns no vector results or Cosmos DB rejects vector queries, check that these paths line up:

- `cosmos.sources[].embedding_field`, such as `e`
- vector embedding policy path, such as `/e`
- vector index path, such as `/e`
- excluded vector internals path, such as `/e/*`

### Authentication Fails

For key auth, set the relevant `use_rbac_auth` value to `false` and provide the configured key.

For RBAC, sign in with Azure CLI or use an environment supported by `DefaultAzureCredential`. Make sure your identity has the required data-plane role for Cosmos DB and the required access to Azure OpenAI.

### Empty Output

Check that:

- `--questions-path` points to an existing JSON file.
- The JSON file contains an array of question objects.
- Each object has `question_id` and `question_text`.
- Retrieval settings are not all set to zero.
- The configured Cosmos DB containers contain uploaded documents.

## Next Reading

- [Concepts](Concepts.md) explains the terms used above.
- [Docs overview](README.md) links the full documentation set.
- The root README remains the shortest setup path.
