# Agentic Retrieval Docs

These docs explain the root Agentic Retrieval workflow: loading JSON or JSONL data, storing embeddings in Azure Cosmos DB, retrieving relevant chunks, and generating answers with the decomposed RAG pipeline.

They focus on the sample-data path configured by the root `config.yaml.example` and files under `data/`. For PubMed / PMC Open Access ingestion, use the PubMed-specific documentation and configuration under `pubmed/`.

## Contents

- [Overview](Overview.md) introduces what Agentic Retrieval is, why single-shot RAG falls short, and walks through the end-to-end retrieval + iterative gap-filling flow.
- [Concepts](Concepts.md) explains the main terms used in this repository, including sources, embeddings, Cosmos DB vector and full-text search, diversity selection, reranking, and decomposed RAG.
- [How to use](How-to-use.md) walks through dependencies, configuration, document upload, retrieval, outputs, timing, and common troubleshooting steps.

## Quick Path

1. Install Python dependencies from the repository root.
2. Copy `config.yaml.example` to `config.yaml`.
3. Fill in Azure OpenAI, embedding, Cosmos DB, and `cosmos.sources` settings.
4. Run `cosmos_db_upload.py` to ingest documents.
5. Run `agentic_retriever.py` with a questions file to generate answers.

See [How to use](How-to-use.md) for the complete flow.
