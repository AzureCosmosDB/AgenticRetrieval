"""Answer questions: vector search → expand via references → pick closest refs.

For each question:
1. Vector-search top-k documents (default 10)
2. For each hit, get its references (pmid→pmcid), fetch their embeddings from Cosmos
3. Pick the ref_k references closest to the query vector (default 2)
4. Combine: up to k + k*ref_k documents (e.g. 10 + 20 = 30)
5. Answer with LLM

Usage:
    python answer_questions_ref.py --config config.pubmed.yaml \
        --questions complex_questions.json --output answers_refs.json \
        --top 10 --ref-k 2
"""

import argparse, csv, json, logging, sys, time
import numpy as np
import yaml
from azure.cosmos import CosmosClient
from azure.identity import AzureCliCredential, get_bearer_token_provider
from openai import AzureOpenAI
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
for n in ("azure.core.pipeline.policies.http_logging_policy", "azure.identity", "azure.cosmos"):
    logging.getLogger(n).setLevel(logging.WARNING)


def load_pmid_to_pmcid(csv_path: str) -> dict[str, str]:
    mapping = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, quotechar='"'):
            pmcid = (row.get("PMCID") or "").strip()
            pmid = (row.get("PMID") or "").strip()
            if pmcid and pmid:
                mapping[pmid] = pmcid
    return mapping


def get_embedding(text: str, embed_client: AzureOpenAI, model: str, dimensions: int, max_input_chars: int = 8000) -> list[float]:
    return embed_client.embeddings.create(
        input=text[:max_input_chars], model=model, dimensions=dimensions
    ).data[0].embedding


def vector_search(container, query_vector: list[float], top: int) -> list[dict]:
    sql = """
    SELECT TOP @top
        c.id, c.pmcid, c.title, c.journal_title, c.abstract, c.pub_year,
        c.doi, c.full_text, c.embedding, c.references,
        VectorDistance(c.embedding, @embedding) AS similarity
    FROM c ORDER BY VectorDistance(c.embedding, @embedding)
    """
    return list(container.query_items(
        query=sql,
        parameters=[{"name": "@embedding", "value": query_vector}, {"name": "@top", "value": top}],
        enable_cross_partition_query=True,
    ))


def fetch_docs_by_pmcids(container, pmcids: list[str]) -> list[dict]:
    if not pmcids:
        return []
    placeholders = ", ".join(f"@p{i}" for i in range(len(pmcids)))
    sql = f"""
    SELECT c.id, c.pmcid, c.title, c.journal_title, c.abstract, c.pub_year,
           c.doi, c.full_text, c.embedding
    FROM c WHERE c.pmcid IN ({placeholders})
    """
    params = [{"name": f"@p{i}", "value": p} for i, p in enumerate(pmcids)]
    return list(container.query_items(query=sql, parameters=params, enable_cross_partition_query=True))


def refs_to_pmcids(refs: list[dict], pmid_map: dict[str, str]) -> list[str]:
    result = []
    for r in refs:
        pmcid = (r.get("pmcid") or "").strip()
        if not pmcid:
            pmid = (r.get("pmid") or "").strip()
            pmcid = pmid_map.get(pmid, "")
        if pmcid:
            result.append(pmcid)
    return result


def closest_select(docs: list[dict], query_vec: list[float], k: int) -> list[dict]:
    """Select k docs closest to query_vec by cosine similarity."""
    if len(docs) <= k:
        return docs
    q = np.array(query_vec, dtype=np.float64)
    q_norm = np.linalg.norm(q)
    if q_norm < 1e-12:
        return docs[:k]
    q = q / q_norm
    scored = []
    for doc in docs:
        emb = doc.get("embedding")
        if emb and isinstance(emb, list) and len(emb) > 0:
            v = np.array(emb, dtype=np.float64)
            v_norm = np.linalg.norm(v)
            sim = float(np.dot(q, v) / max(v_norm, 1e-12))
            scored.append((sim, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:k]]


def build_context(docs: list[dict], max_chars: int = 230000) -> str:
    parts, total = [], 0
    for i, doc in enumerate(docs, 1):
        title = doc.get("title", "(no title)")
        pmcid = doc.get("pmcid", "?")
        abstract = doc.get("abstract", "")
        full_text = doc.get("full_text", "")
        sim = doc.get("similarity")
        sim_str = f" (similarity: {sim:.4f})" if sim is not None else ""
        section = f"[{i}] PMCID: {pmcid}{sim_str}\nTitle: {title}\n"
        if abstract:
            section += f"Abstract: {abstract}\n"
        if full_text:
            remaining = max_chars - total - len(section) - 200
            if remaining > 500:
                section += f"Full text excerpt: {full_text[:6000][:remaining]}\n"
        if total + len(section) > max_chars:
            break
        parts.append(section)
        total += len(section)
    return "\n".join(parts)


def answer_question(question: str, context: str, llm_client: AzureOpenAI,
                    model: str, system_prompt: str, temperature: float,
                    max_completion_tokens: int, max_retries: int) -> str:
    prompt = f"""Based on the following retrieved PubMed articles, answer this question:

Question: {question}

Retrieved Articles:
{context}

Provide a comprehensive answer synthesizing the information from these articles."""
    for attempt in range(max_retries):
        try:
            resp = llm_client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system_prompt},
                          {"role": "user", "content": prompt}],
                temperature=temperature, max_completion_tokens=max_completion_tokens,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            log.warning(f"LLM error (attempt {attempt+1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(min(5.0 * (2 ** attempt), 60))
            else:
                return f"Error: {e}"


def main():
    pa = argparse.ArgumentParser(description="Answer questions: vector + reference expansion + closest selection")
    pa.add_argument("--config", default="config.pubmed.yaml", help="Path to YAML config file")
    pa.add_argument("--questions", default=None)
    pa.add_argument("--output", default=None)
    pa.add_argument("--top", "-k", type=int, default=None, help="Initial vector search docs")
    pa.add_argument("--ref-k", type=int, default=None, help="Closest reference docs per hit")
    args = pa.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cosmos_cfg = cfg.get("cosmos", {})
    llm_cfg = cfg.get("llm", {})
    embed_cfg = cfg.get("embedding", {})
    answer_cfg = cfg.get("answer", {})

    # Resolve parameters: CLI overrides > config > defaults
    questions_path = args.questions or answer_cfg.get("questions", "questions.json")
    output_path = args.output or answer_cfg.get("output", "answers_refs.json")
    top_k = args.top if args.top is not None else int(answer_cfg.get("top", 10))
    ref_k = args.ref_k if args.ref_k is not None else int(answer_cfg.get("ref_k", 2))

    cosmos_uri = cosmos_cfg.get("uri", "")
    database_name = cosmos_cfg.get("database_name", "pubmed")
    container_name = cosmos_cfg.get("sources", [{}])[0].get("container_name", "articles")
    llm_endpoint = llm_cfg.get("llm_endpoint", "")
    llm_model = llm_cfg.get("llm_model", "gpt-5.4")
    llm_api_version = llm_cfg.get("api_version", "2024-12-01-preview")
    llm_token_scope = llm_cfg.get("token_scope", "https://cognitiveservices.azure.com/.default")
    llm_temperature = float(llm_cfg.get("temperature", 0.0))
    llm_max_tokens = int(llm_cfg.get("max_completion_tokens", 4096))
    llm_max_retries = int(llm_cfg.get("max_retries", 5))
    embed_endpoint = embed_cfg.get("embed_endpoint", "")
    embed_api_key = embed_cfg.get("embed_api_key", "")
    embed_api_version = embed_cfg.get("api_version", "2024-12-01-preview")
    embed_model = embed_cfg.get("embed_model", "text-embedding-3-small")
    embed_dims = int(embed_cfg.get("embed_dimensions", 1536))
    embed_max_input_chars = int(embed_cfg.get("max_input_chars", 8000))
    pmc_ids_csv = answer_cfg.get("pmc_ids_csv", "")
    max_context_chars = int(answer_cfg.get("max_context_chars", 230000))
    system_prompt = answer_cfg.get("system_prompt", "You are a biomedical research assistant.").strip()

    with open(questions_path) as f:
        raw = json.load(f)
    questions = [item["question"] for item in raw] if raw and isinstance(raw[0], dict) else raw
    log.info(f"Loaded {len(questions)} questions")

    pmid_map = load_pmid_to_pmcid(pmc_ids_csv) if pmc_ids_csv else {}
    log.info(f"Loaded {len(pmid_map)} PMID→PMCID mappings")

    cred = AzureCliCredential()
    container = CosmosClient(cosmos_uri, credential=cred) \
        .get_database_client(database_name).get_container_client(container_name)
    embed_client = AzureOpenAI(
        azure_endpoint=embed_endpoint, api_key=embed_api_key, api_version=embed_api_version)
    llm_client = AzureOpenAI(
        azure_endpoint=llm_endpoint,
        azure_ad_token_provider=get_bearer_token_provider(cred, llm_token_scope),
        api_version=llm_api_version)
    log.info(f"Strategy: {top_k} vector docs, {ref_k} closest refs each → max {top_k + top_k * ref_k} docs")

    results = []
    for i, question in enumerate(tqdm(questions, desc="Processing")):
        log.info(f"[{i+1}/{len(questions)}] {question[:80]}...")
        query_vec = get_embedding(question, embed_client, embed_model, embed_dims, embed_max_input_chars)

        # 1. Vector search
        hits = vector_search(container, query_vec, top=top_k)
        seen_pmcids = {d.get("pmcid") for d in hits}
        log.info(f"  Retrieved {len(hits)} hits")

        # 2. For each hit, get reference PMCIDs, fetch them, pick ref_k closest to query
        ref_docs_all = []
        if ref_k > 0:
            for doc in hits:
                refs = doc.get("references") or []
                ref_pmcids = [p for p in refs_to_pmcids(refs, pmid_map) if p not in seen_pmcids]
                if not ref_pmcids:
                    continue
                fetched = fetch_docs_by_pmcids(container, ref_pmcids)
                if not fetched:
                    continue
                selected = closest_select(fetched, query_vec, ref_k)
                for s in selected:
                    if s.get("pmcid") not in seen_pmcids:
                        seen_pmcids.add(s.get("pmcid"))
                        ref_docs_all.append(s)

        all_docs = hits + ref_docs_all
        log.info(f"  Total docs: {len(hits)} hits + {len(ref_docs_all)} ref expansions = {len(all_docs)}")

        # 3. Answer
        context = build_context(all_docs, max_chars=max_context_chars)
        answer = answer_question(question, context, llm_client,
                                 model=llm_model, system_prompt=system_prompt,
                                 temperature=llm_temperature,
                                 max_completion_tokens=llm_max_tokens,
                                 max_retries=llm_max_retries)

        # 4. Collect
        retrieved = [{"pmcid": d.get("pmcid"), "title": d.get("title"),
                       "journal": d.get("journal_title"), "year": d.get("pub_year"),
                       "similarity": d.get("similarity"), "source": "vector" if d in hits else "reference"}
                      for d in all_docs]
        results.append({"question": question, "answer": answer, "retrieved_documents": retrieved,
                         "vector_count": len(hits), "ref_count": len(ref_docs_all)})

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    log.info(f"Done. Saved {len(results)} answers to {output_path}")


if __name__ == "__main__":
    main()
