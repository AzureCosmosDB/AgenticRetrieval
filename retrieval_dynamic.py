import argparse, asyncio, json, re, time, yaml
from pathlib import Path
import httpx
import openai
import tiktoken
from openai import AsyncAzureOpenAI
from azure.identity import AzureCliCredential, get_bearer_token_provider
from azure.identity.aio import AzureCliCredential as AsyncAzureCliCredential
from azure.cosmos.aio import CosmosClient
from prompts import RETRIEVAL_PROMPT, ANSWER_PROMPT

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="config.yaml", help="Path to config file")
args = parser.parse_args()

cfg = yaml.safe_load(Path(args.config).read_text())
llm_cfg, embed_cfg, cosmos_cfg = cfg["llm"], cfg["embedding"], cfg["cosmos"]
sources = cosmos_cfg["sources"]

# LLM client
token_provider = get_bearer_token_provider(AzureCliCredential(), llm_cfg["token_scope"]) if llm_cfg["use_rbac_auth"] else None
llm_client = AsyncAzureOpenAI(
    api_version=llm_cfg["api_version"], azure_endpoint=llm_cfg["llm_endpoint"],
    **({"azure_ad_token_provider": token_provider} if token_provider else {"api_key": llm_cfg["llm_api_key"]}),
)

# Embed client
embed_client = AsyncAzureOpenAI(
    api_version=embed_cfg["api_version"], azure_endpoint=embed_cfg["embed_endpoint"],
    api_key=embed_cfg["embed_api_key"],
)

# Per-source retrieval config
_source_cfg = {s["id"]: s["retrieval"] for s in sources}
_source_embed_field = {s["id"]: s["embedding_field"] for s in sources}
_all_embed_fields = set(_source_embed_field.values())
MAX_RETRIES = int(llm_cfg["max_retries"])

STOPWORDS = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "a's", "able", "about", "above", "according", "accordingly", "across", "actually", "after", "afterwards", "again", "against", "ain't", "all", "allow", "allows", "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "an", "and", "another", "any", "anybody", "anyhow", "anyone", "anything", "anyway", "anyways", "anywhere", "apart", "appear", "appreciate", "appropriate", "are", "aren't", "around", "as", "aside", "ask", "asking", "associated", "at", "available", "away", "awfully", "b", "be", "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "behind", "being", "believe", "below", "beside", "besides", "best", "better", "between", "beyond", "both", "brief", "but", "by", "c", "c'mon", "c's", "came", "can", "can't", "cannot", "cant", "cause", "causes", "certain", "certainly", "changes", "clearly", "co", "com", "come", "comes", "concerning", "consequently", "consider", "considering", "contain", "containing", "contains", "corresponding", "could", "couldn't", "course", "currently", "d", "definitely", "described", "despite", "did", "didn't", "different", "do", "does", "doesn't", "doing", "don", "don't", "done", "down", "downwards", "during", "e", "each", "edu", "eg", "eight", "either", "else", "elsewhere", "enough", "entirely", "especially", "et", "etc", "even", "ever", "every", "everybody", "everyone", "everything", "everywhere", "ex", "exactly", "example", "except", "f", "far", "few", "fifth", "first", "five", "followed", "following", "follows", "for", "former", "formerly", "forth", "four", "from", "further", "furthermore", "g", "get", "gets", "getting", "given", "gives", "go", "goes", "going", "gone", "got", "gotten", "greetings", "h", "had", "hadn't", "happens", "hardly", "has", "hasn't", "have", "haven't", "having", "he", "he's", "hello", "help", "hence", "her", "here", "here's", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "hi", "him", "himself", "his", "hither", "hopefully", "how", "howbeit", "however", "i", "i'd", "i'll", "i'm", "i've", "ie", "if", "ignored", "immediate", "in", "inasmuch", "inc", "indeed", "indicate", "indicated", "indicates", "inner", "insofar", "instead", "into", "inward", "is", "isn't", "it", "it'd", "it'll", "it's", "its", "itself", "j", "just", "k", "keep", "keeps", "kept", "know", "known", "knows", "l", "last", "lately", "later", "latter", "latterly", "least", "less", "lest", "let", "let's", "like", "liked", "likely", "little", "ll", "look", "looking", "looks", "ltd", "m", "mainly", "make", "many", "may", "maybe", "me", "mean", "meanwhile", "merely", "might", "more", "moreover", "most", "mostly", "mr", "mrs", "ms", "much", "must", "my", "myself", "n", "name", "namely", "nd", "near", "nearly", "necessary", "need", "needs", "neither", "never", "nevertheless", "new", "next", "nine", "no", "nobody", "non", "none", "noone", "nor", "normally", "not", "nothing", "novel", "now", "nowhere", "o", "obviously", "of", "off", "often", "oh", "ok", "okay", "old", "on", "once", "one", "ones", "only", "onto", "or", "other", "others", "otherwise", "ought", "our", "ours", "ourselves", "out", "outside", "over", "overall", "own", "p", "particular", "particularly", "per", "perhaps", "placed", "please", "plus", "possible", "presumably", "probably", "provides", "q", "que", "quite", "qv", "r", "rather", "rd", "re", "really", "reasonably", "regarding", "regardless", "regards", "relatively", "respectively", "right", "s", "said", "same", "saw", "say", "saying", "says", "second", "secondly", "see", "seeing", "seem", "seemed", "seeming", "seems", "seen", "self", "selves", "sensible", "sent", "serious", "seriously", "seven", "several", "shall", "she", "should", "shouldn't", "since", "six", "so", "some", "somebody", "somehow", "someone", "something", "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", "specified", "specify", "specifying", "still", "sub", "such", "sup", "sure", "t", "t's", "take", "taken", "tell", "tends", "th", "than", "thank", "thanks", "thanx", "that", "that's", "thats", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "there's", "thereafter", "thereby", "therefore", "therein", "theres", "thereupon", "these", "they", "they'd", "they'll", "they're", "they've", "think", "third", "this", "thorough", "thoroughly", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "took", "toward", "towards", "tried", "tries", "truly", "try", "trying", "twice", "two", "u", "un", "under", "unfortunately", "unless", "unlikely", "until", "unto", "up", "upon", "us", "use", "used", "useful", "uses", "using", "usually", "v", "value", "various", "ve", "very", "via", "viz", "vs", "w", "want", "wants", "was", "wasn't", "way", "we", "we'd", "we'll", "we're", "we've", "welcome", "well", "went", "were", "weren't", "what", "what's", "whatever", "when", "whence", "whenever", "where", "where's", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "who's", "whoever", "whole", "whom", "whose", "why", "will", "willing", "wish", "with", "within", "without", "won't", "wonder", "would", "wouldn't", "x", "y", "yes", "yet", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves", "z", "zero"}

# Build source configs for fulltext
_source_ft_fields = {s["id"]: s["retrieval"]["fulltext_fields"] for s in sources}

# Ranker setup
ranker_cfg = cfg["ranker"]
USE_RANKER = ranker_cfg["use_ranker"]
if USE_RANKER:
    _ranker_url = f"https://{ranker_cfg['account_name']}.{ranker_cfg['region']}.{ranker_cfg['url_suffix']}"
    _ranker_batch_size = ranker_cfg["batch_size"]
    _ranker_max_retries = ranker_cfg["max_retries"]
    if ranker_cfg["read_token_from_path"]:
        _ranker_token = Path(ranker_cfg["access_token_path"]).read_text().strip()
    else:
        from azure.identity import AzureCliCredential as SyncCli
        _ranker_token = SyncCli(tenant_id=ranker_cfg["tenant_id"]).get_token(ranker_cfg["token_scope"]).token
    _ranker_headers = {"Authorization": f"Bearer {_ranker_token}", "Content-Type": "application/json"}
    _ranker_http = httpx.AsyncClient(timeout=120)


async def llm_call(**kwargs) -> str:
    for attempt in range(MAX_RETRIES):
        try:
            resp = await llm_client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content
        except (openai.BadRequestError, openai.RateLimitError, openai.APIStatusError) as e:
            if attempt + 1 >= MAX_RETRIES:
                return f"LLM call failed after {MAX_RETRIES} attempts: {type(e).__name__}: {str(e)}"
            wait = min(5.0 * (2 ** attempt), 300)
            print(f"LLM error ({type(e).__name__}), retry {attempt+1}/{MAX_RETRIES} in {wait}s")
            print(e)
            await asyncio.sleep(wait)


async def embed(text: str) -> list[float]:
    r = await embed_client.embeddings.create(input=[text], model=embed_cfg["embed_model"])
    return [float(x) for x in r.data[0].embedding[:embed_cfg.get("embed_dimensions", 1536)]]


async def vector_search(container, emb: list[float], top_k: int, embed_field: str = "embedding") -> list[dict]:
    sql = f"SELECT TOP @k c, VectorDistance(c.{embed_field}, @emb) AS score FROM c ORDER BY VectorDistance(c.{embed_field}, @emb)"
    results = []
    async for item in container.query_items(query=sql, parameters=[{"name": "@k", "value": top_k}, {"name": "@emb", "value": emb}]):
        doc = item.get("c", item)
        results.append(doc)
    return results


async def fulltext_search_field(container, field: str, query: str, top_k: int) -> list[dict]:
    terms = [t for t in re.findall(r"\w+", query) if t.lower() not in STOPWORDS and len(t) > 1]
    if not terms or top_k <= 0:
        return []
    chunks = [terms[i:i+5] for i in range(0, len(terms), 5)]
    score_exprs = []
    for ch in chunks:
        args = ", ".join('"' + t.replace('"', '') + '"' for t in ch)
        score_exprs.append(f"FullTextScore(c.{field}, {args})")
    order = f"ORDER BY RANK {score_exprs[0]}" if len(score_exprs) == 1 else f"ORDER BY RANK RRF({', '.join(score_exprs)})"
    sql = f"SELECT TOP {top_k} * FROM c {order}"
    items = []
    try:
        async for item in container.query_items(query=sql, parameters=[]):
            items.append(item)
    except Exception as e:
        print(f"Fulltext error ({field}): {e}")
    return items


async def fulltext_search(container, fields: list[str], query: str, top_k: int) -> list[dict]:
    if not fields or top_k <= 0:
        return []
    if len(fields) == 1:
        return await fulltext_search_field(container, fields[0], query, top_k)
    per_field = await asyncio.gather(*(fulltext_search_field(container, f, query, top_k) for f in fields))
    # Client-side RRF merge
    scores, doc_map = {}, {}
    for field_items in per_field:
        for rank, item in enumerate(field_items):
            did = item.get("id", "")
            if did:
                scores[did] = scores.get(did, 0.0) + 1.0 / (60 + rank + 1)
                doc_map.setdefault(did, item)
    return [doc_map[did] for did in sorted(scores, key=scores.get, reverse=True)[:top_k]]


def format_doc(doc: dict) -> str:
    exclude = {"_rid", "_self", "_etag", "_attachments", "_ts", "_score", "e"} | _all_embed_fields
    return "\n".join(f"{k}: {v}" for k, v in doc.items() if k not in exclude and v)


def parse_tool_calls(text: str) -> list[str]:
    return re.findall(r'SearchTool\(\s*["\'](.+?)["\']\s*\)', text)


def parse_prune_calls(text: str) -> list[str]:
    m = re.search(r'Prune\s*\((.*?)\)', text, re.DOTALL)
    if not m:
        return []
    return re.findall(r'["\']([^"\']+)["\']', m.group(1))


async def rerank(query: str, docs: list[str], top_k: int) -> list[str]:
    if not USE_RANKER or not docs:
        return docs[:top_k]
    body = {"query": query, "documents": docs, "return_documents": False, "top_k": top_k, "batch_size": _ranker_batch_size}
    for attempt in range(_ranker_max_retries):
        resp = await _ranker_http.post(_ranker_url, headers=_ranker_headers, json=body)
        if resp.status_code in (429, 502, 503) and attempt + 1 < _ranker_max_retries:
            await asyncio.sleep(2 ** attempt)
            continue
        resp.raise_for_status()
        indices = [s["index"] for s in resp.json().get("Scores", [])[:top_k]]
        return [docs[i] for i in indices if i < len(docs)]
    return docs


RERANK_MULTIPLIER = ranker_cfg["rerank_multiplier"]
CONTEXT_LIMIT = llm_cfg["context_limit"]

_enc = tiktoken.get_encoding("o200k_base")

def count_tokens(messages: list[dict]) -> int:
    # Per OpenAI: each message has ~4 overhead tokens + content tokens
    return sum(4 + len(_enc.encode(m["content"])) for m in messages) + 2

async def search_one_query(q: str, containers: dict) -> tuple[list[str], float]:
    emb = await embed(q)
    t = time.perf_counter()
    # Vector searches
    tasks = []
    for sid, ret in _source_cfg.items():
        if sid in containers:
            tasks.append(vector_search(containers[sid], emb, ret["search_k"] * RERANK_MULTIPLIER, _source_embed_field[sid]))
    # Fulltext searches (in parallel with vector)
    for sid, fields in _source_ft_fields.items():
        if sid in containers:
            tasks.append(fulltext_search(containers[sid], fields, q, _source_cfg[sid]["fulltext_search_k"] * RERANK_MULTIPLIER))
    results = await asyncio.gather(*tasks)
    # Dedupe by doc id
    seen, all_docs = set(), []
    for doc_list in results:
        for doc in doc_list:
            did = doc.get("id", "")
            if did not in seen:
                seen.add(did)
                all_docs.append(doc)
    total_k = sum(ret["search_k"] + ret["fulltext_search_k"] for ret in _source_cfg.values())
    result = await rerank(q, [format_doc(d) for d in all_docs], total_k)
    return result, time.perf_counter() - t

async def process_question(q_obj: dict, containers: dict) -> dict:
    t0 = time.perf_counter()
    query = q_obj["question_text"]
    gt_answer = q_obj.get("answer", "")
    print(f"\n{'='*60}\nProcessing [{q_obj.get('question_id', '')}]: {query}\n{'='*60}")
    prompt = RETRIEVAL_PROMPT.format(query=query, N=10, prune_k=cfg["prune_k"])
    messages = [{"role": "user", "content": prompt}]
    round_num = 0
    search_time = 0.0
    llm_time = 0.0
    total_vector_searches = 0
    total_prune_calls = 0
    doc_cache = {}  # id -> formatted text

    async def fetch_doc(doc_id):
        if doc_id in doc_cache:
            return doc_cache[doc_id]
        for c in containers.values():
            try:
                async for item in c.query_items(query="SELECT * FROM c WHERE c.id = @id", parameters=[{"name": "@id", "value": doc_id}]):
                    formatted = format_doc(item)
                    doc_cache[doc_id] = formatted
                    return formatted
            except Exception as e:
                print(f"Error fetching doc {doc_id}: {e}")
                continue
        return None

    while True:
        round_num += 1
        print(f"\n--- Round {round_num} ---")
        t_llm = time.perf_counter()
        text = await llm_call(
            messages=messages, model=llm_cfg["llm_model"],
            temperature=llm_cfg["temperature"], max_completion_tokens=llm_cfg["max_completion_tokens"],
        )
        llm_time += time.perf_counter() - t_llm
        print(f"LLM output:\n{text}\n")

        # Check for Prune calls
        prune_ids = parse_prune_calls(text)
        if prune_ids:
            total_prune_calls += 1
            print(f"Prune requested with {len(prune_ids)} doc IDs")
            fetched = await asyncio.gather(*(fetch_doc(d) for d in prune_ids))
            pruned_context = "\n\n".join(
                f"<doc id=\"{did}\">\n{doc}\n</doc>"
                for did, doc in zip(prune_ids, fetched) if doc
            )
            # Reset messages: keep only original prompt + pruned docs
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": text},
                {"role": "user", "content": f"Pruned context (these are the only documents you have now):\n\n{pruned_context}\n\nContinue: issue more searches, prune again, or output the final document list."},
            ]
            continue

        # Check for SearchTool calls
        queries = parse_tool_calls(text)
        if not queries:  # final document list — extract doc IDs and generate answer
            doc_ids = re.findall(r'["\']([^"\']+)["\']', text)
            print(f"Final document IDs: {doc_ids}")
            fetched = await asyncio.gather(*(fetch_doc(d) for d in doc_ids))
            documents = "\n\n".join(f"[{i+1}] {doc}" for i, doc in enumerate(fetched) if doc)
            t_llm = time.perf_counter()
            answer = await llm_call(
                messages=[{"role": "user", "content": ANSWER_PROMPT.format(query=query, documents=documents)}],
                model=llm_cfg["llm_model"], temperature=llm_cfg["temperature"],
                max_completion_tokens=llm_cfg["max_completion_tokens"],
            )
            llm_time += time.perf_counter() - t_llm
            # print(f"\n--- Final Answer ---\n{answer}")
            elapsed = round(time.perf_counter() - t0, 2)
            print(f"Time: {elapsed}s")
            return {
                "question_id": q_obj.get("question_id", ""),
                "query": query, "answer": answer,
                "ground_truth": gt_answer,
                "model": llm_cfg["llm_model"],
                "rounds": round_num,
                "elapsed_seconds": elapsed,
                "llm_seconds": round(llm_time, 2),
                "search_seconds": round(search_time, 2),
                "total_vector_searches": total_vector_searches,
                "total_prune_calls": total_prune_calls,
                "number_of_documents_for_final_answer": len([f for f in fetched if f])
            }

        print(f"Parsed tool calls: {queries}")
        all_results = await asyncio.gather(*(search_one_query(q, containers) for q in queries))
        all_docs = [r[0] for r in all_results]
        search_time += max(r[1] for r in all_results)
        total_vector_searches += len(queries) * len(containers)
        doc_lines = []
        for q, docs in zip(queries, all_docs):
            doc_lines.append(f'Results for SearchTool("{q}"):')
            for i, d in enumerate(docs, 1):
                doc_lines.append(f"  [{i}] {d}")
                # Cache docs by extracting id from formatted text
                id_match = re.search(r'^id: (.+)$', d, re.MULTILINE)
                if id_match:
                    doc_cache[id_match.group(1)] = d
        context = "\n".join(doc_lines)
        new_msgs = [{"role": "assistant", "content": text}, {"role": "user", "content": f"Here are the retrieved documents:\n\n{context}"}]
        token_est = count_tokens(messages + new_msgs)
        token_info = f"Token usage: {token_est} / {CONTEXT_LIMIT}"
        print(token_info)
        messages.append({"role": "assistant", "content": text})
        messages.append({"role": "user", "content": f"Here are the retrieved documents:\n\n{context}\n\n{token_info}\nContinue: issue more searches, prune, or output the final document list."})


async def main():
    questions = json.loads(Path(cfg["paths"]["questions_path"]).read_text())

    cosmos = CosmosClient(cosmos_cfg["uri"], credential=AsyncAzureCliCredential())
    db = cosmos.get_database_client(cosmos_cfg["database_name"])
    containers = {s["id"]: db.get_container_client(s["container_name"]) for s in sources}

    results = []
    for q_obj in questions:
        result = await process_question(q_obj, containers)
        results.append(result)

    output_dir = Path(cfg["paths"]["output_root"])
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / "dynamic" / f"results_{timestamp}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {len(results)} results to {output_path}")

    await cosmos.close()

asyncio.run(main())
