"""Standard scaffold with native function calling over Cosmos DB (vector + fulltext + ranker)."""
from tqdm import tqdm
import argparse, asyncio, json, re, time, yaml
from pathlib import Path
import httpx, openai as oai
import tiktoken
from openai import AsyncAzureOpenAI
from azure.identity import AzureCliCredential, get_bearer_token_provider
from azure.identity.aio import AzureCliCredential as AsyncAzureCliCredential
from azure.cosmos.aio import CosmosClient

STOPWORDS = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "a's", "able", "about", "above", "according", "accordingly", "across", "actually", "after", "afterwards", "again", "against", "ain't", "all", "allow", "allows", "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "an", "and", "another", "any", "anybody", "anyhow", "anyone", "anything", "anyway", "anyways", "anywhere", "apart", "appear", "appreciate", "appropriate", "are", "aren't", "around", "as", "aside", "ask", "asking", "associated", "at", "available", "away", "awfully", "b", "be", "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "behind", "being", "believe", "below", "beside", "besides", "best", "better", "between", "beyond", "both", "brief", "but", "by", "c", "c'mon", "c's", "came", "can", "can't", "cannot", "cant", "cause", "causes", "certain", "certainly", "changes", "clearly", "co", "com", "come", "comes", "concerning", "consequently", "consider", "considering", "contain", "containing", "contains", "corresponding", "could", "couldn't", "course", "currently", "d", "definitely", "described", "despite", "did", "didn't", "different", "do", "does", "doesn't", "doing", "don", "don't", "done", "down", "downwards", "during", "e", "each", "edu", "eg", "eight", "either", "else", "elsewhere", "enough", "entirely", "especially", "et", "etc", "even", "ever", "every", "everybody", "everyone", "everything", "everywhere", "ex", "exactly", "example", "except", "f", "far", "few", "fifth", "first", "five", "followed", "following", "follows", "for", "former", "formerly", "forth", "four", "from", "further", "furthermore", "g", "get", "gets", "getting", "given", "gives", "go", "goes", "going", "gone", "got", "gotten", "greetings", "h", "had", "hadn't", "happens", "hardly", "has", "hasn't", "have", "haven't", "having", "he", "he's", "hello", "help", "hence", "her", "here", "here's", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "hi", "him", "himself", "his", "hither", "hopefully", "how", "howbeit", "however", "i", "i'd", "i'll", "i'm", "i've", "ie", "if", "ignored", "immediate", "in", "inasmuch", "inc", "indeed", "indicate", "indicated", "indicates", "inner", "insofar", "instead", "into", "inward", "is", "isn't", "it", "it'd", "it'll", "it's", "its", "itself", "j", "just", "k", "keep", "keeps", "kept", "know", "known", "knows", "l", "last", "lately", "later", "latter", "latterly", "least", "less", "lest", "let", "let's", "like", "liked", "likely", "little", "ll", "look", "looking", "looks", "ltd", "m", "mainly", "make", "many", "may", "maybe", "me", "mean", "meanwhile", "merely", "might", "more", "moreover", "most", "mostly", "mr", "mrs", "ms", "much", "must", "my", "myself", "n", "name", "namely", "nd", "near", "nearly", "necessary", "need", "needs", "neither", "never", "nevertheless", "new", "next", "nine", "no", "nobody", "non", "none", "noone", "nor", "normally", "not", "nothing", "novel", "now", "nowhere", "o", "obviously", "of", "off", "often", "oh", "ok", "okay", "old", "on", "once", "one", "ones", "only", "onto", "or", "other", "others", "otherwise", "ought", "our", "ours", "ourselves", "out", "outside", "over", "overall", "own", "p", "particular", "particularly", "per", "perhaps", "placed", "please", "plus", "possible", "presumably", "probably", "provides", "q", "que", "quite", "qv", "r", "rather", "rd", "re", "really", "reasonably", "regarding", "regardless", "regards", "relatively", "respectively", "right", "s", "said", "same", "saw", "say", "saying", "says", "second", "secondly", "see", "seeing", "seem", "seemed", "seeming", "seems", "seen", "self", "selves", "sensible", "sent", "serious", "seriously", "seven", "several", "shall", "she", "should", "shouldn't", "since", "six", "so", "some", "somebody", "somehow", "someone", "something", "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", "specified", "specify", "specifying", "still", "sub", "such", "sup", "sure", "t", "t's", "take", "taken", "tell", "tends", "th", "than", "thank", "thanks", "thanx", "that", "that's", "thats", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "there's", "thereafter", "thereby", "therefore", "therein", "theres", "thereupon", "these", "they", "they'd", "they'll", "they're", "they've", "think", "third", "this", "thorough", "thoroughly", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "took", "toward", "towards", "tried", "tries", "truly", "try", "trying", "twice", "two", "u", "un", "under", "unfortunately", "unless", "unlikely", "until", "unto", "up", "upon", "us", "use", "used", "useful", "uses", "using", "usually", "v", "value", "various", "ve", "very", "via", "viz", "vs", "w", "want", "wants", "was", "wasn't", "way", "we", "we'd", "we'll", "we're", "we've", "welcome", "well", "went", "were", "weren't", "what", "what's", "whatever", "when", "whence", "whenever", "where", "where's", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "who's", "whoever", "whole", "whom", "whose", "why", "will", "willing", "wish", "with", "within", "without", "won't", "wonder", "would", "wouldn't", "x", "y", "yes", "yet", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves", "z", "zero"}

_enc = tiktoken.get_encoding("o200k_base")
def count_tokens(msgs):
    return sum(4 + len(_enc.encode(m["content"] if isinstance(m, dict) and "content" in m and isinstance(m["content"], str) else json.dumps(m) if isinstance(m, dict) else str(m))) for m in msgs) + 2

# --- Search helpers ---
async def embed(text):
    r = await embed_client.embeddings.create(input=[text], model=embed_cfg["embed_model"])
    dim = embed_cfg.get("embed_dimensions", 1536)
    raw = [float(x) for x in r.data[0].embedding]
    if len(raw) >= dim:
        return raw[:dim]
    return raw + [0.0] * (dim - len(raw))

_SAFE_FIELD_RE = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$')

async def vec_search(container, emb, top_k, ef):
    if not _SAFE_FIELD_RE.match(ef):
        raise ValueError(f"Invalid embedding field name: {ef!r}")
    sql = f"SELECT TOP @k c, VectorDistance(c.{ef}, @emb) AS score FROM c ORDER BY VectorDistance(c.{ef}, @emb)"
    return [item.get("c", item) async for item in container.query_items(query=sql, parameters=[{"name":"@k","value":top_k},{"name":"@emb","value":emb}])]

async def ft_field(container, field, query, top_k):
    if not _SAFE_FIELD_RE.match(field):
        raise ValueError(f"Invalid fulltext field name: {field!r}")
    terms = [t for t in re.findall(r"\w+", query) if t.lower() not in STOPWORDS and len(t) > 1]
    if not terms or top_k <= 0: return []
    chunks = [terms[i:i+5] for i in range(0, len(terms), 5)]
    exprs = [f'FullTextScore(c.{field}, {", ".join(chr(34)+t.replace(chr(34),"")+chr(34) for t in ch)})' for ch in chunks]
    order = f"ORDER BY RANK {exprs[0]}" if len(exprs)==1 else f"ORDER BY RANK RRF({', '.join(exprs)})"
    try: return [item async for item in container.query_items(query=f"SELECT TOP {top_k} * FROM c {order}", parameters=[])]
    except: return []

async def ft_search(container, fields, query, top_k):
    if not fields or top_k <= 0: return []
    if len(fields) == 1: return await ft_field(container, fields[0], query, top_k)
    pf = await asyncio.gather(*(ft_field(container, f, query, top_k) for f in fields))
    sc, dm = {}, {}
    for items in pf:
        for r, it in enumerate(items):
            d = it.get("id","")
            if d: sc[d] = sc.get(d,0) + 1/(60+r+1); dm.setdefault(d, it)
    return [dm[d] for d in sorted(sc, key=sc.get, reverse=True)[:top_k]]

async def rerank(query, docs, top_k):
    if not USE_RANKER or not docs: return docs[:top_k]
    body = {"query": query, "documents": docs, "return_documents": False, "top_k": top_k, "batch_size": _r_bs}
    for att in range(_r_mr):
        resp = await _r_http.post(_r_url, headers=_r_hdr, json=body)
        if resp.status_code in (429,502,503) and att+1 < _r_mr: await asyncio.sleep(2**att); continue
        resp.raise_for_status()
        return [docs[s["index"]] for s in resp.json().get("Scores",[])[:top_k] if s["index"] < len(docs)]
    return docs

def fmt(doc):
    ex = {"_rid","_self","_etag","_attachments","_ts","_score","e"} | _all_embed
    return "\n".join(f"{k}: {v}" for k,v in doc.items() if k not in ex and v)

async def do_search(query, containers):
    emb = await embed(query)
    tasks = []
    for sid, ret in _source_cfg.items():
        if sid in containers:
            tasks.append(vec_search(containers[sid], emb, ret["search_k"]*RERANK_MUL, _source_embed[sid]))
    for sid, fields in _source_ft.items():
        if sid in containers:
            tasks.append(ft_search(containers[sid], fields, query, _source_cfg[sid]["fulltext_search_k"]*RERANK_MUL))
    results = await asyncio.gather(*tasks)
    seen, all_d = set(), []
    for dl in results:
        for d in dl:
            did = d.get("id","")
            if did not in seen: seen.add(did); all_d.append(d)
    total_k = sum(r["search_k"]+r["fulltext_search_k"] for r in _source_cfg.values())
    ranked = await rerank(query, [fmt(d) for d in all_d], total_k)
    return json.dumps([{"docid": re.search(r'^id: (.+)$', d, re.MULTILINE).group(1) if re.search(r'^id: (.+)$', d, re.MULTILINE) else "", "snippet": d[:2000]} for d in ranked])

async def do_get_doc(docid, containers):
    for container_id, c in containers.items():
        try:
            async for item in c.query_items(query="SELECT * FROM c WHERE c.id=@id", parameters=[{"name":"@id","value":docid}]):
                return json.dumps({"docid": docid, "text": fmt(item)})
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"  [get_document] Cosmos query failed for container={container_id}, docid={docid}: {e}")
            continue
    return json.dumps({"error": f"Not found: {docid}"})

async def do_prune(docids, containers, doc_cache):
    parts = []
    for did in docids[:PRUNE_K]:
        if did in doc_cache:
            parts.append(f'<doc id="{did}">\n{doc_cache[did]}\n</doc>')
            continue
        found = False
        for container_id, c in containers.items():
            if found:
                break
            try:
                async for item in c.query_items(query="SELECT * FROM c WHERE c.id=@id", parameters=[{"name":"@id","value":did}]):
                    text = fmt(item)
                    doc_cache[did] = text
                    parts.append(f'<doc id="{did}">\n{text}\n</doc>')
                    found = True
                    break
            except asyncio.CancelledError:
                raise
            except Exception as e:
                print(f"  [prune] Cosmos query failed for container={container_id}, docid={did}: {e}")
                continue
    return "Pruned context (only these documents remain):\n\n" + "\n\n".join(parts)

TOOLS = [
    {"type":"function","function":{"name":"search","description":"Search knowledge base. Returns top results with docid and snippet.","parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}},
    {"type":"function","function":{"name":"get_document","description":"Get full document by docid.","parameters":{"type":"object","properties":{"docid":{"type":"string"}},"required":["docid"]}}},
    {"type":"function","function":{"name":"prune","description":"Keep only the specified most relevant document IDs and discard all others from context. Use when context is large to free up space for more searches.","parameters":{"type":"object","properties":{"docids":{"type":"array","items":{"type":"string"},"description":"List of document IDs to keep"}},"required":["docids"]}}},
]

async def process_question(q_obj, containers):
    t0 = time.perf_counter()
    query = q_obj["question_text"]
    qid = q_obj.get("question_id", "")
    print(f"\n{'='*60}\n[{qid}]: {query}\n{'='*60}")
    msgs = [{"role": "user", "content": QUERY_TEMPLATE.format(question=query)}]
    tc = {"search": 0, "get_document": 0, "prune": 0}
    doc_cache = {}
    initial_msg = msgs[0]
    retries = 0
    for iteration in range(50):
        try:
            r = await llm.chat.completions.create(model=llm_cfg["llm_model"], messages=msgs, tools=TOOLS, tool_choice="auto", temperature=llm_cfg.get("temperature", 0), max_completion_tokens=llm_cfg["max_completion_tokens"])
            retries = 0
        except (oai.BadRequestError, oai.RateLimitError, oai.APIStatusError) as e:
            retries += 1; print(f"  LLM error ({retries}/{MAX_RETRIES}): {e}")
            if retries >= MAX_RETRIES:
                elapsed = round(time.perf_counter() - t0, 2)
                print(f"  Max retries ({MAX_RETRIES}) exceeded, returning partial result")
                return {"question_id": qid, "query": query, "answer": "", "ground_truth": q_obj.get("answer",""),
                        "model": llm_cfg["llm_model"], "rounds": iteration+1, "elapsed_seconds": elapsed,
                        "tool_calls": tc, "error": f"Max retries exceeded: {e}"}
            await asyncio.sleep(min(5*2**retries, 300)); continue
        m = r.choices[0].message
        msgs.append(m.model_dump(exclude_none=True))
        if not m.tool_calls:
            answer = m.content or ""
            print(f"  Answer: {answer[:200]}...")
            elapsed = round(time.perf_counter() - t0, 2)
            print(f"  Elapsed: {elapsed}s")
            return {"question_id": qid, "query": query, "answer": answer, "ground_truth": q_obj.get("answer",""),
                    "model": llm_cfg["llm_model"], "rounds": iteration+1, "elapsed_seconds": elapsed,
                    "tool_calls": tc}
        for t in m.tool_calls:
            tc[t.function.name] = tc.get(t.function.name, 0) + 1
        # Enforce: prune must be the sole tool call in a turn
        call_names = [t.function.name for t in m.tool_calls]
        if "prune" in call_names and len(call_names) > 1:
            print(f"  [warn] prune mixed with other calls; returning error for non-prune calls")
            for t in m.tool_calls:
                if t.function.name != "prune":
                    msgs.append({"role": "tool", "tool_call_id": t.id,
                                 "content": json.dumps({"error": "prune must be the only tool call in a turn; re-issue this call separately."})})
            # Execute only the prune call
            prune_call = next(t for t in m.tool_calls if t.function.name == "prune")
            try:
                a = json.loads(prune_call.function.arguments)
            except (json.JSONDecodeError, TypeError) as e:
                msgs.append({"role": "tool", "tool_call_id": prune_call.id,
                             "content": json.dumps({"error": f"Malformed tool arguments: {e}"})})
                continue
            out = await do_prune(a["docids"], containers, doc_cache)
            msgs.clear()
            msgs.append(initial_msg)
            msgs.append({"role": "assistant", "content": "I'll prune the context to focus on the most relevant documents."})
            msgs.append({"role": "user", "content": out})
            print(f"  [prune] Kept {len(a['docids'])} docs, context reset")
            continue
        async def _exec(t):
            try:
                a = json.loads(t.function.arguments)
            except (json.JSONDecodeError, TypeError) as e:
                return t, json.dumps({"error": f"Malformed tool arguments: {e}"}), False
            if t.function.name == "search":
                out = await do_search(a["query"], containers)
                try:
                    for h in json.loads(out):
                        if h.get("snippet"): doc_cache[h["docid"]] = h["snippet"]
                except: pass
            elif t.function.name == "get_document":
                out = await do_get_doc(a["docid"], containers)
                try:
                    d = json.loads(out)
                    if d.get("text"): doc_cache[d["docid"]] = d["text"]
                except: pass
            elif t.function.name == "prune":
                out = await do_prune(a["docids"], containers, doc_cache)
                return t, out, True  # signal prune
            else:
                out = json.dumps({"error": f"Unknown tool: {t.function.name}"})
            return t, out, False
        print(f"  Executing {len(m.tool_calls)} tool calls in parallel...")
        results = await asyncio.gather(*(_exec(t) for t in m.tool_calls))
        pruned = False
        for t, out, is_prune in results:
            if is_prune:
                msgs.clear()
                msgs.append(initial_msg)
                msgs.append({"role": "assistant", "content": "I'll prune the context to focus on the most relevant documents."})
                msgs.append({"role": "user", "content": out})
                try:
                    prune_args = json.loads(t.function.arguments)
                    print(f"  [prune] Kept {len(prune_args['docids'])} docs, context reset")
                except (json.JSONDecodeError, TypeError, KeyError):
                    print(f"  [prune] context reset")
                pruned = True
                break
        if pruned:
            continue
        for t, out, _ in results:
            msgs.append({"role": "tool", "tool_call_id": t.id, "content": out})
            try:
                a = json.loads(t.function.arguments)
                print(f"  [{t.function.name}] {list(a.values())[0][:80] if isinstance(list(a.values())[0], str) else '...'}")
            except (json.JSONDecodeError, TypeError):
                print(f"  [{t.function.name}] (malformed args)")
        token_est = count_tokens(msgs)
        msgs.append({"role": "user", "content": f"Token usage: {token_est} / {CONTEXT_LIMIT}"})
        print(f"  Token usage: {token_est} / {CONTEXT_LIMIT}")
    elapsed = round(time.perf_counter() - t0, 2)
    return {"question_id": qid, "query": query, "answer": "", "ground_truth": q_obj.get("answer",""),
            "model": llm_cfg["llm_model"], "rounds": 50, "elapsed_seconds": elapsed, "tool_calls": tc}

async def main():
    questions = json.loads(Path(cfg["paths"]["questions_path"]).read_text())
    use_rbac_auth = cosmos_cfg.get("use_rbac_auth", False)
    credential = AsyncAzureCliCredential() if use_rbac_auth else cosmos_cfg["key"]
    cosmos = CosmosClient(cosmos_cfg["uri"], credential=credential)
    db = cosmos.get_database_client(cosmos_cfg["database_name"])
    containers = {s["id"]: db.get_container_client(s["container_name"]) for s in sources}

    results = []
    for q in tqdm(questions):
        results.append(await process_question(q, containers))

    out = Path(cfg["paths"]["output_root"]) / "standard" / f"results_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {len(results)} results to {out}")
    await cosmos.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_dynamic.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    llm_cfg, embed_cfg, cosmos_cfg = cfg["llm"], cfg["embedding"], cfg["cosmos"]
    sources = cosmos_cfg["sources"]
    _source_cfg = {s["id"]: s["retrieval"] for s in sources}
    _source_embed = {s["id"]: s["embedding_field"] for s in sources}
    _source_ft = {s["id"]: s["retrieval"]["fulltext_fields"] for s in sources}
    _all_embed = set(_source_embed.values())
    MAX_RETRIES, RERANK_MUL = int(llm_cfg["max_retries"]), cfg["ranker"]["rerank_multiplier"]
    PRUNE_K = cfg.get("prune_k", 20)
    CONTEXT_LIMIT = llm_cfg.get("context_limit", 270000)

    # Clients
    tp = get_bearer_token_provider(AzureCliCredential(), llm_cfg["token_scope"]) if llm_cfg["use_rbac_auth"] else None
    llm = AsyncAzureOpenAI(api_version=llm_cfg["api_version"], azure_endpoint=llm_cfg["llm_endpoint"],
        **({"azure_ad_token_provider": tp} if tp else {"api_key": llm_cfg["llm_api_key"]}))
    embed_tp = get_bearer_token_provider(AzureCliCredential(), embed_cfg["token_scope"]) if embed_cfg.get("use_rbac_auth") else None
    embed_client = AsyncAzureOpenAI(api_version=embed_cfg["api_version"], azure_endpoint=embed_cfg["embed_endpoint"],
        **({"azure_ad_token_provider": embed_tp} if embed_tp else {"api_key": embed_cfg["embed_api_key"]}))

    # Ranker
    rcfg = cfg["ranker"]
    USE_RANKER = rcfg["use_ranker"]
    if USE_RANKER:
        _r_url = f"https://{rcfg['account_name']}.{rcfg['region']}.{rcfg['url_suffix']}"
        _r_bs, _r_mr = rcfg["batch_size"], rcfg["max_retries"]
        if rcfg["read_token_from_path"]:
            _r_tok = Path(rcfg["access_token_path"]).read_text().strip()
        else:
            _ranker_tenant = str(rcfg.get("tenant_id") or "").strip()
            _ranker_cred = AzureCliCredential(tenant_id=_ranker_tenant) if _ranker_tenant else AzureCliCredential()
            _r_tok = _ranker_cred.get_token(rcfg["token_scope"]).token
        _r_hdr = {"Authorization": f"Bearer {_r_tok}", "Content-Type": "application/json"}
        _r_http = httpx.AsyncClient(timeout=120)

    QUERY_TEMPLATE = """You are a deep research agent. Answer the question by using the search, get_document, and prune tools. Search multiple times with diverse queries. Do not give up early.

Available tools:
- search(query): Search the knowledge base. Returns top results with docid and snippet.
- get_document(docid): Get full document text by docid.
- prune(docids): Keep only the specified documents (up to """ + str(PRUNE_K) + """) and discard the rest from context. Use this when context is getting large to focus on the most relevant documents.

Question: {question}

Format: Explanation: ... Exact Answer: ... Confidence: N%"""

    asyncio.run(main())
