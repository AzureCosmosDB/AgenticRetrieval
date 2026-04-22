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
from utils.fulltext import fulltext_search
from utils.ranker import rerank_documents

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

async def rerank(query, docs, top_k):
    if not USE_RANKER or not docs: return docs[:top_k]
    indices = await rerank_documents(_r_http, _r_url, _r_tok, query, docs, top_k, _r_bs, _r_mr)
    if indices is None:
        return docs[:top_k]
    return [docs[i] for i in indices]

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
            tasks.append(fulltext_search(containers[sid], fields, query, _source_cfg[sid]["fulltext_search_k"]*RERANK_MUL))
    results = await asyncio.gather(*tasks)
    seen, all_d = set(), []
    for dl in results:
        for d in dl:
            did = d.get("id","")
            if did not in seen: seen.add(did); all_d.append(d)
    total_k = sum(r["search_k"]+r["fulltext_search_k"] for r in _source_cfg.values())
    texts = [fmt(d) for d in all_d]
    ranked_texts = await rerank(query, texts, total_k)
    # Map ranked texts back to source docs by index
    text_to_idx = {id(t): i for i, t in enumerate(texts)}
    ranked_indices = [text_to_idx[id(t)] for t in ranked_texts]
    return json.dumps([{"docid": all_d[i].get("id", ""), "snippet": ranked_texts[j][:2000]} for j, i in enumerate(ranked_indices)])

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
                except asyncio.CancelledError: raise
                except Exception as e:
                    print(f"  [search] failed to parse/cache search results: {e}")
            elif t.function.name == "get_document":
                out = await do_get_doc(a["docid"], containers)
                try:
                    d = json.loads(out)
                    if d.get("text"): doc_cache[d["docid"]] = d["text"]
                except asyncio.CancelledError: raise
                except Exception as e:
                    print(f"  [get_document] failed to parse/cache doc result for {a.get('docid','?')}: {e}")
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
    if args.max_questions is not None:
        questions = questions[:args.max_questions]
    use_rbac_auth = cosmos_cfg.get("use_rbac_auth", False)
    credential = AsyncAzureCliCredential() if use_rbac_auth else None
    cosmos = CosmosClient(cosmos_cfg["uri"], credential=credential or cosmos_cfg["key"])
    db = cosmos.get_database_client(cosmos_cfg["database_name"])
    containers = {s["id"]: db.get_container_client(s["container_name"]) for s in sources}

    try:
        results = []
        for q in tqdm(questions):
            results.append(await process_question(q, containers))

        out = Path(cfg["paths"]["output_root"]) / "standard" / f"results_{time.strftime('%Y%m%d_%H%M%S')}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2))
        print(f"\nSaved {len(results)} results to {out}")
    finally:
        await cosmos.close()
        if credential is not None:
            await credential.close()
        await llm.close()
        await embed_client.close()
        if USE_RANKER:
            await _r_http.aclose()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_dynamic.yaml")
    parser.add_argument("--max-questions", type=int, default=None, help="Only answer the first N questions")
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
