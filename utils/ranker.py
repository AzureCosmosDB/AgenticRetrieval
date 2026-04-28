"""Shared semantic ranker helper.

This module has no heavy dependencies (no CONFIG, no agentic_retriever) so it
can be imported safely from both cosmos_retriever.py and dynamic_retriever.py.
"""

import asyncio

import httpx


async def rerank_documents(
    http_client: httpx.AsyncClient,
    url: str,
    access_token: str,
    query: str,
    documents: list[str],
    top_k: int,
    batch_size: int = 32,
    max_retries: int = 5,
) -> list[int] | None:
    """Call the semantic ranker and return ranked indices, or None on failure.

    Returns a list of indices into *documents* ordered by ranker score
    (best first), or ``None`` if all attempts failed.
    """
    if not documents or top_k <= 0:
        return list(range(min(len(documents), top_k)))

    body = {
        "query": query,
        "documents": documents,
        "return_documents": False,
        "top_k": top_k,
        "batch_size": batch_size,
    }
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }

    for attempt in range(max_retries):
        try:
            resp = await http_client.post(url, headers=headers, json=body)
            if resp.status_code in (429, 502, 503) and attempt + 1 < max_retries:
                await asyncio.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            scores = resp.json().get("Scores", [])
            return [s["index"] for s in scores[:top_k] if s["index"] < len(documents)]
        except Exception as e:
            print(f"  [rerank_documents] attempt {attempt+1} failed: {e}")
            if attempt + 1 < max_retries:
                await asyncio.sleep(2 ** attempt)
                continue
            return None

    return None
