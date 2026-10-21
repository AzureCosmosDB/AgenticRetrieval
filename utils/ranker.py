"""Shared semantic ranker helper.

This module has no heavy dependencies (no CONFIG, no dynamic_retriever) so it
can be imported safely from both cosmos_retriever.py and dynamic_retriever.py.
"""

import asyncio
from collections.abc import Callable

import httpx


class RankerUnavailableError(RuntimeError):
    """Raised when ranker.fail_on_error is set and the semantic ranker cannot be used."""


async def verify_ranker_token(
    access_token: str | Callable[[], str], fail_on_error: bool,
) -> str | None:
    """Acquire a ranker token now so authentication problems surface at startup.

    Returns None on success. On failure raises RankerUnavailableError when
    fail_on_error is set; otherwise returns the problem description so the
    caller can log it and continue without a working ranker.
    """
    if not callable(access_token):
        return None
    try:
        await asyncio.to_thread(access_token)
    except Exception as error:
        message = (
            f"Semantic ranker is unavailable: could not acquire an access token "
            f"({type(error).__name__}: {str(error)[:300]}). Check the ranker token scope and "
            "tenant and run 'az login'; set ranker.fail_on_error: false to continue with "
            "unranked results instead."
        )
        if fail_on_error:
            raise RankerUnavailableError(message) from error
        return message
    return None


async def rerank_documents(
    http_client: httpx.AsyncClient,
    url: str,
    access_token: str | Callable[[], str],
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

    # The semantic ranker silently returns Scores=[] when any document in the
    # request is an empty string. Diagnose and abort cleanly so callers get a
    # useful signal instead of 5 retries.
    empty_idxs = [i for i, d in enumerate(documents) if not (isinstance(d, str) and d.strip())]
    if empty_idxs:
        print(
            f"  [rerank_documents] aborting: {len(empty_idxs)}/{len(documents)} documents are empty "
            f"(first empty idx={empty_idxs[0]}). The ranker rejects payloads containing empty strings."
        )
        return None

    body = {
        "query": query,
        "documents": documents,
        "return_documents": False,
        "top_k": top_k,
        "batch_size": batch_size,
    }
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }

    transient_empty = 0
    last_response_preview: str | None = None
    last_exception: Exception | None = None

    for attempt in range(max_retries):
        try:
            token = (
                await asyncio.to_thread(access_token)
                if callable(access_token) else access_token
            )
            headers["Authorization"] = f"Bearer {token}"
            resp = await http_client.post(url, headers=headers, json=body)
            if resp.status_code in (401, 403):
                print(
                    f"  [rerank_documents] authorization failed (HTTP {resp.status_code}); "
                    "check the token scope, tenant, and ranker access permissions."
                )
                return None
            if resp.status_code in (429, 502, 503) and attempt + 1 < max_retries:
                await asyncio.sleep(min(2 ** attempt, 8))
                continue
            resp.raise_for_status()
            response_json = resp.json()
            scores = response_json.get("Scores", [])
            if len(scores) != top_k:
                transient_empty += 1
                last_response_preview = str(response_json)[:500]
                if attempt + 1 < max_retries:
                    # Service is flaky – short backoff and try again silently.
                    await asyncio.sleep(0.5)
                    continue

            return [s["index"] for s in scores[:top_k] if s["index"] < len(documents)]
        except Exception as e:
            last_exception = e
            if attempt + 1 < max_retries:
                await asyncio.sleep(min(2 ** attempt, 8))
                continue
            break

    # All retries exhausted – emit one consolidated diagnostic.
    if last_exception is not None:
        print(f"  [rerank_documents] gave up after {max_retries} attempts: {last_exception}")
    else:
        print(
            f"  [rerank_documents] gave up after {max_retries} transient empty-Scores responses "
            f"(top_k={top_k}, n_docs={len(documents)}, batch_size={batch_size})"
        )
        if last_response_preview:
            print(f"  [rerank_documents]   last response: {last_response_preview}")
    return None
