"""Shared fulltext search helpers and stopwords.

This module has no heavy dependencies (no CONFIG, no dynamic_retriever) so it
can be imported safely from both cosmos_retriever.py and dynamic_retriever.py.
"""

import asyncio
import logging
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, TypedDict

from utils.candidates import candidate_identity, candidate_projection

logger = logging.getLogger(__name__)

_FULLTEXT_FIELD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*(?:\[\])?")
_FULLTEXT_SCORE_MAX_TERMS = 30


def is_fulltext_field_path(field: str) -> bool:
    return bool(_FULLTEXT_FIELD_RE.fullmatch(field))


def fulltext_field_from_index_path(path: str) -> str:
    segments = path.strip("/").split("/")
    if segments[-1] == "?":
        segments.pop()
    array_suffix = bool(segments and segments[-1] == "[]")
    if array_suffix:
        segments.pop()
    field = ".".join(segments) + ("[]" if array_suffix else "")
    if not is_fulltext_field_path(field):
        raise ValueError(f"Unsupported full-text index path: {path!r}")
    return field

# Comprehensive BM25 stopwords list
STOPWORDS = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "a's", "able", "about", "above", "according", "accordingly", "across", "actually", "after", "afterwards", "again", "against", "ain't", "all", "allow", "allows", "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "an", "and", "another", "any", "anybody", "anyhow", "anyone", "anything", "anyway", "anyways", "anywhere", "apart", "appear", "appreciate", "appropriate", "are", "aren't", "around", "as", "aside", "ask", "asking", "associated", "at", "available", "away", "awfully", "b", "be", "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "behind", "being", "believe", "below", "beside", "besides", "best", "better", "between", "beyond", "both", "brief", "but", "by", "c", "c'mon", "c's", "came", "can", "can't", "cannot", "cant", "cause", "causes", "certain", "certainly", "changes", "clearly", "co", "com", "come", "comes", "concerning", "consequently", "consider", "considering", "contain", "containing", "contains", "corresponding", "could", "couldn't", "course", "currently", "d", "definitely", "described", "despite", "did", "didn't", "different", "do", "does", "doesn't", "doing", "don", "don't", "done", "down", "downwards", "during", "e", "each", "edu", "eg", "eight", "either", "else", "elsewhere", "enough", "entirely", "especially", "et", "etc", "even", "ever", "every", "everybody", "everyone", "everything", "everywhere", "ex", "exactly", "example", "except", "f", "far", "few", "fifth", "first", "five", "followed", "following", "follows", "for", "former", "formerly", "forth", "four", "from", "further", "furthermore", "g", "get", "gets", "getting", "given", "gives", "go", "goes", "going", "gone", "got", "gotten", "greetings", "h", "had", "hadn't", "happens", "hardly", "has", "hasn't", "have", "haven't", "having", "he", "he's", "hello", "help", "hence", "her", "here", "here's", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "hi", "him", "himself", "his", "hither", "hopefully", "how", "howbeit", "however", "i", "i'd", "i'll", "i'm", "i've", "ie", "if", "ignored", "immediate", "in", "inasmuch", "inc", "indeed", "indicate", "indicated", "indicates", "inner", "insofar", "instead", "into", "inward", "is", "isn't", "it", "it'd", "it'll", "it's", "its", "itself", "j", "just", "k", "keep", "keeps", "kept", "know", "known", "knows", "l", "last", "lately", "later", "latter", "latterly", "least", "less", "lest", "let", "let's", "like", "liked", "likely", "little", "ll", "look", "looking", "looks", "ltd", "m", "mainly", "make", "many", "may", "maybe", "me", "mean", "meanwhile", "merely", "might", "more", "moreover", "most", "mostly", "mr", "mrs", "ms", "much", "must", "my", "myself", "n", "name", "namely", "nd", "near", "nearly", "necessary", "need", "needs", "neither", "never", "nevertheless", "new", "next", "nine", "no", "nobody", "non", "none", "noone", "nor", "normally", "not", "nothing", "novel", "now", "nowhere", "o", "obviously", "of", "off", "often", "oh", "ok", "okay", "old", "on", "once", "one", "ones", "only", "onto", "or", "other", "others", "otherwise", "ought", "our", "ours", "ourselves", "out", "outside", "over", "overall", "own", "p", "particular", "particularly", "per", "perhaps", "placed", "please", "plus", "possible", "presumably", "probably", "provides", "q", "que", "quite", "qv", "r", "rather", "rd", "re", "really", "reasonably", "regarding", "regardless", "regards", "relatively", "respectively", "right", "s", "said", "same", "saw", "say", "saying", "says", "second", "secondly", "see", "seeing", "seem", "seemed", "seeming", "seems", "seen", "self", "selves", "sensible", "sent", "serious", "seriously", "seven", "several", "shall", "she", "should", "shouldn't", "since", "six", "so", "some", "somebody", "somehow", "someone", "something", "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", "specified", "specify", "specifying", "still", "sub", "such", "sup", "sure", "t", "t's", "take", "taken", "tell", "tends", "th", "than", "thank", "thanks", "thanx", "that", "that's", "thats", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "there's", "thereafter", "thereby", "therefore", "therein", "theres", "thereupon", "these", "they", "they'd", "they'll", "they're", "they've", "think", "third", "this", "thorough", "thoroughly", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "took", "toward", "towards", "tried", "tries", "truly", "try", "trying", "twice", "two", "u", "un", "under", "unfortunately", "unless", "unlikely", "until", "unto", "up", "upon", "us", "use", "used", "useful", "uses", "using", "usually", "v", "value", "various", "ve", "very", "via", "viz", "vs", "w", "want", "wants", "was", "wasn't", "way", "we", "we'd", "we'll", "we're", "we've", "welcome", "well", "went", "were", "weren't", "what", "what's", "whatever", "when", "whence", "whenever", "where", "where's", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "who's", "whoever", "whole", "whom", "whose", "why", "will", "willing", "wish", "with", "within", "without", "won't", "wonder", "would", "wouldn't", "x", "y", "yes", "yet", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves", "z", "zero"}


class FullTextOptions(TypedDict):
    fulltext_query_mode: str
    fulltext_match_filter: bool


def fulltext_search_options(config: Mapping[str, Any]) -> FullTextOptions:
    mode = config.get("fulltext_query_mode", "per_field")
    match_filter = config.get("fulltext_match_filter", False)
    if mode not in ("per_field", "consolidated"):
        raise ValueError("retrieval.fulltext_query_mode must be 'per_field' or 'consolidated'")
    if not isinstance(match_filter, bool):
        raise ValueError("retrieval.fulltext_match_filter must be a boolean")
    return {"fulltext_query_mode": mode, "fulltext_match_filter": match_filter}


@dataclass(frozen=True)
class FullTextQueryMetrics:
    fields: tuple[str, ...]
    elapsed_seconds: float
    request_charge: float | None
    response_count: int
    result_count: int
    succeeded: bool


def fulltext_terms(query: str) -> list[str]:
    return [t for t in re.findall(r"\w+", query) if t.lower() not in STOPWORDS and len(t) > 1]


async def _execute_fulltext_query(
    container, fields: list[str], sql: str, query: str, top_k: int, log_fn,
    metrics_fn: Callable[[FullTextQueryMetrics], None] | None,
) -> list[dict]:
    if log_fn is not None:
        log_fn(sql, top_k, query)
    charges: list[float] = []
    response_count = 0

    def response_hook(headers, _body):
        nonlocal response_count
        response_count += 1
        charge = headers.get("x-ms-request-charge")
        if charge is not None:
            charges.append(float(charge))

    started = time.perf_counter()
    items: list[dict] = []
    succeeded = False
    try:
        async for item in container.query_items(
            query=sql, parameters=[], response_hook=response_hook,
        ):
            items.append(item)
        succeeded = True
        return items
    except asyncio.CancelledError:
        raise
    except Exception as error:
        logger.error("Full-text query failed for fields %s (%s): %s", fields, type(error).__name__, str(error)[:300])
        raise
    finally:
        metrics = FullTextQueryMetrics(
            tuple(fields), time.perf_counter() - started,
            sum(charges) if charges and len(charges) == response_count else None,
            response_count, len(items), succeeded,
        )
        logger.info(
            "fulltext query: fields=%s duration=%.3fs ru=%s responses=%d results=%d succeeded=%s",
            fields, metrics.elapsed_seconds, metrics.request_charge,
            response_count, len(items), succeeded,
        )
        if metrics_fn is not None:
            metrics_fn(metrics)


async def fulltext_search_single_field(
    container, field: str, query: str, top_k: int, log_fn=None, *,
    partition_key_path: str | None = None,
    fulltext_match_filter: bool = False,
    metrics_fn: Callable[[FullTextQueryMetrics], None] | None = None,
) -> list[dict]:
    """Rank scalar text, or return matching documents for a terminal string array."""
    fulltext_search_options({"fulltext_match_filter": fulltext_match_filter})
    if not is_fulltext_field_path(field):
        raise ValueError(f"Invalid full-text field path: {field!r}")
    if top_k <= 0:
        return []
    terms = fulltext_terms(query)
    if not terms:
        return []
    chunks = [terms[i:i + 5] for i in range(0, len(terms), 5)]
    field_expr = f"c.{field.removesuffix('[]')}"
    score_exprs = []
    for term_chunk in chunks:
        args = ", ".join(f'"{term}"' for term in term_chunk)
        score_exprs.append(f"FullTextScore({field_expr}, {args})")
    if not score_exprs:
        return []
    if len(score_exprs) == 1:
        order = f"ORDER BY RANK {score_exprs[0]}"
    else:
        order = f"ORDER BY RANK RRF({', '.join(score_exprs)})"
    projection = candidate_projection(partition_key_path) if partition_key_path is not None else "*"
    if field.endswith("[]"):
        args = ", ".join(f'"{term}"' for term in terms)
        sql = (
            f"SELECT TOP {top_k} {projection} FROM c WHERE EXISTS("
            f"SELECT VALUE element FROM element IN {field_expr} "
            f"WHERE FullTextContainsAny(element, {args}))"
        )
    else:
        args = ", ".join(f'"{term}"' for term in terms)
        where = f"WHERE FullTextContainsAny({field_expr}, {args}) " if fulltext_match_filter else ""
        sql = f"SELECT TOP {top_k} {projection} FROM c {where}{order}"
    return await _execute_fulltext_query(
        container, [field], sql, query, top_k, log_fn, metrics_fn,
    )


async def _fulltext_search_scalars(
    container, fields: list[str], query: str, top_k: int, log_fn, *,
    partition_key_path: str | None, fulltext_match_filter: bool,
    metrics_fn: Callable[[FullTextQueryMetrics], None] | None,
) -> list[dict]:
    terms = fulltext_terms(query)
    args = ", ".join(f'"{term}"' for term in terms)
    # Cosmos rejects more than 30 terms in a FullTextScore call. Larger chunks
    # than the legacy five-term groups reduce internal ranking operations.
    chunks = [terms[i:i + _FULLTEXT_SCORE_MAX_TERMS] for i in range(0, len(terms), _FULLTEXT_SCORE_MAX_TERMS)]
    score_args = [", ".join(f'"{term}"' for term in chunk) for chunk in chunks]
    scores = [f"FullTextScore(c.{field}, {chunk})" for field in fields for chunk in score_args]
    order = scores[0] if len(scores) == 1 else f"RRF({', '.join(scores)})"
    predicates = [f"FullTextContainsAny(c.{field}, {args})" for field in fields]
    where = f"WHERE ({' OR '.join(predicates)}) " if fulltext_match_filter else ""
    projection = candidate_projection(partition_key_path) if partition_key_path is not None else "*"
    sql = f"SELECT TOP {top_k} {projection} FROM c {where}ORDER BY RANK {order}"
    return await _execute_fulltext_query(
        container, fields, sql, query, top_k, log_fn, metrics_fn,
    )


async def fulltext_search(
    container, fields: list[str], query: str, top_k: int, log_fn=None, *,
    partition_key_path: str | None = None,
    fulltext_query_mode: str = "per_field",
    fulltext_match_filter: bool = False,
    metrics_fn: Callable[[FullTextQueryMetrics], None] | None = None,
) -> list[dict]:
    """Fuse per-field lists, or a consolidated scalar list plus native-array lists.

    Consolidation intentionally changes ranking: scalar fields use server RRF,
    and that list and each unranked array list have equal weight in client RRF.
    Legacy per-field ranking remains the default for existing configurations.
    """
    fulltext_search_options({
        "fulltext_query_mode": fulltext_query_mode, "fulltext_match_filter": fulltext_match_filter,
    })
    for field in fields:
        if not is_fulltext_field_path(field):
            raise ValueError(f"Invalid full-text field path: {field!r}")
    fields = list(dict.fromkeys(fields))
    if top_k <= 0 or not fields:
        return []
    terms = fulltext_terms(query)
    if not terms:
        return []
    tasks = []
    if fulltext_query_mode == "consolidated":
        scalars = [field for field in fields if not field.endswith("[]")]
        if scalars:
            tasks.append(_fulltext_search_scalars(
                container, scalars, query, top_k, log_fn,
                partition_key_path=partition_key_path, fulltext_match_filter=fulltext_match_filter,
                metrics_fn=metrics_fn,
            ))
        fields = [field for field in fields if field.endswith("[]")]
    tasks.extend(
        fulltext_search_single_field(
            container, f, query, top_k, log_fn=log_fn,
            partition_key_path=partition_key_path, fulltext_match_filter=fulltext_match_filter,
            metrics_fn=metrics_fn,
        ) for f in fields)
    running = [asyncio.create_task(task) for task in tasks]
    try:
        per_field_results = await asyncio.gather(*running)
    except BaseException:
        for task in running:
            task.cancel()
        await asyncio.gather(*running, return_exceptions=True)
        raise
    rrf_k = 60
    doc_scores: dict[tuple[str, str], float] = {}
    doc_map: dict[tuple[str, str], dict] = {}
    for field_items in per_field_results:
        for rank, item in enumerate(field_items):
            doc_id = candidate_identity(item)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank + 1)
            if doc_id not in doc_map:
                doc_map[doc_id] = item
    sorted_ids = sorted(doc_scores, key=lambda did: doc_scores[did], reverse=True)[:top_k]
    return [doc_map[did] for did in sorted_ids]
