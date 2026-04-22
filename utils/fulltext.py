"""Shared fulltext search helpers and stopwords.

This module has no heavy dependencies (no CONFIG, no agentic_retriever) so it
can be imported safely from both cosmos_retriever.py and dynamic_retriever.py.
"""

import asyncio
import re

# Comprehensive BM25 stopwords list
STOPWORDS = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "a's", "able", "about", "above", "according", "accordingly", "across", "actually", "after", "afterwards", "again", "against", "ain't", "all", "allow", "allows", "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "an", "and", "another", "any", "anybody", "anyhow", "anyone", "anything", "anyway", "anyways", "anywhere", "apart", "appear", "appreciate", "appropriate", "are", "aren't", "around", "as", "aside", "ask", "asking", "associated", "at", "available", "away", "awfully", "b", "be", "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "behind", "being", "believe", "below", "beside", "besides", "best", "better", "between", "beyond", "both", "brief", "but", "by", "c", "c'mon", "c's", "came", "can", "can't", "cannot", "cant", "cause", "causes", "certain", "certainly", "changes", "clearly", "co", "com", "come", "comes", "concerning", "consequently", "consider", "considering", "contain", "containing", "contains", "corresponding", "could", "couldn't", "course", "currently", "d", "definitely", "described", "despite", "did", "didn't", "different", "do", "does", "doesn't", "doing", "don", "don't", "done", "down", "downwards", "during", "e", "each", "edu", "eg", "eight", "either", "else", "elsewhere", "enough", "entirely", "especially", "et", "etc", "even", "ever", "every", "everybody", "everyone", "everything", "everywhere", "ex", "exactly", "example", "except", "f", "far", "few", "fifth", "first", "five", "followed", "following", "follows", "for", "former", "formerly", "forth", "four", "from", "further", "furthermore", "g", "get", "gets", "getting", "given", "gives", "go", "goes", "going", "gone", "got", "gotten", "greetings", "h", "had", "hadn't", "happens", "hardly", "has", "hasn't", "have", "haven't", "having", "he", "he's", "hello", "help", "hence", "her", "here", "here's", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "hi", "him", "himself", "his", "hither", "hopefully", "how", "howbeit", "however", "i", "i'd", "i'll", "i'm", "i've", "ie", "if", "ignored", "immediate", "in", "inasmuch", "inc", "indeed", "indicate", "indicated", "indicates", "inner", "insofar", "instead", "into", "inward", "is", "isn't", "it", "it'd", "it'll", "it's", "its", "itself", "j", "just", "k", "keep", "keeps", "kept", "know", "known", "knows", "l", "last", "lately", "later", "latter", "latterly", "least", "less", "lest", "let", "let's", "like", "liked", "likely", "little", "ll", "look", "looking", "looks", "ltd", "m", "mainly", "make", "many", "may", "maybe", "me", "mean", "meanwhile", "merely", "might", "more", "moreover", "most", "mostly", "mr", "mrs", "ms", "much", "must", "my", "myself", "n", "name", "namely", "nd", "near", "nearly", "necessary", "need", "needs", "neither", "never", "nevertheless", "new", "next", "nine", "no", "nobody", "non", "none", "noone", "nor", "normally", "not", "nothing", "novel", "now", "nowhere", "o", "obviously", "of", "off", "often", "oh", "ok", "okay", "old", "on", "once", "one", "ones", "only", "onto", "or", "other", "others", "otherwise", "ought", "our", "ours", "ourselves", "out", "outside", "over", "overall", "own", "p", "particular", "particularly", "per", "perhaps", "placed", "please", "plus", "possible", "presumably", "probably", "provides", "q", "que", "quite", "qv", "r", "rather", "rd", "re", "really", "reasonably", "regarding", "regardless", "regards", "relatively", "respectively", "right", "s", "said", "same", "saw", "say", "saying", "says", "second", "secondly", "see", "seeing", "seem", "seemed", "seeming", "seems", "seen", "self", "selves", "sensible", "sent", "serious", "seriously", "seven", "several", "shall", "she", "should", "shouldn't", "since", "six", "so", "some", "somebody", "somehow", "someone", "something", "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", "specified", "specify", "specifying", "still", "sub", "such", "sup", "sure", "t", "t's", "take", "taken", "tell", "tends", "th", "than", "thank", "thanks", "thanx", "that", "that's", "thats", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "there's", "thereafter", "thereby", "therefore", "therein", "theres", "thereupon", "these", "they", "they'd", "they'll", "they're", "they've", "think", "third", "this", "thorough", "thoroughly", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "took", "toward", "towards", "tried", "tries", "truly", "try", "trying", "twice", "two", "u", "un", "under", "unfortunately", "unless", "unlikely", "until", "unto", "up", "upon", "us", "use", "used", "useful", "uses", "using", "usually", "v", "value", "various", "ve", "very", "via", "viz", "vs", "w", "want", "wants", "was", "wasn't", "way", "we", "we'd", "we'll", "we're", "we've", "welcome", "well", "went", "were", "weren't", "what", "what's", "whatever", "when", "whence", "whenever", "where", "where's", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "who's", "whoever", "whole", "whom", "whose", "why", "will", "willing", "wish", "with", "within", "without", "won't", "wonder", "would", "wouldn't", "x", "y", "yes", "yet", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves", "z", "zero"}


async def fulltext_search_single_field(container, field: str, query: str, top_k: int) -> list[dict]:
    """Run a fulltext search on a single field, returning ranked results."""
    if top_k <= 0:
        return []
    terms = [t for t in re.findall(r"\w+", query) if t.lower() not in STOPWORDS and len(t) > 1]
    if not terms:
        return []
    chunks = [terms[i:i + 5] for i in range(0, len(terms), 5)]
    field_expr = f"c.{field}"
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
    sql = f"SELECT TOP {top_k} * FROM c {order}"
    try:
        items = []
        async for item in container.query_items(query=sql, parameters=[]):
            items.append(item)
        return items
    except asyncio.CancelledError:
        raise
    except Exception:
        return []


async def fulltext_search(container, fields: list[str], query: str, top_k: int) -> list[dict]:
    """Multi-field fulltext search with client-side RRF merge."""
    if top_k <= 0 or not fields:
        return []
    terms = [t for t in re.findall(r"\w+", query) if t.lower() not in STOPWORDS and len(t) > 1]
    if not terms:
        return []
    if len(fields) == 1:
        return await fulltext_search_single_field(container, fields[0], query, top_k)
    per_field_results = await asyncio.gather(
        *(fulltext_search_single_field(container, f, query, top_k) for f in fields)
    )
    rrf_k = 60
    doc_scores: dict[str, float] = {}
    doc_map: dict[str, dict] = {}
    for field_items in per_field_results:
        for rank, item in enumerate(field_items):
            doc_id = item.get("id", "")
            if not doc_id:
                continue
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank + 1)
            if doc_id not in doc_map:
                doc_map[doc_id] = item
    sorted_ids = sorted(doc_scores, key=lambda did: doc_scores[did], reverse=True)[:top_k]
    return [doc_map[did] for did in sorted_ids]
