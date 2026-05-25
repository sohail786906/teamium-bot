from __future__ import annotations

import re

from rag.vectorstore import VectorStore
from rag.embeddings import EmbeddingEngine
from utils.config import settings

_STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "to", "of", "in",
    "on", "for", "and", "or", "with", "that", "this", "it", "as", "at",
    "by", "from", "what", "which", "who", "when", "where", "why", "how",
    "can", "could", "should", "would",
}


def _extract_keywords(text: str) -> set[str]:
    tokens = re.findall(r"[a-zA-Z0-9_]{3,}", text.lower())
    return {t for t in tokens if t not in _STOP_WORDS}


def _rerank(query: str, results: list[dict]) -> list[dict]:
    query_keywords = _extract_keywords(query)
    if not query_keywords:
        return results

    def score(item: dict) -> float:
        vector_score = item["similarity"]
        chunk_keywords = _extract_keywords(item["chunk_text"])
        if not chunk_keywords:
            return vector_score
        overlap = len(query_keywords & chunk_keywords) / max(1, len(query_keywords))
        return (vector_score * 0.85) + (overlap * 0.15)

    return sorted(results, key=score, reverse=True)


def _diversify(results: list[dict], *, limit: int, max_per_source: int = 3) -> list[dict]:
    selected: list[dict] = []
    counts: dict[str, int] = {}

    for item in results:
        source = item["source"]
        if counts.get(source, 0) >= max_per_source:
            continue
        selected.append(item)
        counts[source] = counts.get(source, 0) + 1
        if len(selected) >= limit:
            break

    if len(selected) < limit:
        for item in results:
            if item not in selected:
                selected.append(item)
                if len(selected) >= limit:
                    break

    return selected


def _build_context(results: list[dict], *, max_chars: int) -> tuple[str, list[dict]]:
    parts: list[str] = []
    sources_used: list[dict] = []
    total = 0

    for item in results:
        snippet = f"[Source: {item['source']}]\n{item['chunk_text']}"
        if parts and (total + len(snippet) + 6) > max_chars:
            break
        parts.append(snippet)
        sources_used.append({
            "chunk_id": item["chunk_id"],
            "source": item["source"],
            "score": item["similarity"],
        })
        total += len(snippet) + 6

    return "\n\n---\n\n".join(parts).strip(), sources_used


def retrieve_context(
    *,
    store: VectorStore,
    embedding_engine: EmbeddingEngine,
    query: str,
    top_k: int | None = None,
    min_score: float = 0.15,
) -> tuple[str, list[dict]]:
    k = top_k or settings.TOP_K
    search_limit = k * 3

    results = store.semantic_search(embedding_engine=embedding_engine, query=query, limit=search_limit)
    if not results:
        return "", []

    filtered = [r for r in results if r["similarity"] >= min_score]
    if filtered:
        results = filtered

    results = _rerank(query, results)
    results = _diversify(results, limit=k, max_per_source=3)
    return _build_context(results, max_chars=settings.MAX_CONTEXT_CHARS)
