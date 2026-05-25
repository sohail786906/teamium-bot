from __future__ import annotations

import re
from typing import Any

from openai import OpenAI

from rag.retriever import retrieve_context
from rag.service import KnowledgeBaseService
from utils.config import settings
from utils.logger import logger

SYSTEM_PROMPT = """You are a professional AI knowledge base assistant.

Rules:
1. You can respond normally to greetings and casual conversation.
2. For factual/document-based questions:
   - Use ONLY the provided context.
   - Do NOT use prior knowledge.
   - Give a clear, accurate answer with high readability.
   - Use clean Markdown formatting.
   - Bold only important keywords.
   - Mention source filename(s) used for the answer.
3. If the answer is not in the context:
   - Reply exactly: "The requested information is not available in the knowledge base."
4. Be concise, clear, and accurate.
"""

NOT_FOUND_RESPONSE = "The requested information is not available in the knowledge base."

_GREETING_PATTERN = re.compile(
    r"^\s*(hi|hello|hey|how are you|good morning|good afternoon|good evening)\s*[!.?]*\s*$", re.I
)
_THANKS_PATTERN = re.compile(r"^\s*(thanks|thank you|thx)\s*[!.?]*\s*$", re.I)
_PROCEDURAL_PATTERN = re.compile(
    r"\b(how to|steps to|step by step|create|setup|configure|install|define|process)\b", re.I
)


def _create_llm_client() -> OpenAI:
    return OpenAI(
        api_key=settings.GROQ_API_KEY,
        base_url=settings.GROQ_BASE_URL,
        timeout=settings.LLM_TIMEOUT,
    )


def _handle_greeting(message: str) -> str | None:
    if _GREETING_PATTERN.match(message):
        return "Hello! How can I assist you today?"
    if _THANKS_PATTERN.match(message):
        return "You're welcome. Feel free to ask anything from the knowledge base."
    return None


def _sanitize_source(name: str) -> str:
    cleaned = re.sub(r"[\x00-\x1F\x7F]", "", (name or "")).strip()
    cleaned = re.sub(r"^[^A-Za-z0-9]+", "", cleaned)
    cleaned = re.sub(r"\s+[0-9a-fA-F]{32}(?=\.[A-Za-z0-9]+$)", "", cleaned)
    return cleaned or "unknown"


def _unique_sources(sources: list[dict], limit: int = 8) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for s in sources:
        name = _sanitize_source(str(s.get("source", "")))
        if name not in seen:
            seen.add(name)
            result.append(name)
            if len(result) >= limit:
                break
    return result


def generate_answer(*, service: KnowledgeBaseService, user_query: str) -> dict[str, Any]:
    greeting = _handle_greeting(user_query)
    if greeting is not None:
        return {"answer": greeting, "sources": [], "suggestions": []}

    if service.vector_store.get_chunk_count() == 0:
        return {"answer": NOT_FOUND_RESPONSE, "sources": [], "suggestions": []}

    is_procedural = bool(_PROCEDURAL_PATTERN.search(user_query))

    context, sources = retrieve_context(
        store=service.vector_store,
        embedding_engine=service.embedding_engine,
        query=user_query,
        top_k=max(settings.TOP_K, 12) if is_procedural else None,
        min_score=0.0 if is_procedural else 0.15,
    )

    if not context:
        return {"answer": NOT_FOUND_RESPONSE, "sources": [], "suggestions": []}

    source_names = _unique_sources(sources)
    source_ref = ", ".join(source_names) if source_names else "knowledge base"

    if is_procedural:
        user_prompt = (
            f"CONTEXT:\n{context}\n\n"
            f"USER QUERY:\n{user_query.strip()}\n\n"
            "INSTRUCTIONS:\n"
            "Answer ONLY from CONTEXT.\n"
            "Use numbered steps for procedures.\n"
            "Include key details as bullet points.\n"
            f"Sources used: {source_ref}\n"
            "End with: Sources: <filenames>\n"
        )
    else:
        user_prompt = (
            f"CONTEXT:\n{context}\n\n"
            f"USER QUERY:\n{user_query.strip()}\n\n"
            "INSTRUCTIONS:\n"
            "Answer ONLY from CONTEXT.\n"
            f'If not found, reply exactly: "{NOT_FOUND_RESPONSE}"\n'
            f"Sources used: {source_ref}\n"
            "End with: Sources: <filenames>\n"
        )

    client = _create_llm_client()
    response = client.chat.completions.create(
        model=settings.GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    answer = (response.choices[0].message.content or "").strip() or NOT_FOUND_RESPONSE

    cleaned_sources = [
        {**s, "source": _sanitize_source(str(s.get("source", "")))}
        for s in sources
    ]

    suggestions = []
    if cleaned_sources:
        primary = _unique_sources(cleaned_sources, limit=1)
        top = primary[0] if primary else "the document"
        suggestions = [
            "Can you explain this in simple terms?",
            "What are the key points?",
            f"Tell me more about {top}.",
        ]

    return {"answer": answer, "sources": cleaned_sources, "suggestions": suggestions}
