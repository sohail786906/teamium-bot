from __future__ import annotations

from typing import Any

from supabase import create_client, Client

from rag.chunker import DocumentChunk
from utils.config import settings
from utils.logger import logger

BATCH_SIZE = 50


class VectorStore:
    """
    Supabase pgvector store.
    Tables: kb_documents → kb_chunks → kb_embeddings (cascade delete).
    """

    def __init__(self) -> None:
        self._client: Client = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_KEY)
        self._cached_chunk_count: int | None = None
        self._cached_doc_count: int | None = None

    def get_chunk_count(self) -> int:
        if self._cached_chunk_count is not None:
            return self._cached_chunk_count
        try:
            resp = self._client.table("kb_chunks").select("chunk_id", count="exact").execute()
            self._cached_chunk_count = resp.count or 0
        except Exception:
            self._cached_chunk_count = 0
        return self._cached_chunk_count

    def get_document_count(self) -> int:
        if self._cached_doc_count is not None:
            return self._cached_doc_count
        try:
            resp = self._client.table("kb_documents").select("document_id", count="exact").execute()
            self._cached_doc_count = resp.count or 0
        except Exception:
            self._cached_doc_count = 0
        return self._cached_doc_count

    def _reset_cache(self) -> None:
        self._cached_chunk_count = None
        self._cached_doc_count = None

    def get_indexed_documents(self) -> dict[str, str]:
        """Returns {file_name: sha256} for all documents."""
        resp = self._client.table("kb_documents").select("file_name, sha256").execute()
        return {row["file_name"]: row["sha256"] or "" for row in (resp.data or [])}

    def list_documents(self) -> list[dict]:
        """Returns all document metadata ordered by upload time."""
        resp = (
            self._client.table("kb_documents")
            .select("document_id, file_name, title, uploaded_at")
            .order("uploaded_at", desc=True)
            .execute()
        )
        return resp.data or []

    def remove_document(self, file_name: str) -> None:
        """Delete document — ON DELETE CASCADE removes chunks and embeddings."""
        self._client.table("kb_documents").delete().eq("file_name", file_name).execute()
        self._reset_cache()

    def index_document(
        self,
        *,
        file_name: str,
        title: str,
        sha256: str,
        chunks: list[DocumentChunk],
        embedding_engine: Any,
    ) -> None:
        """Insert document, chunks, and embeddings."""
        doc_resp = (
            self._client.table("kb_documents")
            .insert({"file_name": file_name, "title": title, "sha256": sha256})
            .execute()
        )
        doc_id = doc_resp.data[0]["document_id"]

        if not chunks:
            return

        chunk_rows = [
            {"document_id": doc_id, "chunk_text": c.text, "chunk_index": idx}
            for idx, c in enumerate(chunks)
        ]

        all_chunk_ids: list[int] = []
        for i in range(0, len(chunk_rows), BATCH_SIZE):
            batch = chunk_rows[i : i + BATCH_SIZE]
            resp = self._client.table("kb_chunks").insert(batch).execute()
            all_chunk_ids.extend(r["chunk_id"] for r in resp.data)

        texts = [c.text for c in chunks]
        vectors = embedding_engine.encode(texts)

        embedding_rows = [
            {
                "chunk_id": all_chunk_ids[idx],
                "embedding": vectors[idx].tolist(),
                "embedding_model": settings.EMBEDDING_MODEL,
            }
            for idx in range(len(all_chunk_ids))
        ]

        for i in range(0, len(embedding_rows), BATCH_SIZE):
            batch = embedding_rows[i : i + BATCH_SIZE]
            self._client.table("kb_embeddings").insert(batch).execute()

        self._reset_cache()
        logger.info("Indexed '%s': %d chunks", file_name, len(chunks))

    def semantic_search(self, *, embedding_engine: Any, query: str, limit: int) -> list[dict]:
        """Vector similarity search via match_chunks RPC."""
        query_text = query.strip()
        if not query_text:
            return []

        query_vector = embedding_engine.encode([query_text])[0].tolist()

        resp = self._client.rpc(
            "match_chunks",
            {
                "query_embedding": query_vector,
                "match_threshold": 0.0,
                "match_count": limit,
            },
        ).execute()

        return [
            {
                "chunk_id": row["chunk_id"],
                "chunk_text": row["chunk_text"],
                "source": row["file_name"],
                "title": row.get("title", ""),
                "similarity": row["similarity"],
            }
            for row in (resp.data or [])
        ]
