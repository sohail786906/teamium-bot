from __future__ import annotations

import hashlib
import os

from rag.chunker import chunk_document
from rag.embeddings import EmbeddingEngine
from rag.loader import parse_uploaded_file
from rag.vectorstore import VectorStore
from utils.config import settings
from utils.logger import logger


class KnowledgeBaseService:
    """
    Core service for document upload, indexing, and retrieval.
    All data lives in Supabase — no local file storage.
    """

    def __init__(self) -> None:
        self._embedding_engine = EmbeddingEngine(settings.EMBEDDING_MODEL)
        self._vector_store = VectorStore()

    @property
    def embedding_engine(self) -> EmbeddingEngine:
        return self._embedding_engine

    @property
    def vector_store(self) -> VectorStore:
        return self._vector_store

    def health(self) -> dict:
        return {
            "status": "operational",
            "documents": self._vector_store.get_document_count(),
            "chunks": self._vector_store.get_chunk_count(),
            "embedding_model": settings.EMBEDDING_MODEL,
            "llm_model": settings.GROQ_MODEL,
        }

    def upload_document(self, *, file_name: str, content: bytes) -> dict:
        max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
        if len(content) > max_bytes:
            raise ValueError(f"File exceeds {settings.MAX_UPLOAD_MB}MB limit.")

        if len(content) == 0:
            raise ValueError("File is empty.")

        content_hash = hashlib.sha256(content).hexdigest()

        existing = self._vector_store.get_indexed_documents()
        if file_name in existing:
            if existing[file_name] == content_hash:
                return {"status": "unchanged", "file_name": file_name, "title": "", "chunks_created": 0}
            self._vector_store.remove_document(file_name)
            logger.info("Replacing document: %s", file_name)

        document = parse_uploaded_file(file_name, content)
        chunks = chunk_document(document)
        title = self._extract_title(document)

        self._vector_store.index_document(
            file_name=document.file_name,
            title=title,
            sha256=content_hash,
            chunks=chunks,
            embedding_engine=self._embedding_engine,
        )

        return {
            "status": "indexed",
            "file_name": file_name,
            "title": title,
            "chunks_created": len(chunks),
        }

    def delete_document(self, file_name: str) -> dict:
        existing = self._vector_store.get_indexed_documents()
        if file_name not in existing:
            raise ValueError(f"Document not found: {file_name}")

        self._vector_store.remove_document(file_name)
        logger.info("Deleted: %s", file_name)
        return {"status": "deleted", "file_name": file_name}

    def list_documents(self) -> list[dict]:
        return self._vector_store.list_documents()

    @staticmethod
    def _extract_title(document) -> str:
        for block in document.blocks:
            if block.kind == "heading" and block.level in (1, 2):
                return block.text
        name = os.path.splitext(document.file_name)[0]
        return name.replace("_", " ").replace("-", " ").strip().title()
