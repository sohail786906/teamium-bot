from __future__ import annotations

from fastapi import APIRouter, HTTPException, Depends, Header, UploadFile, File
from pydantic import BaseModel, Field

from rag.generator import generate_answer
from rag.loader import SUPPORTED_EXTENSIONS
from rag.service import KnowledgeBaseService
from utils.config import settings
from utils.logger import logger

router = APIRouter()
kb_service = KnowledgeBaseService()


def verify_auth(x_api_key: str | None = Header(None)) -> None:
    if not settings.API_SECRET_KEY:
        return
    if x_api_key != settings.API_SECRET_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized.")


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict] = []
    suggestions: list[str] = []


class UploadResponse(BaseModel):
    status: str
    file_name: str
    title: str = ""
    chunks_created: int = 0


@router.get("/health")
def health_check() -> dict:
    return kb_service.health()


@router.post("/upload", response_model=UploadResponse, dependencies=[Depends(verify_auth)])
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="File name is required.")

    ext = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(SUPPORTED_EXTENSIONS)}",
        )

    try:
        content = await file.read()
        result = kb_service.upload_document(file_name=file.filename, content=content)
        return UploadResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Upload failed: %s", e)
        raise HTTPException(status_code=500, detail="Document upload failed.")


@router.get("/documents", dependencies=[Depends(verify_auth)])
def list_documents() -> list[dict]:
    return kb_service.list_documents()


@router.delete("/documents/{file_name}", dependencies=[Depends(verify_auth)])
def delete_document(file_name: str) -> dict:
    try:
        return kb_service.delete_document(file_name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Delete failed: %s", e)
        raise HTTPException(status_code=500, detail="Document deletion failed.")


@router.post("/query", response_model=QueryResponse, dependencies=[Depends(verify_auth)])
def query_knowledge_base(req: QueryRequest) -> QueryResponse:
    try:
        result = generate_answer(service=kb_service, user_query=req.query.strip())
        return QueryResponse(
            answer=result["answer"],
            sources=result.get("sources", []),
            suggestions=result.get("suggestions", []),
        )
    except Exception as e:
        logger.exception("Query failed: %s", e)
        raise HTTPException(status_code=502, detail="Failed to process query.")
