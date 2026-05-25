# Knowledge Base API

Production backend service for AI-powered document Q&A. Documents are uploaded via API, stored and embedded in Supabase pgvector, and queried using semantic search + Groq LLM.

## Tech Stack

- **Database**: Supabase PostgreSQL + pgvector (vector similarity search)
- **Embeddings**: all-MiniLM-L6-v2 (384 dimensions, runs locally)
- **LLM**: Groq API — Llama 3.3 70B
- **Framework**: FastAPI + Uvicorn
- **Deployment**: Docker

## Project Structure

```
app/
  main.py             Application entry point, middleware
  routes.py           API endpoint definitions

rag/
  service.py          Core business logic (upload, delete, list)
  vectorstore.py      Supabase pgvector operations
  retriever.py        Semantic search + reranking
  generator.py        LLM answer generation
  chunker.py          Document text chunking
  loader.py           File parsing (.md, .docx, .txt, .pdf)
  embeddings.py       Sentence-transformers wrapper

utils/
  config.py           Environment variable management
  logger.py           Structured JSON logging
```

## Setup

### 1. Supabase Database

1. Create a project at [supabase.com](https://supabase.com)
2. Go to **SQL Editor** in the dashboard
3. Paste and run the contents of `setup.sql`

This creates:
- `kb_documents` — document registry
- `kb_chunks` — text segments with FK to documents
- `kb_embeddings` — vector(384) with FK to chunks
- `match_chunks()` — RPC function for similarity search
- Row Level Security policies

### 2. Environment Variables

```bash
cp .env.example .env
```

| Variable | Required | Description |
|----------|----------|-------------|
| `SUPABASE_URL` | Yes | Your Supabase project URL |
| `SUPABASE_SERVICE_KEY` | Yes | Service role key (not anon key) |
| `GROQ_API_KEY` | Yes | API key from console.groq.com |
| `API_SECRET_KEY` | Yes | Secret for endpoint authentication |
| `EMBEDDING_MODEL` | No | Default: sentence-transformers/all-MiniLM-L6-v2 |
| `GROQ_MODEL` | No | Default: llama-3.3-70b-versatile |
| `TOP_K` | No | Default: 5 |
| `MAX_UPLOAD_MB` | No | Default: 25 |
| `LLM_TIMEOUT` | No | Default: 30 (seconds) |

### 3. Run Locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 4. Docker Deployment

```bash
docker build -t knowledge-base-api .
docker run -d --env-file .env -p 8000:8000 knowledge-base-api
```

## API Reference

Base path: `/api/v1`

All endpoints (except `/health`) require the `X-API-Key` header.

### Health Check

```
GET /api/v1/health
```

Returns document count, chunk count, model info.

### Upload Document

```
POST /api/v1/upload
Header: X-API-Key: <your-secret-key>
Body: multipart/form-data with field "file"
```

Accepts `.md`, `.docx`, `.txt`, `.pdf`. Parses, chunks, embeds, and stores in database. If the same file name exists and content changed, it replaces the old version.

**Response:**
```json
{
  "status": "indexed",
  "file_name": "guide.md",
  "title": "User Guide",
  "chunks_created": 12
}
```

### List Documents

```
GET /api/v1/documents
Header: X-API-Key: <your-secret-key>
```

Returns all indexed documents with metadata.

### Delete Document

```
DELETE /api/v1/documents/{file_name}
Header: X-API-Key: <your-secret-key>
```

Removes document and all associated chunks/embeddings from database.

### Query Knowledge Base

```
POST /api/v1/query
Header: X-API-Key: <your-secret-key>
Content-Type: application/json

{"query": "How to add a project phase?"}
```

**Response:**
```json
{
  "answer": "To add a project phase...",
  "sources": [{"chunk_id": 42, "source": "guide.md", "score": 0.87}],
  "suggestions": ["Can you explain this in simple terms?"]
}
```

## How It Works

1. **Upload** — File is parsed into text blocks, split into ~900 char chunks preserving paragraph boundaries
2. **Embed** — Each chunk is encoded into a 384-dim vector using all-MiniLM-L6-v2
3. **Store** — Document metadata, chunks, and embeddings saved to Supabase (3 tables)
4. **Query** — User query is embedded, pgvector finds similar chunks, results are reranked by keyword overlap
5. **Generate** — Top chunks + query sent to Groq LLM, which answers strictly from context
