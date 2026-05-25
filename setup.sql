-- ============================================================
-- Knowledge Base API — Supabase Database Setup
-- Run this in Supabase SQL Editor (Dashboard > SQL Editor)
-- ============================================================

-- Enable pgvector extension
create extension if not exists vector;

-- ============================================================
-- kb_documents: Master document registry
-- ============================================================
create table if not exists kb_documents (
    document_id serial primary key,
    file_name text not null unique,
    title text not null default '',
    sha256 text not null default '',
    uploaded_at timestamp with time zone default now()
);

-- ============================================================
-- kb_chunks: Document segments for retrieval
-- ============================================================
create table if not exists kb_chunks (
    chunk_id serial primary key,
    document_id int not null references kb_documents(document_id) on delete cascade,
    chunk_text text not null,
    chunk_index int not null default 0,
    created_at timestamp with time zone default now()
);

create index if not exists idx_kb_chunks_document_id on kb_chunks(document_id);

-- ============================================================
-- kb_embeddings: Vector embeddings for similarity search
-- ============================================================
create table if not exists kb_embeddings (
    embedding_id serial primary key,
    chunk_id int not null references kb_chunks(chunk_id) on delete cascade,
    embedding vector(384) not null,
    embedding_model text not null default 'all-MiniLM-L6-v2',
    created_at timestamp with time zone default now()
);

create index if not exists idx_kb_embeddings_chunk_id on kb_embeddings(chunk_id);

-- ============================================================
-- match_chunks: Semantic search function (called by the API)
-- ============================================================
create or replace function match_chunks(
    query_embedding vector(384),
    match_threshold float default 0.0,
    match_count int default 10
)
returns table (
    chunk_id int,
    chunk_text text,
    file_name text,
    title text,
    similarity float
)
language sql stable
as $$
    select
        kc.chunk_id,
        kc.chunk_text,
        kd.file_name,
        kd.title,
        (1 - (ke.embedding <=> query_embedding))::float as similarity
    from kb_embeddings ke
    join kb_chunks kc on kc.chunk_id = ke.chunk_id
    join kb_documents kd on kd.document_id = kc.document_id
    where (1 - (ke.embedding <=> query_embedding)) > match_threshold
    order by ke.embedding <=> query_embedding
    limit match_count;
$$;

-- ============================================================
-- Row Level Security
-- ============================================================
alter table kb_documents enable row level security;
alter table kb_chunks enable row level security;
alter table kb_embeddings enable row level security;

create policy "Allow service role on kb_documents"
    on kb_documents for all using (true) with check (true);

create policy "Allow service role on kb_chunks"
    on kb_chunks for all using (true) with check (true);

create policy "Allow service role on kb_embeddings"
    on kb_embeddings for all using (true) with check (true);
