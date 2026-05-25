from __future__ import annotations

from dataclasses import dataclass

from rag.loader import ParsedDocument


@dataclass(frozen=True)
class DocumentChunk:
    source: str
    section_path: list[str]
    text: str


def chunk_document(
    doc: ParsedDocument,
    *,
    target_chars: int = 900,
    max_chars: int = 1400,
) -> list[DocumentChunk]:
    if target_chars <= 0 or max_chars <= 0 or max_chars < target_chars:
        raise ValueError("Invalid chunk size parameters.")

    chunks: list[DocumentChunk] = []
    heading_stack: list[tuple[int, str]] = []
    buffer: list[str] = []
    buffer_len = 0

    def _flush() -> None:
        nonlocal buffer_len
        if not buffer:
            return
        text = "\n".join(buffer).strip()
        if text:
            chunks.append(
                DocumentChunk(
                    source=doc.file_name,
                    section_path=[t for _, t in heading_stack],
                    text=text,
                )
            )
        buffer.clear()
        buffer_len = 0

    for block in doc.blocks:
        if block.kind == "heading":
            _flush()
            lvl = block.level or 9
            while heading_stack and heading_stack[-1][0] >= lvl:
                heading_stack.pop()
            heading_stack.append((lvl, block.text))
            continue

        paragraph = (block.text or "").strip()
        if not paragraph:
            continue

        if len(paragraph) > max_chars:
            _flush()
            chunks.append(
                DocumentChunk(
                    source=doc.file_name,
                    section_path=[t for _, t in heading_stack],
                    text=paragraph,
                )
            )
            continue

        addition = len(paragraph) + (1 if buffer else 0)
        if buffer and (buffer_len + addition) > max_chars:
            _flush()

        buffer.append(paragraph)
        buffer_len += addition

        if buffer_len >= target_chars:
            _flush()

    _flush()

    if not chunks:
        raise ValueError(f"No chunks produced from: {doc.file_name}")

    return chunks
