from __future__ import annotations

from .models import Chunk, Document


def chunk_documents(documents: list[Document], max_chunk_chars: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in documents:
        paragraphs = [paragraph.strip() for paragraph in document.text.split(". ") if paragraph.strip()]
        buffer = ""
        index = 0
        for paragraph in paragraphs:
            candidate = f"{buffer}. {paragraph}".strip(". ") if buffer else paragraph
            if len(candidate) <= max_chunk_chars:
                buffer = candidate
                continue
            if buffer:
                chunks.append(
                    Chunk(
                        chunk_id=f"{document.doc_id}-{index}",
                        doc_id=document.doc_id,
                        title=document.title,
                        source_url=document.source_url,
                        text=buffer.strip(),
                    )
                )
                index += 1
            buffer = paragraph
        if buffer:
            chunks.append(
                Chunk(
                    chunk_id=f"{document.doc_id}-{index}",
                    doc_id=document.doc_id,
                    title=document.title,
                    source_url=document.source_url,
                    text=buffer.strip(),
                )
            )
    return chunks
