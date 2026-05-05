from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    source_url: str
    text: str
    path: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    source_url: str
    text: str


@dataclass(frozen=True)
class Triple:
    subject: str
    relation: str
    object: str
    source_chunk_id: str = ""
    source_doc_id: str = ""


@dataclass
class QueryResult:
    question: str
    entity: str
    answer: str
    context: str
    supporting_facts: list[str] = field(default_factory=list)
    retrieved_items: list[str] = field(default_factory=list)
    latency_seconds: float = 0.0
