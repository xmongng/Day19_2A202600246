from __future__ import annotations

import re

from .config import Settings
from .llm import LLMClient
from .models import Chunk, Triple

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "by",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}


def normalize_relation(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_")
    return normalized.upper() or "RELATED_TO"


def normalize_entity(value: str) -> str:
    value = re.sub(r"\s+", " ", value).strip(" ,.-")
    return value


def fallback_extract_triples(chunk: Chunk) -> list[Triple]:
    text = chunk.text
    triples: list[Triple] = []

    title = normalize_entity(chunk.title)
    for match in re.finditer(r"([A-Z][A-Za-z0-9\-]+(?:\s+[A-Z][A-Za-z0-9\-]+){0,4})", text):
        entity = normalize_entity(match.group(1))
        if entity == title:
            continue
        triples.append(Triple(subject=title, relation="MENTIONS", object=entity, source_chunk_id=chunk.chunk_id, source_doc_id=chunk.doc_id))

    lower_text = text.lower()
    if "subfield of" in lower_text:
        for fragment in re.findall(r"([A-Z][^\.]{0,80}?subfield of [A-Za-z\s\-]+)", text):
            parts = re.split(r"subfield of", fragment, flags=re.IGNORECASE)
            if len(parts) == 2:
                triples.append(
                    Triple(
                        subject=normalize_entity(parts[0]),
                        relation="SUBFIELD_OF",
                        object=normalize_entity(parts[1]),
                        source_chunk_id=chunk.chunk_id,
                        source_doc_id=chunk.doc_id,
                    )
                )

    for year in re.findall(r"\b(19\d{2}|20\d{2})\b", text):
        triples.append(Triple(subject=title, relation="MENTIONS_YEAR", object=year, source_chunk_id=chunk.chunk_id, source_doc_id=chunk.doc_id))

    unique: dict[tuple[str, str, str], Triple] = {}
    for triple in triples:
        if not triple.subject or not triple.object:
            continue
        if triple.subject.lower() in STOPWORDS or triple.object.lower() in STOPWORDS:
            continue
        unique[(triple.subject, triple.relation, triple.object)] = triple
    return list(unique.values())


def extract_triples(chunks: list[Chunk], settings: Settings) -> list[Triple]:
    llm = LLMClient(settings)
    triples: list[Triple] = []
    for chunk in chunks:
        if settings.use_llm:
            raw_triples = llm.extract_triples(chunk.text)
            for item in raw_triples:
                subject = normalize_entity(item.get("subject", ""))
                relation = normalize_relation(item.get("relation", "RELATED_TO"))
                obj = normalize_entity(item.get("object", ""))
                if subject and obj:
                    triples.append(
                        Triple(
                            subject=subject,
                            relation=relation,
                            object=obj,
                            source_chunk_id=chunk.chunk_id,
                            source_doc_id=chunk.doc_id,
                        )
                    )
        else:
            triples.extend(fallback_extract_triples(chunk))

    unique: dict[tuple[str, str, str], Triple] = {}
    for triple in triples:
        unique[(triple.subject, triple.relation, triple.object)] = triple
    return list(unique.values())
