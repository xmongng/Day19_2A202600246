from __future__ import annotations

import re
from pathlib import Path

from .models import Document


def _extract_title(lines: list[str], fallback: str) -> str:
    for line in lines:
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


def _extract_source(lines: list[str]) -> str:
    for line in lines:
        if line.startswith("Source:"):
            return line.split(":", 1)[1].strip()
    return ""


def _clean_text(raw_text: str) -> str:
    text = re.sub(r"\[[^\]]+\]", "", raw_text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_markdown_documents(dataset_dir: Path) -> list[Document]:
    documents: list[Document] = []
    for path in sorted(dataset_dir.glob("*.md")):
        lines = path.read_text(encoding="utf-8").splitlines()
        title = _extract_title(lines, path.stem.replace("_", " ").title())
        source_url = _extract_source(lines)
        content_lines = [line for line in lines if not line.startswith("# ") and not line.startswith("Source:")]
        text = _clean_text("\n".join(content_lines))
        documents.append(
            Document(
                doc_id=path.stem,
                title=title,
                source_url=source_url,
                text=text,
                path=str(path),
            )
        )
    return documents
