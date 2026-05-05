from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    root_dir: Path = ROOT_DIR
    dataset_dir: Path = ROOT_DIR / os.getenv("DATASET_DIR", "datasets")
    output_dir: Path = ROOT_DIR / os.getenv("OUTPUT_DIR", "outputs")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    chat_model: str = os.getenv("CHAT_MODEL", "gpt-4o-mini")
    embed_model: str = os.getenv("EMBED_MODEL", "text-embedding-3-small")
    use_llm: bool = os.getenv("USE_LLM", "true").lower() == "true"
    max_chunk_chars: int = int(os.getenv("MAX_CHUNK_CHARS", "1800"))
    graph_hops: int = int(os.getenv("GRAPH_HOPS", "2"))
    top_k: int = int(os.getenv("TOP_K", "4"))

    @property
    def graph_path(self) -> Path:
        return self.output_dir / "graph.json"

    @property
    def triples_path(self) -> Path:
        return self.output_dir / "triples.jsonl"

    @property
    def chunks_path(self) -> Path:
        return self.output_dir / "chunks.jsonl"

    @property
    def evaluation_path(self) -> Path:
        return self.output_dir / "evaluation.csv"

    @property
    def graph_image_path(self) -> Path:
        return self.output_dir / "knowledge_graph.png"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    return settings
