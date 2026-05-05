from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path

import networkx as nx

from .config import Settings
from .models import Chunk, QueryResult
from .query import answer_with_flat_rag, answer_with_graph

BENCHMARK_QUESTIONS = [
    "What is generative AI?",
    "How are large language models related to transformers?",
    "What is the relationship between deep learning and artificial intelligence?",
    "What does retrieval-augmented generation mean?",
    "How did transformers change language models?",
    "What are some applications of generative AI?",
    "How are LLMs trained?",
    "What is tokenization in large language models?",
    "How is dataset cleaning used in LLM pipelines?",
    "What is the role of attention in transformer models?",
    "How is ChatGPT related to GPT models?",
    "What does multimodal mean for LLMs?",
    "How did deep neural networks enable generative AI growth?",
    "What are some limitations of large language models?",
    "How does Flat RAG differ from GraphRAG?",
    "What is the difference between BERT and GPT style models?",
    "Why is deduplication important in graph construction?",
    "How can tools extend LLM capabilities?",
    "How are embeddings used in language processing?",
    "What is RLHF used for in LLM training?",
]


def run_benchmark(graph: nx.DiGraph, chunks: list[Chunk], settings: Settings) -> list[dict[str, str | float]]:
    rows: list[dict[str, str | float]] = []
    for question in BENCHMARK_QUESTIONS:
        graph_result = answer_with_graph(graph, question, settings)
        flat_result = answer_with_flat_rag(chunks, question, settings)
        rows.append(_row_from_results(graph_result, flat_result))
    return rows


def _row_from_results(graph_result: QueryResult, flat_result: QueryResult) -> dict[str, str | float]:
    return {
        "question": graph_result.question,
        "graph_entity": graph_result.entity,
        "graph_answer": graph_result.answer,
        "graph_latency_seconds": round(graph_result.latency_seconds, 4),
        "flat_answer": flat_result.answer,
        "flat_latency_seconds": round(flat_result.latency_seconds, 4),
    }


def save_evaluation(rows: list[dict[str, str | float]], output_path: Path) -> None:
    if not rows:
        return
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
