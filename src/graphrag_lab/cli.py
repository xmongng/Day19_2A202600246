from __future__ import annotations

import argparse
import json

from .chunking import chunk_documents
from .config import get_settings
from .evaluate import run_benchmark, save_evaluation
from .extraction import extract_triples
from .graph_builder import build_graph, load_graph, save_graph, save_triples
from .ingest import load_markdown_documents
from .query import answer_with_flat_rag, answer_with_graph
from .visualize import render_graph


def build_pipeline():
    settings = get_settings()
    documents = load_markdown_documents(settings.dataset_dir)
    chunks = chunk_documents(documents, settings.max_chunk_chars)
    triples = extract_triples(chunks, settings)
    graph = build_graph(triples)
    save_triples(triples, settings.triples_path)
    save_graph(graph, settings.graph_path)
    settings.chunks_path.write_text(json.dumps([chunk.__dict__ for chunk in chunks], ensure_ascii=False, indent=2), encoding="utf-8")
    return settings, documents, chunks, triples, graph


def command_index(_: argparse.Namespace) -> None:
    settings, documents, chunks, triples, graph = build_pipeline()
    print(f"Loaded {len(documents)} documents")
    print(f"Created {len(chunks)} chunks")
    print(f"Extracted {len(triples)} triples")
    print(f"Built graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")


def command_visualize(_: argparse.Namespace) -> None:
    settings = get_settings()
    graph = load_graph(settings.graph_path)
    render_graph(graph, settings.graph_image_path)
    print(f"Saved graph image to {settings.graph_image_path}")


def command_query(args: argparse.Namespace) -> None:
    settings = get_settings()
    graph = load_graph(settings.graph_path)
    chunks_payload = json.loads(settings.chunks_path.read_text(encoding="utf-8"))
    from .models import Chunk

    chunks = [Chunk(**item) for item in chunks_payload]
    if args.mode == "graph":
        result = answer_with_graph(graph, args.question, settings)
    else:
        result = answer_with_flat_rag(chunks, args.question, settings)
    print(result.answer)
    if result.context:
        print("\nContext:\n")
        print(result.context)


def command_evaluate(_: argparse.Namespace) -> None:
    settings = get_settings()
    graph = load_graph(settings.graph_path)
    chunks_payload = json.loads(settings.chunks_path.read_text(encoding="utf-8"))
    from .models import Chunk

    chunks = [Chunk(**item) for item in chunks_payload]
    rows = run_benchmark(graph, chunks, settings)
    save_evaluation(rows, settings.evaluation_path)
    print(f"Saved evaluation to {settings.evaluation_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="GraphRAG lab CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index")
    index_parser.set_defaults(func=command_index)

    visualize_parser = subparsers.add_parser("visualize")
    visualize_parser.set_defaults(func=command_visualize)

    query_parser = subparsers.add_parser("query")
    query_parser.add_argument("question")
    query_parser.add_argument("--mode", choices=["graph", "flat"], default="graph")
    query_parser.set_defaults(func=command_query)

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.set_defaults(func=command_evaluate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
