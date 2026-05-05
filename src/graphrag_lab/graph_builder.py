from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
from networkx.readwrite import json_graph

from .models import Triple


def build_graph(triples: list[Triple]) -> nx.DiGraph:
    graph = nx.DiGraph()
    for triple in triples:
        graph.add_node(triple.subject)
        graph.add_node(triple.object)
        graph.add_edge(
            triple.subject,
            triple.object,
            relation=triple.relation,
            source_chunk_id=triple.source_chunk_id,
            source_doc_id=triple.source_doc_id,
        )
    return graph


def save_triples(triples: list[Triple], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        for triple in triples:
            handle.write(json.dumps(triple.__dict__, ensure_ascii=False) + "\n")


def save_graph(graph: nx.DiGraph, output_path: Path) -> None:
    payload = json_graph.node_link_data(graph)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_graph(input_path: Path) -> nx.DiGraph:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    return json_graph.node_link_graph(payload)
