from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx


def render_graph(graph: nx.DiGraph, output_path: Path) -> None:
    plt.figure(figsize=(16, 10))
    pos = nx.spring_layout(graph, seed=42, k=1.3)
    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_color="lightblue",
        node_size=1800,
        font_size=8,
        arrows=True,
    )
    edge_labels = nx.get_edge_attributes(graph, "relation")
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=7)
    plt.title("GraphRAG Knowledge Graph")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
