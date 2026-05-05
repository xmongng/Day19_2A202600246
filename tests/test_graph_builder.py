from graphrag_lab.graph_builder import build_graph
from graphrag_lab.models import Triple


def test_build_graph_deduplicates_edges_by_graph_behavior():
    triples = [
        Triple(subject="A", relation="RELATED_TO", object="B"),
        Triple(subject="A", relation="RELATED_TO", object="B"),
        Triple(subject="B", relation="RELATED_TO", object="C"),
    ]
    graph = build_graph(triples)
    assert graph.number_of_nodes() == 3
    assert graph.number_of_edges() == 2
