import networkx as nx

from graphrag_lab.query import collect_subgraph_nodes, extract_query_entity, textualize_subgraph


def test_collect_subgraph_nodes_two_hops():
    graph = nx.DiGraph()
    graph.add_edge("Generative AI", "Artificial intelligence", relation="SUBFIELD_OF")
    graph.add_edge("Artificial intelligence", "Deep learning", relation="USES")
    nodes = collect_subgraph_nodes(graph, "Generative AI", 2)
    assert nodes == {"Generative AI", "Artificial intelligence", "Deep learning"}


def test_extract_query_entity_matches_question_text():
    graph = nx.DiGraph()
    graph.add_node("Large language model")
    graph.add_node("Transformer")
    entity = extract_query_entity("How is a large language model related to transformers?", graph)
    assert entity == "Large language model"


def test_textualize_subgraph_outputs_relations():
    graph = nx.DiGraph()
    graph.add_edge("A", "B", relation="RELATED_TO")
    facts = textualize_subgraph(graph, {"A", "B"})
    assert facts == ["A --[RELATED_TO]--> B"]
