from __future__ import annotations

import time
from collections import deque

import networkx as nx

from .config import Settings
from .llm import LLMClient
from .models import Chunk, QueryResult
from .retrieval import FlatRAGRetriever


def extract_query_entity(question: str, graph: nx.DiGraph) -> str:
    lowered = question.lower()
    candidates = sorted(graph.nodes, key=len, reverse=True)
    for node in candidates:
        if node.lower() in lowered:
            return node
    return candidates[0] if candidates else ""


def collect_subgraph_nodes(graph: nx.DiGraph, entity: str, hops: int) -> set[str]:
    if entity not in graph:
        return set()
    visited = {entity}
    queue = deque([(entity, 0)])
    while queue:
        node, depth = queue.popleft()
        if depth >= hops:
            continue
        neighbors = set(graph.successors(node)) | set(graph.predecessors(node))
        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, depth + 1))
    return visited


def textualize_subgraph(graph: nx.DiGraph, nodes: set[str]) -> list[str]:
    facts: list[str] = []
    for source, target, data in graph.subgraph(nodes).edges(data=True):
        relation = data.get("relation", "RELATED_TO")
        facts.append(f"{source} --[{relation}]--> {target}")
    return facts


def answer_with_graph(graph: nx.DiGraph, question: str, settings: Settings) -> QueryResult:
    start = time.perf_counter()
    entity = extract_query_entity(question, graph)
    nodes = collect_subgraph_nodes(graph, entity, settings.graph_hops)
    facts = textualize_subgraph(graph, nodes)
    context = "\n".join(facts)
    answer = context
    if settings.use_llm:
        answer = LLMClient(settings).answer_question(question, context)
    return QueryResult(
        question=question,
        entity=entity,
        answer=answer,
        context=context,
        supporting_facts=facts,
        latency_seconds=time.perf_counter() - start,
    )


def answer_with_flat_rag(chunks: list[Chunk], question: str, settings: Settings) -> QueryResult:
    start = time.perf_counter()
    retriever = FlatRAGRetriever(chunks)
    retrieved_chunks = retriever.retrieve(question, settings.top_k)
    context = "\n\n".join(chunk.text for chunk in retrieved_chunks)
    answer = context
    if settings.use_llm:
        answer = LLMClient(settings).answer_question(question, context)
    return QueryResult(
        question=question,
        entity="",
        answer=answer,
        context=context,
        retrieved_items=[chunk.chunk_id for chunk in retrieved_chunks],
        latency_seconds=time.perf_counter() - start,
    )
