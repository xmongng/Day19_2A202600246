from graphrag_lab.chunking import chunk_documents
from graphrag_lab.graph_builder import build_graph
from graphrag_lab.ingest import load_markdown_documents
from graphrag_lab.extraction import fallback_extract_triples


def test_smoke_pipeline_without_llm(tmp_path):
    dataset_dir = tmp_path / "datasets"
    dataset_dir.mkdir()
    sample = dataset_dir / "sample.md"
    sample.write_text(
        "# Generative AI\n\nSource: https://example.com\n\nGenerative AI is a subfield of artificial intelligence. Transformers are used in large language models.",
        encoding="utf-8",
    )

    documents = load_markdown_documents(dataset_dir)
    chunks = chunk_documents(documents, max_chunk_chars=120)
    triples = []
    for chunk in chunks:
        triples.extend(fallback_extract_triples(chunk))
    graph = build_graph(triples)

    assert documents
    assert chunks
    assert triples
    assert graph.number_of_nodes() > 0
    assert graph.number_of_edges() > 0
