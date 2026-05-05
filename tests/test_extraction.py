from graphrag_lab.extraction import fallback_extract_triples
from graphrag_lab.models import Chunk


def test_fallback_extract_triples_returns_items():
    chunk = Chunk(
        chunk_id="c1",
        doc_id="doc1",
        title="Generative AI",
        source_url="https://example.com",
        text="Generative AI is a subfield of artificial intelligence. ChatGPT is an application of Generative AI introduced in 2022.",
    )
    triples = fallback_extract_triples(chunk)
    assert triples
    assert any(triple.subject for triple in triples)
    assert any(triple.object for triple in triples)
