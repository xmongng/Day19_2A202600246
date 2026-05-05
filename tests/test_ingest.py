from pathlib import Path

from graphrag_lab.ingest import load_markdown_documents


def test_load_markdown_documents():
    dataset_dir = Path(__file__).resolve().parents[1] / "datasets"
    documents = load_markdown_documents(dataset_dir)
    assert documents
    assert all(document.title for document in documents)
    assert all(document.text for document in documents)
