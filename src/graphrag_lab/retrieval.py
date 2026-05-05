from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .models import Chunk


class FlatRAGRetriever:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform([chunk.text for chunk in chunks]) if chunks else None

    def retrieve(self, question: str, top_k: int) -> list[Chunk]:
        if not self.chunks or self.matrix is None:
            return []
        query_vector = self.vectorizer.transform([question])
        similarities = cosine_similarity(query_vector, self.matrix).flatten()
        ranked_indexes = similarities.argsort()[::-1][:top_k]
        return [self.chunks[index] for index in ranked_indexes if similarities[index] > 0]
