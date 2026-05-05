from __future__ import annotations

import json

from openai import OpenAI

from .config import Settings


class LLMClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = OpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None

    def require_client(self) -> OpenAI:
        if not self.client:
            raise RuntimeError("OPENAI_API_KEY is missing. Copy .env.example to .env and set your API key.")
        return self.client

    def extract_triples(self, text: str) -> list[dict[str, str]]:
        client = self.require_client()
        prompt = (
            "Extract knowledge graph triples from the text. Return JSON only in the shape "
            "{\"triples\":[{\"subject\":\"...\",\"relation\":\"...\",\"object\":\"...\"}]}. "
            "Use short canonical entities and uppercase snake case relations. Text: "
            f"{text}"
        )
        response = client.chat.completions.create(
            model=self.settings.chat_model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        payload = json.loads(response.choices[0].message.content)
        return payload.get("triples", [])

    def answer_question(self, question: str, context: str) -> str:
        client = self.require_client()
        prompt = (
            "Answer the question using only the provided context. If the answer is not in the context, say so briefly.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}"
        )
        response = client.chat.completions.create(
            model=self.settings.chat_model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()
