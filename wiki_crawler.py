from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

URLS = [
    "https://en.wikipedia.org/wiki/Generative_AI",
    "https://en.wikipedia.org/wiki/Artificial_intelligence",
    "https://en.wikipedia.org/wiki/Computer",
    "https://en.wikipedia.org/wiki/Deep_learning",
    "https://en.wikipedia.org/wiki/Large_language_model",
]

OUTPUT_DIR = Path(__file__).resolve().parent / "datasets"


def slug_from_url(url: str) -> str:
    path = urlparse(url).path.rstrip("/")
    slug = path.split("/")[-1] or "page"
    return slug.lower()


def fetch_article_text(url: str) -> tuple[str, str]:
    response = requests.get(
        url,
        timeout=30,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; wiki-crawler/1.0; +https://wikipedia.org/)"
        },
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    title_tag = soup.select_one("span.mw-page-title-main") or soup.select_one("title")
    title = title_tag.get_text(strip=True) if title_tag else slug_from_url(url)

    content = soup.select_one("div.mw-parser-output")
    if content is None:
        raise ValueError(f"Could not find article content for {url}")

    article_nodes = content.select(":scope > p, :scope > ul > li, :scope > h2, :scope > h3")
    if len(article_nodes) < 5:
        article_nodes = soup.select("div.mw-content-ltr p, div.mw-content-ltr h2, div.mw-content-ltr h3, div.mw-content-ltr li")
    if len(article_nodes) < 5:
        article_nodes = content.find_all(["p", "h2", "h3", "li"])

    paragraphs: list[str] = []
    seen_text: set[str] = set()
    blocked_headings = {"contents", "references", "external links", "see also", "notes", "citations"}

    for node in article_nodes:
        if node.name in {"h2", "h3"}:
            heading = node.get_text(" ", strip=True)
            heading = re.sub(r"\[[0-9]+\]", "", heading)
            heading = re.sub(r"\s+", " ", heading).strip()
            if heading and heading.lower() not in blocked_headings:
                marker = f"## {heading}" if node.name == "h2" else f"### {heading}"
                if marker not in seen_text:
                    paragraphs.append(marker)
                    seen_text.add(marker)
            continue

        if node.find_parent(["table", "figure", "sup"]):
            continue

        text = node.get_text(" ", strip=True)
        text = re.sub(r"\[[0-9]+\]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) < 20:
            continue
        if text not in seen_text:
            paragraphs.append(text)
            seen_text.add(text)

    paragraphs = [item for item in paragraphs if item.strip()]

    if not paragraphs:
        raise ValueError(f"No article text extracted for {url}")

    body = "\n\n".join(paragraphs)
    markdown = f"# {title}\n\nSource: {url}\n\n{body}\n"
    return title, markdown


def save_markdown(url: str) -> Path:
    _, markdown = fetch_article_text(url)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{slug_from_url(url)}.md"
    output_path.write_text(markdown, encoding="utf-8")
    return output_path


def main() -> None:
    for url in URLS:
        output_path = save_markdown(url)
        print(f"Saved {output_path.name}")


if __name__ == "__main__":
    main()
