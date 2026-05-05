# GraphRAG Lab Rewrite

This project rewrites the notebook workflow into Python modules that use the crawled Wikipedia datasets in `datasets/`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e . pytest
cp .env.example .env
```

Add your API key to `.env`:

```env
OPENAI_API_KEY=your_key_here
```

## Commands

Build graph artifacts:

```bash
python -m graphrag_lab.cli index
```

Render graph image:

```bash
python -m graphrag_lab.cli visualize
```

Run a GraphRAG query:

```bash
python -m graphrag_lab.cli query "How are large language models related to transformers?"
```

Run a Flat RAG query:

```bash
python -m graphrag_lab.cli query "How are large language models related to transformers?" --mode flat
```

Run evaluation:

```bash
python -m graphrag_lab.cli evaluate
```

Run tests:

```bash
pytest
```

## Outputs

Generated files are written under `outputs/`:
- `triples.jsonl`
- `graph.json`
- `chunks.jsonl`
- `knowledge_graph.png`
- `evaluation.csv`

## Notes

- Real API mode is the default when `USE_LLM=true` and `OPENAI_API_KEY` is set.
- For offline tests, set `USE_LLM=false` in `.env`.
