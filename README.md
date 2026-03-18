# Health Data Justice - RAG Evaluation

RAG pipeline for finding relevant sections in health data governance documents.

## Quick Start

**Mac:** Double-click `run.sh` (or run `./run.sh` in Terminal)
**Windows:** Double-click `run.bat`

These scripts handle everything automatically: updates, Python, dependencies, Ollama, and launching the app.

## Setup

```bash
./setup.sh
```

## Web Interface

```bash
source .venv/bin/activate
streamlit run frontend/app.py
```

Opens at http://localhost:8501

**Features:**
- Upload/remove PDFs
- Build search index
- Evaluate queries against gold standard
- Search across all documents

## CLI Usage

### Run evaluation with default queries
```bash
source .venv/bin/activate
python evaluate.py
```

### Run with a custom query
```bash
python evaluate.py --query "your search query here"
```

### Re-index after adding new PDFs
```bash
python evaluate.py --reindex
```

## Adding Documents

1. Place PDF files in `data/pdfs/`
2. Run `python evaluate.py --reindex`

Documents are automatically chunked and indexed.

## Editing Queries

Default queries are in `queries.json`. Edit this file to add or modify queries:

```json
{
  "my_query_name": "The text of your query goes here"
}
```

## Understanding Results

Results are saved to `results/` with timestamps.

### Metrics

- **Recall**: What percentage of the gold standard sections were found?
  - Recall = (found sections) / (total gold standard sections)
  - Higher is better. 90%+ means the query finds most relevant content.

- **Precision** (shown per-query): Of the sections retrieved, how many were actually relevant?
  - Precision = (relevant retrieved) / (total retrieved)
  - Higher means less noise in results.

Example: If gold standard has 15 sections and query finds 14 of them out of 20 retrieved:
- Recall = 14/15 = 93%
- Precision = 14/20 = 70%

## Project Structure

```
src/hdj/                     # Core Python module
├── __init__.py
├── rag.py                   # RAG client wrapper
└── evaluate.py              # Evaluation logic

frontend/                    # Web interface (planned)

data/
├── pdfs/                    # Add your PDFs here
├── gold_standard.json       # Highlighted sections (ground truth)
└── gold_standard/
    └── definition.md        # Data justice definition

queries.json                 # Query configurations
results/                     # Timestamped evaluation outputs
evaluate.py                  # CLI entry point
```

## Technical Notes

- Qwen3-Embedding-4B via Ollama for embeddings
- Python 3.12+ required
- LanceDB for vector storage (file-based, no server)
