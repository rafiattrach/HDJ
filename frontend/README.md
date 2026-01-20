# HDJ Frontend

Web interface for Health Data Justice RAG evaluation.

## Run

```bash
# From project root
source .venv/bin/activate
streamlit run frontend/app.py
```

Opens at http://localhost:8501

## Features

1. **Documents** — Upload and manage PDF files
2. **Build Index** — Create searchable vector index from PDFs
3. **Evaluate** — Test queries against gold standard highlights
4. **Search** — Run queries across all documents

## User Journey

1. Upload your PDFs in the Documents tab
2. Click "Build Index" to process them
3. Go to Evaluate tab, add queries or use defaults
4. Run evaluation to see recall/precision metrics
5. Use Search tab for ad-hoc queries
