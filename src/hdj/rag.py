"""RAG client wrapper for Health Data Justice."""

from pathlib import Path

from haiku.rag.client import HaikuRAG
from haiku.rag.config import AppConfig
from haiku.rag.config.models import (
    EmbeddingsConfig,
    EmbeddingModelConfig,
    SearchConfig,
    ProcessingConfig,
)


def get_config() -> AppConfig:
    """Default RAG configuration using Qwen3-Embedding-4B via Ollama."""
    return AppConfig(
        embeddings=EmbeddingsConfig(
            model=EmbeddingModelConfig(
                provider="ollama",
                name="qwen3-embedding:4b",
                vector_dim=2560,
            )
        ),
        search=SearchConfig(
            limit=20,
            context_radius=1,
        ),
        processing=ProcessingConfig(
            chunk_size=512,
            chunking_tokenizer="Qwen/Qwen3-Embedding-0.6B",
        ),
    )


class HDJRag:
    """Health Data Justice RAG wrapper."""

    def __init__(self, db_path: Path, config: AppConfig | None = None):
        self.db_path = db_path
        self.config = config or get_config()
        self._client: HaikuRAG | None = None

    async def __aenter__(self):
        self._client = HaikuRAG(self.db_path, config=self.config, create=True)
        return self

    async def __aexit__(self, *args):
        if self._client:
            self._client.close()

    @property
    def client(self) -> HaikuRAG:
        if not self._client:
            raise RuntimeError("Use 'async with HDJRag(...) as rag:' context manager")
        return self._client

    async def index_pdfs(self, pdf_dir: Path, force: bool = False) -> int:
        """Index all PDFs in directory. Returns count of indexed documents."""
        docs = await self.client.list_documents()
        pdf_files = list(pdf_dir.glob("*.pdf"))

        if not force and len(docs) >= len(pdf_files):
            return len(docs)

        if force and docs:
            for doc in docs:
                await self.client.delete_document(doc.id)

        for pdf in pdf_files:
            await self.client.create_document_from_source(pdf)

        return len(pdf_files)

    async def search(self, query: str, limit: int = 20) -> list[dict]:
        """Search and return results as dicts."""
        results = await self.client.search(query, limit=limit)
        return [
            {
                "content": r.content,
                "score": r.score,
                "document_uri": r.document_uri,
                "page_numbers": r.page_numbers,
            }
            for r in results
        ]

    async def list_documents(self) -> list[dict]:
        """List all indexed documents."""
        docs = await self.client.list_documents()
        return [
            {
                "id": d.id,
                "uri": d.uri,
                "title": d.title,
            }
            for d in docs
        ]
