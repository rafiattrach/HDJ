"""RAG client wrapper for Health Data Justice."""

import asyncio
import json as _json
import urllib.request
from pathlib import Path

from haiku.rag.client import HaikuRAG
from haiku.rag.config import AppConfig
from haiku.rag.config.models import (
    EmbeddingsConfig,
    EmbeddingModelConfig,
    SearchConfig,
    ProcessingConfig,
)

OLLAMA_BASE_URL = "http://localhost:11434"


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

    async def clear_documents(self) -> None:
        """Delete all indexed documents."""
        docs = await self.client.list_documents()
        for doc in docs:
            await self.client.delete_document(doc.id)

    async def index_single_pdf(self, pdf_path: Path) -> None:
        """Index a single PDF document."""
        await self.client.create_document_from_source(pdf_path)

    async def document_count(self) -> int:
        """Return the number of indexed documents."""
        docs = await self.client.list_documents()
        return len(docs)

    async def _raw_search(self, query: str, limit: int = 20) -> list[dict]:
        """Run a single search and return results as dicts (including chunk_id)."""
        results = await self.client.search(query, limit=limit)
        return [
            {
                "chunk_id": r.chunk_id,
                "content": r.content,
                "score": r.score,
                "document_uri": r.document_uri,
                "page_numbers": r.page_numbers,
            }
            for r in results
        ]

    async def search(
        self, query: str, limit: int = 20, cross_lingual: bool = True
    ) -> list[dict]:
        """Search and return results as dicts.

        When *cross_lingual* is ``True`` (default), the query is translated
        to the other language (DE↔EN) and both variants are searched.
        Results are merged by ``chunk_id`` (max score wins) and the top
        *limit* are returned.
        """
        if not cross_lingual:
            return await self._raw_search(query, limit=limit)

        # Lazy import to avoid loading translation models at module import
        from .translate import translate_query

        original, translated = translate_query(query)

        results_orig = await self._raw_search(original, limit=limit)

        if translated is None:
            return [
                {k: v for k, v in r.items() if k != "chunk_id"}
                for r in results_orig
            ]

        results_trans = await self._raw_search(translated, limit=limit)

        # Merge by chunk_id — max score wins per chunk
        merged: dict[str | None, dict] = {}
        for r in results_orig + results_trans:
            cid = r.get("chunk_id")
            if cid is None:
                # No chunk_id — just keep it (shouldn't happen in practice)
                merged[id(r)] = r
            elif cid not in merged or r["score"] > merged[cid]["score"]:
                merged[cid] = r

        sorted_results = sorted(merged.values(), key=lambda r: r["score"], reverse=True)

        return [
            {k: v for k, v in r.items() if k != "chunk_id"}
            for r in sorted_results[:limit]
        ]

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings for a list of texts using the configured Ollama model.

        Calls the Ollama ``/api/embed`` endpoint directly so we can embed
        arbitrary strings (not just indexed documents).
        """
        model_name = self.config.embeddings.model.name

        def _call_ollama() -> list[list[float]]:
            payload = _json.dumps({"model": model_name, "input": texts}).encode()
            req = urllib.request.Request(
                f"{OLLAMA_BASE_URL}/api/embed",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                return _json.loads(resp.read())["embeddings"]

        return await asyncio.to_thread(_call_ollama)

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
