"""RAG client wrapper for Health Data Justice."""

import asyncio
import json as _json
import re
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


# "Structural" markers that are characteristic of a reference-LIST entry and
# rarely appear in ordinary prose: DOIs, URLs, page/volume numbers, editor
# marks, "Retrieved from", "Accessed".
_STRONG_REFERENCE_RE = re.compile(
    r"\bdoi:"
    r"|https?://"
    r"|\bpp?\.\s*\d"             # p. 12 / pp. 12-14
    r"|\bvol\.\s*\d"
    r"|\beds?\.\b"
    r"|\bretrieved from\b"
    r"|\baccessed\b",
    re.IGNORECASE,
)
# "Weak" markers that signal a citation but also occur in normal argument prose
# (a parenthetical year, "et al."), so a handful of them must not condemn a
# passage on their own.
_WEAK_REFERENCE_RE = re.compile(
    r"\(\d{4}[a-z]?\)"           # (2016)  (2016a)
    r"|\bet al\.?",
    re.IGNORECASE,
)


def is_reference_like(text: str) -> bool:
    """Heuristic: is this chunk predominantly a bibliography / reference list?

    Reference-list chunks rank highly on keyword overlap while carrying no
    argument, so they are filtered out of search results by default. The hard
    part is separating them from prose that merely *cites* prior work. Two
    independent signals are used:

    * **Two or more structural markers** (DOIs, URLs, "pp.", "vol.",
      "Retrieved from", ...) — these almost never occur together in prose, so
      they are strong evidence of a reference list.
    * Otherwise, only flag when citation markers are both **numerous and dense**
      (≥5 markers at ≥8 per 100 words). This catches reference lists made purely
      of "(year)" entries while leaving an argument paragraph that cites a few
      sources untouched.
    """
    words = re.findall(r"\w+", text)
    if len(words) < 8:
        return False
    strong = len(_STRONG_REFERENCE_RE.findall(text))
    if strong >= 2:
        return True
    total = strong + len(_WEAK_REFERENCE_RE.findall(text))
    if total < 5:
        return False
    return total / len(words) * 100 >= 8.0


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
        self,
        query: str,
        limit: int = 20,
        cross_lingual: bool = True,
        drop_references: bool = True,
        with_relevance: bool = True,
    ) -> list[dict]:
        """Search and return results as dicts.

        When *cross_lingual* is ``True`` (default), the query is translated
        to the other language (DE↔EN) and both variants are searched.
        Results are merged by ``chunk_id`` (max score wins) and the top
        *limit* are returned.

        When *drop_references* is ``True`` (default), chunks that look like
        bibliography / citation lists are removed (see :func:`is_reference_like`)
        — but only if doing so still leaves results, so a query that genuinely
        targets reference material is never left empty.

        When *with_relevance* is ``True`` (default), each result gets a
        ``relevance`` field: the cosine similarity between the query and the
        passage (a real 0–1 relevance), distinct from the internal ``score``
        which is a tiny rank-fusion value with no standalone meaning.
        """
        if not cross_lingual:
            merged_results = await self._raw_search(query, limit=limit)
        else:
            # Lazy import to avoid loading translation models at module import
            from .translate import translate_query

            original, translated = translate_query(query)
            results_orig = await self._raw_search(original, limit=limit)

            if translated is None:
                merged_results = results_orig
            else:
                results_trans = await self._raw_search(translated, limit=limit)
                # Merge by chunk_id — max score wins per chunk
                merged: dict[str | None, dict] = {}
                for r in results_orig + results_trans:
                    cid = r.get("chunk_id")
                    if cid is None:
                        merged[id(r)] = r
                    elif cid not in merged or r["score"] > merged[cid]["score"]:
                        merged[cid] = r
                merged_results = sorted(
                    merged.values(), key=lambda r: r["score"], reverse=True
                )

        # Drop bibliography/citation chunks — but never return nothing.
        if drop_references:
            kept = [r for r in merged_results if not is_reference_like(r["content"])]
            if kept:
                merged_results = kept

        merged_results = merged_results[:limit]

        # Attach a true query↔passage relevance (cosine similarity).
        if with_relevance and merged_results:
            try:
                rel = await self._relevance_scores(
                    query, [r["content"] for r in merged_results]
                )
                for r, score in zip(merged_results, rel):
                    r["relevance"] = score
            except Exception:
                for r in merged_results:
                    r["relevance"] = None

        return [
            {k: v for k, v in r.items() if k != "chunk_id"}
            for r in merged_results
        ]

    async def _relevance_scores(self, query: str, texts: list[str]) -> list[float]:
        """Cosine similarity between *query* and each text (0–1)."""
        import numpy as np

        vecs = await self.embed_texts([query] + texts)
        arr = np.asarray(vecs, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        unit = arr / norms
        sims = unit[1:] @ unit[0]
        return [float(s) for s in sims]

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
