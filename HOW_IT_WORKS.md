# Health Data Justice — Corpus Search Tool: How It Works

*Maintainer reference for the tool — what it does, how the pieces fit, and the
reasoning behind each number. The app itself is built for non-technical
researchers; this page is the technical companion.*

**What the tool does:** you give it PDFs and a research question (your working
definition of *health data justice*); it finds the passages that match — so you
can tell whether a PDF is relevant and see what authors actually say about your
topic. Every result traces back to a document and page.

---

## End-to-end flow

The app has seven tabs, in working order:

```
1. Documents        → upload PDFs
2. Prepare Documents→ "index" them: split into ~half-page passages + embed each
3. Search           → ask a question → ranked relevant passages (daily use)
📌 My Findings      → passages you saved while searching
4. Validate Search  → measure how well a question finds passages you marked relevant
5. Activity Log     → audit trail of what was done
6. History          → track Coverage over time as you tune questions
```

### What happens under the hood
1. **Index**: each PDF → passages (~512 tokens) → a vector ("embedding") via the
   `qwen3-embedding:4b` model running locally through **Ollama**. Stored in LanceDB.
2. **Search**: your question is embedded and matched two ways at once —
   **keyword** (exact words) + **vector** (meaning) — then fused (**hybrid search**).
   With *Cross-lingual* on, the question is also translated DE↔EN and both variants
   searched, so a German passage can answer an English question (and vice-versa).
3. **Relevance** of each result = cosine similarity between your question and the
   passage (0–100%).

---

## The numbers on screen (and exactly what they mean)

| Number | Plain meaning | Formula | Range |
|---|---|---|---|
| **Relevance** | How close a passage is to your question (meaning, not just words) | cosine(question, passage) | 0–100% |
| **Coverage** (validation) | Of *your reference passages*, how many the question found | found ÷ total reference passages | 0–100% |
| **Accuracy** (validation) | Of the *results returned*, how many were actually relevant | relevant results ÷ results returned | 0–100% |
| **Word match** | Share of a reference passage's meaningful words present in a result | shared ÷ reference words | 0–100% |
| **Meaning similarity** | Same idea even if worded differently / different language | cosine of meanings | 0–100% |

A reference passage counts as **found** if Word match ≥ threshold **OR** Meaning
similarity ≥ threshold (both sliders live in the Validate tab).

**Reference passage** = a chunk of text *you* marked as "this is relevant" (the
gold standard). **Best-matching / Nearest passage** = the chunk the search
actually returned that overlaps it most. With only one or two PDFs loaded these
are often the *same text* — that is expected, not an error.

---

## Metric design notes (read before changing the math)

These are the non-obvious decisions behind the numbers. They are easy to "fix"
in the wrong direction, so the reasoning is recorded here.

- **Accuracy is a per-chunk ratio: `relevant results ÷ results returned`.** It is
  deliberately *not* `reference-passages-found ÷ results-returned` — that mixes
  two different populations (how many reference passages exist vs. how many
  chunks came back) and can exceed 100% when there are more reference passages
  than retrieved chunks. Keep it per-chunk so it always stays within 0–100%.
  (`src/hdj/evaluate.py`)
- **Relevance is cosine similarity between the question and the passage**, not the
  internal hybrid-search ranking value. The ranking value is a rank-fusion
  number whose maximum is roughly 3% (`2 / (60 + 1)`), so it is only meaningful
  for ordering, never as a standalone percentage. Display the cosine relevance.
  (`src/hdj/rag.py`, `src/hdj/evaluate.py`)
- **Passages are shown in full** in the Validate match/miss panels (they sit
  inside collapsible expanders), so a match can always be read end-to-end.
  (`frontend/app.py`)
- **Bibliography / citation chunks are filtered out of search results by default.**
  Reference-list text otherwise ranks on keyword overlap while carrying no
  argument. A chunk is dropped when it has two or more *structural* markers
  (DOIs, URLs, "pp.", "vol.", "Retrieved from", ...), or when citation markers
  are both numerous and dense — so an argument paragraph that merely cites a few
  sources is kept. The filter never empties a result set, and it is **not**
  applied during Validation (which must measure raw retrieval). Toggle:
  *Hide citations* on the Search tab. (`src/hdj/rag.py`, `is_reference_like`)
- **Every on-screen number is explained in-app** via the "What do these numbers
  mean?" cards on the Search and Validate tabs. (`frontend/ui_help.py`)

---

## Possible next improvements

- **Enable the built-in reranker.** `haiku.rag` ships local cross-encoder
  rerankers (e.g. `mxbai`). Turning one on in `get_config()` (`src/hdj/rag.py`)
  re-scores the top candidates with a model that judges *whether a passage
  actually addresses the question* (not just shared vocabulary), widens the
  candidate pool automatically, and pushes citation lines down further. It needs
  a one-time model download and should be validated against the reference
  passages before adopting.
- **Show more surrounding context** around each hit for long arguments, and
  calibrate the Validation "meaning similarity" threshold against a few
  hand-labelled passages so the reported scores stay trustworthy.

---

## Running it

```bash
./run.sh        # macOS/Linux — starts Ollama, pulls the model on first run, opens localhost:8501
run.bat         # Windows
```
Requires [Ollama](https://ollama.com). First run downloads the embedding model (a
few GB); after that everything is offline and local — no data leaves the machine.

## Testing

```bash
uv run python -m pytest tests/ -q     # unit tests (metrics, citation filter, etc.)
```
A headless-browser UI check lives in `scripts/ui_smoke_test.py` (run it against a
local server on port 8502); it boots every tab and asserts the help cards and
metric labels render without errors.
