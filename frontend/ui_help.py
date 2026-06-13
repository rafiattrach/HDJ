"""Plain-language help content for the Corpus Search Tool UI.

Kept separate from app.py so the explanations of every on-screen number live in
one place, can be reviewed without reading the whole app, and can be rendered in
isolation for testing.
"""

import streamlit as st

GLOSSARY_SEARCH = """
Every result is ranked by **Relevance**.

| Number | What it means | Example |
|---|---|---|
| **Relevance** | How closely a passage matches your question, from **0% (unrelated)** to **100% (almost the same meaning)**. It compares *meaning*, so a passage can score high even if it uses different words. | Question *"algorithmic bias in healthcare"* → a passage about biased clinical risk scores ≈ **70%** |
| **Result #N** | Position in the ranked list. #1 is the best match. | — |
| **Cross-lingual** | Also searches a German/English translation of your question, so a German passage can answer an English question (and vice-versa). | — |
| **Hide citations** | Skips passages that are just bibliography / reference-list entries (dense in *(2016)*, *et al.*, DOIs, links). Turn it off if you *want* citations. | — |

💡 *Relevance compares **meaning**, not exact wording — a passage that discusses your topic in different words (or another language) still scores high.*
"""

GLOSSARY_VALIDATION = """
Validation checks how well a **research question** finds the **reference passages** you marked as relevant.

| Number | Plain meaning | Formula | Example | Range |
|---|---|---|---|---|
| **Coverage** | Of *your reference passages*, how many did the question find? Higher = it reaches more of what matters. | found ÷ total reference passages | 12 of 15 found = **80%** | 0–100% |
| **Accuracy** | Of the *passages the search returned*, how many were actually relevant? Higher = less noise. | relevant results ÷ results returned | 14 of 20 relevant = **70%** | 0–100% |
| **Relevance** | How closely a returned passage matches the meaning of the question. | meaning match (question ↔ passage) | ≈ **70%** | 0–100% |
| **Word match** | Share of a reference passage's meaningful words that also appear in the retrieved passage (common words like *the/of* ignored). Used to decide "found". | shared words ÷ reference words | 6 of 8 = **75%** | 0–100% |
| **Meaning similarity** | Do two passages express the *same idea* even if worded differently or in another language? A backup way to count a match. | meaning match (passage ↔ passage) | German ↔ English match = **82%** | 0–100% |

✅ A reference passage counts as **found** if **Word match ≥ threshold** *or* **Meaning similarity ≥ threshold** (you set the thresholds above).

💡 *Coverage and Accuracy answer two different questions: Coverage = did the search **reach** what matters? Accuracy = how much of what it **returned** was on-target? Both are always 0–100%.*
"""


def metrics_glossary(scope: str = "validation"):
    """Always-available, plain-language legend for the numbers on screen."""
    label = (
        "ℹ️ What does **Relevance** mean? (click to read)"
        if scope == "search"
        else "ℹ️ What do **Coverage, Accuracy & Relevance** mean? (click to read)"
    )
    with st.expander(label, expanded=False):
        st.markdown(GLOSSARY_SEARCH if scope == "search" else GLOSSARY_VALIDATION)
