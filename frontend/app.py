"""
Corpus Search Tool – find relevant passages across a document collection.
"""

import asyncio
import html as html_mod
import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse, unquote

import streamlit as st

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hdj import HDJRag, Evaluator, log_event, load_events, generate_report
from src.hdj.audit import clear_events
from src.hdj.evaluate import save_results
from src.hdj.history import load_all_runs, diff_runs, aggregate_query_performance, format_timestamp

# Paths
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
PDFS_DIR = DATA_DIR / "pdfs"
DEFAULTS_DIR = DATA_DIR / "defaults"
DEFAULT_PDFS_DIR = DEFAULTS_DIR / "pdfs"
GOLD_STANDARD_PATH = DATA_DIR / "gold_standard.json"
DEFAULT_GOLD_STANDARD_PATH = DEFAULTS_DIR / "gold_standard.json"
QUERIES_PATH = ROOT / "queries.json"
DEFAULT_QUERIES_PATH = DEFAULTS_DIR / "queries.json"
RESULTS_DIR = ROOT / "results"
DB_PATH = ROOT / "hdj.lancedb"
INDEX_META_PATH = ROOT / "index_meta.json"
AUDIT_PATH = DATA_DIR / "audit_trail.json"
COLLECTIONS_PATH = DATA_DIR / "collections.json"

# Page config
st.set_page_config(
    page_title="Corpus Search Tool",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Force light theme + fix button colors
st.markdown("""
<style>
    /* Force light theme everywhere */
    :root, .stApp, [data-testid="stAppViewContainer"], .main, 
    [data-testid="stHeader"], .block-container {
        background-color: #ffffff !important;
    }
    
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"], p, span, div, label {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #1a1a1a !important;
    }
    
    .main .block-container {
        max-width: 1100px;
        padding-top: 1rem;
    }
    
    h1 { font-weight: 700; font-size: 2rem; color: #1a1a1a !important; }
    h2 { font-weight: 600; font-size: 1.25rem; color: #1a1a1a !important; margin-top: 1rem; }
    h3 { font-weight: 600; font-size: 1rem; color: #374151 !important; }
    
    /* Text areas */
    .stTextArea textarea, .stTextInput input {
        background-color: #ffffff !important;
        color: #1a1a1a !important;
        border: 1px solid #d1d5db !important;
    }
    
    /* Metric cards */
    [data-testid="stMetric"] {
        background: #f9fafb !important;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e5e7eb;
    }
    [data-testid="stMetricValue"] { color: #1a1a1a !important; }
    [data-testid="stMetricLabel"] { color: #6b7280 !important; }
    
    /* ALL BUTTONS - white text on dark background */
    .stButton > button {
        background-color: #1f2937 !important;
        color: #ffffff !important;
        border: none !important;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        font-weight: 500;
    }
    .stButton > button:hover {
        background-color: #374151 !important;
        color: #ffffff !important;
    }
    .stButton > button p, .stButton > button span {
        color: #ffffff !important;
    }
    
    /* Danger button */
    .danger-btn button {
        background-color: #dc2626 !important;
    }
    .danger-btn button:hover {
        background-color: #b91c1c !important;
    }
    
    /* Info boxes */
    .info-box {
        background: #eff6ff !important;
        border: 1px solid #bfdbfe;
        border-radius: 6px;
        padding: 0.75rem 1rem;
        font-size: 0.875rem;
        color: #1e40af !important;
        margin: 0.75rem 0;
    }
    .info-box * { color: #1e40af !important; }
    
    .def-box {
        background: #fefce8 !important;
        border: 1px solid #fef08a;
        border-radius: 6px;
        padding: 0.75rem 1rem;
        font-size: 0.85rem;
        color: #854d0e !important;
        margin: 0.5rem 0;
    }
    .def-box * { color: #854d0e !important; }
    
    .warn-box {
        background: #fef3c7 !important;
        border: 1px solid #fcd34d;
        border-radius: 6px;
        padding: 0.75rem 1rem;
        font-size: 0.875rem;
        color: #92400e !important;
        margin: 0.75rem 0;
    }
    .warn-box * { color: #92400e !important; }
    
    .success-box {
        background: #dcfce7 !important;
        border: 1px solid #86efac;
        border-radius: 6px;
        padding: 0.75rem 1rem;
        font-size: 0.875rem;
        color: #166534 !important;
        margin: 0.75rem 0;
    }
    .success-box * { color: #166534 !important; }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { background-color: #ffffff !important; }
    .stTabs [data-baseweb="tab"] { color: #1a1a1a !important; }
    
    /* Expanders */
    .streamlit-expanderHeader { 
        background-color: #f9fafb !important; 
        color: #1a1a1a !important; 
    }
    
    /* Multiselect tags - white text on dark pills */
    [data-baseweb="tag"] span { color: #ffffff !important; }
    [data-baseweb="tag"] svg { fill: #ffffff !important; }

    /* Hide branding */
    #MainMenu, footer { visibility: hidden; }
    
    hr { margin: 1.5rem 0; border: none; border-top: 1px solid #e5e7eb; }

    /* Overlap highlighting */
    .overlap-hit {
        background-color: #bbf7d0;
        padding: 1px 2px;
        border-radius: 2px;
        color: #166534 !important;
    }
    .overlap-miss {
        background-color: #fecaca;
        padding: 2px 4px;
        border-radius: 3px;
        color: #991b1b !important;
        font-size: 0.875rem;
        line-height: 1.6;
    }
    .score-badge {
        display: inline-block;
        background-color: #dbeafe;
        color: #1e40af !important;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .semantic-badge {
        display: inline-block;
        background-color: #fef3c7;
        color: #92400e !important;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .rank-badge {
        display: inline-block;
        background-color: #ede9fe;
        color: #6d28d9 !important;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .score-badge[title], .rank-badge[title] { cursor: help; }
    .chunk-meta {
        font-size: 0.75rem;
        color: #9ca3af !important;
    }
</style>
""", unsafe_allow_html=True)


# ============ Helper Functions ============

def init_state():
    defaults = {
        "indexed_pdfs": [],
        "index_timestamp": None,
        "eval_results": None,
        "search_results": None,
        "search_translated_query": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def get_pdf_list() -> list[Path]:
    PDFS_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(PDFS_DIR.glob("*.pdf"))


def load_queries() -> dict[str, str]:
    if QUERIES_PATH.exists():
        with open(QUERIES_PATH) as f:
            return json.load(f)
    if DEFAULT_QUERIES_PATH.exists():
        with open(DEFAULT_QUERIES_PATH) as f:
            return json.load(f)
    return {}


def save_queries(queries: dict[str, str]):
    with open(QUERIES_PATH, "w") as f:
        json.dump(queries, f, indent=2)


def load_gold_standard() -> list[dict]:
    if GOLD_STANDARD_PATH.exists():
        with open(GOLD_STANDARD_PATH) as f:
            return json.load(f)
    return []


def save_gold_standard(entries: list[dict]):
    with open(GOLD_STANDARD_PATH, "w") as f:
        json.dump(entries, f, indent=2)


def load_collections() -> dict:
    if COLLECTIONS_PATH.exists():
        with open(COLLECTIONS_PATH) as f:
            return json.load(f)
    return {"collections": {}}


def save_collections(data: dict):
    with open(COLLECTIONS_PATH, "w") as f:
        json.dump(data, f, indent=2)


def add_to_collection(collection_name: str, passage: dict):
    data = load_collections()
    if collection_name not in data["collections"]:
        data["collections"][collection_name] = {
            "created": datetime.now().isoformat(),
            "passages": [],
        }
    data["collections"][collection_name]["passages"].append(passage)
    save_collections(data)


def load_index_meta() -> dict:
    """Load index metadata (timestamp, PDFs)."""
    if INDEX_META_PATH.exists():
        with open(INDEX_META_PATH) as f:
            return json.load(f)
    return {}


def save_index_meta(pdfs: list[str]):
    """Save index metadata."""
    meta = {
        "timestamp": datetime.now().isoformat(),
        "pdfs": pdfs
    }
    with open(INDEX_META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def reset_to_defaults():
    """Reset everything to defaults."""
    log_event(AUDIT_PATH, "reset_all")
    clear_events(AUDIT_PATH)

    # Reset gold standard
    if DEFAULT_GOLD_STANDARD_PATH.exists():
        shutil.copy(DEFAULT_GOLD_STANDARD_PATH, GOLD_STANDARD_PATH)
    
    # Reset queries
    if DEFAULT_QUERIES_PATH.exists():
        shutil.copy(DEFAULT_QUERIES_PATH, QUERIES_PATH)
    
    # Clear user PDFs and restore defaults
    if PDFS_DIR.exists():
        for pdf in PDFS_DIR.glob("*.pdf"):
            pdf.unlink()
    
    if DEFAULT_PDFS_DIR.exists():
        PDFS_DIR.mkdir(parents=True, exist_ok=True)
        for pdf in DEFAULT_PDFS_DIR.glob("*.pdf"):
            shutil.copy(pdf, PDFS_DIR / pdf.name)
    
    # Clear collections
    if COLLECTIONS_PATH.exists():
        COLLECTIONS_PATH.unlink()

    # Clear index
    if DB_PATH.exists():
        shutil.rmtree(DB_PATH)
    if INDEX_META_PATH.exists():
        INDEX_META_PATH.unlink()
    
    # Clear session state
    st.session_state.indexed_pdfs = []
    st.session_state.index_timestamp = None
    st.session_state.eval_results = None
    st.session_state.search_results = None


def get_gold_standard_by_file(entries: list[dict]) -> dict[str, list[tuple[int, dict]]]:
    by_file = {}
    for i, e in enumerate(entries):
        src = e.get("source_file", "unknown")
        if src not in by_file:
            by_file[src] = []
        by_file[src].append((i, e))
    return by_file


async def build_index(force: bool = False) -> tuple[int, list[str]]:
    pdfs = get_pdf_list()
    pdf_names = [p.name for p in pdfs]
    
    async with HDJRag(DB_PATH) as rag:
        count = await rag.index_pdfs(PDFS_DIR, force=force)
        return count, pdf_names


async def run_evaluation(
    queries: dict[str, str],
    limit: int = 20,
    overlap_threshold: float = 0.3,
    semantic_threshold: float = 0.75,
):
    """Run evaluation against gold standard with configurable overlap."""
    evaluator = Evaluator.from_json(
        GOLD_STANDARD_PATH,
        overlap_threshold=overlap_threshold,
        semantic_threshold=semantic_threshold,
    )

    async with HDJRag(DB_PATH) as rag:
        results = []
        for name, query in queries.items():
            result = await evaluator.run_query(rag, name, query, limit)
            results.append(result)

        save_results(results, RESULTS_DIR)
        return results


async def search_documents(query: str, limit: int = 10, cross_lingual: bool = True):
    async with HDJRag(DB_PATH) as rag:
        return await rag.search(query, limit=limit, cross_lingual=cross_lingual)


def _do_build_index(force: bool) -> tuple[int, list[str]]:
    """Build index with per-document progress tracking."""
    pdfs = get_pdf_list()
    pdf_names = [p.name for p in pdfs]
    total = len(pdfs)

    if total == 0:
        return 0, []

    progress = st.progress(0, text="Starting...")
    status = st.status("Preparing documents...", expanded=True)

    loop = asyncio.new_event_loop()
    try:
        rag = HDJRag(DB_PATH)
        loop.run_until_complete(rag.__aenter__())
        try:
            doc_count = loop.run_until_complete(rag.document_count())

            if not force and doc_count >= total:
                status.update(label="Documents already prepared", state="complete")
                progress.empty()
                return doc_count, pdf_names

            if force:
                status.write("Clearing old index...")
                loop.run_until_complete(rag.clear_documents())

            for i, pdf in enumerate(pdfs):
                progress.progress(
                    i / total,
                    text=f"Processing: {pdf.name} ({i + 1}/{total})",
                )
                status.write(f"📄 {pdf.name}")
                loop.run_until_complete(rag.index_single_pdf(pdf))

            progress.progress(1.0, text="Complete!")
            status.update(
                label=f"Prepared {total} documents", state="complete"
            )
            return total, pdf_names
        finally:
            loop.run_until_complete(rag.__aexit__(None, None, None))
    finally:
        loop.close()


def _do_run_evaluation(
    queries: dict[str, str],
    limit: int,
    overlap_threshold: float,
    semantic_threshold: float,
) -> list:
    """Run evaluation with per-query progress tracking."""
    evaluator = Evaluator.from_json(
        GOLD_STANDARD_PATH,
        overlap_threshold=overlap_threshold,
        semantic_threshold=semantic_threshold,
    )
    total = len(queries)

    progress = st.progress(0, text="Preparing...")
    status = st.status("Running validation...", expanded=True)

    loop = asyncio.new_event_loop()
    try:
        rag = HDJRag(DB_PATH)
        loop.run_until_complete(rag.__aenter__())
        try:
            # Pre-compute gold embeddings (heavy, one-time step)
            status.write("Computing reference passage embeddings...")
            try:
                evaluator._gold_embeddings = loop.run_until_complete(
                    rag.embed_texts(evaluator.gold_standard)
                )
            except Exception:
                status.write(
                    "⚠️ Could not pre-compute embeddings (is Ollama running?)"
                )

            results = []
            for i, (name, query) in enumerate(queries.items()):
                progress.progress(
                    i / total,
                    text=f"Testing: {name} ({i + 1}/{total})",
                )
                result = loop.run_until_complete(
                    evaluator.run_query(rag, name, query, limit)
                )
                results.append(result)
                emoji = (
                    "🟢" if result.recall >= 0.8
                    else "🟡" if result.recall >= 0.5
                    else "🔴"
                )
                status.write(
                    f"{emoji} {name} — Coverage: {result.recall:.0%}"
                )

            save_results(results, RESULTS_DIR)
            progress.progress(1.0, text="Complete!")
            status.update(label="Validation complete!", state="complete")
            return results
        finally:
            loop.run_until_complete(rag.__aexit__(None, None, None))
    finally:
        loop.close()


_MATCH_TYPE_LABELS = {
    "substring": "exact text match",
    "word_overlap": "shared keywords",
    "semantic": "cross-lingual meaning match",
}

# ============ UI Helpers ============

def show_info(text: str):
    st.markdown(f'<div class="info-box">{text}</div>', unsafe_allow_html=True)

def show_def(title: str, text: str):
    st.markdown(f'<div class="def-box"><strong>{title}:</strong> {text}</div>', unsafe_allow_html=True)

def show_warn(text: str):
    st.markdown(f'<div class="warn-box">⚠️ {text}</div>', unsafe_allow_html=True)

def show_success(text: str):
    st.markdown(f'<div class="success-box">✓ {text}</div>', unsafe_allow_html=True)

def _pretty_doc_name(document_uri: str | None) -> str:
    """Best-effort filename from a document URI/path."""
    if not document_uri:
        return "Unknown PDF"
    try:
        parsed = urlparse(document_uri)
        # Handle file:// URIs
        if parsed.scheme in {"file"} and parsed.path:
            return Path(unquote(parsed.path)).name or document_uri
        # Handle plain paths
        if parsed.scheme == "" and document_uri:
            return Path(document_uri).name or document_uri
        # Fallback
        return Path(unquote(parsed.path)).name or document_uri
    except Exception:
        return document_uri


def highlight_overlap(text: str, overlapping_words: list[str]) -> str:
    """Return HTML with overlapping words wrapped in highlight spans."""
    if not overlapping_words:
        return html_mod.escape(text)

    word_set = {w.lower() for w in overlapping_words}
    # Split on word boundaries but keep delimiters (punctuation/spaces)
    tokens = re.split(r'(\W+)', text)
    parts = []
    for token in tokens:
        # Strip punctuation for matching but preserve original token
        stripped = re.sub(r'[^\w]', '', token).lower()
        if stripped and stripped in word_set:
            parts.append(f'<span class="overlap-hit">{html_mod.escape(token)}</span>')
        else:
            parts.append(html_mod.escape(token))
    return "".join(parts)


# ============ Main App ============

init_state()

# Load index meta on startup
index_meta = load_index_meta()
if index_meta:
    st.session_state.indexed_pdfs = index_meta.get("pdfs", [])
    st.session_state.index_timestamp = index_meta.get("timestamp")

# ===== HEADER =====
col1, col2 = st.columns([4, 1])
with col1:
    st.title("📚 Corpus Search Tool")
    st.caption("Find relevant passages across your document collection")
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="danger-btn">', unsafe_allow_html=True)
    if st.button("🚨 Reset All", help="Reset PDFs, index, gold standard, and queries to defaults"):
        reset_to_defaults()
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

st.divider()

# ===== TABS =====
tab1, tab2, tab3, tab_findings, tab4, tab5, tab6 = st.tabs([
    "1. Documents",
    "2. Prepare Documents",
    "3. Search",
    "📌 My Findings",
    "4. Validate Search",
    "5. Activity Log",
    "6. History",
])


# ============ TAB 1: Documents ============
with tab1:
    st.header("Step 1: Add PDF Documents")
    
    show_info("Upload the PDF documents you want to analyze. After uploading, go to 'Prepare Documents' to make them searchable.")
    
    # Upload
    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    
    if uploaded_files:
        for f in uploaded_files:
            dest = PDFS_DIR / f.name
            if not dest.exists():
                with open(dest, "wb") as out:
                    out.write(f.getbuffer())
                log_event(AUDIT_PATH, "pdf_uploaded", {"filename": f.name})
                st.success(f"✓ Added {f.name}")
    
    # Current PDFs
    pdfs = get_pdf_list()
    indexed_pdfs = set(st.session_state.get("indexed_pdfs", []))
    
    st.subheader(f"Current PDFs ({len(pdfs)})")
    
    if pdfs:
        for pdf in pdfs:
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                # Show if indexed
                if pdf.name in indexed_pdfs:
                    st.markdown(f"📄 **{pdf.name}** ✓ indexed")
                else:
                    st.markdown(f"📄 {pdf.name} _(not indexed)_")
            with col3:
                if st.button("🗑️", key=f"del_{pdf.name}", help="Remove this PDF"):
                    log_event(AUDIT_PATH, "pdf_deleted", {"filename": pdf.name})
                    pdf.unlink()
                    # Warn if it was indexed
                    if pdf.name in indexed_pdfs:
                        st.warning("This PDF was indexed. Rebuild the index to update.")
                    st.rerun()
    else:
        st.info("No PDFs yet. Upload some above, or click 'Reset All' to restore defaults.")


# ============ TAB 2: Build Index ============
with tab2:
    st.header("Step 2: Prepare Documents")

    show_def("What this does", "Prepares your PDFs so they can be searched. You must re-run this when PDFs change.")
    
    pdfs = get_pdf_list()
    pdf_names = [p.name for p in pdfs]
    
    # Current index status
    st.subheader("Index Status")
    
    index_meta = load_index_meta()
    
    if index_meta and index_meta.get("pdfs"):
        timestamp = index_meta.get("timestamp", "Unknown")
        try:
            ts = datetime.fromisoformat(timestamp)
            timestamp_str = ts.strftime("%Y-%m-%d %H:%M:%S")
        except:
            timestamp_str = timestamp
        
        indexed_pdfs = index_meta.get("pdfs", [])
        
        show_success(f"Index built on {timestamp_str}")
        
        with st.expander(f"📦 Indexed PDFs ({len(indexed_pdfs)})", expanded=True):
            for name in indexed_pdfs:
                if name in pdf_names:
                    st.markdown(f"✓ {name}")
                else:
                    st.markdown(f"⚠️ {name} _(file removed)_")
        
        # Check for unindexed PDFs
        unindexed = set(pdf_names) - set(indexed_pdfs)
        if unindexed:
            show_warn(f"New PDFs not in index: {', '.join(unindexed)}. Click 'Rebuild' to include them.")
    else:
        show_warn("Documents not prepared yet. Click 'Prepare Documents' below.")
    
    # Build buttons
    st.subheader("Build")
    st.markdown(f"**PDFs available:** {len(pdfs)}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔨 Prepare Documents", disabled=len(pdfs) == 0, help="Prepare (skips existing)"):
            count, names = _do_build_index(force=False)
            meta = save_index_meta(names)
            st.session_state.indexed_pdfs = names
            st.session_state.index_timestamp = meta["timestamp"]
            log_event(AUDIT_PATH, "index_built", {"pdfs": names, "force": False})
            st.rerun()

    with col2:
        if st.button("🔄 Re-prepare Documents", disabled=len(pdfs) == 0, help="Delete and rebuild from scratch"):
            count, names = _do_build_index(force=True)
            meta = save_index_meta(names)
            st.session_state.indexed_pdfs = names
            st.session_state.index_timestamp = meta["timestamp"]
            log_event(AUDIT_PATH, "index_built", {"pdfs": names, "force": True})
            st.rerun()


# ============ TAB 3: Search ============
with tab4:
    st.header("Step 4: Validate Search Quality")
    
    show_info("Test how well different research questions find relevant passages. Define your reference passages (known relevant text), then run questions against them.")
    
    # Check if index exists
    index_meta = load_index_meta()
    if not index_meta.get("pdfs"):
        show_warn("Prepare your documents first (Step 2) before validating.")
        st.stop()
    
    indexed_pdfs = index_meta.get("pdfs", [])
    
    # ===== REFERENCE PASSAGES SECTION =====
    st.subheader("Reference Passages")
    
    col1, col2 = st.columns(2)
    with col1:
        with st.expander("📖 What are reference passages?", expanded=False):
            st.markdown("""
            **Reference passages** are sections from your PDFs that you already
            know are relevant to your research. You paste them here so the tool
            can measure how well each research question finds them.

            - **Coverage** = What % of your reference passages did the search find? (Higher = better)
            - **Accuracy** = What % of returned results actually matched a reference passage? (Higher = less noise)
            """)
    with col2:
        with st.expander("🔧 How does the search work?", expanded=False):
            st.markdown("""
            The tool reads each PDF and breaks it into short passages
            (roughly half a page each). When you type a research question,
            it finds the passages whose meaning is closest to your question
            and ranks them by relevance (0–100%).

            **Validation matching:**
            - **Word match** — counts how many meaningful words are shared
              between a reference passage and a retrieved passage (common
              words like "the" and "of" are ignored)
            - **Meaning similarity** — checks whether two passages express
              the same idea, even if worded differently or in different
              languages (German ↔ English)
            """)
    
    gold = load_gold_standard()
    by_file = get_gold_standard_by_file(gold)
    
    # Show gold standard sections (text only, no notes)
    if gold:
        st.markdown(f"**{len(gold)} reference passages** defined")
        
        for filename, entries_with_idx in by_file.items():
            # Check if this file is in the index
            in_index = filename in indexed_pdfs
            status = "" if in_index else " ⚠️ not in index"
            
            with st.expander(f"📄 {filename} ({len(entries_with_idx)} sections){status}", expanded=False):
                if not in_index:
                    show_warn(f"This PDF is not in the current index. Add it and rebuild.")
                
                for idx, entry in entries_with_idx:
                    st.markdown(f"**Section {idx+1}**")
                    
                    new_text = st.text_area(
                        "Text", value=entry["text"], height=100,
                        key=f"gs_text_{idx}", label_visibility="collapsed"
                    )
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("💾 Save", key=f"gs_save_{idx}"):
                            gold[idx]["text"] = new_text
                            save_gold_standard(gold)
                            log_event(AUDIT_PATH, "gold_standard_edited", {
                                "source_file": entry.get("source_file", "unknown"),
                                "text_preview": new_text[:100],
                            })
                            st.success("Saved!")
                            st.rerun()
                    with col2:
                        if st.button("🗑️ Delete", key=f"gs_del_{idx}"):
                            log_event(AUDIT_PATH, "gold_standard_deleted", {
                                "source_file": entry.get("source_file", "unknown"),
                                "text_preview": entry.get("text", "")[:100],
                            })
                            del gold[idx]
                            save_gold_standard(gold)
                            st.rerun()
                    
                    st.divider()
    else:
        st.info("No reference passages defined yet. Add some below.")
    
    # Add new gold standard section - only for indexed PDFs
    st.markdown("**Add a section:**")
    
    if indexed_pdfs:
        new_file = st.selectbox("PDF (must be in index)", indexed_pdfs, key="gs_new_file")
        new_text = st.text_area(
            "Paste relevant text from this PDF",
            placeholder="Copy and paste a section that's relevant to data justice...",
            key="gs_new_text"
        )
        
        if st.button("➕ Add Reference Passage"):
            if new_text.strip():
                gold.append(
                    {
                        "source_file": new_file,
                        "text": new_text.strip(),
                    }
                )
                save_gold_standard(gold)
                log_event(AUDIT_PATH, "gold_standard_added", {
                    "source_file": new_file,
                    "text_preview": new_text.strip()[:100],
                })
                st.success("Added!")
                st.rerun()
            else:
                st.error("Please enter some text.")
    else:
        st.info("Prepare some PDFs first to add reference passages.")
    
    st.divider()
    
    # ===== RESEARCH QUESTIONS SECTION =====
    st.subheader("Research Questions")
    
    queries = load_queries()
    
    if queries:
        st.markdown(f"**{len(queries)} research questions** defined")
        
        for name, text in list(queries.items()):
            with st.expander(f"📝 {name}", expanded=False):
                new_text = st.text_area(
                    "Query", value=text, height=100,
                    key=f"q_edit_{name}", label_visibility="collapsed"
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("💾 Save", key=f"q_save_{name}"):
                        queries[name] = new_text
                        save_queries(queries)
                        log_event(AUDIT_PATH, "query_edited", {
                            "name": name,
                            "text_preview": new_text[:100],
                        })
                        st.success("Saved!")
                        st.rerun()
                with col2:
                    if st.button("🗑️ Delete", key=f"q_del_{name}"):
                        log_event(AUDIT_PATH, "query_deleted", {"name": name})
                        del queries[name]
                        save_queries(queries)
                        st.rerun()
    else:
        st.info("No research questions defined. Add one below.")
    
    # Add new research question
    st.markdown("**Add a question:**")
    col1, col2 = st.columns([1, 3])
    with col1:
        new_q_name = st.text_input("Name", placeholder="my_query", key="new_q_name")
    with col2:
        new_q_text = st.text_area("Question text", placeholder="What to search for...", height=80, key="new_q_text")
    
    if st.button("➕ Add Question") and new_q_name and new_q_text:
        queries[new_q_name] = new_q_text
        save_queries(queries)
        log_event(AUDIT_PATH, "query_added", {
            "name": new_q_name,
            "text_preview": new_q_text[:100],
        })
        st.success(f"Added '{new_q_name}'")
        st.rerun()
    
    st.divider()
    
    # ===== RUN VALIDATION =====
    st.subheader("Run Validation")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Questions", len(queries))
    with col2:
        st.metric("Reference Passages", len(gold))
    with col3:
        limit = st.number_input(
            "Results per question",
            min_value=5,
            max_value=100,
            value=20,
            help="How many passages to retrieve for each question.",
        )
    with col4:
        overlap_threshold = st.slider(
            "Match strictness",
            min_value=0.1,
            max_value=0.9,
            value=0.3,
            step=0.05,
            help=(
                "How much of a reference passage must appear in a retrieved passage "
                "to count as 'found'.\n\n"
                "0.3 = lenient (≥30% of words match)\n"
                "0.5 = stricter (≥50%)\n"
                "0.7+ = very strict"
            ),
        )
    with col5:
        semantic_threshold = st.slider(
            "Meaning similarity threshold",
            min_value=0.5,
            max_value=0.95,
            value=0.75,
            step=0.05,
            help=(
                "Minimum meaning similarity for a cross-lingual match.\n\n"
                "When word matching fails but the meaning is similar enough, "
                "the reference passage counts as found.\n\n"
                "0.75 = balanced (default)\n"
                "0.85+ = stricter"
            ),
        )
    
    can_run = len(queries) > 0 and len(gold) > 0
    
    if not can_run:
        if len(queries) == 0:
            st.warning("Add at least one research question above.")
        if len(gold) == 0:
            st.warning("Add at least one reference passage above.")
    
    if st.button("▶️ Run Validation", disabled=not can_run):
        # Clear old results first
        st.session_state.eval_results = None
        results = _do_run_evaluation(
            queries, limit, overlap_threshold, semantic_threshold,
        )
        st.session_state.eval_results = results
        best = max(results, key=lambda r: r.recall)
        log_event(AUDIT_PATH, "evaluation_run", {
            "query_names": list(queries.keys()),
            "gold_sections_count": len(gold),
            "results_limit": limit,
            "overlap_threshold": overlap_threshold,
            "semantic_threshold": semantic_threshold,
            "best_query": best.name,
            "best_recall": round(best.recall, 3),
        })
        st.rerun()
    
    # Results
    if st.session_state.eval_results:
        st.divider()
        st.subheader("Results")
        
        results = st.session_state.eval_results
        sorted_results = sorted(results, key=lambda x: x.recall, reverse=True)
        
        # Summary
        for r in sorted_results:
            emoji = "🟢" if r.recall >= 0.8 else "🟡" if r.recall >= 0.5 else "🔴"
            cols = st.columns([2, 1, 1, 2])
            with cols[0]:
                st.markdown(f"{emoji} **{r.name}**")
            with cols[1]:
                st.markdown(f'<span title="What percentage of reference passages were found by this question">Coverage: <strong>{r.recall:.0%}</strong></span>', unsafe_allow_html=True)
            with cols[2]:
                st.markdown(f'<span title="What percentage of retrieved results were actually relevant">Accuracy: {r.precision:.0%}</span>', unsafe_allow_html=True)
            with cols[3]:
                st.markdown(f'<span title="How many reference passages were matched out of the total">Found {r.found}/{r.total_gold} reference passages</span>', unsafe_allow_html=True)
        
        st.caption(
            "Coverage = % of reference passages found | "
            "Accuracy = % of results that were relevant | "
            "Hover any value for details"
        )

        st.divider()
        best = sorted_results[0]
        show_success(f"Best: '{best.name}' with {best.recall:.0%} coverage")
        
        # Details
        for r in sorted_results:
            with st.expander(f"📊 {r.name} — {r.recall:.0%} coverage"):
                st.markdown("**Research question:**")
                st.markdown(f'<div style="background:#f9fafb; border:1px solid #e5e7eb; border-radius:6px; padding:0.75rem; font-size:0.9rem; color:#1a1a1a; margin:0.5rem 0; white-space:pre-wrap;">{html_mod.escape(r.query)}</div>', unsafe_allow_html=True)

                # Retrieved chunks summary
                if r.retrieved_chunks:
                    with st.expander(f"📦 Retrieved Passages ({len(r.retrieved_chunks)})", expanded=False):
                        for chunk in r.retrieved_chunks[:20]:
                            doc = _pretty_doc_name(chunk.document_uri)
                            pages = ", ".join(map(str, chunk.page_numbers)) if chunk.page_numbers else "—"
                            preview = chunk.content[:120].replace("\n", " ")
                            if len(chunk.content) > 120:
                                preview += "..."
                            st.markdown(
                                f'<span class="rank-badge" title="Position in search results">#{chunk.rank}</span> '
                                f'<span class="score-badge" title="Relevance score: how closely this passage matches your question (0–100%)">Relevance {chunk.score:.1%}</span> '
                                f'{html_mod.escape(doc)} · Pages {pages}<br>'
                                f'<span class="chunk-meta">{html_mod.escape(preview)}</span>',
                                unsafe_allow_html=True,
                            )
                            st.markdown("---")

                # Matched sections
                if r.match_details:
                    st.markdown(f"**Matched passages ({len(r.match_details)}):**")
                    for i, detail in enumerate(r.match_details):
                        sem_label = f" · {detail.semantic_similarity:.0%} meaning similarity" if detail.semantic_similarity > 0 else ""
                        match_label = _MATCH_TYPE_LABELS.get(detail.match_type, detail.match_type)
                        with st.expander(f"Match {i+1} — {match_label} · {detail.overlap_ratio:.0%} word match{sem_label}", expanded=False):
                            col_g, col_c = st.columns(2)
                            with col_g:
                                st.markdown("**Reference passage:**")
                                highlighted = highlight_overlap(detail.gold_text_preview, detail.overlapping_words)
                                st.markdown(highlighted, unsafe_allow_html=True)
                            with col_c:
                                if detail.matched_chunk:
                                    chunk = detail.matched_chunk
                                    st.markdown("**Best matching passage:**")
                                    badges = (
                                        f'<span class="rank-badge" title="Position in search results">#{chunk.rank}</span> '
                                        f'<span class="score-badge" title="Relevance score: how closely this passage matches your question (0–100%)">Relevance {chunk.score:.1%}</span>'
                                    )
                                    if detail.semantic_similarity > 0:
                                        badges += f' <span class="semantic-badge">{detail.semantic_similarity:.0%} meaning similarity</span>'
                                    st.markdown(badges, unsafe_allow_html=True)
                                    chunk_preview = chunk.content[:200].replace("\n", " ")
                                    if len(chunk.content) > 200:
                                        chunk_preview += "..."
                                    chunk_highlighted = highlight_overlap(chunk_preview, detail.overlapping_words)
                                    st.markdown(chunk_highlighted, unsafe_allow_html=True)

                # Missed sections
                if r.miss_details:
                    st.markdown(f"**Missed passages ({len(r.miss_details)}):**")
                    for i, detail in enumerate(r.miss_details):
                        sem_label = f" · {detail.semantic_similarity:.0%} meaning similarity" if detail.semantic_similarity > 0 else ""
                        with st.expander(f"Missed {i+1} — only {detail.overlap_ratio:.0%} word match{sem_label} with nearest passage", expanded=True):
                            st.markdown(
                                f'<div class="overlap-miss">{html_mod.escape(detail.gold_text_preview)}</div>',
                                unsafe_allow_html=True,
                            )
                            if detail.matched_chunk:
                                chunk = detail.matched_chunk
                                doc = _pretty_doc_name(chunk.document_uri)
                                badges = (
                                    f'<span class="rank-badge" title="Position in search results">#{chunk.rank}</span> '
                                    f'<span class="score-badge" title="Relevance score: how closely this passage matches your question (0–100%)">Relevance {chunk.score:.1%}</span>'
                                )
                                if detail.semantic_similarity > 0:
                                    badges += f' <span class="semantic-badge">{detail.semantic_similarity:.0%} meaning similarity</span>'
                                st.markdown(
                                    f'**Nearest passage:** {badges} '
                                    f'· {html_mod.escape(doc)}',
                                    unsafe_allow_html=True,
                                )
                                st.markdown(
                                    f"**{detail.overlap_ratio:.0%}** word match "
                                    f"({len(detail.overlapping_words)} shared words) — "
                                    f"below the {int(overlap_threshold * 100)}% threshold needed to count as a match"
                                )
                                chunk_preview = chunk.content[:200].replace("\n", " ")
                                if len(chunk.content) > 200:
                                    chunk_preview += "..."
                                st.markdown(f'<div style="background:#f9fafb; border:1px solid #e5e7eb; border-radius:6px; padding:0.75rem; font-size:0.9rem; color:#1a1a1a; margin:0.5rem 0; white-space:pre-wrap;">{html_mod.escape(chunk_preview)}</div>', unsafe_allow_html=True)
                            else:
                                st.markdown("No passages retrieved.")
                elif r.missed_texts:
                    # Fallback for old-format results
                    st.markdown(f"**Missed passages ({len(r.missed_texts)}):**")
                    for i, text in enumerate(r.missed_texts):
                        with st.expander(f"Missed {i+1} ({len(text)} chars)", expanded=True):
                            st.markdown(f'<div style="background:#f9fafb; border:1px solid #e5e7eb; border-radius:6px; padding:0.75rem; font-size:0.9rem; color:#1a1a1a; margin:0.5rem 0; white-space:pre-wrap;">{html_mod.escape(text)}</div>', unsafe_allow_html=True)

        # Export Report button
        st.divider()
        report_config = {
            "embedding_model": "Qwen3-Embedding-4B (via Ollama)",
            "chunk_size": 512,
            "search_method": "Hybrid (vector + full-text with RRF reranking)",
            "results_limit": limit,
            "overlap_threshold": overlap_threshold,
            "semantic_threshold": semantic_threshold,
            "indexed_pdfs": list(index_meta.get("pdfs", [])),
        }
        report_md = generate_report(
            results=results,
            queries=queries,
            gold_standard=gold,
            config=report_config,
            audit_events=load_events(AUDIT_PATH),
        )
        st.download_button(
            "📥 Export Report",
            data=report_md,
            file_name=f"search_validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
        )


# ============ TAB 4: Validate Search ============
with tab3:
    st.header("Step 3: Search Documents")
    
    show_info("Search across all your documents with any research question.")
    
    # Check if index exists
    index_meta = load_index_meta()
    if not index_meta.get("pdfs"):
        show_warn("Prepare your documents first (Step 2) before searching.")
        st.stop()
    
    query = st.text_area(
        "What are you looking for?",
        placeholder="Type a research question or describe what you want to find...",
        height=100
    )

    if not query:
        st.caption(
            'Example questions: _"How is algorithmic bias discussed in the context of healthcare?"_ · '
            '_"What evidence exists for disparities in data collection?"_ · '
            '_"Which policies address equity in health data governance?"_'
        )

    col1, col2, col3 = st.columns([2, 1, 1])
    with col2:
        limit = st.number_input("Max results", min_value=5, max_value=2000, value=10, key="search_limit", help="More results = more comprehensive but slower to scroll through.")
    with col3:
        cross_lingual = st.checkbox(
            "Cross-lingual",
            value=True,
            help="Also search with the question translated to the other language (DE↔EN). Uses offline translation.",
        )

    if st.button("🔍 Search", disabled=not query):
        translated_query = None
        if cross_lingual:
            try:
                from src.hdj.translate import translate_query as _tq
                _, translated_query = _tq(query)
            except Exception:
                pass
        with st.spinner("Searching..."):
            results = asyncio.run(search_documents(query, limit, cross_lingual=cross_lingual))
            st.session_state.search_results = results
            st.session_state.search_translated_query = translated_query
            st.session_state.search_query = query
            log_event(AUDIT_PATH, "search_performed", {
                "query_preview": query[:100],
                "results_count": len(results),
                "limit": limit,
                "cross_lingual": cross_lingual,
            })
        st.rerun()

    if st.session_state.search_results:
        st.divider()
        translated_q = st.session_state.get("search_translated_query")
        if translated_q:
            st.caption(f"Also searched with translation: _{translated_q}_")
        results = st.session_state.search_results

        # Summary line
        doc_names = {_pretty_doc_name(r.get("document_uri")) for r in results}
        st.markdown(f"**Found {len(results)} relevant passages across {len(doc_names)} documents**")

        for i, r in enumerate(results):
            pages = r.get("page_numbers", [])
            content = r.get("content", "")
            score = r.get("score", 0.0)
            doc = _pretty_doc_name(r.get("document_uri"))

            # Page label
            if pages:
                if len(pages) == 1:
                    page_label = f"Page {pages[0]}"
                else:
                    page_label = f"Pages {', '.join(map(str, pages))}"
            else:
                page_label = ""

            label = f"Result {i+1} · Relevance: {score:.1%} · {doc}"
            if page_label:
                label += f" · {page_label}"

            with st.expander(label, expanded=i < 3):
                # Document name as header
                st.markdown(f"**📄 {doc}**" + (f" — {page_label}" if page_label else ""))

                # Relevance progress bar
                st.progress(min(score, 1.0), text=f"Relevance: {score:.1%}")

                # Passage content as styled div instead of disabled textarea
                st.markdown(
                    f'<div style="background:#f9fafb; border:1px solid #e5e7eb; border-radius:6px; '
                    f'padding:0.75rem; font-size:0.9rem; color:#1a1a1a; margin:0.5rem 0; '
                    f'white-space:pre-wrap; line-height:1.6; max-height:400px; overflow-y:auto;">'
                    f'{html_mod.escape(content)}</div>',
                    unsafe_allow_html=True,
                )

                # Copy Citation button
                citation_text = content[:150].replace("\n", " ")
                if len(content) > 150:
                    citation_text += "..."
                citation = f'"{citation_text}" ({doc}'
                if page_label:
                    citation += f", {page_label}"
                citation += ")"
                col_cite, col_save = st.columns(2)
                with col_cite:
                    if st.button("📋 Copy Citation", key=f"cite_{i}"):
                        st.code(citation, language=None)
                        st.caption("Select and copy the text above.")
                with col_save:
                    collections_data = load_collections()
                    existing_collections = list(collections_data["collections"].keys())
                    save_options = existing_collections + ["+ New collection..."]
                    selected_col = st.selectbox(
                        "Save to",
                        save_options,
                        key=f"save_col_{i}",
                        label_visibility="collapsed",
                        placeholder="Save to collection...",
                    )
                    if selected_col == "+ New collection...":
                        new_col_name = st.text_input(
                            "Collection name",
                            key=f"new_col_name_{i}",
                            placeholder="Enter collection name...",
                        )
                    if st.button("📌 Save Passage", key=f"save_{i}"):
                        target_collection = selected_col
                        if selected_col == "+ New collection...":
                            new_col_name = st.session_state.get(f"new_col_name_{i}", "").strip()
                            if not new_col_name:
                                st.warning("Please enter a name for the new collection.")
                                st.stop()
                            target_collection = new_col_name
                        search_q = st.session_state.get("search_query", "")
                        add_to_collection(target_collection, {
                            "content": content,
                            "document": doc,
                            "page_numbers": pages,
                            "relevance_score": round(score, 4),
                            "saved_at": datetime.now().isoformat(),
                            "search_query": search_q,
                        })
                        log_event(AUDIT_PATH, "passage_saved", {
                            "collection": target_collection,
                            "document": doc,
                        })
                        st.success(f"Saved to '{target_collection}'!")
                        st.rerun()


# ============ TAB: My Findings ============
with tab_findings:
    st.header("📌 My Findings")

    collections_data = load_collections()
    all_collections = collections_data.get("collections", {})

    total_passages = sum(len(c["passages"]) for c in all_collections.values())

    if not all_collections:
        st.info("No saved passages yet. Use the Search tab to find passages and save them here.")
    else:
        st.markdown(f"**{len(all_collections)} collections** with **{total_passages} saved passages**")

        # Export buttons
        col_csv, col_md = st.columns(2)
        with col_csv:
            # Build CSV
            csv_lines = ["Collection,Document,Page Numbers,Relevance,Saved At,Search Query,Passage"]
            for cname, cdata in all_collections.items():
                for p in cdata["passages"]:
                    escaped = p["content"].replace('"', '""').replace("\n", " ")
                    pgs = "; ".join(map(str, p.get("page_numbers", [])))
                    csv_lines.append(
                        f'"{cname}","{p.get("document", "")}","{pgs}",'
                        f'{p.get("relevance_score", 0)},"{p.get("saved_at", "")}",'
                        f'"{p.get("search_query", "")}","{escaped}"'
                    )
            st.download_button(
                "📥 Export CSV",
                data="\n".join(csv_lines),
                file_name=f"findings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )
        with col_md:
            # Build Markdown
            md_lines = ["# My Findings", ""]
            for cname, cdata in all_collections.items():
                md_lines.append(f"## {cname}")
                md_lines.append(f"Created: {cdata.get('created', '')}")
                md_lines.append("")
                for j, p in enumerate(cdata["passages"], 1):
                    preview = p["content"][:200].replace("\n", " ")
                    if len(p["content"]) > 200:
                        preview += "..."
                    pgs = ", ".join(map(str, p.get("page_numbers", [])))
                    md_lines.append(f'{j}. **{p.get("document", "")}**' + (f" (p. {pgs})" if pgs else ""))
                    md_lines.append(f'   > "{preview}"')
                    md_lines.append(f'   - Relevance: {p.get("relevance_score", 0):.1%}')
                    md_lines.append(f'   - Search query: {p.get("search_query", "")}')
                    md_lines.append("")
                md_lines.append("")
            st.download_button(
                "📥 Export Markdown",
                data="\n".join(md_lines),
                file_name=f"findings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
            )

        st.divider()

        # Display each collection
        for cname, cdata in all_collections.items():
            passages = cdata["passages"]
            with st.expander(f"📁 {cname} ({len(passages)} passages)", expanded=True):
                # Delete collection button
                if st.button(f"🗑️ Delete collection", key=f"del_col_{cname}"):
                    del all_collections[cname]
                    save_collections(collections_data)
                    log_event(AUDIT_PATH, "collection_deleted", {"collection": cname})
                    st.rerun()

                for j, p in enumerate(passages):
                    doc_name = p.get("document", "Unknown")
                    pgs = p.get("page_numbers", [])
                    page_str = f"Page {', '.join(map(str, pgs))}" if pgs else ""

                    st.markdown(f"**📄 {doc_name}**" + (f" — {page_str}" if page_str else ""))
                    st.progress(min(p.get("relevance_score", 0), 1.0), text=f"Relevance: {p.get('relevance_score', 0):.1%}")

                    preview = p["content"][:300].replace("\n", " ")
                    if len(p["content"]) > 300:
                        preview += "..."
                    st.markdown(
                        f'<div style="background:#f9fafb; border:1px solid #e5e7eb; border-radius:6px; '
                        f'padding:0.75rem; font-size:0.85rem; color:#1a1a1a; margin:0.25rem 0; '
                        f'white-space:pre-wrap; line-height:1.5;">'
                        f'{html_mod.escape(preview)}</div>',
                        unsafe_allow_html=True,
                    )

                    col_meta, col_del = st.columns([3, 1])
                    with col_meta:
                        st.caption(f"Query: _{p.get('search_query', '')}_ · Saved: {p.get('saved_at', '')[:16]}")
                    with col_del:
                        if st.button("🗑️ Remove", key=f"del_p_{cname}_{j}"):
                            passages.pop(j)
                            save_collections(collections_data)
                            log_event(AUDIT_PATH, "passage_removed", {
                                "collection": cname,
                                "document": doc_name,
                            })
                            st.rerun()
                    st.divider()


# ============ TAB 5: Activity Log ============
with tab5:
    st.header("Activity Log")

    show_info("All actions taken in this tool are logged here for transparency and reproducibility.")

    audit_events = load_events(AUDIT_PATH)

    # Action type filter
    if audit_events:
        all_actions = sorted({e.get("action", "") for e in audit_events})
        selected_actions = st.multiselect(
            "Filter by action type",
            options=all_actions,
            default=all_actions,
        )

        filtered = [e for e in audit_events if e.get("action") in selected_actions]
        filtered.reverse()  # newest first

        st.markdown(f"**{len(filtered)} events** (of {len(audit_events)} total)")

        ACTION_LABELS = {
            "pdf_uploaded": "PDF Uploaded",
            "pdf_deleted": "PDF Deleted",
            "index_built": "Documents Prepared",
            "gold_standard_added": "Reference Passage Added",
            "gold_standard_edited": "Reference Passage Edited",
            "gold_standard_deleted": "Reference Passage Deleted",
            "query_added": "Question Added",
            "query_edited": "Question Edited",
            "query_deleted": "Question Deleted",
            "evaluation_run": "Validation Run",
            "search_performed": "Search Performed",
            "passage_saved": "Passage Saved",
            "passage_removed": "Passage Removed",
            "collection_deleted": "Collection Deleted",
            "reset_all": "Reset All",
        }

        for event in filtered:
            ts = event.get("timestamp", "")
            action = event.get("action", "")
            details = event.get("details", {})
            label = ACTION_LABELS.get(action, action)

            # Build a compact details summary
            parts = []
            for k, v in details.items():
                if isinstance(v, list):
                    parts.append(f"{k}: {', '.join(str(x) for x in v)}")
                elif isinstance(v, str) and len(v) > 60:
                    parts.append(f"{k}: {v[:60]}...")
                else:
                    parts.append(f"{k}: {v}")
            detail_str = " · ".join(parts) if parts else ""

            try:
                ts_formatted = datetime.fromisoformat(ts).strftime("%Y-%m-%d %H:%M:%S")
            except (ValueError, TypeError):
                ts_formatted = ts

            if detail_str:
                st.markdown(f"`{ts_formatted}` **{label}** — {detail_str}")
            else:
                st.markdown(f"`{ts_formatted}` **{label}**")

        st.divider()

        # Export as Markdown
        export_lines = ["# Activity Log", ""]
        for event in filtered:
            ts = event.get("timestamp", "")
            action = event.get("action", "")
            details = event.get("details", {})
            label = ACTION_LABELS.get(action, action)
            parts = []
            for k, v in details.items():
                if isinstance(v, list):
                    parts.append(f"{k}: {', '.join(str(x) for x in v)}")
                else:
                    parts.append(f"{k}: {v}")
            detail_str = " · ".join(parts) if parts else ""
            if detail_str:
                export_lines.append(f"- **{ts}** — {label} ({detail_str})")
            else:
                export_lines.append(f"- **{ts}** — {label}")

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "📥 Export Activity Log",
                data="\n".join(export_lines),
                file_name=f"hdj_activity_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
            )
        with col2:
            st.markdown('<div class="danger-btn">', unsafe_allow_html=True)
            if st.button("🗑️ Clear Log"):
                clear_events(AUDIT_PATH)
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("No activity recorded yet. Actions will appear here as you use the tool.")


# ============ TAB 6: History ============
with tab6:
    import pandas as pd

    runs = load_all_runs(RESULTS_DIR)

    if not runs:
        st.info("No evaluation runs found. Run a validation first to see history.")
    else:
        # --- Sub-section 1: Coverage Over Time ---
        st.subheader("Coverage Over Time")

        # Build DataFrame for line chart: rows=runs, cols=query names
        all_query_names: list[str] = []
        for run in runs:
            for qn in run.query_names:
                if qn not in all_query_names:
                    all_query_names.append(qn)

        chart_data: dict[str, list[float | None]] = {qn: [] for qn in all_query_names}
        chart_index: list[str] = []

        for run in runs:
            chart_index.append(format_timestamp(run.timestamp))
            for qn in all_query_names:
                chart_data[qn].append(run.recalls.get(qn))

        df = pd.DataFrame(chart_data, index=chart_index)
        st.line_chart(df)

        # Summary: best coverage per run
        st.markdown("**Run summaries:**")
        for run in runs:
            best_r = max(run.recalls.values()) if run.recalls else 0.0
            color = "green" if best_r >= 0.7 else ("orange" if best_r >= 0.4 else "red")
            st.markdown(
                f"- **{format_timestamp(run.timestamp)}** — "
                f"best coverage: :{color}[{best_r:.1%}] "
                f"({run.best_query})"
            )

        st.divider()

        # --- Sub-section 2: Compare Two Runs ---
        st.subheader("Compare Two Runs")

        ts_labels = {
            run.timestamp: format_timestamp(run.timestamp) for run in runs
        }
        ts_list = [run.timestamp for run in runs]

        col_a, col_b = st.columns(2)
        with col_a:
            baseline_ts = st.selectbox(
                "Baseline",
                ts_list,
                index=0,
                format_func=lambda t: ts_labels[t],
            )
        with col_b:
            compare_ts = st.selectbox(
                "Compare",
                ts_list,
                index=len(ts_list) - 1,
                format_func=lambda t: ts_labels[t],
            )

        if baseline_ts == compare_ts:
            st.warning("Select two different runs to compare.")
        else:
            diff = diff_runs(RESULTS_DIR, baseline_ts, compare_ts)

            if diff.total_gold_changed != 0:
                direction = "more" if diff.total_gold_changed > 0 else "fewer"
                st.info(
                    f"Reference passage count changed: "
                    f"{abs(diff.total_gold_changed)} {direction} passages"
                )

            if diff.queries_added:
                st.markdown(f"**Queries added:** {', '.join(diff.queries_added)}")
            if diff.queries_removed:
                st.markdown(f"**Queries removed:** {', '.join(diff.queries_removed)}")

            st.markdown("**Coverage change per query:**")
            for q, delta in sorted(diff.recall_changes.items()):
                if delta > 0:
                    st.markdown(f"- **{q}**: :green[+{delta:.1%}]")
                elif delta < 0:
                    st.markdown(f"- **{q}**: :red[{delta:.1%}]")
                else:
                    st.markdown(f"- **{q}**: no change")

            # Expandable gained/lost per query
            for q in sorted(diff.recall_changes.keys()):
                g = diff.gained.get(q, [])
                l = diff.lost.get(q, [])
                if g or l:
                    with st.expander(f"Details: {q} (+{len(g)} / -{len(l)})"):
                        if g:
                            st.markdown("**Gained:**")
                            for txt in g:
                                st.markdown(f"- :green[{txt[:120]}]")
                        if l:
                            st.markdown("**Lost:**")
                            for txt in l:
                                st.markdown(f"- :red[{txt[:120]}]")

        st.divider()

        # --- Sub-section 3: Reference Passage Reachability ---
        st.subheader("Reference Passage Reachability")

        trackers = aggregate_query_performance(RESULTS_DIR, runs=runs)

        if trackers:
            never_found = sum(1 for t in trackers if not t.ever_found)
            always_found = sum(1 for t in trackers if t.find_rate == 1.0)

            m1, m2, m3 = st.columns(3)
            m1.metric("Total tracked", len(trackers))
            m2.metric("Never found", never_found)
            m3.metric("Always found", always_found)

            # Bar chart of find rates
            bar_df = pd.DataFrame({
                "Find rate": [t.find_rate for t in trackers],
            }, index=[t.preview[:60] + "..." if len(t.preview) > 60 else t.preview for t in trackers])
            st.bar_chart(bar_df)

            # Expandable details per passage (hardest first)
            st.markdown("**Passage details** (hardest to find first):")
            for t in trackers:
                label = t.preview[:80] + "..." if len(t.preview) > 80 else t.preview
                rate_color = "green" if t.find_rate >= 0.7 else ("orange" if t.find_rate >= 0.3 else "red")
                with st.expander(f":{rate_color}[{t.find_rate:.0%}] {label}"):
                    if t.found_by:
                        st.markdown("**Found by:**")
                        for q, timestamps in sorted(t.found_by.items()):
                            ts_str = ", ".join(format_timestamp(ts) for ts in timestamps)
                            st.markdown(f"- {q}: {ts_str}")
                    if t.missed_by:
                        st.markdown("**Missed by:**")
                        for q, timestamps in sorted(t.missed_by.items()):
                            ts_str = ", ".join(format_timestamp(ts) for ts in timestamps)
                            st.markdown(f"- {q}: {ts_str}")
        else:
            st.info("No passage data to track. Run some evaluations first.")


# Footer
st.divider()
st.caption("Corpus Search Tool · Documents → Prepare → Search → Validate → History")
