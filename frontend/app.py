"""
Health Data Justice - RAG Evaluation Interface
"""

import asyncio
import json
import shutil
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse, unquote

import streamlit as st

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hdj import HDJRag, Evaluator
from src.hdj.evaluate import save_results

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

# Page config
st.set_page_config(
    page_title="HDJ RAG",
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
    
    /* Hide branding */
    #MainMenu, footer { visibility: hidden; }
    
    hr { margin: 1.5rem 0; border: none; border-top: 1px solid #e5e7eb; }
</style>
""", unsafe_allow_html=True)


# ============ Helper Functions ============

def init_state():
    defaults = {
        "indexed_pdfs": [],
        "index_timestamp": None,
        "eval_results": None,
        "search_results": None,
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
    queries: dict[str, str], limit: int = 20, overlap_threshold: float = 0.3
):
    """Run evaluation against gold standard with configurable overlap."""
    evaluator = Evaluator.from_json(GOLD_STANDARD_PATH, overlap_threshold=overlap_threshold)
    
    async with HDJRag(DB_PATH) as rag:
        results = []
        for name, query in queries.items():
            result = await evaluator.run_query(rag, name, query, limit)
            results.append(result)
        
        save_results(results, RESULTS_DIR)
        return results


async def search_documents(query: str, limit: int = 10):
    async with HDJRag(DB_PATH) as rag:
        return await rag.search(query, limit=limit)


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
    st.title("📚 Health Data Justice")
    st.caption("RAG evaluation for policy documents")
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="danger-btn">', unsafe_allow_html=True)
    if st.button("🚨 Reset All", help="Reset PDFs, index, gold standard, and queries to defaults"):
        reset_to_defaults()
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

st.divider()

# ===== TABS =====
tab1, tab2, tab3, tab4 = st.tabs([
    "1️⃣ Documents", 
    "2️⃣ Build Index", 
    "3️⃣ Evaluate", 
    "4️⃣ Search"
])


# ============ TAB 1: Documents ============
with tab1:
    st.header("Step 1: Add PDF Documents")
    
    show_info("Upload the PDF documents you want to analyze. After uploading, go to 'Build Index' to make them searchable.")
    
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
                    pdf.unlink()
                    # Warn if it was indexed
                    if pdf.name in indexed_pdfs:
                        st.warning("This PDF was indexed. Rebuild the index to update.")
                    st.rerun()
    else:
        st.info("No PDFs yet. Upload some above, or click 'Reset All' to restore defaults.")


# ============ TAB 2: Build Index ============
with tab2:
    st.header("Step 2: Build Search Index")
    
    show_def("What this does", "Converts your PDFs into searchable chunks. You must rebuild when PDFs change.")
    
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
        show_warn("No index built yet. Click 'Build Index' below.")
    
    # Build buttons
    st.subheader("Build")
    st.markdown(f"**PDFs available:** {len(pdfs)}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔨 Build Index", disabled=len(pdfs) == 0, help="Build (skips existing)"):
            with st.spinner("Building index... This may take a few minutes."):
                count, names = asyncio.run(build_index(force=False))
                meta = save_index_meta(names)
                st.session_state.indexed_pdfs = names
                st.session_state.index_timestamp = meta["timestamp"]
            st.success(f"Done! {count} chunks from {len(names)} PDFs.")
            st.rerun()
    
    with col2:
        if st.button("🔄 Rebuild Index", disabled=len(pdfs) == 0, help="Delete and rebuild from scratch"):
            with st.spinner("Rebuilding index..."):
                count, names = asyncio.run(build_index(force=True))
                meta = save_index_meta(names)
                st.session_state.indexed_pdfs = names
                st.session_state.index_timestamp = meta["timestamp"]
            st.success(f"Done! {count} chunks from {len(names)} PDFs.")
            st.rerun()


# ============ TAB 3: Evaluate ============
with tab3:
    st.header("Step 3: Evaluate Queries")
    
    show_info("Test how well different queries find relevant sections. Define your gold standard (relevant sections), then run queries against it.")
    
    # Check if index exists
    index_meta = load_index_meta()
    if not index_meta.get("pdfs"):
        show_warn("Build an index first (Step 2) before evaluating.")
        st.stop()
    
    indexed_pdfs = index_meta.get("pdfs", [])
    
    # ===== GOLD STANDARD SECTION =====
    st.subheader("Gold Standard")
    
    col1, col2 = st.columns(2)
    with col1:
        with st.expander("📖 What is the gold standard?", expanded=False):
            st.markdown("""
            **Gold standard** = sections in your PDFs that you've marked as relevant.
            
            - **Recall** = What % of gold standard did we find? (Higher = better coverage)
            - **Precision** = What % of results were relevant? (Higher = less noise)
            """)
    with col2:
        with st.expander("🔧 How does the search work?", expanded=False):
            st.markdown("""
            **Embedding Model:** `Qwen3-Embedding-4B` (via Ollama)
            - Converts text into 2560-dimensional vectors
            - Multilingual, optimized for semantic similarity
            
            **Chunking:** 512 tokens per chunk
            - PDFs are split into overlapping chunks
            - Each chunk is embedded separately
            
            **Search:** Vector similarity (cosine)
            - Your query is embedded the same way
            - Finds chunks with most similar vectors
            - Score = how similar (0-100%)
            
            **Database:** LanceDB (local vector store)
            """)
    
    gold = load_gold_standard()
    by_file = get_gold_standard_by_file(gold)
    
    # Show gold standard sections (text only, no notes)
    if gold:
        st.markdown(f"**{len(gold)} sections** defined")
        
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
                            st.success("Saved!")
                            st.rerun()
                    with col2:
                        if st.button("🗑️ Delete", key=f"gs_del_{idx}"):
                            del gold[idx]
                            save_gold_standard(gold)
                            st.rerun()
                    
                    st.divider()
    else:
        st.info("No gold standard sections defined yet. Add some below.")
    
    # Add new gold standard section - only for indexed PDFs
    st.markdown("**Add a section:**")
    
    if indexed_pdfs:
        new_file = st.selectbox("PDF (must be in index)", indexed_pdfs, key="gs_new_file")
        new_text = st.text_area(
            "Paste relevant text from this PDF",
            placeholder="Copy and paste a section that's relevant to data justice...",
            key="gs_new_text"
        )
        
        if st.button("➕ Add to Gold Standard"):
            if new_text.strip():
                gold.append(
                    {
                        "source_file": new_file,
                        "text": new_text.strip(),
                    }
                )
                save_gold_standard(gold)
                st.success("Added!")
                st.rerun()
            else:
                st.error("Please enter some text.")
    else:
        st.info("Index some PDFs first to add gold standard sections.")
    
    st.divider()
    
    # ===== QUERIES SECTION =====
    st.subheader("Queries")
    
    queries = load_queries()
    
    if queries:
        st.markdown(f"**{len(queries)} queries** defined")
        
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
                        st.success("Saved!")
                        st.rerun()
                with col2:
                    if st.button("🗑️ Delete", key=f"q_del_{name}"):
                        del queries[name]
                        save_queries(queries)
                        st.rerun()
    else:
        st.info("No queries defined. Add one below.")
    
    # Add new query
    st.markdown("**Add a query:**")
    col1, col2 = st.columns([1, 3])
    with col1:
        new_q_name = st.text_input("Name", placeholder="my_query", key="new_q_name")
    with col2:
        new_q_text = st.text_area("Query text", placeholder="What to search for...", height=80, key="new_q_text")
    
    if st.button("➕ Add Query") and new_q_name and new_q_text:
        queries[new_q_name] = new_q_text
        save_queries(queries)
        st.success(f"Added '{new_q_name}'")
        st.rerun()
    
    st.divider()
    
    # ===== RUN EVALUATION =====
    st.subheader("Run Evaluation")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Queries", len(queries))
    with col2:
        st.metric("Gold Standard", len(gold))
    with col3:
        limit = st.number_input(
            "Results per query",
            min_value=5,
            max_value=100,
            value=20,
            help="How many chunks to retrieve for each query.",
        )
    with col4:
        overlap_threshold = st.slider(
            "Gold match strictness",
            min_value=0.1,
            max_value=0.9,
            value=0.3,
            step=0.05,
            help=(
                "How much of a gold passage must overlap with a retrieved chunk "
                "to count as 'found'.\n\n"
                "0.3 = lenient (≥30% of words overlap)\n"
                "0.5 = stricter (≥50%)\n"
                "0.7+ = very strict"
            ),
        )
    
    can_run = len(queries) > 0 and len(gold) > 0
    
    if not can_run:
        if len(queries) == 0:
            st.warning("Add at least one query above.")
        if len(gold) == 0:
            st.warning("Add at least one gold standard section above.")
    
    if st.button("▶️ Run Evaluation", disabled=not can_run):
        # Clear old results first
        st.session_state.eval_results = None
        with st.spinner("Running evaluation..."):
            results = asyncio.run(
                run_evaluation(queries, limit, overlap_threshold=overlap_threshold)
            )
            st.session_state.eval_results = results
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
                st.markdown(f"Recall: **{r.recall:.0%}**")
            with cols[2]:
                st.markdown(f"Precision: {r.precision:.0%}")
            with cols[3]:
                st.markdown(f"Found {r.found}/{r.total_gold}")
        
        st.divider()
        best = sorted_results[0]
        show_success(f"Best: '{best.name}' with {best.recall:.0%} recall")
        
        # Details
        for r in sorted_results:
            with st.expander(f"📊 {r.name} — {r.recall:.0%} recall"):
                st.markdown("**Query:**")
                st.text_area("", value=r.query, height=80, disabled=True, key=f"res_q_{r.name}", label_visibility="collapsed")
                
                if r.missed_texts:
                    st.markdown(f"**Missed sections ({len(r.missed_texts)}):**")
                    for i, text in enumerate(r.missed_texts):
                        with st.expander(f"Missed {i+1} ({len(text)} chars)", expanded=True):
                            # Calculate height based on text length (roughly 80 chars per line, 20px per line)
                            lines = max(5, min(20, len(text) // 80 + 3))
                            st.text_area("", value=text, height=lines * 25, disabled=True, key=f"missed_{r.name}_{i}", label_visibility="collapsed")


# ============ TAB 4: Search ============
with tab4:
    st.header("Step 4: Search Documents")
    
    show_info("Search across all indexed documents with any query.")
    
    # Check if index exists
    index_meta = load_index_meta()
    if not index_meta.get("pdfs"):
        show_warn("Build an index first (Step 2) before searching.")
        st.stop()
    
    query = st.text_area(
        "What are you looking for?",
        placeholder="Describe what you want to find...",
        height=100
    )
    
    col1, col2 = st.columns([3, 1])
    with col2:
        limit = st.number_input("Max results", min_value=5, max_value=2000, value=10, key="search_limit", help="Up to 2000 results. Your M1 Max can handle this easily. More results = more comprehensive but slower to scroll through.")
    
    if st.button("🔍 Search", disabled=not query):
        with st.spinner("Searching..."):
            results = asyncio.run(search_documents(query, limit))
            st.session_state.search_results = results
        st.rerun()
    
    if st.session_state.search_results:
        st.divider()
        results = st.session_state.search_results
        st.markdown(f"**Found {len(results)} sections:**")
        
        for i, r in enumerate(results):
            pages = r.get("page_numbers", [])
            content = r.get("content", "")
            doc = _pretty_doc_name(r.get("document_uri"))

            label = f"Result {i+1}"
            label += f" · {doc}"
            if pages:
                label += f" · Pages {', '.join(map(str, pages))}"

            with st.expander(label, expanded=i < 3):
                st.text_area(
                    "",
                    value=content,
                    height=180,
                    disabled=True,
                    key=f"search_{i}",
                    label_visibility="collapsed",
                )


# Footer
st.divider()
st.caption("Health Data Justice RAG Tool · Step through the tabs in order: Documents → Build Index → Evaluate → Search")
