import os
import tempfile

# ── MUST be the very first import ──────────────────────────────────────────
# Sets TRANSFORMERS_VERBOSITY and warning filters before any lib loads.
from src.logger import get_logger

import streamlit as st

from src.chunk_embed import EmbedData, chunk_text, make_figure_chunks
from src.pinecone_index import PineconeIndex
from src.rag_engine import generate_response
from src.retriever import Retriever
from src.utils import convert_pdf
from src.vector_store import VectorStore

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Validate critical env vars at startup — fail fast with a clear message
# ---------------------------------------------------------------------------
_api_key = os.getenv("OPENROUTER_API_KEY", "")
if not _api_key or _api_key.startswith("your_") or "=" in _api_key:
    st.error(
        "**OpenRouter API key is missing or malformed.**\n\n"
        "Open `.env` and set:\n```\nOPENROUTER_API_KEY=sk-or-v1-...\n```\n"
        "Make sure the value does **not** include the variable name as a prefix."
    )
    st.stop()

_pinecone_key = os.getenv("PINECONE_API_KEY", "")
if not _pinecone_key or _pinecone_key.startswith("your_") or "=" in _pinecone_key:
    st.error(
        "**Pinecone API key is missing or malformed.**\n\n"
        "Open `.env` and set:\n```\nPINECONE_API_KEY=...\n```\n"
        "Make sure the value does **not** include the variable name as a prefix."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Shared singletons via st.cache_resource
# ---------------------------------------------------------------------------
# @st.cache_resource creates exactly ONE instance for the entire app process,
# shared across all sessions and reruns.
# ---------------------------------------------------------------------------

@st.cache_resource
def _get_vector_index() -> VectorStore:
    logger.info("Initialising shared PineconeIndex (cache_resource)")
    return PineconeIndex()


@st.cache_resource
def _get_embed_data() -> EmbedData:
    logger.info("Initialising shared EmbedData (cache_resource)")
    return EmbedData()


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Multimodal RAG",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Multimodal RAG with Docling")
st.caption("Ask questions over complex PDFs — text, tables, images, and formulas.")

# ---------------------------------------------------------------------------
# Session state  (messages + file list only — heavy objects live in cache_resource)
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files: list[str] = []

# ---------------------------------------------------------------------------
# Auto-reconnect — if Pinecone already has data from a previous session/run,
# unlock the chat interface immediately without re-uploading.
# ---------------------------------------------------------------------------
if not st.session_state.indexed_files:
    try:
        _existing_count = _get_vector_index().count()
        if _existing_count > 0:
            logger.info("Existing index detected (%d vectors) — restoring session", _existing_count)
            st.session_state.indexed_files = [f"restored-session ({_existing_count} chunks)"]
    except Exception as e:
        logger.warning("Could not check existing index: %s", e)

# ---------------------------------------------------------------------------
# Sidebar — upload & index
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Document Upload")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file:
        # Only block re-indexing for exact filename matches, not restored-session labels
        already_indexed = uploaded_file.name in st.session_state.indexed_files

        if already_indexed:
            st.info(f"`{uploaded_file.name}` is already indexed.")
        else:
            if st.button("Process & Index", type="primary"):
                logger.info("=== Starting indexing pipeline for: %s ===", uploaded_file.name)

                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                try:
                    with st.status("Processing PDF...", expanded=True) as status:

                        # Step 1 — Parse
                        st.write("Parsing with Docling (tables, formulas, images)...")
                        logger.info("Step 1/5 — Parsing PDF with Docling")
                        parse_result = convert_pdf(tmp_path)
                        markdown_text = parse_result["markdown"]
                        figures       = parse_result["figures"]
                        logger.info(
                            "Step 1/5 — Done. Markdown: %d chars | Figures: %d",
                            len(markdown_text), len(figures),
                        )

                        # Step 2 — Chunk
                        # Text chunks  : sliding window over the full markdown
                        # Figure chunks: ONE dedicated chunk per figure (pure signal)
                        st.write("Chunking text + creating dedicated figure chunks...")
                        logger.info("Step 2/5 — Chunking text and figures")
                        text_chunks   = chunk_text(markdown_text, source=uploaded_file.name)
                        figure_chunks = make_figure_chunks(
                            figures, source=uploaded_file.name, start_id=len(text_chunks)
                        )
                        chunks = text_chunks + figure_chunks
                        st.write(
                            f"Created **{len(text_chunks)}** text chunks "
                            f"+ **{len(figure_chunks)}** figure chunks "
                            f"= **{len(chunks)}** total."
                        )
                        logger.info(
                            "Step 2/5 — Done. %d text + %d figure = %d total chunks",
                            len(text_chunks), len(figure_chunks), len(chunks),
                        )

                        # Step 3 — Embed  (API call — no local model)
                        st.write("Generating embeddings via OpenRouter...")
                        logger.info("Step 3/5 — Generating embeddings for %d chunks", len(chunks))
                        embeddings = _get_embed_data().embed_chunks(chunks)
                        logger.info("Step 3/5 — Done. %d embeddings generated", len(embeddings))

                        # Step 4 — Index
                        st.write("Indexing in Pinecone cloud...")
                        logger.info("Step 4/5 — Upserting into Pinecone")
                        _get_vector_index().upsert(chunks, embeddings)

                        # Step 5 — Finalise
                        st.session_state.indexed_files = [
                            f for f in st.session_state.indexed_files
                            if not f.startswith("restored-session")
                        ]
                        st.session_state.indexed_files.append(uploaded_file.name)
                        status.update(label="Done!", state="complete", expanded=False)

                    total = _get_vector_index().count()
                    logger.info(
                        "=== Indexing complete: %s — %d new chunks, %d total ===",
                        uploaded_file.name, len(chunks), total,
                    )
                    st.success(
                        f"Indexed `{uploaded_file.name}` — "
                        f"{len(chunks)} chunks added ({total} total)."
                    )
                finally:
                    os.unlink(tmp_path)

    # Show indexed files
    if st.session_state.indexed_files:
        st.divider()
        st.subheader("Indexed files")
        for fname in st.session_state.indexed_files:
            if fname.startswith("restored-session"):
                st.markdown(f"- 🔄 `{fname}` *(reconnected from disk)*")
            else:
                st.markdown(f"- ✅ `{fname}`")

        if st.button("Reset (clear index)", type="secondary"):
            logger.info("Reset triggered — clearing Pinecone namespace and cached resources")
            _get_vector_index().delete_collection()
            # Clear cache_resource so fresh instances are created after reset
            _get_vector_index.clear()
            _get_embed_data.clear()
            st.session_state.indexed_files = []
            st.session_state.messages = []
            st.rerun()

# ---------------------------------------------------------------------------
# Main — chat interface
# ---------------------------------------------------------------------------
if not st.session_state.indexed_files:
    st.info("Upload and index a PDF using the sidebar to start chatting.")
    st.stop()

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask a question about the document..."):
    logger.info("=== New query received: '%s' ===", prompt[:100])

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Retrieve context
    logger.info("Fetching context chunks from Pinecone")
    retriever = Retriever(_get_vector_index(), _get_embed_data())
    context_chunks = retriever.retrieve(prompt)
    logger.info("Retrieved %d context chunks", len(context_chunks))

    # Display assistant response
    with st.chat_message("assistant"):
        with st.expander("Retrieved sources", expanded=False):
            for i, chunk in enumerate(context_chunks):
                st.markdown(
                    f"**Chunk {i + 1}** &nbsp; score: `{chunk['score']}` &nbsp; "
                    f"source: `{chunk['source']}`"
                )
                st.text(chunk["text"][:400] + ("..." if len(chunk["text"]) > 400 else ""))

        # Stream response
        logger.info("Streaming LLM response...")
        response_text = ""
        placeholder = st.empty()

        for token in generate_response(prompt, context_chunks):
            response_text += token
            placeholder.markdown(response_text + "▌")

        placeholder.markdown(response_text)
        logger.info("=== Response complete (%d chars) ===", len(response_text))

    st.session_state.messages.append({"role": "assistant", "content": response_text})
