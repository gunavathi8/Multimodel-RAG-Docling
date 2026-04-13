import os
from typing import Any

# logger import before openai to ensure env is set
from src.logger import get_logger

import tiktoken
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logger = get_logger(__name__)

CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", 1024))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))
EMBED_MODEL   = os.getenv("EMBED_MODEL", "google/gemini-embedding-001")

# tiktoken cl100k_base for fast token counting — no local model needed
_TOKENIZER = tiktoken.get_encoding("cl100k_base")

_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)


def _validate_chunking_config() -> None:
    if CHUNK_SIZE <= 0:
        raise ValueError(f"CHUNK_SIZE must be > 0, got {CHUNK_SIZE}")
    if CHUNK_OVERLAP < 0:
        raise ValueError(f"CHUNK_OVERLAP must be >= 0, got {CHUNK_OVERLAP}")
    if CHUNK_OVERLAP >= CHUNK_SIZE:
        raise ValueError(
            f"CHUNK_OVERLAP ({CHUNK_OVERLAP}) must be smaller than CHUNK_SIZE ({CHUNK_SIZE})"
        )


# ---------------------------------------------------------------------------
# Chunking  (unchanged — pure token math, no model needed)
# ---------------------------------------------------------------------------

def chunk_text(text: str, source: str) -> list[dict[str, Any]]:
    """
    Split *text* into overlapping token-based chunks.
    Returns a list of dicts: text, source, chunk_id, chunk_type.
    """
    _validate_chunking_config()

    if not text or not text.strip():
        logger.warning("Empty text received for '%s' — no chunks created", source)
        return []

    logger.info("Tokenising text from '%s' (CHUNK_SIZE=%d, OVERLAP=%d)", source, CHUNK_SIZE, CHUNK_OVERLAP)
    tokens = _TOKENIZER.encode(text)
    logger.info("Total tokens: %d", len(tokens))

    chunks: list[dict[str, Any]] = []
    start = 0
    chunk_id = 0

    while start < len(tokens):
        end = min(start + CHUNK_SIZE, len(tokens))
        chunk_text_str = _TOKENIZER.decode(tokens[start:end])
        chunks.append({
            "text": chunk_text_str,
            "source": source,
            "chunk_id": chunk_id,
            "chunk_type": "text",
        })
        chunk_id += 1
        start += CHUNK_SIZE - CHUNK_OVERLAP

    logger.info("Chunking complete — %d text chunks created from '%s'", len(chunks), source)
    return chunks


def make_figure_chunks(
    figures: list[dict],
    source: str,
    start_id: int = 0,
) -> list[dict[str, Any]]:
    """
    Create ONE dedicated chunk per figure description.

    Why dedicated chunks?
    Regular text chunks embed an average of ~1024 tokens of mixed content,
    which dilutes the figure signal.  A standalone figure chunk embeds ONLY
    the description → pure signal → much higher retrieval score for visual queries.

    The figure text is prefixed with 'Figure N description:' so the embedding
    model understands it is a visual element description.
    """
    chunks = []
    for i, fig in enumerate(figures):
        chunks.append({
            "text": f"Figure {fig['num']} description: {fig['description']}",
            "source": source,
            "chunk_id": start_id + i,
            "chunk_type": "figure",
        })
    logger.info("Created %d dedicated figure chunks from '%s'", len(chunks), source)
    return chunks


# ---------------------------------------------------------------------------
# Embeddings via OpenRouter API  (no local model, no GPU needed)
# ---------------------------------------------------------------------------

class EmbedData:
    """Calls the OpenRouter embeddings endpoint — no weights downloaded locally."""

    # OpenRouter recommends batches of ≤ 96 texts per request
    _BATCH_SIZE = 96

    def __init__(self) -> None:
        logger.info("Embedding provider: OpenRouter | Model: %s", EMBED_MODEL)

    def _call_api(self, texts: list[str]) -> list[list[float]]:
        response = _client.embeddings.create(
            model=EMBED_MODEL,
            input=texts,
        )
        # sort by index to preserve order (API may return out-of-order)
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in sorted_data]

    def embed_chunks(self, chunks: list[dict[str, Any]]) -> list[list[float]]:
        """Embed all chunks in batches — returns list of vectors."""
        if not chunks:
            logger.warning("embed_chunks called with 0 chunks — returning empty embedding list")
            return []

        texts = [c["text"] for c in chunks]
        logger.info("Embedding %d chunks via OpenRouter (batch_size=%d)...", len(texts), self._BATCH_SIZE)

        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), self._BATCH_SIZE):
            batch = texts[i : i + self._BATCH_SIZE]
            logger.info("  Batch %d/%d (%d texts)", i // self._BATCH_SIZE + 1,
                        -(-len(texts) // self._BATCH_SIZE), len(batch))
            all_embeddings.extend(self._call_api(batch))

        if not all_embeddings:
            logger.warning("Embedding API returned 0 vectors")
            return []

        logger.info("Embeddings done — %d vectors, dim=%d", len(all_embeddings), len(all_embeddings[0]))
        return all_embeddings

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        logger.info("Embedding query: '%s'", query[:80])
        vector = self._call_api([query])[0]
        logger.info("Query embedding done — vector dim: %d", len(vector))
        return vector
