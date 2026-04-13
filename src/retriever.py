import os
from typing import Any

from dotenv import load_dotenv

from src.chunk_embed import EmbedData
from src.logger import get_logger
from src.vector_store import VectorStore

load_dotenv()

logger = get_logger(__name__)
TOP_K = int(os.getenv("TOP_K", 5))


class Retriever:
    def __init__(self, index: VectorStore, embed_data: EmbedData) -> None:
        self.index = index
        self.embed_data = embed_data
        logger.info("Retriever initialised (TOP_K=%d)", TOP_K)

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[dict[str, Any]]:
        """
        Embed the query, search vector DB, and return the top-k chunks
        with their text, source filename, and similarity score.
        """
        logger.info("Retrieving top-%d chunks for query: '%s'", top_k, query[:80])

        query_vector = self.embed_data.embed_query(query)
        chunks = self.index.search(query_vector, top_k=top_k)

        for i, c in enumerate(chunks):
            logger.info(
                "  Chunk %d | score=%.4f | type=%-6s | source=%s | chunk_id=%d",
                i + 1, c["score"], c["chunk_type"], c["source"], c["chunk_id"],
            )

        return chunks
