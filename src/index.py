import os
import uuid
from typing import Any

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from src.logger import get_logger
from src.vector_store import VectorStore

load_dotenv()

logger = get_logger(__name__)

QDRANT_PATH     = os.getenv("QDRANT_PATH", "./qdrant_storage")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "multimodal_rag")
VECTOR_SIZE     = int(os.getenv("EMBED_VECTOR_SIZE", 3072))  # gemini-embedding-001 default


class QdrantIndex(VectorStore):
    """Local Qdrant vector store (no Docker required)."""

    def __init__(self) -> None:
        logger.info("Initialising Qdrant local client at path: %s", QDRANT_PATH)
        self.client = QdrantClient(path=QDRANT_PATH)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        existing = {c.name for c in self.client.get_collections().collections}
        if COLLECTION_NAME not in existing:
            logger.info("Collection '%s' not found — creating (dim=%d, distance=COSINE)", COLLECTION_NAME, VECTOR_SIZE)
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=VECTOR_SIZE,
                    distance=Distance.COSINE,
                ),
            )
            logger.info("Collection '%s' created", COLLECTION_NAME)
        else:
            logger.info("Collection '%s' already exists", COLLECTION_NAME)

    def upsert(
        self,
        chunks: list[dict[str, Any]],
        embeddings: list[list[float]],
    ) -> None:
        """Store chunks with their embeddings."""
        logger.info("Upserting %d vectors into collection '%s'", len(chunks), COLLECTION_NAME)
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=emb,
                payload={
                    "text": chunk["text"],
                    "source": chunk["source"],
                    "chunk_id": chunk["chunk_id"],
                    "chunk_type": chunk.get("chunk_type", "text"),
                },
            )
            for chunk, emb in zip(chunks, embeddings)
        ]
        self.client.upsert(collection_name=COLLECTION_NAME, points=points)
        logger.info("Upsert complete — %d points stored", len(points))

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Return normalized top-k chunks from Qdrant."""
        logger.info("Searching collection '%s' for top-%d results", COLLECTION_NAME, top_k)
        # qdrant-client >= 1.10 replaced .search() with .query_points()
        response = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )
        raw_results = response.points
        logger.info("Search complete — %d results returned", len(raw_results))

        return [
            {
                "text": r.payload["text"],
                "source": r.payload["source"],
                "chunk_id": r.payload["chunk_id"],
                "chunk_type": r.payload.get("chunk_type", "text"),
                "score": round(r.score, 4),
            }
            for r in raw_results
        ]

    def count(self) -> int:
        """Return total number of indexed vectors."""
        total = self.client.count(collection_name=COLLECTION_NAME).count
        logger.info("Collection '%s' total vector count: %d", COLLECTION_NAME, total)
        return total

    def delete_collection(self) -> None:
        """Wipe the collection (used on reset)."""
        logger.info("Deleting collection '%s'", COLLECTION_NAME)
        self.client.delete_collection(collection_name=COLLECTION_NAME)
        logger.info("Collection deleted — recreating empty collection")
        self._ensure_collection()
