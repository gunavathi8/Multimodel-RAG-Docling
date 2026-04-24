import os
import uuid
from typing import Any

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

from src.logger import get_logger
from src.vector_store import VectorStore

load_dotenv()

logger = get_logger(__name__)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "multimodal-rag")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
VECTOR_SIZE = int(os.getenv("EMBED_VECTOR_SIZE", 3072))


class PineconeIndex(VectorStore):
    """Pinecone Cloud vector store."""

    def __init__(
        self,
        index_name: str | None = None,
        namespace: str | None = None,
        vector_size: int | None = None,
    ) -> None:
        if not PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY is required for PineconeIndex")

        self.index_name = index_name or PINECONE_INDEX_NAME
        self.namespace = PINECONE_NAMESPACE if namespace is None else namespace
        self.vector_size = vector_size or VECTOR_SIZE

        logger.info(
            "Initialising Pinecone client | index=%s namespace=%s cloud=%s region=%s dim=%d",
            self.index_name,
            self.namespace or "<default>",
            PINECONE_CLOUD,
            PINECONE_REGION,
            self.vector_size,
        )
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self._ensure_index()
        self.index = self.pc.Index(self.index_name)

    def _ensure_index(self) -> None:
        existing = self._list_index_names()
        if self.index_name in existing:
            logger.info("Pinecone index '%s' already exists", self.index_name)
            return

        logger.info("Creating Pinecone index '%s' (dim=%d, metric=cosine)", self.index_name, self.vector_size)
        self.pc.create_index(
            name=self.index_name,
            dimension=self.vector_size,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=PINECONE_CLOUD,
                region=PINECONE_REGION,
            ),
        )
        logger.info("Pinecone index '%s' created", self.index_name)

    def _list_index_names(self) -> set[str]:
        listing = self.pc.list_indexes()
        if isinstance(listing, list):
            return {
                item["name"] if isinstance(item, dict) else str(getattr(item, "name", ""))
                for item in listing
                if (item.get("name") if isinstance(item, dict) else getattr(item, "name", None))
            }
        names = getattr(listing, "names", None)
        if callable(names):
            return set(names())
        indexes = getattr(listing, "indexes", None)
        if indexes:
            return {
                item["name"] if isinstance(item, dict) else str(getattr(item, "name", ""))
                for item in indexes
                if (item.get("name") if isinstance(item, dict) else getattr(item, "name", None))
            }
        return set()

    def upsert(self, chunks: list[dict[str, Any]], embeddings: list[list[float]]) -> None:
        logger.info(
            "Upserting into Pinecone | chunks=%d embeddings=%d namespace=%s",
            len(chunks),
            len(embeddings),
            self.namespace or "<default>",
        )
        vectors = [
            {
                "id": str(uuid.uuid4()),
                "values": emb,
                "metadata": {
                    "text": chunk["text"],
                    "source": chunk["source"],
                    "chunk_id": chunk["chunk_id"],
                    "chunk_type": chunk.get("chunk_type", "text"),
                },
            }
            for chunk, emb in zip(chunks, embeddings)
        ]
        self.index.upsert(vectors=vectors, namespace=self.namespace)
        logger.info("Pinecone upsert complete | vectors=%d", len(vectors))

    def search(self, query_vector: list[float], top_k: int = 5) -> list[dict[str, Any]]:
        logger.info(
            "Searching Pinecone | top_k=%d namespace=%s",
            top_k,
            self.namespace or "<default>",
        )
        response = self.index.query(
            vector=query_vector,
            top_k=top_k,
            namespace=self.namespace,
            include_metadata=True,
        )
        matches = response.get("matches", [])
        logger.info("Pinecone search complete | matches=%d", len(matches))

        rows: list[dict[str, Any]] = []
        for match in matches:
            metadata = match.get("metadata", {})
            rows.append(
                {
                    "text": metadata.get("text", ""),
                    "source": metadata.get("source", "unknown"),
                    "chunk_id": metadata.get("chunk_id", -1),
                    "chunk_type": metadata.get("chunk_type", "text"),
                    "score": round(float(match.get("score", 0.0)), 4),
                }
            )
        return rows

    def count(self) -> int:
        logger.info("Fetching Pinecone index stats for count")
        stats = self.index.describe_index_stats()
        namespaces = stats.get("namespaces", {})
        if self.namespace:
            total = int(namespaces.get(self.namespace, {}).get("vector_count", 0))
        else:
            total = int(stats.get("total_vector_count", 0))
        logger.info("Pinecone vector count=%d namespace=%s", total, self.namespace or "<default>")
        return total

    def delete_collection(self) -> None:
        logger.info("Deleting all vectors from Pinecone namespace=%s", self.namespace or "<default>")
        self.index.delete(delete_all=True, namespace=self.namespace)
        logger.info("Pinecone namespace cleared")
