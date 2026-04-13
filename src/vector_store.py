from __future__ import annotations

from typing import Any, Protocol


class VectorStore(Protocol):
    """
    Minimal contract for vector DB backends used by the app.

    `search` must return normalized chunks with `text/source/chunk_id/chunk_type/score`.
    """

    def upsert(self, chunks: list[dict[str, Any]], embeddings: list[list[float]]) -> None:
        ...

    def search(self, query_vector: list[float], top_k: int = 5) -> list[dict[str, Any]]:
        ...

    def count(self) -> int:
        ...

    def delete_collection(self) -> None:
        ...
