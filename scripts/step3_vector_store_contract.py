#!/usr/bin/env python3
"""
Step 3 standalone test: vector store contract behavior using current backend.

Creates an isolated temporary collection, upserts test vectors, runs search,
and verifies normalized result shape.
"""

from __future__ import annotations

import os
import uuid
from typing import Any


def _build_vector(dim: int, hot_index: int) -> list[float]:
    v = [0.0] * dim
    if 0 <= hot_index < dim:
        v[hot_index] = 1.0
    return v


def _assert_result_shape(row: dict[str, Any]) -> None:
    required = {"text", "source", "chunk_id", "chunk_type", "score"}
    missing = required.difference(row.keys())
    if missing:
        raise AssertionError(f"Missing keys in search result: {sorted(missing)}")


def main() -> None:
    test_collection = f"contract_test_{uuid.uuid4().hex[:8]}"
    os.environ["COLLECTION_NAME"] = test_collection

    # Import after setting env so src.index uses the isolated collection.
    from src.index import QdrantIndex, VECTOR_SIZE  # pylint: disable=import-outside-toplevel

    print("=== STEP 3 - VECTOR STORE CONTRACT TEST ===")
    print(f"Using temporary collection: {test_collection}")
    print(f"Vector size: {VECTOR_SIZE}")

    index = QdrantIndex()
    try:
        chunks = [
            {
                "text": "Transformer uses self-attention for sequence modeling.",
                "source": "contract-test.pdf",
                "chunk_id": 0,
                "chunk_type": "text",
            },
            {
                "text": "Figure 1 description: encoder-decoder architecture diagram.",
                "source": "contract-test.pdf",
                "chunk_id": 1,
                "chunk_type": "figure",
            },
        ]
        embeddings = [
            _build_vector(VECTOR_SIZE, 1),
            _build_vector(VECTOR_SIZE, 2),
        ]

        index.upsert(chunks, embeddings)
        total = index.count()
        print(f"Upserted rows count: {total}")

        results = index.search(_build_vector(VECTOR_SIZE, 1), top_k=2)
        print(f"Search rows returned: {len(results)}")

        for i, row in enumerate(results, start=1):
            _assert_result_shape(row)
            print(
                f"Result {i}: score={row['score']} type={row['chunk_type']} "
                f"chunk_id={row['chunk_id']} source={row['source']}"
            )

        print("Contract test: PASS")
    finally:
        index.delete_collection()
        print("Temporary collection deleted")


if __name__ == "__main__":
    main()
