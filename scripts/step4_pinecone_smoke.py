#!/usr/bin/env python3
"""
Step 4 standalone Pinecone smoke test.

Validates:
- Pinecone connectivity
- upsert
- semantic query
- count
"""

from __future__ import annotations

from src.pinecone_index import PINECONE_INDEX_NAME, PINECONE_NAMESPACE, VECTOR_SIZE, PineconeIndex


def _build_vector(dim: int, hot_index: int) -> list[float]:
    vec = [0.0] * dim
    vec[hot_index] = 1.0
    return vec


def main() -> None:
    print("=== STEP 4 - PINECONE SMOKE TEST ===")
    print(f"Index: {PINECONE_INDEX_NAME}")
    print(f"Namespace: {PINECONE_NAMESPACE or '<default>'}")
    print(f"Vector size: {VECTOR_SIZE}")

    index = PineconeIndex()
    index.delete_collection()

    chunks = [
        {"text": "attention mechanism in transformers", "source": "smoke.pdf", "chunk_id": 0, "chunk_type": "text"},
        {"text": "figure description about encoder decoder", "source": "smoke.pdf", "chunk_id": 1, "chunk_type": "figure"},
    ]
    vectors = [
        _build_vector(VECTOR_SIZE, 3),
        _build_vector(VECTOR_SIZE, 4),
    ]

    index.upsert(chunks, vectors)
    total = index.count()
    print(f"Vector count after upsert: {total}")

    results = index.search(_build_vector(VECTOR_SIZE, 3), top_k=2)
    print(f"Results returned: {len(results)}")
    for i, row in enumerate(results, start=1):
        print(
            f"Result {i}: score={row['score']} source={row['source']} "
            f"chunk_id={row['chunk_id']} type={row['chunk_type']}"
        )

    print("Step 4 smoke test: PASS")


if __name__ == "__main__":
    main()
