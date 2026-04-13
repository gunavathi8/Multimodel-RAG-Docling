#!/usr/bin/env python3
"""
Step 1 baseline diagnostics for the multimodal Docling RAG pipeline.

What this checks:
- PDF parsing output size
- extracted figure count
- chunking count and token distribution
- first chunk previews
"""

from __future__ import annotations

import argparse
from pathlib import Path

import tiktoken

from src.chunk_embed import chunk_text
from src.utils import convert_pdf


def _preview(text: str, limit: int = 220) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[:limit].rstrip() + "..."


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Step 1 baseline diagnostics.")
    parser.add_argument(
        "--pdf",
        default="data/document.pdf",
        help="Path to input PDF (default: data/document.pdf)",
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf).resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    print(f"[Step 1] Running baseline diagnostics for: {pdf_path}")
    parse_result = convert_pdf(str(pdf_path))

    markdown = parse_result["markdown"]
    figures = parse_result["figures"]
    chunks = chunk_text(markdown, source=pdf_path.name)

    tokenizer = tiktoken.get_encoding("cl100k_base")
    token_counts = [len(tokenizer.encode(c["text"])) for c in chunks]

    print("\n=== Parse Metrics ===")
    print(f"Markdown chars: {len(markdown)}")
    print(f"Figures extracted: {len(figures)}")

    print("\n=== Chunk Metrics ===")
    print(f"Total text chunks: {len(chunks)}")
    if token_counts:
        avg_tokens = sum(token_counts) / len(token_counts)
        print(f"Chunk tokens (min/avg/max): {min(token_counts)}/{avg_tokens:.1f}/{max(token_counts)}")
    else:
        print("Chunk tokens: no chunks created")

    print("\n=== Figure Samples ===")
    if figures:
        for fig in figures[:2]:
            print(f"Figure {fig['num']}: {_preview(fig['description'])}")
    else:
        print("No figures detected")

    print("\n=== Chunk Previews ===")
    if chunks:
        for i, chunk in enumerate(chunks[:2], start=1):
            print(f"Chunk {i} (chunk_id={chunk['chunk_id']}): {_preview(chunk['text'])}")
    else:
        print("No text chunks created")


if __name__ == "__main__":
    main()
