#!/usr/bin/env python3
"""
Standalone step-by-step test runner for the multimodal Docling RAG pipeline.

Steps covered:
1) Docling multimodal extraction (text/tables/figures)
2) PDF -> markdown conversion
3) text chunking
4) figure chunk creation
5) optional embedding smoke test
"""

from __future__ import annotations

import argparse
from pathlib import Path

import tiktoken

from src.chunk_embed import EmbedData, chunk_text, make_figure_chunks
from src.utils import convert_pdf, extract_multimodal_debug


def _preview(text: str, limit: int = 320) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[:limit].rstrip() + "..."


def _print_header(title: str) -> None:
    print(f"\n{'=' * 18} {title} {'=' * 18}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run standalone step tests.")
    parser.add_argument("--pdf", default="data/document.pdf", help="Input PDF path")
    parser.add_argument(
        "--with-embedding",
        action="store_true",
        help="Also run embedding smoke test",
    )
    parser.add_argument(
        "--skip-image-summary",
        action="store_true",
        help="Skip LLM image summarization in Step 1 debug",
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf).resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    print(f"Running step tests for: {pdf_path}")

    # Step 1: Docling multimodal debug
    _print_header("STEP 1 - DOCLING MULTIMODAL DEBUG")
    debug_report = extract_multimodal_debug(
        str(pdf_path),
        summarize_images=not args.skip_image_summary,
    )
    stats = debug_report["stats"]
    print(
        "Counts:",
        f"text_blocks={stats['text_blocks']},",
        f"section_headers={stats['section_headers']},",
        f"tables={stats['tables']},",
        f"figures={stats['figures']},",
        f"other_textual_items={stats['other_textual_items']}",
    )

    if debug_report["table_samples"]:
        print("\nTable samples:")
        for i, table_md in enumerate(debug_report["table_samples"], start=1):
            print(f"\n[Table {i}]")
            print(_preview(table_md, limit=700))
    else:
        print("\nTable samples: none")

    if debug_report["figure_samples"]:
        print("\nFigure samples:")
        for fig in debug_report["figure_samples"]:
            print(
                f"- Figure {fig['num']} ({fig['width']}x{fig['height']}): "
                f"{_preview(fig['description'])}"
            )
    else:
        print("\nFigure samples: none")

    # Step 2: convert_pdf output
    _print_header("STEP 2 - CONVERT PDF")
    parse_result = convert_pdf(str(pdf_path))
    markdown = parse_result["markdown"]
    figures = parse_result["figures"]
    print(f"Markdown chars: {len(markdown)}")
    print(f"Figures returned: {len(figures)}")
    print(f"Markdown preview: {_preview(markdown)}")

    # Step 3: text chunking
    _print_header("STEP 3 - CHUNK TEXT")
    text_chunks = chunk_text(markdown, source=pdf_path.name)
    print(f"Text chunks: {len(text_chunks)}")
    tokenizer = tiktoken.get_encoding("cl100k_base")
    token_counts = [len(tokenizer.encode(c["text"])) for c in text_chunks]
    if token_counts:
        avg_tokens = sum(token_counts) / len(token_counts)
        print(f"Chunk tokens min/avg/max: {min(token_counts)}/{avg_tokens:.1f}/{max(token_counts)}")
        print(f"First text chunk preview: {_preview(text_chunks[0]['text'])}")
    else:
        print("No text chunks produced")

    # Step 4: figure chunk creation
    _print_header("STEP 4 - FIGURE CHUNKS")
    figure_chunks = make_figure_chunks(figures, source=pdf_path.name, start_id=len(text_chunks))
    print(f"Figure chunks: {len(figure_chunks)}")
    if figure_chunks:
        print(f"First figure chunk preview: {_preview(figure_chunks[0]['text'])}")
    else:
        print("No figure chunks produced")

    # Step 5: embedding smoke test (optional)
    _print_header("STEP 5 - EMBEDDING SMOKE TEST")
    if args.with_embedding:
        sample = (text_chunks[:2] + figure_chunks[:1]) if text_chunks else figure_chunks[:1]
        if not sample:
            print("Skipped: no sample chunks available")
            return
        embedder = EmbedData()
        vectors = embedder.embed_chunks(sample)
        if vectors:
            print(f"Embedding vectors: {len(vectors)} (dim={len(vectors[0])})")
        else:
            print("Embedding returned no vectors")
    else:
        print("Skipped (pass --with-embedding to run)")


if __name__ == "__main__":
    main()
