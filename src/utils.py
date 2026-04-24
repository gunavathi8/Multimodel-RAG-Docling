import base64
import io
import os
from typing import Literal

# logger import must come before any heavy imports
from src.logger import get_logger

from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI
from PIL import Image

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

try:
    from docling_core.types.doc import (
        PictureItem,
        TableItem,
        SectionHeaderItem,
        TextItem,
        ListItem,
    )
except ImportError:
    from docling.datamodel.document import (   # type: ignore[no-redef]
        PictureItem,
        TableItem,
        SectionHeaderItem,
        TextItem,
        ListItem,
    )

load_dotenv()

logger = get_logger(__name__)

VISION_MODEL = os.getenv("OPENROUTER_LLM_MODEL", "google/gemma-3-27b-it")
AZURE_OPENAI_CHAT_MODEL = os.getenv("AZURE_OPENAI_CHAT_MODEL", "gpt-4o-mini")
AZURE_OPENAI_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", AZURE_OPENAI_CHAT_MODEL)
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")

_openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)
_azure_client = (
    AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
    )
    if AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY
    else None
)


# ---------------------------------------------------------------------------
# Docling converter
# ---------------------------------------------------------------------------

def _build_converter() -> DocumentConverter:
    logger.info("Building Docling converter (table structure + formula enrichment + picture extraction)")
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_table_structure = True
    pipeline_options.do_formula_enrichment = True
    pipeline_options.generate_picture_images = True

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )


# ---------------------------------------------------------------------------
# Image summarisation via OpenRouter multimodal LLM
# ---------------------------------------------------------------------------

def _summarize_image(
    image: Image.Image,
    figure_num: int,
    provider: Literal["openrouter", "azure"] = "openrouter",
) -> str:
    """Send image to OpenRouter multimodal LLM and return a text description."""
    if provider == "azure":
        if _azure_client is None:
            raise ValueError("Azure selected for image summarization but AZURE_OPENAI_ENDPOINT/API_KEY missing")
        client = _azure_client
        model = AZURE_OPENAI_CHAT_DEPLOYMENT
    else:
        client = _openrouter_client
        model = VISION_MODEL

    logger.info("Summarising Figure %d | provider=%s model=%s", figure_num, provider, model)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    b64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{b64_image}"
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                "Describe this image in detail. "
                                "Focus on any data, trends, labels, axes, charts, "
                                "diagrams, or key visual information present."
                            ),
                        },
                    ],
                }
            ],
            max_tokens=512,
        )
        summary = response.choices[0].message.content.strip()
        logger.info("Figure %d summarised (%d chars)", figure_num, len(summary))
        return summary
    except Exception as e:
        logger.warning("Figure %d summary failed: %s", figure_num, e)
        return f"[Image summary unavailable: {e}]"


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert_pdf(pdf_path: str, provider: Literal["openrouter", "azure"] = "openrouter") -> dict:
    """
    Parse a PDF with Docling and return a dict with two keys:

    - "markdown"  : full document text with inline figure references for context.
                    Used for token-based chunking of the running prose.
    - "figures"   : list of {num, description} dicts — one per extracted image.
                    Each figure is later indexed as its OWN dedicated chunk so
                    its embedding is pure signal and not diluted by surrounding text.

    Element handling:
    - Text / paragraphs  → raw text, reading order preserved
    - Section headers    → markdown headings (#, ## …) based on heading level
    - Tables             → markdown table syntax  (| col | col |)
    - Formulas           → LaTeX-style inline text
    - Images / charts    → summarised by vision LLM;
                           summary injected inline in markdown AND stored
                           separately in "figures" list for dedicated indexing
    """
    logger.info("Starting PDF conversion: %s | provider=%s", pdf_path, provider)
    converter = _build_converter()

    logger.info("Running Docling document conversion...")
    result = converter.convert(pdf_path)
    doc = result.document
    logger.info("Docling conversion complete — iterating document items")

    parts: list[str] = []
    figures: list[dict] = []        # collected separately for dedicated chunks
    picture_count = 0
    table_count = 0
    text_count = 0

    for element, _level in doc.iterate_items():

        # ── Images / Charts / Diagrams ──────────────────────────────────────
        if isinstance(element, PictureItem):
            picture_count += 1
            try:
                pil_image = element.get_image(doc)
                if pil_image is not None:
                    summary = _summarize_image(pil_image, picture_count, provider=provider)
                else:
                    logger.warning("Figure %d: no image data returned", picture_count)
                    summary = "[Image could not be extracted]"
            except Exception as e:
                logger.warning("Figure %d: extraction error — %s", picture_count, e)
                summary = "[Image extraction failed]"

            # 1. Keep inline reference so surrounding text has context
            parts.append(f"**[Figure {picture_count}]:** {summary}")
            # 2. Also store separately — will become its own dedicated chunk
            figures.append({"num": picture_count, "description": summary})

        # ── Tables ───────────────────────────────────────────────────────────
        elif isinstance(element, TableItem):
            table_count += 1
            logger.info("Processing table %d", table_count)
            try:
                table_md = element.export_to_markdown(doc)
                parts.append(table_md)
            except Exception as e:
                logger.warning("Table %d markdown export failed: %s", table_count, e)
                fallback = getattr(element, "text", None)
                if fallback:
                    parts.append(fallback)

        # ── Section headers (h1–h6) ──────────────────────────────────────────
        elif isinstance(element, SectionHeaderItem):
            h_level = int(getattr(element, "level", 2))
            prefix = "#" * max(1, min(h_level, 6))
            text = getattr(element, "text", "").strip()
            if text:
                parts.append(f"{prefix} {text}")

        # ── Text, list items, formulas, captions, footnotes … ────────────────
        else:
            text = getattr(element, "text", None)
            if text and text.strip():
                text_count += 1
                parts.append(text.strip())

    logger.info(
        "Document parsed — %d text blocks, %d tables, %d figures",
        text_count, table_count, picture_count,
    )
    markdown = "\n\n".join(parts)
    logger.info(
        "Markdown assembled — %d chars | %d figures extracted separately",
        len(markdown), len(figures),
    )
    return {"markdown": markdown, "figures": figures}


def extract_multimodal_debug(
    pdf_path: str,
    max_table_samples: int = 3,
    max_figure_samples: int = 3,
    summarize_images: bool = True,
    provider: Literal["openrouter", "azure"] = "openrouter",
) -> dict:
    """
    Debug helper to inspect Docling multimodal extraction quality.

    Returns structured output with:
    - element counts
    - sample table markdown blocks
    - sample figure descriptions (optionally via vision model)
    """
    logger.info("Starting multimodal debug extraction: %s | provider=%s", pdf_path, provider)
    converter = _build_converter()
    result = converter.convert(pdf_path)
    doc = result.document

    stats = {
        "text_blocks": 0,
        "section_headers": 0,
        "tables": 0,
        "figures": 0,
        "other_textual_items": 0,
    }
    table_samples: list[str] = []
    figure_samples: list[dict] = []

    for element, _level in doc.iterate_items():
        if isinstance(element, TableItem):
            stats["tables"] += 1
            if len(table_samples) < max_table_samples:
                try:
                    table_samples.append(element.export_to_markdown(doc))
                except Exception as e:
                    table_samples.append(f"[Table export failed: {e}]")

        elif isinstance(element, PictureItem):
            stats["figures"] += 1
            if len(figure_samples) < max_figure_samples:
                description = "[Image summary skipped]"
                width = None
                height = None
                try:
                    img = element.get_image(doc)
                    if img is not None:
                        width, height = img.size
                        if summarize_images:
                            description = _summarize_image(img, stats["figures"], provider=provider)
                except Exception as e:
                    description = f"[Image extraction failed: {e}]"

                figure_samples.append(
                    {
                        "num": stats["figures"],
                        "width": width,
                        "height": height,
                        "description": description,
                    }
                )

        elif isinstance(element, SectionHeaderItem):
            stats["section_headers"] += 1

        elif isinstance(element, TextItem):
            stats["text_blocks"] += 1

        elif isinstance(element, ListItem):
            stats["other_textual_items"] += 1

        else:
            text = getattr(element, "text", None)
            if text and str(text).strip():
                stats["other_textual_items"] += 1

    logger.info(
        "Multimodal debug summary — text=%d headers=%d tables=%d figures=%d",
        stats["text_blocks"],
        stats["section_headers"],
        stats["tables"],
        stats["figures"],
    )
    return {
        "stats": stats,
        "table_samples": table_samples,
        "figure_samples": figure_samples,
    }
