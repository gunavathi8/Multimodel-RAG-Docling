from typing import Literal

from pydantic import BaseModel, Field


class FaqGenerationRequest(BaseModel):
    llm_provider: Literal["openrouter", "azure"] = "azure"
    categories: list[str] = Field(
        default_factory=lambda: [
            "Document Upload",
            "Mandatory Fields and Levels",
            "Relationship Mapping",
            "Search and Retrieval",
            "Troubleshooting",
        ]
    )
    faqs_per_category: int = Field(default=6, ge=2, le=20)
    retrieval_top_k: int = Field(default=12, ge=4, le=25)
    output_markdown_path: str = "generated/revealr_faq_generated.md"
    output_json_path: str = "generated/revealr_faq_generated.json"
    ingest_generated_faq: bool = False


class FaqItem(BaseModel):
    category: str
    question: str
    answer: str
    evidence_chunk_ids: list[int] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

