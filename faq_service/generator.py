import json
import os
from pathlib import Path
from typing import Any, Literal
import re

from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI

from faq_service.prompts import (
    FAQ_GENERATION_SYSTEM_PROMPT,
    FAQ_USER_STYLE_REWRITE_PROMPT,
    FAQ_VERIFIER_SYSTEM_PROMPT,
)
from faq_service.schemas import FaqGenerationRequest, FaqItem
from src.chunk_embed import EmbedData, chunk_text
from src.logger import get_logger
from src.pinecone_index import PineconeIndex

load_dotenv()

logger = get_logger(__name__)

OPENROUTER_LLM_MODEL = os.getenv("OPENROUTER_LLM_MODEL", "google/gemma-3-27b-it")
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

_USER_STYLE_MARKERS = [
    "i ",
    "i'm",
    "im ",
    "my ",
    "not working",
    "still",
    "failed",
    "failing",
    "why ",
    "what am i",
    "how do i",
    "pls",
    "please",
    "unable",
    "can't",
    "cannot",
]


class FaqGenerationService:
    def _get_client_and_model(self, provider: Literal["openrouter", "azure"]) -> tuple[Any, str]:
        if provider == "azure":
            if _azure_client is None:
                raise ValueError("Azure selected but AZURE_OPENAI_ENDPOINT/API_KEY are missing")
            return _azure_client, AZURE_OPENAI_CHAT_DEPLOYMENT
        return _openrouter_client, OPENROUTER_LLM_MODEL

    def _resolve_index_name(self, provider: Literal["openrouter", "azure"]) -> str:
        if provider == "azure":
            return os.getenv("AZURE_PINECONE_INDEX_NAME", "customer-support-index")
        return os.getenv("PINECONE_INDEX_NAME", "multimodal-rag")

    def _resolve_namespace(self, provider: Literal["openrouter", "azure"]) -> str:
        if provider == "azure":
            return os.getenv("AZURE_PINECONE_NAMESPACE", os.getenv("PINECONE_NAMESPACE", ""))
        return os.getenv("PINECONE_NAMESPACE", "")

    def _retrieve_context_for_category(
        self,
        category: str,
        provider: Literal["openrouter", "azure"],
        top_k: int,
    ) -> list[dict[str, Any]]:
        logger.info("FAQ Context Retrieval | category=%s provider=%s top_k=%d", category, provider, top_k)
        embedder = EmbedData(provider=provider)
        index = PineconeIndex(
            index_name=self._resolve_index_name(provider),
            namespace=self._resolve_namespace(provider),
            vector_size=embedder.vector_size,
        )

        retrieval_query = self._build_category_retrieval_query(category)
        query_vector = embedder.embed_query(retrieval_query)
        chunks = index.search(query_vector, top_k=top_k)
        logger.info("FAQ Context Retrieval Completed | category=%s chunks=%d", category, len(chunks))
        return chunks

    def _build_category_retrieval_query(self, category: str) -> str:
        category_examples: dict[str, str] = {
            "Document Upload": (
                "document upload not working missing fields cannot submit why upload fails "
                "what am i missing mandatory fields upload from sharepoint"
            ),
            "Mandatory Fields and Levels": (
                "why level is mandatory why do i need levels missing required field validation error "
                "what happens if level1 level2 not selected"
            ),
            "Relationship Mapping": (
                "how to create relationship map not able to connect documents map confusion "
                "how to link sections step by step"
            ),
            "Search and Retrieval": (
                "not finding document in search why results missing filters not working "
                "how do i find latest version"
            ),
            "Troubleshooting": (
                "something is wrong not working failing error message what to check first "
                "why this fails and how to fix"
            ),
        }
        base = category_examples.get(category, f"{category} user issues troubleshooting")
        return f"Revealr.ai support context: {base}"

    def _generate_faq_candidates(
        self,
        category: str,
        context_chunks: list[dict[str, Any]],
        provider: Literal["openrouter", "azure"],
        faqs_per_category: int,
    ) -> list[dict[str, Any]]:
        client, model = self._get_client_and_model(provider)
        context_blocks = "\n\n---\n\n".join(
            f"[chunk_id={c['chunk_id']} score={c.get('score', 0)} source={c['source']}]\n{c['text']}"
            for c in context_chunks
        )

        logger.info("FAQ Candidate Generation | category=%s model=%s", category, model)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": FAQ_GENERATION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Category: {category}\n\n"
                        f"Context:\n{context_blocks}\n\n"
                        f"Generate exactly {faqs_per_category} FAQs as strict JSON array only.\n"
                        "Each item keys:\n"
                        "- question (string)\n"
                        "- answer (string)\n"
                        "- evidence_chunk_ids (array of integers)\n"
                        "- confidence (0 to 1)\n"
                        "- style_tag (one of: frustrated, confused, validation, how_to, follow_up)\n"
                        "Rules:\n"
                        "- Use context only.\n"
                        "- Focus on realistic user support questions in natural user phrasing.\n"
                        "- Avoid textbook/document-title style questions.\n"
                        "- At least 50% questions must be problem-style or follow-up style.\n"
                        "- Include some negative phrasing: not working, failing, missing fields, why mandatory.\n"
                        "- Keep answers actionable and concise.\n"
                    ),
                },
            ],
            stream=False,
            temperature=0.2,
        )
        text = (response.choices[0].message.content or "").strip()
        try:
            return json.loads(text)
        except Exception:
            start = text.find("[")
            end = text.rfind("]")
            if start != -1 and end != -1 and end > start:
                return json.loads(text[start : end + 1])
            raise ValueError(f"Could not parse FAQ JSON for category '{category}'")

    def _verify_faq_items(
        self,
        category: str,
        context_chunks: list[dict[str, Any]],
        candidates: list[dict[str, Any]],
        provider: Literal["openrouter", "azure"],
    ) -> list[FaqItem]:
        client, model = self._get_client_and_model(provider)
        context_blocks = "\n\n---\n\n".join(
            f"[chunk_id={c['chunk_id']} score={c.get('score', 0)} source={c['source']}]\n{c['text']}"
            for c in context_chunks
        )

        logger.info("FAQ Verification | category=%s candidate_count=%d", category, len(candidates))
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": FAQ_VERIFIER_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Category: {category}\n\n"
                        f"Context:\n{context_blocks}\n\n"
                        "Candidate FAQs JSON:\n"
                        f"{json.dumps(candidates, ensure_ascii=False)}\n\n"
                        "Return strict JSON array only. Keep only grounded FAQs.\n"
                        "Each item keys:\n"
                        "- question (string)\n"
                        "- answer (string)\n"
                        "- evidence_chunk_ids (array of integers)\n"
                        "- confidence (0 to 1)\n"
                        "- style_tag (one of: frustrated, confused, validation, how_to, follow_up)\n"
                    ),
                },
            ],
            stream=False,
            temperature=0.0,
        )
        text = (response.choices[0].message.content or "").strip()
        try:
            parsed = json.loads(text)
        except Exception:
            start = text.find("[")
            end = text.rfind("]")
            if start != -1 and end != -1 and end > start:
                parsed = json.loads(text[start : end + 1])
            else:
                raise ValueError(f"Could not parse verified FAQ JSON for category '{category}'")

        result: list[FaqItem] = []
        for item in parsed:
            result.append(
                FaqItem(
                    category=category,
                    question=str(item.get("question", "")).strip(),
                    answer=str(item.get("answer", "")).strip(),
                    evidence_chunk_ids=[int(x) for x in item.get("evidence_chunk_ids", []) if str(x).isdigit()],
                    confidence=float(item.get("confidence", 0.0) or 0.0),
                )
            )
        logger.info("FAQ Verification Completed | category=%s accepted=%d", category, len(result))
        return result

    def _looks_user_style(self, question: str) -> bool:
        q = question.lower().strip()
        if len(q.split()) < 5:
            return False
        if any(marker in q for marker in _USER_STYLE_MARKERS):
            return True
        # Allow natural question variants even without first-person pronoun
        return q.startswith(("how can i", "how do i", "why is", "what should i", "where can i"))

    def _rewrite_to_user_style(
        self,
        items: list[FaqItem],
        provider: Literal["openrouter", "azure"],
    ) -> list[FaqItem]:
        client, model = self._get_client_and_model(provider)
        need_rewrite = [item for item in items if not self._looks_user_style(item.question)]
        if not need_rewrite:
            return items

        logger.info("FAQ User-Style Rewrite | items=%d", len(need_rewrite))
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": FAQ_USER_STYLE_REWRITE_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Rewrite only the question text into realistic user-chat style.\n"
                        "Return strict JSON array with keys: original_question, rewritten_question.\n"
                        "Keep intent exactly same.\n\n"
                        f"Input questions:\n{json.dumps([x.question for x in need_rewrite], ensure_ascii=False)}"
                    ),
                },
            ],
            stream=False,
            temperature=0.2,
        )
        text = (response.choices[0].message.content or "").strip()
        try:
            rewrites = json.loads(text)
        except Exception:
            start = text.find("[")
            end = text.rfind("]")
            rewrites = json.loads(text[start : end + 1]) if start != -1 and end != -1 and end > start else []

        rewrite_map: dict[str, str] = {}
        for row in rewrites:
            oq = str(row.get("original_question", "")).strip()
            rq = str(row.get("rewritten_question", "")).strip()
            if oq and rq:
                rewrite_map[oq] = rq

        updated: list[FaqItem] = []
        for item in items:
            new_q = rewrite_map.get(item.question, item.question)
            updated.append(
                FaqItem(
                    category=item.category,
                    question=new_q,
                    answer=item.answer,
                    evidence_chunk_ids=item.evidence_chunk_ids,
                    confidence=item.confidence,
                )
            )
        return updated

    def _dedupe_questions(self, items: list[FaqItem]) -> list[FaqItem]:
        seen: set[str] = set()
        deduped: list[FaqItem] = []
        for item in items:
            key = re.sub(r"\s+", " ", item.question.strip().lower())
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped

    def _write_outputs(self, faqs: list[FaqItem], markdown_path: str, json_path: str) -> None:
        md_file = Path(markdown_path)
        js_file = Path(json_path)
        md_file.parent.mkdir(parents=True, exist_ok=True)
        js_file.parent.mkdir(parents=True, exist_ok=True)

        by_category: dict[str, list[FaqItem]] = {}
        for faq in faqs:
            by_category.setdefault(faq.category, []).append(faq)

        logger.info("Writing FAQ markdown output to %s", md_file)
        lines: list[str] = ["# Revealr.ai FAQ (Generated)\n"]
        for category, items in by_category.items():
            lines.append(f"## {category}\n")
            for i, item in enumerate(items, start=1):
                lines.append(f"### Q{i}. {item.question}\n")
                lines.append(f"{item.answer}\n")
                lines.append(
                    f"_Evidence chunk IDs: {', '.join(str(x) for x in item.evidence_chunk_ids) or 'N/A'} | "
                    f"Confidence: {item.confidence:.2f}_\n"
                )
            lines.append("")
        md_file.write_text("\n".join(lines), encoding="utf-8")

        logger.info("Writing FAQ json output to %s", js_file)
        js_file.write_text(
            json.dumps([item.model_dump() for item in faqs], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _ingest_generated_faq(
        self,
        provider: Literal["openrouter", "azure"],
        markdown_path: str,
    ) -> dict[str, Any]:
        logger.info("FAQ Ingestion Start | provider=%s path=%s", provider, markdown_path)
        text = Path(markdown_path).read_text(encoding="utf-8")
        embedder = EmbedData(provider=provider)
        index = PineconeIndex(
            index_name=self._resolve_index_name(provider),
            namespace=self._resolve_namespace(provider),
            vector_size=embedder.vector_size,
        )
        chunks = chunk_text(text, source=Path(markdown_path).name)
        embeddings = embedder.embed_chunks(chunks)
        index.upsert(chunks, embeddings)
        total = index.count()
        logger.info("FAQ Ingestion Completed | new_chunks=%d index_total=%d", len(chunks), total)
        return {
            "ingested_chunks": len(chunks),
            "index_total_vectors": total,
            "index_name": index.index_name,
            "namespace": index.namespace or "",
        }

    def generate(self, request: FaqGenerationRequest) -> dict[str, Any]:
        logger.info(
            "FAQ Generation Request | provider=%s categories=%d faqs_per_category=%d",
            request.llm_provider,
            len(request.categories),
            request.faqs_per_category,
        )
        all_faqs: list[FaqItem] = []

        for category in request.categories:
            context_chunks = self._retrieve_context_for_category(
                category=category,
                provider=request.llm_provider,
                top_k=request.retrieval_top_k,
            )
            if not context_chunks:
                logger.info("No context chunks found for category=%s, skipping", category)
                continue

            candidates = self._generate_faq_candidates(
                category=category,
                context_chunks=context_chunks,
                provider=request.llm_provider,
                faqs_per_category=request.faqs_per_category,
            )
            verified = self._verify_faq_items(
                category=category,
                context_chunks=context_chunks,
                candidates=candidates,
                provider=request.llm_provider,
            )
            verified = self._rewrite_to_user_style(verified, provider=request.llm_provider)
            all_faqs.extend(verified)

        all_faqs = self._dedupe_questions(all_faqs)

        self._write_outputs(
            faqs=all_faqs,
            markdown_path=request.output_markdown_path,
            json_path=request.output_json_path,
        )

        ingest_result = None
        if request.ingest_generated_faq:
            ingest_result = self._ingest_generated_faq(
                provider=request.llm_provider,
                markdown_path=request.output_markdown_path,
            )

        logger.info("FAQ Generation Completed | total_faqs=%d", len(all_faqs))
        return {
            "status": "success",
            "provider": request.llm_provider,
            "total_faqs": len(all_faqs),
            "categories_processed": len(request.categories),
            "output_markdown_path": request.output_markdown_path,
            "output_json_path": request.output_json_path,
            "ingest_result": ingest_result,
        }
