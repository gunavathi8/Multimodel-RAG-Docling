import os
import tempfile
from collections.abc import Generator
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.chunk_embed import EmbedData, chunk_text, make_figure_chunks
from src.logger import get_logger
from src.pinecone_index import PineconeIndex
from src.rag_engine import generate_response
from src.retriever import Retriever
from src.utils import convert_pdf

load_dotenv()

logger = get_logger(__name__)

app = FastAPI(title="Multimodal RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    generate_answer: bool = True
    llm_provider: Literal["openrouter", "azure"] = "openrouter"


def _validate_env(provider: Literal["openrouter", "azure"]) -> None:
    required = ["PINECONE_API_KEY"]
    if provider == "azure":
        required.extend(["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"])
    else:
        required.append("OPENROUTER_API_KEY")

    missing = [k for k in required if not os.getenv(k)]
    if missing:
        raise HTTPException(status_code=500, detail=f"Missing env vars: {', '.join(missing)}")


def _resolve_index_name(provider: Literal["openrouter", "azure"]) -> str:
    if provider == "azure":
        return os.getenv("AZURE_PINECONE_INDEX_NAME", "customer-support-index")
    return os.getenv("PINECONE_INDEX_NAME", "multimodal-rag")


def _resolve_namespace(provider: Literal["openrouter", "azure"]) -> str:
    if provider == "azure":
        return os.getenv("AZURE_PINECONE_NAMESPACE", os.getenv("PINECONE_NAMESPACE", ""))
    return os.getenv("PINECONE_NAMESPACE", "")


def _build_index(provider: Literal["openrouter", "azure"], vector_size: int) -> PineconeIndex:
    index_name = _resolve_index_name(provider)
    namespace = _resolve_namespace(provider)
    logger.info(
        "Building Pinecone index client | provider=%s index=%s namespace=%s dim=%d",
        provider,
        index_name,
        namespace or "<default>",
        vector_size,
    )
    return PineconeIndex(index_name=index_name, namespace=namespace, vector_size=vector_size)


def _collect_stream(stream: Generator[str, None, None]) -> str:
    parts: list[str] = []
    for token in stream:
        parts.append(token)
    return "".join(parts)


@app.get("/health")
def health() -> dict:
    logger.info("Health check called")
    return {"status": "ok", "service": "multimodal-rag-api"}


@app.post("/ingest/pdf")
async def ingest_pdf(
    file: UploadFile = File(...),
    llm_provider: Literal["openrouter", "azure"] = Form(default="openrouter"),
) -> dict:
    logger.info(
        "Ingest endpoint called | filename=%s content_type=%s provider=%s",
        file.filename,
        file.content_type,
        llm_provider,
    )
    _validate_env(llm_provider)

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        logger.info("Ingest rejected | non-pdf file=%s", file.filename)
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    logger.info("Initialising ingestion dependencies | provider=%s", llm_provider)
    embedder = EmbedData(provider=llm_provider)
    index = _build_index(provider=llm_provider, vector_size=embedder.vector_size)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp_path = tmp.name
        content = await file.read()
        tmp.write(content)

    logger.info("Temporary file created | path=%s bytes=%d", tmp_path, len(content))

    try:
        logger.info("Ingestion Step 1/4 | Parse start")
        parse_result = convert_pdf(tmp_path, provider=llm_provider)
        markdown = parse_result["markdown"]
        figures = parse_result["figures"]
        logger.info("Ingestion Step 1/4 | Parse done | markdown_chars=%d figures=%d", len(markdown), len(figures))

        logger.info("Ingestion Step 2/4 | Chunk start")
        text_chunks = chunk_text(markdown, source=file.filename)
        figure_chunks = make_figure_chunks(figures, source=file.filename, start_id=len(text_chunks))
        chunks = text_chunks + figure_chunks
        logger.info(
            "Ingestion Step 2/4 | Chunk done | text_chunks=%d figure_chunks=%d total=%d",
            len(text_chunks),
            len(figure_chunks),
            len(chunks),
        )

        if not chunks:
            logger.info("Ingestion failed | no chunks generated")
            raise HTTPException(status_code=400, detail="No chunks generated from PDF.")

        logger.info("Ingestion Step 3/4 | Embedding start")
        embeddings = embedder.embed_chunks(chunks)
        logger.info("Ingestion Step 3/4 | Embedding done | vectors=%d", len(embeddings))

        if len(embeddings) != len(chunks):
            logger.info("Ingestion failed | mismatch chunks=%d vectors=%d", len(chunks), len(embeddings))
            raise HTTPException(
                status_code=500,
                detail=f"Embedding count mismatch: chunks={len(chunks)}, vectors={len(embeddings)}",
            )

        logger.info("Ingestion Step 4/4 | Upsert start")
        index.upsert(chunks, embeddings)
        total = index.count()
        logger.info("Ingestion Step 4/4 | Upsert done | total_vectors=%d", total)

        return {
            "status": "success",
            "provider": llm_provider,
            "index_name": index.index_name,
            "namespace": index.namespace or "",
            "file": file.filename,
            "markdown_chars": len(markdown),
            "figures": len(figures),
            "text_chunks": len(text_chunks),
            "figure_chunks": len(figure_chunks),
            "total_chunks": len(chunks),
            "index_total_vectors": total,
        }
    finally:
        try:
            os.unlink(tmp_path)
            logger.info("Temporary file deleted | path=%s", tmp_path)
        except Exception as e:
            logger.warning("Temp file cleanup failed | path=%s err=%s", tmp_path, e)


@app.post("/query")
def query(req: QueryRequest) -> dict:
    logger.info(
        "Query endpoint called | query=%s top_k=%d generate_answer=%s provider=%s",
        req.query[:120],
        req.top_k,
        req.generate_answer,
        req.llm_provider,
    )
    _validate_env(req.llm_provider)

    logger.info("Query Step 1/2 | Retrieval start")
    embedder = EmbedData(provider=req.llm_provider)
    index = _build_index(provider=req.llm_provider, vector_size=embedder.vector_size)
    retriever = Retriever(index, embedder)
    chunks = retriever.retrieve(req.query, top_k=req.top_k)
    logger.info("Query Step 1/2 | Retrieval done | chunks=%d", len(chunks))

    answer = None
    if req.generate_answer:
        logger.info("Query Step 2/2 | Generation start")
        answer = _collect_stream(generate_response(req.query, chunks, provider=req.llm_provider))
        logger.info("Query Step 2/2 | Generation done | answer_chars=%d", len(answer))
    else:
        logger.info("Query Step 2/2 | Generation skipped by request")

    return {
        "query": req.query,
        "top_k": req.top_k,
        "provider": req.llm_provider,
        "index_name": index.index_name,
        "namespace": index.namespace or "",
        "chunks": chunks,
        "answer": answer,
    }
