# Multimodal RAG with Docling, Pinecone, and React

This project builds a multimodal RAG pipeline over PDFs using:
- **Docling** for PDF parsing (text, tables, figures)
- **OpenRouter or Azure OpenAI** for embeddings + answer generation
- **Pinecone** as cloud vector database
- **FastAPI** backend
- **React (Vite)** chat frontend

## Project Structure

- `api_server.py` - Single FastAPI server with ingestion and query endpoints
- `src/utils.py` - Docling conversion + multimodal extraction helpers
- `src/chunk_embed.py` - token chunking + embedding client
- `src/pinecone_index.py` - Pinecone vector store implementation
- `src/retriever.py` - retrieval pipeline
- `frontend/` - React chat UI
- `scripts/` - standalone diagnostics and step test scripts

## Backend API

Single server, multiple endpoints:
- `GET /health`
- `POST /ingest/pdf` - ingest a PDF into Pinecone
- `POST /query` - retrieve chunks and optionally generate answer

## Environment Variables

Create `.env` in repo root:

```env
# OpenRouter (optional if using only Azure)
OPENROUTER_API_KEY=...
OPENROUTER_LLM_MODEL=google/gemma-3-27b-it
EMBED_MODEL=google/gemini-embedding-001
EMBED_VECTOR_SIZE=3072

# Azure OpenAI (optional if using only OpenRouter)
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=...
AZURE_OPENAI_API_VERSION=2024-10-21
AZURE_OPENAI_CHAT_MODEL=gpt-4o-mini
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4o-mini
AZURE_OPENAI_EMBED_MODEL=text-embedding-3-large
AZURE_OPENAI_EMBED_DEPLOYMENT=text-embedding-3-large
AZURE_OPENAI_EMBED_VECTOR_SIZE=3072

PINECONE_API_KEY=...
PINECONE_INDEX_NAME=multimodal-rag
PINECONE_NAMESPACE=default
# Used automatically when llm_provider=azure
AZURE_PINECONE_INDEX_NAME=customer-support-index
AZURE_PINECONE_NAMESPACE=default
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1

CHUNK_SIZE=1024
CHUNK_OVERLAP=100
TOP_K=5
```

Important:
- Pinecone index must be **dense** with metric **cosine**
- `EMBED_VECTOR_SIZE` must match embedding model output dimension
- In Azure mode, `AZURE_OPENAI_EMBED_VECTOR_SIZE` must match Azure embedding output dimension

## Run Backend

```bash
uv sync
uv run uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
```

## Run Frontend

```bash
cd frontend
cp .env.example .env
npm install
npm run dev
```

Frontend default API base URL:
- `VITE_API_BASE_URL=http://localhost:8000`

## API Usage Examples

### 1) Ingest PDF

```bash
curl -X POST "http://localhost:8000/ingest/pdf" \
  -F "file=@data/document.pdf" \
  -F "llm_provider=openrouter"
```

Use Azure provider:
```bash
curl -X POST "http://localhost:8000/ingest/pdf" \
  -F "file=@data/document.pdf" \
  -F "llm_provider=azure"
```

### 2) Query (retrieve + answer)

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query":"What is transformer architecture?","top_k":5,"generate_answer":true,"llm_provider":"openrouter","use_dynamic_retrieval":true}'
```

Use Azure provider:
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query":"How do I reset my account password?","top_k":5,"generate_answer":true,"llm_provider":"azure","use_dynamic_retrieval":true}'
```

### 3) Query (retrieve only)

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query":"List important tables","top_k":5,"generate_answer":false,"llm_provider":"openrouter"}'
```

## Standalone Test Scripts

Run from repo root:

```bash
uv run python scripts/step1_diagnostics.py
uv run python scripts/run_step_tests.py --pdf data/document.pdf
uv run python scripts/step3_vector_store_contract.py
uv run python scripts/step4_pinecone_smoke.py
```

## Notes

- You can query directly without re-ingestion if vectors already exist in the same Pinecone index + namespace.
- Backend CORS is enabled for frontend dev URLs:
  - `http://localhost:5173`
  - `http://127.0.0.1:5173`
