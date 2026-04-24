import os
from collections.abc import Generator
from typing import Literal
from typing import Any

from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI

from prompt_library.customer_support_prompt import (
    CUSTOMER_SUPPORT_SYSTEM_PROMPT,
)
from src.logger import get_logger

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

_SYSTEM_PROMPT = CUSTOMER_SUPPORT_SYSTEM_PROMPT


def _build_user_prompt(query: str, context_chunks: list[dict[str, Any]]) -> str:
    context_blocks = "\n\n---\n\n".join(
        f"[Source: {c['source']} | chunk {c['chunk_id']}]\n{c['text']}"
        for c in context_chunks
    )
    return (
        f"Context from the document:\n\n"
        f"{context_blocks}\n\n"
        f"Question: {query}\n\n"
        f"Answer based only on the context above:"
    )


def generate_response(
    query: str,
    context_chunks: list[dict[str, Any]],
    provider: Literal["openrouter", "azure"] = "openrouter",
    model: str | None = None,
) -> Generator[str, None, None]:
    """
    Stream the LLM response token by token via OpenRouter.
    Yields individual string tokens so the Streamlit UI can display them as they arrive.
    """
    if provider == "azure":
        if _azure_client is None:
            raise ValueError("Azure chat selected but AZURE_OPENAI_ENDPOINT/API_KEY are missing")
        client = _azure_client
        active_model = model or AZURE_OPENAI_CHAT_DEPLOYMENT
    else:
        client = _openrouter_client
        active_model = model or OPENROUTER_LLM_MODEL

    logger.info(
        "Generating response | provider=%s model=%s context_chunks=%d",
        provider,
        active_model,
        len(context_chunks),
    )
    user_prompt = _build_user_prompt(query, context_chunks)
    logger.info("Prompt assembled — context length: %d chars", len(user_prompt))

    stream = client.chat.completions.create(
        model=active_model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        stream=True,
    )

    logger.info("Streaming response from model='%s' provider='%s'...", active_model, provider)
    token_count = 0
    for chunk in stream:
        # Azure/OpenAI streams may include non-token events where `choices` is empty.
        choices = getattr(chunk, "choices", None) or []
        if not choices:
            continue

        delta = getattr(choices[0], "delta", None)
        token = getattr(delta, "content", None) if delta else None
        if token:
            token_count += 1
            yield token

    logger.info("Response stream complete — %d tokens received", token_count)
