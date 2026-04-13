import os
from collections.abc import Generator
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from src.logger import get_logger

load_dotenv()

logger = get_logger(__name__)

LLM_MODEL = os.getenv("OPENROUTER_LLM_MODEL", "google/gemma-3-27b-it")

_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions strictly based on the "
    "provided document context. "
    "If the answer is not contained in the context, say so clearly — "
    "do not make up information."
)


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
    model: str = LLM_MODEL,
) -> Generator[str, None, None]:
    """
    Stream the LLM response token by token via OpenRouter.
    Yields individual string tokens so the Streamlit UI can display them as they arrive.
    """
    logger.info("Generating response via OpenRouter model '%s' with %d context chunks", model, len(context_chunks))
    user_prompt = _build_user_prompt(query, context_chunks)
    logger.info("Prompt assembled — context length: %d chars", len(user_prompt))

    stream = _client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        stream=True,
    )

    logger.info("Streaming response from '%s'...", model)
    token_count = 0
    for chunk in stream:
        token = chunk.choices[0].delta.content
        if token:
            token_count += 1
            yield token

    logger.info("Response stream complete — %d tokens received", token_count)
