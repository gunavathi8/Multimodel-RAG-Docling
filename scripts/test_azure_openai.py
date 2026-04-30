#!/usr/bin/env python3
"""
Standalone Azure OpenAI connectivity test.

Checks:
1) Chat deployment call
2) Embedding deployment call
"""

from __future__ import annotations

import os
import sys
from typing import Any

from dotenv import load_dotenv
from openai import AzureOpenAI


def _mask(value: str, keep: int = 6) -> str:
    if not value:
        return "<missing>"
    if len(value) <= keep:
        return "*" * len(value)
    return f"{value[:keep]}...({len(value)} chars)"


def _print_section(title: str) -> None:
    print(f"\n=== {title} ===")


def _fail(msg: str) -> None:
    print(f"[FAIL] {msg}")
    sys.exit(1)


def _run_chat_test(client: AzureOpenAI, deployment: str) -> None:
    _print_section("Chat Test")
    print(f"Using deployment: {deployment}")

    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "Reply with exactly: AZURE_CHAT_OK"},
        ],
        temperature=0,
        max_tokens=20,
    )
    text = (response.choices[0].message.content or "").strip()
    print(f"Response text: {text}")
    if "AZURE_CHAT_OK" not in text:
        _fail("Chat call succeeded but returned unexpected text.")
    print("[PASS] Chat deployment call succeeded.")


def _run_embedding_test(client: AzureOpenAI, deployment: str) -> None:
    _print_section("Embedding Test")
    print(f"Using deployment: {deployment}")

    response = client.embeddings.create(
        model=deployment,
        input=["Azure embedding health check"],
    )
    vector: list[float] = response.data[0].embedding
    print(f"Embedding dimension: {len(vector)}")
    if len(vector) == 0:
        _fail("Embedding returned empty vector.")
    print("[PASS] Embedding deployment call succeeded.")


def _safe_get_env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def main() -> None:
    load_dotenv()

    endpoint = _safe_get_env("AZURE_OPENAI_ENDPOINT")
    api_key = _safe_get_env("AZURE_OPENAI_API_KEY")
    api_version = _safe_get_env("AZURE_OPENAI_API_VERSION", "2024-10-21")
    chat_deployment = _safe_get_env("AZURE_OPENAI_CHAT_DEPLOYMENT", _safe_get_env("AZURE_OPENAI_CHAT_MODEL"))
    embed_deployment = _safe_get_env("AZURE_OPENAI_EMBED_DEPLOYMENT", _safe_get_env("AZURE_OPENAI_EMBED_MODEL"))

    _print_section("Config")
    print(f"AZURE_OPENAI_ENDPOINT: {_mask(endpoint, keep=28)}")
    print(f"AZURE_OPENAI_API_KEY: {_mask(api_key)}")
    print(f"AZURE_OPENAI_API_VERSION: {api_version or '<missing>'}")
    print(f"AZURE_OPENAI_CHAT_DEPLOYMENT: {chat_deployment or '<missing>'}")
    print(f"AZURE_OPENAI_EMBED_DEPLOYMENT: {embed_deployment or '<missing>'}")

    if not endpoint:
        _fail("AZURE_OPENAI_ENDPOINT is missing.")
    if not api_key:
        _fail("AZURE_OPENAI_API_KEY is missing.")
    if not chat_deployment:
        _fail("AZURE_OPENAI_CHAT_DEPLOYMENT (or AZURE_OPENAI_CHAT_MODEL) is missing.")
    if not embed_deployment:
        _fail("AZURE_OPENAI_EMBED_DEPLOYMENT (or AZURE_OPENAI_EMBED_MODEL) is missing.")

    try:
        client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
        )
    except Exception as e:
        _fail(f"Failed to initialise AzureOpenAI client: {e}")
        return

    try:
        _run_chat_test(client, chat_deployment)
    except Exception as e:
        _fail(f"Chat test failed: {e}")

    try:
        _run_embedding_test(client, embed_deployment)
    except Exception as e:
        _fail(f"Embedding test failed: {e}")

    _print_section("Result")
    print("All Azure OpenAI tests passed.")


if __name__ == "__main__":
    main()
