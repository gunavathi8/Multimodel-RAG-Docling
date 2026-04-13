"""
Centralised logging configuration for the multimodal RAG app.

Import this module FIRST (before any HuggingFace / transformers imports)
to suppress the `__path__` deprecation spam that transformers produces.
"""

import logging
import os
import warnings


# ---------------------------------------------------------------------------
# 1. Silence the `__path__` alias spam from transformers internals
#    These are harmless deprecation notices from lazy-loading in transformers,
#    but they flood the terminal on every startup.
# ---------------------------------------------------------------------------
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")   # also silences a tokenizer fork warning

warnings.filterwarnings(
    "ignore",
    message=".*Accessing `__path__`.*",
)
warnings.filterwarnings(
    "ignore",
    message=".*alias will be removed in future versions.*",
)

# ---------------------------------------------------------------------------
# 2. Turn down noisy third-party loggers to WARNING or ERROR
# ---------------------------------------------------------------------------
_QUIET_LIBS = [
    "httpx",        # OpenAI/OpenRouter HTTP client — very chatty
    "httpcore",
    "openai",
    "PIL",
    "qdrant_client",
    "docling",
    "docling_core",
]

for _lib in _QUIET_LIBS:
    logging.getLogger(_lib).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# 3. Configure the application-level logger
# ---------------------------------------------------------------------------
def get_logger(name: str) -> logging.Logger:
    """
    Return a logger for *name* using the shared app formatter.

    Usage:
        from src.logger import get_logger
        logger = get_logger(__name__)
    """
    logger = logging.getLogger(name)

    # Only add handler once (guard against Streamlit re-runs)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(logging.INFO)
    logger.propagate = False   # prevent double-printing via root logger
    return logger
