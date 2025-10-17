"""
Configuration helpers for ACE.

Loads environment variables (including from a .env file) and provides
utilities for accessing shared settings such as the OpenAI model name.
"""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv


@lru_cache(maxsize=1)
def _ensure_dotenv_loaded() -> None:
    """
    Load environment variables from a .env file if present.

    The default behavior of ``load_dotenv`` searches the current working
    directory and parent directories, so we simply invoke it once and cache
    the side effects to avoid repeated filesystem checks.
    """
    load_dotenv()


def get_openai_model(default: str, env_var: str = "OPENAI_MODEL") -> str:
    """
    Resolve the OpenAI model name from environment variables.

    Args:
        default: Fallback model name if no environment variable is set.
        env_var: Specific environment variable to consult before falling back
            to the shared ``OPENAI_MODEL``.

    Returns:
        Model name string.
    """
    _ensure_dotenv_loaded()

    if env_var and env_var != "OPENAI_MODEL":
        specific = os.getenv(env_var)
        if specific:
            return specific

    return os.getenv("OPENAI_MODEL", default)
