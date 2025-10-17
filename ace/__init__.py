"""
ACE - Agentic Context Engineering

Utilities for capturing, storing, and reusing insights from tool executions.
"""

from ace.core.curator import Curator
from ace.core.reflector import OpenAIReflector
from ace.core.interfaces import Bullet, ToolExecution

from ace.storage.sqlite_storage import SQLiteBulletStorage
from ace.storage.faiss_index import FAISSVectorIndex
from ace.storage.embedder import OpenAIEmbedder

__version__ = "0.1.0"

__all__ = [
    "Curator",
    "OpenAIReflector",
    "Bullet",
    "ToolExecution",
    "SQLiteBulletStorage",
    "FAISSVectorIndex",
    "OpenAIEmbedder",
    "ACEAgent",
]


def __getattr__(name):
    if name == "ACEAgent":
        from ace.agents.openai_agents import ACEAgent as _ACEAgent

        return _ACEAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
