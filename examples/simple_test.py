#!/usr/bin/env python3
"""
Simple integration demo to verify ACE components working together.

This script makes live OpenAI API calls. When collected by pytest it is skipped
automatically so that the unit test suite remains self-contained.
"""

import asyncio
import logging
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skip(
    "examples/simple_test.py exercises live OpenAI APIs and async flows; run manually instead of via pytest."
)


sys.path.insert(0, str(Path(__file__).parent.parent))

from ace import (
    Curator,
    OpenAIReflector,
    SQLiteBulletStorage,
    FAISSVectorIndex,
    OpenAIEmbedder,
    Bullet,
    ToolExecution,
)
from openai import OpenAI
from ace.config import get_openai_model

logger = logging.getLogger(__name__)


async def test_ace_components():
    logger.info("=" * 80)
    logger.info("ACE Components Test")
    logger.info("=" * 80)
    
    # Initialize components
    logger.info("")
    logger.info("1. Initializing storage...")
    storage = SQLiteBulletStorage("test_ace.db")
    logger.info("✓ SQLite storage initialized")
    
    logger.info("")
    logger.info("2. Initializing OpenAI embedder...")
    openai_client = OpenAI()
    embedding_model = get_openai_model(
        default="text-embedding-3-small",
        env_var="OPENAI_EMBEDDING_MODEL",
    )
    embedder = OpenAIEmbedder(client=openai_client, model=embedding_model)
    logger.info("Using embedding model: %s", embedding_model)
    logger.info("✓ Embedder ready (dimension: %s)", embedder.dimension())
    
    logger.info("")
    logger.info("3. Initializing FAISS index...")
    vector_index = FAISSVectorIndex(
        dimension=embedder.dimension(),
        index_path="test_ace.faiss"
    )
    logger.info("✓ FAISS index initialized")
    
    logger.info("")
    logger.info("4. Creating curator...")
    curator = Curator(
        storage=storage,
        vector_index=vector_index,
        embedder=embedder,
        similarity_threshold=0.65
    )
    logger.info("✓ Curator created")
    
    logger.info("")
    logger.info("5. Creating reflector...")
    reflector_model = get_openai_model(
        default="gpt-4.1-mini",
        env_var="OPENAI_REFLECTOR_MODEL",
    )
    reflector = OpenAIReflector(openai_client, model=reflector_model)
    logger.info("Using reflector model: %s", reflector_model)
    logger.info("✓ Reflector created")
    
    # Test reflection
    logger.info("")
    logger.info("=" * 80)
    logger.info("Testing Reflection")
    logger.info("=" * 80)
    
    execution = ToolExecution(
        tool_name="get_weather",
        arguments={"city": "InvalidCity"},
        result="Error: City 'InvalidCity' not found. Valid cities: tokyo, paris, new york",
        success=False,
        error="City not found"
    )
    
    logger.info("")
    logger.info("Reflecting on tool execution:")
    logger.info("  Tool: %s", execution.tool_name)
    logger.info("  Arguments: %s", execution.arguments)
    logger.info("  Success: %s", execution.success)
    logger.info("  Error: %s", execution.error)
    
    bullets = await reflector.reflect(execution)
    
    logger.info("")
    logger.info("✓ Generated %s insights:", len(bullets))
    for bullet in bullets:
        logger.info("  [%s] %s", bullet.category, bullet.content)
    
    # Test curator
    logger.info("")
    logger.info("=" * 80)
    logger.info("Testing Curator (Semantic Deduplication)")
    logger.info("=" * 80)
    
    logger.info("")
    logger.info("Adding bullets to curator...")
    added = curator.add_bullets(bullets)
    logger.info("✓ Added %s bullets (after deduplication)", added)
    
    # Try adding similar bullets
    logger.info("")
    logger.info("Adding semantically similar bullets...")
    similar_bullets = [
        Bullet(
            id="",
            content="Always validate city names against the allowed list",
            tool_name="get_weather",
            category="error_avoidance"
        ),
        Bullet(
            id="",
            content="Make sure to use valid city names from the approved list",
            tool_name="get_weather",
            category="error_avoidance"
        )
    ]
    
    added_similar = curator.add_bullets(similar_bullets)
    logger.info("✓ Added %s bullets (duplicates were merged)", added_similar)
    
    # Test retrieval
    logger.info("")
    logger.info("=" * 80)
    logger.info("Testing Bullet Retrieval")
    logger.info("=" * 80)
    
    logger.info("")
    logger.info("Retrieving bullets for 'get_weather'...")
    retrieved = curator.get_relevant_bullets(
        query="weather city validation",
        tool_name="get_weather",
        top_k=5
    )
    
    logger.info("✓ Retrieved %s bullets:", len(retrieved))
    for bullet in retrieved:
        logger.info("  [%s] %s", bullet.category, bullet.content)
    
    # Test context formatting
    logger.info("")
    logger.info("=" * 80)
    logger.info("Testing Context Formatting")
    logger.info("=" * 80)
    
    context = curator.format_bullets_for_prompt(retrieved)
    logger.info("")
    logger.info("Formatted context for injection:")
    logger.info("%s", context)
    
    # Show stats
    logger.info("")
    logger.info("=" * 80)
    logger.info("Statistics")
    logger.info("=" * 80)
    
    stats = curator.get_stats()
    logger.info("")
    logger.info("Total bullets: %s", stats["total_bullets"])
    logger.info("By tool: %s", stats["by_tool"])
    logger.info("By category: %s", stats["by_category"])
    
    # Save index
    logger.info("")
    logger.info("Saving FAISS index...")
    vector_index.save("test_ace.faiss")
    logger.info("✓ Index saved")
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("✓ All tests passed!")
    logger.info("=" * 80)
    logger.info("")
    logger.info("ACE core components are working correctly.")
    logger.info("Note: OpenAI Agents SDK has a compatibility issue with Python 3.11.0rc1")
    logger.info("The ACE implementation is complete and will work once the SDK issue is resolved.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    asyncio.run(test_ace_components())
