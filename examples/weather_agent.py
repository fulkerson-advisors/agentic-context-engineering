#!/usr/bin/env python3
"""
Example: ACE with OpenAI Agents SDK

Demonstrates automatic learning from tool executions using the Agents SDK.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents import Agent, function_tool
from openai import OpenAI

from ace import (
    ACEAgent,
    Curator,
    OpenAIReflector,
    SQLiteBulletStorage,
    FAISSVectorIndex,
    OpenAIEmbedder,
)
from ace.config import get_openai_model

logger = logging.getLogger(__name__)


# Define tools
@function_tool
def get_weather(city: str, unit: str = "celsius") -> str:
    """
    Get current weather for a city.
    
    Args:
        city: City name
        unit: Temperature unit (celsius or fahrenheit)
    """
    # Simulate weather API with validation
    valid_cities = ["tokyo", "paris", "new york", "london", "san francisco"]
    valid_units = ["celsius", "fahrenheit"]
    
    city_lower = city.lower()
    
    if city_lower not in valid_cities:
        return f"Error: City '{city}' not found. Valid cities: {', '.join(valid_cities)}"
    
    if unit not in valid_units:
        return f"Error: Invalid unit '{unit}'. Must be 'celsius' or 'fahrenheit'"
    
    # Simulate weather data
    weather_data = {
        "tokyo": {"temp_c": 22, "temp_f": 72, "conditions": "sunny"},
        "paris": {"temp_c": 18, "temp_f": 64, "conditions": "rainy"},
        "new york": {"temp_c": 15, "temp_f": 59, "conditions": "cloudy"},
        "london": {"temp_c": 12, "temp_f": 54, "conditions": "foggy"},
        "san francisco": {"temp_c": 20, "temp_f": 68, "conditions": "partly cloudy"},
    }
    
    data = weather_data[city_lower]
    temp = data[f"temp_{unit[0]}"]
    
    return f"Weather in {city}: {temp}°{unit[0].upper()}, {data['conditions']}"


@function_tool
def get_forecast(city: str, days: int = 3) -> str:
    """
    Get weather forecast for a city.
    
    Args:
        city: City name
        days: Number of days (1-7)
    """
    if days < 1 or days > 7:
        return f"Error: Days must be between 1 and 7, got {days}"
    
    valid_cities = ["tokyo", "paris", "new york", "london", "san francisco"]
    if city.lower() not in valid_cities:
        return f"Error: City '{city}' not found"
    
    return f"Forecast for {city} ({days} days): Mostly sunny with temperatures 18-25°C"


async def main():
    logger.info("=" * 80)
    logger.info("ACE with OpenAI Agents SDK - Weather Agent Example")
    logger.info("=" * 80)
    logger.info("")
    logger.info("This example shows ACE learning from tool execution errors and successes.")
    logger.info("Watch how the agent improves over multiple queries!")
    logger.info("")
    
    # Initialize ACE components
    logger.info("Initializing ACE components...")
    
    # Storage
    storage = SQLiteBulletStorage("weather_agent.db")
    
    # Embedder
    logger.info("Initializing OpenAI embedder...")
    openai_client = OpenAI()
    embedding_model = get_openai_model(
        default="text-embedding-3-small",
        env_var="OPENAI_EMBEDDING_MODEL",
    )
    embedder = OpenAIEmbedder(client=openai_client, model=embedding_model)
    logger.info("Using embedding model: %s", embedding_model)
    
    # Vector index
    vector_index = FAISSVectorIndex(
        dimension=embedder.dimension(),
        index_path="weather_agent.faiss"
    )
    
    # Curator
    curator = Curator(
        storage=storage,
        vector_index=vector_index,
        embedder=embedder,
        similarity_threshold=0.65
    )
    
    # Reflector
    reflector_model = get_openai_model(
        default="gpt-4.1-mini",
        env_var="OPENAI_REFLECTOR_MODEL",
    )
    reflector = OpenAIReflector(openai_client, model=reflector_model)
    logger.info("Using reflector model: %s", reflector_model)
    
    # Create base agent
    base_agent = Agent(
        name="Weather Assistant",
        instructions="You are a helpful weather assistant. Use the available tools to answer weather questions.",
        tools=[get_weather, get_forecast],
    )
    
    # Wrap with ACE
    ace_agent = ACEAgent(
        agent=base_agent,
        curator=curator,
        reflector=reflector,
        enable_learning=True
    )
    
    logger.info("✓ ACE initialized")
    logger.info("")
    
    # Test queries
    queries = [
        # Query 1: Will fail - invalid city
        "What's the weather in Berlin?",
        
        # Query 2: Will fail - invalid unit
        "How's the weather in Tokyo in kelvin?",
        
        # Query 3: Will fail - invalid days
        "Give me a 10-day forecast for Paris",
        
        # Query 4: Should work better now
        "What's the weather in London?",
        
        # Query 5: Should work well
        "Tell me the weather in San Francisco and give me a 5-day forecast",
    ]
    
    for i, query in enumerate(queries, 1):
        logger.info("=" * 80)
        logger.info("Query %s: %s", i, query)
        logger.info("=" * 80)
        
        try:
            result = await ace_agent.run(query)
            logger.info("")
            logger.info("✓ Response: %s", result.final_output)
            logger.info("")
        except Exception as e:
            logger.error("")
            logger.error("✗ Error: %s", e)
            logger.error("")
        
        # Small delay between queries
        await asyncio.sleep(1)
    
    # Show learning statistics
    logger.info("")
    logger.info("=" * 80)
    logger.info("Learning Statistics")
    logger.info("=" * 80)
    
    stats = ace_agent.get_stats()
    logger.info("")
    logger.info("Total bullets learned: %s", stats["total_bullets"])
    logger.info("")
    logger.info("By tool:")
    for tool, count in stats['by_tool'].items():
        logger.info("  - %s: %s", tool, count)
    
    logger.info("")
    logger.info("By category:")
    for category, count in stats['by_category'].items():
        logger.info("  - %s: %s", category, count)
    
    logger.info("")
    logger.info("Most used bullets:")
    for bullet in stats['most_used']:
        logger.info("  - [%s uses] %s...", bullet['usage_count'], bullet['content'][:80])
    
    # Show what context would be injected
    logger.info("")
    logger.info("=" * 80)
    logger.info("Learned Context (would be injected into future queries)")
    logger.info("=" * 80)
    
    bullets = curator.get_relevant_bullets(query="weather", top_k=10)
    context = curator.format_bullets_for_prompt(bullets)
    logger.info("%s", context)
    
    # Save vector index
    vector_index.save("weather_agent.faiss")
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("✓ Example complete!")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Key takeaway: ACE automatically learned from errors and will help")
    logger.info("the agent avoid similar mistakes in future conversations!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    asyncio.run(main())
